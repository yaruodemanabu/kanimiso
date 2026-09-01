//! Typical-problem speed / accuracy report (markdown on stdout).
//!
//! Run with `cargo run --release --example benchmark`.

use std::time::Instant;

use amatsuki::{Rng, StandardNormal};
use isuzu::api::{recover, QmleEstimator};
use isuzu::datasets::{make_dp_gaussians, make_hawkes, make_ibp_linear_gaussian, make_ou};
use isuzu::filter::{
    kalman, particle_filter, unscented_kalman, LinearGaussian, ParticleConfig, UkfParams,
};
use isuzu::highfreq::hayashi_yoshida;
use isuzu::linalg::{col_from_slice, mat_from_row_slice};
use isuzu::npbayes::{
    dp_gaussian_mixture_gibbs, ibp_linear_gaussian_gibbs_ex, IbpLinearGaussianFit,
};
use isuzu::optimize::OptOptions;
use isuzu::path::{Path, TickSeries};
use isuzu::prelude::*;
use isuzu::simulate::{simulate_gbm_exact, SimConfig};

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn rmse(xs: &[f64]) -> f64 {
    (xs.iter().map(|e| e * e).sum::<f64>() / xs.len() as f64).sqrt()
}

fn majority_accuracy(pred: &[usize], truth: &[usize]) -> f64 {
    let kp = pred.iter().copied().max().unwrap_or(0) + 1;
    let kt = truth.iter().copied().max().unwrap_or(0) + 1;
    let mut counts = vec![vec![0usize; kt]; kp];
    for (&p, &t) in pred.iter().zip(truth) {
        counts[p][t] += 1;
    }
    let correct: usize = counts
        .iter()
        .map(|row| row.iter().copied().max().unwrap_or(0))
        .sum();
    correct as f64 / pred.len() as f64
}

fn main() {
    println!("# ベンチマーク結果");
    println!();
    println!("ホスト: 単一スレッド、`cargo run --release --example benchmark`。");
    println!("乱数は amatsuki ChaCha8（`seed_rng`）。時刻は `Instant` の壁時計。");
    println!();

    bench_ou_qmle();
    bench_filters();
    bench_hawkes();
    bench_hayashi_yoshida();
    bench_gbm_schemes();
    bench_dp_mixture();
    bench_ibp();
}

fn bench_ou_qmle() {
    println!("## 1. Ornstein–Uhlenbeck の Euler QMLE");
    println!();
    println!("真値 $(\\kappa,\\theta,\\sigma)=(1.3,0,0.4)$、厳密スキームで $T=8$, $n=1500$。");
    println!("出発点 $(0.8,0.1,0.6)$、箱 $[0.1,-1,0.05]\\times[4,1,2]$。5 本の独立パス。");
    println!();
    println!("| seed | 時間 (ms) | $\\|\\hat\\kappa-\\kappa\\|$ | $\\|\\hat\\theta-\\theta\\|$ | $\\|\\hat\\sigma-\\sigma\\|$ |");
    println!("| ---: | ---: | ---: | ---: | ---: |");
    let mut times = Vec::new();
    let mut e0 = Vec::new();
    let mut e1 = Vec::new();
    let mut e2 = Vec::new();
    for seed in [7u64, 11, 19, 29, 41] {
        let toy = make_ou(1.3, 0.0, 0.4, 8.0, 1500, seed).unwrap();
        let mut est = QmleEstimator::new(toy.model.clone(), vec![0.8, 0.1, 0.6])
            .bounds(vec![0.1, -1.0, 0.05], vec![4.0, 1.0, 2.0]);
        est.opt = OptOptions {
            max_iter: 250,
            ..OptOptions::default()
        };
        let t0 = Instant::now();
        let report = recover(&mut est, &toy.path, &toy.truth).unwrap();
        let dt = t0.elapsed();
        times.push(ms(dt));
        e0.push(report.abs_error[0]);
        e1.push(report.abs_error[1]);
        e2.push(report.abs_error[2]);
        println!(
            "| {} | {:.1} | {:.4} | {:.4} | {:.4} |",
            seed,
            ms(dt),
            report.abs_error[0],
            report.abs_error[1],
            report.abs_error[2]
        );
    }
    println!();
    println!(
        "平均時間 **{:.1} ms**。RMSE $(\\kappa,\\theta,\\sigma)=({:.4},{:.4},{:.4})$。",
        mean(&times),
        rmse(&e0),
        rmse(&e1),
        rmse(&e2)
    );
    println!();
}

fn bench_filters() {
    println!("## 2. 線形ガウス状態空間：KF / UKF / SIR");
    println!();
    println!("$x_{{k+1}}=0.9 x_k+w_k$, $y_k=x_k+v_k$, $Q=0.04$, $R=0.16$, $n=200$。粒子 400。");
    println!("精度は真の潜在状態に対する濾波平均の RMSE。3 本。");
    println!();
    println!("| seed | KF ms | UKF ms | SIR ms | KF RMSE | UKF RMSE | SIR RMSE |");
    println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    let model = LinearGaussian::new(
        mat_from_row_slice(1, 1, &[0.9]),
        mat_from_row_slice(1, 1, &[0.04]),
        mat_from_row_slice(1, 1, &[1.0]),
        mat_from_row_slice(1, 1, &[0.16]),
    )
    .unwrap();
    let x0 = col_from_slice(&[0.0]);
    let p0 = mat_from_row_slice(1, 1, &[1.0]);
    let mut rows = Vec::new();
    for seed in [3u64, 17, 31] {
        let n = 200usize;
        let mut rng = seed_rng(seed);
        let mut latent = vec![0.0; n + 1];
        let mut obs = vec![0.0; n + 1];
        let times: Vec<f64> = (0..=n).map(|i| i as f64).collect();
        for i in 1..=n {
            latent[i] = 0.9 * latent[i - 1] + 0.2 * rng.sample(StandardNormal);
            obs[i] = latent[i] + 0.4 * rng.sample(StandardNormal);
        }
        let path = Path::new(times, obs, 1).unwrap();
        let t0 = Instant::now();
        let kf = kalman(&model, &path, &x0, &p0).unwrap();
        let tk = t0.elapsed();
        let t0 = Instant::now();
        let ukf = unscented_kalman(&model, &path, &x0, &p0, UkfParams::default()).unwrap();
        let tu = t0.elapsed();
        let t0 = Instant::now();
        let pf = particle_filter(
            &model,
            &path,
            &x0,
            &p0,
            ParticleConfig {
                n_particles: 400,
                ..ParticleConfig::default()
            },
            &mut rng,
        )
        .unwrap();
        let tp = t0.elapsed();
        let rmse_of = |filt: &[faer::Col<f64>]| {
            let mut s = 0.0;
            for i in 0..=n {
                let e = filt[i][0] - latent[i];
                s += e * e;
            }
            (s / (n + 1) as f64).sqrt()
        };
        let rk = rmse_of(&kf.filtered);
        let ru = rmse_of(&ukf.filtered);
        let rp = rmse_of(&pf.filtered);
        println!(
            "| {} | {:.2} | {:.2} | {:.1} | {:.4} | {:.4} | {:.4} |",
            seed,
            ms(tk),
            ms(tu),
            ms(tp),
            rk,
            ru,
            rp
        );
        rows.push((ms(tk), ms(tu), ms(tp), rk, ru, rp));
    }
    let n = rows.len() as f64;
    let avg = |f: fn(&(f64, f64, f64, f64, f64, f64)) -> f64| rows.iter().map(f).sum::<f64>() / n;
    println!();
    println!(
        "平均: KF {:.2} ms / UKF {:.2} ms / SIR {:.1} ms。平均 RMSE KF {:.4}, UKF {:.4}, SIR {:.4}。",
        avg(|r| r.0),
        avg(|r| r.1),
        avg(|r| r.2),
        avg(|r| r.3),
        avg(|r| r.4),
        avg(|r| r.5)
    );
    println!();
    println!("線形ガウスでは KF が MMSE なので、UKF は数値誤差の範囲で一致し、SIR は粒子分散の分だけ劣るのが正しい。");
    println!();
}

fn bench_hawkes() {
    println!("## 3. 指数 Hawkes の閉形式 MLE");
    println!();
    println!("真値 $(\\mu,\\alpha,\\beta)=(0.8,0.4,1.5)$、$T=80$。出発点 $(0.5,0.3,1.2)$。5 本。");
    println!();
    println!("| seed | 時間 (ms) | $n$ | $\\|\\hat\\mu-\\mu\\|$ | $\\|\\hat\\alpha-\\alpha\\|$ | $\\|\\hat\\beta-\\beta\\|$ |");
    println!("| ---: | ---: | ---: | ---: | ---: | ---: |");
    let mut times = Vec::new();
    let mut e0 = Vec::new();
    let mut e1 = Vec::new();
    let mut e2 = Vec::new();
    for seed in [5u64, 8, 13, 21, 34] {
        let (truth, arr) = make_hawkes(0.8, 0.4, 1.5, 80.0, seed).unwrap();
        let t0 = Instant::now();
        let (fit, _) = ExponentialHawkes::mle(&arr, 0.0, 80.0, [0.5, 0.3, 1.2]).unwrap();
        let dt = t0.elapsed();
        times.push(ms(dt));
        e0.push((fit.mu - truth.mu).abs());
        e1.push((fit.alpha - truth.alpha).abs());
        e2.push((fit.beta - truth.beta).abs());
        println!(
            "| {} | {:.1} | {} | {:.4} | {:.4} | {:.4} |",
            seed,
            ms(dt),
            arr.len(),
            (fit.mu - truth.mu).abs(),
            (fit.alpha - truth.alpha).abs(),
            (fit.beta - truth.beta).abs()
        );
    }
    println!();
    println!(
        "平均時間 **{:.1} ms**。RMSE $(\\mu,\\alpha,\\beta)=({:.4},{:.4},{:.4})$。",
        mean(&times),
        rmse(&e0),
        rmse(&e1),
        rmse(&e2)
    );
    println!();
}

fn bench_hayashi_yoshida() {
    println!("## 4. Hayashi–Yoshida 共分散（非同期ブラウン）");
    println!();
    println!("単位拡散の相関ブラウン $\\rho=0.6$、$T=1$。細かい格子 $N=3000$ を独立に間引き、$\\approx 250$ tick。真値 $\\langle X,Y\\rangle_T=\\rho T=0.6$。5 本。");
    println!();
    println!("| seed | 時間 (ms) | $n_X$ | $n_Y$ | $\\widehat{{\\mathrm{{HY}}}}$ | 絶対誤差 |");
    println!("| ---: | ---: | ---: | ---: | ---: | ---: |");
    let mut times = Vec::new();
    let mut errs = Vec::new();
    let rho = 0.6;
    let t_end = 1.0;
    let n_fine = 3000usize;
    let dt = t_end / n_fine as f64;
    for seed in [2u64, 6, 14, 22, 38] {
        let mut rng = seed_rng(seed);
        let mut x = vec![0.0; n_fine + 1];
        let mut y = vec![0.0; n_fine + 1];
        for i in 1..=n_fine {
            let z1 = rng.sample(StandardNormal);
            let z2 = rng.sample(StandardNormal);
            let dwx = dt.sqrt() * z1;
            let dwy = dt.sqrt() * (rho * z1 + (1.0 - rho * rho).sqrt() * z2);
            x[i] = x[i - 1] + dwx;
            y[i] = y[i - 1] + dwy;
        }
        let keep = |rng: &mut SeededRng, vals: &[f64]| -> TickSeries {
            let mut ts = vec![0.0];
            let mut vs = vec![vals[0]];
            for i in 1..n_fine {
                if rng.next_f64() < 0.08 {
                    ts.push(i as f64 * dt);
                    vs.push(vals[i]);
                }
            }
            ts.push(t_end);
            vs.push(*vals.last().unwrap());
            TickSeries::new(ts, vs).unwrap()
        };
        let sx = keep(&mut rng, &x);
        let sy = keep(&mut rng, &y);
        let t0 = Instant::now();
        let hy = hayashi_yoshida(&sx, &sy);
        let dtm = t0.elapsed();
        times.push(ms(dtm));
        errs.push((hy - rho * t_end).abs());
        println!(
            "| {} | {:.3} | {} | {} | {:.4} | {:.4} |",
            seed,
            ms(dtm),
            sx.n(),
            sy.n(),
            hy,
            (hy - rho * t_end).abs()
        );
    }
    println!();
    println!(
        "平均時間 **{:.3} ms**。絶対誤差の RMSE **{:.4}**（真値 0.6）。",
        mean(&times),
        rmse(&errs)
    );
    println!();
}

fn bench_gbm_schemes() {
    println!("## 5. GBM：厳密スキーム対 Euler–Maruyama");
    println!();
    println!("$\\mu=0.08$, $\\sigma=0.25$, $S_0=1$, $T=2$, $n=400$。終端の $E[S_T]=e^{{\\mu T}}$ との誤差。5 本の 1 パス誤差ではなく、各 seed で 80 パスの標本平均。");
    println!();
    println!("| seed | 厳密 ms | Euler ms | 厳密 $|\\bar S_T-e^{{\\mu T}}|$ | Euler $|\\bar S_T-e^{{\\mu T}}|$ |");
    println!("| ---: | ---: | ---: | ---: | ---: |");
    let model = GeometricBrownianMotion::new(0.08, 0.25).unwrap();
    let samp = Sampling::from_terminal(2.0, 400).unwrap();
    let truth = (0.08_f64 * 2.0).exp();
    let mut te = Vec::new();
    let mut tu = Vec::new();
    let mut ee = Vec::new();
    let mut eu = Vec::new();
    for seed in [1u64, 4, 9, 16, 25] {
        let mut rng = seed_rng(seed);
        let t0 = Instant::now();
        let mut se = 0.0;
        for _ in 0..80 {
            let p = simulate_gbm_exact(&model, &samp, 1.0, &mut rng).unwrap();
            se += p.terminal()[0];
        }
        let de = t0.elapsed();
        let t0 = Instant::now();
        let mut su = 0.0;
        for _ in 0..80 {
            let p = simulate(
                &model,
                &samp,
                &[1.0],
                &mut rng,
                &SimConfig {
                    scheme: Scheme::EulerMaruyama,
                    ..SimConfig::default()
                },
            )
            .unwrap();
            su += p.terminal()[0];
        }
        let du = t0.elapsed();
        te.push(ms(de));
        tu.push(ms(du));
        ee.push((se / 80.0 - truth).abs());
        eu.push((su / 80.0 - truth).abs());
        println!(
            "| {} | {:.1} | {:.1} | {:.4} | {:.4} |",
            seed,
            ms(de),
            ms(du),
            (se / 80.0 - truth).abs(),
            (su / 80.0 - truth).abs()
        );
    }
    println!();
    println!(
        "平均時間 厳密 {:.1} ms / Euler {:.1} ms（80 パス）。平均絶対誤差 厳密 {:.4} / Euler {:.4}。",
        mean(&te),
        mean(&tu),
        mean(&ee),
        mean(&eu)
    );
    println!();
}

fn bench_dp_mixture() {
    println!("## 6. Dirichlet 過程混合（Neal 2000 Alg. 3）");
    println!();
    println!("3 正規 $N(-4,0.4^2)$, $N(0,0.4^2)$, $N(4,0.4^2)$、各 40 点。$\\alpha=0.4$, 40 sweep。精度は多数決ラベル合わせ（Hungarian ではない）。5 本。");
    println!();
    println!("| seed | 時間 (ms) | $\\hat K$ | 多数決精度 |");
    println!("| ---: | ---: | ---: | ---: |");
    let mut times = Vec::new();
    let mut accs = Vec::new();
    for seed in [10u64, 20, 30, 40, 50] {
        let (x, truth) = make_dp_gaussians(40, &[-4.0, 0.0, 4.0], 0.4, seed).unwrap();
        let mut rng = seed_rng(seed + 99);
        let t0 = Instant::now();
        let fit = dp_gaussian_mixture_gibbs(&x, 0.4, 0.4, 0.0, 4.0, 40, &mut rng).unwrap();
        let dt = t0.elapsed();
        let acc = majority_accuracy(&fit.assignments, &truth);
        times.push(ms(dt));
        accs.push(acc);
        println!(
            "| {} | {:.1} | {} | {:.3} |",
            seed,
            ms(dt),
            fit.n_clusters,
            acc
        );
    }
    println!();
    println!(
        "平均時間 **{:.1} ms**。平均多数決精度 **{:.3}**。",
        mean(&times),
        mean(&accs)
    );
    println!();
}

fn bench_ibp() {
    println!("## 7. 線形ガウス IBP の collapsed Gibbs");
    println!();
    println!("$N=28$, $D=3$, $\\alpha=1.2$, $\\sigma_X=0.15$, $\\sigma_A=1$, 10 sweep, $\\kappa_{{\\max}}=3$。Hamming は貪欲列マッチ（Hungarian ではない）。3 本。");
    println!();
    println!("| seed | 時間 (ms) | 真 $K$ | $\\hat K$ | Hamming / $(NK)$ |");
    println!("| ---: | ---: | ---: | ---: | ---: |");
    let mut times = Vec::new();
    let mut ham = Vec::new();
    for seed in [12u64, 18, 27] {
        let (x, z_true, _) = make_ibp_linear_gaussian(28, 1.2, 0.15, 1.0, 3, seed).unwrap();
        let mut rng = seed_rng(seed + 7);
        let t0 = Instant::now();
        let fit: IbpLinearGaussianFit =
            ibp_linear_gaussian_gibbs_ex(&x, 1.2, 0.15, 1.0, 10, 3, &mut rng).unwrap();
        let dt = t0.elapsed();
        let h = fit.z.hamming_after_greedy_match(&z_true).unwrap();
        let denom = (z_true.n * z_true.k.max(fit.z.k).max(1)) as f64;
        times.push(ms(dt));
        ham.push(h as f64 / denom);
        println!(
            "| {} | {:.1} | {} | {} | {:.3} |",
            seed,
            ms(dt),
            z_true.k,
            fit.z.k,
            h as f64 / denom
        );
    }
    println!();
    println!(
        "平均時間 **{:.1} ms**。平均正規化 Hamming **{:.3}**。",
        mean(&times),
        mean(&ham)
    );
    println!();
    println!("IBP はラベル（列の置換）と特徴数 $K$ が確率的なので、Hamming は参考値である。");
    println!();
}
