//! Tier-1 oracles for the 2026-08-30 audit (A1–A6 and the one-line bugs).
//!
//! These compare implementations to closed forms / KF / known vectors, not
//! mere finiteness.

use isuzu::control::hjb_1d;
use isuzu::filter::{
    kalman, particle_filter, regularized_particle_filter, FnSsm, LinearGaussian, ParticleConfig,
    RegularizedConfig, ResamplingScheme,
};
use isuzu::hft::refresh_times;
use isuzu::linalg::{col_from_slice, mat_from_row_slice, try_inverse, van_loan_discretize};
use isuzu::noise::fractional_gaussian_noise;
use isuzu::npbayes::{sample_stick_breaking, StickBreakingKind};
use isuzu::path::{Path, TickSeries};
use isuzu::prelude::*;

const LN_2PI: f64 = 1.8378770664093453;

fn linear_lg() -> LinearGaussian {
    LinearGaussian::new(
        mat_from_row_slice(1, 1, &[0.9]),
        mat_from_row_slice(1, 1, &[0.04]),
        mat_from_row_slice(1, 1, &[1.0]),
        mat_from_row_slice(1, 1, &[0.16]),
    )
    .unwrap()
}

fn n01<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u = rng.next_f64().max(f64::MIN_POSITIVE);
    let v = rng.next_f64();
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

fn simulate_lg(model: &LinearGaussian, n: usize, seed: u64) -> Path {
    let times: Vec<f64> = (0..=n).map(|i| i as f64).collect();
    let mut vals = vec![0.0; n + 1];
    let mut x = 0.0;
    let mut rng = seed_rng(seed);
    for i in 1..=n {
        x = model.f[(0, 0)] * x + model.q[(0, 0)].sqrt() * n01(&mut rng);
        vals[i] = model.h[(0, 0)] * x + model.r[(0, 0)].sqrt() * n01(&mut rng);
    }
    Path::new(times, vals, 1).unwrap()
}

#[test]
fn pf_constant_obs_density_is_t_times_c() {
    let q = mat_from_row_slice(1, 1, &[0.04]);
    let r = mat_from_row_slice(1, 1, &[1.0]);
    let model = FnSsm::new(
        1,
        1,
        |_t, _dt, x, out| out[0] = x[0],
        |_t, _x, out| out[0] = 0.0,
        q,
        r,
    )
    .unwrap();
    let n = 12;
    let times: Vec<f64> = (0..=n).map(|i| i as f64).collect();
    let obs = Path::new(times, vec![0.0; n + 1], 1).unwrap();
    let x0 = col_from_slice(&[0.0]);
    let p0 = mat_from_row_slice(1, 1, &[0.25]);
    let expected = n as f64 * (-0.5 * LN_2PI);
    for (n_part, ess, scheme) in [
        (50, 0.5, ResamplingScheme::Systematic),
        (80, 0.0, ResamplingScheme::Multinomial),
        (40, 1.0, ResamplingScheme::Residual),
    ] {
        let mut rng = seed_rng(3);
        let pf = particle_filter(
            &model,
            &obs,
            &x0,
            &p0,
            ParticleConfig {
                n_particles: n_part,
                ess_ratio: ess,
                resampling: scheme,
                store_particles: false,
            },
            &mut rng,
        )
        .unwrap();
        assert!(
            (pf.loglik - expected).abs() < 1e-9,
            "N={n_part} ess={ess} ll={} vs {expected}",
            pf.loglik
        );
    }
}

#[test]
fn pf_loglik_matches_kalman() {
    let model = linear_lg();
    let obs = simulate_lg(&model, 20, 42);
    let x0 = col_from_slice(&[0.0]);
    let p0 = mat_from_row_slice(1, 1, &[1.0]);
    let kf = kalman(&model, &obs, &x0, &p0).unwrap();
    let mut rng = seed_rng(7);
    let pf = particle_filter(
        &model,
        &obs,
        &x0,
        &p0,
        ParticleConfig {
            n_particles: 1500,
            ess_ratio: 0.5,
            ..ParticleConfig::default()
        },
        &mut rng,
    )
    .unwrap();
    let mut rng2 = seed_rng(8);
    let always = particle_filter(
        &model,
        &obs,
        &x0,
        &p0,
        ParticleConfig {
            n_particles: 1500,
            ess_ratio: 1.0,
            ..ParticleConfig::default()
        },
        &mut rng2,
    )
    .unwrap();
    let mut rng3 = seed_rng(9);
    let rpf = regularized_particle_filter(
        &model,
        &obs,
        &x0,
        &p0,
        RegularizedConfig {
            particle: ParticleConfig {
                n_particles: 1200,
                ess_ratio: 1.0,
                ..ParticleConfig::default()
            },
            bandwidth: 1.0,
        },
        &mut rng3,
    )
    .unwrap();
    // The old +T ln N inflation was ~60 nats at N=20; 3 nats is MC error.
    for (name, ll) in [
        ("sir", pf.loglik),
        ("always", always.loglik),
        ("rpf", rpf.loglik),
    ] {
        assert!(
            (ll - kf.loglik).abs() < 3.0,
            "{name} {ll} vs KF {}",
            kf.loglik
        );
    }
}

#[test]
fn van_loan_ou_q_not_euler() {
    let a = mat_from_row_slice(1, 1, &[-1.2]);
    let b = col_from_slice(&[0.0]);
    let g = mat_from_row_slice(1, 1, &[0.5]);
    let (_f, _u, q) = van_loan_discretize(&a, &b, &g, 1.0).unwrap();
    let exact = 0.25 * (1.0 - (-2.4_f64).exp()) / 2.4;
    let euler = 0.25;
    assert!((q[(0, 0)] - exact).abs() < 1e-10);
    assert!((q[(0, 0)] - euler).abs() > 0.1);
}

#[test]
fn kp15_gbm_one_step_includes_i111() {
    let mu = 0.1;
    let sig = 0.3;
    let m = GeometricBrownianMotion::new(mu, sig).unwrap();
    let dt = 0.2;
    let s = Sampling::from_terminal(dt, 1).unwrap();
    let x0 = 1.0;
    let dw = 0.4;
    // KP1.5 with the extra independent Z that enters ΔZ. We cannot pin Z,
    // but for GBM the ΔZ terms cancel, so the path depends only on ΔW.
    let p = simulate(
        &m,
        &s,
        &[x0],
        &mut seed_rng(1),
        &SimConfig {
            scheme: Scheme::KloedenPlaten15,
            increment_w: Some(vec![dw]),
            ..SimConfig::default()
        },
    )
    .unwrap();
    let i111 = (dw * dw * dw - 3.0 * dt * dw) / 6.0;
    let analytic = x0
        * (1.0
            + mu * dt
            + sig * dw
            + 0.5 * sig * sig * (dw * dw - dt)
            + mu * sig * dw * dt
            + 0.5 * mu * mu * dt * dt
            + sig * sig * sig * i111);
    assert!(
        (p.terminal()[0] - analytic).abs() < 1e-12,
        "kp15 {} vs {analytic}",
        p.terminal()[0]
    );
}

#[test]
fn davies_harte_high_hurst() {
    let mut rng = seed_rng(4);
    for n in [64usize, 100, 128] {
        let x = fractional_gaussian_noise(n, 0.9, &mut rng).expect("H=0.9 must embed");
        assert_eq!(x.len(), n);
        assert!(x.iter().all(|v| v.is_finite()));
        let x95 = fractional_gaussian_noise(n, 0.95, &mut rng).expect("H=0.95 must embed");
        assert_eq!(x95.len(), n);
    }
}

#[test]
fn gbm_exact_scheme_matches_closed_form() {
    let mu = 0.05;
    let sig = 0.2;
    let m = GeometricBrownianMotion::new(mu, sig).unwrap();
    let dt = 1.0;
    let s = Sampling::from_terminal(dt, 1).unwrap();
    let x0 = 100.0;
    let dw = 0.35;
    let p = simulate(
        &m,
        &s,
        &[x0],
        &mut seed_rng(2),
        &SimConfig {
            scheme: Scheme::Exact,
            increment_w: Some(vec![dw]),
            ..SimConfig::default()
        },
    )
    .unwrap();
    let analytic = x0 * ((mu - 0.5 * sig * sig) * dt + sig * dw).exp();
    assert!((p.terminal()[0] - analytic).abs() < 1e-12);
}

#[test]
fn refresh_times_async_not_intersection() {
    let x = TickSeries {
        times: vec![0.0, 0.3, 0.9],
        values: vec![1.0, 2.0, 3.0],
    };
    let y = TickSeries {
        times: vec![0.1, 0.5, 1.0],
        values: vec![10.0, 20.0, 30.0],
    };
    let (t, xv, yv) = refresh_times(&x, &y).unwrap();
    assert!(t.len() >= 2);
    assert_eq!(t[0], 0.1);
    assert_eq!(xv[0], 1.0);
    assert_eq!(yv[0], 10.0);
}

#[test]
fn try_inverse_none_if_singular() {
    let z = isuzu::linalg::mat_zeros(2, 2);
    assert!(try_inverse(&z).is_none());
}

#[test]
fn pitman_yor_first_stick_mean() {
    // V_1 ~ Beta(1−d, θ+d) ⇒ E[π_1] = (1−d)/(1+θ) for residual-atom K large.
    let d = 0.5;
    let theta = 1.0;
    let expect = (1.0 - d) / (1.0 + theta); // 0.25
    let mut rng = seed_rng(11);
    let mut s = 0.0;
    let n = 8_000;
    for _ in 0..n {
        let sb = sample_stick_breaking(
            StickBreakingKind::PitmanYor {
                discount: d,
                strength: theta,
            },
            40,
            &mut rng,
        )
        .unwrap();
        s += sb.weights[0];
    }
    let m = s / n as f64;
    assert!((m - expect).abs() < 0.02, "E[π1]={m} vs {expect}");
}

#[test]
fn hjb_rejects_cfl_violation() {
    let x = [0.0, 1.0, 2.0];
    let u = [0.0];
    let err = hjb_1d(
        &x,
        &u,
        2,
        1.0,
        |_x, _u| 0.0,
        |_x, _u| 10.0,
        |_x, _u| 0.0,
        |_x| 0.0,
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("CFL") || msg.contains("cfl"), "{msg}");
}

#[test]
fn gamma_process_is_not_err() {
    let s = Sampling::from_terminal(1.0, 50).unwrap();
    let p = isuzu::models::gamma_process(&s, 2.0, 0.5, &mut seed_rng(3)).unwrap();
    assert!(p.terminal()[0].is_finite());
    assert!(p.terminal()[0] > 0.0);
}

#[test]
fn weibull_mle_exists_and_uses_censoring() {
    let w = WeibullRenewal::new(1.0, 0.5).unwrap();
    let mut rng = seed_rng(6);
    let arr = w.simulate(0.0, 8.0, &mut rng).unwrap();
    let (fit, ll) = WeibullRenewal::mle(&arr, 0.0, 8.0, [0.8, 0.6]).unwrap();
    assert!(ll.is_finite());
    assert!(fit.shape > 0.0 && fit.scale > 0.0);
    // Survival term makes empty-path loglik = −(T/scale)^k, not 0.
    let empty = WeibullRenewal::new(2.0, 1.0)
        .unwrap()
        .loglik(&[], 0.0, 1.0)
        .unwrap();
    assert!((empty + 1.0).abs() < 1e-14, "empty ll {empty}");
}
