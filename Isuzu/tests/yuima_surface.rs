//! End-to-end checks that the YUIMA-class surface actually runs.

use isuzu::prelude::*;
use isuzu::simulate::simulate_cogarch;

#[test]
fn cir_stays_nonnegative() {
    let m = Cir::new(2.0, 0.08, 0.2).unwrap();
    let s = Sampling::from_terminal(2.0, 800).unwrap();
    let mut rng = seed_rng(4);
    let p = simulate(&m, &s, &[0.08], &mut rng, &SimConfig::default()).unwrap();
    assert!(p.component(0).unwrap().iter().all(|&x| x >= -1e-12));
}

#[test]
fn heston_runs() {
    let m = Heston::new(0.05, 1.5, 0.04, 0.3, -0.6).unwrap();
    let s = Sampling::from_terminal(1.0, 400).unwrap();
    let mut rng = seed_rng(5);
    let p = simulate(&m, &s, &[100.0, 0.04], &mut rng, &SimConfig::default()).unwrap();
    assert_eq!(p.dim(), 2);
    assert!(p.component(0).unwrap().iter().all(|&x| x.is_finite()));
}

#[test]
fn merton_has_jumps() {
    let m = MertonJumpDiffusion::new(0.05, 0.15, 3.0, -0.05, 0.1).unwrap();
    let s = Sampling::from_terminal(1.0, 500).unwrap();
    let mut rng = seed_rng(6);
    let p = simulate(&m, &s, &[100.0], &mut rng, &SimConfig::default()).unwrap();
    let dx = p.increments(0).unwrap();
    let max_abs = dx.iter().map(|x| x.abs()).fold(0.0, f64::max);
    assert!(max_abs > 0.0);
}

#[test]
fn fractional_gbm_runs() {
    let m = FractionalGbm::new(0.0, 0.2, 0.7).unwrap();
    let s = Sampling::from_terminal(1.0, 256).unwrap();
    let mut rng = seed_rng(8);
    let p = simulate(&m, &s, &[1.0], &mut rng, &SimConfig::default()).unwrap();
    assert_eq!(p.n_steps(), 256);
    assert!(p.terminal()[0] > 0.0);
}

#[test]
fn carma_gaussian_driver() {
    let m = Carma::new(
        vec![1.2, 0.8],
        vec![1.0, 0.3],
        0.0,
        LevyMeasure::Gaussian {
            mu: 0.0,
            sigma: 0.4,
        },
    )
    .unwrap();
    assert!(m.is_causal());
    let s = Sampling::from_terminal(2.0, 300).unwrap();
    let mut rng = seed_rng(10);
    let p = simulate(&m, &s, &[0.0, 0.0], &mut rng, &SimConfig::default()).unwrap();
    assert_eq!(p.dim(), 2);
}

#[test]
fn cogarch_cp_runs() {
    let m = Cogarch::cogarch11(
        0.05,
        0.7,
        0.25,
        LevyMeasure::CompoundPoisson {
            intensity: 4.0,
            law: JumpLaw::Normal {
                mu: 0.0,
                sigma: 0.3,
            },
        },
    )
    .unwrap();
    let s = Sampling::from_terminal(1.0, 200).unwrap();
    let mut rng = seed_rng(12);
    let p = simulate_cogarch(&m, &s, 0.0, &[0.1], &mut rng).unwrap();
    assert_eq!(p.dim(), 2);
}

#[test]
fn yuima_object_roundtrip() {
    let model = GeometricBrownianMotion::new(0.1, 0.2).unwrap();
    let sampling = Sampling::from_terminal(0.5, 100).unwrap();
    let mut y = Yuima::new(model, sampling, vec![1.0]).unwrap();
    let mut rng = seed_rng(13);
    y.simulate(&mut rng, Scheme::Milstein).unwrap();
    assert_eq!(y.data().unwrap().n_steps(), 100);
}

#[test]
fn change_point_finds_vol_jump() {
    // Brownian with σ=1 then σ=3
    let mut rng = seed_rng(14);
    let left = Sampling::from_terminal(1.0, 400).unwrap();
    let right = Sampling::regular(1.0, 2.0, 400).unwrap();
    let bm = FnSde::new(1, 1, |_t, _x, a| a[0] = 0.0, |_t, _x, s| s[0] = 1.0);
    let bm3 = FnSde::new(1, 1, |_t, _x, a| a[0] = 0.0, |_t, _x, s| s[0] = 3.0);
    let p1 = simulate(&bm, &left, &[0.0], &mut rng, &SimConfig::default()).unwrap();
    let p2 = simulate(
        &bm3,
        &right,
        &[*p1.terminal().first().unwrap()],
        &mut rng,
        &SimConfig::default(),
    )
    .unwrap();
    let mut times = p1.times()[..p1.n_nodes() - 1].to_vec();
    times.extend_from_slice(p2.times());
    let mut values = p1.as_univariate().unwrap();
    values.pop();
    values.extend(p2.as_univariate().unwrap());
    let path = Path::new(times, values, 1).unwrap();
    let cp = change_point_qv(&path, 0).unwrap();
    assert!(
        (cp.time - 1.0).abs() < 0.25,
        "change-point time {}",
        cp.time
    );
}

#[test]
fn bns_detects_merton_jumps() {
    let m = MertonJumpDiffusion::new(0.0, 0.1, 8.0, 0.0, 0.4).unwrap();
    let s = Sampling::from_terminal(1.0, 800).unwrap();
    let mut rng = seed_rng(15);
    let p = simulate(&m, &s, &[1.0], &mut rng, &SimConfig::default()).unwrap();
    let test = bns_jump_test(&p, 0).unwrap();
    assert!(test.jump_component > 0.0);
}

#[test]
fn milstein_gbm_one_step_closed_form() {
    let mu = 0.05;
    let sig = 0.2;
    let m = GeometricBrownianMotion::new(mu, sig).unwrap();
    let s = Sampling::from_terminal(0.25, 1).unwrap();
    let x0 = 100.0;
    let dw = 0.3;
    let p = simulate(
        &m,
        &s,
        &[x0],
        &mut seed_rng(16),
        &SimConfig {
            scheme: Scheme::Milstein,
            increment_w: Some(vec![dw]),
            ..SimConfig::default()
        },
    )
    .unwrap();
    let dt = 0.25;
    let analytic = x0 + mu * x0 * dt + sig * x0 * dw + 0.5 * sig * sig * x0 * (dw * dw - dt);
    assert!(
        (p.terminal()[0] - analytic).abs() < 1e-12,
        "milstein {} vs {analytic}",
        p.terminal()[0]
    );
}

#[test]
fn fn_sde_time_inhomogeneous() {
    let m = FnSde::new(1, 1, |t, _x, a| a[0] = t, |_t, _x, s| s[0] = 0.1);
    let s = Sampling::from_terminal(1.0, 100).unwrap();
    let mut rng = seed_rng(17);
    let p = simulate(&m, &s, &[0.0], &mut rng, &SimConfig::default()).unwrap();
    // E[X_1] ≈ ∫ t dt = 1/2
    assert!((p.terminal()[0] - 0.5).abs() < 0.2);
}

#[test]
fn lse_ou_drift() {
    let truth = OrnsteinUhlenbeck::new(2.0, 0.0, 0.3).unwrap();
    let s = Sampling::from_terminal(12.0, 2500).unwrap();
    let mut rng = seed_rng(18);
    let path = simulate_ou_exact(&truth, &s, 0.0, &mut rng).unwrap();
    // LSE identifies the drift; pin σ (it does not enter the contrast).
    let fit = lse(
        &truth,
        &path,
        &[1.0, 0.2, 0.3],
        Some(&[0.1, -1.0, 0.3]),
        Some(&[5.0, 1.0, 0.3]),
        OptOptions::default(),
    )
    .unwrap();
    assert!(
        fit.params[0] > 0.4 && fit.params[0] < 6.0 && fit.params[1].abs() < 0.35,
        "lse {:?}",
        fit.params
    );
}
