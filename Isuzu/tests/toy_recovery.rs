//! Toy-data generation + parameter / moment recovery across the catalogue.

use isuzu::api::{recover, QmleEstimator};
use isuzu::control::{kelly_fraction, MertonPortfolio};
use isuzu::datasets::{make_carma_hawkes, make_cir, make_gbm, make_hawkes, make_ou};
use isuzu::hft::{roll_spread, two_scale_rv, Acd11, AlmgrenChriss};
use isuzu::malliavin::{asymptotic_term, moment_summary};
use isuzu::models::point_more::WeibullRenewal;
use isuzu::models::{Ckls, Jacobi};
use isuzu::optimize::OptOptions;
use isuzu::prelude::*;

#[test]
fn recover_ou_sigma() {
    let toy = make_ou(1.4, 0.0, 0.45, 12.0, 2500, 21).unwrap();
    let mut est = QmleEstimator::new(toy.model.clone(), vec![1.0, 0.1, 0.7])
        .bounds(vec![0.1, -1.0, 0.05], vec![4.0, 1.0, 2.0]);
    let report = recover(&mut est, &toy.path, &toy.truth).unwrap();
    assert!(
        report.abs_error[2] < 0.12,
        "OU sigma err {:?}",
        report.abs_error
    );
}

#[test]
fn recover_gbm_sigma() {
    let toy = make_gbm(0.08, 0.25, 2.0, 800, 3).unwrap();
    let mut est = QmleEstimator::new(toy.model.clone(), vec![0.0, 0.4])
        .bounds(vec![-1.0, 0.05], vec![1.0, 1.0]);
    est.opt = OptOptions {
        max_iter: 200,
        ..OptOptions::default()
    };
    let report = recover(&mut est, &toy.path, &toy.truth).unwrap();
    assert!(report.abs_error[1] < 0.12, "GBM {:?}", report.abs_error);
}

#[test]
fn cir_mean_near_theta() {
    let toy = make_cir(2.0, 0.09, 0.15, 4.0, 800, 8).unwrap();
    let x = toy.path.as_univariate().unwrap();
    let m = x.iter().sum::<f64>() / x.len() as f64;
    assert!((m - 0.09).abs() < 0.05, "CIR mean {m}");
}

#[test]
fn hawkes_mle_recovers_mu() {
    let (truth, arr) = make_hawkes(0.8, 0.4, 1.5, 120.0, 5).unwrap();
    let (fit, _) = ExponentialHawkes::mle(&arr, 0.0, 120.0, [0.5, 0.3, 1.2]).unwrap();
    assert!(
        (fit.mu - truth.mu).abs() < 0.45,
        "hawkes mu {} vs {}",
        fit.mu,
        truth.mu
    );
}

#[test]
fn carma_hawkes_loglik_finite() {
    let (model, arr) = make_carma_hawkes(0.5, vec![1.1], vec![0.4], 15.0, 9).unwrap();
    let ll = model.loglik(&arr, 0.0, 15.0).unwrap();
    assert!(ll.is_finite());
}

#[test]
fn ckls_and_jacobi_simulate() {
    let c = Ckls::cev(0.05, 0.2, 0.8).unwrap();
    let j = Jacobi::new(3.0, 0.4, 0.3).unwrap();
    let s = Sampling::from_terminal(1.0, 200).unwrap();
    let mut rng = seed_rng(11);
    let p1 = simulate(&c, &s, &[1.0], &mut rng, &SimConfig::default()).unwrap();
    let p2 = simulate(&j, &s, &[0.4], &mut rng, &SimConfig::default()).unwrap();
    assert!(p1.terminal()[0].is_finite());
    assert!(p2
        .component(0)
        .unwrap()
        .iter()
        .all(|x| (0.0..=1.0).contains(x) || x.abs() < 0.05));
}

#[test]
fn weibull_loglik_prefers_truth() {
    let w = WeibullRenewal::new(1.5, 0.4).unwrap();
    let mut rng = seed_rng(12);
    let arr = w.simulate(0.0, 30.0, &mut rng).unwrap();
    let ll0 = w.loglik(&arr, 0.0, 30.0).unwrap();
    let wrong = WeibullRenewal::new(3.0, 1.2).unwrap();
    let ll1 = wrong.loglik(&arr, 0.0, 30.0).unwrap();
    assert!(ll0 > ll1, "ll truth {ll0} vs wrong {ll1}");
}

#[test]
fn merton_closed_form() {
    let m = MertonPortfolio::new(0.12, 0.02, 0.2, 2.0).unwrap();
    assert!((m.myopic_weight() - 1.25).abs() < 1e-12);
    assert!((kelly_fraction(0.12, 0.02, 0.2).unwrap() - 2.5).abs() < 1e-12);
}

#[test]
fn almgren_chriss_and_acd() {
    let ac = AlmgrenChriss {
        x0: 100.0,
        n: 20,
        tau: 0.05,
        sigma: 0.3,
        eta: 0.02,
        gamma: 0.001,
        lambda: 1e-5,
    };
    let (h, tr) = ac.schedule().unwrap();
    assert!((h[0] - 100.0).abs() < 1e-9);
    assert!(tr.iter().sum::<f64>() > 99.0);

    let durs: Vec<f64> = (0..80).map(|i| 0.2 + 0.01 * ((i % 7) as f64)).collect();
    let (fit, ll) = Acd11::mle(&durs, [0.05, 0.1, 0.4]).unwrap();
    assert!(ll.is_finite());
    assert!(fit.omega > 0.0);
}

#[test]
fn hft_roll_and_tsrv() {
    let toy = make_gbm(0.0, 0.2, 1.0, 400, 14).unwrap();
    let _ = roll_spread(&toy.path, 0).unwrap();
    let tsrv = two_scale_rv(&toy.path, 0, 5).unwrap();
    assert!(tsrv >= 0.0);
}

#[test]
fn malliavin_ou_moments() {
    let m = OrnsteinUhlenbeck::new(1.5, 0.0, 0.25).unwrap();
    let s = Sampling::from_terminal(0.4, 60).unwrap();
    let mut rng = seed_rng(16);
    let sum = moment_summary(&m, &s, 0.8, 800, &mut rng).unwrap();
    let exact = 0.8 * (-1.5 * 0.4_f64).exp();
    assert!((sum.mean - exact).abs() < 0.12);
    let ae = asymptotic_term(&m, 0.8, 0.4, |x| x, |_| 0.0, 30).unwrap();
    assert!((ae.d0 - exact).abs() < 0.05);
}

#[test]
fn lead_lag_and_hy_still_work() {
    let toy = make_gbm(0.0, 0.3, 1.0, 300, 19).unwrap();
    let a = TickSeries::new(toy.path.times().to_vec(), toy.path.component(0).unwrap()).unwrap();
    let b = a.shift_time(0.03);
    let grid = lead_lag_grid(-0.08, 0.08, 17).unwrap();
    let ll = lead_lag(&a, &b, &grid).unwrap();
    assert!((ll.theta + 0.03).abs() < 0.021);
    let hy = hayashi_yoshida(&a, &a);
    assert!((hy - a.increments().iter().map(|d| d * d).sum::<f64>()).abs() < 1e-10);
}
