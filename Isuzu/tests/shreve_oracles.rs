//! Shreve roadmap completion-condition oracles (analytic / MC / PDE / tree).
//!
//! These check the explicit 完了条件: SE scaling, control-variate VR,
//! discounted GBM martingale, LSM vs European vs dual, bond and Merton MC,
//! T-forward identity, and cross-engine agreement.

use isuzu::finance::black_scholes::{call as bs_call, put as bs_put, BlackScholesMarket};
use isuzu::finance::jumps::{merton_call, merton_call_mc, merton_call_pide};
use isuzu::finance::market::{asset_forward, t_forward_price, FlatCurve};
use isuzu::finance::monte_carlo::{price_sde, VarianceReduction};
use isuzu::finance::payoff::{EuropeanCall, PathPayoff};
use isuzu::finance::pde::cn_call_error;
use isuzu::finance::rates::{cir_bond, vasicek_bond};
use isuzu::finance::stopping::{
    european_put_dual_upper, lsm_american_put, LongstaffSchwartzConfig,
};
use isuzu::finance::tree::{crr_vs_bs_call, trinomial_price};
use isuzu::finance::{DiscountCurve, MonteCarloEstimate};
use isuzu::models::{Cir, GeometricBrownianMotion, OrnsteinUhlenbeck};
use isuzu::path::Path;
use isuzu::prelude::*;
use isuzu::scheme::Scheme;

struct IdentitySpot;

impl PathPayoff for IdentitySpot {
    fn payoff(&self, path: &Path) -> isuzu::Result<f64> {
        Ok(path.terminal()[0])
    }
}

fn exact_sim() -> SimConfig {
    SimConfig {
        scheme: Scheme::Exact,
        ..SimConfig::default()
    }
}

fn price_call_mc(n_paths: usize, vr: VarianceReduction, seed: u64) -> MonteCarloEstimate {
    let model = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
    let samp = Sampling::from_terminal(1.0, 4).unwrap();
    let pay = EuropeanCall::new(0, 100.0).unwrap();
    let mut rng = seed_rng(seed);
    price_sde(
        &model,
        &pay,
        &samp,
        &[100.0],
        0.05,
        n_paths,
        &exact_sim(),
        vr,
        Some(100.0),
        &mut rng,
    )
    .unwrap()
}

#[test]
fn mc_standard_error_scales_like_sqrt_n() {
    let a = price_call_mc(
        400,
        VarianceReduction {
            antithetic: false,
            european_control: false,
        },
        11,
    );
    let b = price_call_mc(
        1_600,
        VarianceReduction {
            antithetic: false,
            european_control: false,
        },
        12,
    );
    let ratio = a.standard_error / b.standard_error;
    assert!(
        (1.4..=2.8).contains(&ratio),
        "SE ratio {ratio} ({} vs {}) not ~2",
        a.standard_error,
        b.standard_error
    );
}

#[test]
fn european_control_reduces_variance() {
    let raw = price_call_mc(
        1_500,
        VarianceReduction {
            antithetic: false,
            european_control: false,
        },
        21,
    );
    let cv = price_call_mc(
        1_500,
        VarianceReduction {
            antithetic: false,
            european_control: true,
        },
        21,
    );
    let vr = cv
        .diagnostics
        .variance_reduction_ratio
        .expect("control variate should report a VR ratio");
    assert!(vr < 1.0, "VR ratio {vr} should be < 1");
    assert!(
        cv.diagnostics.sample_variance < raw.diagnostics.sample_variance,
        "cv var {} raw {}",
        cv.diagnostics.sample_variance,
        raw.diagnostics.sample_variance
    );
}

#[test]
fn discounted_gbm_is_a_martingale() {
    let model = GeometricBrownianMotion::new(0.05, 0.25).unwrap();
    let samp = Sampling::from_terminal(1.0, 8).unwrap();
    let mut rng = seed_rng(3);
    let est = price_sde(
        &model,
        &IdentitySpot,
        &samp,
        &[100.0],
        0.05,
        4_000,
        &exact_sim(),
        VarianceReduction::default(),
        None,
        &mut rng,
    )
    .unwrap();
    assert!(
        (est.estimate - 100.0).abs() < 3.0 * est.standard_error + 0.05,
        "E[e^{{-rT}} S_T]={} se={}",
        est.estimate,
        est.standard_error
    );
}

#[test]
fn lsm_sits_between_european_and_dual() {
    let mut rng = seed_rng(9);
    let cfg = LongstaffSchwartzConfig::<isuzu::finance::PolynomialBasis>::default();
    let lsm = lsm_american_put(
        36.0,
        40.0,
        0.06,
        0.2,
        1.0,
        40,
        2_000,
        2_000,
        &cfg,
        &exact_sim(),
        &mut rng,
    )
    .unwrap();
    let euro = bs_put(
        &BlackScholesMarket::new(36.0, 0.06, 0.0, 0.2, 1.0).unwrap(),
        40.0,
    )
    .unwrap()
    .price;
    let dual = european_put_dual_upper(36.0, 40.0, 0.06, 0.2, 1.0, 20, 800, &exact_sim(), &mut rng)
        .unwrap();
    assert!(
        lsm.lower_bound.estimate + 0.05 >= euro,
        "LSM {} < euro {euro}",
        lsm.lower_bound.estimate
    );
    assert!(
        lsm.lower_bound.estimate <= dual.estimate + 3.0 * dual.standard_error + 0.25,
        "LSM {} > dual {}",
        lsm.lower_bound.estimate,
        dual.estimate
    );
}

#[test]
fn vasicek_and_cir_bond_match_mc_discount() {
    let v = OrnsteinUhlenbeck::new(0.5, 0.03, 0.02).unwrap();
    let analytic = vasicek_bond(&v, 0.03, 1.0).unwrap().price;
    let samp = Sampling::from_terminal(1.0, 40).unwrap();
    let mut rng = seed_rng(4);
    let mut acc = isuzu::finance::OnlineMoments::default();
    for _ in 0..2_000 {
        let p = simulate(&v, &samp, &[0.03], &mut rng, &exact_sim()).unwrap();
        let mut integ = 0.0;
        for i in 0..p.n_steps() {
            let dt = p.times()[i + 1] - p.times()[i];
            integ += 0.5 * (p.state(i)[0] + p.state(i + 1)[0]) * dt;
        }
        acc.push((-integ).exp());
    }
    let mc = MonteCarloEstimate::from_moments(&acc, 1.96);
    assert!(
        (mc.estimate - analytic).abs() < 3.0 * mc.standard_error + 0.002,
        "Vasicek bond MC {} vs {analytic}",
        mc.estimate
    );

    let c = Cir::new(0.8, 0.04, 0.08).unwrap();
    let cir_a = cir_bond(&c, 0.04, 1.0).unwrap().price;
    let mut acc2 = isuzu::finance::OnlineMoments::default();
    let n = 40;
    let dt = 1.0 / n as f64;
    for _ in 0..1_500 {
        let mut r = 0.04;
        let mut integ = 0.0;
        for _ in 0..n {
            let r1 = c.sample_exact(r, dt, &mut rng).unwrap();
            integ += 0.5 * (r + r1) * dt;
            r = r1;
        }
        acc2.push((-integ).exp());
    }
    let mc2 = MonteCarloEstimate::from_moments(&acc2, 1.96);
    assert!(
        (mc2.estimate - cir_a).abs() < 3.0 * mc2.standard_error + 0.004,
        "CIR bond MC {} vs {cir_a}",
        mc2.estimate
    );
}

#[test]
fn merton_series_matches_mc_and_pide() {
    let series = merton_call(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 0.8, -0.1, 0.15, 20).unwrap();
    let mut rng = seed_rng(6);
    let mc = merton_call_mc(
        100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 0.8, -0.1, 0.15, 40, 3_000, &mut rng,
    )
    .unwrap();
    assert!(
        (mc.estimate - series).abs() < 3.0 * mc.standard_error + 0.15,
        "Merton MC {} vs series {series}",
        mc.estimate
    );
    let pide =
        merton_call_pide(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 0.8, -0.1, 0.15, 90, 45).unwrap();
    assert!(
        (pide - series).abs() < 0.25,
        "PIDE {pide} vs series {series}"
    );
}

#[test]
fn cross_engine_atm_call() {
    let mkt = BlackScholesMarket::new(100.0, 0.05, 0.0, 0.2, 1.0).unwrap();
    let analytic = bs_call(&mkt, 100.0).unwrap().price;
    let (crr, _) = crr_vs_bs_call(100.0, 100.0, 0.05, 0.2, 1.0, 200).unwrap();
    let pde_err = cn_call_error(100.0, 100.0, 0.05, 0.2, 1.0, 80, 80).unwrap();
    let tri = trinomial_price(100.0, 100.0, 0.05, 0.2, 1.0, 80, true, false).unwrap();
    let mc = price_call_mc(
        6_000,
        VarianceReduction {
            antithetic: true,
            european_control: true,
        },
        7,
    );
    assert!((crr - analytic).abs() < 0.08, "CRR {crr} vs {analytic}");
    assert!(pde_err.abs() < 0.15, "CN error {pde_err}");
    assert!(
        (tri - analytic).abs() < 0.1,
        "trinomial {tri} vs {analytic}"
    );
    assert!(
        (mc.estimate - analytic).abs() < 3.0 * mc.standard_error + 0.05,
        "MC {} vs {analytic}",
        mc.estimate
    );
}

#[test]
fn t_forward_equals_cash_over_bond() {
    let mkt = BlackScholesMarket::new(100.0, 0.05, 0.0, 0.2, 1.0).unwrap();
    let cash = bs_call(&mkt, 100.0).unwrap().price;
    let p0t = FlatCurve::new(0.05).unwrap().discount(1.0).unwrap();
    let fwd = t_forward_price(cash, p0t).unwrap();
    assert!((fwd - cash / p0t).abs() < 1e-14);
    assert!((asset_forward(100.0, 0.05, 0.0, 1.0).unwrap() - mkt.forward()).abs() < 1e-14);
    // Undiscounted MC of (S_T−K)+ equals the T-forward price.
    let model = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
    let samp = Sampling::from_terminal(1.0, 4).unwrap();
    let pay = EuropeanCall::new(0, 100.0).unwrap();
    let mut rng = seed_rng(8);
    let mut acc = isuzu::finance::OnlineMoments::default();
    for _ in 0..4_000 {
        let p = simulate(&model, &samp, &[100.0], &mut rng, &exact_sim()).unwrap();
        acc.push(PathPayoff::payoff(&pay, &p).unwrap());
    }
    let undisc = MonteCarloEstimate::from_moments(&acc, 1.96);
    assert!(
        (undisc.estimate - fwd).abs() < 3.0 * undisc.standard_error + 0.05,
        "E[(S_T-K)+]={} vs T-forward {fwd}",
        undisc.estimate
    );
}

#[test]
fn call_monotone_and_seed_reproducible() {
    let lo = BlackScholesMarket::new(90.0, 0.05, 0.0, 0.2, 1.0).unwrap();
    let hi = BlackScholesMarket::new(110.0, 0.05, 0.0, 0.2, 1.0).unwrap();
    assert!(bs_call(&hi, 100.0).unwrap().price > bs_call(&lo, 100.0).unwrap().price);
    let thin = BlackScholesMarket::new(100.0, 0.05, 0.0, 0.1, 1.0).unwrap();
    let fat = BlackScholesMarket::new(100.0, 0.05, 0.0, 0.4, 1.0).unwrap();
    assert!(bs_call(&fat, 100.0).unwrap().price > bs_call(&thin, 100.0).unwrap().price);
    let p_lo = bs_put(&lo, 90.0).unwrap().price;
    let p_hi = bs_put(&lo, 110.0).unwrap().price;
    assert!(p_hi > p_lo);
    let a = price_call_mc(800, VarianceReduction::default(), 99);
    let b = price_call_mc(800, VarianceReduction::default(), 99);
    assert!((a.estimate - b.estimate).abs() < 1e-14);
    assert!(BlackScholesMarket::new(f64::NAN, 0.05, 0.0, 0.2, 1.0).is_err());
}
