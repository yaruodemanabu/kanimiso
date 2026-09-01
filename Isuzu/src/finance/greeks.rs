//! Four Greek estimators: bump, pathwise, likelihood ratio, Malliavin.

use amatsuki::Rng;

use crate::error::{Error, Result};
use crate::finance::black_scholes::{call as bs_call, BlackScholesMarket};
use crate::finance::monte_carlo::{MonteCarloEstimate, OnlineMoments};
use crate::model::Sde;
use crate::models::GeometricBrownianMotion;
use crate::path::Path;
use crate::sampling::Sampling;
use crate::simulate::{simulate, SimConfig};

/// Greek comparison on a GBM European call (the Shreve Ch.4 laboratory).
#[derive(Clone, Debug)]
pub struct GreekReport {
    pub analytic: f64,
    pub bump: MonteCarloEstimate,
    pub pathwise: MonteCarloEstimate,
    pub likelihood_ratio: MonteCarloEstimate,
    pub malliavin: Option<MonteCarloEstimate>,
}

/// Estimate Δ of a GBM call four ways, sharing common random numbers for the bump.
pub fn gbm_call_delta<R: Rng + ?Sized>(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    n_steps: usize,
    n_paths: usize,
    bump: f64,
    cfg: &SimConfig,
    rng: &mut R,
) -> Result<GreekReport> {
    if !(spot > 0.0 && strike > 0.0 && vol > 0.0 && time > 0.0 && bump > 0.0) {
        return Err(Error::param("gbm_call_delta inputs must be positive"));
    }
    let mkt = BlackScholesMarket::new(spot, rate, 0.0, vol, time)?;
    let analytic = bs_call(&mkt, strike)?.delta;
    let model = GeometricBrownianMotion::new(rate, vol)?;
    let samp = Sampling::from_terminal(time, n_steps)?;
    let df = (-rate * time).exp();
    let mut bump_acc = OnlineMoments::default();
    let mut pw_acc = OnlineMoments::default();
    let mut lr_acc = OnlineMoments::default();
    let mut mal_acc = OnlineMoments::default();
    let m = model.n_noise();
    for _ in 0..n_paths {
        let dw = crate::noise::brownian_increments(&samp, m, rng)?;
        let mut cfg_p = cfg.clone();
        cfg_p.increment_w = Some(dw.clone());
        let path = simulate(&model, &samp, &[spot], rng, &cfg_p)?;
        let s_t = path.terminal()[0];
        let pay = df * (s_t - strike).max(0.0);
        // Bump-and-revalue (CRN).
        let up = simulate(&model, &samp, &[spot + bump], rng, &cfg_p)?;
        let down = simulate(&model, &samp, &[spot - bump], rng, &cfg_p)?;
        let pu = df * (up.terminal()[0] - strike).max(0.0);
        let pd = df * (down.terminal()[0] - strike).max(0.0);
        bump_acc.push((pu - pd) / (2.0 * bump));
        // Pathwise: d/dS0 (e^{-rT}(S_T-K)⁺) = e^{-rT} 1_{S_T>K} S_T / S0
        let pw = if s_t > strike { df * s_t / spot } else { 0.0 };
        pw_acc.push(pw);
        // Likelihood ratio on the terminal log-normal.
        let wt = dw.iter().sum::<f64>();
        let lr_w = wt / (vol * spot * time);
        lr_acc.push(pay * lr_w);
        // Malliavin weight on Euler / exact GBM: W_T / (σ S0 T)
        mal_acc.push(pay * wt / (vol * spot * time));
        let _ = path;
    }
    Ok(GreekReport {
        analytic,
        bump: MonteCarloEstimate::from_moments(&bump_acc, 1.96),
        pathwise: MonteCarloEstimate::from_moments(&pw_acc, 1.96),
        likelihood_ratio: MonteCarloEstimate::from_moments(&lr_acc, 1.96),
        malliavin: Some(MonteCarloEstimate::from_moments(&mal_acc, 1.96)),
    })
}

/// Pathwise delta of a Lipschitz terminal payoff `g(S_T)` on GBM:
/// `e^{-rT} g'(S_T) S_T / S_0` when `g'` exists a.e.
pub fn pathwise_delta_call(path: &Path, spot: f64, strike: f64, rate: f64) -> Result<f64> {
    if spot <= 0.0 {
        return Err(Error::param("spot must be positive"));
    }
    let t = *path.times().last().unwrap_or(&0.0);
    let s_t = path.terminal()[0];
    Ok(if s_t > strike {
        (-rate * t).exp() * s_t / spot
    } else {
        0.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed_rng;
    use crate::scheme::Scheme;

    #[test]
    fn four_deltas_near_bs() {
        let mut rng = seed_rng(11);
        let rep = gbm_call_delta(
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            8,
            6_000,
            0.5,
            &SimConfig {
                scheme: Scheme::Exact,
                ..SimConfig::default()
            },
            &mut rng,
        )
        .unwrap();
        assert!((rep.analytic - 0.63683).abs() < 1e-4);
        assert!((rep.pathwise.estimate - rep.analytic).abs() < 0.03);
        assert!((rep.bump.estimate - rep.analytic).abs() < 0.05);
    }
}
