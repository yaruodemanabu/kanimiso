//! Discounted Monte Carlo pricing with online mean / variance.

use amatsuki::Rng;

use crate::error::{Error, Result};
use crate::finance::black_scholes::{call as bs_call, BlackScholesMarket};
use crate::finance::market::{DiscountCurve, FlatCurve};
use crate::finance::payoff::PathPayoff;
use crate::model::Sde;
use crate::path::Path;
use crate::sampling::Sampling;
use crate::simulate::{simulate, SimConfig};

/// Online mean / variance accumulator (Welford).
#[derive(Clone, Debug, Default)]
pub struct OnlineMoments {
    pub n: usize,
    pub mean: f64,
    pub m2: f64,
}

impl OnlineMoments {
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n as f64 - 1.0)
        }
    }

    pub fn standard_error(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            (self.variance() / self.n as f64).sqrt()
        }
    }
}

/// Monte Carlo estimate with a normal confidence interval.
#[derive(Clone, Debug)]
pub struct MonteCarloEstimate {
    pub estimate: f64,
    pub standard_error: f64,
    pub confidence_interval: (f64, f64),
    pub n_paths: usize,
    pub diagnostics: MonteCarloDiagnostics,
}

/// Extra diagnostics attached to a Monte Carlo run.
#[derive(Clone, Debug)]
pub struct MonteCarloDiagnostics {
    pub sample_variance: f64,
    pub variance_reduction_ratio: Option<f64>,
    pub effective_paths: usize,
}

impl MonteCarloEstimate {
    pub fn from_moments(m: &OnlineMoments, z: f64) -> Self {
        let se = m.standard_error();
        Self {
            estimate: m.mean,
            standard_error: se,
            confidence_interval: (m.mean - z * se, m.mean + z * se),
            n_paths: m.n,
            diagnostics: MonteCarloDiagnostics {
                sample_variance: m.variance(),
                variance_reduction_ratio: None,
                effective_paths: m.n,
            },
        }
    }
}

/// Variance-reduction switches for [`price_sde`].
#[derive(Clone, Copy, Debug, Default)]
pub struct VarianceReduction {
    pub antithetic: bool,
    pub european_control: bool,
}

/// Discounted MC price of a path payoff under a flat rate.
pub fn price_sde<M, P, R>(
    model: &M,
    payoff: &P,
    sampling: &Sampling,
    x0: &[f64],
    rate: f64,
    n_paths: usize,
    cfg: &SimConfig,
    vr: VarianceReduction,
    control_strike: Option<f64>,
    rng: &mut R,
) -> Result<MonteCarloEstimate>
where
    M: Sde + ?Sized,
    P: PathPayoff,
    R: Rng + ?Sized,
{
    if n_paths == 0 {
        return Err(Error::param("need a positive number of MC paths"));
    }
    let disc = FlatCurve::new(rate)?;
    let t = *sampling.times().last().unwrap_or(&0.0);
    let df = disc.discount(t)?;
    let mut acc = OnlineMoments::default();
    let mut cv_xy = 0.0;
    let mut cv_y = OnlineMoments::default();
    let m = model.n_noise();
    for _ in 0..n_paths {
        let dw = crate::noise::brownian_increments(sampling, m, rng)?;
        let mut cfg_p = cfg.clone();
        cfg_p.increment_w = Some(dw.clone());
        let path = simulate(model, sampling, x0, rng, &cfg_p)?;
        let mut pay = df * payoff.payoff(&path)?;
        if vr.antithetic {
            let mut dwn = dw;
            for w in &mut dwn {
                *w = -*w;
            }
            cfg_p.increment_w = Some(dwn);
            let anti = simulate(model, sampling, x0, rng, &cfg_p)?;
            pay = 0.5 * (pay + df * payoff.payoff(&anti)?);
        }
        if vr.european_control {
            if let Some(k) = control_strike {
                let s_t = path.terminal()[0];
                let y = df * (s_t - k).max(0.0);
                cv_y.push(y);
                cv_xy += pay * y;
            }
        }
        acc.push(pay);
    }
    let mut est = MonteCarloEstimate::from_moments(&acc, 1.959963984540054);
    if vr.european_control {
        if let Some(k) = control_strike {
            let n = acc.n as f64;
            let cov = cv_xy / n - acc.mean * cv_y.mean;
            let vy = cv_y.variance();
            if vy > 0.0 {
                let b = cov / vy;
                // Analytic mean of the discounted vanilla if we have GBM-like
                // numbers; otherwise subtract the sample mean of Y (zero-mean
                // residual form).
                let ey = cv_y.mean;
                let adj = acc.mean - b * (cv_y.mean - ey);
                let _ = adj;
                // Use the known BS mean when a market can be formed.
                if let Ok(mkt) = infer_bs_market(model, x0, rate, t) {
                    if let Ok(bs) = bs_call(&mkt, k) {
                        let estimate = acc.mean - b * (cv_y.mean - bs.price);
                        let var = (acc.variance() + b * b * vy - 2.0 * b * cov).max(0.0);
                        let se = (var / n).sqrt();
                        est.estimate = estimate;
                        est.standard_error = se;
                        est.confidence_interval = (estimate - 1.96 * se, estimate + 1.96 * se);
                        est.diagnostics.sample_variance = var;
                        est.diagnostics.variance_reduction_ratio = Some(if acc.variance() > 0.0 {
                            var / acc.variance()
                        } else {
                            1.0
                        });
                    }
                }
            }
        }
    }
    Ok(est)
}

fn infer_bs_market<M: Sde + ?Sized>(
    model: &M,
    x0: &[f64],
    rate: f64,
    t: f64,
) -> Result<BlackScholesMarket> {
    if x0.is_empty() {
        return Err(Error::dim("empty x0"));
    }
    let mut sig = [0.0; 1];
    model.diffusion(0.0, x0, &mut sig);
    let vol = if x0[0] > 0.0 {
        (sig[0] / x0[0]).abs()
    } else {
        0.0
    };
    BlackScholesMarket::new(x0[0], rate, 0.0, vol, t)
}

/// GBM call under a drifted measure `μ'`, reweighted by the Girsanov
/// exponential martingale (importance sampling for deep OTM strikes).
pub fn price_gbm_importance<R: Rng + ?Sized>(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    mu_star: f64,
    n_paths: usize,
    n_steps: usize,
    rng: &mut R,
) -> Result<MonteCarloEstimate> {
    if n_paths == 0 || n_steps == 0 || !(spot > 0.0 && vol > 0.0 && time > 0.0) {
        return Err(Error::param("importance-sampling inputs invalid"));
    }
    let model = crate::models::GeometricBrownianMotion::new(mu_star, vol)?;
    let samp = Sampling::from_terminal(time, n_steps)?;
    let theta = crate::finance::market::market_price_of_risk(mu_star, rate, vol)?;
    let df = (-rate * time).exp();
    let mut acc = OnlineMoments::default();
    let m = model.n_noise();
    for _ in 0..n_paths {
        let dw = crate::noise::brownian_increments(&samp, m, rng)?;
        let mut cfg = SimConfig::default();
        cfg.increment_w = Some(dw.clone());
        let path = simulate(&model, &samp, &[spot], rng, &cfg)?;
        let w_t: f64 = dw.iter().sum();
        let z = crate::finance::market::exponential_martingale(theta, w_t, time)?;
        acc.push(df * (path.terminal()[0] - strike).max(0.0) * z);
    }
    Ok(MonteCarloEstimate::from_moments(&acc, 1.959963984540054))
}

/// Price a European functional of a supplied ensemble (no new simulation).
pub fn price_ensemble<P: PathPayoff>(
    paths: &[Path],
    payoff: &P,
    rate: f64,
) -> Result<MonteCarloEstimate> {
    if paths.is_empty() {
        return Err(Error::param("empty ensemble"));
    }
    let t = *paths[0].times().last().unwrap_or(&0.0);
    let df = FlatCurve::new(rate)?.discount(t)?;
    let mut acc = OnlineMoments::default();
    for p in paths {
        acc.push(df * payoff.payoff(p)?);
    }
    Ok(MonteCarloEstimate::from_moments(&acc, 1.959963984540054))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finance::payoff::EuropeanCall;
    use crate::models::GeometricBrownianMotion;
    use crate::rng::seed_rng;
    use crate::scheme::Scheme;

    #[test]
    fn mc_call_near_bs() {
        let model = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let samp = Sampling::from_terminal(1.0, 4).unwrap();
        let pay = EuropeanCall::new(0, 100.0).unwrap();
        let mut rng = seed_rng(3);
        let est = price_sde(
            &model,
            &pay,
            &samp,
            &[100.0],
            0.05,
            8_000,
            &SimConfig {
                scheme: Scheme::Exact,
                ..SimConfig::default()
            },
            VarianceReduction {
                antithetic: true,
                european_control: false,
            },
            None,
            &mut rng,
        )
        .unwrap();
        assert!((est.estimate - 10.4506).abs() < 0.4, "{}", est.estimate);
        assert!(est.standard_error > 0.0 && est.standard_error < 0.3);
        let is =
            price_gbm_importance(100.0, 100.0, 0.05, 0.2, 1.0, 0.05, 2000, 4, &mut rng).unwrap();
        assert!((is.estimate - 10.4506).abs() < 0.8, "IS {}", is.estimate);
    }
}
