//! Poisson emission. `log_prob` uses `ln Γ(k+1)`, never `ln(density)`.

use super::Emission;
use crate::context::FitCtx;
use crate::special::ln_gamma;
use signlred::{Issue, IssueCode, Result};

/// Poisson count emission with rate `λ > 0`.
#[derive(Clone, Debug, PartialEq)]
pub struct Poisson {
    /// Mean count.
    pub rate: f64,
}

impl Poisson {
    /// Poisson(`rate`). `rate ≤ 0` makes every `log_prob` `−∞`.
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }
}

/// Posterior-weighted counts for a Poisson M-step.
#[derive(Clone, Debug, Default)]
pub struct PoissonStats {
    mass: f64,
    weighted_count: f64,
}

impl Emission for Poisson {
    type Observation = f64;
    type SufficientStats = PoissonStats;

    fn log_prob(&self, obs: &Self::Observation) -> f64 {
        let k = *obs;
        if !k.is_finite() || k < 0.0 || (k - k.round()).abs() > 1e-12 {
            return f64::NEG_INFINITY;
        }
        if !self.rate.is_finite() || self.rate <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let kk = k.round();
        kk * self.rate.ln() - self.rate - ln_gamma(kk + 1.0)
    }

    fn accumulate(&self, obs: &Self::Observation, weight: f64, stats: &mut Self::SufficientStats) {
        if !weight.is_finite() || weight <= 0.0 || !obs.is_finite() || *obs < 0.0 {
            return;
        }
        stats.mass += weight;
        stats.weighted_count += weight * obs.max(0.0);
    }

    fn maximize(&mut self, stats: &Self::SufficientStats, ctx: &mut FitCtx) -> Result<()> {
        if stats.mass <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnreachableState)
                    .message("Poisson M-step saw zero posterior mass")
                    .build(),
            );
            return Ok(());
        }
        let rate = stats.weighted_count / stats.mass;
        if !rate.is_finite() || rate <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::EmissionDegenerate)
                    .message(format!("Poisson M-step produced non-positive rate {rate}"))
                    .build(),
            );
            return Ok(());
        }
        self.rate = rate;
        Ok(())
    }
}
