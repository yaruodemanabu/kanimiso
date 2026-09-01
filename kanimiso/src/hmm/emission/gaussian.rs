//! Diagonal Gaussian emission. `log_prob` is evaluated in the log domain.

use super::Emission;
use crate::context::FitCtx;
use signlred::{Issue, IssueCode, NumericalCompromise, Result};

const LN_2PI: f64 = 1.8378770664093453;

/// Diagonal-covariance Gaussian. Mean and variance are runtime vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct Gaussian {
    /// Coordinate-wise mean.
    pub mean: Vec<f64>,
    /// Coordinate-wise variance (must be `> 0` for a finite density).
    pub var: Vec<f64>,
}

impl Gaussian {
    /// Diagonal Gaussian with the given mean and variance.
    pub fn new(mean: Vec<f64>, var: Vec<f64>) -> Self {
        Self { mean, var }
    }

    /// Univariate `N(mean, var)`.
    pub fn univariate(mean: f64, var: f64) -> Self {
        Self::new(vec![mean], vec![var])
    }
}

/// Posterior-weighted sums for a diagonal-Gaussian M-step.
#[derive(Clone, Debug, Default)]
pub struct GaussianStats {
    mass: f64,
    sum_x: Vec<f64>,
    sum_x2: Vec<f64>,
}

impl Emission for Gaussian {
    type Observation = Vec<f64>;
    type SufficientStats = GaussianStats;

    fn log_prob(&self, obs: &Self::Observation) -> f64 {
        let d = self.mean.len().min(self.var.len()).min(obs.len());
        if d == 0 || self.mean.len() != obs.len() || self.var.len() != obs.len() {
            return f64::NEG_INFINITY;
        }
        let mut s = 0.0_f64;
        for j in 0..d {
            let v = self.var[j];
            if !v.is_finite() || v <= 0.0 {
                return f64::NEG_INFINITY;
            }
            let z = obs[j] - self.mean[j];
            if !z.is_finite() {
                return f64::NEG_INFINITY;
            }
            s += LN_2PI + v.ln() + z * z / v;
        }
        -0.5 * s
    }

    fn accumulate(&self, obs: &Self::Observation, weight: f64, stats: &mut Self::SufficientStats) {
        if !weight.is_finite() || weight <= 0.0 {
            return;
        }
        let d = obs.len();
        if stats.sum_x.len() != d {
            stats.sum_x = vec![0.0; d];
            stats.sum_x2 = vec![0.0; d];
        }
        stats.mass += weight;
        for j in 0..d {
            let x = obs[j];
            if x.is_finite() {
                stats.sum_x[j] += weight * x;
                stats.sum_x2[j] += weight * x * x;
            }
        }
    }

    fn maximize(&mut self, stats: &Self::SufficientStats, ctx: &mut FitCtx) -> Result<()> {
        if stats.mass <= 0.0 || stats.sum_x.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::UnreachableState)
                    .message("Gaussian M-step saw zero posterior mass")
                    .build(),
            );
            return Ok(());
        }
        let d = stats.sum_x.len();
        let floor = ctx.policy.emission_var_floor;
        let mut mean = vec![0.0; d];
        let mut var = vec![0.0; d];
        for j in 0..d {
            mean[j] = stats.sum_x[j] / stats.mass;
            let second = stats.sum_x2[j] / stats.mass;
            let raw = second - mean[j] * mean[j];
            if raw < floor {
                ctx.push(
                    Issue::builder(IssueCode::NumericalUnderflow)
                        .message(format!("Gaussian variance {raw} floored to {floor}"))
                        .compromise(NumericalCompromise::new(
                            "unrestricted Gaussian variance MLE",
                            format!("var.max({floor})"),
                            "M-step second moment collapsed",
                            "this state's scale is a floor, not an identified variance",
                        ))
                        .metric("raw_variance", raw)
                        .metric("emission_var_floor", floor)
                        .build(),
                );
                var[j] = floor;
            } else {
                var[j] = raw;
            }
        }
        self.mean = mean;
        self.var = var;
        Ok(())
    }
}
