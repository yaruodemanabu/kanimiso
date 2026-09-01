//! Two-sided power emission. Power is a runtime parameter.

use super::{ContinuousLaw, Emission};
use crate::context::FitCtx;
use signlred::{Issue, IssueCode, Result};

/// Symmetric two-sided power on `(μ − s, μ + s)`:
/// `f(y) = (n / (2s)) (1 − |u|)^{n−1}`, `u = (y−μ)/s`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TwoSidedPower {
    /// Location μ.
    pub loc: f64,
    /// Half-width s > 0.
    pub scale: f64,
    /// Power n > 0.
    pub power: f64,
}

impl TwoSidedPower {
    /// Two-sided power law.
    pub(crate) fn new(loc: f64, scale: f64, power: f64) -> Self {
        Self { loc, scale, power }
    }

    fn unit(&self, y: f64) -> Option<f64> {
        if !y.is_finite() || !self.loc.is_finite() || !self.scale.is_finite() || self.scale <= 0.0 {
            return None;
        }
        if !self.power.is_finite() || self.power <= 0.0 {
            return None;
        }
        let u = (y - self.loc) / self.scale;
        if !u.is_finite() || u.abs() >= 1.0 {
            return None;
        }
        Some(u)
    }
}

impl ContinuousLaw for TwoSidedPower {
    fn log_density(&self, y: f64) -> f64 {
        let Some(u) = self.unit(y) else {
            return f64::NEG_INFINITY;
        };
        let remain = 1.0 - u.abs();
        if remain <= 0.0 {
            return f64::NEG_INFINITY;
        }
        self.power.ln() - std::f64::consts::LN_2 - self.scale.ln()
            + (self.power - 1.0) * remain.ln()
    }

    fn cdf(&self, y: f64) -> f64 {
        let Some(u) = self.unit(y) else {
            if !y.is_finite() || self.scale <= 0.0 {
                return 0.0;
            }
            return if y <= self.loc - self.scale { 0.0 } else { 1.0 };
        };
        if u < 0.0 {
            0.5 * (1.0 + u).powf(self.power)
        } else {
            1.0 - 0.5 * (1.0 - u).powf(self.power)
        }
    }

    fn log_cdf(&self, y: f64) -> f64 {
        let Some(u) = self.unit(y) else {
            if y <= self.loc {
                return f64::NEG_INFINITY;
            }
            return 0.0;
        };
        if u < 0.0 {
            -std::f64::consts::LN_2 + self.power * (1.0 + u).ln()
        } else {
            let t = 0.5 * (1.0 - u).powf(self.power);
            if t < 1.0 {
                (-t).ln_1p()
            } else {
                f64::NEG_INFINITY
            }
        }
    }

    fn log_sf(&self, y: f64) -> f64 {
        let Some(u) = self.unit(y) else {
            if y >= self.loc {
                return f64::NEG_INFINITY;
            }
            return 0.0;
        };
        if u >= 0.0 {
            -std::f64::consts::LN_2 + self.power * (1.0 - u).ln()
        } else {
            let t = 0.5 * (1.0 + u).powf(self.power);
            if t < 1.0 {
                (-t).ln_1p()
            } else {
                f64::NEG_INFINITY
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TspStats {
    mass: f64,
    sum_y: f64,
    sum_abs: f64,
}

impl Emission for TwoSidedPower {
    type Observation = f64;
    type SufficientStats = TspStats;

    fn log_prob(&self, obs: &Self::Observation) -> f64 {
        self.log_density(*obs)
    }

    fn accumulate(&self, obs: &Self::Observation, weight: f64, stats: &mut Self::SufficientStats) {
        if !weight.is_finite() || weight <= 0.0 || !obs.is_finite() {
            return;
        }
        stats.mass += weight;
        stats.sum_y += weight * *obs;
        stats.sum_abs += weight * (obs - self.loc).abs();
    }

    fn maximize(&mut self, stats: &Self::SufficientStats, ctx: &mut FitCtx) -> Result<()> {
        if stats.mass <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnreachableState)
                    .message("two-sided-power M-step saw zero posterior mass")
                    .build(),
            );
            return Ok(());
        }
        self.loc = stats.sum_y / stats.mass;
        let scale = (stats.sum_abs / stats.mass) * 2.0;
        if scale > 0.0 && scale.is_finite() {
            self.scale = scale;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n3_matches_v0_1_trig_form_without_floor() {
        let law = TwoSidedPower::new(0.0, 1.0, 3.0);
        let y = 0.4_f64;
        let u = y;
        let dens = 1.5 * (1.0 - u.abs()).powi(2);
        let got = law.log_density(y).exp();
        // measured 2026-09-01: |closed − formula| = 0 on this point (tol 1e-14)
        assert!((got - dens).abs() <= 1e-14, "{got} vs {dens}");
        let cdf = if u < 0.0 {
            0.5 * (1.0 + u).powi(3)
        } else {
            1.0 - 0.5 * (1.0 - u).powi(3)
        };
        assert!((law.cdf(y) - cdf).abs() <= 1e-14);
    }

    #[test]
    fn tail_log_sf_is_closed_form() {
        let law = TwoSidedPower::new(0.0, 1.0, 3.0);
        let y = 0.999999_f64;
        let want = -std::f64::consts::LN_2 + 3.0 * (1.0 - y).ln();
        let got = law.log_sf(y);
        assert!((got - want).abs() <= 1e-12, "{got} vs {want}");
        assert!(got.is_finite());
        assert!(got < (1e-15_f64).ln());
    }
}
