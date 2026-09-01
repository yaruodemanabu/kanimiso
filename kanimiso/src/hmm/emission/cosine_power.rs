//! Cosine-power emission (AGENTS.md §4.2). Power is a runtime parameter.

use super::{ContinuousLaw, Emission};
use crate::context::FitCtx;
use crate::special::{betainc_reg, ln_gamma};
use signlred::{Issue, IssueCode, Result};

const LN_2: f64 = std::f64::consts::LN_2;

/// Cosine-power density on `(μ − s, μ + s)` with power `n ≥ 0`.
///
/// `log_prob` is the closed form `n log cos θ − log Z_n − log s`. Support
/// outside `|θ| < π/2` is `−∞` (no density-then-`ln`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CosinePower {
    /// Location μ.
    pub loc: f64,
    /// Half-width s > 0.
    pub scale: f64,
    /// Power n (0 is uniform on the support).
    pub power: f64,
}

impl CosinePower {
    /// Cosine-power law. `scale` must be positive; `power` must be finite and
    /// `≥ 0`.
    pub(crate) fn new(loc: f64, scale: f64, power: f64) -> Self {
        Self { loc, scale, power }
    }

    fn theta(&self, y: f64) -> Option<f64> {
        if !y.is_finite() || !self.loc.is_finite() || !self.scale.is_finite() || self.scale <= 0.0 {
            return None;
        }
        if !self.power.is_finite() || self.power < 0.0 {
            return None;
        }
        let u = (y - self.loc) / self.scale;
        let theta = std::f64::consts::FRAC_PI_2 * u;
        if !theta.is_finite() || theta.abs() >= std::f64::consts::FRAC_PI_2 {
            return None;
        }
        Some(theta)
    }
}

/// `log Z_n = log 2 − ½ log π + log Γ((n+1)/2) − log Γ(n/2 + 1)`.
pub(crate) fn log_normalizer(n: f64) -> f64 {
    LN_2 - 0.5 * std::f64::consts::PI.ln() + ln_gamma((n + 1.0) / 2.0) - ln_gamma(n / 2.0 + 1.0)
}

impl ContinuousLaw for CosinePower {
    fn log_density(&self, y: f64) -> f64 {
        let Some(theta) = self.theta(y) else {
            return f64::NEG_INFINITY;
        };
        let c = theta.cos();
        if c <= 0.0 {
            return f64::NEG_INFINITY;
        }
        self.power * c.ln() - log_normalizer(self.power) - self.scale.ln()
    }

    fn cdf(&self, y: f64) -> f64 {
        let Some(theta) = self.theta(y) else {
            if !y.is_finite() || self.scale <= 0.0 {
                return 0.0;
            }
            return if y <= self.loc - self.scale { 0.0 } else { 1.0 };
        };
        let s2 = theta.sin().powi(2);
        let inc = betainc_reg(0.5, (self.power + 1.0) / 2.0, s2);
        0.5 + 0.5 * theta.signum() * inc
    }

    fn log_cdf(&self, y: f64) -> f64 {
        let Some(theta) = self.theta(y) else {
            if y <= self.loc {
                return f64::NEG_INFINITY;
            }
            return 0.0;
        };
        if theta >= 0.0 {
            let f = self.cdf(y);
            if f > 0.0 && f < 1.0 {
                f.ln()
            } else if f >= 1.0 {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        } else {
            let s2 = theta.sin().powi(2);
            let inc = betainc_reg(0.5, (self.power + 1.0) / 2.0, s2);
            if inc > 0.0 {
                (-LN_2) + inc.ln()
            } else {
                f64::NEG_INFINITY
            }
        }
    }

    fn log_sf(&self, y: f64) -> f64 {
        let Some(theta) = self.theta(y) else {
            if y >= self.loc {
                return f64::NEG_INFINITY;
            }
            return 0.0;
        };
        if theta >= 0.0 {
            // 1−F = ½ I_{cos²θ}((n+1)/2, 1/2)  — complement, no subtraction.
            let c2 = theta.cos().powi(2);
            let inc = betainc_reg((self.power + 1.0) / 2.0, 0.5, c2);
            if inc > 0.0 {
                (-LN_2) + inc.ln()
            } else {
                f64::NEG_INFINITY
            }
        } else {
            let f = self.cdf(y);
            if f < 1.0 {
                (1.0 - f).ln()
            } else {
                f64::NEG_INFINITY
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CosineStats {
    mass: f64,
    sum_y: f64,
    sum_abs: f64,
}

impl Emission for CosinePower {
    type Observation = f64;
    type SufficientStats = CosineStats;

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
                    .message("cosine-power M-step saw zero posterior mass")
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
    fn normalizer_matches_closed_form_n0_n2() {
        // n=0 → Z=2; n=2 → Z=1 (AGENTS.md §4.2)
        let z0 = log_normalizer(0.0).exp();
        let z2 = log_normalizer(2.0).exp();
        assert!((z0 - 2.0).abs() < 1e-12, "Z0={z0}");
        assert!((z2 - 1.0).abs() < 1e-12, "Z2={z2}");
    }

    #[test]
    fn n3_cdf_matches_trig_closed_form() {
        let law = CosinePower::new(0.0, 1.0, 3.0);
        for u in [-0.8, -0.3, 0.0, 0.3, 0.8] {
            let y = u;
            let s = (std::f64::consts::FRAC_PI_2 * u).sin();
            let closed = 0.5 + 0.75 * s - 0.25 * s * s * s;
            let got = law.cdf(y);
            // measured 2026-09-01: |betainc − trig| on these u; tol = 4×
            assert!(
                (got - closed).abs() <= 4e-14,
                "u={u}: got {got} closed {closed}"
            );
        }
    }

    #[test]
    fn n3_log_prob_matches_direct_formula() {
        let law = CosinePower::new(0.0, 1.0, 3.0);
        let y = 0.25_f64;
        let theta = std::f64::consts::FRAC_PI_2 * y;
        let want = 3.0 * theta.cos().ln() - log_normalizer(3.0);
        let got = law.log_density(y);
        assert!((got - want).abs() <= 1e-14, "{got} vs {want}");
        assert_eq!(law.log_density(2.0), f64::NEG_INFINITY);
    }

    #[test]
    fn right_tail_log_sf_is_finite_and_not_clamped() {
        // u=0.999: complement I_{cos²} must stay finite and << ln(1e-15)
        let law = CosinePower::new(0.0, 1.0, 3.0);
        let y = 0.9999_f64;
        let ls = law.log_sf(y);
        assert!(ls.is_finite(), "log_sf={ls}");
        let clamp = (1e-15_f64).ln();
        assert!(
            ls < clamp,
            "complement log_sf {ls} must be smaller than the forbidden R7 clamp {clamp}"
        );
    }

    #[test]
    fn scipy_mpmath_golden() {
        let raw = include_str!("../../../../golden/cosine_power.json");
        let payload: serde_json::Value = serde_json::from_str(raw).unwrap();
        let mut worst_cdf = 0.0_f64;
        let mut worst_lp = 0.0_f64;
        let mut worst_quad = 0.0_f64;
        for case in payload["cases"].as_array().unwrap() {
            let n = case["n"].as_f64().unwrap();
            let y = case["y"].as_f64().unwrap();
            let loc = case["loc"].as_f64().unwrap();
            let scale = case["scale"].as_f64().unwrap();
            let law = CosinePower::new(loc, scale, n);
            let lp = law.log_density(y);
            let cdf = law.cdf(y);
            let lsf = law.log_sf(y);
            let exp_lp = case["log_prob"].as_f64().unwrap();
            let exp_cdf = case["cdf"].as_f64().unwrap();
            let exp_lsf = case["log_sf"].as_f64().unwrap();
            let quad = case["mpmath_cdf"].as_f64().unwrap();
            worst_lp = worst_lp.max((lp - exp_lp).abs());
            worst_cdf = worst_cdf.max((cdf - exp_cdf).abs());
            worst_quad = worst_quad.max((cdf - quad).abs());
            assert!(
                (lp - exp_lp).abs() <= 4e-13,
                "n={n} y={y} log_prob {lp} vs {exp_lp}"
            );
            assert!(
                (cdf - exp_cdf).abs() <= 4e-13,
                "n={n} y={y} cdf {cdf} vs {exp_cdf}"
            );
            if exp_lsf.is_finite() {
                assert!(
                    (lsf - exp_lsf).abs() <= 4e-12,
                    "n={n} y={y} log_sf {lsf} vs {exp_lsf}"
                );
            }
            assert!(
                (cdf - quad).abs() <= 4e-12,
                "n={n} y={y} cdf {cdf} mpmath {quad}"
            );
        }
        // measured 2026-09-01 vs scipy 1.18.1 / mpmath 1.4.1
        assert!(worst_lp <= 4e-13 && worst_cdf <= 4e-13 && worst_quad <= 4e-12);
    }
}
