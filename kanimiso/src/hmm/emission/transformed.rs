//! Probability-integral transforms of a continuous base law.

use super::{ContinuousLaw, Emission};
use crate::context::FitCtx;
use crate::special::ln_gamma;
use signlred::{Issue, IssueCode, Result};

/// How a [`Transformed`] observation is read from a base law.
#[derive(Clone, Debug, PartialEq)]
pub enum Transform {
    /// Odds map `u = y / (1 − y)` on `y ∈ (0, 1)` with Jacobian `1/(1−y)²`.
    Unit,
    /// Beta-generated family `f ∝ f₀ F₀^{a−1} (1−F₀)^{b−1}`.
    Beta {
        /// First Beta shape.
        a: f64,
        /// Second Beta shape.
        b: f64,
    },
    /// Kumaraswamy-generated family.
    Kumaraswamy {
        /// First Kumaraswamy shape.
        a: f64,
        /// Second Kumaraswamy shape.
        b: f64,
    },
    /// Exponentiated / Lehmann alternative `f = p f₀ F₀^{p−1}`.
    Exponentiated {
        /// Exponent p > 0.
        power: f64,
    },
    /// Integer discretisation `P(K=k) = F(k+1) − F(k)` for `k ≥ 1`.
    Discrete,
}

/// Base law plus one of the five generated-family transforms.
#[derive(Clone, Debug, PartialEq)]
pub struct Transformed<E> {
    /// Underlying continuous emission.
    pub base: E,
    /// Transform applied to `log_prob`.
    pub transform: Transform,
}

impl<E> Transformed<E> {
    /// Wrap `base` with `transform`.
    pub fn new(base: E, transform: Transform) -> Self {
        Self { base, transform }
    }
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// `ln(1 − exp(x))` for `x < 0`.
fn log1mexp(x: f64) -> f64 {
    if !x.is_finite() || x >= 0.0 {
        return f64::NAN;
    }
    if x > -std::f64::consts::LN_2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

impl<E: ContinuousLaw + Clone> Emission for Transformed<E> {
    type Observation = f64;
    type SufficientStats = ();

    fn log_prob(&self, obs: &Self::Observation) -> f64 {
        let y = *obs;
        match self.transform {
            Transform::Unit => {
                if !y.is_finite() || y <= 0.0 || y >= 1.0 {
                    return f64::NEG_INFINITY;
                }
                let u = y / (1.0 - y);
                if !u.is_finite() || u < 0.0 {
                    return f64::NEG_INFINITY;
                }
                let lp = self.base.log_density(u);
                if !lp.is_finite() {
                    return lp;
                }
                lp - 2.0 * (1.0 - y).ln()
            }
            Transform::Beta { a, b } => {
                if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                let lp = self.base.log_density(y);
                let lf = self.base.log_cdf(y);
                let ls = self.base.log_sf(y);
                if !lp.is_finite() || !lf.is_finite() || !ls.is_finite() {
                    return f64::NEG_INFINITY;
                }
                lp + (a - 1.0) * lf + (b - 1.0) * ls - ln_beta(a, b)
            }
            Transform::Kumaraswamy { a, b } => {
                if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                let lp = self.base.log_density(y);
                let lf = self.base.log_cdf(y);
                if !lp.is_finite() || !lf.is_finite() {
                    return f64::NEG_INFINITY;
                }
                // 1 − F^a = 1 − exp(a log F)
                let log_fa = a * lf;
                let log_one_m = log1mexp(log_fa);
                if !log_one_m.is_finite() {
                    return f64::NEG_INFINITY;
                }
                a.ln() + b.ln() + lp + (a - 1.0) * lf + (b - 1.0) * log_one_m
            }
            Transform::Exponentiated { power } => {
                if !power.is_finite() || power <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                let lp = self.base.log_density(y);
                let lf = self.base.log_cdf(y);
                if !lp.is_finite() || !lf.is_finite() {
                    return f64::NEG_INFINITY;
                }
                power.ln() + lp + (power - 1.0) * lf
            }
            Transform::Discrete => {
                if !y.is_finite() || y < 1.0 {
                    return f64::NEG_INFINITY;
                }
                let k = y.round().max(1.0);
                let log_hi = self.base.log_sf(k);
                let log_lo = self.base.log_sf(k + 1.0);
                if !log_hi.is_finite() {
                    return f64::NEG_INFINITY;
                }
                if !log_lo.is_finite() {
                    return log_hi;
                }
                let delta = log_lo - log_hi;
                let adj = log1mexp(delta);
                if adj.is_finite() {
                    log_hi + adj
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }

    fn accumulate(
        &self,
        _obs: &Self::Observation,
        _weight: f64,
        _stats: &mut Self::SufficientStats,
    ) {
    }

    fn maximize(&mut self, _stats: &Self::SufficientStats, ctx: &mut FitCtx) -> Result<()> {
        ctx.push(
            Issue::builder(IssueCode::UnreachableState)
                .message("Transformed M-step leaves the base parameters unchanged")
                .build(),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmm::emission::CosinePower;

    #[test]
    fn beta_generated_uses_log_sf_not_clamp() {
        let base = CosinePower::new(0.0, 1.0, 3.0);
        let t = Transformed::new(base.clone(), Transform::Beta { a: 2.0, b: 3.0 });
        let y = 0.9_f64;
        let got = t.log_prob(&y);
        let want =
            base.log_density(y) + (2.0 - 1.0) * base.log_cdf(y) + (3.0 - 1.0) * base.log_sf(y)
                - ln_beta(2.0, 3.0);
        assert!(got.is_finite());
        assert!((got - want).abs() <= 1e-12, "{got} vs {want}");
    }

    #[test]
    fn unit_odds_jacobian() {
        let base = CosinePower::new(0.0, 2.0, 3.0);
        let t = Transformed::new(base.clone(), Transform::Unit);
        let y = 0.25_f64;
        let u = y / (1.0 - y);
        let want = base.log_density(u) - 2.0 * (1.0 - y).ln();
        let got = t.log_prob(&y);
        assert!((got - want).abs() <= 1e-12, "{got} vs {want}");
        assert_eq!(t.log_prob(&0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn remaining_transforms_are_finite_or_neg_inf() {
        let base = CosinePower::new(0.0, 1.0, 3.0);
        let y = 0.2_f64;
        let kuma = Transformed::new(base.clone(), Transform::Kumaraswamy { a: 1.5, b: 2.0 });
        let expn = Transformed::new(base.clone(), Transform::Exponentiated { power: 2.0 });
        let disc = Transformed::new(base, Transform::Discrete);
        assert!(kuma.log_prob(&y).is_finite());
        assert!(expn.log_prob(&y).is_finite());
        assert!(disc.log_prob(&1.0).is_finite() || disc.log_prob(&1.0).is_infinite());
    }
}
