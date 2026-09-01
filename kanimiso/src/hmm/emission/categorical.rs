//! Finite-support categorical emission.

use super::Emission;
use crate::context::FitCtx;
use signlred::{Issue, IssueCode, Result};

/// Discrete emission over `{0, …, n_symbols-1}`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Categorical {
    /// `p[symbol]`. Must be non-negative; `log_prob` is `−∞` on zeros.
    pub probs: Vec<f64>,
}

impl Categorical {
    /// Categorical with the given probability vector (not renormalised here).
    pub(crate) fn new(probs: Vec<f64>) -> Self {
        Self { probs }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CategoricalStats {
    counts: Vec<f64>,
}

impl Emission for Categorical {
    type Observation = usize;
    type SufficientStats = CategoricalStats;

    fn log_prob(&self, obs: &Self::Observation) -> f64 {
        let p = self.probs.get(*obs).copied().unwrap_or(0.0);
        if p > 0.0 && p.is_finite() {
            p.ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    fn accumulate(&self, obs: &Self::Observation, weight: f64, stats: &mut Self::SufficientStats) {
        if !weight.is_finite() || weight <= 0.0 {
            return;
        }
        let n = self.probs.len().max(*obs + 1);
        if stats.counts.len() < n {
            stats.counts.resize(n, 0.0);
        }
        stats.counts[*obs] += weight;
    }

    fn maximize(&mut self, stats: &Self::SufficientStats, ctx: &mut FitCtx) -> Result<()> {
        let n = self.probs.len().max(stats.counts.len());
        if n == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmissionDegenerate)
                    .message("categorical emission has an empty alphabet")
                    .build(),
            );
            return Ok(());
        }
        let mut counts = stats.counts.clone();
        counts.resize(n, 0.0);
        let mass: f64 = counts.iter().sum();
        if mass <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnreachableState)
                    .message("categorical M-step saw zero posterior mass")
                    .build(),
            );
            return Ok(());
        }
        self.probs = counts.iter().map(|c| c / mass).collect();
        Ok(())
    }
}
