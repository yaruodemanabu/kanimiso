//! Emission distributions for [`crate::hmm::HiddenMarkovModel`].

mod categorical;
mod gaussian;
mod poisson;

pub(crate) use categorical::Categorical;
pub(crate) use gaussian::Gaussian;
pub(crate) use poisson::Poisson;

use crate::context::FitCtx;
use signlred::Result;

/// One hidden-state emission. The observation type is the data, not a
/// parameter value (AGENTS.md R3).
pub(crate) trait Emission: Clone {
    /// Observation stored in a sequence (`Vec<f64>`, count, or symbol).
    type Observation;
    /// Running sums for one Baum–Welch M-step.
    type SufficientStats: Default;
    /// `log p(obs | self)`. Outside support this is `−∞`. `NaN` is forbidden.
    fn log_prob(&self, obs: &Self::Observation) -> f64;
    /// Add a posterior-weighted observation to `stats`.
    fn accumulate(&self, obs: &Self::Observation, weight: f64, stats: &mut Self::SufficientStats);
    /// Replace parameters with the MLE of `stats`.
    fn maximize(&mut self, stats: &Self::SufficientStats, ctx: &mut FitCtx) -> Result<()>;
}
