//! Emission distributions for [`crate::hmm::HiddenMarkovModel`].

mod categorical;
mod cosine_power;
mod gaussian;
mod poisson;
mod transformed;
mod two_sided_power;

pub use categorical::{Categorical, CategoricalStats};
pub use cosine_power::{CosinePower, CosineStats};
pub use gaussian::{Gaussian, GaussianStats};
pub use poisson::{Poisson, PoissonStats};
pub use transformed::{Transform, Transformed};
pub use two_sided_power::{TspStats, TwoSidedPower};

use crate::context::FitCtx;
use signlred::Result;

/// Continuous law with a CDF and log-complement (no `(1−F).ln()` clamp).
pub(crate) trait ContinuousLaw {
    fn log_density(&self, y: f64) -> f64;
    fn cdf(&self, y: f64) -> f64;
    fn log_cdf(&self, y: f64) -> f64;
    fn log_sf(&self, y: f64) -> f64;
}

/// One hidden-state emission. The observation type is the data, not a
/// parameter value (AGENTS.md R3).
pub trait Emission: Clone {
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
