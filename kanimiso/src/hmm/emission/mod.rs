//! Emission distributions and sufficient-statistic contracts for generic HMMs.
//!
//! Implementations validate matrix observations before inference, evaluate
//! probabilities in the log domain, and expose unfloored maximum-likelihood
//! updates through [`Emission`].

mod categorical;
mod gaussian;
mod poisson;

use crate::context::FitCtx;
use crate::data::Matrix;
use signlred::Result;

/// Categorical emissions and their weighted-count accumulator.
pub use categorical::{CategoricalEmission, CategoricalStats};
/// Diagonal Gaussian emissions and their weighted-Welford accumulator.
pub use gaussian::{GaussianEmission, GaussianStats};
/// Poisson emissions and their weighted-count accumulator.
pub use poisson::{PoissonEmission, PoissonStats};

/// Distribution contract used by generic HMM inference and Baum–Welch updates.
///
/// Matrix conversion is fallible so invalid observations never get rounded,
/// clamped, or silently replaced before likelihood evaluation.
pub trait Emission: Clone {
    /// Validated observation representation consumed by this distribution.
    type Observation: Clone;
    /// Distribution-specific accumulator used by the emission M-step.
    type SufficientStats: Default;

    /// Validate and convert all matrix rows into emission observations.
    fn observations(x: &Matrix) -> Result<Vec<Self::Observation>>;

    /// Evaluate an observation directly in the log domain without flooring.
    fn log_prob(&self, obs: &Self::Observation) -> f64;

    /// Add one posterior-weighted observation to `stats`.
    fn accumulate(&self, obs: &Self::Observation, weight: f64, stats: &mut Self::SufficientStats);

    /// Replace parameters with their MLE when `stats` identify the emission.
    ///
    /// Zero posterior occupancy leaves parameters unchanged and records an
    /// `UnreachableState` warning in `ctx`.
    fn maximize(&mut self, stats: &Self::SufficientStats, ctx: &mut FitCtx) -> Result<()>;
}
