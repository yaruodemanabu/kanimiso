//! Hidden Markov models with one generic chain implementation.
//!
//! [`HiddenMarkovModel`] owns Baum–Welch orchestration and delegates the two
//! chain recursions to one log-space forward–backward kernel and one Viterbi
//! kernel. Distribution-specific mathematics lives behind [`Emission`]; the
//! built-in implementations are [`GaussianEmission`], [`CategoricalEmission`],
//! and [`PoissonEmission`]. Chain probabilities are explicit, `left_right` is
//! a runtime option, and fitting never silently floors or normalizes them.
//!
//! Impossible observation sequences are reported with
//! [`signlred::IssueCode::ScaleFactorZero`]. Extremely small but finite
//! likelihoods remain in the log domain and may record
//! [`signlred::IssueCode::ForwardUnderflow`] without losing probability mass.

mod emission;
mod forward_backward;
mod model;
mod viterbi;

#[cfg(test)]
mod oracle_tests;

pub use emission::{
    CategoricalEmission, CategoricalStats, Emission, GaussianEmission, GaussianStats,
    PoissonEmission, PoissonStats,
};
pub use model::HiddenMarkovModel;
