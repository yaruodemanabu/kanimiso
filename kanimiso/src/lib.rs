//! `kanimiso` — Pure Rust machine learning and statistics.
//!
//! v0.2 is being reconstituted from a small verified core (see the repo
//! `AGENTS.md`). Claims of coverage track [`coverage::verified`], not the raw
//! [`coverage::inventory`]. Until `0.2.0-alpha.1` this crate does **not**
//! claim equivalence to scikit-learn, statsmodels, sktime, tslearn, hmmlearn,
//! or river.
//!
//! Constraints honoured by this crate:
//! - linear algebra goes through [`faer`]
//! - no `unsafe`, no non-Rust native libraries
//! - every fit/predict/partial_fit talks to [`signlred`] (quality errors) and
//!   [`ojizou_san`] (quality logging)
//! - incremental algorithms emit [`ojizou_san::IncrementalExplain`]
//!
//! A silent successful fit is a bug.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod anomaly;
pub(crate) mod bandit;
pub(crate) mod bayes;
mod bridge;
pub(crate) mod classification;
pub(crate) mod cluster;
pub(crate) mod compose;
pub mod context;
pub(crate) mod covariance;
pub mod coverage;
pub mod data;
pub(crate) mod decompose;
pub(crate) mod discriminant;
pub(crate) mod ensemble;
pub(crate) mod feature;
pub(crate) mod filters;
pub(crate) mod glm;
pub(crate) mod gp;
pub(crate) mod histgb;
pub mod hmm;
pub(crate) mod iv;
pub(crate) mod kernel_pca;
pub(crate) mod linalg;
pub mod linear_model;
pub(crate) mod manifold;
pub mod metrics;
pub(crate) mod mixed;
pub(crate) mod model_selection;
pub(crate) mod multiclass;
pub(crate) mod multinomial;
pub(crate) mod multioutput;
pub(crate) mod naive_bayes;
pub(crate) mod neighbors;
pub(crate) mod neural;
pub mod online;
pub(crate) mod panel;
pub(crate) mod preprocess;
pub(crate) mod random_projection;
pub(crate) mod reducer;
pub(crate) mod rng;
pub(crate) mod robust;
pub(crate) mod semi;
pub mod special;
pub(crate) mod stats;
pub(crate) mod svm;
pub(crate) mod text;
pub(crate) mod topic;
pub mod traits;
pub(crate) mod tree;
pub(crate) mod tsa;
pub mod tslearn;
pub(crate) mod validate;
pub(crate) mod vecm;

pub use context::FitCtx;
pub use coverage::{inventory, verified, Algorithm, CoverageStatus};
pub use data::{Matrix, Vector};
pub use ojizou_san as log;
pub use signlred;
pub use traits::{Fit, FitSeries, FitUnsupervised, PartialFit, Predict, Transform};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
