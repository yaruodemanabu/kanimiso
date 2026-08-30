//! `kanimiso` — Pure Rust machine learning and statistics.
//!
//! Scope: the computational surface of **scikit-learn**, **statsmodels**,
//! **sktime**, **tslearn**, **hmmlearn**, and **river**.
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
pub mod bandit;
pub mod bayes;
pub mod classification;
pub mod cluster;
pub mod compose;
pub mod context;
pub mod covariance;
pub mod coverage;
pub mod data;
pub mod decompose;
pub mod discriminant;
pub mod ensemble;
pub mod feature;
pub mod filters;
pub mod glm;
pub mod gp;
pub mod histgb;
pub mod hmm;
pub mod iv;
pub mod kernel_pca;
pub mod linalg;
pub mod linear_model;
pub mod manifold;
pub mod metrics;
pub mod mixed;
pub mod model_selection;
pub mod multinomial;
pub mod multioutput;
pub mod naive_bayes;
pub mod neighbors;
pub mod neural;
pub mod online;
pub mod preprocess;
pub mod reducer;
pub mod rng;
pub mod robust;
pub mod semi;
pub mod special;
pub mod stats;
pub mod svm;
pub mod topic;
pub mod traits;
pub mod tree;
pub mod tsa;
pub mod tslearn;
pub mod validate;
pub mod vecm;

pub use context::FitCtx;
pub use coverage::{inventory, Algorithm};
pub use data::{Matrix, Vector};
pub use ojizou_san as log;
pub use signlred;
pub use traits::{Fit, FitSeries, FitUnsupervised, PartialFit, Predict, Transform};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
