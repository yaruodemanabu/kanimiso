//! Regression models for a carefully annotated first statistical analysis.
#![forbid(unsafe_code)]
// Preserve signlred's report-carrying error contract at every public boundary.
#![allow(clippy::result_large_err)]
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]
pub use ojizou_san::Session;
pub use signlred::{Policy, Qualified, Result};
pub use tsutsumi::{context, data, linalg, special, traits, validate, Matrix, Vector};
pub mod annotation;
pub mod linear;
pub use annotation::{AnalysisOptions, Annotation, Topic};
pub mod regression;
pub use regression::{
    Coefficient, Family, FittedRegression, GeneralizedLinearModel, LinearModel,
    RegressionDiagnostics,
};
pub mod additive;
pub use additive::{
    AdditiveModel, FittedAdditive, GeneralizedAdditiveModel, LinearAdditiveModel, SplineTerm,
};
pub mod explain;
pub use explain::{linear_shap, LinearExplanation};
pub mod mixed;
pub use mixed::{FittedMixed, GeneralizedMixedModel, Likelihood, LinearMixedModel, MixedModel};
