//! Forest and boosting estimators sharing [`oldwood`]'s ordinary CART kernel.
//!
//! `mayoi-no-mori` owns ensemble policy: resampling, random feature and
//! threshold proposals, additive losses, histogram binning, and ordered
//! categorical statistics. Ordinary CART gain and traversal remain in
//! `oldwood`; histogram Newton gain and isolation partitions are distinct
//! objectives with their own compact arenas, not duplicate CART evaluators.
//!
//! The [`LightGbmRegressor`] and [`LightGbmClassifier`] implement the
//! histogram/Newton subset documented by this crate; they are not bindings to
//! the `LightGBM` project. Likewise, [`CatBoostRegressor`] and
//! [`CatBoostClassifier`] provide leakage-resistant ordered target statistics
//! followed by this crate's boosting core, not `CatBoost` model-file
//! compatibility. See the crate README for the precise support matrix.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod adaboost;
mod catboost;
mod data;
mod error;
mod forest;
mod gradient;
mod histogram;
mod isolation;
mod numeric;
mod options;
mod random;
mod strategy;

pub use adaboost::{
    AdaBoostClassifier, AdaBoostR2Options, AdaBoostRegressor, FittedAdaBoostClassifier,
    FittedAdaBoostRegressor, SammeOptions,
};
pub use catboost::{
    CatBoostClassifier, CatBoostRegressor, FittedCatBoostClassifier, FittedCatBoostRegressor,
};
pub use error::{Error, Result};
pub use forest::{
    ExtraTreesClassifier, ExtraTreesRegressor, FittedForestClassifier, FittedForestRegressor,
    RandomForestClassifier, RandomForestRegressor,
};
pub use gradient::{
    FittedGradientBoostingClassifier, FittedGradientBoostingRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor,
};
pub use histogram::{
    FittedLightGbmClassifier, FittedLightGbmRegressor, LightGbmClassifier, LightGbmRegressor,
};
pub use isolation::{FittedIsolationForest, IsolationForest, IsolationForestOptions};
pub use oldwood::{
    ClassificationCriterion, DenseMatrix, MatrixView, RegressionCriterion, TreeOptions,
};
pub use options::{
    BoostingOptions, CatBoostOptions, FeatureSampling, ForestOptions, LightGbmOptions,
};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
