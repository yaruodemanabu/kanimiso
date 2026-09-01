//! Tree **ensembles** built on [`oldwood`] and [`faer`].
//!
//! Random forests, extra-trees, gradient boosting, AdaBoost, Isolation
//! Forest, and random-trees embedding live here. Single-tree CART grow
//! stays in `oldwood`. This crate does not depend on linfa, smartcore, or
//! any other packaged ML solver — only `faer` and `oldwood`.
//!
//! kanimiso wraps these types with `signlred` / `ojizou-san` quality gates.

#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

mod boosting;
mod forest;
mod isolation;

pub use boosting::{
    fit_adaboost, fit_adaboost_r2, fit_gbc, fit_gbr, AdaBoostAlgorithm, AdaBoostR2Spec,
    AdaBoostR2Stop, AdaBoostSpec, AdaBoostStop, BoostStop, FittedAdaBoost, FittedAdaBoostR2,
    FittedGbc, FittedGbr, GbcSpec, GbrSpec,
};
pub use forest::{
    grow_forest_class, grow_forest_reg, ForestClassifier, ForestRegressor, ForestSpec,
};
pub use isolation::{
    fit_embedding, fit_isolation, EmbeddingSpec, FittedEmbedding, FittedIsolation, IsolationSpec,
};
pub use oldwood::{isolation_c_factor, GrowSpec};
