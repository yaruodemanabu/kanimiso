//! Deterministic classification and regression trees.
//!
//! `oldwood` implements one CART split engine in safe Rust. It accepts any
//! matrix through [`MatrixView`], supports non-negative sample weights, and
//! exposes fitted nodes as an immutable arena for inspection.

#![forbid(unsafe_code)]

mod criterion;
mod error;
mod matrix;
mod model;
mod node;
mod numeric;
mod options;
mod split;
mod strategy;
mod validation;

pub use criterion::{ClassificationCriterion, RegressionCriterion};
pub use error::Error;
pub use matrix::{DenseMatrix, MatrixView};
pub use model::{DecisionTreeClassifier, DecisionTreeRegressor, FittedClassifier, FittedRegressor};
pub use node::{ArenaNode, ClassificationValue, NodeKind};
pub use options::TreeOptions;
pub use strategy::{Exhaustive, SplitContext, SplitStrategy};

/// Result type used by all fallible operations in this crate.
pub type Result<T> = core::result::Result<T, Error>;
