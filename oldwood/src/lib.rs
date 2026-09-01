//! CART and isolation-tree **core** on [`faer::Mat<f64>`].
//!
//! This crate is the numeric kernel for a single tree: impurity, split
//! search, grow, and walk. It does not depend on signlred, ojizou-san, or
//! any packaged ML crate (linfa, smartcore, …). Forests, boosting, and
//! isolation ensembles live in [`mayoi_no_mori`](https://docs.rs/mayoi-no-mori)
//! (workspace crate `mayoi-no-mori`).
//!
//! Design matrices are borrowed as `&faer::Mat<f64>` so the same `faer`
//! version (AGENTS.md D6) crosses the kanimiso boundary.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

mod grow;
mod impurity;
mod node;
mod predict;
mod rng;
mod split;

pub use grow::{grow_class, grow_iso, grow_reg, rewrite_logistic_leaves, GrowSpec};
pub use impurity::{class_index, gini, majority, mse_of, weighted_counts};
pub use node::{class_leaf, is_class_stump, isolation_c_factor, ClassNode, IsoNode, RegNode};
pub use predict::{
    iso_leaf_code, iso_path, predict_class_labels, predict_class_one, predict_class_proba,
    predict_reg, predict_reg_one,
};
pub use rng::Rng;
pub use split::{best_class_split, best_reg_split, feature_subset, split_index};
