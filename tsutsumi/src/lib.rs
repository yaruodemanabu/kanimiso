//! Shared Pure Rust numerical kernels with independent oracle tests.
//!
//! The default `linalg` feature provides faer matrices, decomposition/solve
//! diagnostics, special functions, and explicit-trace Nelder–Mead. Disabling
//! default features leaves the dependency-free matrix-view contract.
#![forbid(unsafe_code)]
// The shared public Failure intentionally owns a complete, inspectable report.
#![allow(clippy::result_large_err)]
#![deny(missing_docs)]
#![cfg_attr(feature = "linalg", doc = include_str!("../README.md"))]

mod matrix_view;
pub use matrix_view::MatrixView;
#[cfg(feature = "linalg")]
pub mod context;
#[cfg(feature = "linalg")]
pub mod data;
#[cfg(feature = "linalg")]
pub mod linalg;
#[cfg(feature = "linalg")]
pub mod optimize;
#[cfg(feature = "linalg")]
pub mod special;
#[cfg(feature = "linalg")]
pub mod traits;
#[cfg(feature = "linalg")]
pub mod validate;
#[cfg(feature = "linalg")]
pub use context::FitCtx;
#[cfg(feature = "linalg")]
pub use data::{Matrix, Vector};
#[cfg(feature = "linalg")]
pub mod quadrature;
