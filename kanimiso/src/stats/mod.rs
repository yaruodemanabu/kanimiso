//! Verified statistical-process estimators.
//!
//! The v0.2 surface is intentionally limited to algorithms backed by an
//! independent oracle and explicit numerical-quality reporting.

mod process;

pub use process::{process_mle, ProcessMleFit};
