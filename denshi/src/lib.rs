//! Online prediction and regret-minimization kernels in safe, pure Rust.
//!
//! The crate separates decisions from feedback: inspect a learner's decision,
//! evaluate it, then provide the full-information loss or gradient.
#![forbid(unsafe_code)]
mod error;
mod gradient;
mod hedge;

pub use error::Error;
pub use gradient::OnlineGradientDescent;
pub use hedge::Hedge;
/// Result type used by fallible operations in this crate.
pub type Result<T> = core::result::Result<T, Error>;
