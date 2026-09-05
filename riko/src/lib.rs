//! Stochastic and adversarial multi-armed-bandit policies in safe, pure Rust.
//!
//! Rewards are bounded to `[0, 1]`. A [`Choice`] binds feedback to the round
//! and sampling probability that produced it, preventing stale updates.
#![forbid(unsafe_code)]
mod choice;
mod error;
mod exp;
mod ucb;
pub use choice::Choice;
pub use error::Error;
pub use exp::ExpWeights;
pub use ucb::Ucb;
/// Result type used by fallible operations in this crate.
pub type Result<T> = core::result::Result<T, Error>;
