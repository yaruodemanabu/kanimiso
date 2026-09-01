//! Crate-wide error type.

use thiserror::Error;

/// Recoverable failure from model construction, simulation, or inference.
#[derive(Debug, Error)]
pub enum Error {
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    #[error("invalid parameter: {0}")]
    Parameter(String),
    #[error("invalid sampling: {0}")]
    Sampling(String),
    #[error("simulation failed: {0}")]
    Simulation(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("numerical failure: {0}")]
    Numeric(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn dim(msg: impl Into<String>) -> Self {
        Self::Dimension(msg.into())
    }
    pub(crate) fn param(msg: impl Into<String>) -> Self {
        Self::Parameter(msg.into())
    }
    pub(crate) fn sampling(msg: impl Into<String>) -> Self {
        Self::Sampling(msg.into())
    }
    pub(crate) fn sim(msg: impl Into<String>) -> Self {
        Self::Simulation(msg.into())
    }
    pub(crate) fn infer(msg: impl Into<String>) -> Self {
        Self::Inference(msg.into())
    }
    pub(crate) fn numeric(msg: impl Into<String>) -> Self {
        Self::Numeric(msg.into())
    }
    pub(crate) fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }
}
