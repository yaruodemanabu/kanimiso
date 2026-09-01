//! Error types shared by optimal-transport modules.

use std::fmt;

/// Errors returned by `wormhole` calculations.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// An input vector or matrix is empty.
    EmptyInput {
        /// Human-readable input name.
        name: &'static str,
    },
    /// Matrix or vector dimensions do not match.
    ShapeMismatch {
        /// Description of the required shape relationship.
        context: &'static str,
        /// Observed left-hand shape.
        left: (usize, usize),
        /// Observed right-hand shape.
        right: (usize, usize),
    },
    /// A distribution weight is negative or non-finite.
    InvalidWeight {
        /// Index of the invalid weight.
        index: usize,
        /// Invalid value.
        value: f64,
    },
    /// Two balanced distributions have different total masses.
    MassMismatch {
        /// Source mass.
        source: f64,
        /// Target mass.
        target: f64,
    },
    /// A cost-matrix entry is not finite.
    InvalidCost {
        /// Cost row.
        row: usize,
        /// Cost column.
        column: usize,
        /// Invalid value.
        value: f64,
    },
    /// A named option is outside its mathematical domain.
    InvalidParameter {
        /// Option name.
        name: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// The requested constraints admit no feasible plan.
    Infeasible {
        /// Short explanation of the conflicting constraints.
        context: &'static str,
    },
    /// An iterative algorithm did not meet its stopping criterion.
    DidNotConverge {
        /// Algorithm name.
        algorithm: &'static str,
        /// Iterations completed.
        iterations: usize,
        /// Last measured residual.
        residual: f64,
    },
    /// A `faer` factorization or solve failed.
    LinearAlgebra {
        /// Operation that failed.
        operation: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { name } => write!(f, "{name} must be non-empty"),
            Self::ShapeMismatch {
                context,
                left,
                right,
            } => write!(
                f,
                "shape mismatch for {context}: left={left:?}, right={right:?}"
            ),
            Self::InvalidWeight { index, value } => {
                write!(f, "weight {index} is invalid: {value}")
            }
            Self::MassMismatch { source, target } => {
                write!(
                    f,
                    "balanced masses differ: source={source}, target={target}"
                )
            }
            Self::InvalidCost { row, column, value } => {
                write!(f, "cost at ({row}, {column}) is invalid: {value}")
            }
            Self::InvalidParameter { name, requirement } => {
                write!(f, "{name} must be {requirement}")
            }
            Self::Infeasible { context } => write!(f, "infeasible transport: {context}"),
            Self::DidNotConverge {
                algorithm,
                iterations,
                residual,
            } => write!(
                f,
                "{algorithm} did not converge after {iterations} iterations \
                 (residual={residual})"
            ),
            Self::LinearAlgebra { operation } => {
                write!(f, "faer linear algebra failed during {operation}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Result alias used throughout `wormhole`.
pub type Result<T> = std::result::Result<T, Error>;
