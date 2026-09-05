//! Error values shared by ensemble estimators.

use core::fmt;

/// Result type returned by fallible `mayoi-no-mori` operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Invalid input, invalid configuration, or an underlying CART failure.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The underlying `oldwood` CART kernel rejected an operation.
    Tree(oldwood::Error),
    /// A matrix has no training rows.
    EmptyTrainingData,
    /// A matrix has no feature columns.
    EmptyFeatures,
    /// A target or weight vector has the wrong length.
    Length {
        /// Input whose length is wrong.
        name: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// A floating-point input is NaN or infinite.
    NonFinite {
        /// Kind of value that failed validation.
        name: &'static str,
        /// Zero-based position in its flattened or vector representation.
        index: usize,
    },
    /// A weight is negative.
    NegativeWeight {
        /// Zero-based sample index.
        index: usize,
    },
    /// Every sample has zero weight.
    NoPositiveWeight,
    /// An estimator option is outside its documented domain.
    InvalidOption {
        /// Option name.
        name: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Prediction data has a different feature count than training data.
    FeatureCount {
        /// Training feature count.
        expected: usize,
        /// Prediction feature count.
        actual: usize,
    },
    /// A categorical feature index is repeated or outside the matrix.
    InvalidCategoricalFeature {
        /// Offending feature index.
        feature: usize,
    },
    /// This estimator requires exactly two target classes.
    BinaryClasses {
        /// Number of distinct classes supplied.
        actual: usize,
    },
    /// A checked ensemble calculation produced an unrepresentable value.
    NumericalOverflow {
        /// Calculation that failed.
        operation: &'static str,
    },
}

impl From<oldwood::Error> for Error {
    fn from(value: oldwood::Error) -> Self {
        Self::Tree(value)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tree(error) => write!(formatter, "CART kernel: {error}"),
            Self::EmptyTrainingData => formatter.write_str("training data has no rows"),
            Self::EmptyFeatures => formatter.write_str("training data has no feature columns"),
            Self::Length {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} length {actual} does not match matrix row count {expected}"
            ),
            Self::NonFinite { name, index } => {
                write!(formatter, "{name} at position {index} is not finite")
            }
            Self::NegativeWeight { index } => {
                write!(formatter, "sample weight at index {index} is negative")
            }
            Self::NoPositiveWeight => {
                formatter.write_str("at least one sample weight must be positive")
            }
            Self::InvalidOption { name, requirement } => {
                write!(formatter, "{name} must be {requirement}")
            }
            Self::FeatureCount { expected, actual } => write!(
                formatter,
                "prediction matrix has {actual} columns; fitted model requires {expected}"
            ),
            Self::InvalidCategoricalFeature { feature } => {
                write!(formatter, "categorical feature index {feature} is invalid")
            }
            Self::BinaryClasses { actual } => {
                write!(
                    formatter,
                    "binary classifier requires 2 classes; found {actual}"
                )
            }
            Self::NumericalOverflow { operation } => {
                write!(formatter, "{operation} produced an unrepresentable result")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tree(error) => Some(error),
            _ => None,
        }
    }
}
