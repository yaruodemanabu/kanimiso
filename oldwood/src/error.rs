use core::fmt;

/// Input, configuration, or numerical failure.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Row-major storage length does not equal `rows * columns`.
    InvalidMatrixStorage {
        /// Requested row count.
        rows: usize,
        /// Requested column count.
        columns: usize,
        /// Supplied value count.
        values: usize,
    },
    /// Training requires at least one row.
    EmptyTrainingData,
    /// Training requires at least one feature column.
    EmptyFeatures,
    /// Target length differs from the matrix row count.
    TargetLength {
        /// Matrix row count.
        expected: usize,
        /// Target length.
        actual: usize,
    },
    /// Sample-weight length differs from the matrix row count.
    WeightLength {
        /// Matrix row count.
        expected: usize,
        /// Weight length.
        actual: usize,
    },
    /// A feature value is NaN or infinite.
    NonFiniteFeature {
        /// Zero-based row.
        row: usize,
        /// Zero-based column.
        column: usize,
    },
    /// A regression target is NaN or infinite.
    NonFiniteTarget {
        /// Zero-based row.
        row: usize,
    },
    /// A sample weight is negative, NaN, or infinite.
    InvalidWeight {
        /// Zero-based row.
        row: usize,
    },
    /// No row has positive sample weight.
    NoPositiveWeight,
    /// A tree option violates its documented domain.
    InvalidOption {
        /// Name of the invalid option.
        name: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Prediction feature count differs from the training feature count.
    FeatureCount {
        /// Training feature count.
        expected: usize,
        /// Prediction feature count.
        actual: usize,
    },
    /// A split strategy returned a feature outside the matrix.
    InvalidStrategyFeature {
        /// Returned feature index.
        feature: usize,
        /// Matrix column count.
        columns: usize,
    },
    /// A split strategy returned a non-finite or out-of-range threshold.
    InvalidStrategyThreshold {
        /// Feature for which the threshold was returned.
        feature: usize,
    },
    /// A checked floating-point calculation was not representable.
    NumericalOverflow {
        /// Calculation that failed.
        operation: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMatrixStorage {
                rows,
                columns,
                values,
            } => write!(
                formatter,
                "matrix shape {rows}x{columns} requires a different number of values than {values}"
            ),
            Self::EmptyTrainingData => formatter.write_str("training data has no rows"),
            Self::EmptyFeatures => formatter.write_str("training data has no feature columns"),
            Self::TargetLength { expected, actual } => write!(
                formatter,
                "target length {actual} does not match matrix row count {expected}"
            ),
            Self::WeightLength { expected, actual } => write!(
                formatter,
                "sample-weight length {actual} does not match matrix row count {expected}"
            ),
            Self::NonFiniteFeature { row, column } => {
                write!(formatter, "feature at row {row}, column {column} is not finite")
            }
            Self::NonFiniteTarget { row } => {
                write!(formatter, "regression target at row {row} is not finite")
            }
            Self::InvalidWeight { row } => write!(
                formatter,
                "sample weight at row {row} must be finite and non-negative"
            ),
            Self::NoPositiveWeight => {
                formatter.write_str("at least one sample weight must be positive")
            }
            Self::InvalidOption { name, requirement } => {
                write!(formatter, "{name} must be {requirement}")
            }
            Self::FeatureCount { expected, actual } => write!(
                formatter,
                "prediction matrix has {actual} columns; fitted tree requires {expected}"
            ),
            Self::InvalidStrategyFeature { feature, columns } => write!(
                formatter,
                "split strategy returned feature {feature}, but matrix has {columns} columns"
            ),
            Self::InvalidStrategyThreshold { feature } => write!(
                formatter,
                "split strategy returned a non-finite or out-of-range threshold for feature {feature}"
            ),
            Self::NumericalOverflow { operation } => {
                write!(formatter, "{operation} produced an unrepresentable result")
            }
        }
    }
}

impl std::error::Error for Error {}
