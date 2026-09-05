use core::fmt;

/// Invalid configuration, input, or numerical result.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A collection that must contain at least one element is empty.
    Empty { name: &'static str },
    /// An input vector has the wrong length.
    Length {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A floating-point input is NaN or infinite.
    NonFinite { name: &'static str, index: usize },
    /// An option is outside its documented domain.
    InvalidOption {
        name: &'static str,
        requirement: &'static str,
    },
    /// A loss lies outside the algorithm's declared range.
    LossOutOfRange { index: usize },
    /// A checked calculation produced an unrepresentable result.
    NumericalOverflow { operation: &'static str },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { name } => write!(f, "{name} must not be empty"),
            Self::Length {
                name,
                expected,
                actual,
            } => write!(f, "{name} has length {actual}; expected {expected}"),
            Self::NonFinite { name, index } => write!(f, "{name} at index {index} is not finite"),
            Self::InvalidOption { name, requirement } => write!(f, "{name} must be {requirement}"),
            Self::LossOutOfRange { index } => write!(f, "loss at index {index} is outside [0, 1]"),
            Self::NumericalOverflow { operation } => {
                write!(f, "{operation} produced an unrepresentable result")
            }
        }
    }
}
impl std::error::Error for Error {}
