use core::fmt;

/// Invalid configuration, interaction, or numerical result.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A policy was configured without arms.
    NoArms,
    /// An option is outside its documented domain.
    InvalidOption {
        name: &'static str,
        requirement: &'static str,
    },
    /// A unit-interval sampling variate is invalid.
    InvalidSample,
    /// An arm index is outside the configured action set.
    InvalidArm { arm: usize, arms: usize },
    /// A reward is NaN, infinite, or outside `[0, 1]`.
    InvalidReward,
    /// Feedback does not match the policy state that produced the choice.
    StaleChoice,
    /// A checked calculation produced an unrepresentable value.
    NumericalOverflow { operation: &'static str },
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoArms => f.write_str("a bandit policy requires at least one arm"),
            Self::InvalidOption { name, requirement } => write!(f, "{name} must be {requirement}"),
            Self::InvalidSample => f.write_str("sample must be finite and in [0, 1)"),
            Self::InvalidArm { arm, arms } => write!(f, "arm {arm} is outside 0..{arms}"),
            Self::InvalidReward => f.write_str("reward must be finite and in [0, 1]"),
            Self::StaleChoice => {
                f.write_str("choice was not produced by the policy's current state")
            }
            Self::NumericalOverflow { operation } => {
                write!(f, "{operation} produced an unrepresentable result")
            }
        }
    }
}
impl std::error::Error for Error {}
