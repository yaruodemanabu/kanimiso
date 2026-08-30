//! Severity of a quality issue.

use core::fmt;

/// How seriously a quality issue undermines the result.
///
/// Ordered from most to least severe. [`Policy`](crate::Policy) decides which
/// levels abort the computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// The computation must stop. Returning a number would be a lie.
    Fatal,
    /// The result is not trustworthy enough to return as success.
    Error,
    /// The result exists but is compromised or statistically fragile.
    Warning,
    /// The result is usable; the caller should still read the note.
    Advisory,
    /// Audit trail only.
    Info,
}

impl Severity {
    /// Human label used in logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fatal => "fatal",
            Self::Error => "error",
            Self::Warning => "warning",
            Self::Advisory => "advisory",
            Self::Info => "info",
        }
    }

    /// Whether this severity is at least as serious as `floor`.
    pub fn is_at_least(self, floor: Self) -> bool {
        self <= floor
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_is_most_severe_first() {
        assert!(Severity::Fatal < Severity::Error);
        assert!(Severity::Error < Severity::Warning);
        assert!(Severity::Warning < Severity::Advisory);
        assert!(Severity::Advisory < Severity::Info);
    }
}
