//! A value together with the quality report that justifies using it.

use crate::report::Report;
use core::fmt;

/// Successful computation + the quality contract that came with it.
///
/// Dropping `report` is allowed by the type system and forbidden by the crate
/// contract. Downstream crates should persist it via `ojizou-san`.
#[derive(Debug, Clone, PartialEq)]
pub struct Qualified<T> {
    /// The numeric or structured result.
    pub value: T,
    /// Warnings, compromises, and advisories collected during the computation.
    pub report: Report,
}

impl<T> Qualified<T> {
    /// Wrap a value with an already-finished report. Prefer [`Report::finish`].
    pub fn new(value: T, report: Report) -> Self {
        Self { value, report }
    }

    /// Map the inner value, keeping the same report.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Qualified<U> {
        Qualified {
            value: f(self.value),
            report: self.report,
        }
    }

    /// Split into value and report.
    pub fn into_parts(self) -> (T, Report) {
        (self.value, self.report)
    }

    /// True when the value is accompanied by any warning-or-worse issue.
    pub fn is_compromised(&self) -> bool {
        self.report.has_warning() || self.report.has_error() || self.report.has_fatal()
    }
}

impl<T: fmt::Display> fmt::Display for Qualified<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}\n{}", self.value, self.report)
    }
}
