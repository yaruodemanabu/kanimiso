//! Fatal / error-severity quality failure.

use crate::issue::Issue;
use crate::report::Report;
use core::fmt;

/// A computation that must not return a value.
///
/// The primary issue is the most severe aborting finding. The full [`Report`]
/// is retained so callers (and `ojizou-san`) can see every preceding warning.
#[derive(Debug, Clone, PartialEq)]
pub struct Failure {
    /// Most severe aborting issue.
    pub primary: Issue,
    /// Complete ledger, including non-aborting issues collected first.
    pub report: Report,
}

impl Failure {
    /// Convenience: a single-issue failure.
    pub fn from_issue(
        algorithm: impl Into<String>,
        operation: impl Into<String>,
        issue: Issue,
    ) -> Self {
        let mut report = Report::new(algorithm, operation);
        report.push(issue.clone());
        Self {
            primary: issue,
            report,
        }
    }

    /// Borrow the primary issue.
    pub fn primary(&self) -> &Issue {
        &self.primary
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "signlred failure in {}::{}: {}",
            self.report.algorithm, self.report.operation, self.primary
        )
    }
}

impl std::error::Error for Failure {}
