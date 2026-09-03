//! Per-call glue: `signlred` report + `ojizou-san` session + policy.

use ojizou_san::Session;
use signlred::{Failure, Issue, Policy, Qualified, Report, Result};

/// Working context for one estimator call.
pub struct FitCtx {
    /// Quality logger.
    pub session: Session,
    /// Accumulating quality report.
    pub report: Report,
    /// Abort / warn thresholds.
    pub policy: Policy,
    record_completion: bool,
}

impl FitCtx {
    /// Open a context. The session emits `fit_started` when `operation == "fit"`.
    pub fn new(algorithm: impl Into<String>, operation: impl Into<String>) -> Self {
        let algorithm = algorithm.into();
        let operation = operation.into();
        Self {
            session: Session::new(&algorithm, &operation),
            report: Report::new(&algorithm, &operation),
            policy: Policy::default(),
            record_completion: true,
        }
    }

    /// Open with an existing session (nested step).
    pub fn with_session(session: Session) -> Self {
        let report = Report::new(session.algorithm(), session.operation());
        Self {
            session,
            report,
            policy: Policy::default(),
            record_completion: true,
        }
    }

    /// Keep nested iteration breadcrumbs but let the enclosing computation
    /// ingest the merged report and record the single terminal event.
    pub(crate) fn suppress_completion_recording(&mut self) {
        self.record_completion = false;
    }

    /// Push an issue under the active policy.
    pub fn push(&mut self, issue: Issue) {
        self.report.push_with_policy(self.policy.clone(), issue);
    }

    /// Finish: log and return `Qualified` or `Failure`.
    pub fn finish<T>(self, value: T) -> Result<Qualified<T>> {
        let FitCtx {
            session,
            report,
            policy,
            record_completion,
        } = self;
        match report.finish_with_policy(policy, value) {
            Ok(q) => {
                if record_completion {
                    session.finish_ok(&q);
                }
                Ok(q)
            }
            Err(e) => {
                if record_completion {
                    session.finish_err(&e);
                }
                Err(e)
            }
        }
    }

    /// Finish a context that cannot produce a value, even when a caller policy
    /// deliberately relaxes the recorded issue's default abort threshold.
    pub(crate) fn finish_failure(self) -> Failure {
        let primary = self
            .report
            .clone()
            .into_failure(&self.policy)
            .map(|failure| failure.primary)
            .or_else(|| self.report.primary().cloned())
            .expect("forced failure context must contain an issue");
        let failure = Failure {
            primary,
            report: self.report,
        };
        if self.record_completion {
            self.session.finish_err(&failure);
        }
        failure
    }

    /// Merge a failed nested computation and finish the outer session.
    pub(crate) fn merge_failure(mut self, failure: Failure) -> Failure {
        let nested_primary = failure.primary;
        let outer_primary = self
            .report
            .issues()
            .iter()
            .filter(|issue| self.policy.must_abort(issue))
            .min_by_key(|issue| issue.severity)
            .cloned();
        self.report.merge(failure.report);
        let primary = outer_primary
            .filter(|issue| issue.severity < nested_primary.severity)
            .unwrap_or(nested_primary);
        let combined = Failure {
            primary,
            report: self.report,
        };
        if self.record_completion {
            self.session.finish_err(&combined);
        }
        combined
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use signlred::{IssueCode, Severity};

    #[test]
    fn nested_failure_remains_primary_when_outer_issue_does_not_abort() {
        let mut outer = FitCtx::new("context", "outer");
        outer.policy.abort_at = Severity::Fatal;
        outer.push(
            Issue::builder(IssueCode::NonFiniteOutput)
                .message("outer error is non-aborting under this policy")
                .build(),
        );

        let mut nested_policy = Policy::default();
        nested_policy.abort_at = Severity::Warning;
        let mut nested_report = Report::new("context", "nested");
        nested_report.push_with_policy(
            nested_policy.clone(),
            Issue::builder(IssueCode::MaxIterReached)
                .message("nested warning is aborting under its policy")
                .build(),
        );
        let nested = nested_report
            .into_failure(&nested_policy)
            .expect("nested warning must abort");
        let combined = outer.merge_failure(nested);
        assert_eq!(combined.primary.code, IssueCode::MaxIterReached);
        assert!(combined.report.contains(IssueCode::NonFiniteOutput));
        assert!(combined.report.contains(IssueCode::MaxIterReached));
    }
}
