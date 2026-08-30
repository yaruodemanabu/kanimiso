//! Per-call glue: `signlred` report + `ojizou-san` session + policy.

use ojizou_san::Session;
use signlred::{Issue, Policy, Qualified, Report, Result};

/// Working context for one estimator call.
pub struct FitCtx {
    /// Quality logger.
    pub session: Session,
    /// Accumulating quality report.
    pub report: Report,
    /// Abort / warn thresholds.
    pub policy: Policy,
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
        }
    }

    /// Open with an existing session (nested step).
    pub fn with_session(session: Session) -> Self {
        let report = Report::new(session.algorithm(), session.operation());
        Self {
            session,
            report,
            policy: Policy::default(),
        }
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
        } = self;
        match report.finish_with_policy(policy, value) {
            Ok(q) => {
                session.finish_ok(&q);
                Ok(q)
            }
            Err(e) => {
                session.finish_err(&e);
                Err(e)
            }
        }
    }
}
