//! Quality-responsible logging for machine learning and linear algebra.
//!
//! `ojizou-san` is the durable half of the quality contract that [`signlred`]
//! defines in-process. Algorithms must record:
//!
//! - every [`signlred::Issue`] they raise
//! - every numerical compromise (intended vs actual computation)
//! - every meaningless-fit diagnosis
//! - every incremental / online update, with explainability
//! - optimization traces that justify claiming convergence
//!
//! This is not a generic application logger. A silent `fit` is a bug.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod event;
mod explain;
mod journal;
mod ledger;
mod session;
mod sink;

pub use event::{Event, EventKind};
pub use explain::{IncrementalExplain, TrustLevel};
pub use journal::CompromiseJournal;
pub use ledger::Ledger;
pub use session::Session;
pub use sink::{FnSink, MemorySink, Sink, StderrSink};

use signlred::{Issue, Report};

/// Push every issue in a report into a session as quality events.
pub fn ingest_report(session: &Session, report: &Report) {
    for issue in report.issues() {
        session.record_issue(issue);
    }
}

/// Render an issue as a single-line quality log.
pub fn render_issue(issue: &Issue) -> String {
    format!("{issue}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use signlred::{Issue, IssueCode, Meaninglessness, NumericalCompromise, Report};

    #[test]
    fn session_records_compromise_and_meaningless() {
        let session = Session::new("ols", "fit");
        let mut report = Report::new("ols", "fit");
        report.push(
            Issue::builder(IssueCode::RidgeFallbackUsed)
                .message("added λ=1e-8 because X'X was indefinite at working precision")
                .compromise(NumericalCompromise::new(
                    "OLS via Cholesky(X'X)",
                    "ridge λ=1e-8 then Cholesky",
                    "smallest eigenvalue was −1e-18",
                    "the estimand is no longer OLS; it is a tiny ridge",
                ))
                .build(),
        );
        report.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .meaninglessness(Meaninglessness::vacuous("coefficients", "rank 0", "stop"))
                .build(),
        );
        ingest_report(&session, &report);
        let events = session.ledger().events();
        assert!(events
            .iter()
            .any(|e| e.kind == EventKind::NumericalCompromise));
        assert!(events
            .iter()
            .any(|e| e.kind == EventKind::MeaninglessResult));
        assert!(!session.journal().is_empty());
    }
}
