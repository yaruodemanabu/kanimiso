//! Per-operation logging session used by every `kanimiso` algorithm.

use crate::event::{Event, EventKind};
use crate::explain::IncrementalExplain;
use crate::journal::CompromiseJournal;
use crate::ledger::Ledger;
use crate::sink::Sink;
use signlred::{Failure, Issue, Qualified, Report};
use std::sync::{Arc, Mutex};

/// A named quality session: algorithm + operation + ledger + journal.
///
/// Cheap to clone (shared ledger). Algorithms should clone the session into
/// nested steps rather than creating a silent child logger.
#[derive(Clone)]
pub struct Session {
    algorithm: String,
    operation: String,
    ledger: Ledger,
    journal: Arc<Mutex<CompromiseJournal>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("algorithm", &self.algorithm)
            .field("operation", &self.operation)
            .field("ledger_len", &self.ledger.len())
            .finish()
    }
}

impl Session {
    /// Open a session. Emits `FitStarted` when `operation` is `"fit"`.
    pub fn new(algorithm: impl Into<String>, operation: impl Into<String>) -> Self {
        let algorithm = algorithm.into();
        let operation = operation.into();
        let session = Self {
            algorithm: algorithm.clone(),
            operation: operation.clone(),
            ledger: Ledger::new(),
            journal: Arc::new(Mutex::new(CompromiseJournal::new())),
        };
        if operation == "fit" {
            session.emit(Event::new(EventKind::FitStarted, &algorithm, &operation));
        }
        session
    }

    /// Child session sharing the ledger/journal but with a new operation name.
    pub fn child(&self, operation: impl Into<String>) -> Self {
        Self {
            algorithm: self.algorithm.clone(),
            operation: operation.into(),
            ledger: self.ledger.clone(),
            journal: Arc::clone(&self.journal),
        }
    }

    /// Algorithm name.
    pub fn algorithm(&self) -> &str {
        &self.algorithm
    }

    /// Operation name.
    pub fn operation(&self) -> &str {
        &self.operation
    }

    /// Shared ledger.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Snapshot of the compromise journal.
    pub fn journal(&self) -> CompromiseJournal {
        self.journal
            .lock()
            .expect("compromise journal poisoned")
            .clone()
    }

    /// Fan-out sink.
    pub fn add_sink(&self, sink: Arc<dyn Sink>) {
        self.ledger.add_sink(sink);
    }

    /// Emit a pre-built event (algorithm/operation overwritten to this session).
    pub fn emit(&self, mut event: Event) -> Event {
        event.algorithm = self.algorithm.clone();
        event.operation = self.operation.clone();
        let stored = self.ledger.append(event);
        self.journal
            .lock()
            .expect("compromise journal poisoned")
            .ingest_event(&stored);
        stored
    }

    /// Record a `signlred` issue (picks kind from severity / compromise / meaning).
    pub fn record_issue(&self, issue: &Issue) -> Event {
        let kind = EventKind::from_severity(issue.severity);
        self.emit(Event::new(kind, &self.algorithm, &self.operation).with_issue(issue.clone()))
    }

    /// Ingest a whole report.
    pub fn ingest(&self, report: &Report) {
        for issue in report.issues() {
            self.record_issue(issue);
        }
    }

    /// Record incremental explainability (mandatory for `partial_fit`).
    pub fn record_incremental(&self, expl: IncrementalExplain) -> Event {
        self.emit(
            Event::new(
                EventKind::IncrementalExplanation,
                &self.algorithm,
                &self.operation,
            )
            .with_incremental(expl),
        )
    }

    /// Mark a successful fit. Empty ledgers are themselves a quality bug; this
    /// always writes `FitFinished` so the contract is visible.
    pub fn finish_ok<T>(&self, qualified: &Qualified<T>) -> Event {
        self.ingest(&qualified.report);
        self.emit(
            Event::new(EventKind::FitFinished, &self.algorithm, &self.operation)
                .message("qualified success")
                .field("n_issues", qualified.report.issues().len().to_string())
                .field("compromised", qualified.is_compromised().to_string()),
        )
    }

    /// Mark a failed fit.
    pub fn finish_err(&self, failure: &Failure) -> Event {
        self.ingest(&failure.report);
        self.emit(
            Event::new(EventKind::FitFailed, &self.algorithm, &self.operation)
                .with_issue(failure.primary.clone())
                .message(failure.to_string()),
        )
    }

    /// Optimization breadcrumb.
    pub fn step(&self, iteration: u64, loss: f64, grad_norm: Option<f64>) -> Event {
        let mut ev = Event::new(
            EventKind::OptimizationStep,
            &self.algorithm,
            &self.operation,
        )
        .metric("iteration", iteration as f64)
        .metric("loss", loss);
        if let Some(g) = grad_norm {
            ev = ev.metric("grad_norm", g);
        }
        self.emit(ev)
    }

    /// Convergence claim — must include the criterion that was satisfied.
    pub fn converged(&self, criterion: impl Into<String>, iteration: u64) -> Event {
        self.emit(
            Event::new(EventKind::Convergence, &self.algorithm, &self.operation)
                .message(criterion.into())
                .metric("iteration", iteration as f64),
        )
    }

    /// Divergence claim.
    pub fn diverged(&self, why: impl Into<String>) -> Event {
        self.emit(Event::new(EventKind::Divergence, &self.algorithm, &self.operation).message(why))
    }
}
