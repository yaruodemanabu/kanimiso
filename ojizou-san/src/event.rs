//! Structured quality events.

use crate::explain::IncrementalExplain;
use signlred::{Issue, Severity};
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Kind of quality event. These are ML/LA events, not HTTP access logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// An estimator began a batch fit.
    FitStarted,
    /// An estimator finished a batch fit (success or qualified success).
    FitFinished,
    /// An estimator aborted.
    FitFailed,
    /// A streaming / incremental update.
    PartialFit,
    /// A prediction / forecast / decode step.
    Predict,
    /// A feature transform.
    Transform,
    /// A `signlred` warning-severity issue.
    QualityWarning,
    /// A `signlred` error-severity issue (usually followed by `FitFailed`).
    QualityError,
    /// A `signlred` fatal issue.
    QualityFatal,
    /// A numerical compromise was taken.
    NumericalCompromise,
    /// A completed computation was diagnosed as meaningless.
    MeaninglessResult,
    /// Narrative explainability for an incremental update.
    IncrementalExplanation,
    /// One iteration of an optimizer.
    OptimizationStep,
    /// Optimizer claimed convergence.
    Convergence,
    /// Optimizer diverged or exploded.
    Divergence,
    /// Advisory / info breadcrumb.
    Audit,
}

impl EventKind {
    /// Human label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FitStarted => "fit_started",
            Self::FitFinished => "fit_finished",
            Self::FitFailed => "fit_failed",
            Self::PartialFit => "partial_fit",
            Self::Predict => "predict",
            Self::Transform => "transform",
            Self::QualityWarning => "quality_warning",
            Self::QualityError => "quality_error",
            Self::QualityFatal => "quality_fatal",
            Self::NumericalCompromise => "numerical_compromise",
            Self::MeaninglessResult => "meaningless_result",
            Self::IncrementalExplanation => "incremental_explanation",
            Self::OptimizationStep => "optimization_step",
            Self::Convergence => "convergence",
            Self::Divergence => "divergence",
            Self::Audit => "audit",
        }
    }

    /// Map a `signlred` severity onto a quality event kind (the issue itself).
    pub fn from_severity(severity: Severity) -> Self {
        match severity {
            Severity::Fatal => Self::QualityFatal,
            Severity::Error => Self::QualityError,
            Severity::Warning => Self::QualityWarning,
            Severity::Advisory | Severity::Info => Self::Audit,
        }
    }
}

/// One append-only quality log record.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// Nanoseconds since UNIX epoch (best effort).
    pub timestamp_ns: u128,
    /// Event kind.
    pub kind: EventKind,
    /// Algorithm name.
    pub algorithm: String,
    /// Operation name.
    pub operation: String,
    /// Sequence number inside the ledger.
    pub seq: u64,
    /// Optional copied issue.
    pub issue: Option<Issue>,
    /// Optional incremental narrative.
    pub incremental: Option<IncrementalExplain>,
    /// Structured fields (`"n" -> "100"`).
    pub fields: BTreeMap<String, String>,
    /// Free-text body.
    pub message: String,
}

impl Event {
    /// Build an event with a fresh timestamp.
    pub fn new(
        kind: EventKind,
        algorithm: impl Into<String>,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            timestamp_ns: now_ns(),
            kind,
            algorithm: algorithm.into(),
            operation: operation.into(),
            seq: 0,
            issue: None,
            incremental: None,
            fields: BTreeMap::new(),
            message: String::new(),
        }
    }

    /// Attach a field.
    pub fn field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    /// Attach a numeric field (rendered with debug).
    pub fn metric(self, key: impl Into<String>, value: f64) -> Self {
        self.field(key, format!("{value:.8e}"))
    }

    /// Set the body.
    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// Attach an issue and upgrade the kind when the issue is a compromise
    /// or a meaninglessness diagnosis.
    pub fn with_issue(mut self, issue: Issue) -> Self {
        if issue.is_meaningless() {
            self.kind = EventKind::MeaninglessResult;
        } else if issue.is_compromise() {
            self.kind = EventKind::NumericalCompromise;
        } else if matches!(
            self.kind,
            EventKind::QualityWarning
                | EventKind::QualityError
                | EventKind::QualityFatal
                | EventKind::Audit
        ) {
            // keep
        } else {
            self.kind = EventKind::from_severity(issue.severity);
        }
        if self.message.is_empty() {
            self.message = issue.to_string();
        }
        self.issue = Some(issue);
        self
    }

    /// Attach incremental explainability (also sets kind).
    pub fn with_incremental(mut self, expl: IncrementalExplain) -> Self {
        self.kind = EventKind::IncrementalExplanation;
        if self.message.is_empty() {
            self.message = expl.narrative.clone();
        }
        self.incremental = Some(expl);
        self
    }

    /// Single-line render used by [`crate::StderrSink`].
    pub fn render(&self) -> String {
        let mut line = format!(
            "ojizou-san seq={} kind={} algo={} op={}",
            self.seq,
            self.kind.as_str(),
            self.algorithm,
            self.operation
        );
        if !self.message.is_empty() {
            line.push(' ');
            line.push_str(&self.message);
        }
        if !self.fields.is_empty() {
            line.push_str(" fields=");
            for (i, (k, v)) in self.fields.iter().enumerate() {
                if i > 0 {
                    line.push(',');
                }
                line.push_str(k);
                line.push('=');
                line.push_str(v);
            }
        }
        line
    }
}

fn now_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}
