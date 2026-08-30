//! A single quality issue.

use crate::codes::IssueCode;
use crate::compromise::NumericalCompromise;
use crate::domain::Domain;
use crate::incremental::IncrementalQuality;
use crate::location::Location;
use crate::meaningless::Meaninglessness;
use crate::severity::Severity;
use core::fmt;

/// One quality finding attached to a computation.
#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    /// Stable machine-readable code.
    pub code: IssueCode,
    /// Scientific domain.
    pub domain: Domain,
    /// Severity after policy application (builders start from the code default).
    pub severity: Severity,
    /// Short title (defaults to the code slug).
    pub title: String,
    /// Human-readable diagnosis.
    pub message: String,
    /// Optional numerical substitution record.
    pub numerical_compromise: Option<NumericalCompromise>,
    /// Optional meaninglessness diagnosis.
    pub meaninglessness: Option<Meaninglessness>,
    /// Optional incremental-update evidence.
    pub incremental: Option<IncrementalQuality>,
    /// Named scalar metrics that justify the finding.
    pub metrics: Vec<(String, f64)>,
    /// What the caller should do.
    pub remediation: String,
    /// Where the finding was raised.
    pub location: Location,
}

impl Issue {
    /// Start a builder with code defaults for domain, severity, and remediation.
    pub fn builder(code: IssueCode) -> IssueBuilder {
        IssueBuilder {
            code,
            domain: code.default_domain(),
            severity: code.default_severity(),
            title: code.as_str().to_string(),
            message: String::new(),
            numerical_compromise: None,
            meaninglessness: None,
            incremental: None,
            metrics: Vec::new(),
            remediation: code.default_remediation().to_string(),
            location: Location::default(),
        }
    }

    /// Shortcut: code + message + default location.
    pub fn new(code: IssueCode, message: impl Into<String>) -> Self {
        Self::builder(code).message(message).build()
    }

    /// True when this issue documents a numerical compromise.
    pub fn is_compromise(&self) -> bool {
        self.numerical_compromise.is_some()
            || matches!(
                self.code,
                IssueCode::PseudoinverseUsed
                    | IssueCode::RidgeFallbackUsed
                    | IssueCode::TruncatedSvdUsed
                    | IssueCode::JitterInjected
            )
    }

    /// True when this issue says the result has no interpretive value.
    pub fn is_meaningless(&self) -> bool {
        self.meaninglessness.is_some()
            || matches!(
                self.code,
                IssueCode::MeaninglessFit
                    | IssueCode::ConstantTarget
                    | IssueCode::PredictionsAreConstant
                    | IssueCode::InterceptOnlyCollapse
                    | IssueCode::RankZero
            )
    }
}

impl fmt::Display for Issue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}: {} — {}",
            self.severity, self.code, self.title, self.message
        )?;
        if let Some(c) = &self.numerical_compromise {
            write!(f, " | {c}")?;
        }
        if let Some(m) = &self.meaninglessness {
            write!(f, " | {m}")?;
        }
        if let Some(i) = &self.incremental {
            write!(f, " | {i}")?;
        }
        if !self.metrics.is_empty() {
            write!(f, " | metrics=")?;
            for (i, (k, v)) in self.metrics.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, "{k}={v:.6e}")?;
            }
        }
        write!(f, " | remediation: {}", self.remediation)
    }
}

/// Fluent constructor for [`Issue`].
#[derive(Debug, Clone)]
pub struct IssueBuilder {
    code: IssueCode,
    domain: Domain,
    severity: Severity,
    title: String,
    message: String,
    numerical_compromise: Option<NumericalCompromise>,
    meaninglessness: Option<Meaninglessness>,
    incremental: Option<IncrementalQuality>,
    metrics: Vec<(String, f64)>,
    remediation: String,
    location: Location,
}

impl IssueBuilder {
    /// Override domain.
    pub fn domain(mut self, domain: Domain) -> Self {
        self.domain = domain;
        self
    }

    /// Override severity (policy may raise it later).
    pub fn severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// Override title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the diagnosis text.
    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// Attach a numerical compromise.
    pub fn compromise(mut self, c: NumericalCompromise) -> Self {
        self.numerical_compromise = Some(c);
        self
    }

    /// Attach a meaninglessness diagnosis.
    pub fn meaninglessness(mut self, m: Meaninglessness) -> Self {
        self.meaninglessness = Some(m);
        self
    }

    /// Attach incremental evidence.
    pub fn incremental(mut self, i: IncrementalQuality) -> Self {
        self.incremental = Some(i);
        self
    }

    /// Push a named metric.
    pub fn metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.push((name.into(), value));
        self
    }

    /// Override remediation.
    pub fn remediation(mut self, text: impl Into<String>) -> Self {
        self.remediation = text.into();
        self
    }

    /// Set source location.
    pub fn location(mut self, location: Location) -> Self {
        self.location = location;
        self
    }

    /// Finish the issue. Empty messages fall back to the remediation sentence.
    pub fn build(self) -> Issue {
        let message = if self.message.is_empty() {
            self.remediation.clone()
        } else {
            self.message
        };
        Issue {
            code: self.code,
            domain: self.domain,
            severity: self.severity,
            title: self.title,
            message,
            numerical_compromise: self.numerical_compromise,
            meaninglessness: self.meaninglessness,
            incremental: self.incremental,
            metrics: self.metrics,
            remediation: self.remediation,
            location: self.location,
        }
    }
}
