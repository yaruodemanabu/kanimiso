//! Accumulator of quality issues for one operation.

use crate::compromise::NumericalCompromise;
use crate::failure::Failure;
use crate::issue::Issue;
use crate::policy::Policy;
use crate::qualified::Qualified;
use crate::severity::Severity;
use crate::{IssueCode, Result};
use core::fmt;

/// Quality ledger for a single algorithm operation (`fit`, `predict`, …).
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// Estimator / procedure name (`"ols"`, `"gaussian_hmm"`).
    pub algorithm: String,
    /// Operation name (`"fit"`, `"partial_fit"`, `"forecast"`).
    pub operation: String,
    issues: Vec<Issue>,
    /// n, when known.
    pub n_samples: Option<usize>,
    /// p, when known.
    pub n_features: Option<usize>,
    /// Free parameters, when known.
    pub n_parameters: Option<usize>,
}

impl Report {
    /// Create an empty report.
    pub fn new(algorithm: impl Into<String>, operation: impl Into<String>) -> Self {
        Self {
            algorithm: algorithm.into(),
            operation: operation.into(),
            issues: Vec::new(),
            n_samples: None,
            n_features: None,
            n_parameters: None,
        }
    }

    /// Record the design shape.
    pub fn set_sample_shape(&mut self, n_samples: usize, n_features: usize) {
        self.n_samples = Some(n_samples);
        self.n_features = Some(n_features);
    }

    /// Record the number of free parameters.
    pub fn set_n_parameters(&mut self, n_parameters: usize) {
        self.n_parameters = Some(n_parameters);
    }

    /// Apply policy and push an issue.
    pub fn push(&mut self, issue: Issue) {
        self.push_with_policy(Policy::default(), issue);
    }

    /// Apply an explicit policy and push.
    pub fn push_with_policy(&mut self, policy: Policy, issue: Issue) {
        self.issues.push(policy.apply(issue));
    }

    /// Issues in insertion order.
    pub fn issues(&self) -> &[Issue] {
        &self.issues
    }

    /// Consume and return issues.
    pub fn into_issues(self) -> Vec<Issue> {
        self.issues
    }

    /// Merge another report's issues into this one. Shape metadata is kept
    /// from `self` unless missing.
    pub fn merge(&mut self, other: Report) {
        if self.n_samples.is_none() {
            self.n_samples = other.n_samples;
        }
        if self.n_features.is_none() {
            self.n_features = other.n_features;
        }
        if self.n_parameters.is_none() {
            self.n_parameters = other.n_parameters;
        }
        self.issues.extend(other.issues);
    }

    /// True if any issue is `Fatal`.
    pub fn has_fatal(&self) -> bool {
        self.issues.iter().any(|i| i.severity == Severity::Fatal)
    }

    /// True if any issue is `Error` or worse.
    pub fn has_error(&self) -> bool {
        self.issues.iter().any(|i| i.severity <= Severity::Error)
    }

    /// True if any issue is `Warning`.
    pub fn has_warning(&self) -> bool {
        self.issues.iter().any(|i| i.severity == Severity::Warning)
    }

    /// True if any issue is a numerical compromise.
    pub fn has_compromise(&self) -> bool {
        self.issues.iter().any(Issue::is_compromise)
    }

    /// True if any issue is a meaninglessness diagnosis.
    pub fn has_meaningless(&self) -> bool {
        self.issues.iter().any(Issue::is_meaningless)
    }

    /// Iterator over compromise records.
    pub fn compromises(&self) -> impl Iterator<Item = &NumericalCompromise> {
        self.issues
            .iter()
            .filter_map(|i| i.numerical_compromise.as_ref())
    }

    /// Whether a given code is present.
    pub fn contains(&self, code: IssueCode) -> bool {
        self.issues.iter().any(|i| i.code == code)
    }

    /// Most severe issue, if any.
    pub fn primary(&self) -> Option<&Issue> {
        self.issues.iter().min_by_key(|i| i.severity)
    }

    /// Finish with the default policy.
    pub fn finish<T>(self, value: T) -> Result<Qualified<T>> {
        self.finish_with_policy(Policy::default(), value)
    }

    /// Finish: abort on policy, otherwise wrap the value.
    pub fn finish_with_policy<T>(self, policy: Policy, value: T) -> Result<Qualified<T>> {
        if let Some(failure) = self.clone().into_failure(&policy) {
            return Err(failure);
        }
        Ok(Qualified {
            value,
            report: self,
        })
    }

    /// Convert this report into a [`Failure`] if policy requires abort.
    pub fn into_failure(self, policy: &Policy) -> Option<Failure> {
        let aborting: Vec<&Issue> = self
            .issues
            .iter()
            .filter(|i| policy.must_abort(i))
            .collect();
        if aborting.is_empty() {
            return None;
        }
        let primary = aborting
            .into_iter()
            .min_by_key(|i| i.severity)
            .cloned()
            .expect("non-empty");
        Some(Failure {
            primary,
            report: self,
        })
    }

    /// Human-readable multi-line dump.
    pub fn render(&self) -> String {
        let mut out = format!(
            "signlred report algorithm={} operation={} n={:?} p={:?} params={:?} issues={}\n",
            self.algorithm,
            self.operation,
            self.n_samples,
            self.n_features,
            self.n_parameters,
            self.issues.len()
        );
        for (i, issue) in self.issues.iter().enumerate() {
            out.push_str(&format!("  [{i}] {issue}\n"));
        }
        out
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}
