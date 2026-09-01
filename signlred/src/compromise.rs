//! Record of a numerical compromise: a different computation than requested.

use core::fmt;

/// Class of numerical substitution (AGENTS.md D8).
#[derive(Debug, Clone, PartialEq)]
pub enum CompromiseKind {
    /// Free-form compromise; read the string fields.
    Unspecified,
    /// A log-probability was replaced by `floor` (only when `Policy::log_prob_floor` is `Some`).
    ProbabilityClamped {
        /// Value before the floor.
        original: f64,
        /// Floor that was applied.
        floor: f64,
    },
    /// Linear-domain evaluation underflowed; the computation continued in log space.
    LogDomainFallback,
}

impl Default for CompromiseKind {
    fn default() -> Self {
        Self::Unspecified
    }
}

/// What was asked for, what was actually computed, and why that substitution
/// changes the meaning of the result.
///
/// `ojizou-san` persists these records. Discarding them hides that the returned
/// number is not the estimand the caller named.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericalCompromise {
    /// Estimand or algorithm the caller requested (`"OLS via (X'X)^{-1}X'y"`).
    pub original_intent: String,
    /// Computation that actually ran (`"thin QR + truncated SVD, rank 3"`).
    pub actual_computation: String,
    /// Why the substitution was required at working precision.
    pub why_necessary: String,
    /// Crude absolute / residual error bound when one can be stated.
    pub expected_error_bound: Option<f64>,
    /// Modelling or numerical assumptions that no longer hold.
    pub assumptions_violated: Vec<String>,
    /// How a human should change their interpretation of coefficients / p-values.
    pub interpretability_impact: String,
    /// Structured kind; [`CompromiseKind::Unspecified`] for legacy free-form records.
    pub kind: CompromiseKind,
}

impl NumericalCompromise {
    /// Builder-style constructor.
    pub fn new(
        original_intent: impl Into<String>,
        actual_computation: impl Into<String>,
        why_necessary: impl Into<String>,
        interpretability_impact: impl Into<String>,
    ) -> Self {
        Self {
            original_intent: original_intent.into(),
            actual_computation: actual_computation.into(),
            why_necessary: why_necessary.into(),
            expected_error_bound: None,
            assumptions_violated: Vec::new(),
            interpretability_impact: interpretability_impact.into(),
            kind: CompromiseKind::Unspecified,
        }
    }

    /// A log-probability was clamped to `Policy::log_prob_floor`.
    pub fn probability_clamped(original: f64, floor: f64) -> Self {
        Self {
            original_intent: "retain the computed log-probability".into(),
            actual_computation: format!("clamp log-probability {original} to {floor}"),
            why_necessary: "Policy::log_prob_floor is Some; the caller opted into a floor".into(),
            expected_error_bound: Some((original - floor).abs()),
            assumptions_violated: vec!["untruncated support of the density".into()],
            interpretability_impact:
                "tails are heavier than the model; EM and model selection are biased toward this floor"
                    .into(),
            kind: CompromiseKind::ProbabilityClamped { original, floor },
        }
    }

    /// Linear-domain evaluation underflowed; work continued in the log domain.
    pub fn log_domain_fallback(why: impl Into<String>) -> Self {
        Self {
            original_intent: "evaluate a density or scale factor in linear space".into(),
            actual_computation: "log-domain evaluation (logsumexp / shifted exp)".into(),
            why_necessary: why.into(),
            expected_error_bound: None,
            assumptions_violated: Vec::new(),
            interpretability_impact:
                "the numeric path changed; the estimand is the same if the log-space identity holds"
                    .into(),
            kind: CompromiseKind::LogDomainFallback,
        }
    }

    /// Attach a residual / perturbation bound.
    pub fn with_bound(mut self, bound: f64) -> Self {
        self.expected_error_bound = Some(bound);
        self
    }

    /// Record an assumption that was broken by the compromise.
    pub fn violate(mut self, assumption: impl Into<String>) -> Self {
        self.assumptions_violated.push(assumption.into());
        self
    }
}

impl fmt::Display for NumericalCompromise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "compromise: intended [{}] actually [{}] because [{}]; interpretation: {}",
            self.original_intent,
            self.actual_computation,
            self.why_necessary,
            self.interpretability_impact
        )?;
        if let Some(b) = self.expected_error_bound {
            write!(f, " (bound≈{b:e})")?;
        }
        Ok(())
    }
}
