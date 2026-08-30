//! Record of a numerical compromise: a different computation than requested.

use core::fmt;

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
