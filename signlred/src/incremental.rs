//! Quality and explainability of a single online / incremental update.

use core::fmt;

/// Evidence attached to every `partial_fit` / recursive update.
///
/// Incremental algorithms change an estimand. This record is the *reason* the
/// new state should (or should not) be believed: what moved, how much
/// information arrived, and whether identification survived forgetting.
#[derive(Debug, Clone, PartialEq)]
pub struct IncrementalQuality {
    /// 0-based index of this update since initialization.
    pub update_index: u64,
    /// Observations consumed in this batch.
    pub batch_size: usize,
    /// Cumulative raw observation count (no forgetting).
    pub n_seen: u64,
    /// Effective sample size after forgetting / windowing.
    pub effective_sample_size: f64,
    /// Forgetting factor λ ∈ (0, 1], if the algorithm uses one.
    pub forgetting_factor: Option<f64>,
    /// L2 norm of the parameter delta induced by this batch.
    pub parameter_delta_norm: Option<f64>,
    /// Max abs coordinate-wise parameter delta.
    pub parameter_delta_max: Option<f64>,
    /// Names of the coordinates that moved most (highest |Δ|).
    pub top_moved_parameters: Vec<(String, f64)>,
    /// Residual / loss *before* the update.
    pub loss_before: Option<f64>,
    /// Residual / loss *after* the update.
    pub loss_after: Option<f64>,
    /// Approximate information in the batch (e.g. Δℓ, Fisher trace, SSE drop).
    pub information_gain: Option<f64>,
    /// Optional drift statistic computed on this batch.
    pub drift_statistic: Option<f64>,
    /// Whether the post-update state is still identified.
    pub still_identified: bool,
    /// Whether this batch was treated as warm-up rather than inference.
    pub warmup: bool,
    /// Narrative: what changed and why a human should care.
    pub explanation: String,
}

impl IncrementalQuality {
    /// Start an explanation for update `k`.
    pub fn new(update_index: u64, batch_size: usize, n_seen: u64) -> Self {
        Self {
            update_index,
            batch_size,
            n_seen,
            effective_sample_size: n_seen as f64,
            forgetting_factor: None,
            parameter_delta_norm: None,
            parameter_delta_max: None,
            top_moved_parameters: Vec::new(),
            loss_before: None,
            loss_after: None,
            information_gain: None,
            drift_statistic: None,
            still_identified: true,
            warmup: false,
            explanation: String::new(),
        }
    }

    /// True when the batch added no usable information.
    pub fn is_uninformative(&self, info_eps: f64) -> bool {
        match self.information_gain {
            Some(g) => !g.is_finite() || g.abs() <= info_eps,
            None => self
                .parameter_delta_norm
                .map(|d| !d.is_finite() || d <= info_eps)
                .unwrap_or(false),
        }
    }

    /// Loss improvement (`before - after`) when both sides exist.
    pub fn loss_drop(&self) -> Option<f64> {
        match (self.loss_before, self.loss_after) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => Some(a - b),
            _ => None,
        }
    }
}

impl fmt::Display for IncrementalQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "update#{} batch={} n_seen={} n_eff={:.4} identified={} warmup={}",
            self.update_index,
            self.batch_size,
            self.n_seen,
            self.effective_sample_size,
            self.still_identified,
            self.warmup
        )?;
        if let Some(d) = self.parameter_delta_norm {
            write!(f, " ||Δθ||={d:.6e}")?;
        }
        if let Some(g) = self.information_gain {
            write!(f, " info={g:.6e}")?;
        }
        if !self.explanation.is_empty() {
            write!(f, " — {}", self.explanation)?;
        }
        Ok(())
    }
}
