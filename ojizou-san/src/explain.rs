//! Incremental-learning explainability records.

use signlred::IncrementalQuality;
use std::collections::BTreeMap;
use std::fmt;

/// How much a human should trust the post-update state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrustLevel {
    /// Warm-up, unidentified, or zero-information update.
    DoNotUse,
    /// Identified but fragile (small n_eff, drift, large jump).
    Fragile,
    /// Ordinary update; still read the narrative.
    Usable,
}

impl TrustLevel {
    /// Human label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DoNotUse => "do_not_use",
            Self::Fragile => "fragile",
            Self::Usable => "usable",
        }
    }
}

impl fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Human-and-machine explanation of one incremental update.
///
/// Online algorithms in `kanimiso` must produce this on every `partial_fit`.
/// A parameter vector without a narrative is not an acceptable result.
#[derive(Debug, Clone, PartialEq)]
pub struct IncrementalExplain {
    /// Underlying numerical quality record.
    pub quality: IncrementalQuality,
    /// What in the model changed (parameters, sufficient stats, tree splits).
    pub what_changed: String,
    /// Why the algorithm applied that change (gradient, split gain, assignment).
    pub why_changed: String,
    /// Quality of the state before the batch.
    pub before_quality: String,
    /// Quality of the state after the batch.
    pub after_quality: String,
    /// Named contributions (`"feature:age" -> 0.42` of ‖Δθ‖²).
    pub contribution: BTreeMap<String, f64>,
    /// Trust recommendation.
    pub trust: TrustLevel,
    /// Full narrative assembled for logs.
    pub narrative: String,
}

impl IncrementalExplain {
    /// Build from a [`IncrementalQuality`] plus the required prose fields.
    pub fn from_quality(
        quality: IncrementalQuality,
        what_changed: impl Into<String>,
        why_changed: impl Into<String>,
        before_quality: impl Into<String>,
        after_quality: impl Into<String>,
    ) -> Self {
        let what_changed = what_changed.into();
        let why_changed = why_changed.into();
        let before_quality = before_quality.into();
        let after_quality = after_quality.into();
        let trust = trust_of(&quality);
        let narrative = format!(
            "update#{} (batch={}, n_seen={}, n_eff={:.4}, identified={}, warmup={}, trust={}): \
             what={} why={} before={} after={} ||Δθ||={:?} info={:?}",
            quality.update_index,
            quality.batch_size,
            quality.n_seen,
            quality.effective_sample_size,
            quality.still_identified,
            quality.warmup,
            trust,
            what_changed,
            why_changed,
            before_quality,
            after_quality,
            quality.parameter_delta_norm,
            quality.information_gain
        );
        Self {
            quality,
            what_changed,
            why_changed,
            before_quality,
            after_quality,
            contribution: BTreeMap::new(),
            trust,
            narrative,
        }
    }

    /// Insert a contribution share (need not sum to 1; the logger does not renormalize).
    pub fn contribute(mut self, name: impl Into<String>, share: f64) -> Self {
        self.contribution.insert(name.into(), share);
        self
    }
}

fn trust_of(q: &IncrementalQuality) -> TrustLevel {
    if q.warmup || !q.still_identified || q.is_uninformative(1e-15) {
        TrustLevel::DoNotUse
    } else if q.effective_sample_size < 30.0
        || q.drift_statistic
            .map(|d| d.is_finite() && d > 3.0)
            .unwrap_or(false)
    {
        TrustLevel::Fragile
    } else {
        TrustLevel::Usable
    }
}

impl fmt::Display for IncrementalExplain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.narrative)
    }
}
