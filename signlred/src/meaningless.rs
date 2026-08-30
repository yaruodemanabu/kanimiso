//! Diagnosis that a completed computation has no interpretive value.

use core::fmt;

/// How badly a number misleads if it is treated as a real finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpretiveValue {
    /// The number is well-defined but answers a different question than claimed.
    Misleading,
    /// The number is an artifact of a degenerate problem (0/0, 1-1, empty class).
    Vacuous,
    /// Publishing the number would be actively false (e.g. a p-value with no null).
    False,
}

impl InterpretiveValue {
    /// Human label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Misleading => "misleading",
            Self::Vacuous => "vacuous",
            Self::False => "false",
        }
    }
}

impl fmt::Display for InterpretiveValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a fit or statistic must not be consumed as knowledge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Meaninglessness {
    /// What numeric object was produced (`"OLS coefficient vector"`).
    pub what_was_computed: String,
    /// Why that object has no (or the wrong) meaning.
    pub why_meaningless: String,
    /// How it fails as interpretation.
    pub interpretive_value: InterpretiveValue,
    /// What a careful analyst should do instead.
    pub suggested_action: String,
}

impl Meaninglessness {
    /// Constructor.
    pub fn new(
        what_was_computed: impl Into<String>,
        why_meaningless: impl Into<String>,
        interpretive_value: InterpretiveValue,
        suggested_action: impl Into<String>,
    ) -> Self {
        Self {
            what_was_computed: what_was_computed.into(),
            why_meaningless: why_meaningless.into(),
            interpretive_value,
            suggested_action: suggested_action.into(),
        }
    }

    /// Vacuous-result helper used throughout `kanimiso`.
    pub fn vacuous(
        what_was_computed: impl Into<String>,
        why_meaningless: impl Into<String>,
        suggested_action: impl Into<String>,
    ) -> Self {
        Self::new(
            what_was_computed,
            why_meaningless,
            InterpretiveValue::Vacuous,
            suggested_action,
        )
    }
}

impl fmt::Display for Meaninglessness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "meaningless({}): {} — {} | action: {}",
            self.interpretive_value,
            self.what_was_computed,
            self.why_meaningless,
            self.suggested_action
        )
    }
}
