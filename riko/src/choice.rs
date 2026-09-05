/// Auditable result of selecting an arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Choice {
    arm: usize,
    probability: f64,
    round: usize,
}
impl Choice {
    pub(crate) const fn new(arm: usize, probability: f64, round: usize) -> Self {
        Self {
            arm,
            probability,
            round,
        }
    }
    /// Selected zero-based arm.
    #[must_use]
    pub const fn arm(self) -> usize {
        self.arm
    }
    /// Probability with which the policy selected this arm.
    #[must_use]
    pub const fn probability(self) -> f64 {
        self.probability
    }
    pub(crate) const fn round(self) -> usize {
        self.round
    }
}
