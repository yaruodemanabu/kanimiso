use crate::{ucb::validate_feedback, Choice, Error, Result};

/// Exponential-weights policy for adversarial bandit rewards.
///
/// Randomness is supplied explicitly to [`select`](Self::select), making the
/// interaction deterministic and independent of any RNG implementation.
#[derive(Clone, Debug)]
pub struct ExpWeights {
    exploration: f64,
    log_weights: Vec<f64>,
    rounds: usize,
}
impl ExpWeights {
    /// Creates a policy with uniform exploration in `(0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns an error for zero arms or exploration outside `(0, 1]`.
    pub fn new(arms: usize, exploration: f64) -> Result<Self> {
        if arms == 0 {
            return Err(Error::NoArms);
        }
        if !(exploration.is_finite() && 0.0 < exploration && exploration <= 1.0) {
            return Err(Error::InvalidOption {
                name: "exploration",
                requirement: "finite and in (0, 1]",
            });
        }
        Ok(Self {
            exploration,
            log_weights: vec![0.0; arms],
            rounds: 0,
        })
    }
    /// Current sampling distribution.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn probabilities(&self) -> Vec<f64> {
        let largest = self
            .log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut weights: Vec<f64> = self
            .log_weights
            .iter()
            .map(|weight| (*weight - largest).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let arms = weights.len() as f64;
        for weight in &mut weights {
            *weight = (1.0 - self.exploration) * *weight / total + self.exploration / arms;
        }
        weights
    }
    /// Maps an external uniform variate in `[0, 1)` to an arm.
    ///
    /// # Errors
    ///
    /// Returns an error unless `sample` is finite and lies in `[0, 1)`.
    pub fn select(&self, sample: f64) -> Result<Choice> {
        if !sample.is_finite() || !(0.0..1.0).contains(&sample) {
            return Err(Error::InvalidSample);
        }
        let probabilities = self.probabilities();
        let mut cumulative = 0.0;
        for (arm, probability) in probabilities.iter().copied().enumerate() {
            cumulative += probability;
            if sample < cumulative || arm + 1 == probabilities.len() {
                return Ok(Choice::new(arm, probability, self.rounds));
            }
        }
        unreachable!("non-empty normalized distribution selects an arm")
    }
    /// Applies the importance-weighted reward estimate for `choice`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid rewards, stale choices, or overflow.
    #[allow(clippy::cast_precision_loss)]
    pub fn update(&mut self, choice: Choice, reward: f64) -> Result<()> {
        validate_feedback(choice, reward, self.log_weights.len(), self.rounds)?;
        let expected_probability = self.probabilities()[choice.arm()];
        if choice.probability().to_bits() != expected_probability.to_bits() {
            return Err(Error::StaleChoice);
        }
        let estimated_reward = reward / choice.probability();
        let increment = self.exploration * estimated_reward / self.log_weights.len() as f64;
        let next = self.log_weights[choice.arm()] + increment;
        if !next.is_finite() {
            return Err(Error::NumericalOverflow {
                operation: "exponential bandit weight",
            });
        }
        self.log_weights[choice.arm()] = next;
        self.rounds += 1;
        Ok(())
    }
    /// Number of completed interactions.
    #[must_use]
    pub const fn rounds(&self) -> usize {
        self.rounds
    }
}
