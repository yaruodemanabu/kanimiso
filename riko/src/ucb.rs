use crate::{Choice, Error, Result};

/// Upper-confidence-bound policy for bounded stochastic rewards.
///
/// Every arm is selected once, in index order. Later rounds maximize
/// `mean + exploration * sqrt(ln(total pulls) / arm pulls)`, breaking ties by
/// the smallest arm index.
#[derive(Clone, Debug)]
pub struct Ucb {
    exploration: f64,
    pulls: Vec<usize>,
    reward_sums: Vec<f64>,
    rounds: usize,
}
impl Ucb {
    /// Creates a policy for rewards in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns an error for zero arms or invalid exploration.
    pub fn new(arms: usize, exploration: f64) -> Result<Self> {
        if arms == 0 {
            return Err(Error::NoArms);
        }
        if !exploration.is_finite() || exploration < 0.0 {
            return Err(Error::InvalidOption {
                name: "exploration",
                requirement: "finite and non-negative",
            });
        }
        Ok(Self {
            exploration,
            pulls: vec![0; arms],
            reward_sums: vec![0.0; arms],
            rounds: 0,
        })
    }
    /// Selects an arm deterministically from the current statistics.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn select(&self) -> Choice {
        if let Some(arm) = self.pulls.iter().position(|pulls| *pulls == 0) {
            return Choice::new(arm, 1.0, self.rounds);
        }
        let log_rounds = (self.rounds as f64).ln();
        let mut arm = 0;
        for candidate in 1..self.pulls.len() {
            if self.score(candidate, log_rounds) > self.score(arm, log_rounds) {
                arm = candidate;
            }
        }
        Choice::new(arm, 1.0, self.rounds)
    }
    /// Incorporates the reward associated with the most recent choice.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid rewards, stale choices, or overflow.
    pub fn update(&mut self, choice: Choice, reward: f64) -> Result<()> {
        validate_feedback(choice, reward, self.pulls.len(), self.rounds)?;
        let sum = self.reward_sums[choice.arm()] + reward;
        if !sum.is_finite() {
            return Err(Error::NumericalOverflow {
                operation: "cumulative arm reward",
            });
        }
        self.reward_sums[choice.arm()] = sum;
        self.pulls[choice.arm()] += 1;
        self.rounds += 1;
        Ok(())
    }
    #[allow(clippy::cast_precision_loss)]
    fn score(&self, arm: usize, log_rounds: f64) -> f64 {
        self.reward_sums[arm] / self.pulls[arm] as f64
            + self.exploration * (log_rounds / self.pulls[arm] as f64).sqrt()
    }
    /// Pull counts by arm.
    #[must_use]
    pub fn pulls(&self) -> &[usize] {
        &self.pulls
    }
    /// Empirical reward mean, or `None` for an unpulled arm.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_reward(&self, arm: usize) -> Option<f64> {
        self.pulls
            .get(arm)
            .and_then(|pulls| (*pulls > 0).then(|| self.reward_sums[arm] / *pulls as f64))
    }
    /// Number of completed interactions.
    #[must_use]
    pub const fn rounds(&self) -> usize {
        self.rounds
    }
}

pub(crate) fn validate_feedback(
    choice: Choice,
    reward: f64,
    arms: usize,
    round: usize,
) -> Result<()> {
    if choice.arm() >= arms {
        return Err(Error::InvalidArm {
            arm: choice.arm(),
            arms,
        });
    }
    if choice.round() != round {
        return Err(Error::StaleChoice);
    }
    if !reward.is_finite() || !(0.0..=1.0).contains(&reward) {
        return Err(Error::InvalidReward);
    }
    Ok(())
}
