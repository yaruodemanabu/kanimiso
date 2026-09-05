use crate::{Error, Result};

/// Exponentially weighted forecaster for full-information expert advice.
///
/// Each round, call [`probabilities`](Self::probabilities), incur the expected
/// loss of that mixture, and pass every expert's loss to [`update`](Self::update).
#[derive(Clone, Debug)]
pub struct Hedge {
    learning_rate: f64,
    cumulative_losses: Vec<f64>,
    learner_loss: f64,
    rounds: usize,
}

impl Hedge {
    /// Creates a forecaster for `experts`, with losses constrained to `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns an error for zero experts or an invalid learning rate.
    pub fn new(experts: usize, learning_rate: f64) -> Result<Self> {
        if experts == 0 {
            return Err(Error::Empty { name: "experts" });
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(Error::InvalidOption {
                name: "learning_rate",
                requirement: "finite and positive",
            });
        }
        Ok(Self {
            learning_rate,
            cumulative_losses: vec![0.0; experts],
            learner_loss: 0.0,
            rounds: 0,
        })
    }

    /// Returns the current normalized expert mixture.
    #[must_use]
    pub fn probabilities(&self) -> Vec<f64> {
        let largest = self
            .cumulative_losses
            .iter()
            .map(|loss| -self.learning_rate * loss)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut weights: Vec<f64> = self
            .cumulative_losses
            .iter()
            .map(|loss| (-self.learning_rate * loss - largest).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= total;
        }
        weights
    }

    /// Records one complete loss vector and returns the mixture's expected loss.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed losses or an overflowing cumulative loss.
    pub fn update(&mut self, losses: &[f64]) -> Result<f64> {
        validate_losses(losses, self.cumulative_losses.len())?;
        let probabilities = self.probabilities();
        let round_loss: f64 = probabilities
            .iter()
            .zip(losses)
            .map(|(probability, loss)| probability * loss)
            .sum();
        let next = self.learner_loss + round_loss;
        if !next.is_finite() {
            return Err(Error::NumericalOverflow {
                operation: "cumulative learner loss",
            });
        }
        let mut next_losses = self.cumulative_losses.clone();
        for (cumulative, loss) in next_losses.iter_mut().zip(losses) {
            *cumulative += loss;
            if !cumulative.is_finite() {
                return Err(Error::NumericalOverflow {
                    operation: "cumulative expert loss",
                });
            }
        }
        self.cumulative_losses = next_losses;
        self.learner_loss = next;
        self.rounds += 1;
        Ok(round_loss)
    }

    /// Number of completed rounds.
    #[must_use]
    pub const fn rounds(&self) -> usize {
        self.rounds
    }
    /// Cumulative expected loss incurred by the learner.
    #[must_use]
    pub const fn cumulative_loss(&self) -> f64 {
        self.learner_loss
    }
    /// Per-expert cumulative losses.
    #[must_use]
    pub fn expert_losses(&self) -> &[f64] {
        &self.cumulative_losses
    }
    /// External regret against the best fixed expert in hindsight.
    #[must_use]
    pub fn regret(&self) -> f64 {
        self.learner_loss
            - self
                .cumulative_losses
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
    }
}

fn validate_losses(losses: &[f64], expected: usize) -> Result<()> {
    if losses.len() != expected {
        return Err(Error::Length {
            name: "losses",
            expected,
            actual: losses.len(),
        });
    }
    for (index, loss) in losses.iter().copied().enumerate() {
        if !loss.is_finite() {
            return Err(Error::NonFinite {
                name: "losses",
                index,
            });
        }
        if !(0.0..=1.0).contains(&loss) {
            return Err(Error::LossOutOfRange { index });
        }
    }
    Ok(())
}
