use crate::{Error, Result};

/// Projected online gradient descent on an Euclidean ball.
///
/// The caller supplies one convex loss gradient per round. The decision used
/// for that round must be read before calling [`update`](Self::update).
#[derive(Clone, Debug)]
pub struct OnlineGradientDescent {
    learning_rate: f64,
    radius: f64,
    decision: Vec<f64>,
    cumulative_gradient: Vec<f64>,
    rounds: usize,
}

impl OnlineGradientDescent {
    /// Creates a zero-initialized learner with a fixed step size and radius.
    ///
    /// # Errors
    ///
    /// Returns an error for zero dimension or a non-positive/non-finite option.
    pub fn new(dimension: usize, learning_rate: f64, radius: f64) -> Result<Self> {
        if dimension == 0 {
            return Err(Error::Empty { name: "dimension" });
        }
        for (name, value) in [("learning_rate", learning_rate), ("radius", radius)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::InvalidOption {
                    name,
                    requirement: "finite and positive",
                });
            }
        }
        Ok(Self {
            learning_rate,
            radius,
            decision: vec![0.0; dimension],
            cumulative_gradient: vec![0.0; dimension],
            rounds: 0,
        })
    }

    /// Current feasible decision vector.
    #[must_use]
    pub fn decision(&self) -> &[f64] {
        &self.decision
    }
    /// Number of incorporated gradients.
    #[must_use]
    pub const fn rounds(&self) -> usize {
        self.rounds
    }
    /// Sum of all gradients, useful for evaluating linearized regret.
    #[must_use]
    pub fn cumulative_gradient(&self) -> &[f64] {
        &self.cumulative_gradient
    }

    /// Applies `x <- projection(x - learning_rate * gradient)`.
    ///
    /// # Errors
    ///
    /// Returns an error for a mismatched, non-finite, or overflowing gradient.
    pub fn update(&mut self, gradient: &[f64]) -> Result<()> {
        if gradient.len() != self.decision.len() {
            return Err(Error::Length {
                name: "gradient",
                expected: self.decision.len(),
                actual: gradient.len(),
            });
        }
        for (index, value) in gradient.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(Error::NonFinite {
                    name: "gradient",
                    index,
                });
            }
        }
        let mut next_decision = self.decision.clone();
        let mut next_cumulative = self.cumulative_gradient.clone();
        let mut squared_norm = 0.0;
        for ((decision, cumulative), gradient) in next_decision
            .iter_mut()
            .zip(&mut next_cumulative)
            .zip(gradient.iter().copied())
        {
            *decision -= self.learning_rate * gradient;
            *cumulative += gradient;
            squared_norm = decision.mul_add(*decision, squared_norm);
            if !decision.is_finite() || !cumulative.is_finite() {
                return Err(Error::NumericalOverflow {
                    operation: "gradient update",
                });
            }
        }
        let norm = squared_norm.sqrt();
        if norm > self.radius {
            let scale = self.radius / norm;
            for value in &mut next_decision {
                *value *= scale;
            }
        }
        self.decision = next_decision;
        self.cumulative_gradient = next_cumulative;
        self.rounds += 1;
        Ok(())
    }
}
