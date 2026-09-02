//! Finite categorical emission with unfloored maximum-likelihood updates.

use super::Emission;
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use signlred::{Failure, Issue, IssueCode, Result, Severity};

/// A finite categorical emission parameterized by normalized probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalEmission {
    probabilities: Vector,
}

/// Weighted category counts for a categorical M-step.
///
/// Create an empty accumulator with [`Default`] and populate it through
/// [`Emission::accumulate`]; count storage is intentionally private.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CategoricalStats {
    counts: Vec<f64>,
}

impl CategoricalEmission {
    /// Construct an emission by normalizing explicit category weights.
    ///
    /// The vector must be non-empty, finite, non-negative, and have positive
    /// total weight. Zero individual weights remain exact zero probabilities.
    pub fn from_weights(weights: Vector) -> Result<Self> {
        if weights.is_empty() {
            return Err(failure(
                "from_weights",
                IssueCode::InvalidParameter,
                "CategoricalEmission requires at least one category weight",
            ));
        }
        if weights.as_slice().iter().any(|weight| !weight.is_finite()) {
            return Err(failure(
                "from_weights",
                IssueCode::NonFiniteInput,
                "CategoricalEmission weights must be finite",
            ));
        }
        if weights.as_slice().iter().any(|weight| *weight < 0.0) {
            return Err(failure(
                "from_weights",
                IssueCode::InvalidParameter,
                "CategoricalEmission weights must be non-negative",
            ));
        }
        let scale = weights.as_slice().iter().copied().fold(0.0_f64, f64::max);
        if scale == 0.0 {
            return Err(failure(
                "from_weights",
                IssueCode::InvalidParameter,
                "CategoricalEmission weights must have positive sum",
            ));
        }
        let scaled_sum: f64 = weights
            .as_slice()
            .iter()
            .map(|weight| *weight / scale)
            .sum();
        if !scaled_sum.is_finite() || scaled_sum <= 0.0 {
            return Err(failure(
                "from_weights",
                IssueCode::InvalidParameter,
                "CategoricalEmission normalized weight sum is invalid",
            ));
        }
        let probabilities = Vector::from_iter(
            weights
                .as_slice()
                .iter()
                .map(|weight| (*weight / scale) / scaled_sum),
        );
        Ok(Self { probabilities })
    }

    /// Borrow the normalized category probabilities in category-index order.
    pub fn probabilities(&self) -> &Vector {
        &self.probabilities
    }

    fn poison(stats: &mut CategoricalStats, categories: usize) {
        stats.counts = vec![f64::NAN; categories];
    }
}

impl Emission for CategoricalEmission {
    type Observation = usize;
    type SufficientStats = CategoricalStats;

    fn observations(x: &Matrix) -> Result<Vec<Self::Observation>> {
        if x.ncols() != 1 {
            return Err(failure(
                "observations",
                IssueCode::DimensionMismatch,
                format!(
                    "CategoricalEmission observations require exactly one column, got {}",
                    x.ncols()
                ),
            ));
        }
        let upper_exclusive = usize::MAX as f64 + 1.0;
        let mut observations = Vec::with_capacity(x.nrows());
        for row in 0..x.nrows() {
            let value = x.get(row, 0);
            if !value.is_finite() {
                return Err(failure(
                    "observations",
                    IssueCode::NonFiniteInput,
                    format!("CategoricalEmission observation at row {row} is not finite"),
                ));
            }
            if value < 0.0 || value.fract() != 0.0 || value >= upper_exclusive {
                return Err(failure(
                    "observations",
                    IssueCode::InvalidParameter,
                    format!(
                        "CategoricalEmission observation at row {row} must be a non-negative integer in usize range; got {value}"
                    ),
                ));
            }
            observations.push(value as usize);
        }
        Ok(observations)
    }

    fn log_prob(&self, obs: &Self::Observation) -> f64 {
        match self.probabilities.as_slice().get(*obs).copied() {
            Some(probability) if probability > 0.0 => probability.ln(),
            _ => f64::NEG_INFINITY,
        }
    }

    fn accumulate(&self, obs: &Self::Observation, weight: f64, stats: &mut Self::SufficientStats) {
        if weight == 0.0 {
            return;
        }
        let categories = self.probabilities.len();
        if !weight.is_finite() || weight < 0.0 || *obs >= categories {
            Self::poison(stats, categories);
            return;
        }
        if stats.counts.is_empty() {
            stats.counts = vec![0.0; categories];
        }
        if stats.counts.len() != categories {
            Self::poison(stats, categories);
            return;
        }
        stats.counts[*obs] += weight;
        if !stats.counts[*obs].is_finite() {
            Self::poison(stats, categories);
        }
    }

    fn maximize(&mut self, stats: &Self::SufficientStats, ctx: &mut FitCtx) -> Result<()> {
        if stats.counts.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::UnreachableState)
                    .severity(Severity::Warning)
                    .message(
                        "CategoricalEmission maximize received zero posterior occupancy; parameters are unchanged",
                    )
                    .build(),
            );
            return Ok(());
        }
        if stats.counts.len() != self.probabilities.len() {
            return Err(failure(
                "maximize",
                IssueCode::DimensionMismatch,
                "CategoricalEmission count dimension differs from the emission",
            ));
        }
        if stats
            .counts
            .iter()
            .any(|count| !count.is_finite() || *count < 0.0)
        {
            return Err(failure(
                "maximize",
                IssueCode::NonFiniteInput,
                "CategoricalEmission counts must be finite and non-negative",
            ));
        }
        let occupancy: f64 = stats.counts.iter().sum();
        if occupancy == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnreachableState)
                    .severity(Severity::Warning)
                    .message(
                        "CategoricalEmission maximize received zero posterior occupancy; parameters are unchanged",
                    )
                    .build(),
            );
            return Ok(());
        }
        if !occupancy.is_finite() || occupancy < 0.0 {
            return Err(failure(
                "maximize",
                IssueCode::NonFiniteInput,
                "CategoricalEmission posterior occupancy is invalid",
            ));
        }
        self.probabilities = Vector::from_iter(stats.counts.iter().map(|count| *count / occupancy));
        Ok(())
    }
}

fn failure(operation: &'static str, code: IssueCode, message: impl Into<String>) -> Failure {
    Failure::from_issue(
        "CategoricalEmission",
        operation,
        Issue::builder(code).message(message).build(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_normalize_and_log_prob_matches_closed_form() {
        let emission = CategoricalEmission::from_weights(Vector::from_slice(&[2.0, 1.0, 1.0]))
            .expect("valid weights");
        assert_eq!(emission.probabilities().as_slice(), &[0.5, 0.25, 0.25]);
        let got = emission.log_prob(&0);
        // Measured |error| = 0.0 on 2026-09-02; allow two ulps around -ln(2).
        assert!((got + std::f64::consts::LN_2).abs() <= 2.0 * f64::EPSILON);
    }

    #[test]
    fn constructor_rejects_invalid_weights() {
        for (weights, code) in [
            (vec![0.0, 0.0], IssueCode::InvalidParameter),
            (vec![1.0, -1.0], IssueCode::InvalidParameter),
            (vec![1.0, f64::NAN], IssueCode::NonFiniteInput),
        ] {
            let failure = CategoricalEmission::from_weights(Vector::from_slice(&weights))
                .expect_err("invalid weights");
            assert_eq!(failure.primary().code, code);
            assert_eq!(failure.report.algorithm, "CategoricalEmission");
            assert_eq!(failure.report.operation, "from_weights");
        }
    }

    #[test]
    fn observations_reject_wrong_shape_and_invalid_codes() {
        let wrong_shape = Matrix::zeros(2, 2);
        let failure = CategoricalEmission::observations(&wrong_shape).unwrap_err();
        assert_eq!(failure.primary().code, IssueCode::DimensionMismatch);

        for value in [f64::NAN, -1.0, 1.5, usize::MAX as f64 + 1.0] {
            let x = Matrix::from_row_major(1, 1, &[value]);
            let failure = CategoricalEmission::observations(&x).unwrap_err();
            assert!(matches!(
                failure.primary().code,
                IssueCode::NonFiniteInput | IssueCode::InvalidParameter
            ));
            assert_eq!(failure.report.operation, "observations");
        }

        let upper_exclusive = usize::MAX as f64 + 1.0;
        let exponent = usize::BITS.saturating_sub(53);
        let spacing_below_upper = if exponent > 0 {
            2.0_f64.powi(exponent as i32)
        } else {
            1.0
        };
        let largest_representable = upper_exclusive - spacing_below_upper;
        let x = Matrix::from_row_major(1, 1, &[largest_representable]);
        let observations = CategoricalEmission::observations(&x).expect("in-range integer");
        assert_eq!(observations[0] as f64, largest_representable);
    }

    #[test]
    fn m_step_uses_unfloored_mle() {
        let mut emission = CategoricalEmission::from_weights(Vector::from_slice(&[1.0, 1.0, 1.0]))
            .expect("valid weights");
        let mut stats = CategoricalStats::default();
        emission.accumulate(&0, 1.0, &mut stats);
        emission.accumulate(&2, 3.0, &mut stats);
        let mut ctx = FitCtx::new("CategoricalEmission", "maximize");
        emission
            .maximize(&stats, &mut ctx)
            .expect("identified M-step");
        assert_eq!(emission.probabilities().as_slice(), &[0.25, 0.0, 0.75]);
        assert_eq!(emission.log_prob(&1), f64::NEG_INFINITY);
    }

    #[test]
    fn zero_occupancy_warns_and_preserves_parameters() {
        let mut emission = CategoricalEmission::from_weights(Vector::from_slice(&[3.0, 1.0]))
            .expect("valid weights");
        let before = emission.clone();
        let mut ctx = FitCtx::new("CategoricalEmission", "maximize");
        emission
            .maximize(&CategoricalStats::default(), &mut ctx)
            .expect("warning is non-fatal");
        assert_eq!(emission, before);
        assert!(ctx.report.contains(IssueCode::UnreachableState));
        assert!(ctx.report.has_warning());
    }
}
