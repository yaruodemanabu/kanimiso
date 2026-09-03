//! Poisson emission with exact non-negative integer observations.

use super::Emission;
use crate::context::FitCtx;
use crate::data::Matrix;
use crate::special::ln_gamma;
use signlred::{Failure, Issue, IssueCode, Result, Severity};

/// A Poisson count emission with a finite, non-negative rate.
#[derive(Clone, Debug, PartialEq)]
pub struct PoissonEmission {
    rate: f64,
}

/// Weighted count sum and posterior occupancy for a Poisson M-step.
///
/// Create an empty accumulator with [`Default`] and populate it through
/// [`Emission::accumulate`]; its numerical fields are intentionally private.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoissonStats {
    weight: f64,
    weighted_sum: f64,
}

impl PoissonEmission {
    /// Construct a Poisson emission from a finite, non-negative rate.
    ///
    /// A zero rate is valid and represents a point mass at count zero.
    pub fn new(rate: f64) -> Result<Self> {
        if !rate.is_finite() {
            return Err(failure(
                "new",
                IssueCode::NonFiniteInput,
                "PoissonEmission rate must be finite",
            ));
        }
        if rate < 0.0 {
            return Err(failure(
                "new",
                IssueCode::InvalidParameter,
                "PoissonEmission rate must be non-negative",
            ));
        }
        Ok(Self { rate })
    }

    /// Return the finite, non-negative Poisson rate.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    fn poison(stats: &mut PoissonStats) {
        stats.weight = f64::NAN;
        stats.weighted_sum = f64::NAN;
    }
}

impl Emission for PoissonEmission {
    type Observation = u64;
    type SufficientStats = PoissonStats;

    fn observations(x: &Matrix) -> Result<Vec<Self::Observation>> {
        if x.ncols() != 1 {
            return Err(failure(
                "observations",
                IssueCode::DimensionMismatch,
                format!(
                    "PoissonEmission observations require exactly one column, got {}",
                    x.ncols()
                ),
            ));
        }
        let upper_exclusive = u64::MAX as f64 + 1.0;
        let mut observations = Vec::with_capacity(x.nrows());
        for row in 0..x.nrows() {
            let value = x.get(row, 0);
            if !value.is_finite() {
                return Err(failure(
                    "observations",
                    IssueCode::NonFiniteInput,
                    format!("PoissonEmission observation at row {row} is not finite"),
                ));
            }
            if value < 0.0 || value.fract() != 0.0 || value >= upper_exclusive {
                return Err(failure(
                    "observations",
                    IssueCode::InvalidParameter,
                    format!(
                        "PoissonEmission observation at row {row} must be a non-negative integer in u64 range; got {value}"
                    ),
                ));
            }
            observations.push(value as u64);
        }
        Ok(observations)
    }

    fn log_prob(&self, obs: &Self::Observation) -> f64 {
        if self.rate == 0.0 {
            return if *obs == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        let count = *obs as f64;
        count * self.rate.ln() - self.rate - ln_gamma(count + 1.0)
    }

    fn accumulate(&self, obs: &Self::Observation, weight: f64, stats: &mut Self::SufficientStats) {
        if weight == 0.0 {
            return;
        }
        if !weight.is_finite() || weight < 0.0 {
            Self::poison(stats);
            return;
        }
        stats.weight += weight;
        stats.weighted_sum += weight * (*obs as f64);
        if !stats.weight.is_finite() || !stats.weighted_sum.is_finite() {
            Self::poison(stats);
        }
    }

    fn maximize(&mut self, stats: &Self::SufficientStats, ctx: &mut FitCtx) -> Result<()> {
        if stats.weight == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnreachableState)
                    .severity(Severity::Warning)
                    .message(
                        "PoissonEmission maximize received zero posterior occupancy; parameters are unchanged",
                    )
                    .build(),
            );
            return Ok(());
        }
        if !stats.weight.is_finite()
            || stats.weight < 0.0
            || !stats.weighted_sum.is_finite()
            || stats.weighted_sum < 0.0
        {
            return Err(failure(
                "maximize",
                IssueCode::NonFiniteInput,
                "PoissonEmission sufficient statistics must be finite and non-negative",
            ));
        }
        let rate = stats.weighted_sum / stats.weight;
        if !rate.is_finite() || rate < 0.0 {
            return Err(failure(
                "maximize",
                IssueCode::NonFiniteInput,
                "PoissonEmission MLE rate is invalid",
            ));
        }
        self.rate = rate;
        Ok(())
    }
}

fn failure(operation: &'static str, code: IssueCode, message: impl Into<String>) -> Failure {
    Failure::from_issue(
        "PoissonEmission",
        operation,
        Issue::builder(code).message(message).build(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_prob_matches_closed_form_and_zero_rate_contract() {
        let emission = PoissonEmission::new(2.0).expect("valid rate");
        let got = emission.log_prob(&3);
        let expected = 3.0 * 2.0_f64.ln() - 2.0 - 6.0_f64.ln();
        // Measured |error| = 2.22e-16 on 2026-09-02; tol is 4x.
        assert!((got - expected).abs() <= 4.0 * f64::EPSILON);

        let point_mass = PoissonEmission::new(0.0).expect("zero is valid");
        assert_eq!(point_mass.log_prob(&0), 0.0);
        assert_eq!(point_mass.log_prob(&1), f64::NEG_INFINITY);
    }

    #[test]
    fn constructor_rejects_invalid_rate() {
        for (rate, code) in [
            (-1.0, IssueCode::InvalidParameter),
            (f64::NAN, IssueCode::NonFiniteInput),
            (f64::INFINITY, IssueCode::NonFiniteInput),
        ] {
            let failure = PoissonEmission::new(rate).expect_err("invalid rate");
            assert_eq!(failure.primary().code, code);
            assert_eq!(failure.report.algorithm, "PoissonEmission");
            assert_eq!(failure.report.operation, "new");
        }
    }

    #[test]
    fn observations_reject_wrong_shape_and_invalid_counts() {
        let wrong_shape = Matrix::zeros(2, 2);
        let failure = PoissonEmission::observations(&wrong_shape).unwrap_err();
        assert_eq!(failure.primary().code, IssueCode::DimensionMismatch);

        for value in [f64::NAN, -1.0, 1.5, u64::MAX as f64 + 1.0] {
            let x = Matrix::from_row_major(1, 1, &[value]);
            let failure = PoissonEmission::observations(&x).unwrap_err();
            assert!(matches!(
                failure.primary().code,
                IssueCode::NonFiniteInput | IssueCode::InvalidParameter
            ));
            assert_eq!(failure.report.operation, "observations");
        }

        let upper_exclusive = u64::MAX as f64 + 1.0;
        let largest_representable = upper_exclusive - 2048.0;
        let x = Matrix::from_row_major(1, 1, &[largest_representable]);
        let observations = PoissonEmission::observations(&x).expect("in-range integer");
        assert_eq!(observations[0] as f64, largest_representable);
    }

    #[test]
    fn m_step_uses_weighted_mean_without_floor() {
        let mut emission = PoissonEmission::new(7.0).expect("valid rate");
        let mut stats = PoissonStats::default();
        emission.accumulate(&1, 1.0, &mut stats);
        emission.accumulate(&3, 3.0, &mut stats);
        let mut ctx = FitCtx::new("PoissonEmission", "maximize");
        emission
            .maximize(&stats, &mut ctx)
            .expect("identified M-step");
        assert_eq!(emission.rate(), 2.5);

        let mut zero_stats = PoissonStats::default();
        emission.accumulate(&0, 2.0, &mut zero_stats);
        emission
            .maximize(&zero_stats, &mut ctx)
            .expect("zero-rate MLE is valid");
        assert_eq!(emission.rate(), 0.0);
    }

    #[test]
    fn zero_occupancy_warns_and_preserves_rate() {
        let mut emission = PoissonEmission::new(3.0).expect("valid rate");
        let mut ctx = FitCtx::new("PoissonEmission", "maximize");
        emission
            .maximize(&PoissonStats::default(), &mut ctx)
            .expect("warning is non-fatal");
        assert_eq!(emission.rate(), 3.0);
        assert!(ctx.report.contains(IssueCode::UnreachableState));
        assert!(ctx.report.has_warning());
    }
}
