//! Diagonal Gaussian emission and weighted-Welford sufficient statistics.

use super::Emission;
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use signlred::{Failure, Issue, IssueCode, Result, Severity};

/// A multivariate Gaussian emission with one variance per observation dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianEmission {
    mean: Vector,
    variance: Vector,
}

/// Weighted-Welford sufficient statistics for a diagonal Gaussian M-step.
///
/// Create an empty accumulator with [`Default`] and populate it through
/// [`Emission::accumulate`]; its numerical fields are intentionally private.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GaussianStats {
    weight: f64,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl GaussianEmission {
    /// Construct a diagonal Gaussian from its mean and variance vectors.
    ///
    /// Both vectors must have the same non-zero length, every value must be
    /// finite, and every variance must be strictly positive.
    pub fn new(mean: Vector, variance: Vector) -> Result<Self> {
        if mean.is_empty() || mean.len() != variance.len() {
            return Err(failure(
                "new",
                IssueCode::DimensionMismatch,
                format!(
                    "GaussianEmission mean dimension {} and variance dimension {} must be equal and non-zero",
                    mean.len(),
                    variance.len()
                ),
            ));
        }
        if mean.as_slice().iter().any(|value| !value.is_finite())
            || variance.as_slice().iter().any(|value| !value.is_finite())
        {
            return Err(failure(
                "new",
                IssueCode::NonFiniteInput,
                "GaussianEmission parameters must be finite",
            ));
        }
        if variance.as_slice().iter().any(|value| *value <= 0.0) {
            return Err(failure(
                "new",
                IssueCode::InvalidParameter,
                "GaussianEmission variances must be strictly positive",
            ));
        }
        Ok(Self { mean, variance })
    }

    /// Borrow the mean vector, in observation-dimension order.
    pub fn mean(&self) -> &Vector {
        &self.mean
    }

    /// Borrow the strictly positive diagonal variance vector.
    pub fn variance(&self) -> &Vector {
        &self.variance
    }

    fn poison(stats: &mut GaussianStats, dimension: usize) {
        stats.weight = f64::NAN;
        stats.mean = vec![f64::NAN; dimension];
        stats.m2 = vec![f64::NAN; dimension];
    }
}

impl Emission for GaussianEmission {
    type Observation = Vec<f64>;
    type SufficientStats = GaussianStats;

    fn observations(x: &Matrix) -> Result<Vec<Self::Observation>> {
        if x.ncols() == 0 {
            return Err(failure(
                "observations",
                IssueCode::DimensionMismatch,
                "GaussianEmission observations require at least one column",
            ));
        }
        let mut observations = Vec::with_capacity(x.nrows());
        for row in 0..x.nrows() {
            let mut observation = Vec::with_capacity(x.ncols());
            for column in 0..x.ncols() {
                let value = x.get(row, column);
                if !value.is_finite() {
                    return Err(failure(
                        "observations",
                        IssueCode::NonFiniteInput,
                        format!(
                            "GaussianEmission observation at row {row}, column {column} is not finite"
                        ),
                    ));
                }
                observation.push(value);
            }
            observations.push(observation);
        }
        Ok(observations)
    }

    fn log_prob(&self, obs: &Self::Observation) -> f64 {
        if obs.len() != self.mean.len() || obs.iter().any(|value| !value.is_finite()) {
            return f64::NEG_INFINITY;
        }
        let mut quadratic_and_log_det = 0.0;
        for (dimension, value) in obs.iter().enumerate() {
            let variance = self.variance[dimension];
            let residual = *value - self.mean[dimension];
            quadratic_and_log_det += variance.ln() + residual * residual / variance;
        }
        -0.5 * (self.mean.len() as f64 * std::f64::consts::TAU.ln() + quadratic_and_log_det)
    }

    fn accumulate(&self, obs: &Self::Observation, weight: f64, stats: &mut Self::SufficientStats) {
        if weight == 0.0 {
            return;
        }
        let dimension = self.mean.len();
        if !weight.is_finite()
            || weight < 0.0
            || obs.len() != dimension
            || obs.iter().any(|value| !value.is_finite())
        {
            Self::poison(stats, dimension);
            return;
        }
        if stats.weight == 0.0 && stats.mean.is_empty() && stats.m2.is_empty() {
            stats.weight = weight;
            stats.mean = obs.clone();
            stats.m2 = vec![0.0; dimension];
            return;
        }
        if !stats.weight.is_finite()
            || stats.weight < 0.0
            || stats.mean.len() != dimension
            || stats.m2.len() != dimension
        {
            Self::poison(stats, dimension);
            return;
        }
        let total_weight = stats.weight + weight;
        if !total_weight.is_finite() || total_weight <= 0.0 {
            Self::poison(stats, dimension);
            return;
        }
        for index in 0..dimension {
            let delta = obs[index] - stats.mean[index];
            let next_mean = stats.mean[index] + delta * weight / total_weight;
            stats.m2[index] += weight * delta * (obs[index] - next_mean);
            stats.mean[index] = next_mean;
        }
        stats.weight = total_weight;
    }

    fn maximize(&mut self, stats: &Self::SufficientStats, ctx: &mut FitCtx) -> Result<()> {
        if !ctx.policy.near_zero_variance.is_finite() || ctx.policy.near_zero_variance < 0.0 {
            return Err(failure(
                "maximize",
                IssueCode::InvalidParameter,
                format!(
                    "GaussianEmission maximize requires a finite, non-negative near_zero_variance; got {}",
                    ctx.policy.near_zero_variance
                ),
            ));
        }
        if stats.weight == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnreachableState)
                    .severity(Severity::Warning)
                    .message(
                        "GaussianEmission maximize received zero posterior occupancy; parameters are unchanged",
                    )
                    .build(),
            );
            return Ok(());
        }
        if !stats.weight.is_finite() || stats.weight < 0.0 {
            return Err(failure(
                "maximize",
                IssueCode::NonFiniteInput,
                "GaussianEmission sufficient-statistic weight is invalid",
            ));
        }
        let dimension = self.mean.len();
        if stats.mean.len() != dimension || stats.m2.len() != dimension {
            return Err(failure(
                "maximize",
                IssueCode::DimensionMismatch,
                "GaussianEmission sufficient-statistic dimension differs from the emission",
            ));
        }
        if stats
            .mean
            .iter()
            .chain(stats.m2.iter())
            .any(|value| !value.is_finite())
        {
            return Err(failure(
                "maximize",
                IssueCode::NonFiniteInput,
                "GaussianEmission sufficient statistics must be finite",
            ));
        }
        let variances: Vec<f64> = stats.m2.iter().map(|m2| *m2 / stats.weight).collect();
        if variances.iter().any(|variance| !variance.is_finite()) {
            return Err(failure(
                "maximize",
                IssueCode::NonFiniteOutput,
                "GaussianEmission MLE variance is not finite",
            ));
        }
        if variances
            .iter()
            .any(|variance| *variance <= ctx.policy.near_zero_variance)
        {
            return Err(failure(
                "maximize",
                IssueCode::EmissionDegenerate,
                format!(
                    "GaussianEmission MLE variance is at or below near_zero_variance={} and cannot be floored",
                    ctx.policy.near_zero_variance
                ),
            ));
        }
        self.mean = Vector::from_slice(&stats.mean);
        self.variance = Vector::from_slice(&variances);
        Ok(())
    }
}

fn failure(operation: &'static str, code: IssueCode, message: impl Into<String>) -> Failure {
    Failure::from_issue(
        "GaussianEmission",
        operation,
        Issue::builder(code).message(message).build(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_prob_matches_standard_normal_closed_form() {
        let emission =
            GaussianEmission::new(Vector::from_slice(&[0.0]), Vector::from_slice(&[1.0]))
                .expect("valid Gaussian");
        let got = emission.log_prob(&vec![0.0]);
        let expected = -0.5 * std::f64::consts::TAU.ln();
        // Measured |error| = 0.0 on 2026-09-02; allow two ulps.
        assert!((got - expected).abs() <= 2.0 * f64::EPSILON);
    }

    #[test]
    fn constructor_and_observations_reject_invalid_input() {
        let mismatch =
            GaussianEmission::new(Vector::from_slice(&[0.0]), Vector::from_slice(&[1.0, 1.0]))
                .unwrap_err();
        assert_eq!(mismatch.primary().code, IssueCode::DimensionMismatch);
        assert_eq!(mismatch.report.algorithm, "GaussianEmission");
        assert_eq!(mismatch.report.operation, "new");

        let non_positive =
            GaussianEmission::new(Vector::from_slice(&[0.0]), Vector::from_slice(&[0.0]))
                .unwrap_err();
        assert_eq!(non_positive.primary().code, IssueCode::InvalidParameter);

        let non_finite = Matrix::from_row_major(1, 1, &[f64::NAN]);
        let failure = GaussianEmission::observations(&non_finite).unwrap_err();
        assert_eq!(failure.primary().code, IssueCode::NonFiniteInput);
        assert_eq!(failure.report.operation, "observations");
    }

    #[test]
    fn weighted_welford_m_step_matches_closed_form() {
        let mut emission =
            GaussianEmission::new(Vector::from_slice(&[0.0]), Vector::from_slice(&[1.0]))
                .expect("valid Gaussian");
        let mut stats = GaussianStats::default();
        emission.accumulate(&vec![1.0], 1.0, &mut stats);
        emission.accumulate(&vec![3.0], 3.0, &mut stats);
        assert_eq!(stats.weight, 4.0);
        assert_eq!(stats.mean, vec![2.5]);
        assert_eq!(stats.m2, vec![3.0]);

        let mut ctx = FitCtx::new("GaussianEmission", "maximize");
        emission
            .maximize(&stats, &mut ctx)
            .expect("identified M-step");
        assert_eq!(emission.mean().as_slice(), &[2.5]);
        assert_eq!(emission.variance().as_slice(), &[0.75]);
    }

    #[test]
    fn degenerate_m_step_fails_without_changing_parameters() {
        let mut emission =
            GaussianEmission::new(Vector::from_slice(&[10.0]), Vector::from_slice(&[2.0]))
                .expect("valid Gaussian");
        let before = emission.clone();
        let mut stats = GaussianStats::default();
        emission.accumulate(&vec![4.0], 1.0, &mut stats);
        emission.accumulate(&vec![4.0], 1.0, &mut stats);
        let mut ctx = FitCtx::new("GaussianEmission", "maximize");
        let failure = emission.maximize(&stats, &mut ctx).unwrap_err();
        assert_eq!(failure.primary().code, IssueCode::EmissionDegenerate);
        assert_eq!(failure.report.operation, "maximize");
        assert_eq!(emission, before);
    }

    #[test]
    fn invalid_variance_policy_is_rejected() {
        let mut emission =
            GaussianEmission::new(Vector::from_slice(&[0.0]), Vector::from_slice(&[1.0]))
                .expect("valid Gaussian");
        let before = emission.clone();
        let mut stats = GaussianStats::default();
        emission.accumulate(&vec![-1.0], 1.0, &mut stats);
        emission.accumulate(&vec![1.0], 1.0, &mut stats);

        for invalid in [f64::NAN, -1.0] {
            let mut ctx = FitCtx::new("GaussianEmission", "maximize");
            ctx.policy.near_zero_variance = invalid;
            let failure = emission.maximize(&stats, &mut ctx).unwrap_err();
            assert_eq!(failure.primary().code, IssueCode::InvalidParameter);
            assert_eq!(failure.report.operation, "maximize");
            assert_eq!(emission, before);
        }
    }

    #[test]
    fn zero_occupancy_warns_and_preserves_parameters() {
        let mut emission =
            GaussianEmission::new(Vector::from_slice(&[2.0]), Vector::from_slice(&[3.0]))
                .expect("valid Gaussian");
        let before = emission.clone();
        let mut ctx = FitCtx::new("GaussianEmission", "maximize");
        emission
            .maximize(&GaussianStats::default(), &mut ctx)
            .expect("warning is non-fatal");
        assert_eq!(emission, before);
        assert!(ctx.report.contains(IssueCode::UnreachableState));
        assert!(ctx.report.has_warning());
    }
}
