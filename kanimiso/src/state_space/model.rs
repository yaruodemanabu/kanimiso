use crate::data::{Matrix, Vector};
use crate::linalg::symmetric_eigen;
use signlred::{Failure, Issue, IssueCode, Policy, Report, Result};

/// Known-parameter, time-invariant linear Gaussian state-space model.
///
/// The timing convention is
/// `x_0 ~ N(initial_mean, initial_covariance)`,
/// `x_t = transition * x_(t-1) + transition_offset + eta_t` for `t >= 1`,
/// and `y_t = observation * x_t + observation_offset + epsilon_t`.  Process
/// and observation noises are independent with the supplied covariance
/// matrices.
#[derive(Clone, Debug)]
pub struct LinearGaussianStateSpace {
    pub(super) transition: Matrix,
    pub(super) observation: Matrix,
    pub(super) process_covariance: Matrix,
    pub(super) observation_covariance: Matrix,
    pub(super) transition_offset: Vector,
    pub(super) observation_offset: Vector,
    pub(super) initial_mean: Vector,
    pub(super) initial_covariance: Matrix,
    pub(super) policy: Policy,
}

impl LinearGaussianStateSpace {
    /// Construct and validate a dense state-space model.
    ///
    /// Covariances may be positive semidefinite.  No jitter, ridge, floor, or
    /// pseudoinverse is inserted: an innovation or smoother system that is not
    /// positive definite fails at the time where it is needed.
    pub fn new(
        transition: Matrix,
        observation: Matrix,
        process_covariance: Matrix,
        observation_covariance: Matrix,
        transition_offset: Vector,
        observation_offset: Vector,
        initial_mean: Vector,
        initial_covariance: Matrix,
        policy: Policy,
    ) -> Result<Self> {
        let mut report = Report::new("linear_gaussian_state_space", "new");
        let state_dim = transition.nrows();
        let observation_dim = observation.nrows();
        report.set_sample_shape(state_dim, observation_dim);

        if state_dim == 0 || observation_dim == 0 {
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("state and observation dimensions must both be nonzero")
                    .build(),
            );
        }
        let shapes_match = transition.ncols() == state_dim
            && observation.ncols() == state_dim
            && process_covariance.shape() == (state_dim, state_dim)
            && observation_covariance.shape() == (observation_dim, observation_dim)
            && transition_offset.len() == state_dim
            && observation_offset.len() == observation_dim
            && initial_mean.len() == state_dim
            && initial_covariance.shape() == (state_dim, state_dim);
        if !shapes_match {
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!(
                        "expected T={state_dim}x{state_dim}, Z={observation_dim}x{state_dim}, Q={state_dim}x{state_dim}, H={observation_dim}x{observation_dim}, c.len={state_dim}, d.len={observation_dim}, a0.len={state_dim}, P0={state_dim}x{state_dim}; got T={}x{}, Z={}x{}, Q={}x{}, H={}x{}, c.len={}, d.len={}, a0.len={}, P0={}x{}",
                        transition.nrows(),
                        transition.ncols(),
                        observation.nrows(),
                        observation.ncols(),
                        process_covariance.nrows(),
                        process_covariance.ncols(),
                        observation_covariance.nrows(),
                        observation_covariance.ncols(),
                        transition_offset.len(),
                        observation_offset.len(),
                        initial_mean.len(),
                        initial_covariance.nrows(),
                        initial_covariance.ncols()
                    ))
                    .build(),
            );
        }
        if state_dim == 0 || observation_dim == 0 || !shapes_match {
            return Err(unconditional_failure(report));
        }

        validate_policy(&mut report, &policy);
        if !report.issues().is_empty() {
            return Err(unconditional_failure(report));
        }

        validate_finite_matrix(&mut report, &policy, "transition", &transition);
        validate_finite_matrix(&mut report, &policy, "observation", &observation);
        validate_finite_vector(
            &mut report,
            &policy,
            "transition offset",
            &transition_offset,
        );
        validate_finite_vector(
            &mut report,
            &policy,
            "observation offset",
            &observation_offset,
        );
        validate_finite_matrix(
            &mut report,
            &policy,
            "process covariance",
            &process_covariance,
        );
        validate_finite_matrix(
            &mut report,
            &policy,
            "observation covariance",
            &observation_covariance,
        );
        validate_finite_matrix(
            &mut report,
            &policy,
            "initial covariance",
            &initial_covariance,
        );
        validate_finite_vector(&mut report, &policy, "initial mean", &initial_mean);

        if !report.issues().is_empty() {
            return Err(unconditional_failure(report));
        }

        validate_covariance(
            &mut report,
            &policy,
            "process covariance",
            &process_covariance,
        );
        validate_covariance(
            &mut report,
            &policy,
            "observation covariance",
            &observation_covariance,
        );
        validate_covariance(
            &mut report,
            &policy,
            "initial covariance",
            &initial_covariance,
        );

        if !report.issues().is_empty() {
            return Err(unconditional_failure(report));
        }

        Ok(Self {
            transition,
            observation,
            process_covariance,
            observation_covariance,
            transition_offset,
            observation_offset,
            initial_mean,
            initial_covariance,
            policy,
        })
    }
}

fn unconditional_failure(report: Report) -> Failure {
    let primary = report
        .primary()
        .cloned()
        .expect("model validation failure must contain an issue");
    Failure { primary, report }
}

fn validate_policy(report: &mut Report, policy: &Policy) {
    if !policy.residual_tol.is_finite() || policy.residual_tol <= 0.0 {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::InvalidParameter)
                .message(format!(
                    "residual_tol={} must be finite and positive",
                    policy.residual_tol
                ))
                .build(),
        );
    }
    let warn_is_valid =
        policy.condition_number_warn.is_finite() && policy.condition_number_warn >= 1.0;
    let error_is_valid =
        policy.condition_number_error.is_finite() && policy.condition_number_error >= 1.0;
    if !warn_is_valid {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::InvalidParameter)
                .message(format!(
                    "condition_number_warn={} must be finite and at least one",
                    policy.condition_number_warn
                ))
                .build(),
        );
    }
    if !error_is_valid {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::InvalidParameter)
                .message(format!(
                    "condition_number_error={} must be finite and at least one",
                    policy.condition_number_error
                ))
                .build(),
        );
    }
    if warn_is_valid
        && error_is_valid
        && policy.condition_number_warn > policy.condition_number_error
    {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::InvalidParameter)
                .message(format!(
                    "condition_number_warn={} must not exceed condition_number_error={}",
                    policy.condition_number_warn, policy.condition_number_error
                ))
                .build(),
        );
    }
}

fn validate_finite_vector(report: &mut Report, policy: &Policy, name: &str, vector: &Vector) {
    if !vector.as_slice().iter().all(|value| value.is_finite()) {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::NonFiniteInput)
                .message(format!("{name} contains NaN or infinity"))
                .build(),
        );
    }
}

fn validate_finite_matrix(report: &mut Report, policy: &Policy, name: &str, matrix: &Matrix) {
    let is_finite =
        (0..matrix.nrows()).all(|i| (0..matrix.ncols()).all(|j| matrix.get(i, j).is_finite()));
    if !is_finite {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::NonFiniteInput)
                .message(format!("{name} contains NaN or infinity"))
                .build(),
        );
    }
}

fn validate_covariance(report: &mut Report, policy: &Policy, name: &str, matrix: &Matrix) {
    let scale = matrix_max_abs(matrix);
    let mut asymmetry = 0.0_f64;
    for i in 0..matrix.nrows() {
        for j in 0..i {
            asymmetry = asymmetry.max((matrix.get(i, j) - matrix.get(j, i)).abs());
        }
    }
    if asymmetry != 0.0 {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::NonPositiveDefinite)
                .message(format!(
                    "{name} is not exactly symmetric: maximum asymmetry is {asymmetry:.4e}"
                ))
                .metric("maximum_asymmetry", asymmetry)
                .metric("covariance_scale", scale)
                .build(),
        );
        return;
    }
    let minimum_variance = (0..matrix.nrows())
        .map(|i| matrix.get(i, i))
        .fold(f64::INFINITY, f64::min);
    if minimum_variance < 0.0 {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::NonPositiveDefinite)
                .message(format!(
                    "{name} has negative diagonal variance {minimum_variance:.4e}"
                ))
                .metric("minimum_variance", minimum_variance)
                .metric("covariance_scale", scale)
                .build(),
        );
        return;
    }
    let Some((eigenvalues, _)) = symmetric_eigen(report, matrix.inner(), policy) else {
        return;
    };
    let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    if minimum < 0.0 {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::NonPositiveDefinite)
                .message(format!(
                    "{name} has negative minimum eigenvalue {minimum:.4e}"
                ))
                .metric("minimum_eigenvalue", minimum)
                .metric("covariance_scale", scale)
                .build(),
        );
    }
}

fn matrix_max_abs(matrix: &Matrix) -> f64 {
    let mut maximum = 0.0_f64;
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            maximum = maximum.max(matrix.get(i, j).abs());
        }
    }
    maximum
}

#[cfg(test)]
mod tests {
    use super::*;
    use signlred::Severity;

    fn valid_model(policy: Policy) -> Result<LinearGaussianStateSpace> {
        LinearGaussianStateSpace::new(
            Matrix::from_row_major(2, 2, &[1.0, 0.1, 0.0, 1.0]),
            Matrix::from_row_major(1, 2, &[1.0, 0.0]),
            Matrix::from_row_major(2, 2, &[0.2, 0.0, 0.0, 0.1]),
            Matrix::from_row_major(1, 1, &[0.3]),
            Vector::zeros(2),
            Vector::zeros(1),
            Vector::zeros(2),
            Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            policy,
        )
    }

    #[test]
    fn malformed_shape_is_unconditionally_rejected_before_covariance_indexing() {
        let policy = Policy {
            abort_at: Severity::Fatal,
            ..Policy::default()
        };
        let result = LinearGaussianStateSpace::new(
            Matrix::from_row_major(1, 2, &[1.0, 0.0]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Matrix::from_row_major(1, 2, &[1.0, 0.0]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Vector::zeros(1),
            Vector::zeros(1),
            Vector::zeros(1),
            Matrix::from_row_major(1, 2, &[1.0, 0.0]),
            policy,
        );
        let failure = result.expect_err("malformed shapes are model-invalid for every policy");
        assert_eq!(failure.primary.code, IssueCode::DimensionMismatch);
    }

    #[test]
    fn policy_fields_used_by_state_space_are_validated() {
        let invalid_policies = [
            Policy {
                residual_tol: f64::NAN,
                ..Policy::default()
            },
            Policy {
                condition_number_warn: f64::INFINITY,
                ..Policy::default()
            },
            Policy {
                condition_number_error: 10.0,
                condition_number_warn: 100.0,
                ..Policy::default()
            },
        ];
        for policy in invalid_policies {
            let failure = valid_model(policy).expect_err("invalid policy must reject construction");
            assert_eq!(failure.primary.code, IssueCode::InvalidParameter);
        }
    }

    #[test]
    fn covariance_requires_exact_symmetry_even_under_nonaborting_policy() {
        let policy = Policy {
            abort_at: Severity::Fatal,
            ..Policy::default()
        };
        let result = LinearGaussianStateSpace::new(
            Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            Matrix::from_row_major(1, 2, &[1.0, 0.0]),
            Matrix::from_row_major(2, 2, &[0.2, f64::EPSILON, 0.0, 0.1]),
            Matrix::from_row_major(1, 1, &[0.3]),
            Vector::zeros(2),
            Vector::zeros(1),
            Vector::zeros(2),
            Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            policy,
        );
        let failure = result.expect_err("nonzero covariance asymmetry must not be repaired");
        assert_eq!(failure.primary.code, IssueCode::NonPositiveDefinite);
    }

    #[test]
    fn every_negative_covariance_direction_is_rejected() {
        let negative_variance = LinearGaussianStateSpace::new(
            Matrix::from_row_major(1, 1, &[1.0]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Matrix::from_row_major(1, 1, &[-f64::MIN_POSITIVE]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Vector::zeros(1),
            Vector::zeros(1),
            Vector::zeros(1),
            Matrix::from_row_major(1, 1, &[1.0]),
            Policy::default(),
        );
        assert_eq!(
            negative_variance
                .expect_err("negative variance")
                .primary
                .code,
            IssueCode::NonPositiveDefinite
        );

        let indefinite = LinearGaussianStateSpace::new(
            Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            Matrix::from_row_major(1, 2, &[1.0, 0.0]),
            Matrix::from_row_major(2, 2, &[1.0, 2.0, 2.0, 1.0]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Vector::zeros(2),
            Vector::zeros(1),
            Vector::zeros(2),
            Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            Policy::default(),
        );
        assert_eq!(
            indefinite.expect_err("negative eigenvalue").primary.code,
            IssueCode::NonPositiveDefinite
        );
    }

    #[test]
    fn exactly_symmetric_semidefinite_covariances_are_accepted_unchanged() {
        let model = LinearGaussianStateSpace::new(
            Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            Matrix::from_row_major(1, 2, &[1.0, 0.0]),
            Matrix::zeros(2, 2),
            Matrix::zeros(1, 1),
            Vector::zeros(2),
            Vector::zeros(1),
            Vector::zeros(2),
            Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            Policy::default(),
        )
        .expect("positive-semidefinite covariances are valid model inputs");
        assert_eq!(model.process_covariance.to_row_major(), vec![0.0; 4]);
        assert_eq!(model.observation_covariance.to_row_major(), vec![0.0]);
    }
}
