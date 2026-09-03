use super::model::LinearGaussianStateSpace;
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{
    chol_solve_matrix_with_context, matrix_add, matrix_is_finite, matrix_multiply, matrix_subtract,
    matrix_symmetrized, matrix_transpose, matrix_write_row, vector_is_finite,
};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Qualified, Result, Severity};

/// Complete output of a time-invariant Kalman filter pass.
#[derive(Clone, Debug)]
pub struct KalmanFilterResult {
    /// Prior state mean at each observation time (`time × state`).
    pub predicted_mean: Matrix,
    /// Posterior state mean after each observation update (`time × state`).
    pub filtered_mean: Matrix,
    /// Prior state covariance at each observation time.
    pub predicted_covariance: Vec<Matrix>,
    /// Posterior state covariance after each observation update.
    pub filtered_covariance: Vec<Matrix>,
    /// Observed measurement-column indices at each time.
    pub observed_columns: Vec<Vec<usize>>,
    /// Compact innovations in the order given by [`Self::observed_columns`].
    pub innovations: Vec<Vector>,
    /// Compact innovation covariance for each time.
    pub innovation_covariances: Vec<Matrix>,
    /// Exact Gaussian log-likelihood contribution at each time.
    pub log_likelihood_contributions: Vector,
    /// Sum of [`Self::log_likelihood_contributions`].
    pub log_likelihood: f64,
}

impl LinearGaussianStateSpace {
    /// Run the Kalman filter.
    ///
    /// `observations` is `time × observation_dim`.  `observed_mask`, when
    /// present, is row-major with the same element count.  Masked cells still
    /// require a finite placeholder; their numerical value is never read.
    pub fn filter(
        &self,
        observations: &Matrix,
        observed_mask: Option<&[bool]>,
        session: &Session,
    ) -> Result<Qualified<KalmanFilterResult>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.policy = self.policy.clone();
        ctx.report
            .set_sample_shape(observations.nrows(), observations.ncols());
        let Some(value) = self.filter_into(observations, observed_mask, &mut ctx) else {
            return Err(ctx.finish_failure());
        };
        ctx.finish(value)
    }

    pub(super) fn filter_into(
        &self,
        observations: &Matrix,
        observed_mask: Option<&[bool]>,
        ctx: &mut FitCtx,
    ) -> Option<KalmanFilterResult> {
        let time_count = observations.nrows();
        let observation_dim = self.observation.nrows();
        let state_dim = self.transition.nrows();
        if time_count == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("state-space filtering requires at least one time point")
                    .build(),
            );
        }
        if observations.ncols() != observation_dim {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!(
                        "observations have {} columns; model expects {observation_dim}",
                        observations.ncols()
                    ))
                    .build(),
            );
        }
        if let Some(mask) = observed_mask {
            let expected = time_count.saturating_mul(observation_dim);
            if mask.len() != expected {
                ctx.push(
                    Issue::builder(IssueCode::DimensionMismatch)
                        .message(format!(
                            "observed mask has length {}; expected time*observation_dim={expected}",
                            mask.len()
                        ))
                        .build(),
                );
            }
        }
        let observations_are_finite = (0..observations.nrows())
            .all(|i| (0..observations.ncols()).all(|j| observations.get(i, j).is_finite()));
        if !observations_are_finite {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteInput)
                    .message(
                        "observations contain NaN or infinity; represent missingness only with observed_mask",
                    )
                    .build(),
            );
        }
        if ctx.report.has_error() {
            return None;
        }

        let is_observed = |time: usize, column: usize| {
            observed_mask
                .map(|mask| mask[time * observation_dim + column])
                .unwrap_or(true)
        };
        let total_observed = (0..time_count)
            .map(|time| {
                (0..observation_dim)
                    .filter(|column| is_observed(time, *column))
                    .count()
            })
            .sum::<usize>();
        if total_observed == 0 {
            ctx.push(
                Issue::builder(IssueCode::AllMissing)
                    .severity(Severity::Advisory)
                    .message(
                        "every observation is masked; returning prior propagation with zero log-likelihood",
                    )
                    .build(),
            );
        }

        let mut predicted_mean = Matrix::zeros(time_count, state_dim);
        let mut filtered_mean = Matrix::zeros(time_count, state_dim);
        let mut predicted_covariance = Vec::with_capacity(time_count);
        let mut filtered_covariance = Vec::with_capacity(time_count);
        let mut observed_columns = Vec::with_capacity(time_count);
        let mut innovations = Vec::with_capacity(time_count);
        let mut innovation_covariances = Vec::with_capacity(time_count);
        let mut log_likelihood_contributions = Vector::zeros(time_count);
        let mut prior_mean = self.initial_mean.clone();
        let mut prior_covariance = self.initial_covariance.clone();
        let mut log_likelihood = 0.0;

        for time in 0..time_count {
            if time > 0 {
                let previous_mean = filtered_mean.row(time - 1);
                prior_mean = self
                    .transition
                    .matvec(&previous_mean)
                    .add(&self.transition_offset);
                prior_covariance = matrix_symmetrized(&matrix_add(
                    &matrix_multiply(
                        &matrix_multiply(&self.transition, &filtered_covariance[time - 1]),
                        &matrix_transpose(&self.transition),
                    ),
                    &self.process_covariance,
                ));
            }
            if !vector_is_finite(&prior_mean) || !matrix_is_finite(&prior_covariance) {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message(format!(
                            "Kalman state prediction is non-finite at time {time}; state_dim={state_dim}"
                        ))
                        .build(),
                );
                return None;
            }
            matrix_write_row(&mut predicted_mean, time, &prior_mean);
            predicted_covariance.push(prior_covariance.clone());

            let columns = (0..observation_dim)
                .filter(|column| is_observed(time, *column))
                .collect::<Vec<_>>();
            if columns.is_empty() {
                matrix_write_row(&mut filtered_mean, time, &prior_mean);
                filtered_covariance.push(prior_covariance.clone());
                observed_columns.push(columns);
                innovations.push(Vector::zeros(0));
                innovation_covariances.push(Matrix::zeros(0, 0));
                continue;
            }

            let observed =
                Vector::from_iter(columns.iter().map(|column| observations.get(time, *column)));
            let design = select_rows(&self.observation, &columns);
            let noise = principal_submatrix(&self.observation_covariance, &columns);
            let predicted_observation = self
                .observation
                .matvec(&prior_mean)
                .add(&self.observation_offset);
            let selected_prediction =
                Vector::from_iter(columns.iter().map(|column| predicted_observation[*column]));
            if !vector_is_finite(&selected_prediction) {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message(format!(
                            "Kalman observation prediction is non-finite at time {time}; observation_dim={observation_dim}, observed_columns={columns:?}"
                        ))
                        .build(),
                );
                return None;
            }
            let innovation = observed.sub(&selected_prediction);
            let design_prior = matrix_multiply(&design, &prior_covariance);
            let innovation_covariance = matrix_symmetrized(&matrix_add(
                &matrix_multiply(&design_prior, &matrix_transpose(&design)),
                &noise,
            ));

            let mut right_hand_sides = Matrix::zeros(columns.len(), state_dim + 1);
            for i in 0..columns.len() {
                right_hand_sides.set(i, 0, innovation[i]);
                for j in 0..state_dim {
                    right_hand_sides.set(i, j + 1, design_prior.get(i, j));
                }
            }
            let solve_context = format!(
                "Kalman innovation covariance at time {time}; state_dim={state_dim}, innovation_dim={}, observed_columns={columns:?}",
                columns.len()
            );
            let solved = chol_solve_matrix_with_context(
                &mut ctx.report,
                innovation_covariance.inner(),
                &right_hand_sides,
                &ctx.policy,
                &solve_context,
            )?;
            let inverse_innovation = solved.solution.column(0);
            let mut kalman_gain = Matrix::zeros(state_dim, columns.len());
            for i in 0..state_dim {
                for j in 0..columns.len() {
                    kalman_gain.set(i, j, solved.solution.get(j, i + 1));
                }
            }
            let posterior_mean = prior_mean.add(&kalman_gain.matvec(&innovation));
            let identity =
                Matrix::from_fn(state_dim, state_dim, |i, j| if i == j { 1.0 } else { 0.0 });
            let identity_minus_gain_design =
                matrix_subtract(&identity, &matrix_multiply(&kalman_gain, &design));
            let posterior_covariance = matrix_symmetrized(&matrix_add(
                &matrix_multiply(
                    &matrix_multiply(&identity_minus_gain_design, &prior_covariance),
                    &matrix_transpose(&identity_minus_gain_design),
                ),
                &matrix_multiply(
                    &matrix_multiply(&kalman_gain, &noise),
                    &matrix_transpose(&kalman_gain),
                ),
            ));
            let quadratic = innovation.dot(&inverse_innovation);
            if !quadratic.is_finite() || quadratic < 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::Indefinite)
                        .message(format!(
                            "innovation quadratic form at time {time} is {quadratic}"
                        ))
                        .build(),
                );
                return None;
            }
            let contribution = -0.5
                * (columns.len() as f64 * std::f64::consts::TAU.ln()
                    + solved.log_determinant
                    + quadratic);
            if !contribution.is_finite()
                || !vector_is_finite(&posterior_mean)
                || !matrix_is_finite(&posterior_covariance)
            {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message(format!(
                            "Kalman update produced non-finite output at time {time}"
                        ))
                        .build(),
                );
                return None;
            }

            let accumulated_log_likelihood = log_likelihood + contribution;
            if !accumulated_log_likelihood.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message(format!(
                            "Kalman total log-likelihood became {accumulated_log_likelihood} after time {time}; innovation_dim={}, observed_columns={columns:?}",
                            columns.len()
                        ))
                        .build(),
                );
                return None;
            }
            matrix_write_row(&mut filtered_mean, time, &posterior_mean);
            filtered_covariance.push(posterior_covariance);
            observed_columns.push(columns);
            innovations.push(innovation);
            innovation_covariances.push(innovation_covariance);
            log_likelihood_contributions[time] = contribution;
            log_likelihood = accumulated_log_likelihood;
        }

        Some(KalmanFilterResult {
            predicted_mean,
            filtered_mean,
            predicted_covariance,
            filtered_covariance,
            observed_columns,
            innovations,
            innovation_covariances,
            log_likelihood_contributions,
            log_likelihood,
        })
    }
}

fn select_rows(matrix: &Matrix, rows: &[usize]) -> Matrix {
    Matrix::from_fn(rows.len(), matrix.ncols(), |i, j| matrix.get(rows[i], j))
}

fn principal_submatrix(matrix: &Matrix, indices: &[usize]) -> Matrix {
    Matrix::from_fn(indices.len(), indices.len(), |i, j| {
        matrix.get(indices[i], indices[j])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use signlred::{Policy, Severity};

    fn scalar_model(
        transition: f64,
        observation: f64,
        process_variance: f64,
        observation_variance: f64,
        initial_mean: f64,
        initial_variance: f64,
        policy: Policy,
    ) -> LinearGaussianStateSpace {
        LinearGaussianStateSpace::new(
            Matrix::from_row_major(1, 1, &[transition]),
            Matrix::from_row_major(1, 1, &[observation]),
            Matrix::from_row_major(1, 1, &[process_variance]),
            Matrix::from_row_major(1, 1, &[observation_variance]),
            Vector::from_slice(&[0.0]),
            Vector::from_slice(&[0.0]),
            Vector::from_slice(&[initial_mean]),
            Matrix::from_row_major(1, 1, &[initial_variance]),
            policy,
        )
        .expect("valid scalar model")
    }

    #[test]
    fn impossible_innovation_is_failure_under_relaxed_policy() {
        let policy = Policy {
            abort_at: Severity::Fatal,
            ..Policy::default()
        };
        let model = scalar_model(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, policy);
        let failure = model
            .filter(
                &Matrix::from_row_major(1, 1, &[1.0]),
                None,
                &Session::new("state-space", "relaxed-singular-filter"),
            )
            .expect_err("a missing Cholesky factor cannot produce a filter result");
        assert_eq!(failure.primary.code, IssueCode::CholeskyFailed);
        assert!(failure.primary.message.contains("time 0"));
        assert!(failure.primary.message.contains("observed_columns=[0]"));
    }

    #[test]
    fn nonfinite_prediction_is_checked_before_an_all_missing_update() {
        let policy = Policy {
            abort_at: Severity::Fatal,
            ..Policy::default()
        };
        let model = scalar_model(f64::MAX, 1.0, 0.0, 1.0, 2.0, 1.0, policy);
        let failure = model
            .filter(
                &Matrix::from_row_major(2, 1, &[0.0, 0.0]),
                Some(&[false, false]),
                &Session::new("state-space", "nonfinite-missing-prediction"),
            )
            .expect_err("overflowed prior propagation cannot be returned as a value");
        assert_eq!(failure.primary.code, IssueCode::NonFiniteOutput);
        assert!(failure.primary.message.contains("time 1"));
    }

    #[test]
    fn nonfinite_accumulated_log_likelihood_is_failure() {
        let policy = Policy {
            abort_at: Severity::Fatal,
            ..Policy::default()
        };
        let model = scalar_model(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, policy);
        let observations = Matrix::from_row_major(4, 1, &[1e154, 1e154, 1e154, 1e154]);
        let failure = model
            .filter(
                &observations,
                None,
                &Session::new("state-space", "log-likelihood-overflow"),
            )
            .expect_err("an infinite accumulated likelihood cannot be returned");
        assert_eq!(failure.primary.code, IssueCode::NonFiniteOutput);
        assert!(failure.primary.message.contains("total log-likelihood"));
    }

    #[test]
    fn maximum_finite_scalar_covariance_does_not_overflow_during_symmetrization() {
        let model = scalar_model(1.0, 0.0, 0.0, 1.0, 0.0, f64::MAX, Policy::default());
        let result = model
            .filter(
                &Matrix::from_row_major(1, 1, &[0.0]),
                None,
                &Session::new("state-space", "maximum-finite-covariance"),
            )
            .expect("a representable finite innovation covariance must remain finite")
            .value;

        // A zero observation loading makes the measurement independent of
        // the state, so the exact posterior covariance remains f64::MAX.
        assert_eq!(result.innovation_covariances[0].get(0, 0), 1.0);
        assert_eq!(result.filtered_covariance[0].get(0, 0), f64::MAX);
        assert!(result.log_likelihood.is_finite());
    }
}
