use super::filter::KalmanFilterResult;
use super::model::LinearGaussianStateSpace;
use crate::context::FitCtx;
use crate::data::Matrix;
use crate::linalg::{
    chol_solve_matrix_with_context, matrix_add, matrix_is_finite, matrix_multiply, matrix_subtract,
    matrix_symmetrized, matrix_transpose, matrix_write_row, vector_is_finite,
};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Qualified, Result};

/// Kalman filter output plus Rauch--Tung--Striebel smoothed states.
#[derive(Clone, Debug)]
pub struct KalmanSmootherResult {
    /// Forward-filter result used by the backward pass.
    pub filter: KalmanFilterResult,
    /// State mean conditional on every observed measurement (`time × state`).
    pub smoothed_mean: Matrix,
    /// State covariance conditional on every observed measurement.
    pub smoothed_covariance: Vec<Matrix>,
}

impl LinearGaussianStateSpace {
    /// Run one Kalman filter followed by an RTS backward smoother.
    pub fn smooth(
        &self,
        observations: &Matrix,
        observed_mask: Option<&[bool]>,
        session: &Session,
    ) -> Result<Qualified<KalmanSmootherResult>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.policy = self.policy.clone();
        ctx.report
            .set_sample_shape(observations.nrows(), observations.ncols());
        let Some(filter) = self.filter_into(observations, observed_mask, &mut ctx) else {
            return Err(ctx.finish_failure());
        };
        let state_dim = self.transition.nrows();
        let mut smoothed_mean = filter.filtered_mean.clone();
        let mut smoothed_covariance = filter.filtered_covariance.clone();

        // A trailing run of missing measurements contains no information that
        // can revise an earlier filtered state.  Starting the RTS recursion at
        // the final observed time is also essential for valid deterministic
        // models: their later predicted covariance may be singular, but no
        // inverse is mathematically needed when there is no later evidence.
        let last_observed_time = filter
            .observed_columns
            .iter()
            .rposition(|columns| !columns.is_empty());
        if let Some(last_observed_time) = last_observed_time {
            for time in (0..last_observed_time).rev() {
                let filtered_covariance = &filter.filtered_covariance[time];
                let right_hand_side = matrix_multiply(&self.transition, filtered_covariance);
                if (0..right_hand_side.nrows())
                    .all(|i| (0..right_hand_side.ncols()).all(|j| right_hand_side.get(i, j) == 0.0))
                {
                    // Cov(x_t, x_{t+1} | y_0:t) is exactly zero.  In a
                    // Gaussian model the future is then independent of x_t,
                    // so the zero smoother gain is exact and no inverse of a
                    // possibly singular predicted covariance is required.
                    continue;
                }
                let observed_columns = &filter.observed_columns[time + 1];
                let solve_context = format!(
                    "RTS predicted-state covariance for transition {time}->{}; state_dim={state_dim}, next_observed_columns={observed_columns:?}",
                    time + 1
                );
                let Some(solved) = chol_solve_matrix_with_context(
                    &mut ctx.report,
                    filter.predicted_covariance[time + 1].inner(),
                    &right_hand_side,
                    &ctx.policy,
                    &solve_context,
                ) else {
                    return Err(ctx.finish_failure());
                };
                let smoother_gain = matrix_transpose(&solved.solution);
                let filtered_state = filter.filtered_mean.row(time);
                let next_smoothed = smoothed_mean.row(time + 1);
                let next_predicted = filter.predicted_mean.row(time + 1);
                let correction = next_smoothed.sub(&next_predicted);
                let current_smoothed = filtered_state.add(&smoother_gain.matvec(&correction));
                let covariance_correction = matrix_subtract(
                    &smoothed_covariance[time + 1],
                    &filter.predicted_covariance[time + 1],
                );
                let current_covariance = matrix_symmetrized(&matrix_add(
                    filtered_covariance,
                    &matrix_multiply(
                        &matrix_multiply(&smoother_gain, &covariance_correction),
                        &matrix_transpose(&smoother_gain),
                    ),
                ));
                if !vector_is_finite(&current_smoothed) || !matrix_is_finite(&current_covariance) {
                    ctx.push(
                        Issue::builder(IssueCode::NonFiniteOutput)
                            .message(format!(
                                "RTS smoother produced non-finite output at time {time}"
                            ))
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                }
                matrix_write_row(&mut smoothed_mean, time, &current_smoothed);
                smoothed_covariance[time] = current_covariance;
            }
        }

        ctx.finish(KalmanSmootherResult {
            filter,
            smoothed_mean,
            smoothed_covariance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Vector;
    use signlred::{IssueCode, Policy, Severity};

    #[test]
    fn impossible_backward_solve_is_failure_under_relaxed_policy() {
        let policy = Policy {
            abort_at: Severity::Fatal,
            ..Policy::default()
        };
        let model = LinearGaussianStateSpace::new(
            Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            Matrix::zeros(2, 2),
            Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            Vector::zeros(2),
            Vector::zeros(2),
            Vector::zeros(2),
            Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 0.0]),
            policy,
        )
        .expect("semidefinite process covariance is valid");
        let failure = model
            .smooth(
                &Matrix::zeros(2, 2),
                None,
                &Session::new("state-space", "relaxed-singular-smoother"),
            )
            .expect_err("a missing RTS factor cannot produce a smoother result");
        assert_eq!(failure.primary.code, IssueCode::CholeskyFailed);
        assert!(failure.primary.message.contains("RTS predicted-state"));
        assert!(failure.primary.message.contains("transition 0->1"));
        assert!(failure
            .primary
            .message
            .contains("next_observed_columns=[0, 1]"));
    }

    #[test]
    fn zero_cross_covariance_needs_no_inverse_even_with_future_observations() {
        let model = LinearGaussianStateSpace::new(
            Matrix::from_row_major(1, 1, &[0.0]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Matrix::from_row_major(1, 1, &[0.0]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Vector::from_slice(&[0.0]),
            Vector::from_slice(&[0.0]),
            Vector::from_slice(&[2.0]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Policy::default(),
        )
        .expect("semidefinite process covariance is valid");
        let result = model
            .smooth(
                &Matrix::from_row_major(2, 1, &[4.0, 100.0]),
                None,
                &Session::new("state-space", "zero-cross-covariance"),
            )
            .expect("zero cross-covariance has the exact zero smoother gain")
            .value;

        // At time zero, N(2, 1) conditioned on 4 = x_0 + N(0, 1) is
        // N(3, 1/2).  Since x_1 is deterministically zero, y_1 is independent
        // of x_0 and cannot revise that posterior despite being observed.
        assert_eq!(result.smoothed_mean.get(0, 0), 3.0);
        assert_eq!(result.smoothed_covariance[0].get(0, 0), 0.5);
        assert_eq!(
            result.smoothed_mean.get(0, 0),
            result.filter.filtered_mean.get(0, 0)
        );
        assert_eq!(
            result.smoothed_covariance[0].get(0, 0),
            result.filter.filtered_covariance[0].get(0, 0)
        );
    }

    #[test]
    fn trailing_missing_measurements_do_not_require_an_rts_inverse() {
        let model = LinearGaussianStateSpace::new(
            Matrix::from_row_major(1, 1, &[0.0]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Matrix::from_row_major(1, 1, &[0.0]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Vector::from_slice(&[0.0]),
            Vector::from_slice(&[0.0]),
            Vector::from_slice(&[2.0]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Policy::default(),
        )
        .expect("semidefinite process covariance is valid");
        let observations = Matrix::from_row_major(3, 1, &[4.0, 0.0, 0.0]);
        let result = model
            .smooth(
                &observations,
                Some(&[true, false, false]),
                &Session::new("state-space", "trailing-missing-smoother"),
            )
            .expect("no future evidence requires no singular RTS solve")
            .value;

        // Conditioning x_0 ~ N(2, 1) on y_0 = x_0 + e, e ~ N(0, 1),
        // gives N(3, 1/2).  The zero transition makes every later state
        // deterministic at zero; missing measurements cannot change either.
        assert_eq!(result.smoothed_mean.to_row_major(), vec![3.0, 0.0, 0.0]);
        assert_eq!(
            result
                .smoothed_covariance
                .iter()
                .map(|covariance| covariance.get(0, 0))
                .collect::<Vec<_>>(),
            vec![0.5, 0.0, 0.0]
        );
        assert_eq!(
            result.smoothed_mean.to_row_major(),
            result.filter.filtered_mean.to_row_major()
        );
    }

    #[test]
    fn all_missing_smoothing_is_exact_prior_propagation() {
        let model = LinearGaussianStateSpace::new(
            Matrix::from_row_major(1, 1, &[0.0]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Matrix::from_row_major(1, 1, &[0.0]),
            Matrix::from_row_major(1, 1, &[0.0]),
            Vector::from_slice(&[1.0]),
            Vector::from_slice(&[0.0]),
            Vector::from_slice(&[2.0]),
            Matrix::from_row_major(1, 1, &[0.0]),
            Policy::default(),
        )
        .expect("a deterministic state-space model is valid");
        let result = model
            .smooth(
                &Matrix::from_row_major(3, 1, &[0.0, 0.0, 0.0]),
                Some(&[false, false, false]),
                &Session::new("state-space", "all-missing-smoother"),
            )
            .expect("absence of data requires no covariance inverse");

        assert!(result.report.contains(IssueCode::AllMissing));
        assert_eq!(
            result.value.smoothed_mean.to_row_major(),
            vec![2.0, 1.0, 1.0]
        );
        assert_eq!(
            result.value.smoothed_mean.to_row_major(),
            result.value.filter.filtered_mean.to_row_major()
        );
        assert!(result
            .value
            .smoothed_covariance
            .iter()
            .all(|covariance| covariance.get(0, 0) == 0.0));
    }
}
