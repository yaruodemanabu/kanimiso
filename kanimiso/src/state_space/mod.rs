//! Verified, time-invariant linear Gaussian state-space inference.
//!
//! The core owns one Kalman recursion for dense runtime-sized matrices.  It
//! supports explicit element-wise missingness, Joseph covariance updates, the
//! exact Gaussian innovation likelihood, and Rauch--Tung--Striebel smoothing.
//! Public innovation outputs are compact over the observed channels.  The
//! committed oracle additionally stores full-channel intermediates solely to
//! audit masking identities; those full intermediates are not part of the API.

mod filter;
mod model;
mod smoother;

pub use filter::KalmanFilterResult;
pub use model::LinearGaussianStateSpace;
pub use smoother::KalmanSmootherResult;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Matrix, Vector};
    use ojizou_san::Session;
    use serde_json::Value;
    use signlred::{IssueCode, Policy};

    fn scalar_model() -> LinearGaussianStateSpace {
        LinearGaussianStateSpace::new(
            Matrix::from_row_major(1, 1, &[1.0]),
            Matrix::from_row_major(1, 1, &[3.0]),
            Matrix::from_row_major(1, 1, &[0.5]),
            Matrix::from_row_major(1, 1, &[5.0]),
            Vector::from_slice(&[0.0]),
            Vector::from_slice(&[0.0]),
            Vector::from_slice(&[2.0]),
            Matrix::from_row_major(1, 1, &[4.0]),
            Policy::default(),
        )
        .expect("valid scalar model")
    }

    fn partially_missing_model() -> LinearGaussianStateSpace {
        LinearGaussianStateSpace::new(
            Matrix::from_row_major(2, 2, &[0.8, 0.1, 0.0, 0.9]),
            Matrix::from_row_major(2, 2, &[1.0, 0.2, 0.3, 1.0]),
            Matrix::from_row_major(2, 2, &[0.04, 0.01, 0.01, 0.03]),
            Matrix::from_row_major(2, 2, &[0.09, 0.02, 0.02, 0.16]),
            Vector::from_slice(&[0.0, 0.0]),
            Vector::from_slice(&[0.0, 0.0]),
            Vector::from_slice(&[0.2, -0.1]),
            Matrix::from_row_major(2, 2, &[0.5, 0.1, 0.1, 0.4]),
            Policy::default(),
        )
        .expect("valid partially missing model")
    }

    #[test]
    fn scalar_update_matches_the_closed_form() {
        let observations = Matrix::from_row_major(1, 1, &[7.0]);
        let result = scalar_model()
            .filter(&observations, None, &Session::new("state-space", "filter"))
            .expect("filter")
            .value;
        let expected_mean = 94.0 / 41.0;
        let expected_covariance = 20.0 / 41.0;
        let expected_log_likelihood =
            -0.5 * (std::f64::consts::TAU.ln() + 41.0_f64.ln() + 1.0 / 41.0);
        let mean_error = (result.filtered_mean.get(0, 0) - expected_mean).abs();
        let covariance_error =
            (result.filtered_covariance[0].get(0, 0) - expected_covariance).abs();
        let likelihood_error = (result.log_likelihood - expected_log_likelihood).abs();
        assert_eq!(mean_error, 0.0);
        // Measured 5.551e-17 on 2026-09-03; tolerance is approximately 4x.
        assert!(covariance_error <= 2.3e-16);
        assert_eq!(likelihood_error, 0.0);
    }

    #[test]
    fn explicit_missing_mask_skips_only_the_measurement_update() {
        let observations =
            Matrix::from_row_major(4, 2, &[0.4, -0.2, 0.0, 0.1, 0.3, 0.0, -0.1, 0.5]);
        let mask = [true, true, false, true, false, false, true, false];
        let result = partially_missing_model()
            .filter(
                &observations,
                Some(&mask),
                &Session::new("state-space", "filter"),
            )
            .expect("filter")
            .value;
        assert!(result.observed_columns[2].is_empty());
        assert!(result.innovations[2].is_empty());
        assert_eq!(result.log_likelihood_contributions[2], 0.0);
        for state in 0..2 {
            assert_eq!(
                result.filtered_mean.get(2, state),
                result.predicted_mean.get(2, state)
            );
            for other in 0..2 {
                assert_eq!(
                    result.filtered_covariance[2].get(state, other),
                    result.predicted_covariance[2].get(state, other)
                );
            }
        }
        let expected_next = Matrix::from_row_major(2, 2, &[0.8, 0.1, 0.0, 0.9])
            .matvec(&result.filtered_mean.row(2));
        for state in 0..2 {
            assert_eq!(result.predicted_mean.get(3, state), expected_next[state]);
        }
    }

    #[test]
    fn filtering_is_causal_and_the_final_smoothed_state_is_filtered() {
        let observations =
            Matrix::from_row_major(4, 2, &[0.4, -0.2, 0.0, 0.1, 0.3, 0.0, -0.1, 0.5]);
        let changed_future =
            Matrix::from_row_major(4, 2, &[0.4, -0.2, 0.0, 0.1, 0.3, 0.0, 200.0, -300.0]);
        let model = partially_missing_model();
        let first = model
            .filter(
                &observations,
                None,
                &Session::new("state-space", "filter-a"),
            )
            .expect("first filter")
            .value;
        let second = model
            .filter(
                &changed_future,
                None,
                &Session::new("state-space", "filter-b"),
            )
            .expect("second filter")
            .value;
        for time in 0..3 {
            for state in 0..2 {
                assert_eq!(
                    first.filtered_mean.get(time, state),
                    second.filtered_mean.get(time, state)
                );
            }
        }
        let smoothed = model
            .smooth(&observations, None, &Session::new("state-space", "smooth"))
            .expect("smooth")
            .value;
        for state in 0..2 {
            assert_eq!(
                smoothed.smoothed_mean.get(3, state),
                smoothed.filter.filtered_mean.get(3, state)
            );
            for other in 0..2 {
                assert_eq!(
                    smoothed.smoothed_covariance[3].get(state, other),
                    smoothed.filter.filtered_covariance[3].get(state, other)
                );
            }
        }
    }

    #[test]
    fn invalid_mask_nonfinite_placeholder_and_singular_innovation_fail() {
        let observations = Matrix::from_row_major(1, 2, &[0.0, 0.0]);
        let bad_mask = partially_missing_model()
            .filter(
                &observations,
                Some(&[true]),
                &Session::new("state-space", "bad-mask"),
            )
            .expect_err("mask length must fail");
        assert_eq!(bad_mask.primary.code, IssueCode::DimensionMismatch);

        let nonfinite = Matrix::from_row_major(1, 2, &[0.0, f64::NAN]);
        let bad_value = partially_missing_model()
            .filter(
                &nonfinite,
                Some(&[true, false]),
                &Session::new("state-space", "bad-value"),
            )
            .expect_err("masked placeholders must still be finite");
        assert_eq!(bad_value.primary.code, IssueCode::NonFiniteInput);

        let singular = LinearGaussianStateSpace::new(
            Matrix::from_row_major(1, 1, &[1.0]),
            Matrix::from_row_major(1, 1, &[1.0]),
            Matrix::from_row_major(1, 1, &[0.0]),
            Matrix::from_row_major(1, 1, &[0.0]),
            Vector::from_slice(&[0.0]),
            Vector::from_slice(&[0.0]),
            Vector::from_slice(&[0.0]),
            Matrix::from_row_major(1, 1, &[0.0]),
            Policy::default(),
        )
        .expect("PSD covariances are valid at construction");
        let failure = singular
            .filter(
                &Matrix::from_row_major(1, 1, &[1.0]),
                None,
                &Session::new("state-space", "singular"),
            )
            .expect_err("singular innovation must not be floored");
        assert_eq!(failure.primary.code, IssueCode::CholeskyFailed);
    }

    fn decimal(value: &Value) -> f64 {
        value
            .as_str()
            .expect("golden decimal string")
            .parse::<f64>()
            .expect("finite f64 golden value")
    }

    fn json_vector(value: &Value) -> Vector {
        Vector::from_iter(value.as_array().expect("vector array").iter().map(decimal))
    }

    fn json_matrix(value: &Value) -> Matrix {
        let rows = value.as_array().expect("matrix rows");
        let columns = rows
            .first()
            .map_or(0, |row| row.as_array().expect("matrix row array").len());
        Matrix::from_fn(rows.len(), columns, |i, j| {
            decimal(&rows[i].as_array().expect("matrix row array")[j])
        })
    }

    fn json_matrix_sequence(value: &Value) -> Vec<Matrix> {
        value
            .as_array()
            .expect("matrix sequence")
            .iter()
            .map(json_matrix)
            .collect()
    }

    fn json_observations(value: &Value) -> (Matrix, Vec<bool>) {
        let rows = value.as_array().expect("observation rows");
        let columns = rows
            .first()
            .expect("non-empty observations")
            .as_array()
            .expect("observation row")
            .len();
        let mut mask = Vec::with_capacity(rows.len() * columns);
        for row in rows {
            let row = row.as_array().expect("observation row");
            assert_eq!(row.len(), columns, "observation row width");
            mask.extend(row.iter().map(|cell| !cell.is_null()));
        }
        let observations = Matrix::from_fn(rows.len(), columns, |i, j| {
            let cell = &rows[i].as_array().expect("observation row")[j];
            if cell.is_null() {
                0.0
            } else {
                decimal(cell)
            }
        });
        (observations, mask)
    }

    fn observe_error(
        path: String,
        actual: f64,
        expected: f64,
        maximum_absolute: &mut (f64, String),
        maximum_relative: &mut (f64, String),
    ) {
        assert!(actual.is_finite(), "{path}: actual value is not finite");
        assert!(expected.is_finite(), "{path}: expected value is not finite");
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(1.0);
        assert!(absolute.is_finite(), "{path}: absolute error is not finite");
        assert!(relative.is_finite(), "{path}: relative error is not finite");
        if absolute > maximum_absolute.0 {
            *maximum_absolute = (absolute, path.clone());
        }
        if relative > maximum_relative.0 {
            *maximum_relative = (relative, path);
        }
    }

    fn compare_vector(
        path: &str,
        actual: &Vector,
        expected: &Value,
        maximum_absolute: &mut (f64, String),
        maximum_relative: &mut (f64, String),
    ) {
        let expected = expected.as_array().expect("expected vector");
        assert_eq!(actual.len(), expected.len(), "{path} length");
        for (index, value) in expected.iter().enumerate() {
            observe_error(
                format!("{path}[{index}]"),
                actual[index],
                decimal(value),
                maximum_absolute,
                maximum_relative,
            );
        }
    }

    fn compare_matrix(
        path: &str,
        actual: &Matrix,
        expected: &Value,
        maximum_absolute: &mut (f64, String),
        maximum_relative: &mut (f64, String),
    ) {
        let expected = expected.as_array().expect("expected matrix");
        assert_eq!(actual.nrows(), expected.len(), "{path} rows");
        for (i, row) in expected.iter().enumerate() {
            let row = row.as_array().expect("expected matrix row");
            assert_eq!(actual.ncols(), row.len(), "{path}[{i}] columns");
            for (j, value) in row.iter().enumerate() {
                observe_error(
                    format!("{path}[{i}][{j}]"),
                    actual.get(i, j),
                    decimal(value),
                    maximum_absolute,
                    maximum_relative,
                );
            }
        }
    }

    fn compare_matrix_sequence(
        path: &str,
        actual: &[Matrix],
        expected: &Value,
        maximum_absolute: &mut (f64, String),
        maximum_relative: &mut (f64, String),
    ) {
        let expected = expected.as_array().expect("expected matrix sequence");
        assert_eq!(actual.len(), expected.len(), "{path} length");
        for (time, matrix) in actual.iter().enumerate() {
            compare_matrix(
                &format!("{path}[{time}]"),
                matrix,
                &expected[time],
                maximum_absolute,
                maximum_relative,
            );
        }
    }

    #[test]
    fn joint_gaussian_decimal_golden_replays_filter_and_smoother() {
        let fixture: Value = serde_json::from_str(include_str!("../../../golden/state_space.json"))
            .expect("state-space golden JSON");
        let cases = fixture["cases"].as_array().expect("state-space cases");
        assert_eq!(
            cases.len(),
            fixture["case_count"].as_u64().expect("case count") as usize
        );
        let mut maximum_absolute = (0.0, String::new());
        let mut maximum_relative = (0.0, String::new());

        for case in cases {
            let name = case["name"].as_str().expect("case name");
            let input = &case["input"];
            let expected = &case["expected"];
            let model = LinearGaussianStateSpace::new(
                json_matrix(&input["transition_matrix"]),
                json_matrix(&input["observation_matrix"]),
                json_matrix(&input["process_covariance"]),
                json_matrix(&input["observation_covariance"]),
                json_vector(&input["transition_offset"]),
                json_vector(&input["observation_offset"]),
                json_vector(&input["initial_predicted_mean"]),
                json_matrix(&input["initial_predicted_covariance"]),
                Policy::default(),
            )
            .unwrap_or_else(|failure| panic!("{name} model: {failure}"));
            let (observations, mask) = json_observations(&input["observations"]);
            let result = model
                .smooth(
                    &observations,
                    Some(&mask),
                    &Session::new(name, "golden-smooth"),
                )
                .unwrap_or_else(|failure| panic!("{name} smooth: {failure}"))
                .value;
            let filtered = &result.filter;

            compare_matrix(
                &format!("{name}.predicted_means"),
                &filtered.predicted_mean,
                &expected["predicted_means"],
                &mut maximum_absolute,
                &mut maximum_relative,
            );
            compare_matrix(
                &format!("{name}.filtered_means"),
                &filtered.filtered_mean,
                &expected["filtered_means"],
                &mut maximum_absolute,
                &mut maximum_relative,
            );
            compare_matrix(
                &format!("{name}.smoothed_means"),
                &result.smoothed_mean,
                &expected["smoothed_means"],
                &mut maximum_absolute,
                &mut maximum_relative,
            );
            compare_matrix_sequence(
                &format!("{name}.predicted_covariances"),
                &filtered.predicted_covariance,
                &expected["predicted_covariances"],
                &mut maximum_absolute,
                &mut maximum_relative,
            );
            compare_matrix_sequence(
                &format!("{name}.filtered_covariances"),
                &filtered.filtered_covariance,
                &expected["filtered_covariances"],
                &mut maximum_absolute,
                &mut maximum_relative,
            );
            compare_matrix_sequence(
                &format!("{name}.smoothed_covariances"),
                &result.smoothed_covariance,
                &expected["smoothed_covariances"],
                &mut maximum_absolute,
                &mut maximum_relative,
            );
            compare_matrix_sequence(
                &format!("{name}.innovation_covariances"),
                &filtered.innovation_covariances,
                &expected["observed_innovation_covariances"],
                &mut maximum_absolute,
                &mut maximum_relative,
            );
            let expected_innovations = expected["observed_innovations"]
                .as_array()
                .expect("expected innovations");
            assert_eq!(filtered.innovations.len(), expected_innovations.len());
            for (time, innovation) in filtered.innovations.iter().enumerate() {
                compare_vector(
                    &format!("{name}.innovations[{time}]"),
                    innovation,
                    &expected_innovations[time],
                    &mut maximum_absolute,
                    &mut maximum_relative,
                );
            }
            compare_vector(
                &format!("{name}.loglik_by_time"),
                &filtered.log_likelihood_contributions,
                &expected["loglik_by_time"],
                &mut maximum_absolute,
                &mut maximum_relative,
            );
            observe_error(
                format!("{name}.total_loglik"),
                filtered.log_likelihood,
                decimal(&expected["total_loglik"]),
                &mut maximum_absolute,
                &mut maximum_relative,
            );

            let expected_indices = expected["observed_indices"]
                .as_array()
                .expect("expected observed indices");
            assert_eq!(filtered.observed_columns.len(), expected_indices.len());
            for (time, indices) in filtered.observed_columns.iter().enumerate() {
                let expected_row = expected_indices[time]
                    .as_array()
                    .expect("expected observed-index row")
                    .iter()
                    .map(|value| value.as_u64().expect("observed index") as usize)
                    .collect::<Vec<_>>();
                assert_eq!(*indices, expected_row, "{name}.observed_indices[{time}]");
            }

            let full_expected = json_matrix_sequence(&expected["full_innovation_covariances"]);
            for (time, expected_covariance) in full_expected.iter().enumerate() {
                let actual = crate::linalg::matrix_symmetrized(&crate::linalg::matrix_add(
                    &crate::linalg::matrix_multiply(
                        &crate::linalg::matrix_multiply(
                            &model.observation,
                            &filtered.predicted_covariance[time],
                        ),
                        &crate::linalg::matrix_transpose(&model.observation),
                    ),
                    &model.observation_covariance,
                ));
                compare_matrix(
                    &format!("{name}.full_innovation_covariances[{time}]"),
                    &actual,
                    &expected["full_innovation_covariances"][time],
                    &mut maximum_absolute,
                    &mut maximum_relative,
                );
                assert_eq!(actual.shape(), expected_covariance.shape());
            }
        }

        // Measured 3.553e-15 on 2026-09-03; tolerance is approximately 4.2x.
        assert!(
            maximum_absolute.0 <= 1.5e-14,
            "max abs {} at {}",
            maximum_absolute.0,
            maximum_absolute.1
        );
        // Measured 8.327e-16 on 2026-09-03; tolerance is approximately 4.1x.
        assert!(
            maximum_relative.0 <= 3.4e-15,
            "max rel {} at {}",
            maximum_relative.0,
            maximum_relative.1
        );
    }
}
