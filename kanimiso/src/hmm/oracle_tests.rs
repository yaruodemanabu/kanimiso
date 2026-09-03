//! Replay tests for the independent standard-library Decimal HMM oracle.

use super::viterbi::viterbi_path;
use super::{CategoricalEmission, Emission, GaussianEmission, HiddenMarkovModel, PoissonEmission};
use crate::data::{Matrix, Vector};
use ojizou_san::Session;
use serde_json::Value;
use signlred::Policy;

#[derive(Default)]
struct FixedErrors {
    gaussian_loglik: f64,
    gaussian_viterbi: f64,
    categorical_loglik: f64,
    categorical_viterbi: f64,
    poisson_loglik: f64,
    poisson_viterbi: f64,
}

fn decimal(value: &Value) -> f64 {
    value
        .as_str()
        .expect("Decimal value encoded as a string")
        .parse::<f64>()
        .expect("Decimal value representable as binary64")
}

fn decimal_vector(value: &Value) -> Vector {
    Vector::from_iter(
        value
            .as_array()
            .expect("Decimal vector")
            .iter()
            .map(decimal),
    )
}

fn decimal_matrix(value: &Value) -> Matrix {
    let rows = value.as_array().expect("Decimal matrix rows");
    let row_count = rows.len();
    let column_count = rows
        .first()
        .expect("non-empty Decimal matrix")
        .as_array()
        .expect("Decimal matrix row")
        .len();
    Matrix::from_fn(row_count, column_count, |row, column| {
        decimal(&rows[row].as_array().expect("Decimal matrix row")[column])
    })
}

fn expected_path(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("Viterbi path")
        .iter()
        .map(|state| state.as_u64().expect("state index") as f64)
        .collect()
}

fn integer_observations(value: &Value) -> Matrix {
    let observations = value.as_array().expect("integer observations");
    Matrix::from_fn(observations.len(), 1, |row, _| {
        observations[row].as_u64().expect("non-negative count") as f64
    })
}

fn check_fixed_model<E: Emission>(
    case: &Value,
    model: &HiddenMarkovModel<E>,
    observations: &Matrix,
) -> (f64, f64) {
    let name = case["name"].as_str().expect("case name");
    let session = Session::new(format!("hmm-decimal-{name}"), "fixed");
    let score = model
        .score(observations, &session)
        .unwrap_or_else(|failure| panic!("{name} score: {failure:?}"));
    assert!(score.report.issues().is_empty(), "{name} score report");
    let decoded = model
        .decode(observations, &session)
        .unwrap_or_else(|failure| panic!("{name} decode: {failure:?}"));
    assert!(decoded.report.issues().is_empty(), "{name} decode report");
    let expected = &case["expected"];
    let path = expected_path(&expected["viterbi_path"]);
    assert_eq!(decoded.value.as_slice(), path, "{name} public path");

    let parsed = E::observations(observations)
        .unwrap_or_else(|failure| panic!("{name} observations: {failure:?}"));
    let log_emissions: Vec<Vec<f64>> = parsed
        .iter()
        .map(|observation| {
            model
                .emissions()
                .iter()
                .map(|emission| emission.log_prob(observation))
                .collect()
        })
        .collect();
    let (kernel_path, kernel_score) =
        viterbi_path(model.initial(), model.transition(), &log_emissions);
    assert_eq!(kernel_path.as_slice(), path, "{name} kernel path");

    let state_count = model.emissions().len();
    let path_count = (0..observations.nrows()).fold(1_u64, |count, _| count * state_count as u64);
    assert_eq!(
        expected["path_count"].as_u64(),
        Some(path_count),
        "{name} exhaustive path count"
    );

    (
        (score.value - decimal(&expected["log_likelihood"])).abs(),
        (kernel_score - decimal(&expected["viterbi_log_probability"])).abs(),
    )
}

fn observe_scalar_error(actual: f64, expected: &Value, maximum: &mut f64) {
    *maximum = maximum.max((actual - decimal(expected)).abs());
}

#[test]
fn fixed_parameter_families_match_decimal_brute_force() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../golden/hmm.json")).expect("golden/hmm.json");
    let cases = fixture["cases"].as_array().expect("HMM cases");
    assert_eq!(
        cases.len(),
        fixture["case_count"].as_u64().expect("case count") as usize
    );
    assert_eq!(
        fixture["metadata"]["provenance"].as_str(),
        Some("independent Decimal brute force; not hmmlearn")
    );

    let mut errors = FixedErrors::default();
    for case in cases {
        let input = &case["input"];
        let initial = decimal_vector(&input["initial"]);
        let transition = decimal_matrix(&input["transition"]);
        let max_iter = input["baum_welch_iterations"].as_u64().unwrap_or(1) as usize;
        match case["family"].as_str().expect("emission family") {
            "gaussian" => {
                let emissions = input["emissions"]
                    .as_array()
                    .expect("Gaussian emissions")
                    .iter()
                    .map(|emission| {
                        GaussianEmission::new(
                            decimal_vector(&emission["mean"]),
                            decimal_vector(&emission["variance"]),
                        )
                        .expect("valid Gaussian oracle parameters")
                    })
                    .collect();
                let observations = decimal_matrix(&input["observations"]);
                let model = HiddenMarkovModel::new(
                    initial,
                    transition,
                    emissions,
                    max_iter,
                    false,
                    Policy::default(),
                )
                .expect("valid Gaussian oracle model");
                let (loglik, viterbi) = check_fixed_model(case, &model, &observations);
                errors.gaussian_loglik = errors.gaussian_loglik.max(loglik);
                errors.gaussian_viterbi = errors.gaussian_viterbi.max(viterbi);
            }
            "categorical" => {
                let emissions = input["emissions"]
                    .as_array()
                    .expect("categorical emissions")
                    .iter()
                    .map(|emission| {
                        CategoricalEmission::from_weights(decimal_vector(&emission["weights"]))
                            .expect("valid categorical oracle weights")
                    })
                    .collect();
                let observations = integer_observations(&input["observations"]);
                let model = HiddenMarkovModel::new(
                    initial,
                    transition,
                    emissions,
                    max_iter,
                    false,
                    Policy::default(),
                )
                .expect("valid categorical oracle model");
                let (loglik, viterbi) = check_fixed_model(case, &model, &observations);
                errors.categorical_loglik = errors.categorical_loglik.max(loglik);
                errors.categorical_viterbi = errors.categorical_viterbi.max(viterbi);
            }
            "poisson" => {
                let emissions = input["emissions"]
                    .as_array()
                    .expect("Poisson emissions")
                    .iter()
                    .map(|emission| {
                        PoissonEmission::new(decimal(&emission["rate"]))
                            .expect("valid Poisson oracle rate")
                    })
                    .collect();
                let observations = integer_observations(&input["observations"]);
                let model = HiddenMarkovModel::new(
                    initial,
                    transition,
                    emissions,
                    max_iter,
                    false,
                    Policy::default(),
                )
                .expect("valid Poisson oracle model");
                let (loglik, viterbi) = check_fixed_model(case, &model, &observations);
                errors.poisson_loglik = errors.poisson_loglik.max(loglik);
                errors.poisson_viterbi = errors.poisson_viterbi.max(viterbi);
            }
            family => panic!("unsupported oracle family {family}"),
        }
    }

    eprintln!(
        "HMM Decimal fixed max_abs: gaussian(loglik={:.17e}, viterbi={:.17e}), \
         categorical(loglik={:.17e}, viterbi={:.17e}), \
         poisson(loglik={:.17e}, viterbi={:.17e})",
        errors.gaussian_loglik,
        errors.gaussian_viterbi,
        errors.categorical_loglik,
        errors.categorical_viterbi,
        errors.poisson_loglik,
        errors.poisson_viterbi,
    );

    // Measured on 2026-09-03: Gaussian loglik 0, Gaussian Viterbi
    // 1.77635683940025046e-15, and both categorical maxima
    // 1.77635683940025046e-15. The 7.2e-15 bound is approximately 4x the
    // largest nonzero error in these two families.
    assert!(errors.gaussian_loglik <= 7.2e-15);
    assert!(errors.gaussian_viterbi <= 7.2e-15);
    assert!(errors.categorical_loglik <= 7.2e-15);
    assert!(errors.categorical_viterbi <= 7.2e-15);
    // Measured Poisson errors were 1.59872115546022542e-14 for loglik and
    // 1.42108547152020037e-14 for Viterbi; each bound is approximately 4x.
    assert!(errors.poisson_loglik <= 6.4e-14);
    assert!(errors.poisson_viterbi <= 5.7e-14);
}

#[test]
fn gaussian_baum_welch_matches_two_decimal_iterations() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../golden/hmm.json")).expect("golden/hmm.json");
    let case = fixture["cases"]
        .as_array()
        .expect("HMM cases")
        .iter()
        .find(|case| case["name"] == "gaussian_univariate_two_state")
        .expect("Gaussian Baum--Welch case");
    let input = &case["input"];
    let expected = &case["baum_welch"];
    let emissions = input["emissions"]
        .as_array()
        .expect("Gaussian emissions")
        .iter()
        .map(|emission| {
            GaussianEmission::new(
                decimal_vector(&emission["mean"]),
                decimal_vector(&emission["variance"]),
            )
            .expect("valid Gaussian oracle parameters")
        })
        .collect();
    let observations = decimal_matrix(&input["observations"]);
    let iterations = input["baum_welch_iterations"]
        .as_u64()
        .expect("Baum--Welch iterations") as usize;
    assert_eq!(expected["iterations"].as_u64(), Some(iterations as u64));
    let model = HiddenMarkovModel::new(
        decimal_vector(&input["initial"]),
        decimal_matrix(&input["transition"]),
        emissions,
        iterations,
        false,
        Policy::default(),
    )
    .expect("valid Gaussian Baum--Welch model");
    let fitted = model
        .fit(
            &observations,
            &Session::new("hmm-decimal-gaussian", "baum-welch"),
        )
        .expect("two Gaussian Baum--Welch iterations");
    assert!(fitted.report.issues().is_empty());

    let mut parameter_error = 0.0_f64;
    for (state, value) in fitted.value.initial().as_slice().iter().enumerate() {
        observe_scalar_error(*value, &expected["initial"][state], &mut parameter_error);
    }
    for source in 0..fitted.value.transition().nrows() {
        for destination in 0..fitted.value.transition().ncols() {
            observe_scalar_error(
                fitted.value.transition().get(source, destination),
                &expected["transition"][source][destination],
                &mut parameter_error,
            );
        }
    }
    for (state, emission) in fitted.value.emissions().iter().enumerate() {
        for coordinate in 0..emission.mean().len() {
            observe_scalar_error(
                emission.mean()[coordinate],
                &expected["emissions"][state]["mean"][coordinate],
                &mut parameter_error,
            );
            observe_scalar_error(
                emission.variance()[coordinate],
                &expected["emissions"][state]["variance"][coordinate],
                &mut parameter_error,
            );
        }
    }
    let after_score = fitted
        .value
        .score(
            &observations,
            &Session::new("hmm-decimal-gaussian", "after-score"),
        )
        .expect("score after two iterations")
        .value;
    let score_error = (after_score - decimal(&expected["log_likelihood_after_fit"])).abs();
    let after_path = fitted
        .value
        .decode(
            &observations,
            &Session::new("hmm-decimal-gaussian", "after-path"),
        )
        .expect("path after two iterations")
        .value;
    assert_eq!(
        after_path.as_slice(),
        expected_path(&expected["viterbi_path_after_fit"])
    );
    eprintln!(
        "HMM Decimal Baum--Welch max_abs: parameters={parameter_error:.17e}, score={score_error:.17e}"
    );

    // Measured on 2026-09-03: parameters 4.44089209850062616e-16 and score
    // 1.77635683940025046e-15. Both bounds are approximately 4x.
    assert!(parameter_error <= 1.8e-15);
    assert!(score_error <= 7.2e-15);
}
