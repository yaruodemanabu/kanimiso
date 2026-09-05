use number_ruler::{
    linear_shap, AdditiveModel, AnalysisOptions, Family, GeneralizedLinearModel, Likelihood,
    LinearModel, Matrix, MixedModel, Session, SplineTerm, Vector,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("../golden/regression.json")).unwrap()
}
fn matrix(value: &Value) -> Matrix {
    let rows = value.as_array().unwrap();
    Matrix::from_fn(rows.len(), rows[0].as_array().unwrap().len(), |i, j| {
        rows[i][j].as_f64().unwrap()
    })
}
fn vector(value: &Value) -> Vector {
    Vector::from_iter(
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap()),
    )
}
fn family(value: &Value) -> Family {
    match value.as_str().unwrap() {
        "Gaussian" => Family::Gaussian,
        "Binomial" => Family::Binomial,
        "Poisson" => Family::Poisson,
        _ => unreachable!(),
    }
}
fn session() -> Session {
    Session::new("number_ruler", "test")
}
fn error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / (1.0 + expected.abs())
}
fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && error(actual, expected) <= tolerance,
        "{actual} vs {expected}: error={} > {tolerance}",
        error(actual, expected)
    );
}

#[test]
fn external_lm_and_glm_oracles() {
    for case in fixture()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "regression")
    {
        let x = matrix(&case["x"]);
        let y = vector(&case["y"]);
        let model = GeneralizedLinearModel {
            family: family(&case["family"]),
            ..Default::default()
        }
        .fit(&x, &y, &session())
        .unwrap();
        let fit = model.value;
        assert!(fit.annotations.len() >= 4 && !model.report.issues().is_empty());
        let mut worst: f64 = 0.0;
        for (j, &b) in fit.beta.as_slice().iter().enumerate() {
            worst = worst.max(error(b, case["beta"][j].as_f64().unwrap()));
            worst = worst.max(error(
                fit.diagnostics.coefficients[j].standard_error.unwrap(),
                case["se"][j].as_f64().unwrap(),
            ));
            worst = worst.max(error(
                fit.diagnostics.coefficients[j].p_value.unwrap(),
                case["p"][j].as_f64().unwrap(),
            ));
        }
        for (i, &actual) in fit.fitted_values.as_slice().iter().enumerate() {
            worst = worst.max(error(actual, case["fitted"][i].as_f64().unwrap()));
        }
        for i in 0..fit.beta.len() {
            for j in 0..fit.beta.len() {
                worst = worst.max(error(
                    fit.covariance.as_ref().unwrap().get(i, j),
                    case["covariance"][i][j].as_f64().unwrap(),
                ));
            }
        }
        worst = worst.max(error(
            fit.diagnostics.deviance,
            case["deviance"].as_f64().unwrap(),
        ));
        eprintln!(
            "{:?} statsmodels maximum scaled error: {worst:e}",
            fit.family
        );
        // Measured 2026-09-05: Gaussian 1.81e-14, Binomial 2.15e-9,
        // Poisson 1.59e-14; approximately fourfold margins.
        assert!(
            worst
                < if fit.family == Family::Binomial {
                    8.6e-9
                } else {
                    7.3e-14
                }
        );
        close(
            fit.diagnostics.leverage.as_slice().iter().sum(),
            fit.beta.len() as f64,
            128.0 * f64::EPSILON,
        );
    }
}

#[test]
fn external_mixed_likelihood_oracles() {
    for case in fixture()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "mixed" || c["kind"] == "glmm")
    {
        let family = family(&case["family"]);
        let config = MixedModel {
            family,
            likelihood: if case["likelihood"] == "Restricted" {
                Likelihood::Restricted
            } else {
                Likelihood::Maximum
            },
            quadrature_points: 64,
            options: AnalysisOptions {
                max_iterations: 800,
                ..Default::default()
            },
        };
        let x = matrix(&case["x"]);
        let y = vector(&case["y"]);
        let groups: Vec<_> = case["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let result = config.fit(&x, &y, &groups, &session()).unwrap();
        let fit = result.value;
        let mut worst = error(fit.log_likelihood, case["loglik"].as_f64().unwrap());
        for j in 0..fit.beta.len() {
            worst = worst.max(error(fit.beta[j], case["beta"][j].as_f64().unwrap()));
        }
        worst = worst.max(error(
            fit.random_intercept_variance,
            case["random_variance"].as_f64().unwrap(),
        ));
        if let Some(value) = fit.residual_variance {
            worst = worst.max(error(value, case["residual_variance"].as_f64().unwrap()));
            for (j, &(_, value)) in fit.random_effects.iter().enumerate() {
                worst = worst.max(error(value, case["effects"][j].as_f64().unwrap()));
            }
        }
        eprintln!(
            "{:?} {:?} mixed maximum scaled error: {worst:e}; ll={} variance={}",
            family, config.likelihood, fit.log_likelihood, fit.random_intercept_variance
        );
        // Measured max 1.60e-8 on 2026-09-05; fourfold margin.
        assert!(worst < 6.4e-8);
        assert!(fit.annotations.len() >= 5);
        assert!(fit.predict_marginal(&x, &session()).is_ok());
        assert!(fit.predict_conditional(&x, &groups, &session()).is_ok());
        assert!(fit
            .predict_conditional(&x, &vec![999; x.nrows()], &session())
            .is_err());
    }
}

#[test]
fn gaussian_and_generalized_fit_are_row_permutation_equivariant() {
    for case in fixture()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "regression")
    {
        let x = matrix(&case["x"]);
        let y = vector(&case["y"]);
        let estimator = GeneralizedLinearModel {
            family: family(&case["family"]),
            ..Default::default()
        };
        let original = estimator.fit(&x, &y, &session()).unwrap().value;
        let reversed = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| x.get(x.nrows() - 1 - i, j));
        let response = Vector::from_iter(y.as_slice().iter().rev().copied());
        let other = estimator
            .fit(&reversed, &response, &session())
            .unwrap()
            .value;
        for j in 0..original.beta.len() {
            close(other.beta[j], original.beta[j], 1e-8);
        }
    }
}

#[test]
fn additive_effects_reconstruct_predictor_and_reuse_training_basis() {
    for case in fixture()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "regression")
    {
        let x = matrix(&case["x"]);
        let y = vector(&case["y"]);
        let estimator = AdditiveModel {
            family: family(&case["family"]),
            terms: vec![
                SplineTerm {
                    feature: 0,
                    knots: vec![-0.5, 0.5],
                },
                SplineTerm {
                    feature: 1,
                    knots: vec![0.0],
                },
            ],
            penalty: 0.25,
            ..Default::default()
        };
        let fit = estimator.fit(&x, &y, &session()).unwrap().value;
        assert!(fit.regression.covariance.is_none());
        assert!(fit
            .regression
            .diagnostics
            .coefficients
            .iter()
            .all(|c| c.p_value.is_none()));
        let effects = fit.term_effects(&x, &session()).unwrap().value;
        let predictions = fit.predict(&x, &session()).unwrap().value;
        for i in 0..x.nrows() {
            close(
                predictions[i],
                estimator
                    .family
                    .mean(fit.regression.beta[0] + effects.get(i, 0) + effects.get(i, 1)),
                64.0 * f64::EPSILON,
            );
            let single = Matrix::from_fn(1, x.ncols(), |_, j| x.get(i, j));
            close(
                fit.predict(&single, &session()).unwrap().value[0],
                predictions[i],
                64.0 * f64::EPSILON,
            );
        }
        for j in 0..effects.ncols() {
            close(
                (0..effects.nrows()).map(|i| effects.get(i, j)).sum(),
                0.0,
                256.0 * f64::EPSILON,
            );
        }
    }
}

#[test]
fn unpenalized_linear_additive_model_reduces_to_lm_without_knots() {
    let data = fixture();
    let case = &data["cases"][0];
    let x = matrix(&case["x"]);
    let y = vector(&case["y"]);
    let lm = LinearModel::default()
        .fit(&x, &y, &session())
        .unwrap()
        .value;
    let additive = AdditiveModel {
        terms: (0..x.ncols())
            .map(|feature| SplineTerm {
                feature,
                knots: vec![],
            })
            .collect(),
        ..Default::default()
    }
    .fit(&x, &y, &session())
    .unwrap()
    .value;
    for i in 0..y.len() {
        close(
            lm.fitted_values[i],
            additive.regression.fitted_values[i],
            64.0 * f64::EPSILON,
        );
    }
}

#[test]
fn shap_matches_exhaustive_coalitions_on_linear_predictor_scale() {
    for case in fixture()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "regression")
    {
        let x = matrix(&case["x"]);
        let y = vector(&case["y"]);
        let fit = GeneralizedLinearModel {
            family: family(&case["family"]),
            ..Default::default()
        }
        .fit(&x, &y, &session())
        .unwrap()
        .value;
        let attribution = linear_shap(&fit, &x, &x, &session()).unwrap().value;
        // Independently average predictions after every coalition intervention.
        for row in 0..x.nrows() {
            let value = |mask: usize| {
                (0..x.nrows())
                    .map(|background| {
                        fit.beta[0]
                            + (0..2)
                                .map(|j| {
                                    fit.beta[j + 1]
                                        * x.get(
                                            if mask & (1 << j) != 0 {
                                                row
                                            } else {
                                                background
                                            },
                                            j,
                                        )
                                })
                                .sum::<f64>()
                    })
                    .sum::<f64>()
                    / x.nrows() as f64
            };
            for j in 0..2 {
                let other = 1 << (1 - j);
                let oracle = 0.5 * ((value(1 << j) - value(0)) + (value(3) - value(other)));
                close(
                    attribution.contributions.get(row, j),
                    oracle,
                    128.0 * f64::EPSILON,
                );
            }
            let reconstructed = attribution.base_value
                + (0..2)
                    .map(|j| attribution.contributions.get(row, j))
                    .sum::<f64>();
            close(
                fit.family.mean(reconstructed),
                fit.fitted_values[row],
                128.0 * f64::EPSILON,
            );
        }
        assert!(attribution
            .annotations
            .iter()
            .any(|a| a.statement.contains("conditional")));
    }
}

#[test]
fn invalid_inputs_fail_without_panics_or_silent_imputation() {
    let x = Matrix::from_fn(20, 1, |i, _| i as f64);
    let y = Vector::from_iter((0..20).map(|i| (i % 3) as f64));
    assert!(LinearModel::default()
        .fit(&x, &Vector::zeros(1), &session())
        .is_err());
    assert!(LinearModel::default()
        .fit(&x, &Vector::filled(20, 1.0), &session())
        .is_err());
    let invalid = Matrix::from_fn(20, 1, |i, _| if i == 3 { f64::NAN } else { i as f64 });
    assert!(LinearModel::default()
        .fit(&invalid, &y, &session())
        .is_err());
    for family in [Family::Binomial, Family::Poisson] {
        assert!(GeneralizedLinearModel {
            family,
            ..Default::default()
        }
        .fit(&x, &Vector::filled(20, 0.25), &session())
        .is_err());
    }
    let separation = Vector::from_iter((0..20).map(|i| f64::from(i >= 10)));
    assert!(GeneralizedLinearModel {
        family: Family::Binomial,
        ..Default::default()
    }
    .fit(&x, &separation, &session())
    .is_err());
    let fit = LinearModel::default()
        .fit(&x, &y, &session())
        .unwrap()
        .value;
    assert!(fit.predict(&Matrix::zeros(2, 2), &session()).is_err());
    assert!(fit.predict(&invalid, &session()).is_err());
    assert!(linear_shap(&fit, &Matrix::zeros(0, 1), &x, &session()).is_err());
    let outside = Matrix::from_row_major(1, 1, &[50.0]);
    assert!(fit
        .predict(&outside, &session())
        .unwrap()
        .report
        .issues()
        .iter()
        .any(|i| i.message.contains("outside training range")));
    assert!(MixedModel::default()
        .fit(&x, &y, &[0; 20], &session())
        .is_err());
    assert!(AdditiveModel {
        terms: vec![SplineTerm {
            feature: 0,
            knots: vec![50.0]
        }],
        ..Default::default()
    }
    .fit(&x, &y, &session())
    .is_err());
}

#[test]
fn normal_tail_oracle_retains_tiny_nonzero_probabilities() {
    let mut worst: f64 = 0.0;
    for case in fixture()["normal_tails"].as_array().unwrap() {
        let z = case["z"].as_f64().unwrap();
        let expected = case["p"].as_f64().unwrap();
        let actual = tsutsumi::special::norm_pvalue_two_sided(z);
        assert!(actual > 0.0);
        assert_eq!(actual, tsutsumi::special::norm_pvalue_two_sided(-z));
        worst = worst.max((actual / expected - 1.0).abs());
    }
    eprintln!("normal tail relative error: {worst:e}");
    // Measured 9.24e-14 relative on 2026-09-05, fourfold margin.
    assert!(worst < 3.7e-13);
}

#[test]
fn mixed_fit_is_invariant_to_group_labels_and_row_order() {
    let data = fixture();
    for case in data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "mixed" || c["kind"] == "glmm")
    {
        let x = matrix(&case["x"]);
        let y = vector(&case["y"]);
        let groups: Vec<_> = case["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g.as_u64().unwrap())
            .collect();
        let config = MixedModel {
            family: family(&case["family"]),
            likelihood: if case["likelihood"] == "Restricted" {
                Likelihood::Restricted
            } else {
                Likelihood::Maximum
            },
            quadrature_points: 64,
            options: AnalysisOptions {
                max_iterations: 800,
                ..Default::default()
            },
        };
        let original = config.fit(&x, &y, &groups, &session()).unwrap().value;
        let reverse = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| x.get(x.nrows() - 1 - i, j));
        let response = Vector::from_iter(y.as_slice().iter().rev().copied());
        let relabeled: Vec<_> = groups.iter().rev().map(|g| 100 - g).collect();
        let other = config
            .fit(&reverse, &response, &relabeled, &session())
            .unwrap()
            .value;
        let mut worst = error(
            original.random_intercept_variance,
            other.random_intercept_variance,
        );
        for j in 0..original.beta.len() {
            worst = worst.max(error(original.beta[j], other.beta[j]));
        }
        worst = worst.max(error(original.log_likelihood, other.log_likelihood));
        eprintln!("mixed permutation {:?}: {worst:e}", config.family);
        // Measured max 2.16e-8 on 2026-09-05; fourfold margin.
        assert!(worst < 8.64e-8);
    }
}

#[test]
fn zero_random_variance_face_reduces_to_the_population_linear_model() {
    let x = Matrix::from_fn(32, 1, |i, _| (i % 4) as f64 - 1.5);
    let y =
        Vector::from_iter((0..32).map(|i| 1.0 + 2.0 * x.get(i, 0) + [-0.5, 0.5, 0.5, -0.5][i % 4]));
    let groups: Vec<_> = (0..32).map(|i| i / 4).collect();
    let reference = LinearModel::default()
        .fit(&x, &y, &session())
        .unwrap()
        .value;
    for likelihood in [Likelihood::Maximum, Likelihood::Restricted] {
        let fit = MixedModel {
            likelihood,
            ..Default::default()
        }
        .fit(&x, &y, &groups, &session())
        .unwrap()
        .value;
        assert_eq!(fit.random_intercept_variance, 0.0);
        for j in 0..fit.beta.len() {
            close(fit.beta[j], reference.beta[j], 32.0 * f64::EPSILON);
        }
        assert!(fit.random_effects.iter().all(|(_, value)| *value == 0.0));
    }
}

#[test]
fn quality_policy_cannot_turn_invalid_input_into_a_fit() {
    let options = AnalysisOptions {
        policy: number_ruler::Policy {
            abort_at: signlred::Severity::Fatal,
            abort_on_meaningless: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let x = Matrix::from_fn(20, 1, |i, _| i as f64);
    assert!(LinearModel {
        options: options.clone()
    }
    .fit(&x, &Vector::zeros(1), &session())
    .is_err());
    let invalid = AnalysisOptions {
        max_iterations: 0,
        ..Default::default()
    };
    assert!(GeneralizedLinearModel {
        family: Family::Poisson,
        options: invalid
    }
    .fit(
        &x,
        &Vector::from_iter((0..20).map(|i| (i % 3) as f64)),
        &session()
    )
    .is_err());
    let invalid = AnalysisOptions {
        policy: number_ruler::Policy {
            optimizer_objective_tol: f64::NAN,
            ..Default::default()
        },
        ..options
    };
    assert!(LinearModel { options: invalid }
        .fit(&x, &Vector::zeros(20), &session())
        .is_err());
}

#[test]
fn additive_penalty_matches_a_scalar_ridge_closed_form() {
    let x = Matrix::from_fn(24, 1, |i, _| i as f64);
    let y =
        Vector::from_iter((0..24).map(|i| 1.0 + 0.8 * x.get(i, 0) + ((i * 7) % 11) as f64 / 10.0));
    let config = AdditiveModel {
        terms: vec![SplineTerm {
            feature: 0,
            knots: vec![],
        }],
        penalty: 2.0,
        ..Default::default()
    };
    let fit = config.fit(&x, &y, &session()).unwrap().value;
    let center = 11.5;
    let scale = 11.5;
    let basis: Vec<_> = (0..24).map(|i| (x.get(i, 0) - center) / scale).collect();
    let expected = basis
        .iter()
        .enumerate()
        .map(|(i, &v)| v * (y[i] - y.mean()))
        .sum::<f64>()
        / (basis.iter().map(|v| v * v).sum::<f64>() + config.penalty);
    let worst =
        error(fit.regression.beta[1], expected).max(error(fit.regression.beta[0], y.mean()));
    eprintln!("scalar additive ridge error: {worst:e}");
    assert!(worst < 64.0 * f64::EPSILON);
    let effective = 1.0
        + basis.iter().map(|v| v * v).sum::<f64>()
            / (basis.iter().map(|v| v * v).sum::<f64>() + config.penalty);
    close(
        fit.regression.diagnostics.leverage.as_slice().iter().sum(),
        effective,
        64.0 * f64::EPSILON,
    );
}

#[test]
fn additive_identification_uses_selected_terms_not_unused_columns() {
    let x = Matrix::from_fn(24, 40, |i, j| if j == 0 { i as f64 } else { 0.0 });
    let y =
        Vector::from_iter((0..24).map(|i| 1.0 + 0.8 * x.get(i, 0) + ((i * 7) % 11) as f64 / 10.0));
    let config = AdditiveModel {
        terms: vec![SplineTerm {
            feature: 0,
            knots: vec![],
        }],
        ..Default::default()
    };
    assert!(config.fit(&x, &y, &session()).is_ok());
}
