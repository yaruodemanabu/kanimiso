//! RiskMetrics-style EWMA conditional-variance model.

use super::common::{
    gaussian_qml_profile, inspect_scale_invariant_univariate, normalized_log_squares,
};
use crate::context::FitCtx;
use crate::data::Vector;
use crate::optimize::{decode_open_interval, encode_open_interval, NelderMead};
use crate::special::logsumexp;
use crate::traits::FitSeries;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, NumericalCompromise, Qualified, Result, Severity};

/// RiskMetrics-style EWMA variance.
///
/// \(h_t=\lambda h_{t-1}+(1-\lambda)\varepsilon_{t-1}^2\).
/// This recurrence is also used by arch `EWMAVariance`, but kanimiso uses a
/// full-sample mean-square initialization and configurable closed search
/// bounds; output parity with arch is not claimed.
#[derive(Clone, Debug)]
pub struct EwmaVol {
    /// Fixed decay; `None` QMLE-tunes \(\lambda\).
    pub lambda: Option<f64>,
    /// Closed QMLE search interval inside `(0, 1)`.
    pub search_bounds: (f64, f64),
    /// Deterministic QMLE starting value inside `search_bounds`.
    pub search_initial: f64,
    /// Number of coarse intervals used to select a QMLE search basin.
    pub search_grid_intervals: usize,
    /// Shared unconstrained solver used after the bounded basin scan.
    /// Its [`signlred::Policy`] also governs the enclosing EWMA diagnostics.
    pub optimizer: NelderMead,
}

impl Default for EwmaVol {
    fn default() -> Self {
        Self {
            lambda: None,
            search_bounds: (0.5, 0.999),
            search_initial: 0.94,
            search_grid_intervals: 64,
            optimizer: NelderMead::default(),
        }
    }
}

impl EwmaVol {
    /// RiskMetrics \(\lambda=0.94\).
    pub fn riskmetrics() -> Self {
        Self {
            lambda: Some(0.94),
            ..Self::default()
        }
    }

    /// QMLE \(\lambda\).
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted EWMA variances.
#[derive(Clone, Debug)]
pub struct FittedEwmaVol {
    /// Decay \(\lambda\).
    pub lambda: f64,
    /// In-sample conditional variances.
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

struct EwmaLogPath {
    scale: f64,
    normalized_log_squares: Vec<f64>,
    log_variances: Vec<f64>,
}

fn ewma_normalized_log_variance(e: &[f64], lam: f64) -> Option<EwmaLogPath> {
    if !lam.is_finite() || !(lam > 0.0 && lam < 1.0) {
        return None;
    }
    let normalized = normalized_log_squares(e)?;
    if normalized.scale == 0.0 {
        return Some(EwmaLogPath {
            scale: normalized.scale,
            normalized_log_squares: normalized.values,
            log_variances: vec![f64::NEG_INFINITY; e.len()],
        });
    }
    let mut log_variance = vec![normalized.log_mean_square; e.len()];
    let log_lambda = lam.ln();
    let log_one_minus_lambda = (-lam).ln_1p();
    for time in 1..e.len() {
        log_variance[time] = logsumexp(&[
            log_lambda + log_variance[time - 1],
            log_one_minus_lambda + normalized.values[time - 1],
        ]);
        if !log_variance[time].is_finite() {
            return None;
        }
    }
    Some(EwmaLogPath {
        scale: normalized.scale,
        normalized_log_squares: normalized.values,
        log_variances: log_variance,
    })
}

fn ewma_qml_objective(e: &[f64], lam: f64) -> f64 {
    let Some(path) = ewma_normalized_log_variance(e, lam) else {
        return f64::NAN;
    };
    if path.scale == 0.0 || path.log_variances.is_empty() {
        return f64::NAN;
    }
    ewma_profile_from_log_variance(&path)
}

fn tracked_ewma_qml_objective(e: &[f64], lam: f64, overflowed: &mut usize) -> f64 {
    let objective = ewma_qml_objective(e, lam);
    if objective == f64::INFINITY {
        *overflowed += 1;
    }
    objective
}

fn ewma_profile_from_log_variance(path: &EwmaLogPath) -> f64 {
    gaussian_qml_profile(&path.normalized_log_squares, &path.log_variances)
}

fn ewma_qml_derivative(e: &[f64], lam: f64) -> f64 {
    let Some(path) = ewma_normalized_log_variance(e, lam) else {
        return f64::NAN;
    };
    if path.scale == 0.0 || path.log_variances.is_empty() {
        return f64::NAN;
    }
    let log_lambda = lam.ln();
    let log_one_minus_lambda = (-lam).ln_1p();
    let mut relative_variance_derivative = 0.0;
    let mut objective_derivative = 0.0;
    for time in 0..path.log_variances.len() {
        if relative_variance_derivative != 0.0 {
            let standardized_square = if path.normalized_log_squares[time] == f64::NEG_INFINITY {
                0.0
            } else {
                (path.normalized_log_squares[time] - path.log_variances[time]).exp()
            };
            objective_derivative +=
                0.5 * relative_variance_derivative * (1.0 - standardized_square);
        }
        if time + 1 < path.log_variances.len() {
            let next_log_variance = path.log_variances[time + 1];
            let previous_variance_weight =
                (log_lambda + path.log_variances[time] - next_log_variance).exp();
            let shock_weight = if path.normalized_log_squares[time] == f64::NEG_INFINITY {
                0.0
            } else {
                (log_one_minus_lambda + path.normalized_log_squares[time] - next_log_variance).exp()
            };
            relative_variance_derivative = previous_variance_weight
                * (lam.recip() + relative_variance_derivative)
                - shock_weight / (1.0 - lam);
        }
    }
    objective_derivative
}

#[cfg(test)]
fn ewma_nll(e: &[f64], lam: f64) -> f64 {
    let Some(path) = ewma_normalized_log_variance(e, lam) else {
        return f64::NAN;
    };
    if path.scale == 0.0 || path.log_variances.is_empty() {
        return f64::NAN;
    }
    ewma_profile_from_log_variance(&path) + e.len() as f64 * path.scale.ln()
}

fn ewma_sigma2(e: &[f64], lam: f64) -> Vec<f64> {
    let Some(path) = ewma_normalized_log_variance(e, lam) else {
        return vec![f64::NAN; e.len()];
    };
    if path.scale == 0.0 {
        return vec![0.0; path.log_variances.len()];
    }
    let log_scale_square = 2.0 * path.scale.ln();
    path.log_variances
        .into_iter()
        .map(|log_variance| (log_scale_square + log_variance).exp())
        .collect()
}

fn relative_objective_gap(left: f64, right: f64) -> f64 {
    if left == right {
        return 0.0;
    }
    if !left.is_finite() || !right.is_finite() {
        return f64::INFINITY;
    }
    let scale = left.abs().max(right.abs());
    if scale == 0.0 {
        return 0.0;
    }
    let scaled_gap = (left / scale - right / scale).abs();
    scaled_gap * (scale / (1.0 + left.abs().min(right.abs())))
}

impl FitSeries for EwmaVol {
    type Fitted = FittedEwmaVol;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedEwmaVol>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.policy = self.optimizer.policy.clone();
        inspect_scale_invariant_univariate(&mut ctx, y);
        if y.is_empty() || y.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(ctx.finish_failure());
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        if !mean.is_finite() || e.as_slice().iter().any(|value| !value.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("EWMA demeaning produced a non-finite residual")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }

        let lam = if let Some(fixed) = self.lambda {
            if !fixed.is_finite() || !(fixed > 0.0 && fixed < 1.0) {
                ctx.push(
                    Issue::builder(IssueCode::InvalidParameter)
                        .message(format!("fixed EWMA λ={fixed} must be finite and in (0, 1)"))
                        .metric("lambda", fixed)
                        .build(),
                );
                return Err(ctx.finish_failure());
            }
            if e.max_abs() == 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::DegenerateDistribution)
                        .message("fixed-λ EWMA has zero conditional variance on a constant series")
                        .build(),
                );
            }
            fixed
        } else {
            let (lower, upper) = self.search_bounds;
            if !lower.is_finite()
                || !upper.is_finite()
                || !(0.0 < lower && lower < upper && upper < 1.0)
                || !self.search_initial.is_finite()
                || !(lower < self.search_initial && self.search_initial < upper)
                || self.search_grid_intervals < 3
                || !self.optimizer.policy.optimizer_objective_tol.is_finite()
                || self.optimizer.policy.optimizer_objective_tol <= 0.0
            {
                ctx.push(
                    Issue::builder(IssueCode::InvalidParameter)
                        .message(
                            "EWMA QMLE requires finite 0 < lower < initial < upper < 1, at least three grid intervals, and a positive finite objective tolerance",
                        )
                        .build(),
                );
                return Err(ctx.finish_failure());
            }
            if e.max_abs() == 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("EWMA λ is unidentified because every demeaned residual is zero")
                        .build(),
                );
                return Err(ctx.finish_failure());
            }

            let width = upper - lower;
            let step = width / self.search_grid_intervals as f64;
            let mut overflowed_objectives = 0usize;
            let mut grid_best_lambda = self.search_initial;
            let initial_objective = tracked_ewma_qml_objective(
                e.as_slice(),
                grid_best_lambda,
                &mut overflowed_objectives,
            );
            if initial_objective.is_nan() || initial_objective == f64::NEG_INFINITY {
                ctx.push(
                    Issue::builder(IssueCode::LossIsNan)
                        .message(format!(
                            "EWMA QMLE initial objective is NaN or negative infinity at λ={grid_best_lambda}"
                        ))
                        .metric("lambda", grid_best_lambda)
                        .build(),
                );
                return Err(ctx.finish_failure());
            }
            let mut grid_best_objective = f64::INFINITY;
            let mut best_interior = None;
            if initial_objective.is_finite() {
                grid_best_objective = initial_objective;
                best_interior = Some((self.search_initial, initial_objective));
            }
            let mut grid_min = f64::INFINITY;
            let mut grid_max = f64::NEG_INFINITY;
            let mut finite_grid_objectives = 0usize;
            let mut lower_objective = f64::INFINITY;
            let mut upper_objective = f64::INFINITY;
            for index in 0..=self.search_grid_intervals {
                let candidate = if index == self.search_grid_intervals {
                    upper
                } else {
                    lower + width * (index as f64 / self.search_grid_intervals as f64)
                };
                let candidate_objective =
                    tracked_ewma_qml_objective(e.as_slice(), candidate, &mut overflowed_objectives);
                if candidate_objective.is_nan() || candidate_objective == f64::NEG_INFINITY {
                    ctx.push(
                        Issue::builder(IssueCode::LossIsNan)
                            .message(format!(
                                "EWMA QMLE grid objective is NaN or negative infinity at λ={candidate}"
                            ))
                            .metric("lambda", candidate)
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                }
                if index == 0 {
                    lower_objective = candidate_objective;
                } else if index == self.search_grid_intervals {
                    upper_objective = candidate_objective;
                }
                if !candidate_objective.is_finite() {
                    continue;
                }
                finite_grid_objectives += 1;
                grid_min = grid_min.min(candidate_objective);
                grid_max = grid_max.max(candidate_objective);
                if candidate_objective < grid_best_objective {
                    grid_best_lambda = candidate;
                    grid_best_objective = candidate_objective;
                }
                if index > 0
                    && index < self.search_grid_intervals
                    && best_interior
                        .map(|(_, objective)| candidate_objective < objective)
                        .unwrap_or(true)
                {
                    best_interior = Some((candidate, candidate_objective));
                }
            }
            if !grid_best_objective.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NumericalOverflow)
                        .severity(Severity::Fatal)
                        .message(
                            "every sampled EWMA QMLE objective overflowed to positive infinity",
                        )
                        .metric("overflowed_objectives", overflowed_objectives as f64)
                        .build(),
                );
                return Err(ctx.finish_failure());
            }
            ctx.session.step(0, grid_best_objective, None);
            if initial_objective.is_finite()
                && finite_grid_objectives == self.search_grid_intervals + 1
                && relative_objective_gap(grid_min, grid_max)
                    <= self.optimizer.policy.optimizer_objective_tol
            {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message(
                            "EWMA λ is unidentified because the sampled QMLE objective is flat at the configured tolerance",
                        )
                        .build(),
                );
                return Err(ctx.finish_failure());
            }

            let mut best_interior = if let Some((seed, seed_objective)) = best_interior {
                let neighbor = if seed - step > lower {
                    seed - step
                } else {
                    seed + step
                };
                let Some(seed_unconstrained) = encode_open_interval(seed, lower, upper) else {
                    ctx.push(
                        Issue::builder(IssueCode::NonFiniteOutput)
                            .message("EWMA QMLE could not encode its interior grid seed")
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                };
                let Some(neighbor_unconstrained) = encode_open_interval(neighbor, lower, upper)
                else {
                    ctx.push(
                        Issue::builder(IssueCode::NonFiniteOutput)
                            .message("EWMA QMLE could not encode its neighboring grid seed")
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                };
                let simplex = [
                    Vector::from_slice(&[seed_unconstrained]),
                    Vector::from_slice(&[neighbor_unconstrained]),
                ];
                let optimized = match self.optimizer.minimize_nested(
                    &simplex,
                    |point| {
                        decode_open_interval(point[0], lower, upper)
                            .map(|candidate| {
                                tracked_ewma_qml_objective(
                                    e.as_slice(),
                                    candidate,
                                    &mut overflowed_objectives,
                                )
                            })
                            .unwrap_or(f64::NAN)
                    },
                    &ctx.session.child("qml"),
                ) {
                    Ok(qualified) => qualified,
                    Err(failure) => return Err(ctx.merge_failure(failure)),
                };
                let (optimization, optimizer_report) = optimized.into_parts();
                ctx.report.merge(optimizer_report);
                let Some(interior_lambda) =
                    decode_open_interval(optimization.point[0], lower, upper)
                else {
                    ctx.push(
                        Issue::builder(IssueCode::NonFiniteOutput)
                            .message("EWMA QMLE could not decode the optimizer result")
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                };
                let interior_objective = tracked_ewma_qml_objective(
                    e.as_slice(),
                    interior_lambda,
                    &mut overflowed_objectives,
                );
                if !interior_objective.is_finite() {
                    ctx.push(
                        Issue::builder(IssueCode::NonFiniteOutput)
                            .message("EWMA QMLE optimizer returned a non-finite best objective")
                            .metric("lambda", interior_lambda)
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                }
                if seed_objective < interior_objective {
                    (seed, seed_objective)
                } else {
                    (interior_lambda, interior_objective)
                }
            } else {
                (self.search_initial, f64::INFINITY)
            };
            if grid_best_lambda > lower
                && grid_best_lambda < upper
                && grid_best_objective < best_interior.1
            {
                best_interior = (grid_best_lambda, grid_best_objective);
            }
            let minimum = best_interior.1.min(lower_objective).min(upper_objective);
            let tolerance = self.optimizer.policy.optimizer_objective_tol;
            let lower_equivalent = relative_objective_gap(lower_objective, minimum) <= tolerance;
            let upper_equivalent = relative_objective_gap(upper_objective, minimum) <= tolerance;
            let lower_derivative =
                lower_equivalent.then(|| ewma_qml_derivative(e.as_slice(), lower));
            let upper_derivative =
                upper_equivalent.then(|| ewma_qml_derivative(e.as_slice(), upper));
            if lower_derivative.is_some_and(|derivative| !derivative.is_finite())
                || upper_derivative.is_some_and(|derivative| !derivative.is_finite())
            {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message(
                            "an objective-equivalent EWMA QMLE boundary derivative is non-finite",
                        )
                        .metric("lower_lambda", lower)
                        .metric("upper_lambda", upper)
                        .build(),
                );
                return Err(ctx.finish_failure());
            }
            if lower_equivalent && upper_equivalent {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message(
                            "EWMA λ is unidentified because both search bounds are objective-equivalent to the sampled minimum",
                        )
                        .metric(
                            "lower_relative_objective_gap",
                            relative_objective_gap(lower_objective, minimum),
                        )
                        .metric(
                            "upper_relative_objective_gap",
                            relative_objective_gap(upper_objective, minimum),
                        )
                        .metric("objective_tolerance", tolerance)
                        .metric(
                            "lower_objective_derivative",
                            lower_derivative.expect("equivalent lower bound has a derivative"),
                        )
                        .metric(
                            "upper_objective_derivative",
                            upper_derivative.expect("equivalent upper bound has a derivative"),
                        )
                        .build(),
                );
                return Err(ctx.finish_failure());
            }
            let lower_parameter_gap = (best_interior.0 - lower).abs() / (1.0 + lower.abs());
            let upper_parameter_gap = (upper - best_interior.0).abs() / (1.0 + upper.abs());
            let active_boundary = if lower_equivalent
                && (lower_objective <= best_interior.1
                    || lower_derivative.is_some_and(|derivative| derivative >= 0.0))
                && lower_objective <= upper_objective
            {
                Some((
                    "lower",
                    lower,
                    relative_objective_gap(lower_objective, minimum),
                    lower_parameter_gap,
                    lower_derivative.expect("equivalent lower bound has a derivative"),
                ))
            } else if upper_equivalent
                && (upper_objective <= best_interior.1
                    || upper_derivative.is_some_and(|derivative| derivative <= 0.0))
                && upper_objective <= lower_objective
            {
                Some((
                    "upper",
                    upper,
                    relative_objective_gap(upper_objective, minimum),
                    upper_parameter_gap,
                    upper_derivative.expect("equivalent upper bound has a derivative"),
                ))
            } else {
                None
            };
            let selected_lambda =
                if let Some((name, boundary, objective_gap, parameter_gap, objective_derivative)) =
                    active_boundary
                {
                    ctx.push(
                        Issue::builder(IssueCode::ParameterAtBoundary)
                            .message(format!(
                                "EWMA QMLE selected the active {name} search bound λ={boundary}"
                            ))
                            .metric("lambda", boundary)
                            .metric("lower", lower)
                            .metric("upper", upper)
                            .metric("relative_objective_gap", objective_gap)
                            .metric("objective_tolerance", tolerance)
                            .metric("objective_derivative", objective_derivative)
                            .metric("relative_lambda_gap", parameter_gap)
                            .metric("best_sampled_lambda", best_interior.0)
                            .build(),
                    );
                    boundary
                } else {
                    best_interior.0
                };
            if overflowed_objectives > 0 {
                ctx.push(
                    Issue::builder(IssueCode::NumericalOverflow)
                        .severity(Severity::Advisory)
                        .message(format!(
                            "{overflowed_objectives} dominated EWMA QMLE candidate objectives overflowed to positive infinity"
                        ))
                        .metric("overflowed_objectives", overflowed_objectives as f64)
                        .metric("selected_lambda", selected_lambda)
                        .build(),
                );
            }
            selected_lambda
        };

        let sigma2 = ewma_sigma2(e.as_slice(), lam);
        if sigma2.iter().any(|value| !value.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("EWMA conditional-variance recurrence produced a non-finite value")
                    .metric("lambda", lam)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let underflowed = sigma2.iter().filter(|value| **value == 0.0).count();
        if e.max_abs() > 0.0 && underflowed > 0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalUnderflow)
                    .message(format!(
                        "{underflowed} EWMA conditional variances underflowed to zero"
                    ))
                    .metric("underflowed_variances", underflowed as f64)
                    .compromise(NumericalCompromise::new(
                        "return every positive EWMA conditional variance in binary64",
                        "return zero where the physical-scale variance is below binary64 range",
                        "the normalized recurrence is positive, but rescaling its variance underflowed",
                        "zero entries are representational underflow, not zero process variance",
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedEwmaVol {
            lambda: lam,
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_qmle_matches_independent_decimal_bounded_oracle() {
        let y = Vector::from_slice(&[
            8.0, -8.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.25, -0.25, -2.0, 2.0,
        ]);
        let fitted = EwmaVol::new()
            .fit_series(&y, &Session::new("ewma", "oracle"))
            .expect("interior EWMA optimum");
        let expected_lambda = 0.614_948_768_511_978_5;
        let expected_nll = 18.650_768_452_114_544;
        let lambda_error = (fitted.value.lambda - expected_lambda).abs();
        let actual_nll = ewma_nll(fitted.value.resid.as_slice(), fitted.value.lambda);
        // Measured λ error 7.548e-10 on 2026-09-02; tolerance is about 4x.
        assert!(lambda_error <= 3.1e-9, "λ error {lambda_error:e}");
        // Decimal oracle versus fitted binary64 measured 3.553e-15; tolerance is about 4x.
        assert!((actual_nll - expected_nll).abs() <= 1.4e-14);
        assert!(!fitted.report.contains(IssueCode::ParameterAtBoundary));

        for index in 0..=64 {
            let candidate = 0.5 + (0.999 - 0.5) * (index as f64 / 64.0);
            assert!(actual_nll <= ewma_nll(fitted.value.resid.as_slice(), candidate));
        }
    }

    #[test]
    fn ewma_decimal_golden_replay() {
        let raw = include_str!("../../../golden/ewma_qml.json");
        let payload: serde_json::Value = serde_json::from_str(raw).expect("golden/ewma_qml.json");
        let cases = payload["cases"].as_array().expect("cases");
        assert_eq!(cases.len(), 5, "oracle generator documents five cases");
        let parse = |value: &serde_json::Value| {
            value
                .as_str()
                .expect("decimal string")
                .parse::<f64>()
                .expect("finite binary64 value")
        };
        let mut maximum_lambda_error = 0.0_f64;
        let mut maximum_nll_error = 0.0_f64;
        let mut maximum_sigma2_error = 0.0_f64;

        for case in cases {
            let name = case["name"].as_str().expect("case name");
            let observations = case["input"]["observations"]
                .as_array()
                .expect("observations")
                .iter()
                .map(&parse)
                .collect::<Vec<_>>();
            let bounds = case["input"]["search_bounds"]
                .as_array()
                .expect("search bounds");
            let lower = parse(&bounds[0]);
            let upper = parse(&bounds[1]);
            let mut model = EwmaVol::new();
            model.search_bounds = (lower, upper);
            model.search_initial = if lower < 0.94 && 0.94 < upper {
                0.94
            } else {
                lower + (upper - lower) * 0.5
            };
            let result = model.fit_series(
                &Vector::from_slice(&observations),
                &Session::new("ewma", name),
            );
            let outcome = case["expected"]["outcome"].as_str().expect("outcome");
            if outcome == "failure" {
                let failure = result.expect_err("oracle expects failure");
                assert_eq!(failure.primary.code, IssueCode::UnidentifiedModel);
                continue;
            }

            let fitted = result.expect("oracle expects a fitted EWMA");
            let expected_lambda = parse(&case["expected"]["lambda"]);
            let lambda_error = (fitted.value.lambda - expected_lambda).abs();
            maximum_lambda_error = maximum_lambda_error.max(lambda_error);
            let selection = case["expected"]["selection"].as_str().expect("selection");
            if selection == "interior" {
                // Measured max |Δλ|=7.548e-10 on 2026-09-02; tolerance is about 4x.
                assert!(lambda_error <= 3.1e-9, "{name}: λ error {lambda_error:e}");
                assert!(!fitted.report.contains(IssueCode::ParameterAtBoundary));
            } else {
                assert_eq!(fitted.value.lambda, expected_lambda, "{name}");
                assert!(fitted.report.contains(IssueCode::ParameterAtBoundary));
            }

            let expected_nll = parse(&case["expected"]["nll_without_gaussian_constant"]);
            let actual_nll = ewma_nll(fitted.value.resid.as_slice(), fitted.value.lambda);
            maximum_nll_error = maximum_nll_error.max((actual_nll - expected_nll).abs());
            let expected_sigma2 = case["expected"]["sigma2"].as_array().expect("sigma2");
            assert_eq!(fitted.value.sigma2.len(), expected_sigma2.len());
            for (actual, expected) in fitted.value.sigma2.as_slice().iter().zip(expected_sigma2) {
                maximum_sigma2_error = maximum_sigma2_error.max((*actual - parse(expected)).abs());
            }
        }

        eprintln!(
            "EWMA Decimal golden max errors: λ={maximum_lambda_error:e}, NLL={maximum_nll_error:e}, sigma2={maximum_sigma2_error:e}"
        );
        // Measured maxima on 2026-09-02: NLL 3.553e-15, sigma² 4.868e-8.
        // Both tolerances are approximately four times the observed error (R9).
        assert!(maximum_nll_error <= 1.4e-14);
        assert!(maximum_sigma2_error <= 2.0e-7);
    }

    #[test]
    fn ewma_qmle_finds_lower_bound_despite_competing_basin() {
        let y = Vector::from_slice(&[8.0, -8.0, 4.0, -4.0, 2.0, -2.0, 1.0, -1.0]);
        let fitted = EwmaVol::new()
            .fit_series(&y, &Session::new("ewma", "boundary"))
            .expect("boundary EWMA optimum");
        assert_eq!(fitted.value.lambda, 0.5);
        let actual_nll = ewma_nll(fitted.value.resid.as_slice(), fitted.value.lambda);
        // Decimal oracle versus f64 replay measured below 1.90e-15; tolerance is about 4x.
        assert!((actual_nll - 15.064_887_636_096_91).abs() <= 8.0e-15);
        assert!(fitted.report.contains(IssueCode::ParameterAtBoundary));
        assert!(fitted.is_compromised());
    }

    #[test]
    fn ewma_constant_series_exposes_unidentified_decay() {
        let y = Vector::from_slice(&[7.0, 7.0, 7.0, 7.0]);
        let failure = EwmaVol::new()
            .fit_series(&y, &Session::new("ewma", "constant"))
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::UnidentifiedModel);
        assert!(failure.report.contains(IssueCode::UnidentifiedModel));
    }

    #[test]
    fn ewma_fixed_lambda_is_exact_and_invalid_values_fail() {
        let y = Vector::from_slice(&[3.0, -1.0, 0.0, -2.0]);
        let mut fixed = EwmaVol::new();
        fixed.lambda = Some(0.9999);
        let fitted = fixed
            .fit_series(&y, &Session::new("ewma", "fixed"))
            .expect("valid fixed lambda");
        assert_eq!(fitted.value.lambda, 0.9999);
        assert!((fitted.value.sigma2[0] - 3.5).abs() <= 16.0 * f64::EPSILON);
        let expected_second = 0.9999 * 3.5 + (1.0 - 0.9999) * 9.0;
        // Measured error was below 2 ulps on 2026-09-02; eight ulps allow libm variation.
        assert!((fitted.value.sigma2[1] - expected_second).abs() <= 8.0 * f64::EPSILON);

        for invalid in [0.0, -0.0, 1.0, -0.2, f64::NAN, f64::INFINITY] {
            let mut model = EwmaVol::new();
            model.lambda = Some(invalid);
            let failure = model
                .fit_series(&y, &Session::new("ewma", "invalid-fixed"))
                .unwrap_err();
            assert_eq!(failure.primary.code, IssueCode::InvalidParameter);
        }
    }

    #[test]
    fn ewma_qmle_honors_custom_bounds_with_an_interior_optimum() {
        let interior = Vector::from_slice(&[
            8.0, -8.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.25, -0.25, -2.0, 2.0,
        ]);
        let mut custom = EwmaVol::new();
        custom.search_bounds = (0.55, 0.70);
        custom.search_initial = 0.62;
        let custom = custom
            .fit_series(&interior, &Session::new("ewma", "custom-bounds"))
            .expect("interior optimum in custom bounds");
        let expected_lambda = 0.614_948_768_511_978_5;
        // The default-bound run measured 7.548e-10 error; this allows about 4x.
        assert!((custom.value.lambda - expected_lambda).abs() <= 3.1e-9);
        assert!(!custom.report.contains(IssueCode::ParameterAtBoundary));
    }

    #[test]
    fn ewma_qmle_is_scale_invariant_without_a_variance_floor() {
        let values = [
            8.0, -8.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.25, -0.25, -2.0, 2.0,
        ];
        let mut base_model = EwmaVol::new();
        base_model.optimizer.policy.abort_at = Severity::Warning;
        let base = base_model
            .fit_series(
                &Vector::from_slice(&values),
                &Session::new("ewma", "base-scale"),
            )
            .expect("base-scale EWMA");
        let factor = 1e-100;
        let scaled_values = values.map(|value| value * factor);
        let mut scaled_model = EwmaVol::new();
        scaled_model.optimizer.policy.abort_at = Severity::Warning;
        let scaled = scaled_model
            .fit_series(
                &Vector::from_slice(&scaled_values),
                &Session::new("ewma", "small-scale"),
            )
            .expect("small-scale EWMA");
        assert!(!base.report.contains(IssueCode::ConstantFeature));
        assert!(!scaled.report.contains(IssueCode::ConstantFeature));
        let mut maximum_relative_variance_error = 0.0_f64;
        for (base_variance, scaled_variance) in base
            .value
            .sigma2
            .as_slice()
            .iter()
            .zip(scaled.value.sigma2.as_slice())
        {
            let expected = base_variance * factor * factor;
            let relative_error = (scaled_variance / expected - 1.0).abs();
            maximum_relative_variance_error = maximum_relative_variance_error.max(relative_error);
        }
        eprintln!(
            "EWMA scale-invariance: base λ={}, scaled λ={}, |Δλ|={:e}, max relative sigma² error={maximum_relative_variance_error:e}",
            base.value.lambda,
            scaled.value.lambda,
            (scaled.value.lambda - base.value.lambda).abs()
        );
        // Log-difference normalization measured |Δλ|=1.882e-9 on 2026-09-02;
        // the tolerance is about four times that optimizer-level error (R9).
        assert!((scaled.value.lambda - base.value.lambda).abs() <= 7.6e-9);
        // The induced maximum relative sigma² error measured 1.995e-8; this is 4x.
        assert!(maximum_relative_variance_error <= 8.0e-8);
    }

    #[test]
    fn ewma_reports_physical_variance_underflow() {
        let factor = 1e-200;
        let y = Vector::from_iter(
            [8.0, -8.0, 1.0, -1.0, 0.5, -0.5]
                .into_iter()
                .map(|value| value * factor),
        );
        let fitted = EwmaVol::riskmetrics()
            .fit_series(&y, &Session::new("ewma", "variance-underflow"))
            .expect("underflow is returned only with an explicit quality report");
        assert!(fitted.report.contains(IssueCode::NumericalUnderflow));
        assert!(fitted.is_compromised());
        assert!(fitted
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|variance| *variance == 0.0));
    }

    #[test]
    fn ewma_log_variance_survives_long_zero_runs() {
        let mut values = vec![0.0; 1_200];
        values[0] = 1.0;
        values[1] = -1.0;
        let fitted = EwmaVol::new()
            .fit_series(
                &Vector::from_iter(values),
                &Session::new("ewma", "long-zero-run"),
            )
            .expect("log-variance recurrence must remain finite");
        assert_eq!(fitted.value.lambda, 0.5);
        assert!(fitted.report.contains(IssueCode::ParameterAtBoundary));
        assert!(fitted.report.contains(IssueCode::NumericalUnderflow));
        assert!(fitted
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|variance| variance.is_finite()));
        assert!(fitted
            .value
            .sigma2
            .as_slice()
            .iter()
            .any(|variance| *variance == 0.0));
    }

    #[test]
    fn ewma_qmle_skips_positive_infinity_candidates_with_a_finite_optimum() {
        let payload: serde_json::Value =
            serde_json::from_str(include_str!("../../../golden/ewma_qml.json"))
                .expect("golden/ewma_qml.json");
        let case = &payload["extended_real_case"];
        let parse = |value: &serde_json::Value| {
            value
                .as_str()
                .expect("decimal string")
                .parse::<f64>()
                .expect("binary64 value")
        };
        let recipe = &case["input_recipe"];
        let mut values = recipe["prefix"]
            .as_array()
            .expect("prefix")
            .iter()
            .map(&parse)
            .collect::<Vec<_>>();
        values.extend(std::iter::repeat_n(
            0.0,
            recipe["zero_count"].as_u64().expect("zero count") as usize,
        ));
        values.extend(
            recipe["suffix"]
                .as_array()
                .expect("suffix")
                .iter()
                .map(&parse),
        );
        assert_eq!(
            values.len(),
            recipe["expanded_length"].as_u64().expect("length") as usize
        );
        let bounds = recipe["search_bounds"].as_array().expect("bounds");
        let lower = parse(&bounds[0]);
        let upper = parse(&bounds[1]);
        assert_eq!(
            case["binary64_endpoint_statuses"]["lower_bound"]["nll_status"],
            "positive_infinity"
        );
        assert_eq!(ewma_qml_objective(&values, lower), f64::INFINITY);
        assert!(ewma_qml_objective(&values, upper).is_finite());

        let mut model = EwmaVol::new();
        model.search_bounds = (lower, upper);
        let fitted = model
            .fit_series(
                &Vector::from_iter(values),
                &Session::new("ewma", "late-shock-overflow"),
            )
            .expect("overflowed dominated candidates must not hide a finite optimum");
        let upper_derivative = ewma_qml_derivative(fitted.value.resid.as_slice(), upper);
        let actual_nll = ewma_nll(fitted.value.resid.as_slice(), upper);
        let expected_nll = parse(&case["decimal_evidence"]["endpoint_nlls"]["upper_bound"]);
        let expected_derivative = parse(&case["decimal_evidence"]["derivative_at_upper_bound"]);
        let nll_error = (actual_nll - expected_nll).abs();
        let derivative_error = (upper_derivative - expected_derivative).abs();
        eprintln!(
            "late-shock EWMA selected lambda={}, NLL error={nll_error:e}, derivative error={derivative_error:e}",
            fitted.value.lambda
        );
        assert_eq!(fitted.value.lambda, upper);
        assert!(ewma_qml_objective(fitted.value.resid.as_slice(), fitted.value.lambda).is_finite());
        // Measured Decimal errors on 2026-09-02: NLL 3.956e-11 and derivative 5.490e-8.
        // Tolerances are approximately four times those observed errors (R9).
        assert!(nll_error <= 1.6e-10);
        assert!(derivative_error <= 2.2e-7);
        assert!(fitted.report.contains(IssueCode::NumericalOverflow));
        assert!(fitted.report.contains(IssueCode::ParameterAtBoundary));
        assert!(!fitted.report.contains(IssueCode::LossIsNan));
        assert!(!fitted.report.contains(IssueCode::UnidentifiedModel));
        assert!(upper_derivative < 0.0);
        assert!(fitted
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|variance| variance.is_finite()));
    }

    #[test]
    fn ewma_inactive_infinite_boundary_derivative_cannot_veto_a_finite_fit() {
        let mut values = Vec::with_capacity(2_306);
        values.extend([1.0, -1.0]);
        values.extend(std::iter::repeat_n(0.0, 2_300));
        let tiny = (-400.0_f64).exp();
        values.extend([tiny, -tiny, 1.0, -1.0]);
        assert_eq!(ewma_qml_objective(&values, 0.5), f64::INFINITY);
        assert!(ewma_qml_derivative(&values, 0.5).is_nan());
        assert!(ewma_qml_objective(&values, 0.999).is_finite());

        let fitted = EwmaVol::new()
            .fit_series(
                &Vector::from_iter(values),
                &Session::new("ewma", "inactive-infinite-boundary"),
            )
            .expect("an inactive infinite endpoint cannot veto a finite QMLE fit");
        assert_eq!(fitted.value.lambda, 0.999);
        assert!(fitted.report.contains(IssueCode::NumericalOverflow));
        assert!(fitted.report.contains(IssueCode::ParameterAtBoundary));
        assert!(!fitted.report.contains(IssueCode::LossIsNan));
    }

    #[test]
    fn ewma_log_objective_preserves_residuals_across_extreme_dynamic_range() {
        let mut values = Vec::with_capacity(3_002);
        values.extend([1e150, -1e150]);
        values.extend((0..3_000).map(|index| if index % 2 == 0 { 1e-200 } else { -1e-200 }));
        assert_eq!(
            1e-200 / 1e150,
            0.0,
            "the direct normalized quotient underflows"
        );

        let objective = ewma_qml_objective(&values, 0.5);
        let mut truncated = values.clone();
        truncated[2..].fill(0.0);
        let truncated_objective = ewma_qml_objective(&truncated, 0.5);
        let retained_effect = (objective - truncated_objective).abs();
        eprintln!(
            "extreme-range EWMA objective={objective:e}, zero-truncated={truncated_objective:e}, effect={retained_effect:e}"
        );
        assert!(objective.is_finite());
        assert!(truncated_objective.is_finite());
        assert!(retained_effect > 1.0);

        let mut fixed = EwmaVol::new();
        fixed.lambda = Some(0.5);
        let fitted = fixed
            .fit_series(
                &Vector::from_iter(values),
                &Session::new("ewma", "extreme-dynamic-range"),
            )
            .expect("the log recurrence must retain every representable residual");
        assert!(fitted
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|variance| variance.is_finite()));
        assert!(fitted.report.contains(IssueCode::NumericalUnderflow));
    }

    #[test]
    fn ewma_near_bound_interior_optimum_is_not_snapped_by_solver_tolerance() {
        let y = Vector::from_slice(&[
            8.0, -8.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.25, -0.25, -2.0, 2.0,
        ]);
        let lower = 0.6149;
        let mut model = EwmaVol::new();
        model.search_bounds = (lower, 0.999);
        model.optimizer.policy.optimizer_objective_tol = 1e-7;
        model.optimizer.policy.optimizer_parameter_tol = 1e-4;
        let fitted = model
            .fit_series(&y, &Session::new("ewma", "near-bound-interior"))
            .expect("the known interior optimum remains inside the custom interval");
        let expected = 0.614_948_768_511_978_5;
        let error = (fitted.value.lambda - expected).abs();
        eprintln!(
            "near-bound EWMA selected lambda={}, error={error:e}",
            fitted.value.lambda
        );
        assert!(fitted.value.lambda > lower);
        assert!(!fitted.report.contains(IssueCode::ParameterAtBoundary));
        // Measured 6.958e-10 on 2026-09-02; tolerance is about four times that.
        assert!(error <= 2.8e-9);
    }

    #[test]
    fn relative_objective_gap_orders_positive_infinity_explicitly() {
        assert_eq!(relative_objective_gap(f64::INFINITY, f64::INFINITY), 0.0);
        assert_eq!(relative_objective_gap(f64::INFINITY, 1.0), f64::INFINITY);
        assert_eq!(relative_objective_gap(1.0, f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn ewma_analytic_objective_derivative_matches_centered_difference() {
        let residuals = [
            8.0, -8.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.25, -0.25, -2.0, 2.0,
        ];
        let lambda = 0.7;
        let step = f64::EPSILON.cbrt();
        let numerical = (ewma_qml_objective(&residuals, lambda + step)
            - ewma_qml_objective(&residuals, lambda - step))
            / (2.0 * step);
        let analytic = ewma_qml_derivative(&residuals, lambda);
        let error = (analytic - numerical).abs();
        eprintln!("EWMA derivative analytic={analytic:e}, centered={numerical:e}, error={error:e}");
        // Measured 5.210e-9 on 2026-09-02; tolerance is about four times that.
        assert!(error <= 2.1e-8);
    }

    #[test]
    fn ewma_qmle_rejects_invalid_search_configuration() {
        let y = Vector::from_slice(&[2.0, -1.0, 0.0, -1.0]);
        let mut model = EwmaVol::new();
        model.search_grid_intervals = 2;
        let failure = model
            .fit_series(&y, &Session::new("ewma", "invalid-search"))
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::InvalidParameter);
    }

    #[test]
    fn ewma_preserves_nested_optimizer_reports_and_strict_failures() {
        let y = Vector::from_slice(&[
            8.0, -8.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.25, -0.25, -2.0, 2.0,
        ]);

        let mut capped = EwmaVol::new();
        capped.optimizer.max_iterations = 1;
        let capped_session = Session::new("ewma", "capped");
        let capped = capped
            .fit_series(&y, &capped_session)
            .expect("iteration cap returns a qualified EWMA fit");
        assert!(capped.report.contains(IssueCode::MaxIterReached));
        assert!(capped.is_compromised());
        assert_eq!(
            capped_session
                .ledger()
                .events()
                .iter()
                .filter(|event| {
                    event.kind == ojizou_san::EventKind::QualityWarning
                        && event.issue.as_ref().map(|issue| issue.code)
                            == Some(IssueCode::MaxIterReached)
                })
                .count(),
            1,
            "the merged optimizer issue must be ingested exactly once"
        );
        assert_eq!(
            capped_session
                .ledger()
                .of_kind(ojizou_san::EventKind::FitFinished)
                .len(),
            1,
            "only the enclosing EWMA computation records a terminal event"
        );

        let mut strict = EwmaVol::new();
        strict.optimizer.max_iterations = 1;
        strict.optimizer.policy.abort_at = Severity::Warning;
        let strict_session = Session::new("ewma", "strict-capped");
        let failure = strict.fit_series(&y, &strict_session).unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::MaxIterReached);
        assert!(failure.report.contains(IssueCode::MaxIterReached));
        assert_eq!(
            strict_session
                .ledger()
                .events()
                .iter()
                .filter(|event| {
                    event.kind == ojizou_san::EventKind::QualityWarning
                        && event.issue.as_ref().map(|issue| issue.code)
                            == Some(IssueCode::MaxIterReached)
                })
                .count(),
            1
        );
        assert_eq!(
            strict_session
                .ledger()
                .of_kind(ojizou_san::EventKind::FitFailed)
                .len(),
            1
        );
    }

    #[test]
    fn ewma_optimizer_policy_also_governs_boundary_diagnostics() {
        let y = Vector::from_slice(&[8.0, -8.0, 4.0, -4.0, 2.0, -2.0, 1.0, -1.0]);
        let mut strict = EwmaVol::new();
        strict.optimizer.policy.abort_at = Severity::Warning;
        let failure = strict
            .fit_series(&y, &Session::new("ewma", "strict-boundary"))
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::ParameterAtBoundary);
        assert!(failure.report.contains(IssueCode::ParameterAtBoundary));

        let mut relaxed = EwmaVol::new();
        relaxed.optimizer.policy.abort_at = Severity::Fatal;
        let unidentified = relaxed
            .fit_series(
                &Vector::from_slice(&[7.0, 7.0, 7.0, 7.0]),
                &Session::new("ewma", "relaxed-unidentified"),
            )
            .expect_err("an unidentified model cannot manufacture a lambda");
        assert_eq!(unidentified.primary.code, IssueCode::UnidentifiedModel);
    }

    #[test]
    fn fixed_ewma_reports_zero_variance_as_degenerate() {
        let fitted = EwmaVol::riskmetrics()
            .fit_series(
                &Vector::from_slice(&[7.0, 7.0, 7.0, 7.0]),
                &Session::new("ewma", "fixed-constant"),
            )
            .expect("fixed lambda remains defined on a constant series");
        assert!(fitted
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|variance| *variance == 0.0));
        assert!(fitted.report.contains(IssueCode::DegenerateDistribution));
    }
}
