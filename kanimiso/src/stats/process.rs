use crate::context::FitCtx;
use crate::data::Vector;
use ojizou_san::Session;
use signlred::{
    scan_finite, Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result,
    Severity,
};

/// Gaussian process with exponential covariance
/// (statsmodels `ProcessMLE` lite).
///
/// Covariance parameters are not identification `p`.
#[derive(Clone, Debug)]
pub struct ProcessMleFit {
    /// Estimated mean.
    pub mean: f64,
    /// Marginal variance.
    pub sigma2: f64,
    /// Exponential range \(\rho\).
    pub range: f64,
}

/// Profile a mean-plus-exponential-covariance process on irregular time.
///
/// `t` is the observation time; `y` is the response. Both vectors must have
/// equal length, contain only finite values, and timestamps must be distinct.
/// A deterministic five-point grid over \(\rho\) is scored by the exact
/// Gaussian profile likelihood, including \(\log\det C_\rho\). The profiled
/// mean and marginal variance are exact at every grid point; only the range
/// search is discretized.
pub fn process_mle(t: &Vector, y: &Vector, session: &Session) -> Result<Qualified<ProcessMleFit>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if t.len() != y.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "process_mle needs paired timestamps and responses, got {} and {}",
                    t.len(),
                    y.len()
                ))
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    let n = t.len();
    ctx.report.set_sample_shape(n, 1);
    ctx.report.set_n_parameters(3);
    if n == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("process_mle needs at least three timestamp-response pairs")
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    if n < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message(format!(
                    "process_mle needs at least three observations to identify mean, variance, and range; got {n}"
                ))
                .metric("n", n as f64)
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    let mut input_is_finite = true;
    if let Some(issue) = scan_finite(t.as_slice()).to_issue("process_mle timestamps") {
        ctx.push(issue);
        input_is_finite = false;
    }
    if let Some(issue) = scan_finite(y.as_slice()).to_issue("process_mle responses") {
        ctx.push(issue);
        input_is_finite = false;
    }
    if !input_is_finite {
        return Err(ctx.finish_failure());
    }

    let mut observations: Vec<(f64, f64)> = t
        .as_slice()
        .iter()
        .copied()
        .zip(y.as_slice().iter().copied())
        .collect();
    observations.sort_by(|left, right| left.0.total_cmp(&right.0));
    if let Some(duplicate) = observations
        .windows(2)
        .find(|pair| pair[0].0 == pair[1].0)
        .map(|pair| pair[0].0)
    {
        ctx.push(
            Issue::builder(IssueCode::DuplicateIndex)
                .severity(Severity::Error)
                .message(format!(
                    "process_mle timestamp {duplicate} occurs more than once; the zero-nugget exponential covariance is singular"
                ))
                .metric("duplicate_timestamp", duplicate)
                .meaninglessness(Meaninglessness::vacuous(
                    "exponential-process likelihood",
                    "two noiseless observations at one process location make the covariance singular",
                    "aggregate duplicate timestamps or fit an explicit measurement-noise model",
                ))
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    let times: Vec<f64> = observations.iter().map(|pair| pair.0).collect();
    let responses: Vec<f64> = observations.iter().map(|pair| pair.1).collect();
    if responses.iter().all(|value| *value == responses[0]) {
        ctx.push(
            Issue::builder(IssueCode::ConstantTarget)
                .message("process_mle marginal variance is zero for an exactly constant response")
                .meaninglessness(Meaninglessness::vacuous(
                    "exponential-process range",
                    "a zero-variance process has no identifiable correlation range",
                    "supply a response with observed variation",
                ))
                .build(),
        );
        return Err(ctx.finish_failure());
    }

    let mut gaps = Vec::with_capacity(n - 1);
    for pair in times.windows(2) {
        let gap = pair[1] - pair[0];
        if !gap.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message(
                        "process_mle timestamp subtraction overflowed while forming an adjacent gap",
                    )
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        gaps.push(gap);
    }
    let mut sorted_gaps = gaps.clone();
    sorted_gaps.sort_by(f64::total_cmp);
    let median_gap = sorted_gaps[sorted_gaps.len() / 2];
    let range_multipliers = [1.0_f64, 2.0, 4.0, 8.0, 16.0];
    let mut ranges = [0.0_f64; 5];
    for (range, multiplier) in ranges.iter_mut().zip(range_multipliers) {
        *range = median_gap * multiplier;
        if !range.is_finite() || *range <= 0.0 {
            let code = if *range == 0.0 {
                IssueCode::NumericalUnderflow
            } else {
                IssueCode::NumericalOverflow
            };
            ctx.push(
                Issue::builder(code)
                    .severity(Severity::Error)
                    .message(format!(
                        "process_mle range grid cannot represent median_gap={median_gap} times multiplier={multiplier}"
                    ))
                    .metric("median_gap", median_gap)
                    .metric("range_multiplier", multiplier)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
    }

    let response_scale = responses
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if response_scale == 0.0 {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("process_mle response scale is zero")
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    let normalized_response: Vec<f64> = responses
        .iter()
        .map(|value| value / response_scale)
        .collect();

    let mut best: Option<(usize, f64, ProcessProfile)> = None;
    for (grid_index, range) in ranges.iter().copied().enumerate() {
        let profile = match exponential_process_profile(
            &gaps,
            &normalized_response,
            range,
            ctx.policy.underflow_guard,
        ) {
            Ok(profile) => profile,
            Err(mut issue) => {
                issue.message = format!(
                    "process_mle range-grid trial {grid_index} at rho={range} failed: {}",
                    issue.message
                );
                issue.metrics.push(("range".into(), range));
                ctx.push(issue);
                return Err(ctx.finish_failure());
            }
        };
        if best
            .as_ref()
            .is_none_or(|(_, objective, _)| profile.objective < *objective)
        {
            best = Some((grid_index, profile.objective, profile));
        }
    }
    let (best_index, _, profile) = best.expect("the non-empty validated range grid was evaluated");
    let mean = profile.mean * response_scale;
    let standard_deviation = profile.sigma2.sqrt() * response_scale;
    let sigma2 = standard_deviation * standard_deviation;
    if !mean.is_finite() || !sigma2.is_finite() || sigma2 <= 0.0 {
        let code = if sigma2 == 0.0 {
            IssueCode::NumericalUnderflow
        } else {
            IssueCode::NumericalOverflow
        };
        ctx.push(
            Issue::builder(code)
                .severity(Severity::Error)
                .message(format!(
                    "process_mle could not rescale mean={mean} and variance={sigma2} to response units"
                ))
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    if best_index == 0 || best_index + 1 == ranges.len() {
        ctx.push(
            Issue::builder(IssueCode::ParameterAtBoundary)
                .message(format!(
                    "process_mle selected endpoint {} of the five-point range grid",
                    best_index + 1
                ))
                .metric("range", ranges[best_index])
                .build(),
        );
    }
    ctx.push(
        Issue::builder(IssueCode::DidNotConverge)
            .severity(Severity::Advisory)
            .message(
                "process_mle profiles the exact Gaussian likelihood on a deterministic five-point range grid rather than continuously optimizing the range",
            )
            .compromise(NumericalCompromise::new(
                "continuous maximization of the Gaussian exponential-covariance likelihood over the positive range",
                "exact profiled mean and variance at median-gap range multiples 1, 2, 4, 8, and 16",
                "the range estimate is quantized by the grid; no nugget or variance floor is injected",
                "treat the reported range as a grid-resolution estimate",
            ))
            .build(),
    );
    ctx.finish(ProcessMleFit {
        mean,
        sigma2,
        range: ranges[best_index],
    })
}

#[derive(Clone, Copy, Debug)]
struct ProcessProfile {
    mean: f64,
    sigma2: f64,
    objective: f64,
}

/// Exact OU innovation factorization of an exponential-covariance likelihood.
///
/// For sorted observations, `phi_i = exp(-gap_i / range)` and the covariance
/// determinant is `product_i (1 - phi_i^2)`. This is algebraically identical
/// to dense GLS, while avoiding a second implementation of a generic matrix
/// inverse and remaining stable for `phi_i` close to one via `exp_m1`.
fn exponential_process_profile(
    gaps: &[f64],
    response: &[f64],
    range: f64,
    underflow_guard: f64,
) -> std::result::Result<ProcessProfile, Issue> {
    if response.len() != gaps.len() + 1
        || response.is_empty()
        || !range.is_finite()
        || range <= 0.0
        || !underflow_guard.is_finite()
        || underflow_guard <= 0.0
    {
        return Err(Issue::builder(IssueCode::InvalidParameter)
            .message(
                "exponential-process profile received inconsistent dimensions, range, or underflow guard",
            )
            .build());
    }
    let mut mean_weight = 1.0_f64;
    let mut mean_rhs = response[0];
    let mut log_determinant = 0.0_f64;
    let mut correlations = Vec::with_capacity(gaps.len());
    let mut innovation_variances = Vec::with_capacity(gaps.len());
    for (index, gap) in gaps.iter().copied().enumerate() {
        let scaled_gap = gap / range;
        let correlation = (-scaled_gap).exp();
        let one_minus_correlation = -(-scaled_gap).exp_m1();
        let innovation_variance = one_minus_correlation * (1.0 + correlation);
        if !scaled_gap.is_finite() {
            return Err(Issue::builder(IssueCode::NumericalOverflow)
                .message(format!(
                    "scaled exponential-process gap at index {index} is not representable"
                ))
                .metric("gap", gap)
                .metric("range", range)
                .build());
        }
        if scaled_gap <= 0.0 || !innovation_variance.is_finite() || innovation_variance <= 0.0 {
            return Err(
                Issue::builder(IssueCode::NearSingular)
                    .message(format!(
                        "adjacent exponential correlation at gap index {index} is singular at working precision"
                    ))
                    .metric("gap", gap)
                    .metric("scaled_gap", scaled_gap)
                    .build(),
            );
        }
        if correlation == 0.0 {
            return Err(Issue::builder(IssueCode::NumericalUnderflow)
                .message(format!(
                    "exponential correlation at gap index {index} underflowed to zero"
                ))
                .metric("gap", gap)
                .metric("range", range)
                .build());
        }
        if innovation_variance < underflow_guard {
            return Err(Issue::builder(IssueCode::NearSingular)
                .message(format!(
                    "exponential innovation variance at gap index {index} is below Policy::underflow_guard"
                ))
                .metric("innovation_variance", innovation_variance)
                .metric("underflow_guard", underflow_guard)
                .build());
        }
        let innovation =
            response[index + 1] - response[index] + one_minus_correlation * response[index];
        mean_weight += one_minus_correlation / (1.0 + correlation);
        mean_rhs += innovation / (1.0 + correlation);
        log_determinant += innovation_variance.ln();
        correlations.push(correlation);
        innovation_variances.push(innovation_variance);
    }
    let mean = mean_rhs / mean_weight;
    let mut quadratic = (response[0] - mean).powi(2);
    for index in 0..gaps.len() {
        let one_minus_correlation = innovation_variances[index] / (1.0 + correlations[index]);
        let residual = response[index + 1] - response[index]
            + one_minus_correlation * (response[index] - mean);
        quadratic += residual * residual / innovation_variances[index];
    }
    let sigma2 = quadratic / response.len() as f64;
    let objective = response.len() as f64 * ((std::f64::consts::TAU).ln() + 1.0 + sigma2.ln())
        + log_determinant;
    if !mean.is_finite()
        || !sigma2.is_finite()
        || sigma2 <= 0.0
        || !objective.is_finite()
        || !log_determinant.is_finite()
    {
        return Err(
            Issue::builder(IssueCode::NonFiniteOutput)
                .message(format!(
                    "exponential-process profile produced mean={mean}, sigma2={sigma2}, objective={objective}"
                ))
                .build(),
        );
    }
    Ok(ProcessProfile {
        mean,
        sigma2,
        objective,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn process_mle_decimal(value: &serde_json::Value) -> f64 {
        value
            .as_str()
            .expect("process_mle golden decimals are strings")
            .parse::<f64>()
            .expect("process_mle golden decimal must fit f64")
    }

    fn process_mle_decimal_array(value: &serde_json::Value) -> Vec<f64> {
        value
            .as_array()
            .expect("process_mle golden vector")
            .iter()
            .map(process_mle_decimal)
            .collect()
    }

    #[test]
    fn process_mle_matches_independent_decimal_dense_gls_oracle() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../golden/process_mle.json"))
                .expect("golden/process_mle.json");
        let cases = fixture["cases"].as_array().expect("process_mle cases");
        let mut max_parameter_error = 0.0_f64;
        let mut max_profile_error = 0.0_f64;
        for case in cases {
            let times = process_mle_decimal_array(&case["times"]);
            let observations = process_mle_decimal_array(&case["observations"]);
            let fitted = process_mle(
                &Vector::from_slice(&times),
                &Vector::from_slice(&observations),
                &Session::new("process_mle", "decimal_oracle"),
            )
            .expect("valid Decimal oracle case");
            let selected = &case["selected"];
            for (actual, expected) in [
                (fitted.value.mean, process_mle_decimal(&selected["mean"])),
                (
                    fitted.value.sigma2,
                    process_mle_decimal(&selected["sigma2"]),
                ),
                (fitted.value.range, process_mle_decimal(&selected["range"])),
            ] {
                max_parameter_error = max_parameter_error.max((actual - expected).abs());
            }

            let mut ordered: Vec<(f64, f64)> = times
                .iter()
                .copied()
                .zip(observations.iter().copied())
                .collect();
            ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
            let gaps: Vec<f64> = ordered
                .windows(2)
                .map(|pair| pair[1].0 - pair[0].0)
                .collect();
            let scale = ordered
                .iter()
                .fold(0.0_f64, |current, pair| current.max(pair.1.abs()));
            let normalized: Vec<f64> = ordered.iter().map(|pair| pair.1 / scale).collect();
            for evaluation in case["evaluations"].as_array().expect("range evaluations") {
                let range = process_mle_decimal(&evaluation["range"]);
                let profile = exponential_process_profile(
                    &gaps,
                    &normalized,
                    range,
                    signlred::Policy::default().underflow_guard,
                )
                .expect("validated oracle range");
                let actual_mean = profile.mean * scale;
                let actual_sigma2 = profile.sigma2 * scale * scale;
                let actual_profile_objective = profile.objective
                    - observations.len() as f64 * ((std::f64::consts::TAU).ln() + 1.0)
                    + 2.0 * observations.len() as f64 * scale.ln();
                for (actual, expected) in [
                    (actual_mean, process_mle_decimal(&evaluation["mean"])),
                    (actual_sigma2, process_mle_decimal(&evaluation["sigma2"])),
                    (
                        actual_profile_objective,
                        process_mle_decimal(&evaluation["profile_objective"]),
                    ),
                ] {
                    max_profile_error = max_profile_error.max((actual - expected).abs());
                }
            }
        }
        println!(
            "process_mle Decimal oracle max parameter abs={max_parameter_error:.17e}, max profile abs={max_profile_error:.17e}"
        );
        // Measured 2026-09-03: 1.77635683940025046e-15; tolerance is 4.1x.
        assert!(max_parameter_error <= 7.3e-15, "{max_parameter_error}");
        // Measured 2026-09-03: 1.42108547152020037e-14; tolerance is 4.1x.
        assert!(max_profile_error <= 5.9e-14, "{max_profile_error}");
    }

    #[test]
    fn process_mle_is_permutation_and_response_affine_equivariant() {
        let times = [0.0, 0.5, 1.7, 3.0, 5.2, 8.1];
        let response = [2.2, 1.7, 3.1, 2.4, 4.0, 3.3];
        let baseline = process_mle(
            &Vector::from_slice(&times),
            &Vector::from_slice(&response),
            &Session::new("process_mle", "properties"),
        )
        .expect("baseline process fit");
        let permutation = [3_usize, 0, 5, 1, 4, 2];
        let permuted = process_mle(
            &Vector::from_iter(permutation.iter().map(|index| times[*index])),
            &Vector::from_iter(permutation.iter().map(|index| response[*index])),
            &Session::new("process_mle", "permutation"),
        )
        .expect("permuted process fit");
        let offset = -4.2_f64;
        let factor = 3.25_f64;
        let transformed = process_mle(
            &Vector::from_slice(&times),
            &Vector::from_iter(response.iter().map(|value| offset + factor * value)),
            &Session::new("process_mle", "affine"),
        )
        .expect("affine response fit");
        let permutation_error = (baseline.value.mean - permuted.value.mean)
            .abs()
            .max((baseline.value.sigma2 - permuted.value.sigma2).abs())
            .max((baseline.value.range - permuted.value.range).abs());
        let affine_error = (transformed.value.mean - (offset + factor * baseline.value.mean))
            .abs()
            .max((transformed.value.sigma2 - factor * factor * baseline.value.sigma2).abs())
            .max((transformed.value.range - baseline.value.range).abs());
        println!(
            "process_mle property max permutation abs={permutation_error:.17e}, affine abs={affine_error:.17e}"
        );
        // Sorting makes this exact for the measured permutation.
        assert_eq!(permutation_error, 0.0);
        // Measured 2026-09-03: 2.66453525910037570e-15; tolerance is 4.1x.
        assert!(affine_error <= 1.1e-14, "{affine_error}");
    }

    #[test]
    fn process_mle_rejects_invalid_data_without_floors_or_row_dropping() {
        let session = Session::new("process_mle", "invalid");
        let mismatch = process_mle(
            &Vector::from_slice(&[0.0, 1.0, 2.0]),
            &Vector::from_slice(&[1.0, 2.0]),
            &session,
        )
        .expect_err("mismatched pairs must fail");
        assert!(mismatch.report.contains(IssueCode::DimensionMismatch));

        let nonfinite = process_mle(
            &Vector::from_slice(&[0.0, f64::NAN, 2.0]),
            &Vector::from_slice(&[1.0, 2.0, 4.0]),
            &session,
        )
        .expect_err("nonfinite timestamps must not be dropped");
        assert!(nonfinite.report.contains(IssueCode::NonFiniteInput));

        let duplicate = process_mle(
            &Vector::from_slice(&[0.0, 1.0, 1.0]),
            &Vector::from_slice(&[1.0, 2.0, 4.0]),
            &session,
        )
        .expect_err("duplicate timestamps need an explicit noise model");
        assert!(duplicate.report.contains(IssueCode::DuplicateIndex));

        let constant = process_mle(
            &Vector::from_slice(&[0.0, 1.0, 2.0]),
            &Vector::from_slice(&[4.0, 4.0, 4.0]),
            &session,
        )
        .expect_err("zero process variance must not be floored");
        assert!(constant.report.contains(IssueCode::ConstantTarget));

        let short = process_mle(
            &Vector::from_slice(&[0.0, 1.0]),
            &Vector::from_slice(&[1.0, 2.0]),
            &session,
        )
        .expect_err("two points do not identify all process parameters");
        assert!(short.report.contains(IssueCode::InsufficientSample));

        let unresolved_grid = process_mle(
            &Vector::from_slice(&[0.0, f64::from_bits(1), 1.0]),
            &Vector::from_slice(&[0.0, 1.0, 0.5]),
            &session,
        )
        .expect_err("a numerically singular range trial must not be skipped");
        assert!(unresolved_grid.report.contains(IssueCode::NearSingular));
    }
}
