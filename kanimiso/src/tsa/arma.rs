//! Exact ARMA coefficient, covariance, impulse-response, and sampling kernels.

use super::common::compensated_sum;
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::least_squares;
use crate::rng::Rng;
use ojizou_san::Session;
use signlred::{scan_finite, Issue, IssueCode, NumericalCompromise, Qualified, Result, Severity};

fn validate_arma_coefficient_finiteness(ctx: &mut FitCtx, ar: &Vector, ma: &Vector) -> bool {
    let mut finite = true;
    if let Some(issue) = scan_finite(ar.as_slice()).to_issue("AR coefficients") {
        ctx.push(issue);
        finite = false;
    }
    if let Some(issue) = scan_finite(ma.as_slice()).to_issue("MA coefficients") {
        ctx.push(issue);
        finite = false;
    }
    finite
}

/// Expand
/// \((1 + \sum_k n_k z^k) / (1 - \sum_k d_k z^k)\), including lag zero.
fn arma_ratio_expansion(
    ctx: &mut FitCtx,
    numerator: &[f64],
    denominator_recurrence: &[f64],
    lags: usize,
    overflow_message: &'static str,
) -> Option<Vector> {
    let mut coefficients = Vector::zeros(lags);
    if lags == 0 {
        return Some(coefficients);
    }
    coefficients[0] = 1.0;
    for lag in 1..lags {
        let direct = numerator.get(lag - 1).copied().unwrap_or(0.0);
        let Some(value) = compensated_sum(
            std::iter::once(direct).chain(
                (1..=denominator_recurrence.len().min(lag))
                    .map(|order| denominator_recurrence[order - 1] * coefficients[lag - order]),
            ),
        ) else {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message(overflow_message)
                    .metric("lag", lag as f64)
                    .build(),
            );
            return None;
        };
        coefficients[lag] = value;
    }
    Some(coefficients)
}

/// ARMA(\(p,q\)) to MA(\(\infty\)) coefficients (statsmodels `arma2ma`).
///
/// Returns exactly `lags` coefficients, including \(\psi_0=1\) when
/// `lags > 0`. With \(\Phi(z)=1-\sum_j\phi_jz^j\) and
/// \(\Theta(z)=1+\sum_j\theta_jz^j\), this expands
/// \(\Psi(z)=\Theta(z)/\Phi(z)\):
/// \(\psi_k=\theta_k+\sum_j\phi_j\psi_{k-j}\). The AR polynomial must be
/// causal; invertibility of the MA numerator is not required.
pub fn arma2ma(
    ar: &Vector,
    ma: &Vector,
    lags: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !validate_arma_coefficient_finiteness(&mut ctx, ar, ma) {
        return Err(ctx.finish_failure());
    }
    if !arma_coefficients_are_causal(ar.as_slice()) {
        ctx.push(
            Issue::builder(IssueCode::CausalityViolated)
                .message("arma2ma requires a causal AR denominator")
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    let Some(psi) = arma_ratio_expansion(
        &mut ctx,
        ma.as_slice(),
        ar.as_slice(),
        lags,
        "arma2ma impulse-response recurrence overflowed",
    ) else {
        return Err(ctx.finish_failure());
    };
    ctx.finish(psi)
}

/// ARMA(\(p,q\)) to AR(\(\infty\)) coefficients (statsmodels `arma2ar`).
///
/// Returns exactly `lags` coefficients of \(\Pi(z)=\Phi(z)/\Theta(z)\),
/// including \(\pi_0=1\) when `lags > 0`. Thus
/// \(\pi_k=-\phi_k-\sum_j\theta_j\pi_{k-j}\), and convolution with the
/// output of [`arma2ma`] is the unit sequence. The MA polynomial must be
/// invertible; causality of the finite AR numerator is not needed to form this
/// inverse-filter prefix.
pub fn arma2ar(
    ar: &Vector,
    ma: &Vector,
    lags: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !validate_arma_coefficient_finiteness(&mut ctx, ar, ma) {
        return Err(ctx.finish_failure());
    }
    let inverse_ma: Vec<f64> = ma
        .as_slice()
        .iter()
        .map(|coefficient| -coefficient)
        .collect();
    if !arma_coefficients_are_causal(&inverse_ma) {
        ctx.push(
            Issue::builder(IssueCode::InvertibilityViolated)
                .message("arma2ar requires an invertible MA denominator")
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    let ar_polynomial: Vec<f64> = ar
        .as_slice()
        .iter()
        .map(|coefficient| -coefficient)
        .collect();
    let Some(pi) = arma_ratio_expansion(
        &mut ctx,
        &ar_polynomial,
        &inverse_ma,
        lags,
        "arma2ar inverse-filter recurrence overflowed",
    ) else {
        return Err(ctx.finish_failure());
    };
    ctx.finish(pi)
}
/// Exact theoretical ARMA ACF (statsmodels `arima_process.arma_acf`).
///
/// The first autocovariances solve the finite ARMA Yule–Walker system;
/// later lags use its exact recurrence. Orders are not identification `p`.
pub fn arma_acf(
    ar: &Vector,
    ma: &Vector,
    nlags: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let Some(acov) = arma_exact_acovariances(&mut ctx, ar, ma, nlags) else {
        return Err(ctx.finish_failure());
    };
    let gamma0 = acov[0];
    let acf = Vector::from_iter(acov.as_slice().iter().map(|gamma| gamma / gamma0));
    ctx.finish(acf)
}
/// Simulate an ARMA series (statsmodels `arma_generate_sample`).
///
/// Innovations are deterministic `N(0,1)` draws for a fixed `seed`; values
/// before the sample are zero. A causal AR denominator is required. MA
/// invertibility is not required for the generative recurrence.
pub fn arma_generate_sample(
    ar: &Vector,
    ma: &Vector,
    n: usize,
    seed: u64,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !validate_arma_coefficient_finiteness(&mut ctx, ar, ma) {
        return Err(ctx.finish_failure());
    }
    if !arma_coefficients_are_causal(ar.as_slice()) {
        ctx.push(
            Issue::builder(IssueCode::CausalityViolated)
                .message("arma_generate_sample requires a causal AR denominator")
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    if n == 0 {
        return ctx.finish(Vector::zeros(0));
    }
    let mut rng = Rng::new(seed);
    let mut e = Vector::zeros(n);
    let mut y = Vector::zeros(n);
    for t in 0..n {
        e[t] = rng.standard_normal();
        let Some(value) = compensated_sum(
            std::iter::once(e[t])
                .chain((0..ma.len().min(t)).map(|lag| ma[lag] * e[t - 1 - lag]))
                .chain((0..ar.len().min(t)).map(|lag| ar[lag] * y[t - 1 - lag])),
        ) else {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("arma_generate_sample recurrence overflowed")
                    .metric("time_index", t as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        y[t] = value;
    }
    if !ar.is_empty() || !ma.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("arma_generate_sample uses Gaussian innovations and zero pre-sample")
                .compromise(NumericalCompromise::new(
                    "stationary start from the ARMA Lyapunov covariance",
                    "zero initial conditions plus N(0,1) innovations",
                    "the first max(p,q) draws are transient",
                    "discard a burn-in before treating the path as stationary",
                ))
                .build(),
        );
    }
    ctx.finish(y)
}

/// Impulse-response coefficients (statsmodels `ArmaProcess.arma2ma`).
///
/// Alias of [`arma2ma`]. Orders are not identification `p`.
pub fn arma_impulse_response(
    ar: &Vector,
    ma: &Vector,
    lags: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    arma2ma(ar, ma, lags, session)
}

/// Exact theoretical ARMA autocovariance (statsmodels `arima_process.arma_acovf`).
///
/// Returns \(\gamma(0),\ldots,\gamma(\mathrm{nlags})\) for unit innovation
/// variance by solving the finite Yule–Walker equations. Orders are not
/// identification `p`.
pub fn arma_acovf(
    ar: &Vector,
    ma: &Vector,
    nlags: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let Some(acov) = arma_exact_acovariances(&mut ctx, ar, ma, nlags) else {
        return Err(ctx.finish_failure());
    };
    ctx.finish(acov)
}

fn arma_exact_acovariances(
    ctx: &mut FitCtx,
    ar: &Vector,
    ma: &Vector,
    nlags: usize,
) -> Option<Vector> {
    let Some(output_len) = nlags.checked_add(1) else {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message("ARMA autocovariance output length overflowed")
                .build(),
        );
        return None;
    };
    if !validate_arma_coefficient_finiteness(ctx, ar, ma) {
        return None;
    }
    if !arma_coefficients_are_causal(ar.as_slice()) {
        ctx.push(
            Issue::builder(IssueCode::CausalityViolated)
                .message("AR coefficients do not define a causal stationary process")
                .build(),
        );
        return None;
    }

    let p = ar.len();
    let q = ma.len();
    let mut psi = vec![0.0_f64; q.saturating_add(1)];
    psi[0] = 1.0;
    for lag in 1..=q {
        let Some(value) = compensated_sum(
            std::iter::once(ma[lag - 1])
                .chain((1..=p.min(lag)).map(|order| ar[order - 1] * psi[lag - order])),
        ) else {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("ARMA impulse-response initialization overflowed")
                    .build(),
            );
            return None;
        };
        psi[lag] = value;
    }

    let forcing = |lag: usize| {
        compensated_sum((lag..=q).map(|index| {
            let theta = if index == 0 { 1.0 } else { ma[index - 1] };
            theta * psi[index - lag]
        }))
    };

    if p == 0 {
        let mut values = Vec::with_capacity(output_len);
        for lag in 0..=nlags {
            let Some(value) = forcing(lag) else {
                ctx.push(
                    Issue::builder(IssueCode::NumericalOverflow)
                        .message("MA autocovariance summation overflowed")
                        .build(),
                );
                return None;
            };
            values.push(value);
        }
        return Some(Vector::from_iter(values));
    }

    let Some(initial_count) = p.checked_add(1) else {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message("ARMA Yule–Walker system dimension overflowed")
                .build(),
        );
        return None;
    };
    let mut system = Matrix::zeros(initial_count, initial_count);
    for lag in 0..initial_count {
        system.set(lag, lag, 1.0);
        for order in 1..=p {
            let gamma_lag = lag.abs_diff(order);
            let updated = system.get(lag, gamma_lag) - ar[order - 1];
            if !updated.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NumericalOverflow)
                        .message("ARMA Yule–Walker system construction overflowed")
                        .build(),
                );
                return None;
            }
            system.set(lag, gamma_lag, updated);
        }
    }
    let mut rhs_values = Vec::with_capacity(initial_count);
    for lag in 0..initial_count {
        let Some(value) = forcing(lag) else {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("ARMA Yule–Walker right-hand side overflowed")
                    .build(),
            );
            return None;
        };
        rhs_values.push(value);
    }
    let rhs = Vector::from_iter(rhs_values);
    let initial = least_squares(&mut ctx.report, &system, &rhs, &ctx.policy)?;
    if initial.as_slice().iter().any(|value| !value.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteOutput)
                .message("ARMA Yule–Walker solve produced a non-finite autocovariance")
                .build(),
        );
        return None;
    }
    if initial[0] <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::NonPositiveDefinite)
                .message("ARMA Yule–Walker solve produced a non-positive variance")
                .metric("gamma0", initial[0])
                .build(),
        );
        return None;
    }

    let mut values = initial.as_slice()[..initial_count.min(output_len)].to_vec();
    for lag in initial_count..=nlags {
        let Some(noise_covariance) = forcing(lag) else {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("ARMA autocovariance forcing term overflowed")
                    .build(),
            );
            return None;
        };
        let Some(value) = compensated_sum(
            (1..=p)
                .map(|order| ar[order - 1] * values[lag - order])
                .chain(std::iter::once(noise_covariance)),
        ) else {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("ARMA autocovariance recurrence overflowed")
                    .build(),
            );
            return None;
        };
        values.push(value);
    }
    Some(Vector::from_iter(values))
}

fn arma_coefficients_are_causal(ar: &[f64]) -> bool {
    let mut coefficients = ar.to_vec();
    for order in (1..=coefficients.len()).rev() {
        let reflection = coefficients[order - 1];
        if !reflection.is_finite() || reflection.abs() >= 1.0 {
            return false;
        }
        let denominator = 1.0 - reflection * reflection;
        let mut previous = vec![0.0_f64; order - 1];
        for index in 0..order - 1 {
            previous[index] =
                (coefficients[index] + reflection * coefficients[order - 2 - index]) / denominator;
            if !previous[index].is_finite() {
                return false;
            }
        }
        coefficients[..order - 1].copy_from_slice(&previous);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn arma_autocovariance_matches_closed_forms_without_tail_truncation() {
        let session = Session::new("arma-acovariance", "closed-forms");
        let near_unit = 0.999_f64;
        let ar = Vector::from_slice(&[near_unit]);
        let empty = Vector::zeros(0);
        let acov = arma_acovf(&ar, &empty, 8, &session)
            .expect("causal near-unit AR(1)")
            .value;
        let acf = arma_acf(&ar, &empty, 8, &session)
            .expect("causal near-unit AR(1) ACF")
            .value;
        let expected_variance = 1.0 / (1.0 - near_unit * near_unit);
        let mut max_absolute_error = (acov[0] - expected_variance).abs();
        for lag in 0..=8 {
            let expected_acf = near_unit.powi(lag as i32);
            max_absolute_error = max_absolute_error.max((acf[lag] - expected_acf).abs());
            max_absolute_error =
                max_absolute_error.max((acov[lag] - expected_variance * expected_acf).abs());
        }

        let phi = 0.7_f64;
        let theta = -0.2_f64;
        let arma = arma_acovf(
            &Vector::from_slice(&[phi]),
            &Vector::from_slice(&[theta]),
            8,
            &session,
        )
        .expect("causal ARMA(1,1)")
        .value;
        let arma_variance = (1.0 + theta * theta + 2.0 * phi * theta) / (1.0 - phi * phi);
        let arma_lag_one = phi * arma_variance + theta;
        max_absolute_error = max_absolute_error.max((arma[0] - arma_variance).abs());
        max_absolute_error = max_absolute_error.max((arma[1] - arma_lag_one).abs());
        for lag in 2..=8 {
            max_absolute_error = max_absolute_error
                .max((arma[lag] - arma_lag_one * phi.powi((lag - 1) as i32)).abs());
        }

        let ma = arma_acovf(&empty, &Vector::from_slice(&[0.4, -0.3]), 4, &session)
            .expect("finite MA(2)")
            .value;
        let expected_ma = [1.25, 0.28, -0.3, 0.0, 0.0];
        for (actual, expected) in ma.as_slice().iter().zip(expected_ma) {
            max_absolute_error = max_absolute_error.max((actual - expected).abs());
        }

        eprintln!("exact ARMA closed-form max_abs={max_absolute_error:.17e}");
        // Measured 2.76827449852135032e-11 on 2026-09-03; tolerance is 3.6×.
        assert!(max_absolute_error <= 1.0e-10);
    }

    #[test]
    fn arma_autocovariance_matches_independent_decimal_lyapunov_oracle() {
        let payload: serde_json::Value =
            serde_json::from_str(include_str!("../../../golden/arma_acov.json"))
                .expect("golden/arma_acov.json");
        let decimal = |value: &serde_json::Value| {
            value
                .as_str()
                .expect("Decimal string")
                .parse::<f64>()
                .expect("binary64 Decimal value")
        };
        let mut max_absolute_error = 0.0_f64;
        let mut max_relative_error = 0.0_f64;
        for case in payload["cases"].as_array().expect("ARMA oracle cases") {
            let input = &case["input"];
            let ar = Vector::from_iter(
                input["ar"]
                    .as_array()
                    .expect("AR coefficients")
                    .iter()
                    .map(&decimal),
            );
            let ma = Vector::from_iter(
                input["ma"]
                    .as_array()
                    .expect("MA coefficients")
                    .iter()
                    .map(&decimal),
            );
            let nlags = input["nlags"].as_u64().expect("nlags") as usize;
            let name = case["name"].as_str().expect("case name");
            let session = Session::new("arma-acovariance", name);
            let actual_acov = arma_acovf(&ar, &ma, nlags, &session)
                .unwrap_or_else(|failure| panic!("{name}: {failure}"))
                .value;
            let actual_acf = arma_acf(&ar, &ma, nlags, &session)
                .unwrap_or_else(|failure| panic!("{name}: {failure}"))
                .value;
            for (actual, expected) in actual_acov
                .as_slice()
                .iter()
                .zip(case["autocovariance"].as_array().expect("autocovariance"))
                .chain(
                    actual_acf
                        .as_slice()
                        .iter()
                        .zip(case["autocorrelation"].as_array().expect("autocorrelation")),
                )
            {
                let expected = decimal(expected);
                let absolute = (actual - expected).abs();
                let relative = absolute / expected.abs().max(f64::MIN_POSITIVE);
                max_absolute_error = max_absolute_error.max(absolute);
                max_relative_error = max_relative_error.max(relative);
            }
        }
        eprintln!(
            "ARMA Decimal-Lyapunov oracle: max_abs={max_absolute_error:.17e}, max_rel={max_relative_error:.17e}"
        );
        // Measured 2.09183781407773495e-11 absolute and
        // 4.18389850766444961e-14 relative on 2026-09-03; tolerances are 4×.
        assert!(max_absolute_error <= 8.4e-11);
        assert!(max_relative_error <= 1.68e-13);
    }

    #[test]
    fn arma_autocovariance_enforces_causality_and_the_yule_walker_recurrence() {
        let session = Session::new("arma-acovariance", "properties");
        let empty = Vector::zeros(0);
        let white = arma_acovf(&empty, &empty, 3, &session)
            .expect("white noise is ARMA(0,0)")
            .value;
        assert_eq!(white.as_slice(), &[1.0, 0.0, 0.0, 0.0]);
        let zero_lag = arma_acf(&empty, &empty, 0, &session)
            .expect("lag zero")
            .value;
        assert_eq!(zero_lag.as_slice(), &[1.0]);

        let coefficients = [1.5_f64, -0.75_f64];
        let gamma = arma_acovf(
            &Vector::from_slice(&coefficients),
            &Vector::from_slice(&[0.2]),
            20,
            &session,
        )
        .expect("stable AR(2) despite sum of absolute coefficients exceeding one")
        .value;
        let mut max_recurrence_error = 0.0_f64;
        for lag in 3..=20 {
            let expected = coefficients[0] * gamma[lag - 1] + coefficients[1] * gamma[lag - 2];
            max_recurrence_error = max_recurrence_error.max((gamma[lag] - expected).abs());
        }
        eprintln!("exact ARMA recurrence max_abs={max_recurrence_error:.17e}");
        assert_eq!(max_recurrence_error, 0.0);

        let noncausal = arma_acovf(&Vector::from_slice(&[1.0]), &empty, 3, &session)
            .expect_err("unit-root autocovariance is undefined");
        assert_eq!(noncausal.primary.code, IssueCode::CausalityViolated);
        let nonfinite = arma_acf(&Vector::from_slice(&[f64::INFINITY]), &empty, 3, &session)
            .expect_err("non-finite coefficient");
        assert_eq!(nonfinite.primary.code, IssueCode::NonFiniteInput);
    }

    #[test]
    fn arma_ratio_expansions_match_closed_forms_and_convolution_identity() {
        let session = Session::new("arma-ratio", "closed-forms");
        let empty = Vector::zeros(0);
        assert!(arma2ma(&empty, &empty, 0, &session)
            .expect("zero-length MA prefix")
            .value
            .is_empty());
        assert!(arma2ar(&empty, &empty, 0, &session)
            .expect("zero-length AR prefix")
            .value
            .is_empty());
        assert_eq!(
            arma2ma(&empty, &empty, 4, &session)
                .expect("white-noise transfer function")
                .value
                .as_slice(),
            &[1.0, 0.0, 0.0, 0.0]
        );

        let phi = 0.5_f64;
        let theta = 0.2_f64;
        let coefficient_count = 12;
        let ar = Vector::from_slice(&[phi]);
        let ma = Vector::from_slice(&[theta]);
        let psi = arma2ma(&ar, &ma, coefficient_count, &session)
            .expect("causal transfer function")
            .value;
        let pi = arma2ar(&ar, &ma, coefficient_count, &session)
            .expect("invertible inverse filter")
            .value;
        let mut maximum_closed_form_error = 0.0_f64;
        let mut maximum_convolution_error = 0.0_f64;
        for lag in 0..coefficient_count {
            let expected_psi = if lag == 0 {
                1.0
            } else {
                (phi + theta) * phi.powi((lag - 1) as i32)
            };
            let expected_pi = if lag == 0 {
                1.0
            } else {
                -(phi + theta) * (-theta).powi((lag - 1) as i32)
            };
            maximum_closed_form_error = maximum_closed_form_error
                .max((psi[lag] - expected_psi).abs())
                .max((pi[lag] - expected_pi).abs());
            let convolution: f64 = (0..=lag).map(|index| psi[index] * pi[lag - index]).sum();
            let expected_convolution = if lag == 0 { 1.0 } else { 0.0 };
            maximum_convolution_error =
                maximum_convolution_error.max((convolution - expected_convolution).abs());
        }
        eprintln!(
            "ARMA ratio identities: closed_form={maximum_closed_form_error:.17e}, \
             convolution={maximum_convolution_error:.17e}"
        );
        // Measured on 2026-09-03: 6.93889390390722838e-18 closed form and
        // 1.38777878078144568e-17 convolution; tolerances are approximately 4×.
        assert!(maximum_closed_form_error <= 2.8e-17);
        assert!(maximum_convolution_error <= 5.6e-17);

        let stable_high_sum = arma2ma(&Vector::from_slice(&[1.5, -0.75]), &empty, 8, &session)
            .expect("Schur-stable AR(2) with sum of absolute coefficients above one");
        assert!(stable_high_sum
            .value
            .as_slice()
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn arma_ratio_expansions_propagate_domain_and_arithmetic_failures() {
        let session = Session::new("arma-ratio", "failure-propagation");
        let empty = Vector::zeros(0);
        let noncausal = arma2ma(&Vector::from_slice(&[1.0]), &empty, 4, &session)
            .expect_err("unit-root AR denominator");
        assert_eq!(noncausal.primary.code, IssueCode::CausalityViolated);
        let noninvertible = arma2ar(&empty, &Vector::from_slice(&[1.0]), 4, &session)
            .expect_err("unit-root MA denominator");
        assert_eq!(noninvertible.primary.code, IssueCode::InvertibilityViolated);
        let nonfinite = arma2ma(&Vector::from_slice(&[f64::NAN]), &empty, 4, &session)
            .expect_err("non-finite AR coefficient");
        assert_eq!(nonfinite.primary.code, IssueCode::NonFiniteInput);

        let ma_overflow = arma2ma(
            &Vector::from_slice(&[0.5]),
            &Vector::from_slice(&[f64::MAX, f64::MAX]),
            3,
            &session,
        )
        .expect_err("overflowed MA expansion must not be replaced by zero");
        assert_eq!(ma_overflow.primary.code, IssueCode::NumericalOverflow);
        let ar_overflow = arma2ar(
            &Vector::from_slice(&[f64::MAX, -f64::MAX]),
            &Vector::from_slice(&[0.5]),
            3,
            &session,
        )
        .expect_err("overflowed inverse expansion must not be replaced by zero");
        assert_eq!(ar_overflow.primary.code, IssueCode::NumericalOverflow);
    }

    #[test]
    fn arma_sample_is_exact_length_deterministic_and_failure_atomic() {
        let session = Session::new("arma-sample", "properties");
        let empty = Vector::zeros(0);
        let phi = 0.4_f64;
        let theta = 0.2_f64;
        let ar = Vector::from_slice(&[phi]);
        let ma = Vector::from_slice(&[theta]);
        assert!(arma_generate_sample(&ar, &ma, 0, 91, &session)
            .expect("empty requested sample")
            .value
            .is_empty());

        let direct = arma_generate_sample(&ar, &ma, 16, 91, &session)
            .expect("direct deterministic sample")
            .value;
        assert!(direct.as_slice().iter().all(|value| value.is_finite()));

        let mut rng = Rng::new(91);
        let innovation_zero = rng.standard_normal();
        let innovation_one = rng.standard_normal();
        let innovation_two = rng.standard_normal();
        let expected = [
            innovation_zero,
            innovation_one + (phi + theta) * innovation_zero,
            innovation_two + (phi + theta) * innovation_one + phi * (phi + theta) * innovation_zero,
        ];
        let maximum_replay_error = direct.as_slice()[..expected.len()]
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("ARMA sample closed-form replay max_abs={maximum_replay_error:.17e}");
        let replay_scale = expected
            .iter()
            .map(|value| value.abs())
            .fold(1.0_f64, f64::max);
        // Measured exactly zero on 2026-09-03. The bound is four binary64
        // roundoffs at the observed scale, rather than an arbitrary decimal floor.
        assert!(maximum_replay_error <= 4.0 * f64::EPSILON * replay_scale);

        let noncausal = arma_generate_sample(&Vector::from_slice(&[1.0]), &empty, 4, 91, &session)
            .expect_err("noncausal process generation");
        assert_eq!(noncausal.primary.code, IssueCode::CausalityViolated);
        let noninvertible_but_generative =
            arma_generate_sample(&empty, &Vector::from_slice(&[1.25]), 8, 91, &session)
                .expect("MA invertibility is not required for generation");
        assert!(noninvertible_but_generative
            .value
            .as_slice()
            .iter()
            .all(|value| value.is_finite()));
        // Seed 3 starts with |z| > 1, so MAX * z must overflow.  This makes the
        // failure precondition explicit instead of depending on a merely large
        // but still finite first innovation (seed 4).
        assert!(Rng::new(3).standard_normal().abs() > 1.0);
        let overflow =
            arma_generate_sample(&empty, &Vector::from_slice(&[f64::MAX]), 2, 3, &session)
                .expect_err("overflow must abort instead of inserting zero");
        assert_eq!(overflow.primary.code, IssueCode::NumericalOverflow);
        let nonfinite = arma_generate_sample(
            &empty,
            &Vector::from_slice(&[f64::INFINITY]),
            2,
            91,
            &session,
        )
        .expect_err("non-finite MA coefficient");
        assert_eq!(nonfinite.primary.code, IssueCode::NonFiniteInput);
    }
}
