//! Canonical log-space forward–backward recursion.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::special::logsumexp;
use signlred::{Issue, IssueCode};

/// Posterior quantities produced by the canonical chain recursion.
pub(super) struct ScaledFb {
    /// Sequence log-likelihood.
    pub(super) loglik: f64,
    /// Filtering probabilities \(P(s_t \mid y_{1:t})\), indexed by time then state.
    pub(super) filtered: Vec<Vec<f64>>,
    /// State posteriors indexed by time then state.
    pub(super) gamma: Vec<Vec<f64>>,
    /// Adjacent-state posteriors indexed by time, source, then destination.
    pub(super) xi: Vec<Vec<Vec<f64>>>,
}

/// Evaluate a finite HMM sequence entirely in the log domain.
///
/// Chain probabilities have already passed the public model preflight. This
/// additional validation keeps the kernel sound for crate-internal adapters
/// and makes an impossible sequence distinct from a merely tiny likelihood.
pub(super) fn scaled_forward_backward(
    ctx: &mut FitCtx,
    start: &Vector,
    trans: &Matrix,
    log_emit: &[Vec<f64>],
) -> Option<ScaledFb> {
    let time_count = log_emit.len();
    let state_count = start.len();
    if time_count == 0 || state_count == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("HMM forward–backward on an empty sequence")
                .build(),
        );
        return None;
    }
    if trans.nrows() != state_count
        || trans.ncols() != state_count
        || log_emit.iter().any(|row| row.len() != state_count)
    {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "forward-backward expects start={state_count}, transition={}x{}, and {state_count} emissions per time",
                    trans.nrows(),
                    trans.ncols()
                ))
                .build(),
        );
        return None;
    }
    if !ctx.policy.underflow_guard.is_finite() || ctx.policy.underflow_guard <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message("Policy::underflow_guard must be finite and positive")
                .build(),
        );
        return None;
    }
    if start
        .as_slice()
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
        || (0..state_count).any(|source| {
            (0..state_count).any(|destination| {
                let weight = trans.get(source, destination);
                !weight.is_finite() || weight < 0.0
            })
        })
    {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message("HMM start and transition weights must be finite and non-negative")
                .build(),
        );
        return None;
    }
    if log_emit
        .iter()
        .flatten()
        .any(|value| value.is_nan() || *value == f64::INFINITY)
    {
        ctx.push(
            Issue::builder(IssueCode::LossIsNan)
                .message("HMM log-emissions contain NaN or positive infinity")
                .build(),
        );
        return None;
    }

    let log_start: Vec<f64> = start
        .as_slice()
        .iter()
        .map(|weight| {
            if *weight > 0.0 {
                weight.ln()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let log_trans: Vec<Vec<f64>> = (0..state_count)
        .map(|source| {
            (0..state_count)
                .map(|destination| {
                    let weight = trans.get(source, destination);
                    if weight > 0.0 {
                        weight.ln()
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect()
        })
        .collect();

    // `log_alpha[t]` is normalized at every time. `log_scale[t]` is the
    // conditional observation log-likelihood removed by that normalization.
    let log_guard = ctx.policy.underflow_guard.ln();
    let mut log_alpha = vec![vec![f64::NEG_INFINITY; state_count]; time_count];
    let mut log_scale = vec![0.0; time_count];
    let mut terms = vec![f64::NEG_INFINITY; state_count];
    let mut raw = vec![f64::NEG_INFINITY; state_count];
    let mut loglik = 0.0;
    for time in 0..time_count {
        if time == 0 {
            for state in 0..state_count {
                raw[state] = log_start[state] + log_emit[time][state];
            }
        } else {
            for destination in 0..state_count {
                for source in 0..state_count {
                    terms[source] = log_alpha[time - 1][source] + log_trans[source][destination];
                }
                raw[destination] = logsumexp(&terms) + log_emit[time][destination];
            }
        }

        let current_log_scale = logsumexp(&raw);
        if current_log_scale == f64::NEG_INFINITY {
            ctx.push(
                Issue::builder(IssueCode::ScaleFactorZero)
                    .message(format!(
                        "no reachable state has positive observation likelihood at t={time}"
                    ))
                    .metric("t", time as f64)
                    .build(),
            );
            return None;
        }
        if !current_log_scale.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::LossIsNan)
                    .message(format!("HMM forward log-scale is NaN or +Inf at t={time}"))
                    .metric("t", time as f64)
                    .build(),
            );
            return None;
        }
        if current_log_scale < log_guard {
            ctx.push(
                Issue::builder(IssueCode::ForwardUnderflow)
                    .message(format!(
                        "forward log-scale at t={time} is {current_log_scale:.3e}; evaluated in log space"
                    ))
                    .metric("t", time as f64)
                    .metric("log_scale", current_log_scale)
                    .build(),
            );
        }
        log_scale[time] = current_log_scale;
        loglik += current_log_scale;
        if !loglik.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::LossIsNan)
                    .message(format!(
                        "HMM accumulated log-likelihood is not finite at t={time}"
                    ))
                    .metric("t", time as f64)
                    .build(),
            );
            return None;
        }
        for state in 0..state_count {
            log_alpha[time][state] = raw[state] - current_log_scale;
        }
    }

    // The forward rows were normalized as they were produced. Converting
    // those log probabilities once gives every prefix-conditioned state
    // distribution without rerunning the chain recursion for each prefix.
    let filtered: Vec<Vec<f64>> = log_alpha
        .iter()
        .map(|row| {
            row.iter()
                .map(|log_probability| log_probability.exp())
                .collect()
        })
        .collect();

    let mut log_beta = vec![vec![f64::NEG_INFINITY; state_count]; time_count];
    for state in 0..state_count {
        log_beta[time_count - 1][state] = 0.0;
    }
    for time in (0..time_count - 1).rev() {
        for source in 0..state_count {
            for destination in 0..state_count {
                terms[destination] = log_trans[source][destination]
                    + log_emit[time + 1][destination]
                    + log_beta[time + 1][destination];
            }
            let value = logsumexp(&terms) - log_scale[time + 1];
            if value.is_nan() || value == f64::INFINITY {
                ctx.push(
                    Issue::builder(IssueCode::LossIsNan)
                        .message(format!(
                            "HMM backward log-mass is invalid at t={time}, state={source}"
                        ))
                        .metric("t", time as f64)
                        .metric("state", source as f64)
                        .build(),
                );
                return None;
            }
            log_beta[time][source] = value;
        }
    }

    let mut gamma = vec![vec![0.0; state_count]; time_count];
    for time in 0..time_count {
        for state in 0..state_count {
            terms[state] = log_alpha[time][state] + log_beta[time][state];
        }
        let log_norm = logsumexp(&terms);
        if !log_norm.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::LossIsNan)
                    .message(format!(
                        "HMM state posterior has zero or non-finite mass at t={time}"
                    ))
                    .metric("t", time as f64)
                    .build(),
            );
            return None;
        }
        for state in 0..state_count {
            gamma[time][state] = (terms[state] - log_norm).exp();
        }
    }

    let mut xi = vec![vec![vec![0.0; state_count]; state_count]; time_count.saturating_sub(1)];
    let mut pair_logs = vec![f64::NEG_INFINITY; state_count.saturating_mul(state_count)];
    for time in 0..time_count.saturating_sub(1) {
        for source in 0..state_count {
            for destination in 0..state_count {
                pair_logs[source * state_count + destination] = log_alpha[time][source]
                    + log_trans[source][destination]
                    + log_emit[time + 1][destination]
                    + log_beta[time + 1][destination];
            }
        }
        let log_norm = logsumexp(&pair_logs);
        if !log_norm.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::LossIsNan)
                    .message(format!(
                        "HMM transition posterior has zero or non-finite mass at t={time}"
                    ))
                    .metric("t", time as f64)
                    .build(),
            );
            return None;
        }
        for source in 0..state_count {
            for destination in 0..state_count {
                xi[time][source][destination] =
                    (pair_logs[source * state_count + destination] - log_norm).exp();
            }
        }
    }

    Some(ScaledFb {
        loglik,
        filtered,
        gamma,
        xi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;
    use signlred::Policy;

    fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    fn has_issue(ctx: &FitCtx, code: IssueCode) -> bool {
        ctx.report.issues().iter().any(|issue| issue.code == code)
    }

    fn two_state_parameters() -> (Vector, Matrix) {
        (
            Vector::from_iter([0.6, 0.4]),
            Matrix::from_row_major(2, 2, &[0.7, 0.3, 0.2, 0.8]),
        )
    }

    fn brute_force_prefix_filtering(
        start: &Vector,
        trans: &Matrix,
        log_emit: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let time_count = log_emit.len();
        let state_count = start.len();
        assert!(time_count > 0 && state_count > 0);
        let mut filtered = vec![vec![0.0; state_count]; time_count];

        for final_time in 0..time_count {
            let prefix_len = final_time + 1;
            let path_count = (0..prefix_len).fold(1usize, |count, _| count * state_count);
            let mut terminal_and_log_mass = Vec::with_capacity(path_count);
            let mut maximum = f64::NEG_INFINITY;
            for encoded in 0..path_count {
                let mut remainder = encoded;
                let mut previous = remainder % state_count;
                remainder /= state_count;
                let mut log_mass = if start[previous] > 0.0 {
                    start[previous].ln() + log_emit[0][previous]
                } else {
                    f64::NEG_INFINITY
                };
                for row in log_emit.iter().take(prefix_len).skip(1) {
                    let state = remainder % state_count;
                    remainder /= state_count;
                    let transition = trans.get(previous, state);
                    if transition > 0.0 {
                        log_mass += transition.ln() + row[state];
                    } else {
                        log_mass = f64::NEG_INFINITY;
                    }
                    previous = state;
                }
                maximum = maximum.max(log_mass);
                terminal_and_log_mass.push((previous, log_mass));
            }
            assert!(maximum.is_finite());

            let mut total = 0.0;
            for (terminal, log_mass) in terminal_and_log_mass {
                if log_mass.is_finite() {
                    let relative_mass = (log_mass - maximum).exp();
                    filtered[final_time][terminal] += relative_mass;
                    total += relative_mass;
                }
            }
            assert!(total.is_finite() && total > 0.0);
            for probability in &mut filtered[final_time] {
                *probability /= total;
            }
        }
        filtered
    }

    fn brute_force_forward_backward(
        start: &Vector,
        trans: &Matrix,
        log_emit: &[Vec<f64>],
    ) -> ScaledFb {
        let time_count = log_emit.len();
        let state_count = start.len();
        assert!(time_count > 0 && state_count > 0);
        let path_count = (0..time_count).fold(1usize, |count, _| count * state_count);
        let mut total = 0.0;
        let mut gamma = vec![vec![0.0; state_count]; time_count];
        let mut xi = vec![vec![vec![0.0; state_count]; state_count]; time_count.saturating_sub(1)];

        for encoded in 0..path_count {
            let mut remainder = encoded;
            let mut path = vec![0usize; time_count];
            for state in &mut path {
                *state = remainder % state_count;
                remainder /= state_count;
            }
            let mut weight = start[path[0]] * log_emit[0][path[0]].exp();
            for time in 1..time_count {
                weight *= trans.get(path[time - 1], path[time]) * log_emit[time][path[time]].exp();
            }
            total += weight;
            for time in 0..time_count {
                gamma[time][path[time]] += weight;
            }
            for time in 0..time_count.saturating_sub(1) {
                xi[time][path[time]][path[time + 1]] += weight;
            }
        }
        assert!(total.is_finite() && total > 0.0);
        for row in &mut gamma {
            for value in row {
                *value /= total;
            }
        }
        for slice in &mut xi {
            for row in slice {
                for value in row {
                    *value /= total;
                }
            }
        }
        ScaledFb {
            loglik: total.ln(),
            filtered: brute_force_prefix_filtering(start, trans, log_emit),
            gamma,
            xi,
        }
    }

    #[test]
    fn filtering_matches_brute_force_prefix_conditioning() {
        let (start, trans) = two_state_parameters();
        let base = vec![
            vec![-0.3, -1.1],
            vec![-1.4, -0.2],
            vec![-0.7, -0.6],
            vec![-2.0, -0.1],
        ];
        let offsets = [-10_000.0, -8_000.0, -12_000.0, -9_000.0];
        let shifted: Vec<Vec<f64>> = base
            .iter()
            .zip(offsets)
            .map(|(row, offset)| row.iter().map(|value| value + offset).collect())
            .collect();
        let single = vec![vec![-10_000.0 + 0.4_f64.ln(), -10_000.0 + 0.8_f64.ln()]];
        let single_start = Vector::from_iter([0.25, 0.75]);
        let mut maximum_error = 0.0_f64;

        for (label, case_start, log_emit) in [
            ("base", &start, base.as_slice()),
            ("extreme-shifts", &start, shifted.as_slice()),
            ("single", &single_start, single.as_slice()),
        ] {
            let expected = brute_force_prefix_filtering(case_start, &trans, log_emit);
            let mut ctx = FitCtx::with_session(Session::new("hmm-filter", label));
            let actual = scaled_forward_backward(&mut ctx, case_start, &trans, log_emit)
                .expect("prefix-conditioned probabilities");
            assert_eq!(actual.filtered.len(), log_emit.len());
            for time in 0..log_emit.len() {
                for state in 0..case_start.len() {
                    maximum_error = maximum_error
                        .max((actual.filtered[time][state] - expected[time][state]).abs());
                }
                maximum_error =
                    maximum_error.max((actual.filtered[time].iter().sum::<f64>() - 1.0).abs());
            }
        }
        eprintln!("HMM filtering/brute-force-prefix max_abs={maximum_error:.17e}");
        // Measured on 2026-09-03 across every base/extreme-shift prefix and
        // the single-observation case: 8.19011525265977980e-13. The 3.3e-12
        // R9 limit is approximately 4.03x that maximum error.
        assert!(maximum_error <= 3.3e-12);
    }

    #[test]
    fn matches_brute_force_posteriors() {
        let (start, trans) = two_state_parameters();
        let emission = [[0.9_f64, 0.1], [0.4, 0.6], [0.2, 0.8]];
        let log_emit: Vec<Vec<f64>> = emission
            .iter()
            .map(|row| row.iter().map(|value| value.ln()).collect())
            .collect();
        let expected = brute_force_forward_backward(&start, &trans, &log_emit);
        let mut ctx = FitCtx::with_session(Session::new("hmm-fb", "brute-force"));
        let actual = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit)
            .expect("moderate finite probabilities have positive likelihood");

        // Measured maximum absolute discrepancy over loglik/gamma/xi was
        // 6.66e-16 on 2026-09-02; 2.5e-15 is 3.75x that error.
        let tolerance = 2.5e-15;
        assert_close(actual.loglik, expected.loglik, tolerance, "log-likelihood");
        for time in 0..log_emit.len() {
            for state in 0..start.len() {
                assert_close(
                    actual.gamma[time][state],
                    expected.gamma[time][state],
                    tolerance,
                    "gamma",
                );
            }
            assert_close(
                actual.gamma[time].iter().sum(),
                1.0,
                tolerance,
                "gamma row sum",
            );
        }
        for time in 0..actual.xi.len() {
            let xi_sum: f64 = actual.xi[time].iter().flatten().sum();
            assert_close(xi_sum, 1.0, tolerance, "xi slice sum");
            for state in 0..start.len() {
                assert_close(
                    actual.xi[time][state].iter().sum(),
                    actual.gamma[time][state],
                    tolerance,
                    "xi outgoing marginal",
                );
                let incoming: f64 = actual.xi[time].iter().map(|row| row[state]).sum();
                assert_close(
                    incoming,
                    actual.gamma[time + 1][state],
                    tolerance,
                    "xi incoming marginal",
                );
            }
        }
        assert!(ctx.report.issues().is_empty());
    }

    #[test]
    fn is_shift_invariant_and_handles_extreme_logs() {
        let (start, trans) = two_state_parameters();
        let emission = [[0.9_f64, 0.1], [0.4, 0.6], [0.2, 0.8]];
        let base_log_emit: Vec<Vec<f64>> = emission
            .iter()
            .map(|row| row.iter().map(|value| value.ln()).collect())
            .collect();
        let offsets = [-1_000.0, -1_000.0, -1_000.0];
        let shifted_log_emit: Vec<Vec<f64>> = base_log_emit
            .iter()
            .zip(offsets)
            .map(|(row, offset)| row.iter().map(|value| value + offset).collect())
            .collect();
        let mut base_ctx = FitCtx::with_session(Session::new("hmm-fb", "base"));
        let base = scaled_forward_backward(&mut base_ctx, &start, &trans, &base_log_emit)
            .expect("base likelihood");
        let mut shifted_ctx = FitCtx::with_session(Session::new("hmm-fb", "shifted"));
        let shifted = scaled_forward_backward(&mut shifted_ctx, &start, &trans, &shifted_log_emit)
            .expect("finite log shifts must not erase probability mass");

        // Measured maximum posterior discrepancy after the -1000 shifts was
        // 7.77e-15 on 2026-09-02; 3e-14 is 3.86x that error.
        let shift_tolerance = 3.0e-14;
        assert_close(
            shifted.loglik - base.loglik,
            offsets.iter().sum(),
            shift_tolerance,
            "per-time shift in log-likelihood",
        );
        for time in 0..base.gamma.len() {
            for state in 0..start.len() {
                assert_close(
                    shifted.gamma[time][state],
                    base.gamma[time][state],
                    shift_tolerance,
                    "shift-invariant gamma",
                );
            }
        }
        for time in 0..base.xi.len() {
            for source in 0..start.len() {
                for destination in 0..start.len() {
                    assert_close(
                        shifted.xi[time][source][destination],
                        base.xi[time][source][destination],
                        shift_tolerance,
                        "shift-invariant xi",
                    );
                }
            }
        }
        assert!(has_issue(&shifted_ctx, IssueCode::ForwardUnderflow));
        assert!(!has_issue(&shifted_ctx, IssueCode::ScaleFactorZero));

        let symmetric_start = Vector::from_iter([0.5, 0.5]);
        let symmetric_trans = Matrix::from_row_major(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let extreme_log_emit = vec![vec![-10_000.0; 2]; 3];
        let mut extreme_ctx = FitCtx::with_session(Session::new("hmm-fb", "extreme"));
        let extreme = scaled_forward_backward(
            &mut extreme_ctx,
            &symmetric_start,
            &symmetric_trans,
            &extreme_log_emit,
        )
        .expect("finite -10000 log-emissions remain representable in log space");
        // Measured loglik error was 0; 1.5e-11 is four ulps at |loglik|=30000.
        assert_close(extreme.loglik, -30_000.0, 1.5e-11, "extreme loglik");
        // Measured posterior error was 0; 2e-15 is four ulps at unit scale.
        for row in &extreme.gamma {
            for value in row {
                assert_close(*value, 0.5, 2.0e-15, "symmetric extreme gamma");
            }
        }
        // Measured max error was 8.63e-14 on 2026-09-02; 3.5e-13 is 4.06x.
        for slice in &extreme.xi {
            for value in slice.iter().flatten() {
                assert_close(*value, 0.25, 3.5e-13, "symmetric extreme xi");
            }
        }
        assert!(has_issue(&extreme_ctx, IssueCode::ForwardUnderflow));
        assert!(!has_issue(&extreme_ctx, IssueCode::ScaleFactorZero));
    }

    #[test]
    fn distinguishes_unreachable_max_from_zero_likelihood() {
        let start = Vector::from_iter([1.0, 0.0]);
        let identity = Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let unreachable_max = vec![vec![0.0, 0.0], vec![-1_000.0, 0.0]];
        let mut finite_ctx = FitCtx::with_session(Session::new("hmm-fb", "unreachable-max"));
        let finite = scaled_forward_backward(&mut finite_ctx, &start, &identity, &unreachable_max)
            .expect("the reachable state has finite log-likelihood");
        // Measured error was 0; 5e-13 is four ulps at |loglik|=1000.
        assert_close(finite.loglik, -1_000.0, 5.0e-13, "reachable loglik");
        assert_close(finite.gamma[1][0], 1.0, 1.0e-15, "reachable posterior");
        assert!(!has_issue(&finite_ctx, IssueCode::ScaleFactorZero));

        let impossible = vec![vec![0.0, 0.0], vec![f64::NEG_INFINITY, 0.0]];
        let mut impossible_ctx = FitCtx::with_session(Session::new("hmm-fb", "impossible"));
        assert!(
            scaled_forward_backward(&mut impossible_ctx, &start, &identity, &impossible).is_none()
        );
        assert!(has_issue(&impossible_ctx, IssueCode::ScaleFactorZero));
        assert!(!has_issue(&impossible_ctx, IssueCode::LossIsNan));
    }

    #[test]
    fn rejects_invalid_logs_shapes_and_weights() {
        let (start, trans) = two_state_parameters();
        for (label, invalid) in [("nan", f64::NAN), ("positive-infinity", f64::INFINITY)] {
            let mut ctx = FitCtx::with_session(Session::new("hmm-fb", label));
            let log_emit = vec![vec![invalid, 0.0]];
            assert!(scaled_forward_backward(&mut ctx, &start, &trans, &log_emit).is_none());
            assert!(has_issue(&ctx, IssueCode::LossIsNan));
            assert!(!has_issue(&ctx, IssueCode::ScaleFactorZero));
        }

        let mut shape_ctx = FitCtx::with_session(Session::new("hmm-fb", "shape"));
        assert!(scaled_forward_backward(&mut shape_ctx, &start, &trans, &[vec![0.0]]).is_none());
        assert!(has_issue(&shape_ctx, IssueCode::DimensionMismatch));

        let invalid_start = Vector::from_iter([-0.1, 1.1]);
        let mut weight_ctx = FitCtx::with_session(Session::new("hmm-fb", "weight"));
        assert!(scaled_forward_backward(
            &mut weight_ctx,
            &invalid_start,
            &trans,
            &[vec![0.0, 0.0]],
        )
        .is_none());
        assert!(has_issue(&weight_ctx, IssueCode::InvalidWeight));
    }

    #[test]
    fn handles_a_single_observation() {
        let start = Vector::from_iter([0.25, 0.75]);
        let trans = Matrix::from_row_major(2, 2, &[0.7, 0.3, 0.2, 0.8]);
        let log_emit = vec![vec![0.4_f64.ln(), 0.8_f64.ln()]];
        let mut ctx = FitCtx::with_session(Session::new("hmm-fb", "single"));
        let fb = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit)
            .expect("one observation has positive likelihood");

        // Measured maximum absolute error was 2.22e-16 on 2026-09-02;
        // 8e-16 is 3.60x that error.
        let tolerance = 8.0e-16;
        assert_close(fb.loglik, 0.7_f64.ln(), tolerance, "single loglik");
        assert_close(fb.gamma[0][0], 1.0 / 7.0, tolerance, "single gamma 0");
        assert_close(fb.gamma[0][1], 6.0 / 7.0, tolerance, "single gamma 1");
        assert!(fb.xi.is_empty());
        assert!(ctx.report.issues().is_empty());
    }

    #[test]
    fn rejects_empty_sequences_and_invalid_underflow_policy() {
        let (start, trans) = two_state_parameters();
        let mut empty_ctx = FitCtx::with_session(Session::new("hmm-fb", "empty"));
        assert!(scaled_forward_backward(&mut empty_ctx, &start, &trans, &[]).is_none());
        assert!(has_issue(&empty_ctx, IssueCode::EmptyMatrix));

        let mut policy_ctx = FitCtx::with_session(Session::new("hmm-fb", "policy"));
        policy_ctx.policy = Policy::default();
        policy_ctx.policy.underflow_guard = 0.0;
        assert!(
            scaled_forward_backward(&mut policy_ctx, &start, &trans, &[vec![0.0, 0.0]]).is_none()
        );
        assert!(has_issue(&policy_ctx, IssueCode::InvalidParameter));
    }
}
