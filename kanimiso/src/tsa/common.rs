//! Shared numerical kernels for the retained time-series models.

use crate::context::FitCtx;
use crate::data::Vector;
use crate::special::logsumexp;
use signlred::{scan_finite, Issue, IssueCode};

#[derive(Clone, Debug)]
pub(super) struct NormalizedLogSquares {
    pub(super) scale: f64,
    pub(super) normalized_values: Vec<f64>,
    pub(super) values: Vec<f64>,
    pub(super) log_mean_square: f64,
}

pub(super) fn normalized_log_squares(values: &[f64]) -> Option<NormalizedLogSquares> {
    if values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    if values.is_empty() {
        return Some(NormalizedLogSquares {
            scale: 0.0,
            normalized_values: Vec::new(),
            values: Vec::new(),
            log_mean_square: f64::NEG_INFINITY,
        });
    }
    let scale = values
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if scale == 0.0 {
        return Some(NormalizedLogSquares {
            scale,
            normalized_values: vec![0.0; values.len()],
            values: vec![f64::NEG_INFINITY; values.len()],
            log_mean_square: f64::NEG_INFINITY,
        });
    }
    let log_scale = scale.ln();
    let normalized_values = values.iter().map(|value| value / scale).collect::<Vec<_>>();
    let log_squares = values
        .iter()
        .map(|value| {
            if *value == 0.0 {
                f64::NEG_INFINITY
            } else {
                2.0 * (value.abs().ln() - log_scale)
            }
        })
        .collect::<Vec<_>>();
    let log_mean_square = logsumexp(&log_squares) - (values.len() as f64).ln();
    log_mean_square.is_finite().then_some(NormalizedLogSquares {
        scale,
        normalized_values,
        values: log_squares,
        log_mean_square,
    })
}

pub(super) fn gaussian_qml_profile(log_squares: &[f64], log_variances: &[f64]) -> f64 {
    log_squares
        .iter()
        .zip(log_variances)
        .map(|(log_square, log_variance)| {
            let standardized_square = if *log_square == f64::NEG_INFINITY {
                0.0
            } else {
                (log_square - log_variance).exp()
            };
            0.5 * (log_variance + standardized_square)
        })
        .sum()
}

pub(super) fn compensated_sum(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    for value in values {
        if !value.is_finite() {
            return None;
        }
        let next = sum + value;
        if !next.is_finite() {
            return None;
        }
        if sum.abs() >= value.abs() {
            correction += (sum - next) + value;
        } else {
            correction += (value - next) + sum;
        }
        if !correction.is_finite() {
            return None;
        }
        sum = next;
    }
    let corrected = sum + correction;
    corrected.is_finite().then_some(corrected)
}
fn ordered_f64_key(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & (1_u64 << 63) == 0 {
        bits | (1_u64 << 63)
    } else {
        !bits
    }
}

fn objectives_tied_within_ulps(left: f64, right: f64, maximum_ulps: usize) -> bool {
    left == right
        || (left.is_finite()
            && right.is_finite()
            && ordered_f64_key(left).abs_diff(ordered_f64_key(right)) <= maximum_ulps as u64)
}

pub(super) fn select_ranked_objective_candidate<T, Objective, Rank>(
    mut candidates: Vec<T>,
    objective_tie_ulps: usize,
    objective: Objective,
    rank: Rank,
) -> Option<T>
where
    Objective: Fn(&T) -> f64,
    Rank: Fn(&T) -> usize,
{
    let minimum_objective = candidates.iter().map(&objective).min_by(f64::total_cmp)?;
    let selected_index = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            objectives_tied_within_ulps(objective(candidate), minimum_objective, objective_tie_ulps)
        })
        .min_by_key(|(_, candidate)| rank(candidate))
        .map(|(index, _)| index)
        .expect("the finite global minimum must be eligible");
    Some(candidates.swap_remove(selected_index))
}
pub(super) fn inspect_scale_invariant_univariate(ctx: &mut FitCtx, y: &Vector) {
    ctx.report.set_sample_shape(y.len(), 1);
    if y.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("univariate series is empty")
                .metric("n", 0.0)
                .build(),
        );
        return;
    }
    if let Some(issue) = scan_finite(y.as_slice()).to_issue("y") {
        ctx.push(issue);
    }
}
