//! Canonical Viterbi maximum-path recursion.

use crate::data::{Matrix, Vector};

/// Return the maximum-probability state path and its log probability.
///
/// The public model runs forward–backward preflight before this kernel, so all
/// dimensions and weights are valid and the sequence has at least one reachable
/// finite path. Strict comparisons make ties deterministic: the lowest-indexed
/// predecessor and terminal state win.
pub(super) fn viterbi_path(start: &Vector, trans: &Matrix, log_emit: &[Vec<f64>]) -> (Vector, f64) {
    let time_count = log_emit.len();
    let state_count = start.len();
    if time_count == 0 || state_count == 0 {
        return (Vector::zeros(time_count), f64::NEG_INFINITY);
    }

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
    let mut delta = vec![vec![f64::NEG_INFINITY; state_count]; time_count];
    let mut predecessor = vec![vec![0usize; state_count]; time_count];
    for state in 0..state_count {
        let log_start = if start[state] > 0.0 {
            start[state].ln()
        } else {
            f64::NEG_INFINITY
        };
        delta[0][state] = log_start + log_emit[0][state];
    }
    for time in 1..time_count {
        for destination in 0..state_count {
            let mut best = f64::NEG_INFINITY;
            let mut argmax = 0usize;
            for source in 0..state_count {
                let candidate = delta[time - 1][source] + log_trans[source][destination];
                if candidate > best {
                    best = candidate;
                    argmax = source;
                }
            }
            delta[time][destination] = best + log_emit[time][destination];
            predecessor[time][destination] = argmax;
        }
    }

    let mut last = 0usize;
    let mut best = f64::NEG_INFINITY;
    for state in 0..state_count {
        if delta[time_count - 1][state] > best {
            best = delta[time_count - 1][state];
            last = state;
        }
    }
    let mut path = vec![0usize; time_count];
    path[time_count - 1] = last;
    for time in (1..time_count).rev() {
        path[time - 1] = predecessor[time][path[time]];
    }
    (
        Vector::from_iter(path.iter().map(|state| *state as f64)),
        best,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force_argmax(
        start: &Vector,
        trans: &Matrix,
        log_emit: &[Vec<f64>],
    ) -> (Vec<usize>, f64) {
        let time_count = log_emit.len();
        let state_count = start.len();
        let path_count = (0..time_count).fold(1usize, |count, _| count * state_count);
        let mut best_path = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        for encoded in 0..path_count {
            let mut remainder = encoded;
            let mut path = vec![0usize; time_count];
            for state in &mut path {
                *state = remainder % state_count;
                remainder /= state_count;
            }
            let mut score = if start[path[0]] > 0.0 {
                start[path[0]].ln() + log_emit[0][path[0]]
            } else {
                f64::NEG_INFINITY
            };
            for time in 1..time_count {
                let weight = trans.get(path[time - 1], path[time]);
                let log_transition = if weight > 0.0 {
                    weight.ln()
                } else {
                    f64::NEG_INFINITY
                };
                score += log_transition + log_emit[time][path[time]];
            }
            if score > best_score {
                best_score = score;
                best_path = path;
            }
        }
        (best_path, best_score)
    }

    #[test]
    fn matches_brute_force_argmax_and_path_log_probability() {
        let start = Vector::from_iter([0.6, 0.4]);
        let trans = Matrix::from_row_major(2, 2, &[0.7, 0.3, 0.2, 0.8]);
        let emission = [[0.9_f64, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]];
        let log_emit: Vec<Vec<f64>> = emission
            .iter()
            .map(|row| row.iter().map(|value| value.ln() - 1_000.0).collect())
            .collect();
        let (expected_path, expected_score) = brute_force_argmax(&start, &trans, &log_emit);
        let (actual_path, actual_score) = viterbi_path(&start, &trans, &log_emit);

        assert_eq!(
            actual_path.as_slice(),
            expected_path
                .iter()
                .map(|state| *state as f64)
                .collect::<Vec<_>>()
        );
        // Measured absolute discrepancy was 0 on 2026-09-03; four ulps at
        // this score magnitude bound the different dynamic-programming order.
        let tolerance = 4.0 * f64::EPSILON * expected_score.abs().max(1.0);
        assert!(
            (actual_score - expected_score).abs() <= tolerance,
            "actual={actual_score:.17e}, expected={expected_score:.17e}, tolerance={tolerance:.3e}"
        );
    }
}
