//! Log-space forward–backward (AGENTS.md §4.3).
//!
//! Every time step stays in the log domain. `logsumexp` subtracts the per-slice
//! maximum before `exp`, so a column of `log_emit[t][·] = −1e4` does not collapse
//! to a zero scale factor.

use crate::context::FitCtx;
use signlred::{Issue, IssueCode};

/// Shared log-sum-exp (one implementation for the v0.2 HMM core).
pub(crate) fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0_f64;
    for &v in xs {
        s += (v - m).exp();
    }
    m + s.ln()
}

pub(crate) fn log_prob(p: f64) -> f64 {
    if p > 0.0 && p.is_finite() {
        p.ln()
    } else {
        f64::NEG_INFINITY
    }
}

/// Posterior occupancies and pairwise transitions from one sequence.
#[derive(Clone, Debug)]
pub(crate) struct ForwardBackward {
    pub loglik: f64,
    pub gamma: Vec<Vec<f64>>,
    pub xi: Vec<Vec<Vec<f64>>>,
}

/// Full log-space forward–backward.
///
/// `log_emit[t][j]` is `log p(o_t | state j)` (`−∞` outside support).
pub(crate) fn forward_backward(
    ctx: &mut FitCtx,
    log_start: &[f64],
    log_trans: &[Vec<f64>],
    log_emit: &[Vec<f64>],
) -> Option<ForwardBackward> {
    let t_len = log_emit.len();
    let s = log_start.len();
    if t_len == 0 || s == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("HMM forward–backward on an empty sequence")
                .build(),
        );
        return None;
    }
    let mut log_alpha = vec![vec![f64::NEG_INFINITY; s]; t_len];
    for j in 0..s {
        log_alpha[0][j] = log_start[j] + log_emit[0][j];
    }
    let mut buf = vec![f64::NEG_INFINITY; s];
    for t in 1..t_len {
        for j in 0..s {
            for i in 0..s {
                buf[i] = log_alpha[t - 1][i] + log_trans[i][j];
            }
            log_alpha[t][j] = logsumexp(&buf) + log_emit[t][j];
        }
    }
    let loglik = logsumexp(&log_alpha[t_len - 1]);
    if !loglik.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::ScaleFactorZero)
                .message("log-space forward mass is −∞; the sequence is impossible under this HMM")
                .build(),
        );
        ctx.push(
            Issue::builder(IssueCode::ForwardUnderflow)
                .message("forward log-mass vanished")
                .build(),
        );
        return None;
    }
    // Equivalent linear-domain scale; warn when the mean per-step mass is tiny.
    let mean_scale = (loglik / t_len as f64).exp();
    if mean_scale < ctx.policy.underflow_guard {
        ctx.push(
            Issue::builder(IssueCode::ForwardUnderflow)
                .message(format!(
                    "mean per-step forward scale {mean_scale:.3e} is below Policy::underflow_guard"
                ))
                .metric("mean_scale", mean_scale)
                .metric("loglik", loglik)
                .build(),
        );
    }

    let mut log_beta = vec![vec![0.0_f64; s]; t_len];
    for t in (0..t_len - 1).rev() {
        for i in 0..s {
            for j in 0..s {
                buf[j] = log_trans[i][j] + log_emit[t + 1][j] + log_beta[t + 1][j];
            }
            log_beta[t][i] = logsumexp(&buf);
        }
    }

    let mut gamma = vec![vec![0.0_f64; s]; t_len];
    for t in 0..t_len {
        let mut nrm = 0.0_f64;
        for j in 0..s {
            let g = (log_alpha[t][j] + log_beta[t][j] - loglik).exp();
            gamma[t][j] = if g.is_finite() { g } else { 0.0 };
            nrm += gamma[t][j];
        }
        if nrm > 0.0 {
            for j in 0..s {
                gamma[t][j] /= nrm;
            }
        }
    }

    let mut xi = vec![vec![vec![0.0_f64; s]; s]; t_len.saturating_sub(1)];
    for t in 0..t_len.saturating_sub(1) {
        let mut nrm = 0.0_f64;
        for i in 0..s {
            for j in 0..s {
                let v =
                    (log_alpha[t][i] + log_trans[i][j] + log_emit[t + 1][j] + log_beta[t + 1][j]
                        - loglik)
                        .exp();
                xi[t][i][j] = if v.is_finite() { v } else { 0.0 };
                nrm += xi[t][i][j];
            }
        }
        if nrm > 0.0 {
            for i in 0..s {
                for j in 0..s {
                    xi[t][i][j] /= nrm;
                }
            }
        }
    }

    Some(ForwardBackward { loglik, gamma, xi })
}

/// Sequence log-likelihood (no posteriors).
pub(crate) fn log_likelihood(
    ctx: &mut FitCtx,
    log_start: &[f64],
    log_trans: &[Vec<f64>],
    log_emit: &[Vec<f64>],
) -> f64 {
    forward_backward(ctx, log_start, log_trans, log_emit)
        .map(|fb| fb.loglik)
        .unwrap_or(f64::NEG_INFINITY)
}
