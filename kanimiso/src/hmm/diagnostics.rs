//! Chain diagnostics (absorbing / unreachable / degenerate).

use crate::context::FitCtx;
use signlred::{Issue, IssueCode};

pub(crate) fn diagnose_chain(ctx: &mut FitCtx, start: &[f64], trans: &[Vec<f64>], occup: &[f64]) {
    let s = start.len();
    let floor = ctx.policy.transition_floor;
    for j in 0..s {
        let incoming: f64 = (0..s)
            .map(|i| {
                trans
                    .get(i)
                    .and_then(|row| row.get(j))
                    .copied()
                    .unwrap_or(0.0)
            })
            .sum();
        if start.get(j).copied().unwrap_or(0.0) <= floor && incoming <= floor * s as f64 {
            ctx.push(
                Issue::builder(IssueCode::UnreachableState)
                    .message(format!("state {j} is unreachable from π and A"))
                    .metric("state", j as f64)
                    .build(),
            );
        }
        if occup.get(j).copied().unwrap_or(0.0) <= floor {
            ctx.push(
                Issue::builder(IssueCode::UnreachableState)
                    .message(format!("state {j} received ~0 posterior occupancy"))
                    .metric("state", j as f64)
                    .metric("occupancy", occup.get(j).copied().unwrap_or(0.0))
                    .build(),
            );
        }
        let diag = trans
            .get(j)
            .and_then(|row| row.get(j))
            .copied()
            .unwrap_or(0.0);
        let off: f64 = (0..s)
            .filter(|&k| k != j)
            .map(|k| {
                trans
                    .get(j)
                    .and_then(|row| row.get(k))
                    .copied()
                    .unwrap_or(0.0)
            })
            .sum();
        if diag >= 1.0 - floor && off <= floor {
            ctx.push(
                Issue::builder(IssueCode::AbsorbingStateOnly)
                    .message(format!("state {j} is absorbing (A[{j},{j}]={diag:.6})"))
                    .metric("state", j as f64)
                    .metric("self_transition", diag)
                    .build(),
            );
        }
    }
}

pub(crate) fn occupancy(gamma: &[Vec<f64>], n_states: usize) -> Vec<f64> {
    let mut occup = vec![0.0_f64; n_states];
    for row in gamma {
        for (j, g) in row.iter().enumerate().take(n_states) {
            occup[j] += *g;
        }
    }
    occup
}
