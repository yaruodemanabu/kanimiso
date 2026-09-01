//! Generic Baum–Welch using [`Emission::{accumulate, maximize}`].

use super::diagnostics::{diagnose_chain, occupancy};
use super::emission::Emission;
use super::forward_backward::{forward_backward, log_prob};
use super::model::HiddenMarkovModel;
use crate::context::FitCtx;
use signlred::Result;

pub(crate) fn baum_welch<E: Emission>(
    model: &mut HiddenMarkovModel<E>,
    obs: &[E::Observation],
    max_iter: usize,
    ctx: &mut FitCtx,
) -> Result<Vec<f64>> {
    let k = model.n_states();
    let mut history = Vec::new();
    for _ in 0..max_iter.max(1) {
        let (log_start, log_trans) = model.log_tables();
        let log_emit = model.log_emit_seq(obs);
        let Some(fb) = forward_backward(ctx, &log_start, &log_trans, &log_emit) else {
            break;
        };
        history.push(fb.loglik);

        let mut start = vec![0.0_f64; k];
        for j in 0..k {
            start[j] = fb.gamma[0][j];
        }
        let mut trans = vec![vec![0.0_f64; k]; k];
        for xi_t in &fb.xi {
            for i in 0..k {
                for j in 0..k {
                    trans[i][j] += xi_t[i][j];
                }
            }
        }
        renormalize_vec(&mut start, ctx.policy.transition_floor);
        renormalize_rows(&mut trans, ctx.policy.transition_floor);

        for (j, emission) in model.emissions.iter_mut().enumerate() {
            let mut stats = E::SufficientStats::default();
            for (t, o) in obs.iter().enumerate() {
                let w = fb.gamma[t][j];
                emission.accumulate(o, w, &mut stats);
            }
            emission.maximize(&stats, ctx)?;
        }
        model.initial = start;
        model.transition = trans;
        diagnose_chain(
            ctx,
            &model.initial,
            &model.transition,
            &occupancy(&fb.gamma, k),
        );
    }
    Ok(history)
}

fn renormalize_vec(v: &mut [f64], floor: f64) {
    let mut s = 0.0_f64;
    for p in v.iter_mut() {
        *p = p.max(floor);
        s += *p;
    }
    if s > 0.0 {
        for p in v.iter_mut() {
            *p /= s;
        }
    } else if !v.is_empty() {
        let u = 1.0 / v.len() as f64;
        v.fill(u);
    }
}

fn renormalize_rows(m: &mut [Vec<f64>], floor: f64) {
    for row in m.iter_mut() {
        renormalize_vec(row, floor);
    }
}

pub(crate) fn log_tables(initial: &[f64], transition: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let log_start: Vec<f64> = initial.iter().copied().map(log_prob).collect();
    let log_trans: Vec<Vec<f64>> = transition
        .iter()
        .map(|row| row.iter().copied().map(log_prob).collect())
        .collect();
    (log_start, log_trans)
}
