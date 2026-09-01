//! [`HiddenMarkovModel<E>`] — one chain, one [`Emission`] family.

use super::baum_welch::{baum_welch, log_tables};
use super::emission::Emission;
use super::forward_backward::log_likelihood;
use super::viterbi::viterbi_path;
use crate::context::FitCtx;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Qualified, Result};

/// Hidden Markov model with one emission object per state.
#[derive(Clone, Debug)]
pub(crate) struct HiddenMarkovModel<E> {
    /// Start distribution `π` (length `K`).
    pub initial: Vec<f64>,
    /// Row-stochastic transitions `A` (`K` × `K`).
    pub transition: Vec<Vec<f64>>,
    /// Per-state emission.
    pub emissions: Vec<E>,
}

impl<E: Emission> HiddenMarkovModel<E> {
    /// Assemble a model. Dimensions are checked on the first scoring call.
    pub(crate) fn new(initial: Vec<f64>, transition: Vec<Vec<f64>>, emissions: Vec<E>) -> Self {
        Self {
            initial,
            transition,
            emissions,
        }
    }

    pub(crate) fn n_states(&self) -> usize {
        self.emissions.len()
    }

    pub(crate) fn log_tables(&self) -> (Vec<f64>, Vec<Vec<f64>>) {
        log_tables(&self.initial, &self.transition)
    }

    pub(crate) fn log_emit_seq(&self, obs: &[E::Observation]) -> Vec<Vec<f64>> {
        obs.iter()
            .map(|o| self.emissions.iter().map(|e| e.log_prob(o)).collect())
            .collect()
    }

    fn check_shape(&self, ctx: &mut FitCtx, t_len: usize) -> bool {
        let k = self.n_states();
        if k == 0 || t_len == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("HMM has no states or the sequence is empty")
                    .build(),
            );
            return false;
        }
        if self.initial.len() != k
            || self.transition.len() != k
            || self.transition.iter().any(|row| row.len() != k)
        {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("π, A, and emissions disagree on the number of states")
                    .build(),
            );
            return false;
        }
        true
    }

    /// Sequence log-likelihood via log-space forward.
    pub(crate) fn log_likelihood(
        &self,
        obs: &[E::Observation],
        session: &Session,
    ) -> Result<Qualified<f64>> {
        let mut ctx = FitCtx::with_session(session.child("score"));
        if !self.check_shape(&mut ctx, obs.len()) {
            return ctx.finish(f64::NEG_INFINITY);
        }
        let (log_start, log_trans) = self.log_tables();
        let log_emit = self.log_emit_seq(obs);
        if log_emit.iter().flatten().any(|v| v.is_nan()) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("Emission::log_prob returned NaN")
                    .build(),
            );
        }
        let ll = log_likelihood(&mut ctx, &log_start, &log_trans, &log_emit);
        ctx.finish(ll)
    }

    /// Viterbi state path.
    pub(crate) fn decode(
        &self,
        obs: &[E::Observation],
        session: &Session,
    ) -> Result<Qualified<Vec<usize>>> {
        let mut ctx = FitCtx::with_session(session.child("decode"));
        if !self.check_shape(&mut ctx, obs.len()) {
            return ctx.finish(Vec::new());
        }
        let (log_start, log_trans) = self.log_tables();
        let log_emit = self.log_emit_seq(obs);
        let (path, _) = viterbi_path(&log_start, &log_trans, &log_emit);
        ctx.finish(path)
    }

    /// Baum–Welch. Returns a fitted clone; `self` is unchanged (`fit` is `&self`).
    pub(crate) fn fit(
        &self,
        obs: &[E::Observation],
        max_iter: usize,
        session: &Session,
    ) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.child("fit"));
        if !self.check_shape(&mut ctx, obs.len()) {
            return ctx.finish(self.clone());
        }
        let mut fitted = self.clone();
        let _history = baum_welch(&mut fitted, obs, max_iter, &mut ctx)?;
        ctx.finish(fitted)
    }
}
