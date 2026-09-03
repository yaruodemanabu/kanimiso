//! Generic hidden Markov model orchestration.
//!
//! Emission-specific probability and sufficient-statistic calculations live in
//! [`super::emission`]. This module owns the single chain-level validation,
//! Baum–Welch, scoring, and decoding implementation.

use super::emission::Emission;
use super::forward_backward::scaled_forward_backward;
use super::viterbi::viterbi_path;
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use ojizou_san::Session;
use signlred::{Failure, Issue, IssueCode, Policy, Qualified, Result};

const ALGORITHM: &str = "hmm.HiddenMarkovModel";

/// A hidden Markov model parameterized by one emission family.
///
/// All states use the same emission type, while each element of `emissions`
/// stores that state's parameters. Chain probabilities are accepted only when
/// they already satisfy the configured probability contract; construction and
/// fitting never silently normalize, floor, or clamp them.
#[derive(Clone, Debug)]
pub struct HiddenMarkovModel<E: Emission> {
    initial: Vector,
    transition: Matrix,
    emissions: Vec<E>,
    max_iter: usize,
    left_right: bool,
    policy: Policy,
}

impl<E: Emission> HiddenMarkovModel<E> {
    /// Construct a validated generic HMM.
    pub fn new(
        initial: Vector,
        transition: Matrix,
        emissions: Vec<E>,
        max_iter: usize,
        left_right: bool,
        policy: Policy,
    ) -> Result<Self> {
        let model = Self {
            initial,
            transition,
            emissions,
            max_iter,
            left_right,
            policy,
        };
        if let Some(issue) = model.validation_issue() {
            return Err(Failure::from_issue(ALGORITHM, "new", issue));
        }
        Ok(model)
    }

    /// Initial-state probability vector.
    pub fn initial(&self) -> &Vector {
        &self.initial
    }

    /// Row-stochastic state-transition matrix.
    pub fn transition(&self) -> &Matrix {
        &self.transition
    }

    /// State-indexed emission parameters.
    pub fn emissions(&self) -> &[E] {
        &self.emissions
    }

    /// Maximum number of Baum–Welch iterations.
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Whether the model forbids transitions to lower-numbered states.
    pub fn left_right(&self) -> bool {
        self.left_right
    }

    /// Numerical and reporting policy used by every operation.
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// Fit a cloned model by Baum–Welch, leaving `self` unchanged.
    pub fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = self.context(session, "fit", x);
        let observations = match Self::observations(x, &mut ctx)? {
            Some(observations) => observations,
            None => return ctx.finish(self.clone()),
        };
        let mut fitted = self.clone();
        let state_count = fitted.emissions.len();

        for iteration in 0..fitted.max_iter {
            let log_emit = match fitted.log_emissions(&observations, &mut ctx) {
                Some(log_emit) => log_emit,
                None => return ctx.finish(fitted),
            };
            let posterior = match scaled_forward_backward(
                &mut ctx,
                &fitted.initial,
                &fitted.transition,
                &log_emit,
            ) {
                Some(posterior) => posterior,
                None => return ctx.finish(fitted),
            };
            ctx.session.step(iteration as u64, -posterior.loglik, None);

            let mut statistics: Vec<E::SufficientStats> =
                (0..state_count).map(|_| Default::default()).collect();
            let mut occupancy = vec![0.0; state_count];
            for (time, observation) in observations.iter().enumerate() {
                for state in 0..state_count {
                    let weight = posterior.gamma[time][state];
                    occupancy[state] += weight;
                    if weight > 0.0 {
                        fitted.emissions[state].accumulate(
                            observation,
                            weight,
                            &mut statistics[state],
                        );
                    }
                }
            }
            for state in 0..state_count {
                if occupancy[state] == 0.0 {
                    ctx.push(
                        Issue::builder(IssueCode::UnreachableState)
                            .message(format!(
                                "state {state} received zero posterior occupancy; its emission parameters were retained"
                            ))
                            .metric("state", state as f64)
                            .metric("occupancy", occupancy[state])
                            .build(),
                    );
                    continue;
                }
                if let Err(failure) = fitted.emissions[state].maximize(&statistics[state], &mut ctx)
                {
                    return Err(Self::record_emission_failure(&mut ctx, failure));
                }
            }

            fitted.initial = Vector::from_slice(&posterior.gamma[0]);
            if observations.len() > 1 {
                for source in 0..state_count {
                    let denominator: f64 = posterior.gamma[..observations.len() - 1]
                        .iter()
                        .map(|row| row[source])
                        .sum();
                    if denominator == 0.0 {
                        ctx.push(
                            Issue::builder(IssueCode::UnreachableState)
                                .message(format!(
                                    "state {source} had zero transition occupancy; transition row was retained"
                                ))
                                .metric("state", source as f64)
                                .metric("occupancy", denominator)
                                .build(),
                        );
                        continue;
                    }
                    for destination in 0..state_count {
                        let numerator: f64 = posterior
                            .xi
                            .iter()
                            .map(|slice| slice[source][destination])
                            .sum();
                        fitted
                            .transition
                            .set(source, destination, numerator / denominator);
                    }
                }
            }
            if fitted.left_right {
                fitted.enforce_left_right();
            }
            if let Some(issue) = fitted.validation_issue() {
                ctx.push(issue);
                return ctx.finish(fitted);
            }
        }

        ctx.finish(fitted)
    }

    /// Return the sequence log-likelihood under the current parameters.
    pub fn score(&self, x: &Matrix, session: &Session) -> Result<Qualified<f64>> {
        let mut ctx = self.context(session, "score", x);
        let observations = match Self::observations(x, &mut ctx)? {
            Some(observations) => observations,
            None => return ctx.finish(f64::NEG_INFINITY),
        };
        let log_emit = match self.log_emissions(&observations, &mut ctx) {
            Some(log_emit) => log_emit,
            None => return ctx.finish(f64::NEG_INFINITY),
        };
        let posterior =
            match scaled_forward_backward(&mut ctx, &self.initial, &self.transition, &log_emit) {
                Some(posterior) => posterior,
                None => return ctx.finish(f64::NEG_INFINITY),
            };
        ctx.finish(posterior.loglik)
    }

    /// Return causal state probabilities for every observation.
    ///
    /// Row `t` is \(P(s_t \mid y_{0:t})\). The canonical normalized
    /// forward recursion computes every row in one \(O(TK^2)\) pass; this
    /// method does not refit the model or rerun each observation prefix.
    pub fn filter_probabilities(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = self.context(session, "filter_probabilities", x);
        let observations = match Self::observations(x, &mut ctx)? {
            Some(observations) => observations,
            None => return ctx.finish(Matrix::zeros(x.nrows(), self.initial.len())),
        };
        let log_emit = match self.log_emissions(&observations, &mut ctx) {
            Some(log_emit) => log_emit,
            None => return ctx.finish(Matrix::zeros(x.nrows(), self.initial.len())),
        };
        let posterior =
            match scaled_forward_backward(&mut ctx, &self.initial, &self.transition, &log_emit) {
                Some(posterior) => posterior,
                None => return ctx.finish(Matrix::zeros(x.nrows(), self.initial.len())),
            };
        ctx.finish(Matrix::from_fn(
            x.nrows(),
            self.initial.len(),
            |time, state| posterior.filtered[time][state],
        ))
    }

    /// Decode the maximum-probability hidden-state path.
    ///
    /// Forward–backward is evaluated first so a genuinely impossible sequence
    /// is reported instead of being converted into an arbitrary Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = self.context(session, "decode", x);
        let observations = match Self::observations(x, &mut ctx)? {
            Some(observations) => observations,
            None => return ctx.finish(Vector::zeros(x.nrows())),
        };
        let log_emit = match self.log_emissions(&observations, &mut ctx) {
            Some(log_emit) => log_emit,
            None => return ctx.finish(Vector::zeros(x.nrows())),
        };
        if scaled_forward_backward(&mut ctx, &self.initial, &self.transition, &log_emit).is_none() {
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let (path, _) = viterbi_path(&self.initial, &self.transition, &log_emit);
        ctx.finish(path)
    }

    fn context(&self, session: &Session, operation: &str, x: &Matrix) -> FitCtx {
        let mut ctx = FitCtx::with_session(session.child(operation));
        ctx.policy = self.policy.clone();
        ctx.report.set_sample_shape(x.nrows(), x.ncols());
        ctx
    }

    fn observations(x: &Matrix, ctx: &mut FitCtx) -> Result<Option<Vec<E::Observation>>> {
        if x.nrows() == 0 || x.ncols() == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("HMM observations must have at least one row and one column")
                    .build(),
            );
            return Ok(None);
        }
        for row in 0..x.nrows() {
            for column in 0..x.ncols() {
                if !x.get(row, column).is_finite() {
                    ctx.push(
                        Issue::builder(IssueCode::NonFiniteInput)
                            .message(format!(
                                "HMM observation at row {row}, column {column} is not finite"
                            ))
                            .metric("row", row as f64)
                            .metric("column", column as f64)
                            .build(),
                    );
                    return Ok(None);
                }
            }
        }
        let observations = match E::observations(x) {
            Ok(observations) => observations,
            Err(failure) => return Err(Self::record_emission_failure(ctx, failure)),
        };
        if observations.len() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!(
                        "emission parser returned {} observations for {} matrix rows",
                        observations.len(),
                        x.nrows()
                    ))
                    .metric("observations", observations.len() as f64)
                    .metric("rows", x.nrows() as f64)
                    .build(),
            );
            return Ok(None);
        }
        Ok(Some(observations))
    }

    fn log_emissions(
        &self,
        observations: &[E::Observation],
        ctx: &mut FitCtx,
    ) -> Option<Vec<Vec<f64>>> {
        let mut values = vec![vec![0.0; self.emissions.len()]; observations.len()];
        for (time, observation) in observations.iter().enumerate() {
            for (state, emission) in self.emissions.iter().enumerate() {
                let value = emission.log_prob(observation);
                if value.is_nan() || value == f64::INFINITY {
                    ctx.push(
                        Issue::builder(IssueCode::NonFiniteOutput)
                            .message(format!(
                                "emission log-probability is NaN or positive infinity at time {time}, state {state}"
                            ))
                            .metric("time", time as f64)
                            .metric("state", state as f64)
                            .build(),
                    );
                    return None;
                }
                values[time][state] = value;
            }
        }
        Some(values)
    }

    fn enforce_left_right(&mut self) {
        self.initial[0] = 1.0;
        for state in 1..self.initial.len() {
            self.initial[state] = 0.0;
        }
        for source in 0..self.transition.nrows() {
            for destination in 0..source {
                self.transition.set(source, destination, 0.0);
            }
        }
    }

    fn record_emission_failure(ctx: &mut FitCtx, failure: Failure) -> Failure {
        let primary = failure.primary;
        ctx.report.merge(failure.report);
        let combined = Failure {
            primary,
            report: ctx.report.clone(),
        };
        ctx.session.finish_err(&combined);
        combined
    }

    fn validation_issue(&self) -> Option<Issue> {
        if !self.policy.probability_sum_tol.is_finite()
            || self.policy.probability_sum_tol <= 0.0
            || self.policy.probability_sum_tol >= 1.0
        {
            return Some(
                Issue::builder(IssueCode::InvalidParameter)
                    .message(
                        "Policy::probability_sum_tol must be finite, positive, and less than one",
                    )
                    .build(),
            );
        }
        if !self.policy.underflow_guard.is_finite() || self.policy.underflow_guard <= 0.0 {
            return Some(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("Policy::underflow_guard must be finite and positive")
                    .build(),
            );
        }
        if self.max_iter == 0 {
            return Some(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("max_iter must be positive")
                    .build(),
            );
        }
        let state_count = self.emissions.len();
        if state_count == 0 {
            return Some(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("an HMM must contain at least one emission state")
                    .build(),
            );
        }
        if self.initial.len() != state_count {
            return Some(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!(
                        "initial probability length {} does not match {state_count} emissions",
                        self.initial.len()
                    ))
                    .build(),
            );
        }
        if self.transition.shape() != (state_count, state_count) {
            return Some(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!(
                        "transition matrix is {}x{} but must be {state_count}x{state_count}",
                        self.transition.nrows(),
                        self.transition.ncols()
                    ))
                    .build(),
            );
        }
        if self
            .initial
            .as_slice()
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Some(
                Issue::builder(IssueCode::InvalidWeight)
                    .message("initial probabilities must be finite and non-negative")
                    .build(),
            );
        }
        for source in 0..state_count {
            for destination in 0..state_count {
                let weight = self.transition.get(source, destination);
                if !weight.is_finite() || weight < 0.0 {
                    return Some(
                        Issue::builder(IssueCode::InvalidWeight)
                            .message(format!(
                                "transition weight at row {source}, column {destination} must be finite and non-negative"
                            ))
                            .metric("row", source as f64)
                            .metric("column", destination as f64)
                            .build(),
                    );
                }
            }
        }
        let initial_sum: f64 = self.initial.as_slice().iter().sum();
        if !initial_sum.is_finite() || (initial_sum - 1.0).abs() > self.policy.probability_sum_tol {
            return Some(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!(
                        "initial probabilities sum to {initial_sum}, expected one within probability_sum_tol"
                    ))
                    .metric("sum", initial_sum)
                    .metric("tolerance", self.policy.probability_sum_tol)
                    .build(),
            );
        }
        for source in 0..state_count {
            let row_sum: f64 = (0..state_count)
                .map(|destination| self.transition.get(source, destination))
                .sum();
            if !row_sum.is_finite() || (row_sum - 1.0).abs() > self.policy.probability_sum_tol {
                return Some(
                    Issue::builder(IssueCode::InvalidWeight)
                        .message(format!(
                            "transition row {source} sums to {row_sum}, expected one within probability_sum_tol"
                        ))
                        .metric("row", source as f64)
                        .metric("sum", row_sum)
                        .metric("tolerance", self.policy.probability_sum_tol)
                        .build(),
                );
            }
        }
        if self.left_right {
            if self.initial[0] != 1.0
                || self.initial.as_slice()[1..]
                    .iter()
                    .any(|weight| *weight != 0.0)
            {
                return Some(
                    Issue::builder(IssueCode::InvalidWeight)
                        .message(
                            "a left-right HMM requires initial probabilities [1, 0, ...] exactly",
                        )
                        .build(),
                );
            }
            for source in 0..state_count {
                for destination in 0..source {
                    if self.transition.get(source, destination) != 0.0 {
                        return Some(
                            Issue::builder(IssueCode::InvalidWeight)
                                .message(format!(
                                    "left-right transition ({source}, {destination}) must be exactly zero"
                                ))
                                .metric("row", source as f64)
                                .metric("column", destination as f64)
                                .build(),
                        );
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::super::emission::{Emission, GaussianEmission};
    use super::*;

    #[derive(Clone, Debug)]
    struct FixedEmission(f64);

    impl Emission for FixedEmission {
        type Observation = f64;
        type SufficientStats = ();

        fn observations(x: &Matrix) -> Result<Vec<Self::Observation>> {
            Ok((0..x.nrows()).map(|row| x.get(row, 0)).collect())
        }

        fn log_prob(&self, _obs: &Self::Observation) -> f64 {
            self.0
        }

        fn accumulate(
            &self,
            _obs: &Self::Observation,
            _weight: f64,
            _stats: &mut Self::SufficientStats,
        ) {
        }

        fn maximize(&mut self, _stats: &Self::SufficientStats, _ctx: &mut FitCtx) -> Result<()> {
            Ok(())
        }
    }

    fn emissions() -> Vec<GaussianEmission> {
        vec![
            GaussianEmission::new(Vector::from_slice(&[-2.0]), Vector::from_slice(&[1.0]))
                .expect("valid state-zero Gaussian"),
            GaussianEmission::new(Vector::from_slice(&[2.0]), Vector::from_slice(&[1.0]))
                .expect("valid state-one Gaussian"),
        ]
    }

    fn observations() -> Matrix {
        Matrix::from_row_major(
            12,
            1,
            &[
                -2.4, -1.8, -2.2, -1.7, -2.1, -1.9, 1.7, 2.2, 1.8, 2.4, 1.9, 2.1,
            ],
        )
    }

    fn symmetric_model(max_iter: usize) -> HiddenMarkovModel<GaussianEmission> {
        HiddenMarkovModel::new(
            Vector::from_slice(&[0.5, 0.5]),
            Matrix::from_row_major(2, 2, &[0.8, 0.2, 0.2, 0.8]),
            emissions(),
            max_iter,
            false,
            Policy::default(),
        )
        .expect("valid symmetric HMM")
    }

    fn assert_new_fails(
        initial: Vector,
        transition: Matrix,
        emissions: Vec<GaussianEmission>,
        max_iter: usize,
        left_right: bool,
        policy: Policy,
        expected: IssueCode,
    ) {
        let failure =
            HiddenMarkovModel::new(initial, transition, emissions, max_iter, left_right, policy)
                .expect_err("invalid HMM must be rejected");
        assert_eq!(failure.primary.code, expected);
    }

    #[test]
    fn constructor_rejects_invalid_shape_weights_sums_and_topology() {
        let valid_transition = Matrix::from_row_major(2, 2, &[0.8, 0.2, 0.2, 0.8]);
        assert_new_fails(
            Vector::zeros(0),
            Matrix::zeros(0, 0),
            Vec::new(),
            1,
            false,
            Policy::default(),
            IssueCode::InvalidParameter,
        );
        assert_new_fails(
            Vector::from_slice(&[1.0]),
            valid_transition.clone(),
            emissions(),
            1,
            false,
            Policy::default(),
            IssueCode::DimensionMismatch,
        );
        assert_new_fails(
            Vector::from_slice(&[0.5, 0.5]),
            Matrix::zeros(2, 1),
            emissions(),
            1,
            false,
            Policy::default(),
            IssueCode::DimensionMismatch,
        );
        assert_new_fails(
            Vector::from_slice(&[-0.5, 1.5]),
            valid_transition.clone(),
            emissions(),
            1,
            false,
            Policy::default(),
            IssueCode::InvalidWeight,
        );
        assert_new_fails(
            Vector::from_slice(&[f64::NAN, 1.0]),
            valid_transition.clone(),
            emissions(),
            1,
            false,
            Policy::default(),
            IssueCode::InvalidWeight,
        );
        assert_new_fails(
            Vector::from_slice(&[0.4, 0.4]),
            valid_transition.clone(),
            emissions(),
            1,
            false,
            Policy::default(),
            IssueCode::InvalidWeight,
        );
        assert_new_fails(
            Vector::from_slice(&[0.5, 0.5]),
            Matrix::from_row_major(2, 2, &[0.8, 0.2, 0.3, 0.8]),
            emissions(),
            1,
            false,
            Policy::default(),
            IssueCode::InvalidWeight,
        );
        assert_new_fails(
            Vector::from_slice(&[0.5, 0.5]),
            valid_transition.clone(),
            emissions(),
            0,
            false,
            Policy::default(),
            IssueCode::InvalidParameter,
        );

        let mut invalid_policy = Policy::default();
        invalid_policy.probability_sum_tol = 0.0;
        assert_new_fails(
            Vector::from_slice(&[0.5, 0.5]),
            valid_transition.clone(),
            emissions(),
            1,
            false,
            invalid_policy,
            IssueCode::InvalidParameter,
        );
        let mut overbroad_policy = Policy::default();
        overbroad_policy.probability_sum_tol = 1.0;
        assert_new_fails(
            Vector::from_slice(&[0.5, 0.5]),
            valid_transition.clone(),
            emissions(),
            1,
            false,
            overbroad_policy,
            IssueCode::InvalidParameter,
        );
        assert_new_fails(
            Vector::from_slice(&[0.5, 0.5]),
            Matrix::from_row_major(2, 2, &[0.8, 0.2, 0.0, 1.0]),
            emissions(),
            1,
            true,
            Policy::default(),
            IssueCode::InvalidWeight,
        );
        assert_new_fails(
            Vector::from_slice(&[1.0, 0.0]),
            Matrix::from_row_major(2, 2, &[0.8, 0.2, 0.1, 0.9]),
            emissions(),
            1,
            true,
            Policy::default(),
            IssueCode::InvalidWeight,
        );
    }

    #[test]
    fn gaussian_fit_does_not_reduce_sequence_score() {
        let model = symmetric_model(4);
        let data = observations();
        let session = Session::new(ALGORITHM, "test");
        let before = model.score(&data, &session).expect("initial score").value;
        let fitted = model.fit(&data, &session).expect("fit").value;
        let after = fitted.score(&data, &session).expect("fitted score").value;

        // Baum–Welch is monotone in exact arithmetic; four times the configured
        // probability-sum tolerance leaves the requested integration margin.
        let tolerance = 4.0 * model.policy().probability_sum_tol;
        assert!(
            after + tolerance >= before,
            "before={before}, after={after}"
        );
    }

    #[test]
    fn state_permutation_preserves_score() {
        let policy = Policy::default();
        let original = HiddenMarkovModel::new(
            Vector::from_slice(&[0.6, 0.4]),
            Matrix::from_row_major(2, 2, &[0.85, 0.15, 0.2, 0.8]),
            emissions(),
            1,
            false,
            policy.clone(),
        )
        .expect("valid original model");
        let mut reversed_emissions = emissions();
        reversed_emissions.reverse();
        let permuted = HiddenMarkovModel::new(
            Vector::from_slice(&[0.4, 0.6]),
            Matrix::from_row_major(2, 2, &[0.8, 0.2, 0.15, 0.85]),
            reversed_emissions,
            1,
            false,
            policy.clone(),
        )
        .expect("valid permuted model");
        let data = observations();
        let session = Session::new(ALGORITHM, "test");
        let original_score = original.score(&data, &session).expect("score").value;
        let permuted_score = permuted.score(&data, &session).expect("score").value;

        // The fixture is an exact state relabeling; four times the configured
        // probability-sum tolerance covers summation-order roundoff.
        let tolerance = 4.0 * policy.probability_sum_tol;
        assert!((original_score - permuted_score).abs() <= tolerance);
    }

    #[test]
    fn left_right_fit_retains_topology() {
        let policy = Policy::default();
        let model = HiddenMarkovModel::new(
            Vector::from_slice(&[1.0, 0.0]),
            Matrix::from_row_major(2, 2, &[0.8, 0.2, 0.0, 1.0]),
            emissions(),
            4,
            true,
            policy,
        )
        .expect("valid left-right model");
        let session = Session::new(ALGORITHM, "test");
        let fitted = model
            .fit(&observations(), &session)
            .expect("left-right fit")
            .value;

        assert_eq!(fitted.initial().as_slice(), &[1.0, 0.0]);
        assert_eq!(fitted.transition().get(1, 0), 0.0);
        assert!(fitted.left_right());
    }

    #[test]
    fn decode_has_one_state_per_observation() {
        let data = observations();
        let decoded = symmetric_model(1)
            .decode(&data, &Session::new(ALGORITHM, "test"))
            .expect("decode")
            .value;
        assert_eq!(decoded.len(), data.nrows());
    }

    #[test]
    fn one_observation_fit_retains_transition_matrix() {
        let model = HiddenMarkovModel::new(
            Vector::from_slice(&[0.5, 0.5]),
            Matrix::from_row_major(2, 2, &[0.7, 0.3, 0.4, 0.6]),
            vec![FixedEmission(0.0), FixedEmission(0.0)],
            2,
            false,
            Policy::default(),
        )
        .expect("valid fixed-emission HMM");
        let before = model.transition().to_row_major();
        let fitted = model
            .fit(
                &Matrix::from_row_major(1, 1, &[3.0]),
                &Session::new(ALGORITHM, "test"),
            )
            .expect("single-observation fit")
            .value;

        assert_eq!(fitted.transition().to_row_major(), before);
    }

    #[test]
    fn decode_resolves_finite_ties_but_rejects_an_impossible_sequence() {
        let data = Matrix::from_row_major(3, 1, &[1.0, 2.0, 3.0]);
        let tied = HiddenMarkovModel::new(
            Vector::from_slice(&[0.5, 0.5]),
            Matrix::from_row_major(2, 2, &[0.5, 0.5, 0.5, 0.5]),
            vec![FixedEmission(0.0), FixedEmission(0.0)],
            1,
            false,
            Policy::default(),
        )
        .expect("valid tied HMM");
        let path = tied
            .decode(&data, &Session::new(ALGORITHM, "tie"))
            .expect("a finite tie has a deterministic path")
            .value;
        assert_eq!(path.as_slice(), &[0.0, 0.0, 0.0]);

        let impossible = HiddenMarkovModel::new(
            Vector::from_slice(&[0.5, 0.5]),
            Matrix::from_row_major(2, 2, &[0.5, 0.5, 0.5, 0.5]),
            vec![
                FixedEmission(f64::NEG_INFINITY),
                FixedEmission(f64::NEG_INFINITY),
            ],
            1,
            false,
            Policy::default(),
        )
        .expect("valid chain with impossible emissions");
        let failure = impossible
            .decode(&data, &Session::new(ALGORITHM, "impossible"))
            .expect_err("forward preflight must reject an impossible sequence");
        assert_eq!(failure.primary.code, IssueCode::ScaleFactorZero);
    }
}
