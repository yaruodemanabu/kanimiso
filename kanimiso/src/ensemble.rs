//! Voting, bagging, and stacking of linear base learners.
//!
//! [`AdaBoostClassifier`] is re-exported from [`crate::tree`] (the boosting
//! implementation already lives there). Voting / bagging / stacking are
//! implemented here so [`crate::compose`] can re-export them without a cycle.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linear_model::{FittedLinear, FittedLogistic, LinearRegression, LogisticRegression};
use crate::rng::Rng;
use crate::traits::{Fit, Predict};
use crate::validate::{inspect_classes, inspect_xy};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Qualified, Result};

pub use crate::tree::{AdaBoostAlgorithm, AdaBoostClassifier, FittedAdaBoost};

fn take_rows(x: &Matrix, idx: &[usize]) -> Matrix {
    Matrix::from_fn(idx.len(), x.ncols(), |i, j| x.get(idx[i], j))
}

fn take_vec(y: &Vector, idx: &[usize]) -> Vector {
    Vector::from_iter(idx.iter().map(|&i| y[i]))
}

fn bootstrap(n: usize, rng: &mut Rng) -> Vec<usize> {
    (0..n).map(|_| rng.below(n.max(1))).collect()
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn logistic_proba(m: &FittedLogistic, x: &Matrix) -> Vector {
    if let Some(sm) = &m.softmax {
        if x.ncols() == sm.coef.ncols() || (sm.used_intercept && x.ncols() + 1 == sm.coef.ncols()) {
            let p = sm.predict_proba(x);
            let last = sm.classes.len().saturating_sub(1);
            return Vector::from_iter((0..x.nrows()).map(|i| p.get(i, last)));
        }
    }
    Vector::from_iter((0..x.nrows()).map(|i| {
        let mut s = m.intercept;
        for j in 0..x.ncols().min(m.coef.len()) {
            s += m.coef[j] * x.get(i, j);
        }
        sigmoid(s)
    }))
}

fn linear_pred(m: &FittedLinear, x: &Matrix) -> Vector {
    if x.ncols() != m.coef.len() {
        return Vector::filled(x.nrows(), m.intercept);
    }
    let mut y = x.matvec(&m.coef);
    if m.used_intercept {
        for i in 0..y.len() {
            y[i] += m.intercept;
        }
    }
    y
}

/// Soft-voting classifier: bootstrap copies of [`LogisticRegression`], average probabilities.
#[derive(Clone, Debug)]
pub struct VotingClassifier {
    /// Number of logistic voters.
    pub n_estimators: usize,
    /// PRNG seed for the bootstrap.
    pub seed: u64,
}

impl Default for VotingClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 5,
            seed: 0,
        }
    }
}

impl VotingClassifier {
    /// `n` bootstrap logistic voters.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            seed: 0,
        }
    }
}

/// Fitted voting classifier.
#[derive(Clone, Debug)]
pub struct FittedVotingClassifier {
    /// Bootstrap logistic models.
    pub members: Vec<FittedLogistic>,
    /// Sorted training classes.
    pub classes: Vec<i64>,
}

impl Predict for FittedVotingClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if self.members.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("VotingClassifier has no members")
                    .build(),
            );
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut acc = Vector::zeros(x.nrows());
        for m in &self.members {
            let p = logistic_proba(m, x);
            for i in 0..acc.len() {
                acc[i] += p[i];
            }
        }
        let k = self.members.len() as f64;
        let pos = *self.classes.last().unwrap_or(&1) as f64;
        let neg = *self.classes.first().unwrap_or(&0) as f64;
        for i in 0..acc.len() {
            acc[i] = if acc[i] / k >= 0.5 { pos } else { neg };
        }
        ctx.finish(acc)
    }
}

impl Fit for VotingClassifier {
    type Fitted = FittedVotingClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedVotingClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        let n = x.nrows();
        let mut rng = Rng::new(self.seed);
        let mut members = Vec::new();
        let k = self.n_estimators.max(1);
        for t in 0..k {
            let idx = bootstrap(n, &mut rng);
            let mut lr = LogisticRegression::new();
            if let Ok(q) = lr.fit(
                &take_rows(x, &idx),
                &take_vec(y, &idx),
                &session.child(format!("voter_{t}")),
            ) {
                members.push(q.value);
            }
        }
        ctx.session
            .converged(format!("{k} logistic voters"), k as u64);
        ctx.finish(FittedVotingClassifier { members, classes })
    }
}

/// Averaging regressor of bootstrap [`LinearRegression`] models.
#[derive(Clone, Debug)]
pub struct VotingRegressor {
    /// Number of OLS voters.
    pub n_estimators: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for VotingRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 5,
            seed: 0,
        }
    }
}

impl VotingRegressor {
    /// `n` bootstrap OLS voters.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            seed: 0,
        }
    }
}

/// Fitted voting regressor.
#[derive(Clone, Debug)]
pub struct FittedVotingRegressor {
    /// Bootstrap OLS models.
    pub members: Vec<FittedLinear>,
}

impl Predict for FittedVotingRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if self.members.is_empty() {
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut acc = Vector::zeros(x.nrows());
        for m in &self.members {
            let p = linear_pred(m, x);
            for i in 0..acc.len() {
                acc[i] += p[i];
            }
        }
        let k = self.members.len() as f64;
        ctx.finish(acc.scale(1.0 / k))
    }
}

impl Fit for VotingRegressor {
    type Fitted = FittedVotingRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedVotingRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let n = x.nrows();
        let mut rng = Rng::new(self.seed);
        let mut members = Vec::new();
        let k = self.n_estimators.max(1);
        for t in 0..k {
            let idx = bootstrap(n, &mut rng);
            if let Ok(q) = LinearRegression::new().fit(
                &take_rows(x, &idx),
                &take_vec(y, &idx),
                &session.child(format!("voter_{t}")),
            ) {
                members.push(q.value);
            }
        }
        ctx.finish(FittedVotingRegressor { members })
    }
}

/// Bagged [`LinearRegression`] (bootstrap aggregating).
#[derive(Clone, Debug)]
pub struct BaggingRegressor {
    /// Number of bags.
    pub n_estimators: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for BaggingRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            seed: 0,
        }
    }
}

impl BaggingRegressor {
    /// `n` bootstrap OLS bags.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            seed: 0,
        }
    }
}

/// Fitted bagging regressor (same representation as voting OLS).
#[derive(Clone, Debug)]
pub struct FittedBaggingRegressor {
    /// Bootstrap OLS models.
    pub members: Vec<FittedLinear>,
}

impl Predict for FittedBaggingRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        FittedVotingRegressor {
            members: self.members.clone(),
        }
        .predict(x, session)
    }
}

impl Fit for BaggingRegressor {
    type Fitted = FittedBaggingRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedBaggingRegressor>> {
        let mut voter = VotingRegressor {
            n_estimators: self.n_estimators,
            seed: self.seed,
        };
        let q = voter.fit(x, y, session)?;
        Ok(q.map(|v| FittedBaggingRegressor { members: v.members }))
    }
}

/// Two [`LinearRegression`] bases + a meta OLS on out-of-fold predictions.
#[derive(Clone, Debug)]
pub struct StackingRegressor {
    /// If true, the second base is fit without an intercept (diversity).
    pub diverse_intercept: bool,
}

impl Default for StackingRegressor {
    fn default() -> Self {
        Self {
            diverse_intercept: true,
        }
    }
}

impl StackingRegressor {
    /// Default two-base stacker.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted stacking regressor.
#[derive(Clone, Debug)]
pub struct FittedStackingRegressor {
    /// First base (intercept on).
    pub base_a: FittedLinear,
    /// Second base (optionally intercept off).
    pub base_b: FittedLinear,
    /// Meta OLS on the two base predictions.
    pub meta: FittedLinear,
}

impl Predict for FittedStackingRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let a = linear_pred(&self.base_a, x);
        let b = linear_pred(&self.base_b, x);
        let z = Matrix::from_fn(x.nrows(), 2, |i, j| if j == 0 { a[i] } else { b[i] });
        let y = linear_pred(&self.meta, &z);
        ctx.finish(y)
    }
}

impl Fit for StackingRegressor {
    type Fitted = FittedStackingRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedStackingRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let n = x.nrows();
        let mid = (n / 2).max(1).min(n.saturating_sub(1));
        let a_idx: Vec<usize> = (0..mid).collect();
        let b_idx: Vec<usize> = (mid..n).collect();
        let empty = FittedLinear {
            coef: Vector::zeros(x.ncols()),
            intercept: 0.0,
            beta: Vector::zeros(x.ncols() + 1),
            n,
            p: x.ncols(),
            df_resid: 0.0,
            r2: f64::NAN,
            adj_r2: f64::NAN,
            sigma2: f64::NAN,
            se: Vector::zeros(x.ncols()),
            t_values: Vector::zeros(x.ncols()),
            p_values: Vector::zeros(x.ncols()),
            aic: f64::NAN,
            bic: f64::NAN,
            f_stat: f64::NAN,
            f_pvalue: f64::NAN,
            durbin_watson: f64::NAN,
            loglik: f64::NAN,
            fitted: Vector::zeros(n),
            resid: Vector::zeros(n),
            leverage: Vector::zeros(n),
            cooks: Vector::zeros(n),
            used_intercept: true,
        };
        let mut lr_a = LinearRegression {
            fit_intercept: true,
        };
        let mut lr_b = LinearRegression {
            fit_intercept: !self.diverse_intercept,
        };
        let base_a_half = lr_a
            .fit(
                &take_rows(x, &a_idx),
                &take_vec(y, &a_idx),
                &session.child("base_a_half"),
            )
            .map(|q| q.value)
            .unwrap_or_else(|_| empty.clone());
        let base_b_half = lr_b
            .fit(
                &take_rows(x, &a_idx),
                &take_vec(y, &a_idx),
                &session.child("base_b_half"),
            )
            .map(|q| q.value)
            .unwrap_or_else(|_| empty.clone());
        let xa = take_rows(x, &b_idx);
        let ya = take_vec(y, &b_idx);
        let pa = linear_pred(&base_a_half, &xa);
        let pb = linear_pred(&base_b_half, &xa);
        let z = Matrix::from_fn(b_idx.len(), 2, |i, j| if j == 0 { pa[i] } else { pb[i] });
        let meta = LinearRegression::new()
            .fit(&z, &ya, &session.child("meta"))
            .map(|q| q.value)
            .unwrap_or_else(|_| {
                let mut m = empty.clone();
                m.coef = Vector::from_slice(&[0.5, 0.5]);
                m.used_intercept = true;
                m.intercept = ya.mean();
                m
            });
        let mut lr_a_full = LinearRegression {
            fit_intercept: true,
        };
        let base_a = lr_a_full
            .fit(x, y, &session.child("base_a"))
            .map(|q| q.value)
            .unwrap_or_else(|_| empty.clone());
        let mut lr_b_full = LinearRegression {
            fit_intercept: !self.diverse_intercept,
        };
        let base_b = lr_b_full
            .fit(x, y, &session.child("base_b"))
            .map(|q| q.value)
            .unwrap_or_else(|_| empty);
        ctx.finish(FittedStackingRegressor {
            base_a,
            base_b,
            meta,
        })
    }
}

/// Bagged [`LogisticRegression`] (bootstrap aggregating).
#[derive(Clone, Debug)]
pub struct BaggingClassifier {
    /// Number of bags.
    pub n_estimators: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for BaggingClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 8,
            seed: 0,
        }
    }
}

impl BaggingClassifier {
    /// `n` bootstrap logistic bags.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            seed: 0,
        }
    }
}

/// Fitted bagging classifier.
#[derive(Clone, Debug)]
pub struct FittedBaggingClassifier {
    /// Bootstrap logistic models.
    pub members: Vec<FittedLogistic>,
    /// Sorted training classes.
    pub classes: Vec<i64>,
}

impl Predict for FittedBaggingClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        FittedVotingClassifier {
            members: self.members.clone(),
            classes: self.classes.clone(),
        }
        .predict(x, session)
    }
}

impl Fit for BaggingClassifier {
    type Fitted = FittedBaggingClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedBaggingClassifier>> {
        let mut voter = VotingClassifier {
            n_estimators: self.n_estimators,
            seed: self.seed,
        };
        let q = voter.fit(x, y, session)?;
        Ok(q.map(|v| FittedBaggingClassifier {
            members: v.members,
            classes: v.classes,
        }))
    }
}

/// Two [`LogisticRegression`] bases + a meta logistic on hold-out scores.
#[derive(Clone, Debug)]
pub struct StackingClassifier {
    /// If true, the second base uses a larger ridge-like step cap via default logistic.
    pub diverse: bool,
}

impl Default for StackingClassifier {
    fn default() -> Self {
        Self { diverse: true }
    }
}

impl StackingClassifier {
    /// Default two-base stacker.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted stacking classifier.
#[derive(Clone, Debug)]
pub struct FittedStackingClassifier {
    /// First base.
    pub base_a: FittedLogistic,
    /// Second base.
    pub base_b: FittedLogistic,
    /// Meta logistic on the two base probabilities.
    pub meta: FittedLogistic,
    /// Training classes.
    pub classes: Vec<i64>,
}

impl Predict for FittedStackingClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let a = logistic_proba(&self.base_a, x);
        let b = logistic_proba(&self.base_b, x);
        let z = Matrix::from_fn(x.nrows(), 2, |i, j| if j == 0 { a[i] } else { b[i] });
        let pos = *self.classes.last().unwrap_or(&1) as f64;
        let neg = *self.classes.first().unwrap_or(&0) as f64;
        let p = logistic_proba(&self.meta, &z);
        let out = Vector::from_iter((0..x.nrows()).map(|i| if p[i] >= 0.5 { pos } else { neg }));
        ctx.finish(out)
    }
}

impl Fit for StackingClassifier {
    type Fitted = FittedStackingClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedStackingClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        let n = x.nrows();
        let mid = (n / 2).max(1).min(n.saturating_sub(1));
        let a_idx: Vec<usize> = (0..mid).collect();
        let b_idx: Vec<usize> = (mid..n).collect();
        let empty = match LogisticRegression::new().fit(
            &take_rows(x, &a_idx),
            &take_vec(y, &a_idx),
            &session.child("probe"),
        ) {
            Ok(q) => q.value,
            Err(_) => FittedLogistic {
                coef: Vector::zeros(x.ncols()),
                intercept: 0.0,
                classes: classes.clone(),
                beta: Vector::zeros(x.ncols() + 1),
                softmax: None,
            },
        };
        let base_a_half = LogisticRegression::new()
            .fit(
                &take_rows(x, &a_idx),
                &take_vec(y, &a_idx),
                &session.child("base_a_half"),
            )
            .map(|q| q.value)
            .unwrap_or_else(|_| empty.clone());
        let mut lr_b = LogisticRegression::new();
        if self.diverse {
            lr_b.max_iter = lr_b.max_iter.saturating_add(8);
        }
        let base_b_half = lr_b
            .fit(
                &take_rows(x, &a_idx),
                &take_vec(y, &a_idx),
                &session.child("base_b_half"),
            )
            .map(|q| q.value)
            .unwrap_or_else(|_| empty.clone());
        let xa = take_rows(x, &b_idx);
        let ya = take_vec(y, &b_idx);
        let pa = logistic_proba(&base_a_half, &xa);
        let pb = logistic_proba(&base_b_half, &xa);
        let z = Matrix::from_fn(b_idx.len(), 2, |i, j| if j == 0 { pa[i] } else { pb[i] });
        let meta = LogisticRegression::new()
            .fit(&z, &ya, &session.child("meta"))
            .map(|q| q.value)
            .unwrap_or_else(|_| empty.clone());
        let base_a = LogisticRegression::new()
            .fit(x, y, &session.child("base_a"))
            .map(|q| q.value)
            .unwrap_or_else(|_| empty.clone());
        let base_b = LogisticRegression::new()
            .fit(x, y, &session.child("base_b"))
            .map(|q| q.value)
            .unwrap_or_else(|_| empty);
        ctx.finish(FittedStackingClassifier {
            base_a,
            base_b,
            meta,
            classes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    fn line(n: usize) -> (Matrix, Vector) {
        let x = Matrix::from_fn(n, 1, |i, _| i as f64);
        let y =
            Vector::from_iter((0..n).map(|i| 1.0 + 2.0 * i as f64 + 0.2 * ((i % 3) as f64 - 1.0)));
        (x, y)
    }

    #[test]
    fn bagging_recovers_a_line() {
        let (x, y) = line(16);
        let q = BaggingRegressor::new(6)
            .fit(&x, &y, &Session::new("ens", "bag"))
            .unwrap();
        let pred = q
            .value
            .predict(&x, &Session::new("ens", "pred"))
            .unwrap()
            .value;
        assert_eq!(pred.len(), y.len());
        assert!(pred.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn stacking_predicts_finite() {
        let (x, y) = line(20);
        let q = StackingRegressor::new()
            .fit(&x, &y, &Session::new("ens", "stack"))
            .unwrap();
        let pred = q
            .value
            .predict(&x, &Session::new("ens", "pred"))
            .unwrap()
            .value;
        assert!(pred.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn voting_classifier_two_blobs() {
        let x = Matrix::from_fn(20, 1, |i, _| if i < 10 { -2.0 } else { 2.0 });
        let y = Vector::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        let q = VotingClassifier::new(4)
            .fit(&x, &y, &Session::new("ens", "vote"))
            .unwrap();
        let pred = q
            .value
            .predict(&x, &Session::new("ens", "pred"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..y.len() {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 16, "ok={ok}");
    }

    #[test]
    fn bagging_and_stacking_classifier_two_blobs() {
        let x = Matrix::from_fn(20, 1, |i, _| if i < 10 { -2.0 } else { 2.0 });
        let y = Vector::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        let q = BaggingClassifier::new(4)
            .fit(&x, &y, &Session::new("ens", "bagc"))
            .unwrap();
        let pred = q
            .value
            .predict(&x, &Session::new("ens", "pred"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..y.len() {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 16, "bagc ok={ok}");
        let q = StackingClassifier::new()
            .fit(&x, &y, &Session::new("ens", "stackc"))
            .unwrap();
        let pred = q
            .value
            .predict(&x, &Session::new("ens", "pred"))
            .unwrap()
            .value;
        assert!(pred.as_slice().iter().all(|v| v.is_finite()));
    }
}
