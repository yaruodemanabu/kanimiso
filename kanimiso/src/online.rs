//! River-style online learning: recursive least squares, linear classifiers,
//! Hoeffding trees, drift detectors, stream clustering, anomaly trees, and
//! rolling metrics.
//!
//! **Contract.** Every [`PartialFit::partial_fit`] returns
//! [`ojizou_san::IncrementalExplain`] and records it on the session. A
//! parameter update without a narrative is a bug. Feature-space changes raise
//! [`IssueCode::FeatureSpaceChangedOnline`]; a `partial_fit` that cannot
//! initialize raises [`IssueCode::PartialFitBeforeInit`].
//!
//! Batch SGD for linear regression already lives in
//! [`crate::linear_model::SgdRegressor`]; it is re-exported here so the river
//! surface has a single import path.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::rng::Rng;
use crate::special::norm_cdf;
use crate::traits::{PartialFit, Predict, Transform};
use faer::Mat;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{
    IncrementalQuality, Issue, IssueCode, NumericalCompromise, Qualified, Result, Severity,
};
use std::collections::HashMap;

pub use crate::linear_model::SgdRegressor;

/// Online recursive least squares with a forgetting factor `λ`.
///
/// Sufficient statistics are the coefficient vector `θ` and the inverse Gram
/// `P`. After a row `(x, y)`:
///
/// ```text
/// K = P x / (λ + xᵀ P x)
/// θ ← θ + K (y − xᵀ θ)
/// P ← (P − K (P x)ᵀ) / λ
/// ```
///
/// The effective sample size is the geometric-weight sum
/// `(1 − λⁿ) / (1 − λ)`, which is `n` when `λ = 1` and approaches `1/(1−λ)`
/// otherwise. A tiny `n_eff` relative to `p` is
/// [`IssueCode::ForgettingErasedIdentification`].
#[derive(Clone, Debug)]
pub struct LinearRegression {
    /// Forgetting factor `λ ∈ (0, 1]`. `λ = 1` is ordinary growing-window RLS.
    pub forgetting_factor: f64,
    /// Prepend a column of ones.
    pub fit_intercept: bool,
    /// Initial `P = p0 I`.
    pub p0: f64,
    theta: Vector,
    p_mat: Option<Mat<f64>>,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            forgetting_factor: 1.0,
            fit_intercept: true,
            p0: 1_000.0,
            theta: Vector::zeros(0),
            p_mat: None,
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl LinearRegression {
    /// RLS with forgetting `λ`.
    pub fn new(forgetting_factor: f64) -> Self {
        Self {
            forgetting_factor,
            ..Self::default()
        }
    }

    /// Slope coefficients (intercept excluded).
    pub fn coef(&self) -> Vector {
        if self.fit_intercept && self.theta.len() > 0 {
            Vector::from_iter((1..self.theta.len()).map(|i| self.theta[i]))
        } else {
            self.theta.clone()
        }
    }

    /// Intercept, or 0 when `fit_intercept` is false.
    pub fn intercept(&self) -> f64 {
        if self.fit_intercept && !self.theta.is_empty() {
            self.theta[0]
        } else {
            0.0
        }
    }

    /// Inverse Gram `P` (covariance of `θ` up to σ²).
    pub fn p_matrix(&self) -> Option<&Mat<f64>> {
        self.p_mat.as_ref()
    }

    fn dim(&self, p_x: usize) -> usize {
        p_x + if self.fit_intercept { 1 } else { 0 }
    }

    fn row_vec(&self, x: &Matrix, i: usize) -> Vector {
        let p = self.dim(x.ncols());
        let mut v = Vector::zeros(p);
        if self.fit_intercept {
            v[0] = 1.0;
            for j in 0..x.ncols() {
                v[j + 1] = x.get(i, j);
            }
        } else {
            for j in 0..x.ncols() {
                v[j] = x.get(i, j);
            }
        }
        v
    }

    fn init(&mut self, p_x: usize) {
        let p = self.dim(p_x);
        self.theta = Vector::zeros(p);
        self.p_mat = Some(Mat::<f64>::from_fn(p, p, |i, j| {
            if i == j {
                self.p0
            } else {
                0.0
            }
        }));
        self.initialized = true;
    }

    fn n_eff(&self) -> f64 {
        let lam = self.forgetting_factor;
        let n = self.n_seen as f64;
        if (1.0 - lam).abs() < 1e-15 {
            n
        } else {
            (1.0 - lam.powf(n)) / (1.0 - lam)
        }
    }

    fn predict_row(&self, x: &Matrix, i: usize) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        let z = self.row_vec(x, i);
        z.dot(&self.theta)
    }
}

impl PartialFit for LinearRegression {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(
                Issue::builder(IssueCode::MissingTarget)
                    .message("online LinearRegression.partial_fit requires y")
                    .build(),
            );
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        if x.nrows() == 0 || x.ncols() == 0 {
            if !self.initialized {
                ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            }
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "empty batch"),
            );
        }
        inspect_online_xy(&mut ctx, x, Some(y));
        if !(self.forgetting_factor > 0.0 && self.forgetting_factor <= 1.0) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!(
                        "forgetting λ={} is not in (0, 1]",
                        self.forgetting_factor
                    ))
                    .build(),
            );
        }
        if !self.initialized {
            self.init(x.ncols());
        } else if self.dim(x.ncols()) != self.theta.len() {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message(format!(
                        "RLS expects {} design columns (incl. intercept); got {}",
                        self.theta.len(),
                        self.dim(x.ncols())
                    ))
                    .build(),
            );
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.theta.clone();
        let loss_before = rls_mse(self, x, y);
        let mut info = 0.0;
        let lam = self.forgetting_factor.clamp(1e-12, 1.0);
        for i in 0..x.nrows() {
            let z = self.row_vec(x, i);
            let p = match self.p_mat.as_mut() {
                Some(p) => p,
                None => break,
            };
            let mut pz = Vector::zeros(z.len());
            for r in 0..z.len() {
                let mut s = 0.0;
                for c in 0..z.len() {
                    s += p[(r, c)] * z[c];
                }
                pz[r] = s;
            }
            let denom = lam + z.dot(&pz);
            if !denom.is_finite() || denom.abs() < 1e-18 {
                ctx.push(
                    Issue::builder(IssueCode::NearSingular)
                        .message("RLS gain denominator vanished")
                        .build(),
                );
                continue;
            }
            let pred = z.dot(&self.theta);
            let err = y[i] - pred;
            info += err * err;
            for j in 0..self.theta.len() {
                self.theta[j] += (pz[j] / denom) * err;
            }
            // P ← (P − (Pz)(Pz)ᵀ / denom) / λ
            for r in 0..z.len() {
                for c in 0..z.len() {
                    p[(r, c)] = (p[(r, c)] - pz[r] * pz[c] / denom) / lam;
                }
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.theta.sub(&before);
        let loss_after = rls_mse(self, x, y);
        let n_eff = self.n_eff();
        let p = self.theta.len();
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = n_eff;
        q.forgetting_factor = Some(self.forgetting_factor);
        q.parameter_delta_norm = Some(delta.norm());
        q.parameter_delta_max = Some(delta.max_abs());
        q.loss_before = Some(loss_before);
        q.loss_after = Some(loss_after);
        q.information_gain = Some(
            (loss_before - loss_after)
                .abs()
                .max(info / x.nrows().max(1) as f64),
        );
        q.still_identified = n_eff > p as f64 && self.n_seen as usize > p;
        q.warmup = self.n_seen < 5 || (self.n_seen as usize) <= p;
        q.explanation = format!(
            "RLS λ={:.4}: {} rows, n_eff={n_eff:.3}, ||Δθ||={:.4e}, mse {:.6e} → {:.6e}",
            self.forgetting_factor,
            x.nrows(),
            delta.norm(),
            loss_before,
            loss_after
        );
        q.top_moved_parameters = top_moved(&delta, 3);
        if q.is_uninformative(ctx.policy.uninformative_info_eps)
            || (loss_before - loss_after).abs() <= ctx.policy.uninformative_info_eps
                && delta.norm() <= ctx.policy.uninformative_info_eps
        {
            ctx.push(
                Issue::builder(IssueCode::UpdateWithZeroInformation)
                    .incremental(q.clone())
                    .message("this RLS batch did not change the residual or θ")
                    .build(),
            );
        }
        if n_eff < ctx.policy.min_effective_sample || n_eff <= p as f64 {
            ctx.push(
                Issue::builder(IssueCode::ForgettingErasedIdentification)
                    .incremental(q.clone())
                    .message(format!(
                        "n_eff={n_eff:.3} (asymptotic 1/(1−λ)={:.3}) is too small to identify {p} parameters",
                        if (1.0 - self.forgetting_factor).abs() < 1e-15 {
                            f64::INFINITY
                        } else {
                            1.0 / (1.0 - self.forgetting_factor)
                        }
                    ))
                    .metric("n_eff", n_eff)
                    .build(),
            );
        }
        if q.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("RLS is still in warmup (n_seen ≤ p or n_seen < 5)")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                format!("θ[{p}] and inverse Gram P"),
                "recursive least squares / Kalman gain on the new rows",
                format!("mse={loss_before:.6e}"),
                format!("mse={loss_after:.6e}"),
            ),
        )
    }
}

impl Predict for LinearRegression {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let y = Vector::from_iter((0..x.nrows()).map(|i| self.predict_row(x, i)));
        ctx.finish(y)
    }
}

fn rls_mse(m: &LinearRegression, x: &Matrix, y: &Vector) -> f64 {
    if x.nrows() == 0 {
        return f64::NAN;
    }
    let mut s = 0.0;
    for i in 0..x.nrows() {
        let e = m.predict_row(x, i) - y[i];
        s += e * e;
    }
    s / x.nrows() as f64
}

/// Online logistic regression by SGD (binary, labels mapped to `{0, 1}`).
#[derive(Clone, Debug)]
pub struct LogisticRegression {
    /// Step size.
    pub learning_rate: f64,
    /// ℓ₂ penalty on the slopes.
    pub l2: f64,
    /// Intercept.
    pub fit_intercept: bool,
    coef: Vector,
    intercept: f64,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            l2: 0.0,
            fit_intercept: true,
            coef: Vector::zeros(0),
            intercept: 0.0,
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl LogisticRegression {
    /// Default online logistic.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current slopes.
    pub fn coef(&self) -> &Vector {
        &self.coef
    }

    /// Current intercept.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

impl PartialFit for LogisticRegression {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        if !self.initialized {
            if x.ncols() == 0 {
                ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
                return finish_explain(
                    ctx,
                    reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
                );
            }
            self.coef = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if self.coef.len() != x.ncols() {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message("online logistic feature dimension changed")
                    .build(),
            );
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        inspect_online_xy(&mut ctx, x, Some(y));
        if self.learning_rate > 1.0 {
            ctx.push(
                Issue::builder(IssueCode::LearningRateTooLarge)
                    .message(format!(
                        "η={} is large for unscaled logistic SGD",
                        self.learning_rate
                    ))
                    .build(),
            );
        }
        let before = self.coef.clone();
        let loss_before = logloss_of(x, y, &self.coef, self.intercept);
        for i in 0..x.nrows() {
            let yi = as01(y[i]);
            let mut eta = self.intercept;
            for j in 0..x.ncols() {
                eta += self.coef[j] * x.get(i, j);
            }
            let mu = sigmoid(eta);
            let g = mu - yi;
            for j in 0..x.ncols() {
                self.coef[j] -= self.learning_rate * (g * x.get(i, j) + self.l2 * self.coef[j]);
            }
            if self.fit_intercept {
                self.intercept -= self.learning_rate * g;
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.coef.sub(&before);
        let loss_after = logloss_of(x, y, &self.coef, self.intercept);
        let expl = pack_linear_explain(
            &mut ctx,
            self.updates,
            x.nrows(),
            self.n_seen,
            &delta,
            loss_before,
            loss_after,
            self.n_seen as usize > x.ncols(),
            self.n_seen < 5,
            "logistic coefficients",
            "SGD on Bernoulli log-loss",
        );
        finish_explain(ctx, expl)
    }
}

impl Predict for LogisticRegression {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut eta = self.intercept;
            for j in 0..x.ncols().min(self.coef.len()) {
                eta += self.coef[j] * x.get(i, j);
            }
            if sigmoid(eta) >= 0.5 {
                1.0
            } else {
                0.0
            }
        }));
        ctx.finish(y)
    }
}

/// Online perceptron (mistake-driven, labels mapped to `±1`).
#[derive(Clone, Debug)]
pub struct Perceptron {
    coef: Vector,
    intercept: f64,
    n_seen: u64,
    updates: u64,
    mistakes: u64,
    initialized: bool,
}

impl Default for Perceptron {
    fn default() -> Self {
        Self {
            coef: Vector::zeros(0),
            intercept: 0.0,
            n_seen: 0,
            updates: 0,
            mistakes: 0,
            initialized: false,
        }
    }
}

impl Perceptron {
    /// Zero-initialized perceptron.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current slopes.
    pub fn coef(&self) -> &Vector {
        &self.coef
    }
}

impl PartialFit for Perceptron {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        if !self.initialized {
            if x.ncols() == 0 {
                ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
                return finish_explain(
                    ctx,
                    reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
                );
            }
            self.coef = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if self.coef.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        inspect_online_xy(&mut ctx, x, Some(y));
        let before = self.coef.clone();
        let mistakes_before = self.mistakes;
        for i in 0..x.nrows() {
            let yi = as_pm(y[i]);
            let mut s = self.intercept;
            for j in 0..x.ncols() {
                s += self.coef[j] * x.get(i, j);
            }
            if yi * s <= 0.0 {
                self.mistakes += 1;
                for j in 0..x.ncols() {
                    self.coef[j] += yi * x.get(i, j);
                }
                self.intercept += yi;
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.coef.sub(&before);
        let m_after = self.mistakes - mistakes_before;
        let loss_before = m_after as f64 / x.nrows().max(1) as f64;
        let expl = pack_linear_explain(
            &mut ctx,
            self.updates,
            x.nrows(),
            self.n_seen,
            &delta,
            loss_before,
            0.0,
            self.n_seen as usize > x.ncols(),
            self.n_seen < 5,
            "perceptron weights",
            "additive update on mistakes (Rosenblatt)",
        );
        finish_explain(ctx, expl)
    }
}

impl Predict for Perceptron {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut s = self.intercept;
            for j in 0..x.ncols().min(self.coef.len()) {
                s += self.coef[j] * x.get(i, j);
            }
            if s >= 0.0 {
                1.0
            } else {
                0.0
            }
        }));
        ctx.finish(y)
    }
}

/// Passive-aggressive PA-I classifier (Crammer et al.).
#[derive(Clone, Debug)]
pub struct PassiveAggressive {
    /// Aggressiveness `C` (maximum step).
    pub c: f64,
    coef: Vector,
    intercept: f64,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for PassiveAggressive {
    fn default() -> Self {
        Self {
            c: 1.0,
            coef: Vector::zeros(0),
            intercept: 0.0,
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl PassiveAggressive {
    /// PA-I with aggressiveness `c`.
    pub fn new(c: f64) -> Self {
        Self {
            c,
            ..Self::default()
        }
    }

    /// Current slopes.
    pub fn coef(&self) -> &Vector {
        &self.coef
    }
}

impl PartialFit for PassiveAggressive {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        if !self.initialized {
            if x.ncols() == 0 {
                ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
                return finish_explain(
                    ctx,
                    reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
                );
            }
            self.coef = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if self.coef.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        inspect_online_xy(&mut ctx, x, Some(y));
        let before = self.coef.clone();
        let mut hinge = 0.0;
        for i in 0..x.nrows() {
            let yi = as_pm(y[i]);
            let mut s = self.intercept;
            let mut n2 = 1.0;
            for j in 0..x.ncols() {
                let v = x.get(i, j);
                s += self.coef[j] * v;
                n2 += v * v;
            }
            let loss = (1.0 - yi * s).max(0.0);
            hinge += loss;
            if loss > 0.0 && n2 > 0.0 {
                let tau = (loss / n2).min(self.c);
                for j in 0..x.ncols() {
                    self.coef[j] += tau * yi * x.get(i, j);
                }
                self.intercept += tau * yi;
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.coef.sub(&before);
        let loss_before = hinge / x.nrows().max(1) as f64;
        let expl = pack_linear_explain(
            &mut ctx,
            self.updates,
            x.nrows(),
            self.n_seen,
            &delta,
            loss_before,
            0.0,
            self.n_seen as usize > x.ncols(),
            self.n_seen < 5,
            "PA-I weights",
            "hinge-minimizing projection with cap C",
        );
        finish_explain(ctx, expl)
    }
}

impl Predict for PassiveAggressive {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut s = self.intercept;
            for j in 0..x.ncols().min(self.coef.len()) {
                s += self.coef[j] * x.get(i, j);
            }
            if s >= 0.0 {
                1.0
            } else {
                0.0
            }
        }));
        ctx.finish(y)
    }
}

/// Very Fast Decision Tree (Hoeffding) for binary classification.
///
/// Each leaf stores class counts and per-feature class-conditional Gaussians.
/// A split is taken when the Gini-gain gap exceeds the Hoeffding bound
/// `√(ln(1/δ) / (2 n))`, or the bound itself falls below `tau`.
#[derive(Clone, Debug)]
pub struct HoeffdingTree {
    /// Split-confidence `δ`.
    pub delta: f64,
    /// Minimum observations in a leaf before a split is considered.
    pub min_samples: usize,
    /// Tie-break threshold on the Hoeffding bound.
    pub tau: f64,
    root: HtNode,
    n_seen: u64,
    updates: u64,
    n_features: usize,
    initialized: bool,
    last_split: Option<String>,
}

#[derive(Clone, Debug)]
enum HtNode {
    Leaf(HtLeaf),
    Split {
        feature: usize,
        threshold: f64,
        left: Box<HtNode>,
        right: Box<HtNode>,
        n: u64,
    },
}

#[derive(Clone, Debug)]
struct HtLeaf {
    id: u64,
    n: u64,
    n_pos: u64,
    n_neg: u64,
    feat_neg: Vec<Gauss>,
    feat_pos: Vec<Gauss>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Gauss {
    n: f64,
    mean: f64,
    m2: f64,
}

impl Gauss {
    fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.n += 1.0;
        let d = x - self.mean;
        self.mean += d / self.n;
        self.m2 += d * (x - self.mean);
    }
    fn std(&self) -> f64 {
        if self.n < 2.0 {
            1e-6
        } else {
            (self.m2 / (self.n - 1.0)).max(0.0).sqrt().max(1e-6)
        }
    }
    fn var(&self) -> f64 {
        if self.n < 2.0 {
            0.0
        } else {
            (self.m2 / (self.n - 1.0)).max(0.0)
        }
    }
}

impl HtLeaf {
    fn new(id: u64, p: usize) -> Self {
        Self {
            id,
            n: 0,
            n_pos: 0,
            n_neg: 0,
            feat_neg: vec![Gauss::default(); p],
            feat_pos: vec![Gauss::default(); p],
        }
    }

    fn push(&mut self, x: &Matrix, i: usize, pos: bool) {
        self.n += 1;
        if pos {
            self.n_pos += 1;
        } else {
            self.n_neg += 1;
        }
        for j in 0..x.ncols().min(self.feat_neg.len()) {
            if pos {
                self.feat_pos[j].push(x.get(i, j));
            } else {
                self.feat_neg[j].push(x.get(i, j));
            }
        }
    }

    fn gini(&self) -> f64 {
        gini_counts(self.n_neg as f64, self.n_pos as f64)
    }

    fn maybe_split(
        &self,
        min_samples: usize,
        delta: f64,
        tau: f64,
    ) -> Option<(usize, f64, f64, f64)> {
        if (self.n as usize) < min_samples || self.n_pos == 0 || self.n_neg == 0 {
            return None;
        }
        let p = self.feat_neg.len();
        let parent = self.gini();
        let mut best = (0usize, 0.0, -1.0);
        let mut second = -1.0;
        for j in 0..p {
            let t1 = self.feat_neg[j].mean;
            let t2 = self.feat_pos[j].mean;
            let t3 = 0.5 * (t1 + t2);
            for &t in &[t1, t2, t3] {
                let gain = self.gain_at(j, t, parent);
                if gain > best.2 {
                    second = best.2;
                    best = (j, t, gain);
                } else if gain > second {
                    second = gain;
                }
            }
        }
        let n = self.n as f64;
        let eps = (0.5 * (1.0 / delta.max(1e-12)).ln() / n).sqrt();
        if best.2 < 0.0 {
            return None;
        }
        if best.2 - second.max(0.0) > eps || eps < tau {
            Some((best.0, best.1, best.2, eps))
        } else {
            None
        }
    }

    fn gain_at(&self, j: usize, t: f64, parent: f64) -> f64 {
        let gn = &self.feat_neg[j];
        let gp = &self.feat_pos[j];
        let p_left_neg = norm_cdf((t - gn.mean) / gn.std());
        let p_left_pos = norm_cdf((t - gp.mean) / gp.std());
        let n_left_neg = self.n_neg as f64 * p_left_neg;
        let n_left_pos = self.n_pos as f64 * p_left_pos;
        let n_right_neg = self.n_neg as f64 - n_left_neg;
        let n_right_pos = self.n_pos as f64 - n_left_pos;
        let n_l = n_left_neg + n_left_pos;
        let n_r = n_right_neg + n_right_pos;
        let n = (self.n as f64).max(1.0);
        if n_l <= 1.0 || n_r <= 1.0 {
            return -1.0;
        }
        parent
            - (n_l * gini_counts(n_left_neg, n_left_pos)
                + n_r * gini_counts(n_right_neg, n_right_pos))
                / n
    }
}

fn gini_counts(n0: f64, n1: f64) -> f64 {
    let n = n0 + n1;
    if n <= 0.0 {
        return 0.0;
    }
    let p0 = n0 / n;
    let p1 = n1 / n;
    1.0 - p0 * p0 - p1 * p1
}

impl Default for HoeffdingTree {
    fn default() -> Self {
        Self {
            delta: 1e-7,
            min_samples: 20,
            tau: 0.05,
            root: HtNode::Leaf(HtLeaf::new(0, 0)),
            n_seen: 0,
            updates: 0,
            n_features: 0,
            initialized: false,
            last_split: None,
        }
    }
}

impl HoeffdingTree {
    /// Default VFDT.
    pub fn new() -> Self {
        Self::default()
    }

    /// Narrative of the most recent leaf split, if any.
    pub fn last_split(&self) -> Option<&str> {
        self.last_split.as_deref()
    }

    /// Reset to an empty root (used by adaptive forests after drift).
    pub fn reset(&mut self) {
        self.root = HtNode::Leaf(HtLeaf::new(0, self.n_features));
        self.n_seen = 0;
        self.last_split = None;
        self.initialized = self.n_features > 0;
    }

    fn predict_one(&self, x: &Matrix, i: usize) -> f64 {
        let mut node = &self.root;
        loop {
            match node {
                HtNode::Leaf(l) => {
                    return if l.n_pos >= l.n_neg { 1.0 } else { 0.0 };
                }
                HtNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    ..
                } => {
                    let v = if *feature < x.ncols() {
                        x.get(i, *feature)
                    } else {
                        0.0
                    };
                    node = if v <= *threshold { left } else { right };
                }
            }
        }
    }
}

fn ht_update(
    node: &mut HtNode,
    x: &Matrix,
    i: usize,
    pos: bool,
    min_samples: usize,
    delta: f64,
    tau: f64,
    next_id: u64,
) -> Option<String> {
    match node {
        HtNode::Split {
            feature,
            threshold,
            left,
            right,
            n,
        } => {
            *n += 1;
            if x.get(i, *feature) <= *threshold {
                ht_update(left, x, i, pos, min_samples, delta, tau, next_id)
            } else {
                ht_update(right, x, i, pos, min_samples, delta, tau, next_id)
            }
        }
        HtNode::Leaf(leaf) => {
            leaf.push(x, i, pos);
            if let Some((feat, thr, gain, eps)) = leaf.maybe_split(min_samples, delta, tau) {
                let p = leaf.feat_neg.len();
                let narrative = format!(
                    "leaf {} split on feature {feat} at {thr:.6} (Gini gain {gain:.4}, Hoeffding ε={eps:.4}, n={})",
                    leaf.id, leaf.n
                );
                *node = HtNode::Split {
                    feature: feat,
                    threshold: thr,
                    left: Box::new(HtNode::Leaf(HtLeaf::new(next_id, p))),
                    right: Box::new(HtNode::Leaf(HtLeaf::new(next_id + 1, p))),
                    n: leaf.n,
                };
                Some(narrative)
            } else {
                None
            }
        }
    }
}

impl PartialFit for HoeffdingTree {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        if !self.initialized {
            if x.ncols() == 0 {
                ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
                return finish_explain(
                    ctx,
                    reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
                );
            }
            self.n_features = x.ncols();
            self.root = HtNode::Leaf(HtLeaf::new(0, x.ncols()));
            self.initialized = true;
        } else if x.ncols() != self.n_features {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        inspect_online_xy(&mut ctx, x, Some(y));
        let mut split_note = None;
        for i in 0..x.nrows() {
            let pos = as01(y[i]) >= 0.5;
            let nid = self.n_seen + 17 + i as u64;
            if let Some(s) = ht_update(
                &mut self.root,
                x,
                i,
                pos,
                self.min_samples,
                self.delta,
                self.tau,
                nid,
            ) {
                split_note = Some(s);
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        if let Some(s) = &split_note {
            self.last_split = Some(s.clone());
        }
        let warmup = self.n_seen < self.min_samples as u64;
        if warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .message(format!(
                        "Hoeffding tree has seen {} < min_samples={}",
                        self.n_seen, self.min_samples
                    ))
                    .build(),
            );
        }
        let info = if split_note.is_some() { 1.0 } else { 0.0 };
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(if split_note.is_some() { 1.0 } else { 0.0 });
        q.information_gain = Some(info);
        q.still_identified = !warmup;
        q.warmup = warmup;
        q.explanation = split_note
            .clone()
            .unwrap_or_else(|| format!("leaf statistics updated on {} rows; no split", x.nrows()));
        if q.is_uninformative(ctx.policy.uninformative_info_eps) && !warmup {
            ctx.push(
                Issue::builder(IssueCode::UpdateWithZeroInformation)
                    .incremental(q.clone())
                    .message("Hoeffding update did not split and added no structural change")
                    .build(),
            );
        }
        let what = split_note
            .clone()
            .unwrap_or_else(|| "per-leaf class counts and Gaussian sufficient stats".into());
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                what,
                "Hoeffding-bound Gini split test (or sufficient-stat update)",
                "pre-batch tree",
                "post-batch tree",
            ),
        )
    }
}

impl Predict for HoeffdingTree {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        ctx.finish(Vector::from_iter(
            (0..x.nrows()).map(|i| self.predict_one(x, i)),
        ))
    }
}

/// Hoeffding tree regressor (river `HoeffdingTreeRegressor`).
///
/// Leaves predict a running mean. A split is taken when the variance-reduction
/// gap exceeds the Hoeffding bound, or the bound itself falls below `tau`.
#[derive(Clone, Debug)]
pub struct HoeffdingRegressor {
    /// Split-confidence `δ`.
    pub delta: f64,
    /// Minimum observations in a leaf before a split is considered.
    pub min_samples: usize,
    /// Tie-break threshold on the Hoeffding bound.
    pub tau: f64,
    root: HrNode,
    n_seen: u64,
    updates: u64,
    n_features: usize,
    initialized: bool,
    last_split: Option<String>,
}

#[derive(Clone, Debug)]
enum HrNode {
    Leaf(HrLeaf),
    Split {
        feature: usize,
        threshold: f64,
        left: Box<HrNode>,
        right: Box<HrNode>,
        n: u64,
    },
}

#[derive(Clone, Debug)]
struct HrLeaf {
    id: u64,
    y: Gauss,
    bins: Vec<HrBin>,
}

#[derive(Clone, Debug, Default)]
struct HrBin {
    gx: Gauss,
    left: Gauss,
    right: Gauss,
}

impl HrLeaf {
    fn new(id: u64, p: usize) -> Self {
        Self {
            id,
            y: Gauss::default(),
            bins: vec![HrBin::default(); p],
        }
    }

    fn push(&mut self, x: &Matrix, i: usize, yi: f64) {
        self.y.push(yi);
        for j in 0..x.ncols().min(self.bins.len()) {
            let xv = x.get(i, j);
            let t = if self.bins[j].gx.n < 1.0 {
                xv
            } else {
                self.bins[j].gx.mean
            };
            if xv <= t {
                self.bins[j].left.push(yi);
            } else {
                self.bins[j].right.push(yi);
            }
            self.bins[j].gx.push(xv);
        }
    }

    fn maybe_split(
        &self,
        min_samples: usize,
        delta: f64,
        tau: f64,
    ) -> Option<(usize, f64, f64, f64)> {
        if (self.y.n as usize) < min_samples {
            return None;
        }
        let parent_var = self.y.var();
        if parent_var <= 1e-18 {
            return None;
        }
        let mut best = (0usize, 0.0, -1.0);
        let mut second = -1.0;
        for (j, bin) in self.bins.iter().enumerate() {
            let nl = bin.left.n;
            let nr = bin.right.n;
            if nl < 2.0 || nr < 2.0 {
                continue;
            }
            let n = self.y.n.max(1.0);
            let red = parent_var - (nl * bin.left.var() + nr * bin.right.var()) / n;
            if red > best.2 {
                second = best.2;
                best = (j, bin.gx.mean, red);
            } else if red > second {
                second = red;
            }
        }
        let eps = (0.5 * (1.0 / delta.max(1e-12)).ln() / self.y.n.max(1.0)).sqrt();
        if best.2 < 0.0 {
            return None;
        }
        if best.2 - second.max(0.0) > eps || eps < tau {
            Some((best.0, best.1, best.2, eps))
        } else {
            None
        }
    }
}

impl Default for HoeffdingRegressor {
    fn default() -> Self {
        Self {
            delta: 1e-7,
            min_samples: 20,
            tau: 0.05,
            root: HrNode::Leaf(HrLeaf::new(0, 0)),
            n_seen: 0,
            updates: 0,
            n_features: 0,
            initialized: false,
            last_split: None,
        }
    }
}

impl HoeffdingRegressor {
    /// Default Hoeffding regressor.
    pub fn new() -> Self {
        Self::default()
    }

    fn reset(&mut self) {
        if self.n_features > 0 {
            self.root = HrNode::Leaf(HrLeaf::new(0, self.n_features));
        }
        self.n_seen = 0;
        self.last_split = None;
    }

    fn predict_one(&self, x: &Matrix, i: usize) -> f64 {
        let mut node = &self.root;
        loop {
            match node {
                HrNode::Leaf(l) => return l.y.mean,
                HrNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    ..
                } => {
                    let v = if *feature < x.ncols() {
                        x.get(i, *feature)
                    } else {
                        0.0
                    };
                    node = if v <= *threshold { left } else { right };
                }
            }
        }
    }
}

fn hr_update(
    node: &mut HrNode,
    x: &Matrix,
    i: usize,
    yi: f64,
    min_samples: usize,
    delta: f64,
    tau: f64,
    next_id: u64,
) -> Option<String> {
    match node {
        HrNode::Split {
            feature,
            threshold,
            left,
            right,
            n,
        } => {
            *n += 1;
            if x.get(i, *feature) <= *threshold {
                hr_update(left, x, i, yi, min_samples, delta, tau, next_id)
            } else {
                hr_update(right, x, i, yi, min_samples, delta, tau, next_id)
            }
        }
        HrNode::Leaf(leaf) => {
            leaf.push(x, i, yi);
            if let Some((feat, thr, gain, eps)) = leaf.maybe_split(min_samples, delta, tau) {
                let p = leaf.bins.len();
                let narrative = format!(
                    "leaf {} split on feature {feat} at {thr:.6e} (var-red {gain:.4e}, ε={eps:.4e}, n={})",
                    leaf.id, leaf.y.n
                );
                *node = HrNode::Split {
                    feature: feat,
                    threshold: thr,
                    left: Box::new(HrNode::Leaf(HrLeaf::new(next_id, p))),
                    right: Box::new(HrNode::Leaf(HrLeaf::new(next_id + 1, p))),
                    n: leaf.y.n as u64,
                };
                Some(narrative)
            } else {
                None
            }
        }
    }
}

impl PartialFit for HoeffdingRegressor {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        if !self.initialized {
            if x.ncols() == 0 {
                ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
                return finish_explain(
                    ctx,
                    reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
                );
            }
            self.n_features = x.ncols();
            self.root = HrNode::Leaf(HrLeaf::new(0, x.ncols()));
            self.initialized = true;
        } else if x.ncols() != self.n_features {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        inspect_online_xy(&mut ctx, x, Some(y));
        let mut split_note = None;
        for i in 0..x.nrows() {
            let nid = self.n_seen + 17 + i as u64;
            if let Some(s) = hr_update(
                &mut self.root,
                x,
                i,
                y[i],
                self.min_samples,
                self.delta,
                self.tau,
                nid,
            ) {
                split_note = Some(s);
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        if let Some(s) = &split_note {
            self.last_split = Some(s.clone());
        }
        let warmup = self.n_seen < self.min_samples as u64;
        if warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .message(format!(
                        "HoeffdingRegressor has seen {} < min_samples={}",
                        self.n_seen, self.min_samples
                    ))
                    .build(),
            );
        }
        let info = if split_note.is_some() { 1.0 } else { 0.0 };
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(if split_note.is_some() { 1.0 } else { 0.0 });
        q.information_gain = Some(info);
        q.still_identified = !warmup;
        q.warmup = warmup;
        q.explanation = split_note
            .clone()
            .unwrap_or_else(|| format!("leaf means updated on {} rows; no split", x.nrows()));
        if q.is_uninformative(ctx.policy.uninformative_info_eps) && !warmup {
            ctx.push(
                Issue::builder(IssueCode::UpdateWithZeroInformation)
                    .incremental(q.clone())
                    .message("HoeffdingRegressor update did not split")
                    .build(),
            );
        }
        let what = split_note
            .clone()
            .unwrap_or_else(|| "per-leaf mean and variance-reduction candidates".into());
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                what,
                "Hoeffding-bound variance-reduction split test",
                "pre-batch tree",
                "post-batch tree",
            ),
        )
    }
}

impl Predict for HoeffdingRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        ctx.finish(Vector::from_iter(
            (0..x.nrows()).map(|i| self.predict_one(x, i)),
        ))
    }
}

/// Online one-hot encoder (river `preprocessing.OneHotEncoder`).
///
/// Width is capped at `n_features × max_categories`. New categories after the
/// first update are a warned feature-space change, not an abort.
#[derive(Clone, Debug)]
pub struct OnlineOneHotEncoder {
    /// Cap on distinct codes per column.
    pub max_categories: usize,
    maps: Vec<HashMap<i64, usize>>,
    n_features: usize,
    initialized: bool,
    updates: u64,
    n_seen: u64,
    new_cats: u64,
}

impl Default for OnlineOneHotEncoder {
    fn default() -> Self {
        Self {
            max_categories: 8,
            maps: Vec::new(),
            n_features: 0,
            initialized: false,
            updates: 0,
            n_seen: 0,
            new_cats: 0,
        }
    }
}

impl OnlineOneHotEncoder {
    /// Encoder with `max_categories` slots per column.
    pub fn new(max_categories: usize) -> Self {
        Self {
            max_categories: max_categories.max(1),
            ..Self::default()
        }
    }
}

impl PartialFit for OnlineOneHotEncoder {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if !self.initialized {
            self.n_features = x.ncols();
            self.maps = vec![HashMap::new(); x.ncols()];
            self.initialized = true;
        } else if x.ncols() != self.n_features {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let mut added = 0u64;
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                let key = v.round() as i64;
                if self.maps[j].contains_key(&key) {
                    continue;
                }
                if self.maps[j].len() >= self.max_categories {
                    ctx.push(
                        Issue::builder(IssueCode::InvalidWeight)
                            .severity(Severity::Warning)
                            .message(format!(
                                "OnlineOneHotEncoder column {j} exceeded max_categories={}",
                                self.max_categories
                            ))
                            .build(),
                    );
                    continue;
                }
                let slot = self.maps[j].len();
                self.maps[j].insert(key, slot);
                added += 1;
                if self.updates > 0 {
                    ctx.push(
                        Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                            .severity(Severity::Warning)
                            .message(format!("new category {key} on column {j}"))
                            .build(),
                    );
                }
            }
        }
        self.new_cats += added;
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(added as f64);
        q.information_gain = Some((added as f64).max(1.0 / self.n_seen.max(1) as f64));
        q.still_identified = self.n_seen >= 1;
        q.warmup = self.n_seen < 2;
        q.explanation = format!(
            "online one-hot added {added} categories on {} rows",
            x.nrows()
        );
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "category maps per column",
                "integer-code one-hot with a fixed width cap",
                "previous category maps",
                "updated category maps",
            ),
        )
    }
}

impl Transform for OnlineOneHotEncoder {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Matrix::zeros(x.nrows(), 0));
        }
        let width = self.n_features * self.max_categories;
        let out = Matrix::from_fn(x.nrows(), width, |i, j| {
            let col = j / self.max_categories;
            let slot = j % self.max_categories;
            if col >= x.ncols() || col >= self.maps.len() {
                return 0.0;
            }
            let v = x.get(i, col);
            if !v.is_finite() {
                return 0.0;
            }
            match self.maps[col].get(&(v.round() as i64)) {
                Some(&s) if s == slot => 1.0,
                _ => 0.0,
            }
        });
        ctx.finish(out)
    }
}

/// Online ordinal encoder (river `preprocessing.OrdinalEncoder`).
///
/// New categories after the first update raise
/// [`IssueCode::FeatureSpaceChangedOnline`] as a warning (the default Error
/// would abort a valid stream).
#[derive(Clone, Debug)]
pub struct OnlineOrdinalEncoder {
    /// Cap on distinct codes per column.
    pub max_categories: usize,
    maps: Vec<HashMap<i64, usize>>,
    n_features: usize,
    initialized: bool,
    updates: u64,
    n_seen: u64,
    new_cats: u64,
}

impl Default for OnlineOrdinalEncoder {
    fn default() -> Self {
        Self {
            max_categories: 16,
            maps: Vec::new(),
            n_features: 0,
            initialized: false,
            updates: 0,
            n_seen: 0,
            new_cats: 0,
        }
    }
}

impl OnlineOrdinalEncoder {
    /// Encoder with `max_categories` codes per column.
    pub fn new(max_categories: usize) -> Self {
        Self {
            max_categories: max_categories.max(1),
            ..Self::default()
        }
    }
}

impl PartialFit for OnlineOrdinalEncoder {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if !self.initialized {
            self.n_features = x.ncols();
            self.maps = vec![HashMap::new(); x.ncols()];
            self.initialized = true;
        } else if x.ncols() != self.n_features {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let mut added = 0u64;
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                let key = v.round() as i64;
                if self.maps[j].contains_key(&key) {
                    continue;
                }
                if self.maps[j].len() >= self.max_categories {
                    ctx.push(
                        Issue::builder(IssueCode::InvalidWeight)
                            .severity(Severity::Warning)
                            .message(format!(
                                "OnlineOrdinalEncoder column {j} exceeded max_categories={}",
                                self.max_categories
                            ))
                            .build(),
                    );
                    continue;
                }
                let slot = self.maps[j].len();
                self.maps[j].insert(key, slot);
                added += 1;
                if self.updates > 0 {
                    ctx.push(
                        Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                            .severity(Severity::Warning)
                            .message(format!("new ordinal category {key} on column {j}"))
                            .build(),
                    );
                }
            }
        }
        self.new_cats += added;
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(added as f64);
        q.information_gain = Some((added as f64).max(1.0 / self.n_seen.max(1) as f64));
        q.still_identified = self.n_seen >= 1;
        q.warmup = self.n_seen < 2;
        q.explanation = format!(
            "online ordinal encoder added {added} codes on {} rows",
            x.nrows()
        );
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "integer codes per column",
                "first-seen category → 0..K−1 with a fixed cap",
                "previous ordinal maps",
                "updated ordinal maps",
            ),
        )
    }
}

impl Transform for OnlineOrdinalEncoder {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Matrix::zeros(x.nrows(), x.ncols()));
        }
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j >= self.maps.len() {
                return -1.0;
            }
            let v = x.get(i, j);
            if !v.is_finite() {
                return -1.0;
            }
            match self.maps[j].get(&(v.round() as i64)) {
                Some(&s) => s as f64,
                None => -1.0,
            }
        });
        ctx.finish(out)
    }
}

/// Outcome of a univariate drift detector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DriftDecision {
    /// No evidence of a change.
    Stable,
    /// Warning region (DDM) or mild evidence.
    Warning {
        /// Detector statistic.
        statistic: f64,
    },
    /// Drift declared; old parameters no longer describe the stream.
    Drift {
        /// Detector statistic.
        statistic: f64,
    },
}

/// ADWIN (Bifet & Gavaldà): adaptive windowing on a scalar stream.
#[derive(Clone, Debug)]
pub struct Adwin {
    /// Confidence `δ`.
    pub delta: f64,
    /// Hard cap on the stored window (oldest points are dropped).
    pub max_window: usize,
    window: Vec<f64>,
    cuts: u64,
}

impl Default for Adwin {
    fn default() -> Self {
        Self {
            delta: 0.002,
            max_window: 2048,
            window: Vec::new(),
            cuts: 0,
        }
    }
}

impl Adwin {
    /// Detector with the given `δ`.
    pub fn new(delta: f64) -> Self {
        Self {
            delta,
            ..Self::default()
        }
    }

    /// Current window length.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Reset the window.
    pub fn reset(&mut self) {
        self.window.clear();
    }

    /// Number of ADWIN cuts performed so far.
    pub fn n_cuts(&self) -> u64 {
        self.cuts
    }

    /// Consume one observation and decide whether the window must shrink.
    pub fn update(&mut self, x: f64, session: &Session) -> Result<Qualified<DriftDecision>> {
        let mut ctx = FitCtx::with_session(session.child("update"));
        if x.is_finite() {
            self.window.push(x);
        }
        if self.window.len() > self.max_window {
            let drop = self.window.len() - self.max_window;
            self.window.drain(0..drop);
        }
        let decision = self.try_cut();
        match decision {
            DriftDecision::Drift { statistic } => {
                ctx.push(
                    Issue::builder(IssueCode::ConceptDriftDetected)
                        .message(format!(
                            "ADWIN cut the window (δ={}, |μ₀−μ₁|={statistic:.4}, n={})",
                            self.delta,
                            self.window.len()
                        ))
                        .metric("drift_statistic", statistic)
                        .build(),
                );
            }
            DriftDecision::Warning { statistic } => {
                ctx.push(
                    Issue::builder(IssueCode::VirtualDriftDetected)
                        .severity(Severity::Advisory)
                        .message(format!("ADWIN warning statistic={statistic:.4}"))
                        .build(),
                );
            }
            DriftDecision::Stable => {}
        }
        ctx.finish(decision)
    }

    fn try_cut(&mut self) -> DriftDecision {
        let n = self.window.len();
        if n < 16 {
            return DriftDecision::Stable;
        }
        let mut prefix = vec![0.0; n + 1];
        for i in 0..n {
            prefix[i + 1] = prefix[i] + self.window[i];
        }
        let total = prefix[n];
        let nf = n as f64;
        for i in 8..(n - 8) {
            let n0 = i as f64;
            let n1 = (n - i) as f64;
            let mu0 = prefix[i] / n0;
            let mu1 = (total - prefix[i]) / n1;
            let diff = (mu0 - mu1).abs();
            let m = 1.0 / (1.0 / n0 + 1.0 / n1);
            let delta_p = (self.delta / nf.max(1.0)).max(1e-16);
            let eps = (0.5 / m * (4.0 / delta_p).ln()).sqrt();
            if diff > eps {
                self.window.drain(0..i);
                self.cuts += 1;
                return DriftDecision::Drift { statistic: diff };
            }
        }
        DriftDecision::Stable
    }
}

/// Gama et al. Drift Detection Method on a 0/1 error stream.
#[derive(Clone, Debug)]
pub struct Ddm {
    n: u64,
    p: f64,
    p_min: f64,
    s_min: f64,
}

impl Default for Ddm {
    fn default() -> Self {
        Self {
            n: 0,
            p: 0.0,
            p_min: f64::INFINITY,
            s_min: f64::INFINITY,
        }
    }
}

impl Ddm {
    /// Fresh DDM detector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with a 0/1 error indicator.
    pub fn update(&mut self, x: f64, session: &Session) -> Result<Qualified<DriftDecision>> {
        let mut ctx = FitCtx::with_session(session.child("update"));
        let e = if x > 0.5 { 1.0 } else { 0.0 };
        self.n += 1;
        self.p += (e - self.p) / self.n as f64;
        let s = (self.p * (1.0 - self.p) / self.n as f64).max(0.0).sqrt();
        if self.p + s < self.p_min + self.s_min {
            self.p_min = self.p;
            self.s_min = s;
        }
        let stat = self.p + s;
        let decision = if self.n >= 30 && stat > self.p_min + 3.0 * self.s_min {
            ctx.push(
                Issue::builder(IssueCode::ConceptDriftDetected)
                    .message(format!("DDM drift: p+s={stat:.4} > p_min+3s_min"))
                    .metric("drift_statistic", stat)
                    .build(),
            );
            self.n = 0;
            self.p = 0.0;
            self.p_min = f64::INFINITY;
            self.s_min = f64::INFINITY;
            DriftDecision::Drift { statistic: stat }
        } else if self.n >= 30 && stat > self.p_min + 2.0 * self.s_min {
            DriftDecision::Warning { statistic: stat }
        } else {
            DriftDecision::Stable
        };
        ctx.finish(decision)
    }
}

/// Two-sided Page–Hinkley test.
#[derive(Clone, Debug)]
pub struct PageHinkley {
    /// Magnitude of change to detect.
    pub delta: f64,
    /// Threshold `λ`.
    pub lambda: f64,
    mean: f64,
    n: u64,
    cum: f64,
    min_cum: f64,
    max_cum: f64,
}

impl Default for PageHinkley {
    fn default() -> Self {
        Self {
            delta: 0.005,
            lambda: 50.0,
            mean: 0.0,
            n: 0,
            cum: 0.0,
            min_cum: 0.0,
            max_cum: 0.0,
        }
    }
}

impl PageHinkley {
    /// Detector with change magnitude `delta` and threshold `lambda`.
    pub fn new(delta: f64, lambda: f64) -> Self {
        Self {
            delta,
            lambda,
            ..Self::default()
        }
    }

    /// Update with a scalar observation.
    pub fn update(&mut self, x: f64, session: &Session) -> Result<Qualified<DriftDecision>> {
        let mut ctx = FitCtx::with_session(session.child("update"));
        if !x.is_finite() {
            return ctx.finish(DriftDecision::Stable);
        }
        self.n += 1;
        self.mean += (x - self.mean) / self.n as f64;
        self.cum += x - self.mean - self.delta;
        if self.cum < self.min_cum {
            self.min_cum = self.cum;
        }
        if self.cum > self.max_cum {
            self.max_cum = self.cum;
        }
        let up = self.cum - self.min_cum;
        let down = self.max_cum - self.cum;
        let stat = up.max(down);
        let decision = if stat > self.lambda {
            ctx.push(
                Issue::builder(IssueCode::ConceptDriftDetected)
                    .message(format!("Page–Hinkley PH={stat:.4} > λ={}", self.lambda))
                    .metric("drift_statistic", stat)
                    .build(),
            );
            self.n = 0;
            self.mean = 0.0;
            self.cum = 0.0;
            self.min_cum = 0.0;
            self.max_cum = 0.0;
            DriftDecision::Drift { statistic: stat }
        } else {
            DriftDecision::Stable
        };
        ctx.finish(decision)
    }
}

/// A handful of Hoeffding trees with a per-tree ADWIN detector.
#[derive(Clone, Debug)]
pub struct AdaptiveRandomForest {
    /// Number of trees.
    pub n_estimators: usize,
    trees: Vec<HoeffdingTree>,
    detectors: Vec<Adwin>,
    n_seen: u64,
    updates: u64,
    initialized: bool,
    rng: Rng,
}

impl Default for AdaptiveRandomForest {
    fn default() -> Self {
        Self {
            n_estimators: 3,
            trees: Vec::new(),
            detectors: Vec::new(),
            n_seen: 0,
            updates: 0,
            initialized: false,
            rng: Rng::new(7),
        }
    }
}

impl AdaptiveRandomForest {
    /// Forest with `n_estimators` trees.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators: n_estimators.max(1),
            ..Self::default()
        }
    }

    fn ensure(&mut self, p: usize) {
        if self.initialized {
            return;
        }
        self.trees = (0..self.n_estimators)
            .map(|_| {
                let mut t = HoeffdingTree::new();
                t.min_samples = 15;
                t.n_features = p;
                t.root = HtNode::Leaf(HtLeaf::new(0, p));
                t.initialized = true;
                t
            })
            .collect();
        self.detectors = (0..self.n_estimators).map(|_| Adwin::new(0.002)).collect();
        self.initialized = true;
    }
}

impl PartialFit for AdaptiveRandomForest {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if self.initialized && self.trees[0].n_features != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        self.ensure(x.ncols());
        inspect_online_xy(&mut ctx, x, Some(y));
        let mut resets = 0u64;
        let mut splits = 0u64;
        for k in 0..self.trees.len() {
            // Poisson(1) bootstrap: include each row with P(count≥1) ≈ 0.63.
            let mut xb = x.clone();
            let mut yb = y.clone();
            let mut keep = 0usize;
            for i in 0..x.nrows() {
                if self.rng.uniform() < 0.632 || x.nrows() == 1 {
                    if keep != i {
                        for j in 0..x.ncols() {
                            xb.set(keep, j, x.get(i, j));
                        }
                        yb[keep] = y[i];
                    }
                    keep += 1;
                }
            }
            if keep == 0 {
                keep = 1;
            }
            let xs = Matrix::from_fn(keep, x.ncols(), |i, j| xb.get(i, j));
            let ys = Vector::from_iter((0..keep).map(|i| yb[i]));
            let q = self.trees[k].partial_fit(&xs, Some(&ys), &session.child("tree"))?;
            if q.value.quality.information_gain.unwrap_or(0.0) > 0.5 {
                splits += 1;
            }
            for i in 0..x.nrows() {
                let pred = self.trees[k].predict_one(x, i);
                let err = if (pred - as01(y[i])).abs() > 0.5 {
                    1.0
                } else {
                    0.0
                };
                let d = self.detectors[k].update(err, &session.child("adwin"))?;
                if matches!(d.value, DriftDecision::Drift { .. }) {
                    self.trees[k].reset();
                    resets += 1;
                    ctx.push(
                        Issue::builder(IssueCode::ConceptDriftDetected)
                            .message(format!("ARF tree {k} reset after ADWIN drift"))
                            .build(),
                    );
                }
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(resets as f64 + 0.1 * splits as f64);
        q.information_gain = Some(splits as f64 + resets as f64);
        q.still_identified = self.n_seen >= 15;
        q.warmup = self.n_seen < 15;
        q.explanation = format!(
            "ARF: {} trees, {splits} splits, {resets} ADWIN resets on {} rows",
            self.trees.len(),
            x.nrows()
        );
        if q.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("adaptive forest still warming up")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                format!("{splits} leaf splits, {resets} tree resets"),
                "bootstrap Hoeffding updates + per-tree ADWIN",
                "pre-batch forest",
                "post-batch forest",
            ),
        )
    }
}

impl Predict for AdaptiveRandomForest {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut votes = 0.0;
            for t in &self.trees {
                votes += t.predict_one(x, i);
            }
            if votes >= self.trees.len() as f64 * 0.5 {
                1.0
            } else {
                0.0
            }
        }));
        ctx.finish(y)
    }
}

/// Adaptive random forest for regression (river `ARFRegressor` lite).
///
/// Each tree is online RLS with a Poisson bootstrap and a per-tree ADWIN
/// detector on absolute residual. Nested `partial_fit` of the leaves adds
/// extra incremental events (allowed).
#[derive(Clone, Debug)]
pub struct AdaptiveRandomForestRegressor {
    /// Number of RLS leaves.
    pub n_estimators: usize,
    trees: Vec<LinearRegression>,
    detectors: Vec<Adwin>,
    n_features: usize,
    n_seen: u64,
    updates: u64,
    initialized: bool,
    rng: Rng,
}

impl Default for AdaptiveRandomForestRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 3,
            trees: Vec::new(),
            detectors: Vec::new(),
            n_features: 0,
            n_seen: 0,
            updates: 0,
            initialized: false,
            rng: Rng::new(11),
        }
    }
}

impl AdaptiveRandomForestRegressor {
    /// Forest with `n_estimators` RLS leaves.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators: n_estimators.max(1),
            ..Self::default()
        }
    }

    fn ensure(&mut self) {
        if self.initialized {
            return;
        }
        self.trees = (0..self.n_estimators)
            .map(|_| LinearRegression::new(1.0))
            .collect();
        self.detectors = (0..self.n_estimators).map(|_| Adwin::new(0.002)).collect();
        self.initialized = true;
    }
}

impl PartialFit for AdaptiveRandomForestRegressor {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if self.initialized && self.n_features != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        self.ensure();
        self.n_features = x.ncols();
        inspect_online_xy(&mut ctx, x, Some(y));
        let mut resets = 0u64;
        let mut delta: f64 = 0.0;
        for k in 0..self.trees.len() {
            let mut xb = x.clone();
            let mut yb = y.clone();
            let mut keep = 0usize;
            for i in 0..x.nrows() {
                if self.rng.uniform() < 0.632 || x.nrows() == 1 {
                    if keep != i {
                        for j in 0..x.ncols() {
                            xb.set(keep, j, x.get(i, j));
                        }
                        yb[keep] = y[i];
                    }
                    keep += 1;
                }
            }
            if keep == 0 {
                keep = 1;
            }
            let xs = Matrix::from_fn(keep, x.ncols(), |i, j| xb.get(i, j));
            let ys = Vector::from_iter((0..keep).map(|i| yb[i]));
            let before = self.trees[k].coef();
            let q = self.trees[k].partial_fit(&xs, Some(&ys), &session.child("rls"))?;
            delta += q.value.quality.parameter_delta_norm.unwrap_or(0.0);
            let after = self.trees[k].coef();
            if before.len() == after.len() {
                delta += after.sub(&before).norm();
            }
            for i in 0..x.nrows() {
                let pred = self.trees[k].predict_row(x, i);
                let err = (pred - y[i]).abs();
                let d = self.detectors[k].update(err, &session.child("adwin"))?;
                if matches!(d.value, DriftDecision::Drift { .. }) {
                    self.trees[k] = LinearRegression::new(1.0);
                    resets += 1;
                    ctx.push(
                        Issue::builder(IssueCode::ConceptDriftDetected)
                            .message(format!("ARF-regressor leaf {k} reset after ADWIN drift"))
                            .build(),
                    );
                }
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta + resets as f64);
        q.information_gain = Some(delta + resets as f64);
        q.still_identified = self.n_seen >= 15;
        q.warmup = self.n_seen < 15;
        q.explanation = format!(
            "ARF-regressor: {} RLS leaves, {resets} ADWIN resets on {} rows",
            self.trees.len(),
            x.nrows()
        );
        if q.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("adaptive forest regressor still warming up")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                format!("{resets} leaf resets, Δθ={delta:.6e}"),
                "bootstrap RLS updates + per-leaf ADWIN on |residual|",
                "pre-batch forest",
                "post-batch forest",
            ),
        )
    }
}

impl Predict for AdaptiveRandomForestRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let n_trees = self.trees.len().max(1) as f64;
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut s = 0.0;
            for t in &self.trees {
                s += t.predict_row(x, i);
            }
            s / n_trees
        }));
        ctx.finish(y)
    }
}

/// Online column-wise standard scaler (Welford).
#[derive(Clone, Debug)]
pub struct OnlineStandardScaler {
    mean: Vector,
    m2: Vector,
    count: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for OnlineStandardScaler {
    fn default() -> Self {
        Self {
            mean: Vector::zeros(0),
            m2: Vector::zeros(0),
            count: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl OnlineStandardScaler {
    /// Empty scaler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Running means.
    pub fn mean(&self) -> &Vector {
        &self.mean
    }
}

impl PartialFit for OnlineStandardScaler {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if !self.initialized {
            self.mean = Vector::zeros(x.ncols());
            self.m2 = Vector::zeros(x.ncols());
            self.count = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if self.mean.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.mean.clone();
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                let n = self.count[j] + 1.0;
                let d = v - self.mean[j];
                self.mean[j] += d / n;
                self.m2[j] += d * (v - self.mean[j]);
                self.count[j] = n;
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.mean.sub(&before);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.norm());
        q.information_gain = Some(delta.norm());
        q.still_identified = self.n_seen >= 2;
        q.warmup = self.n_seen < 2;
        q.explanation = format!("online StandardScaler Welford on {} rows", x.nrows());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "running column means and M2",
                "Welford online moments",
                "previous moments",
                "updated moments",
            ),
        )
    }
}

impl Transform for OnlineStandardScaler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.mean.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j >= p {
                return x.get(i, j);
            }
            let std = if self.count[j] >= 2.0 {
                (self.m2[j] / (self.count[j] - 1.0)).max(0.0).sqrt()
            } else {
                1.0
            };
            let s = if std > 0.0 { std } else { 1.0 };
            (x.get(i, j) - self.mean[j]) / s
        });
        ctx.finish(out)
    }
}

/// Mini-batch / streaming k-means.
#[derive(Clone, Debug)]
pub struct StreamKMeans {
    /// Number of centroids.
    pub k: usize,
    centers: Matrix,
    counts: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for StreamKMeans {
    fn default() -> Self {
        Self {
            k: 2,
            centers: Matrix::zeros(0, 0),
            counts: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl StreamKMeans {
    /// `k` streaming centroids.
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            ..Self::default()
        }
    }

    /// Current centroids (`k × p`).
    pub fn centers(&self) -> &Matrix {
        &self.centers
    }
}

impl PartialFit for StreamKMeans {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        if x.nrows() == 0 || x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "empty"),
            );
        }
        if !self.initialized {
            let k = self.k.min(x.nrows());
            self.centers = Matrix::from_fn(k, x.ncols(), |i, j| x.get(i, j));
            self.counts = Vector::zeros(k);
            self.initialized = true;
        } else if self.centers.ncols() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.centers.clone();
        let mut empty = vec![true; self.centers.nrows()];
        for i in 0..x.nrows() {
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for c in 0..self.centers.nrows() {
                let mut d = 0.0;
                for j in 0..x.ncols() {
                    let t = x.get(i, j) - self.centers.get(c, j);
                    d += t * t;
                }
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            empty[best] = false;
            self.counts[best] += 1.0;
            let eta = 1.0 / self.counts[best];
            for j in 0..x.ncols() {
                let v = self.centers.get(best, j) + eta * (x.get(i, j) - self.centers.get(best, j));
                self.centers.set(best, j, v);
            }
        }
        for (c, is_empty) in empty.iter().enumerate() {
            if *is_empty {
                ctx.push(
                    Issue::builder(IssueCode::EmptyCluster)
                        .message(format!("stream k-means centroid {c} received no points"))
                        .build(),
                );
            }
        }
        if empty.iter().filter(|e| **e).count() + 1 >= self.centers.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DegenerateClusters)
                    .message("stream k-means collapsed toward a single centroid")
                    .build(),
            );
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut dn = 0.0;
        for c in 0..self.centers.nrows() {
            for j in 0..self.centers.ncols() {
                let d = self.centers.get(c, j) - before.get(c, j);
                dn += d * d;
            }
        }
        dn = dn.sqrt();
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(dn);
        q.information_gain = Some(dn);
        q.still_identified = self.n_seen >= self.k as u64;
        q.warmup = self.n_seen < self.k as u64;
        q.explanation = format!("stream k-means assignment + centroid step, ||Δμ||={dn:.4e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "centroid coordinates",
                "online mean update of the assigned cluster",
                "previous centroids",
                "updated centroids",
            ),
        )
    }
}

impl Predict for StreamKMeans {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for c in 0..self.centers.nrows() {
                let mut d = 0.0;
                for j in 0..x.ncols().min(self.centers.ncols()) {
                    let t = x.get(i, j) - self.centers.get(c, j);
                    d += t * t;
                }
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            best as f64
        }));
        ctx.finish(y)
    }
}

/// Half-space trees for online anomaly scoring (Tan, Ting, Liu).
#[derive(Clone, Debug)]
pub struct HalfSpaceTrees {
    /// Number of trees.
    pub n_trees: usize,
    /// Maximum depth.
    pub max_depth: usize,
    trees: Vec<HsNode>,
    mins: Vector,
    maxs: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
    rng: Rng,
}

#[derive(Clone, Debug)]
struct HsNode {
    feature: usize,
    threshold: f64,
    mass: u64,
    depth: usize,
    left: Option<Box<HsNode>>,
    right: Option<Box<HsNode>>,
}

impl Default for HalfSpaceTrees {
    fn default() -> Self {
        Self {
            n_trees: 10,
            max_depth: 8,
            trees: Vec::new(),
            mins: Vector::zeros(0),
            maxs: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
            rng: Rng::new(11),
        }
    }
}

impl HalfSpaceTrees {
    /// Ensemble of half-space trees.
    pub fn new(n_trees: usize, max_depth: usize) -> Self {
        Self {
            n_trees: n_trees.max(1),
            max_depth: max_depth.max(1),
            ..Self::default()
        }
    }

    fn grow(&mut self, depth: usize, p: usize) -> HsNode {
        if depth >= self.max_depth || p == 0 {
            return HsNode {
                feature: 0,
                threshold: 0.0,
                mass: 0,
                depth,
                left: None,
                right: None,
            };
        }
        let feature = self.rng.below(p);
        let lo = self.mins[feature];
        let hi = self.maxs[feature];
        let threshold = if hi > lo {
            self.rng.uniform_range(lo, hi)
        } else {
            lo
        };
        HsNode {
            feature,
            threshold,
            mass: 0,
            depth,
            left: Some(Box::new(self.grow(depth + 1, p))),
            right: Some(Box::new(self.grow(depth + 1, p))),
        }
    }

    fn score_one(&self, x: &Matrix, i: usize) -> f64 {
        if self.n_seen == 0 || self.trees.is_empty() {
            return 0.0;
        }
        let mut s = 0.0;
        for t in &self.trees {
            s += walk_score(t, x, i, self.n_seen);
        }
        s / self.trees.len() as f64
    }
}

fn walk_insert(node: &mut HsNode, x: &Matrix, i: usize) {
    node.mass += 1;
    if node.left.is_none() || node.right.is_none() {
        return;
    }
    let v = if node.feature < x.ncols() {
        x.get(i, node.feature)
    } else {
        0.0
    };
    if v <= node.threshold {
        if let Some(l) = node.left.as_mut() {
            walk_insert(l, x, i);
        }
    } else if let Some(r) = node.right.as_mut() {
        walk_insert(r, x, i);
    }
}

fn walk_score(node: &HsNode, x: &Matrix, i: usize, n_seen: u64) -> f64 {
    let v = if node.feature < x.ncols() {
        x.get(i, node.feature)
    } else {
        0.0
    };
    let child = if v <= node.threshold {
        node.left.as_deref()
    } else {
        node.right.as_deref()
    };
    match child {
        Some(c) if c.mass > 0 || c.left.is_some() => walk_score(c, x, i, n_seen),
        _ => {
            let mass = node.mass.max(1) as f64;
            // High score = rare (low mass relative to n, shallow isolation).
            1.0 - (mass / n_seen.max(1) as f64) * (1.0 + node.depth as f64).recip()
        }
    }
}

impl PartialFit for HalfSpaceTrees {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        if x.ncols() == 0 || x.nrows() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "empty"),
            );
        }
        if !self.initialized {
            self.mins = Vector::from_iter((0..x.ncols()).map(|j| {
                (0..x.nrows())
                    .map(|i| x.get(i, j))
                    .fold(f64::INFINITY, f64::min)
            }));
            self.maxs = Vector::from_iter((0..x.ncols()).map(|j| {
                (0..x.nrows())
                    .map(|i| x.get(i, j))
                    .fold(f64::NEG_INFINITY, f64::max)
            }));
            for j in 0..x.ncols() {
                if !self.mins[j].is_finite()
                    || !self.maxs[j].is_finite()
                    || self.mins[j] == self.maxs[j]
                {
                    self.mins[j] = 0.0;
                    self.maxs[j] = 1.0;
                } else {
                    let pad = 0.1 * (self.maxs[j] - self.mins[j]);
                    self.mins[j] -= pad;
                    self.maxs[j] += pad;
                }
            }
            let p = x.ncols();
            self.trees = (0..self.n_trees).map(|_| self.grow(0, p)).collect();
            self.initialized = true;
        } else if self.mins.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        for i in 0..x.nrows() {
            for t in &mut self.trees {
                walk_insert(t, x, i);
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(x.nrows() as f64);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n_seen > 4;
        q.warmup = self.n_seen < 4;
        q.explanation = format!(
            "half-space trees: mass incremented along {} paths × {} rows",
            self.trees.len(),
            x.nrows()
        );
        if q.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("half-space trees have seen fewer than 4 points")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "per-node mass counts",
                "path increment in a random half-space partition",
                "previous masses",
                "updated masses",
            ),
        )
    }
}

impl Predict for HalfSpaceTrees {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        ctx.finish(Vector::from_iter(
            (0..x.nrows()).map(|i| self.score_one(x, i)),
        ))
    }
}

/// Rolling 0/1 accuracy. `x` is the prediction, `y` the label.
#[derive(Clone, Debug, Default)]
pub struct OnlineAccuracy {
    correct: u64,
    n: u64,
    updates: u64,
}

impl OnlineAccuracy {
    /// Empty metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current accuracy, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.correct as f64 / self.n as f64
        }
    }
}

impl PartialFit for OnlineAccuracy {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        metric_partial(
            session,
            "accuracy",
            x,
            y,
            self.updates,
            self.n,
            |pred, truth| {
                self.n += 1;
                if (pred.round() - truth.round()).abs() < 1e-9 {
                    self.correct += 1;
                }
                self.updates += 1;
                self.score()
            },
        )
    }
}

/// Rolling mean squared error. `x` is the prediction, `y` the target.
#[derive(Clone, Debug, Default)]
pub struct OnlineMse {
    sse: f64,
    n: u64,
    updates: u64,
}

impl OnlineMse {
    /// Empty metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current MSE, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.sse / self.n as f64
        }
    }
}

impl PartialFit for OnlineMse {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        metric_partial(session, "mse", x, y, self.updates, self.n, |pred, truth| {
            let e = pred - truth;
            self.sse += e * e;
            self.n += 1;
            self.updates += 1;
            self.score()
        })
    }
}

/// Rolling coefficient of determination.
#[derive(Clone, Debug, Default)]
pub struct OnlineR2 {
    sse: f64,
    sy: f64,
    sy2: f64,
    n: u64,
    updates: u64,
}

impl OnlineR2 {
    /// Empty metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current R², or NaN when unidentified.
    pub fn score(&self) -> f64 {
        if self.n < 2 {
            return f64::NAN;
        }
        let mean = self.sy / self.n as f64;
        let sst = self.sy2 - self.n as f64 * mean * mean;
        if sst.abs() <= 1e-15 {
            return f64::NAN;
        }
        1.0 - self.sse / sst
    }
}

impl PartialFit for OnlineR2 {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        metric_partial(session, "r2", x, y, self.updates, self.n, |pred, truth| {
            let e = pred - truth;
            self.sse += e * e;
            self.sy += truth;
            self.sy2 += truth * truth;
            self.n += 1;
            self.updates += 1;
            self.score()
        })
    }
}

/// Additive Holt–Winters with online level / trend / seasonal updates.
#[derive(Clone, Debug)]
pub struct HoltWintersOnline {
    /// Level smoothing.
    pub alpha: f64,
    /// Trend smoothing.
    pub beta: f64,
    /// Seasonal smoothing.
    pub gamma: f64,
    /// Seasonal period.
    pub period: usize,
    level: f64,
    trend: f64,
    season: Vec<f64>,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for HoltWintersOnline {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: 0.1,
            gamma: 0.1,
            period: 4,
            level: 0.0,
            trend: 0.0,
            season: Vec::new(),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl HoltWintersOnline {
    /// Additive Holt–Winters with the given period.
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(1),
            season: vec![0.0; period.max(1)],
            ..Self::default()
        }
    }

    /// Current level.
    pub fn level(&self) -> f64 {
        self.level
    }

    /// Current trend.
    pub fn trend(&self) -> f64 {
        self.trend
    }

    fn step(&mut self, y: f64) {
        let p = self.period.max(1);
        if self.season.len() != p {
            self.season = vec![0.0; p];
        }
        let idx = (self.n_seen as usize) % p;
        let s = self.season[idx];
        let last_level = self.level;
        if !self.initialized {
            self.level = y;
            self.initialized = true;
            self.season[idx] = 0.0;
        } else {
            let new_level = self.alpha * (y - s) + (1.0 - self.alpha) * (self.level + self.trend);
            let new_trend = self.beta * (new_level - last_level) + (1.0 - self.beta) * self.trend;
            let new_season = self.gamma * (y - new_level) + (1.0 - self.gamma) * s;
            self.level = new_level;
            self.trend = new_trend;
            self.season[idx] = new_season;
        }
        self.n_seen += 1;
    }
}

impl PartialFit for HoltWintersOnline {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let series: Vec<f64> = if let Some(y) = y {
            y.as_slice().to_vec()
        } else if x.ncols() > 0 {
            (0..x.nrows()).map(|i| x.get(i, 0)).collect()
        } else {
            Vec::new()
        };
        if series.is_empty() {
            if !self.initialized {
                ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            }
            return finish_explain(
                ctx,
                reject_explain(self.updates, 0, self.n_seen, "empty series"),
            );
        }
        let before_l = self.level;
        let before_t = self.trend;
        let mut sse = 0.0;
        for &yi in &series {
            let idx = (self.n_seen as usize) % self.period.max(1);
            let seas = if idx < self.season.len() {
                self.season[idx]
            } else {
                0.0
            };
            let pred = self.level + self.trend + seas;
            if self.initialized {
                let e = yi - pred;
                sse += e * e;
            }
            self.step(yi);
        }
        self.updates += 1;
        let warmup = self.n_seen < self.period as u64;
        if warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .message(format!(
                        "Holt–Winters has seen {} < period={}",
                        self.n_seen, self.period
                    ))
                    .build(),
            );
        }
        if self.n_seen < 2 * self.period as u64 && self.period > 1 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSeasonalCycles)
                    .message(
                        "fewer than two full seasonal cycles; seasonal terms are weakly identified",
                    )
                    .build(),
            );
        }
        let dn = ((self.level - before_l).powi(2) + (self.trend - before_t).powi(2)).sqrt();
        let mut q = IncrementalQuality::new(self.updates - 1, series.len(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(dn);
        q.information_gain = Some(sse);
        q.still_identified = !warmup;
        q.warmup = warmup;
        q.explanation = format!(
            "Holt–Winters additive update: Δlevel={:.4e} Δtrend={:.4e} sse={sse:.4e}",
            self.level - before_l,
            self.trend - before_t
        );
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "level, trend, and seasonal factors",
                "additive Holt–Winters smoothing equations",
                format!("L={before_l:.4} T={before_t:.4}"),
                format!("L={:.4} T={:.4}", self.level, self.trend),
            ),
        )
    }
}

impl Predict for HoltWintersOnline {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let p = self.period.max(1);
        let y = Vector::from_iter((0..x.nrows()).map(|h| {
            let idx = (self.n_seen as usize + h) % p;
            let s = if idx < self.season.len() {
                self.season[idx]
            } else {
                0.0
            };
            self.level + (h as f64 + 1.0) * self.trend + s
        }));
        ctx.finish(y)
    }
}

fn inspect_online_xy(ctx: &mut FitCtx, x: &Matrix, y: Option<&Vector>) {
    let (n, p) = x.shape();
    ctx.report.set_sample_shape(n, p);
    if n == 0 || p == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message(format!("online design is {n}×{p}"))
                .build(),
        );
        return;
    }
    if let Some(y) = y {
        if y.len() != n {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!("y.len()={} but X has {n} rows", y.len()))
                    .build(),
            );
        }
    }
    // Do not call inspect_xy: a 1-row batch is a constant target and would abort.
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

fn as01(y: f64) -> f64 {
    if y >= 0.5 {
        1.0
    } else {
        0.0
    }
}

fn as_pm(y: f64) -> f64 {
    if y >= 0.5 {
        1.0
    } else {
        -1.0
    }
}

fn logloss_of(x: &Matrix, y: &Vector, coef: &Vector, intercept: f64) -> f64 {
    if x.nrows() == 0 || coef.len() != x.ncols() {
        return f64::NAN;
    }
    let mut s = 0.0;
    for i in 0..x.nrows() {
        let mut eta = intercept;
        for j in 0..x.ncols() {
            eta += coef[j] * x.get(i, j);
        }
        let mu = sigmoid(eta).clamp(1e-12, 1.0 - 1e-12);
        let yi = as01(y[i]);
        s += -(yi * mu.ln() + (1.0 - yi) * (1.0 - mu).ln());
    }
    s / x.nrows() as f64
}

fn top_moved(delta: &Vector, k: usize) -> Vec<(String, f64)> {
    let mut idx: Vec<usize> = (0..delta.len()).collect();
    idx.sort_by(|a, b| {
        delta[*b]
            .abs()
            .partial_cmp(&delta[*a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter()
        .take(k)
        .map(|j| (format!("theta[{j}]"), delta[j]))
        .collect()
}

/// Adaptive model rules (river `AMRules`): a default RLS leaf that may split
/// once when a Hoeffding-bound variance reduction is identified.
///
/// Each `partial_fit` names the leaf that moved, the candidate split (if any),
/// and whether the post-update state is still a single unidentified intercept.
#[derive(Clone, Debug)]
pub struct AdaptiveModelRules {
    /// Forgetting factor of each leaf RLS.
    pub forgetting: f64,
    /// Hoeffding δ.
    pub delta: f64,
    p: usize,
    split: Option<(usize, f64)>,
    default: RuleLeaf,
    left: Option<RuleLeaf>,
    right: Option<RuleLeaf>,
    feat_mean: Vector,
    feat_n: u64,
    parent_sse: f64,
    parent_n: u64,
    left_sse: Vector,
    left_n: Vector,
    right_sse: Vector,
    right_n: Vector,
    updates: u64,
    n_seen: u64,
}

#[derive(Clone, Debug)]
struct RuleLeaf {
    theta: Vector,
    p_mat: Option<faer::Mat<f64>>,
    n: u64,
}

impl RuleLeaf {
    fn new(p: usize, p0: f64) -> Self {
        Self {
            theta: Vector::zeros(p),
            p_mat: Some(faer::Mat::<f64>::from_fn(p, p, |i, j| {
                if i == j {
                    p0
                } else {
                    0.0
                }
            })),
            n: 0,
        }
    }

    fn predict(&self, z: &Vector) -> f64 {
        if self.theta.len() != z.len() {
            return 0.0;
        }
        self.theta.dot(z)
    }

    fn update(&mut self, z: &Vector, y: f64, lam: f64) -> f64 {
        let p = z.len();
        if self.theta.len() != p {
            *self = Self::new(p, 1_000.0);
        }
        let pred = self.predict(z);
        let Some(pm) = self.p_mat.as_mut() else {
            return 0.0;
        };
        let mut px = Vector::zeros(p);
        for i in 0..p {
            let mut s = 0.0;
            for j in 0..p {
                s += pm[(i, j)] * z[j];
            }
            px[i] = s;
        }
        let denom = lam + z.dot(&px);
        if denom.abs() <= 1e-18 {
            return 0.0;
        }
        let err = y - pred;
        let mut delta_n = 0.0;
        for i in 0..p {
            let d = px[i] * err / denom;
            self.theta[i] += d;
            delta_n += d * d;
        }
        for i in 0..p {
            for j in 0..p {
                pm[(i, j)] = (pm[(i, j)] - px[i] * px[j] / denom) / lam;
            }
        }
        self.n += 1;
        delta_n.sqrt()
    }
}

impl Default for AdaptiveModelRules {
    fn default() -> Self {
        Self {
            forgetting: 1.0,
            delta: 0.01,
            p: 0,
            split: None,
            default: RuleLeaf::new(0, 1_000.0),
            left: None,
            right: None,
            feat_mean: Vector::zeros(0),
            feat_n: 0,
            parent_sse: 0.0,
            parent_n: 0,
            left_sse: Vector::zeros(0),
            left_n: Vector::zeros(0),
            right_sse: Vector::zeros(0),
            right_n: Vector::zeros(0),
            updates: 0,
            n_seen: 0,
        }
    }
}

impl AdaptiveModelRules {
    /// AMRules with forgetting `λ`.
    pub fn new(forgetting: f64) -> Self {
        Self {
            forgetting,
            ..Self::default()
        }
    }

    fn ensure(&mut self, p_x: usize) {
        let p = p_x + 1;
        if self.p == p {
            return;
        }
        self.p = p;
        self.default = RuleLeaf::new(p, 1_000.0);
        self.feat_mean = Vector::zeros(p_x);
        self.left_sse = Vector::zeros(p_x);
        self.left_n = Vector::zeros(p_x);
        self.right_sse = Vector::zeros(p_x);
        self.right_n = Vector::zeros(p_x);
    }

    fn row(x: &Matrix, i: usize) -> Vector {
        let mut z = Vector::zeros(x.ncols() + 1);
        z[0] = 1.0;
        for j in 0..x.ncols() {
            z[j + 1] = x.get(i, j);
        }
        z
    }

    fn route(&mut self, x: &Matrix, i: usize) -> &mut RuleLeaf {
        match self.split {
            Some((j, t)) if x.get(i, j) <= t => {
                if self.left.is_none() {
                    self.left = Some(RuleLeaf::new(self.p, 1_000.0));
                }
                self.left.as_mut().unwrap()
            }
            Some((_j, _)) => {
                if self.right.is_none() {
                    self.right = Some(RuleLeaf::new(self.p, 1_000.0));
                }
                self.right.as_mut().unwrap()
            }
            None => &mut self.default,
        }
    }

    fn maybe_split(&mut self, ctx: &mut FitCtx) -> Option<(usize, f64, f64)> {
        if self.split.is_some() || self.feat_mean.len() == 0 || self.parent_n < 20 {
            return None;
        }
        let r2 = 1.0; // residual range bound used by Hoeffding
        let eps =
            (r2 * r2 * (1.0 / self.delta.max(1e-12)).ln() / (2.0 * self.parent_n as f64)).sqrt();
        let parent_var = self.parent_sse / self.parent_n as f64;
        let mut best = None;
        let mut best_gain = 0.0;
        let mut second = 0.0;
        for j in 0..self.feat_mean.len() {
            let nl = self.left_n[j];
            let nr = self.right_n[j];
            if nl < 5.0 || nr < 5.0 {
                continue;
            }
            let vl = self.left_sse[j] / nl;
            let vr = self.right_sse[j] / nr;
            let gain = parent_var - (nl * vl + nr * vr) / (nl + nr);
            if gain > best_gain {
                second = best_gain;
                best_gain = gain;
                best = Some((j, self.feat_mean[j], gain));
            } else if gain > second {
                second = gain;
            }
        }
        if let Some((j, t, gain)) = best {
            if gain - second > eps && gain > 0.0 {
                self.split = Some((j, t));
                self.left = Some(RuleLeaf::new(self.p, 1_000.0));
                self.right = Some(RuleLeaf::new(self.p, 1_000.0));
                ctx.push(
                    Issue::builder(IssueCode::ConceptDriftDetected)
                        .severity(Severity::Advisory)
                        .message(format!(
                            "AMRules split on feature {j} at {t:.6e} (gain={gain:.6e}, ε={eps:.6e})"
                        ))
                        .build(),
                );
                return Some((j, t, gain));
            }
        }
        None
    }
}

impl PartialFit for AdaptiveModelRules {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            if self.n_seen == 0 {
                ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            }
            return finish_explain(
                ctx,
                reject_explain(self.updates, 0, self.n_seen, "AMRules needs a target"),
            );
        };
        if x.nrows() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("AMRules x rows ≠ y length")
                    .build(),
            );
        }
        self.ensure(x.ncols());
        let mut dsum = 0.0;
        let mut loss_before = 0.0;
        let mut loss_after = 0.0;
        let mut leaf_hits = [0u64; 3];
        for i in 0..y.len() {
            let z = Self::row(x, i);
            let pred_b = match self.split {
                Some((j, t)) if x.get(i, j) <= t => {
                    self.left.as_ref().map(|l| l.predict(&z)).unwrap_or(0.0)
                }
                Some(_) => self.right.as_ref().map(|l| l.predict(&z)).unwrap_or(0.0),
                None => self.default.predict(&z),
            };
            loss_before += (pred_b - y[i]) * (pred_b - y[i]);
            let which = match self.split {
                Some((j, t)) if x.get(i, j) <= t => 1,
                Some(_) => 2,
                None => 0,
            };
            leaf_hits[which] += 1;
            let lam = self.forgetting;
            let dn = self.route(x, i).update(&z, y[i], lam);
            dsum += dn;
            let pred_a = match self.split {
                Some((j, t)) if x.get(i, j) <= t => {
                    self.left.as_ref().map(|l| l.predict(&z)).unwrap_or(0.0)
                }
                Some(_) => self.right.as_ref().map(|l| l.predict(&z)).unwrap_or(0.0),
                None => self.default.predict(&z),
            };
            let e = pred_a - y[i];
            loss_after += e * e;
            if self.split.is_none() {
                self.parent_sse += e * e;
                self.parent_n += 1;
                self.feat_n += 1;
                for j in 0..x.ncols() {
                    let v = x.get(i, j);
                    self.feat_mean[j] += (v - self.feat_mean[j]) / self.feat_n as f64;
                    if v <= self.feat_mean[j] {
                        self.left_sse[j] += e * e;
                        self.left_n[j] += 1.0;
                    } else {
                        self.right_sse[j] += e * e;
                        self.right_n[j] += 1.0;
                    }
                }
            }
        }
        self.n_seen += y.len() as u64;
        self.updates += 1;
        let split = self.maybe_split(&mut ctx);
        let identified = self.n_seen as usize > self.p.max(1);
        let warmup = self.n_seen < 10;
        let what = if let Some((j, t, g)) = split {
            format!("split on x[{j}]≤{t:.4e} gain={g:.4e}; leaf hits {leaf_hits:?}")
        } else if self.split.is_some() {
            format!("updated split leaves; hits {leaf_hits:?}")
        } else {
            format!("updated default RLS; hits {leaf_hits:?}")
        };
        let expl = pack_linear_explain(
            &mut ctx,
            self.updates,
            y.len(),
            self.n_seen,
            &Vector::from_slice(&[dsum]),
            loss_before,
            loss_after,
            identified,
            warmup,
            &what,
            "RLS leaf update; a Hoeffding test may grow a stump",
        );
        finish_explain(ctx, expl)
    }
}

impl Predict for AdaptiveModelRules {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.p == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
        }
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let z = Self::row(x, i);
            match self.split {
                Some((j, t)) if x.get(i, j) <= t => {
                    self.left.as_ref().map(|l| l.predict(&z)).unwrap_or(0.0)
                }
                Some(_) => self.right.as_ref().map(|l| l.predict(&z)).unwrap_or(0.0),
                None => self.default.predict(&z),
            }
        }));
        ctx.finish(out)
    }
}

fn reject_explain(update: u64, batch: usize, n_seen: u64, why: &str) -> IncrementalExplain {
    IncrementalExplain::from_quality(
        IncrementalQuality::new(update, batch, n_seen),
        "nothing",
        why,
        "invalid",
        "invalid",
    )
}

fn flag_info(ctx: &mut FitCtx, q: &IncrementalQuality) {
    if q.is_uninformative(ctx.policy.uninformative_info_eps) {
        ctx.push(
            Issue::builder(IssueCode::UpdateWithZeroInformation)
                .incremental(q.clone())
                .message("this online update added no usable information")
                .build(),
        );
    }
}

fn pack_linear_explain(
    ctx: &mut FitCtx,
    updates: u64,
    batch: usize,
    n_seen: u64,
    delta: &Vector,
    loss_before: f64,
    loss_after: f64,
    identified: bool,
    warmup: bool,
    what: &str,
    why: &str,
) -> IncrementalExplain {
    let mut q = IncrementalQuality::new(updates.saturating_sub(1), batch, n_seen);
    q.effective_sample_size = n_seen as f64;
    q.parameter_delta_norm = Some(delta.norm());
    q.parameter_delta_max = Some(delta.max_abs());
    q.loss_before = Some(loss_before);
    q.loss_after = Some(loss_after);
    q.information_gain = Some((loss_before - loss_after).abs().max(delta.norm()));
    q.still_identified = identified;
    q.warmup = warmup;
    q.explanation = format!(
        "{what}: ||Δθ||={:.4e} loss {loss_before:.6e} → {loss_after:.6e}",
        delta.norm()
    );
    q.top_moved_parameters = top_moved(delta, 3);
    if q.is_uninformative(ctx.policy.uninformative_info_eps) {
        ctx.push(
            Issue::builder(IssueCode::UpdateWithZeroInformation)
                .incremental(q.clone())
                .message("batch residual / parameters did not move")
                .build(),
        );
    }
    if warmup {
        ctx.push(
            Issue::builder(IssueCode::WarmupIncomplete)
                .incremental(q.clone())
                .message("online linear model is still warming up")
                .build(),
        );
    }
    IncrementalExplain::from_quality(
        q,
        what,
        why,
        format!("loss={loss_before:.6e}"),
        format!("loss={loss_after:.6e}"),
    )
}

fn finish_explain(ctx: FitCtx, expl: IncrementalExplain) -> Result<Qualified<IncrementalExplain>> {
    ctx.session.record_incremental(expl.clone());
    ctx.finish(expl)
}

fn metric_partial<F>(
    session: &Session,
    name: &str,
    x: &Matrix,
    y: Option<&Vector>,
    updates: u64,
    n_seen: u64,
    mut on_pair: F,
) -> Result<Qualified<IncrementalExplain>>
where
    F: FnMut(f64, f64) -> f64,
{
    let mut ctx = FitCtx::with_session(session.child("partial_fit"));
    let Some(y) = y else {
        ctx.push(Issue::builder(IssueCode::MissingTarget).build());
        return finish_explain(ctx, reject_explain(updates, x.nrows(), n_seen, "missing y"));
    };
    if y.len() != x.nrows() {
        ctx.push(Issue::builder(IssueCode::DimensionMismatch).build());
    }
    let before = n_seen;
    let mut last = f64::NAN;
    for i in 0..x.nrows().min(y.len()) {
        last = on_pair(x.get(i, 0), y[i]);
    }
    let after_n = before + x.nrows().min(y.len()) as u64;
    let mut q = IncrementalQuality::new(updates, x.nrows(), after_n);
    q.effective_sample_size = after_n as f64;
    q.parameter_delta_norm = Some(0.0);
    q.information_gain = Some(x.nrows() as f64);
    q.still_identified = after_n > 1;
    q.warmup = after_n < 2;
    q.explanation = format!("online {name} ← {} pairs, score={last:.6}", x.nrows());
    q.loss_after = Some(last);
    if q.warmup {
        ctx.push(
            Issue::builder(IssueCode::WarmupIncomplete)
                .incremental(q.clone())
                .message(format!("online {name} has fewer than 2 observations"))
                .build(),
        );
    }
    finish_explain(
        ctx,
        IncrementalExplain::from_quality(
            q,
            format!("running {name}"),
            "sufficient-stat update of the rolling metric",
            format!("n={before}"),
            format!("n={after_n} score={last:.6}"),
        ),
    )
}

/// Kolmogorov–Smirnov windowing (river `KSWIN`).
///
/// The last `stat_size` observations are compared to the remainder of a
/// sliding window. A KS statistic above the Hoeffding-style threshold is
/// [`IssueCode::ConceptDriftDetected`].
#[derive(Clone, Debug)]
pub struct Kswin {
    /// Window length.
    pub window_size: usize,
    /// Length of the recent comparison sample.
    pub stat_size: usize,
    /// Type-I level \(\alpha\).
    pub alpha: f64,
    window: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl Default for Kswin {
    fn default() -> Self {
        Self {
            window_size: 100,
            stat_size: 30,
            alpha: 0.005,
            window: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl Kswin {
    /// Detector with window `w`.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            ..Self::default()
        }
    }

    fn ks(a: &[f64], b: &[f64]) -> f64 {
        let mut sa = a.to_vec();
        let mut sb = b.to_vec();
        sa.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        sb.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let mut i = 0usize;
        let mut j = 0usize;
        let mut d: f64 = 0.0;
        while i < sa.len() || j < sb.len() {
            let va = sa.get(i).copied().unwrap_or(f64::INFINITY);
            let vb = sb.get(j).copied().unwrap_or(f64::INFINITY);
            if va <= vb {
                i += 1;
            }
            if vb <= va {
                j += 1;
            }
            let fa = i as f64 / sa.len().max(1) as f64;
            let fb = j as f64 / sb.len().max(1) as f64;
            d = d.max((fa - fb).abs());
        }
        d
    }

    /// Consume one scalar and decide.
    pub fn update(&mut self, x: f64, session: &Session) -> Result<Qualified<DriftDecision>> {
        let mut ctx = FitCtx::with_session(session.child("update"));
        if x.is_finite() {
            self.window.push(x);
        }
        if self.window.len() > self.window_size {
            let drop = self.window.len() - self.window_size;
            self.window.drain(0..drop);
        }
        self.n_seen += 1;
        let r = self.stat_size.min(self.window.len() / 2);
        if self.window.len() < self.stat_size.max(16) * 2 {
            return ctx.finish(DriftDecision::Stable);
        }
        let n = self.window.len();
        let recent = &self.window[n - r..];
        let past = &self.window[..n - r];
        let stat = Self::ks(past, recent);
        let crit = ((-0.5 * (self.alpha.max(1e-12) / 2.0).ln())
            * ((past.len() + recent.len()) as f64 / (past.len() * recent.len()) as f64))
            .sqrt();
        if stat > crit {
            ctx.push(
                Issue::builder(IssueCode::ConceptDriftDetected)
                    .message(format!("KSWIN KS={stat:.4} > {crit:.4}"))
                    .metric("drift_statistic", stat)
                    .build(),
            );
            self.window.drain(0..n - r);
            return ctx.finish(DriftDecision::Drift { statistic: stat });
        }
        ctx.finish(DriftDecision::Stable)
    }
}

impl PartialFit for Kswin {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let before = self.window.len();
        let mut fired = 0.0;
        for i in 0..x.nrows() {
            match self.update(x.get(i, 0), &session.child("kswin")) {
                Ok(q) => {
                    if matches!(q.value, DriftDecision::Drift { .. }) {
                        fired += 1.0;
                    }
                }
                Err(e) => {
                    if e.primary().code == IssueCode::ConceptDriftDetected {
                        fired += 1.0;
                    }
                }
            }
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.window.len() as f64;
        q.information_gain = Some(fired);
        q.drift_statistic = Some(fired);
        q.still_identified = self.window.len() >= self.stat_size;
        q.warmup = self.window.len() < self.stat_size.max(16) * 2;
        q.explanation = format!(
            "KSWIN consumed {} rows; window {before}→{}, drift_cuts={fired}",
            x.nrows(),
            self.window.len()
        );
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "sliding KS window",
                "two-sample KS of recent vs remainder",
                format!("window={before}"),
                format!("window={} cuts={fired}", self.window.len()),
            ),
        )
    }
}

/// Hoeffding drift detection (river `HDDM_A`) on a 0/1 error stream.
#[derive(Clone, Debug)]
pub struct Hddm {
    /// Confidence \(\delta\).
    pub delta: f64,
    n: u64,
    mean: f64,
    n_min: u64,
    mean_min: f64,
    updates: u64,
}

impl Default for Hddm {
    fn default() -> Self {
        Self {
            delta: 0.002,
            n: 0,
            mean: 0.0,
            n_min: 0,
            mean_min: f64::INFINITY,
            updates: 0,
        }
    }
}

impl Hddm {
    /// Fresh HDDM_A detector.
    pub fn new() -> Self {
        Self::default()
    }

    fn bound(n: u64, delta: f64) -> f64 {
        if n == 0 {
            return f64::INFINITY;
        }
        (0.5 / n as f64 * (1.0 / delta.max(1e-16)).ln()).sqrt()
    }

    /// Update with a 0/1 error indicator.
    pub fn update(&mut self, x: f64, session: &Session) -> Result<Qualified<DriftDecision>> {
        let mut ctx = FitCtx::with_session(session.child("update"));
        let e = if x > 0.5 { 1.0 } else { 0.0 };
        self.n += 1;
        self.mean += (e - self.mean) / self.n as f64;
        let eps = Self::bound(self.n, self.delta);
        let eps_min = Self::bound(self.n_min.max(1), self.delta);
        if self.mean + eps < self.mean_min + eps_min {
            self.mean_min = self.mean;
            self.n_min = self.n;
        }
        let stat = self.mean - self.mean_min;
        let decision = if self.n >= 30 && self.mean - eps > self.mean_min + eps_min {
            ctx.push(
                Issue::builder(IssueCode::ConceptDriftDetected)
                    .message(format!(
                        "HDDM_A drift: μ-ε={:.4} > μ_min+ε_min",
                        self.mean - eps
                    ))
                    .metric("drift_statistic", stat)
                    .build(),
            );
            self.n = 0;
            self.mean = 0.0;
            self.n_min = 0;
            self.mean_min = f64::INFINITY;
            DriftDecision::Drift { statistic: stat }
        } else if self.n >= 30 && self.mean - 0.5 * eps > self.mean_min + 0.5 * eps_min {
            DriftDecision::Warning { statistic: stat }
        } else {
            DriftDecision::Stable
        };
        ctx.finish(decision)
    }
}

impl PartialFit for Hddm {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let before = self.mean;
        let mut fired = 0.0;
        for i in 0..x.nrows() {
            match self.update(x.get(i, 0), &session.child("hddm")) {
                Ok(q) => {
                    if matches!(q.value, DriftDecision::Drift { .. }) {
                        fired += 1.0;
                    }
                }
                Err(e) => {
                    if e.primary().code == IssueCode::ConceptDriftDetected {
                        fired += 1.0;
                    }
                }
            }
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n);
        q.effective_sample_size = self.n as f64;
        q.information_gain = Some(fired);
        q.drift_statistic = Some(self.mean);
        q.still_identified = self.n >= 30;
        q.warmup = self.n < 30;
        q.explanation = format!(
            "HDDM_A μ {before:.4}→{:.4}, n={}, drift_cuts={fired}",
            self.mean, self.n
        );
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "error-rate mean and Hoeffding bound",
                "HDDM_A comparison to the historical minimum",
                format!("μ={before:.4}"),
                format!("μ={:.4} n={}", self.mean, self.n),
            ),
        )
    }
}

/// Approximate large-margin algorithm (Gentile 2001; river `ALMA`).
#[derive(Clone, Debug)]
pub struct Alma {
    /// Margin parameter \(\alpha \in (0,1]\).
    pub alpha: f64,
    /// Aggressiveness \(C\).
    pub c: f64,
    /// Exponent \(p\) on the decaying rate (usually 2).
    pub p: f64,
    coef: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for Alma {
    fn default() -> Self {
        Self {
            alpha: 0.9,
            c: 1.0,
            p: 2.0,
            coef: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl Alma {
    /// Default ALMA.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current weights (unit-norm after each update).
    pub fn coef(&self) -> &Vector {
        &self.coef
    }
}

impl PartialFit for Alma {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(
                Issue::builder(IssueCode::MissingTarget)
                    .message("ALMA.partial_fit requires y")
                    .build(),
            );
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        if !self.initialized {
            self.coef = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if self.coef.len() != x.ncols() {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message("ALMA feature count changed")
                    .build(),
            );
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.coef.clone();
        let mut n_hit = 0.0;
        let mut loss_before = 0.0;
        for i in 0..x.nrows() {
            let yi = if y[i] >= 0.5 { 1.0 } else { -1.0 };
            let mut pred = 0.0;
            for j in 0..x.ncols() {
                pred += self.coef[j] * x.get(i, j);
            }
            let margin = yi * pred;
            let b = self.alpha * (1.0 + self.c * (self.n_seen as f64 + 1.0).powf(-1.0 / self.p));
            loss_before += (b - margin).max(0.0);
            if margin <= (1.0 - self.alpha) * b.max(1e-12) {
                n_hit += 1.0;
                let eta = self.c / (self.n_seen as f64 + 1.0).sqrt();
                for j in 0..x.ncols() {
                    self.coef[j] += eta * yi * x.get(i, j);
                }
                let nrm = self.coef.norm().max(1e-15);
                for j in 0..x.ncols() {
                    self.coef[j] /= nrm;
                }
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        let delta = self.coef.sub(&before);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.norm());
        q.information_gain = Some(n_hit);
        q.loss_before = Some(loss_before);
        q.still_identified = self.n_seen as usize > x.ncols();
        q.warmup = self.n_seen < 5;
        q.explanation = format!(
            "ALMA: {n_hit} of {} rows inside the large-margin region, ||Δw||={:.4e}",
            x.nrows(),
            delta.norm()
        );
        if n_hit == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UpdateWithZeroInformation)
                    .incremental(q.clone())
                    .message("ALMA left every row outside the update region")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "unit-norm ALMA weights",
                "large-margin multiplicative schedule of Gentile (2001)",
                format!("margin_loss={loss_before:.6e}"),
                format!("updates={n_hit}"),
            ),
        )
    }
}

/// Early drift detection (Baena-García; river `EDDM`) on a 0/1 error stream.
#[derive(Clone, Debug)]
pub struct Eddm {
    n: u64,
    last_error_at: Option<u64>,
    mean_d: f64,
    m2_d: f64,
    n_err: u64,
    max_mean2s: f64,
    updates: u64,
}

impl Default for Eddm {
    fn default() -> Self {
        Self {
            n: 0,
            last_error_at: None,
            mean_d: 0.0,
            m2_d: 0.0,
            n_err: 0,
            max_mean2s: 0.0,
            updates: 0,
        }
    }
}

impl Eddm {
    /// Fresh EDDM detector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with a 0/1 error indicator.
    pub fn update(&mut self, x: f64, session: &Session) -> Result<Qualified<DriftDecision>> {
        let mut ctx = FitCtx::with_session(session.child("update"));
        let e = if x > 0.5 { 1.0 } else { 0.0 };
        self.n += 1;
        if e > 0.5 {
            if let Some(prev) = self.last_error_at {
                let d = (self.n - prev) as f64;
                self.n_err += 1;
                let delta = d - self.mean_d;
                self.mean_d += delta / self.n_err as f64;
                self.m2_d += delta * (d - self.mean_d);
            }
            self.last_error_at = Some(self.n);
        }
        let sd = if self.n_err > 1 {
            (self.m2_d / (self.n_err - 1) as f64).max(0.0).sqrt()
        } else {
            0.0
        };
        let stat = self.mean_d + 2.0 * sd;
        if stat > self.max_mean2s {
            self.max_mean2s = stat;
        }
        let decision = if self.n_err >= 30 && self.max_mean2s > 0.0 && stat < 0.90 * self.max_mean2s
        {
            ctx.push(
                Issue::builder(IssueCode::ConceptDriftDetected)
                    .message(format!("EDDM drift: mean+2s={stat:.4} < 0.9 max"))
                    .metric("drift_statistic", stat)
                    .build(),
            );
            self.n = 0;
            self.last_error_at = None;
            self.mean_d = 0.0;
            self.m2_d = 0.0;
            self.n_err = 0;
            self.max_mean2s = 0.0;
            DriftDecision::Drift { statistic: stat }
        } else if self.n_err >= 30 && self.max_mean2s > 0.0 && stat < 0.95 * self.max_mean2s {
            DriftDecision::Warning { statistic: stat }
        } else {
            DriftDecision::Stable
        };
        ctx.finish(decision)
    }
}

impl PartialFit for Eddm {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let before = self.mean_d;
        let mut fired = 0.0;
        for i in 0..x.nrows() {
            match self.update(x.get(i, 0), &session.child("eddm")) {
                Ok(q) => {
                    if matches!(q.value, DriftDecision::Drift { .. }) {
                        fired += 1.0;
                    }
                }
                Err(e) => {
                    if e.primary().code == IssueCode::ConceptDriftDetected {
                        fired += 1.0;
                    }
                }
            }
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n);
        q.effective_sample_size = self.n as f64;
        q.information_gain = Some(fired);
        q.drift_statistic = Some(self.mean_d);
        q.still_identified = self.n_err >= 30;
        q.warmup = self.n_err < 30;
        q.explanation = format!(
            "EDDM mean-distance {before:.4}→{:.4}, errors={}, drift_cuts={fired}",
            self.mean_d, self.n_err
        );
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "inter-error distance mean and std",
                "EDDM comparison to the historical maximum of mean+2s",
                format!("d̄={before:.4}"),
                format!("d̄={:.4} n_err={}", self.mean_d, self.n_err),
            ),
        )
    }
}

/// Hoeffding drift detection with an EWMA mean (river `HDDM_W`).
#[derive(Clone, Debug)]
pub struct HddmW {
    /// Confidence \(\delta\).
    pub delta: f64,
    /// EWMA weight \(\lambda\).
    pub lambda: f64,
    n: u64,
    mean: f64,
    mean_min: f64,
    n_min: u64,
    updates: u64,
}

impl Default for HddmW {
    fn default() -> Self {
        Self {
            delta: 0.002,
            lambda: 0.05,
            n: 0,
            mean: 0.0,
            mean_min: f64::INFINITY,
            n_min: 0,
            updates: 0,
        }
    }
}

impl HddmW {
    /// Fresh HDDM_W detector.
    pub fn new() -> Self {
        Self::default()
    }

    fn bound(n: u64, lambda: f64, delta: f64) -> f64 {
        if n == 0 {
            return f64::INFINITY;
        }
        let decay = 1.0 - (1.0 - lambda).powi(n as i32);
        let eff = (lambda / decay.max(1e-18)).min(1.0);
        (0.5 * eff * (1.0 / delta.max(1e-16)).ln()).sqrt()
    }

    /// Update with a 0/1 error indicator.
    pub fn update(&mut self, x: f64, session: &Session) -> Result<Qualified<DriftDecision>> {
        let mut ctx = FitCtx::with_session(session.child("update"));
        let e = if x > 0.5 { 1.0 } else { 0.0 };
        self.n += 1;
        let l = self.lambda.clamp(1e-6, 1.0);
        if self.n == 1 {
            self.mean = e;
        } else {
            self.mean = l * e + (1.0 - l) * self.mean;
        }
        let eps = Self::bound(self.n, l, self.delta);
        let eps_min = Self::bound(self.n_min.max(1), l, self.delta);
        if self.mean + eps < self.mean_min + eps_min {
            self.mean_min = self.mean;
            self.n_min = self.n;
        }
        let stat = self.mean - self.mean_min;
        let decision = if self.n >= 30 && self.mean - eps > self.mean_min + eps_min {
            ctx.push(
                Issue::builder(IssueCode::ConceptDriftDetected)
                    .message(format!(
                        "HDDM_W drift: μ-ε={:.4} > μ_min+ε_min",
                        self.mean - eps
                    ))
                    .metric("drift_statistic", stat)
                    .build(),
            );
            self.n = 0;
            self.mean = 0.0;
            self.n_min = 0;
            self.mean_min = f64::INFINITY;
            DriftDecision::Drift { statistic: stat }
        } else if self.n >= 30 && self.mean - 0.5 * eps > self.mean_min + 0.5 * eps_min {
            DriftDecision::Warning { statistic: stat }
        } else {
            DriftDecision::Stable
        };
        ctx.finish(decision)
    }
}

impl PartialFit for HddmW {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let before = self.mean;
        let mut fired = 0.0;
        for i in 0..x.nrows() {
            match self.update(x.get(i, 0), &session.child("hddmw")) {
                Ok(q) => {
                    if matches!(q.value, DriftDecision::Drift { .. }) {
                        fired += 1.0;
                    }
                }
                Err(e) => {
                    if e.primary().code == IssueCode::ConceptDriftDetected {
                        fired += 1.0;
                    }
                }
            }
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n);
        q.effective_sample_size = self.n as f64;
        q.information_gain = Some(fired);
        q.drift_statistic = Some(self.mean);
        q.still_identified = self.n >= 30;
        q.warmup = self.n < 30;
        q.explanation = format!(
            "HDDM_W μ {before:.4}→{:.4}, n={}, drift_cuts={fired}",
            self.mean, self.n
        );
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "EWMA error-rate and a weighted Hoeffding bound",
                "HDDM_W comparison to the historical minimum",
                format!("μ={before:.4}"),
                format!("μ={:.4} n={}", self.mean, self.n),
            ),
        )
    }
}

/// Leveraging bagging: Poisson(\(w\)) online bagging of Hoeffding trees.
#[derive(Clone, Debug)]
pub struct LeveragingBagging {
    /// Number of trees.
    pub n_estimators: usize,
    /// Poisson mean of the leverage weights.
    pub w: f64,
    trees: Vec<HoeffdingTree>,
    n_seen: u64,
    updates: u64,
    initialized: bool,
    rng: Rng,
}

impl Default for LeveragingBagging {
    fn default() -> Self {
        Self {
            n_estimators: 3,
            w: 6.0,
            trees: Vec::new(),
            n_seen: 0,
            updates: 0,
            initialized: false,
            rng: Rng::new(11),
        }
    }
}

impl LeveragingBagging {
    /// Forest with `n_estimators` trees.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators: n_estimators.max(1),
            ..Self::default()
        }
    }

    fn ensure(&mut self, p: usize) {
        if self.initialized {
            return;
        }
        self.trees = (0..self.n_estimators)
            .map(|_| {
                let mut t = HoeffdingTree::new();
                t.min_samples = 15;
                t.n_features = p;
                t.root = HtNode::Leaf(HtLeaf::new(0, p));
                t.initialized = true;
                t
            })
            .collect();
        self.initialized = true;
    }
}

impl PartialFit for LeveragingBagging {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        if !self.initialized {
            self.ensure(x.ncols());
        }
        let mut n_updates = 0u64;
        for i in 0..x.nrows().min(y.len()) {
            let xi = Matrix::from_fn(1, x.ncols(), |_, j| x.get(i, j));
            let yi = Vector::from_slice(&[y[i]]);
            for t in 0..self.trees.len() {
                let k = self.rng.poisson(self.w.max(0.0));
                n_updates += k;
                for _ in 0..k {
                    let _ = self.trees[t].partial_fit(&xi, Some(&yi), &session.child("tree"));
                }
            }
        }
        self.n_seen += x.nrows().min(y.len()) as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(n_updates as f64);
        q.information_gain = Some(n_updates as f64);
        q.still_identified = self.n_seen >= 15;
        q.warmup = self.n_seen < 15;
        q.explanation = format!(
            "LeveragingBagging: {} trees, {n_updates} Poisson-weighted Hoeffding updates",
            self.trees.len()
        );
        if q.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("leveraging bagging still warming up")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                format!("{n_updates} Poisson(w={}) tree updates", self.w),
                "each row is replayed k~Poisson(w) times into every Hoeffding tree",
                "pre-batch forest",
                "post-batch forest",
            ),
        )
    }
}

impl Predict for LeveragingBagging {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut c0 = 0usize;
            let mut c1 = 0usize;
            for t in &self.trees {
                if t.predict_one(x, i) >= 0.5 {
                    c1 += 1;
                } else {
                    c0 += 1;
                }
            }
            if c1 >= c0 {
                1.0
            } else {
                0.0
            }
        }));
        ctx.finish(out)
    }
}

impl Predict for Alma {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut s = 0.0;
            for j in 0..x.ncols().min(self.coef.len()) {
                s += self.coef[j] * x.get(i, j);
            }
            if s >= 0.0 {
                1.0
            } else {
                0.0
            }
        }));
        ctx.finish(out)
    }
}

/// Online SNARIMAX: RLS on AR lags, MA residual lags, and exogenous columns.
#[derive(Clone, Debug)]
pub struct Snarimax {
    /// Autoregressive order.
    pub p: usize,
    /// Moving-average order.
    pub q: usize,
    inner: LinearRegression,
    y_hist: Vec<f64>,
    e_hist: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl Default for Snarimax {
    fn default() -> Self {
        Self {
            p: 1,
            q: 0,
            inner: LinearRegression::new(1.0),
            y_hist: Vec::new(),
            e_hist: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl Snarimax {
    /// SNARIMAX with AR(`p`) and MA(`q`).
    pub fn new(p: usize, q: usize) -> Self {
        Self {
            p,
            q,
            ..Self::default()
        }
    }

    fn feat_row(&self, x: &Matrix, i: usize) -> Vector {
        let dim = self.p + self.q + x.ncols();
        let mut z = Vector::zeros(dim);
        for k in 0..self.p {
            let idx = self.y_hist.len().saturating_sub(k + 1);
            z[k] = if self.y_hist.len() > k {
                self.y_hist[idx]
            } else {
                0.0
            };
        }
        for k in 0..self.q {
            let idx = self.e_hist.len().saturating_sub(k + 1);
            z[self.p + k] = if self.e_hist.len() > k {
                self.e_hist[idx]
            } else {
                0.0
            };
        }
        for j in 0..x.ncols() {
            z[self.p + self.q + j] = x.get(i, j);
        }
        z
    }
}

impl PartialFit for Snarimax {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        inspect_online_xy(&mut ctx, x, Some(y));
        let need = self.p.max(self.q);
        let warmup = self.y_hist.len() < need;
        let mut info = 0.0;
        let mut last_err = 0.0;
        for i in 0..x.nrows().min(y.len()) {
            let z = self.feat_row(x, i);
            let row = Matrix::from_fn(1, z.len(), |_, j| z[j]);
            let yi = Vector::from_slice(&[y[i]]);
            match self
                .inner
                .partial_fit(&row, Some(&yi), &session.child("rls"))
            {
                Ok(q) => {
                    info += q.value.quality.information_gain.unwrap_or(0.0);
                }
                Err(e) => ctx.push(e.primary),
            }
            let pred = if self.inner.coef().len() + 1 == z.len() + 1
                || self.inner.intercept().is_finite()
            {
                let mut s = self.inner.intercept();
                let c = self.inner.coef();
                for j in 0..c.len().min(z.len()) {
                    s += c[j] * z[j];
                }
                s
            } else {
                0.0
            };
            last_err = y[i] - pred;
            self.y_hist.push(y[i]);
            self.e_hist.push(last_err);
            if self.y_hist.len() > 64 {
                self.y_hist.remove(0);
            }
            if self.e_hist.len() > 64 {
                self.e_hist.remove(0);
            }
        }
        self.n_seen += x.nrows().min(y.len()) as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.information_gain = Some(info.abs() + last_err.abs());
        q.still_identified = self.n_seen > (self.p + self.q + x.ncols()) as u64;
        q.warmup = warmup || self.n_seen < 5;
        q.explanation = format!(
            "SNARIMAX AR{} MA{}: last residual={last_err:.4e}, n={}",
            self.p, self.q, self.n_seen
        );
        if q.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("SNARIMAX lag buffers are still filling")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "RLS update of AR/MA/exog coefficients",
                "Hannan–Rissanen-style online regression on lags and x_t",
                format!("n={}", self.n_seen.saturating_sub(x.nrows() as u64)),
                format!("n={} e={last_err:.4e}", self.n_seen),
            ),
        )
    }
}

/// DenStream-style decaying micro-clusters (river `cluster.DenStream`).
#[derive(Clone, Debug)]
pub struct DenStream {
    /// Merge radius.
    pub eps: f64,
    /// Per-step weight decay.
    pub lambda: f64,
    micros: Vec<(Vector, f64)>,
    n_seen: u64,
    updates: u64,
}

impl Default for DenStream {
    fn default() -> Self {
        Self {
            eps: 1.0,
            lambda: 0.99,
            micros: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl DenStream {
    /// DenStream with merge radius `eps`.
    pub fn new(eps: f64) -> Self {
        Self {
            eps,
            ..Self::default()
        }
    }

    /// Current micro-cluster count.
    pub fn n_micro(&self) -> usize {
        self.micros.len()
    }
}

impl PartialFit for DenStream {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if !(self.eps.is_finite() && self.eps > 0.0) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("DenStream eps={} is not positive", self.eps))
                    .build(),
            );
        }
        let lam = self.lambda.clamp(1e-6, 1.0);
        let mut merged = 0u64;
        let mut spawned = 0u64;
        for i in 0..x.nrows() {
            for mc in self.micros.iter_mut() {
                mc.1 *= lam;
            }
            self.micros.retain(|mc| mc.1 >= 1e-3);
            let row = x.row(i);
            let mut best = None;
            let mut best_d = f64::INFINITY;
            for (c, (ctr, _)) in self.micros.iter().enumerate() {
                let mut d = 0.0;
                for j in 0..row.len().min(ctr.len()) {
                    let z = row[j] - ctr[j];
                    d += z * z;
                }
                d = d.sqrt();
                if d < best_d {
                    best_d = d;
                    best = Some(c);
                }
            }
            if let Some(c) = best {
                if best_d <= self.eps {
                    let w = self.micros[c].1;
                    let nw = w + 1.0;
                    for j in 0..self.micros[c].0.len().min(row.len()) {
                        self.micros[c].0[j] = (w * self.micros[c].0[j] + row[j]) / nw;
                    }
                    self.micros[c].1 = nw;
                    merged += 1;
                    continue;
                }
            }
            self.micros.push((row, 1.0));
            spawned += 1;
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        if self.micros.is_empty() && x.nrows() > 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyCluster)
                    .message("DenStream retained no micro-clusters")
                    .build(),
            );
        }
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.micros.iter().map(|m| m.1).sum();
        q.information_gain = Some(merged as f64 + spawned as f64);
        q.still_identified = !self.micros.is_empty();
        q.warmup = self.n_seen < 4;
        q.explanation = format!(
            "DenStream: merged={merged} spawned={spawned} micros={}",
            self.micros.len()
        );
        if q.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("DenStream has seen fewer than 4 points")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "micro-cluster assignment / spawn",
                "decayed CF update if within eps, else a new micro-cluster",
                format!("micros before batch"),
                format!("micros={}", self.micros.len()),
            ),
        )
    }
}

/// Hoeffding Adaptive Tree: a VFDT plus ADWIN on 0/1 error (Bifet & Gavaldà).
#[derive(Clone, Debug)]
pub struct HoeffdingAdaptiveTree {
    tree: HoeffdingTree,
    detector: Adwin,
    n_seen: u64,
    updates: u64,
    resets: u64,
}

impl Default for HoeffdingAdaptiveTree {
    fn default() -> Self {
        Self {
            tree: HoeffdingTree::new(),
            detector: Adwin::new(0.002),
            n_seen: 0,
            updates: 0,
            resets: 0,
        }
    }
}

impl HoeffdingAdaptiveTree {
    /// Default HAT.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for HoeffdingAdaptiveTree {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let q = self.tree.partial_fit(x, y, &session.child("tree"))?;
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            return finish_explain(ctx, q.value);
        };
        inspect_online_xy(&mut ctx, x, Some(y));
        let mut drift = 0u64;
        for i in 0..x.nrows().min(y.len()) {
            let pred = self.tree.predict_one(x, i);
            let err = if (pred - as01(y[i])).abs() > 0.5 {
                1.0
            } else {
                0.0
            };
            match self.detector.update(err, &session.child("adwin")) {
                Ok(d) => {
                    if matches!(d.value, DriftDecision::Drift { .. }) {
                        self.tree.reset();
                        self.detector.reset();
                        self.resets += 1;
                        drift += 1;
                        ctx.push(
                            Issue::builder(IssueCode::ConceptDriftDetected)
                                .message("HAT reset the Hoeffding tree after an ADWIN cut")
                                .build(),
                        );
                    }
                }
                Err(_) => {
                    self.tree.reset();
                    self.detector.reset();
                    self.resets += 1;
                    drift += 1;
                    ctx.push(
                        Issue::builder(IssueCode::ConceptDriftDetected)
                            .message("HAT reset after ADWIN reported drift as an error")
                            .build(),
                    );
                }
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut iq = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        iq.effective_sample_size = self.n_seen as f64;
        iq.parameter_delta_norm =
            Some(drift as f64 + q.value.quality.parameter_delta_norm.unwrap_or(0.0));
        iq.information_gain = Some(drift as f64 + q.value.quality.information_gain.unwrap_or(0.0));
        iq.still_identified = self.n_seen >= 15;
        iq.warmup = self.n_seen < 15;
        iq.explanation = format!(
            "HAT: VFDT update then ADWIN on 0/1 error; {drift} resets (total {})",
            self.resets
        );
        if iq.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(iq.clone())
                    .message("HAT is still warming up")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                iq,
                if drift > 0 {
                    "tree reset after drift"
                } else {
                    "leaf stats updated; no drift"
                },
                "Hoeffding split test plus ADWIN on instantaneous error",
                "pre-batch HAT",
                "post-batch HAT",
            ),
        )
    }
}

impl Predict for HoeffdingAdaptiveTree {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.tree.predict(x, session)
    }
}

/// Hoeffding adaptive tree regressor (river `HoeffdingAdaptiveTreeRegressor`).
///
/// ADWIN watches absolute residual; a cut resets the tree.
#[derive(Clone, Debug)]
pub struct HoeffdingAdaptiveTreeRegressor {
    tree: HoeffdingRegressor,
    detector: Adwin,
    n_seen: u64,
    updates: u64,
    resets: u64,
}

impl Default for HoeffdingAdaptiveTreeRegressor {
    fn default() -> Self {
        Self {
            tree: HoeffdingRegressor::new(),
            detector: Adwin::new(0.002),
            n_seen: 0,
            updates: 0,
            resets: 0,
        }
    }
}

impl HoeffdingAdaptiveTreeRegressor {
    /// Default HAT regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for HoeffdingAdaptiveTreeRegressor {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let q = self.tree.partial_fit(x, y, &session.child("tree"))?;
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            return finish_explain(ctx, q.value);
        };
        inspect_online_xy(&mut ctx, x, Some(y));
        let mut drift = 0u64;
        for i in 0..x.nrows().min(y.len()) {
            let pred = self.tree.predict_one(x, i);
            let err = (pred - y[i]).abs();
            match self.detector.update(err, &session.child("adwin")) {
                Ok(d) => {
                    if matches!(d.value, DriftDecision::Drift { .. }) {
                        self.tree.reset();
                        self.detector.reset();
                        self.resets += 1;
                        drift += 1;
                        ctx.push(
                            Issue::builder(IssueCode::ConceptDriftDetected)
                                .message("HAT-regressor reset the tree after an ADWIN cut")
                                .build(),
                        );
                    }
                }
                Err(_) => {
                    self.tree.reset();
                    self.detector.reset();
                    self.resets += 1;
                    drift += 1;
                    ctx.push(
                        Issue::builder(IssueCode::ConceptDriftDetected)
                            .message("HAT-regressor reset after ADWIN reported drift as an error")
                            .build(),
                    );
                }
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut iq = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        iq.effective_sample_size = self.n_seen as f64;
        iq.parameter_delta_norm =
            Some(drift as f64 + q.value.quality.parameter_delta_norm.unwrap_or(0.0));
        iq.information_gain = Some(drift as f64 + q.value.quality.information_gain.unwrap_or(0.0));
        iq.still_identified = self.n_seen >= 15;
        iq.warmup = self.n_seen < 15;
        iq.explanation = format!(
            "HAT-regressor: VFDT update then ADWIN on |residual|; {drift} resets (total {})",
            self.resets
        );
        if iq.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(iq.clone())
                    .message("HAT-regressor is still warming up")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                iq,
                if drift > 0 {
                    "tree reset after drift"
                } else {
                    "leaf means updated; no drift"
                },
                "Hoeffding variance-reduction plus ADWIN on absolute residual",
                "pre-batch HAT-regressor",
                "post-batch HAT-regressor",
            ),
        )
    }
}

impl Predict for HoeffdingAdaptiveTreeRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.tree.predict(x, session)
    }
}

/// Streaming Random Patches: Hoeffding trees on random feature subsets.
#[derive(Clone, Debug)]
pub struct StreamingRandomPatches {
    /// Number of trees.
    pub n_estimators: usize,
    trees: Vec<HoeffdingTree>,
    masks: Vec<Vec<usize>>,
    n_seen: u64,
    updates: u64,
    initialized: bool,
    rng: Rng,
}

impl Default for StreamingRandomPatches {
    fn default() -> Self {
        Self {
            n_estimators: 3,
            trees: Vec::new(),
            masks: Vec::new(),
            n_seen: 0,
            updates: 0,
            initialized: false,
            rng: Rng::new(19),
        }
    }
}

impl StreamingRandomPatches {
    /// Forest with `n_estimators` patch trees.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators: n_estimators.max(1),
            ..Self::default()
        }
    }

    fn ensure(&mut self, p: usize) {
        if self.initialized {
            return;
        }
        let k = ((p as f64).sqrt().ceil() as usize).clamp(1, p.max(1));
        self.masks = (0..self.n_estimators)
            .map(|_| {
                let mut idx: Vec<usize> = (0..p).collect();
                self.rng.shuffle(&mut idx);
                idx.truncate(k);
                if idx.is_empty() {
                    idx.push(0);
                }
                idx
            })
            .collect();
        self.trees = (0..self.n_estimators)
            .map(|_| {
                let mut t = HoeffdingTree::new();
                t.min_samples = 15;
                t.n_features = k;
                t.root = HtNode::Leaf(HtLeaf::new(0, k));
                t.initialized = true;
                t
            })
            .collect();
        self.initialized = true;
    }
}

impl PartialFit for StreamingRandomPatches {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        inspect_online_xy(&mut ctx, x, Some(y));
        self.ensure(x.ncols());
        let mut n_up = 0u64;
        for t in 0..self.trees.len() {
            let mask = &self.masks[t];
            let xs = Matrix::from_fn(x.nrows(), mask.len(), |i, j| {
                let c = mask[j];
                if c < x.ncols() {
                    x.get(i, c)
                } else {
                    0.0
                }
            });
            let _ = self.trees[t].partial_fit(&xs, Some(y), &session.child("patch"));
            n_up += 1;
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(n_up as f64);
        q.information_gain = Some(n_up as f64);
        q.still_identified = self.n_seen >= 15;
        q.warmup = self.n_seen < 15;
        q.explanation = format!(
            "SRP: {} patch trees, each on {} random features",
            self.trees.len(),
            self.masks.first().map(|m| m.len()).unwrap_or(0)
        );
        if q.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("SRP is still warming up")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                format!("{n_up} patch-tree updates"),
                "each Hoeffding tree sees a random feature subspace",
                "pre-batch SRP",
                "post-batch SRP",
            ),
        )
    }
}

impl Predict for StreamingRandomPatches {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut c0 = 0usize;
            let mut c1 = 0usize;
            for (t, mask) in self.masks.iter().enumerate() {
                let xs = Matrix::from_fn(1, mask.len(), |_, j| {
                    let c = mask[j];
                    if c < x.ncols() {
                        x.get(i, c)
                    } else {
                        0.0
                    }
                });
                if self.trees[t].predict_one(&xs, 0) >= 0.5 {
                    c1 += 1;
                } else {
                    c0 += 1;
                }
            }
            if c1 >= c0 {
                1.0
            } else {
                0.0
            }
        }));
        ctx.finish(out)
    }
}

/// Funk / biased SVD matrix factorization (river `reco.FunkMF`).
///
/// `X` has two columns: user id and item id (rounded). `y` is the rating.
#[derive(Clone, Debug)]
pub struct FunkMf {
    /// Latent rank.
    pub n_factors: usize,
    /// SGD step.
    pub learning_rate: f64,
    /// ℓ2 on factors and biases.
    pub l2: f64,
    mu: f64,
    bu: Vec<f64>,
    bi: Vec<f64>,
    pu: Vec<Vec<f64>>,
    qi: Vec<Vec<f64>>,
    n_seen: u64,
    updates: u64,
}

impl Default for FunkMf {
    fn default() -> Self {
        Self {
            n_factors: 4,
            learning_rate: 0.05,
            l2: 0.01,
            mu: 0.0,
            bu: Vec::new(),
            bi: Vec::new(),
            pu: Vec::new(),
            qi: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl FunkMf {
    /// FunkMF with `k` factors.
    pub fn new(n_factors: usize) -> Self {
        Self {
            n_factors: n_factors.max(1),
            ..Self::default()
        }
    }

    fn grow(&mut self, u: usize, i: usize) {
        let k = self.n_factors.max(1);
        while self.bu.len() <= u {
            self.bu.push(0.0);
            self.pu.push(vec![0.01; k]);
        }
        while self.bi.len() <= i {
            self.bi.push(0.0);
            self.qi.push(vec![0.01; k]);
        }
    }

    fn pred(&self, u: usize, i: usize) -> f64 {
        let mut s = self.mu;
        if u < self.bu.len() {
            s += self.bu[u];
        }
        if i < self.bi.len() {
            s += self.bi[i];
        }
        if u < self.pu.len() && i < self.qi.len() {
            for f in 0..self.n_factors.min(self.pu[u].len()).min(self.qi[i].len()) {
                s += self.pu[u][f] * self.qi[i][f];
            }
        }
        s
    }
}

impl PartialFit for FunkMf {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        inspect_online_xy(&mut ctx, x, Some(y));
        if x.ncols() < 2 {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("FunkMF needs two columns (user, item)")
                    .build(),
            );
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "need user/item"),
            );
        }
        let lr = self.learning_rate.max(1e-6);
        let l2 = self.l2.max(0.0);
        let mut sse = 0.0;
        let mut dsum = 0.0;
        for r in 0..x.nrows().min(y.len()) {
            let u = x.get(r, 0).max(0.0).round() as usize;
            let i = x.get(r, 1).max(0.0).round() as usize;
            self.grow(u, i);
            let pred = self.pred(u, i);
            let e = y[r] - pred;
            sse += e * e;
            self.mu += lr * e;
            self.bu[u] += lr * (e - l2 * self.bu[u]);
            self.bi[i] += lr * (e - l2 * self.bi[i]);
            for f in 0..self.n_factors {
                let pu = self.pu[u][f];
                let qi = self.qi[i][f];
                self.pu[u][f] += lr * (e * qi - l2 * pu);
                self.qi[i][f] += lr * (e * pu - l2 * qi);
                dsum += (self.pu[u][f] - pu).abs() + (self.qi[i][f] - qi).abs();
            }
        }
        self.n_seen += x.nrows().min(y.len()) as u64;
        self.updates += 1;
        let warmup = self.n_seen < 8;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(dsum);
        q.loss_before = Some(sse);
        q.loss_after = Some(sse);
        q.information_gain = Some(dsum);
        q.still_identified = !warmup;
        q.warmup = warmup;
        q.explanation = format!(
            "FunkMF SGD: ||ΔP,Q||₁={dsum:.4e}, SSE={sse:.4e}, users={}, items={}",
            self.bu.len(),
            self.bi.len()
        );
        if warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("FunkMF is still warming up")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                format!("SGD on {} ratings", x.nrows()),
                "biased SVD step: μ, b_u, b_i, p_u, q_i",
                "pre-batch factors",
                format!("μ={:.4e} SSE={sse:.4e}", self.mu),
            ),
        )
    }
}

impl Predict for FunkMf {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() < 2 {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("FunkMF predict needs user/item columns")
                    .build(),
            );
            return ctx.finish(Vector::filled(x.nrows(), self.mu));
        }
        let y = Vector::from_iter((0..x.nrows()).map(|r| {
            let u = x.get(r, 0).max(0.0).round() as usize;
            let i = x.get(r, 1).max(0.0).round() as usize;
            self.pred(u, i)
        }));
        ctx.finish(y)
    }
}

/// CluStream-style q-micro-cluster summary (Aggarwal et al.).
#[derive(Clone, Debug)]
pub struct CluStream {
    /// Maximum number of micro-clusters.
    pub q: usize,
    /// Assign-to-nearest radius; beyond this a new micro is spawned.
    pub radius: f64,
    micros: Vec<(f64, Vector, f64)>,
    n_seen: u64,
    updates: u64,
}

impl Default for CluStream {
    fn default() -> Self {
        Self {
            q: 8,
            radius: 2.0,
            micros: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl CluStream {
    /// CluStream with at most `q` micro-clusters.
    pub fn new(q: usize) -> Self {
        Self {
            q: q.max(1),
            ..Self::default()
        }
    }

    /// Current micro-cluster count.
    pub fn n_micro(&self) -> usize {
        self.micros.len()
    }

    fn center(n: f64, ls: &Vector) -> Vector {
        if n <= 0.0 {
            Vector::zeros(ls.len())
        } else {
            ls.scale(1.0 / n)
        }
    }

    fn merge_nearest(&mut self) {
        if self.micros.len() < 2 {
            return;
        }
        let mut best = (0usize, 1usize, f64::INFINITY);
        for a in 0..self.micros.len() {
            let ca = Self::center(self.micros[a].0, &self.micros[a].1);
            for b in (a + 1)..self.micros.len() {
                let cb = Self::center(self.micros[b].0, &self.micros[b].1);
                let d = ca.sub(&cb).norm();
                if d < best.2 {
                    best = (a, b, d);
                }
            }
        }
        let (a, b, _) = best;
        let (n2, ls2, ss2) = self.micros.remove(b.max(a));
        let (n1, ls1, ss1) = self.micros.remove(a.min(b));
        self.micros.push((n1 + n2, ls1.add(&ls2), ss1 + ss2));
    }
}

impl PartialFit for CluStream {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if !(self.radius.is_finite() && self.radius > 0.0) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!("CluStream radius={} is not positive", self.radius))
                    .build(),
            );
        }
        let mut spawned = 0u64;
        let mut merged = 0u64;
        for i in 0..x.nrows() {
            let row = x.row(i);
            let mut best = None;
            let mut bd = f64::INFINITY;
            for (t, (n, ls, _)) in self.micros.iter().enumerate() {
                let c = Self::center(*n, ls);
                if c.len() != row.len() {
                    continue;
                }
                let d = c.sub(&row).norm();
                if d < bd {
                    bd = d;
                    best = Some(t);
                }
            }
            if let Some(t) = best {
                if bd <= self.radius {
                    let (n, ls, ss) = &mut self.micros[t];
                    *n += 1.0;
                    for j in 0..ls.len().min(row.len()) {
                        ls[j] += row[j];
                    }
                    *ss += row.dot(&row);
                    continue;
                }
            }
            self.micros.push((1.0, row.clone(), row.dot(&row)));
            spawned += 1;
            while self.micros.len() > self.q.max(1) {
                self.merge_nearest();
                merged += 1;
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let warmup = self.n_seen < 4;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((spawned + merged) as f64);
        q.information_gain = Some((spawned + merged) as f64);
        q.still_identified = !warmup;
        q.warmup = warmup;
        q.explanation = format!(
            "CluStream: spawned {spawned}, merged {merged}, micros={}",
            self.micros.len()
        );
        if warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("CluStream has seen fewer than 4 points")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                format!("{spawned} new micros, {merged} merges"),
                "assign to nearest CF if within radius, else spawn and merge the closest pair",
                "pre-batch micros",
                format!("micros={}", self.micros.len()),
            ),
        )
    }
}

impl Predict for CluStream {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.micros.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::filled(x.nrows(), -1.0));
        }
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let row = x.row(i);
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for (t, (n, ls, _)) in self.micros.iter().enumerate() {
                let c = Self::center(*n, ls);
                if c.len() != row.len() {
                    continue;
                }
                let d = c.sub(&row).norm();
                if d < bd {
                    bd = d;
                    best = t;
                }
            }
            best as f64
        }));
        ctx.finish(y)
    }
}

/// Extremely Fast Decision Tree (Hoeffding AnyTime Tree) lite.
///
/// Same VFDT sufficient statistics as [`HoeffdingTree`], but splits are
/// considered earlier (`min_samples` is smaller and `tau` is larger). A
/// re-split of an existing node is not implemented — documented as a
/// compromise versus Manapragada, Webb & Salehi.
#[derive(Clone, Debug)]
pub struct ExtremelyFastDecisionTree {
    tree: HoeffdingTree,
}

impl Default for ExtremelyFastDecisionTree {
    fn default() -> Self {
        let mut tree = HoeffdingTree::new();
        tree.min_samples = 5;
        tree.tau = 0.15;
        Self { tree }
    }
}

impl ExtremelyFastDecisionTree {
    /// Eager VFDT.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for ExtremelyFastDecisionTree {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let q = self.tree.partial_fit(x, y, session)?;
        let mut ctx = FitCtx::with_session(session.child("efdt"));
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .severity(Severity::Advisory)
                .message("EFDT-lite uses an eager Hoeffding bound; it does not re-evaluate existing splits")
                .build(),
        );
        ctx.finish(q.value)
    }
}

impl Predict for ExtremelyFastDecisionTree {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.tree.predict(x, session)
    }
}

/// Biased matrix factorization: `μ + b_u + b_i` (river `reco.BiasedMF`).
#[derive(Clone, Debug)]
pub struct BiasedMf {
    /// SGD step.
    pub learning_rate: f64,
    /// ℓ2 on biases.
    pub l2: f64,
    mu: f64,
    bu: Vec<f64>,
    bi: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl Default for BiasedMf {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            l2: 0.01,
            mu: 0.0,
            bu: Vec::new(),
            bi: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl BiasedMf {
    /// Default biased MF.
    pub fn new() -> Self {
        Self::default()
    }

    fn grow(&mut self, u: usize, i: usize) {
        while self.bu.len() <= u {
            self.bu.push(0.0);
        }
        while self.bi.len() <= i {
            self.bi.push(0.0);
        }
    }

    fn pred(&self, u: usize, i: usize) -> f64 {
        let mut s = self.mu;
        if u < self.bu.len() {
            s += self.bu[u];
        }
        if i < self.bi.len() {
            s += self.bi[i];
        }
        s
    }
}

impl PartialFit for BiasedMf {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };
        inspect_online_xy(&mut ctx, x, Some(y));
        if x.ncols() < 2 {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("BiasedMF needs two columns (user, item)")
                    .build(),
            );
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "need user/item"),
            );
        }
        let lr = self.learning_rate.max(1e-6);
        let l2 = self.l2.max(0.0);
        let mut sse = 0.0;
        let mut dsum = 0.0;
        for r in 0..x.nrows().min(y.len()) {
            let u = x.get(r, 0).max(0.0).round() as usize;
            let i = x.get(r, 1).max(0.0).round() as usize;
            self.grow(u, i);
            let pred = self.pred(u, i);
            let e = y[r] - pred;
            sse += e * e;
            self.mu += lr * e;
            let bu = self.bu[u];
            let bi = self.bi[i];
            self.bu[u] += lr * (e - l2 * bu);
            self.bi[i] += lr * (e - l2 * bi);
            dsum += (self.bu[u] - bu).abs() + (self.bi[i] - bi).abs();
        }
        self.n_seen += x.nrows().min(y.len()) as u64;
        self.updates += 1;
        let warmup = self.n_seen < 8;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(dsum);
        q.still_identified = !warmup;
        q.warmup = warmup;
        q.explanation = format!(
            "BiasedMF SGD: ||Δb||₁={dsum:.4e}, SSE={sse:.4e}, users={}, items={}",
            self.bu.len(),
            self.bi.len()
        );
        if warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("BiasedMF is still warming up")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                format!("SGD on {} ratings", x.nrows()),
                "μ, b_u, b_i only (no latent factors)",
                "pre-batch biases",
                format!("μ={:.4e} SSE={sse:.4e}", self.mu),
            ),
        )
    }
}

impl Predict for BiasedMf {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() < 2 {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("BiasedMF predict needs user/item columns")
                    .build(),
            );
            return ctx.finish(Vector::filled(x.nrows(), self.mu));
        }
        let y = Vector::from_iter((0..x.nrows()).map(|r| {
            let u = x.get(r, 0).max(0.0).round() as usize;
            let i = x.get(r, 1).max(0.0).round() as usize;
            self.pred(u, i)
        }));
        ctx.finish(y)
    }
}

/// Online k-NN classifier (river `neighbors.KNNClassifier`).
///
/// Every labelled row is stored. A 1-row batch uses
/// [`inspect_online_xy`], not [`inspect_xy`].
#[derive(Clone, Debug)]
pub struct KnnClassifier {
    /// Neighbourhood size.
    pub k: usize,
    xs: Vec<Vec<f64>>,
    ys: Vec<i64>,
    n_seen: u64,
    updates: u64,
}

impl Default for KnnClassifier {
    fn default() -> Self {
        Self {
            k: 3,
            xs: Vec::new(),
            ys: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl KnnClassifier {
    /// `k`-NN memory.
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            ..Self::default()
        }
    }
}

impl PartialFit for KnnClassifier {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        if self.n_seen > 0 {
            if let Some(first) = self.xs.first() {
                if first.len() != x.ncols() {
                    ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
                    return finish_explain(
                        ctx,
                        reject_explain(
                            self.updates,
                            x.nrows(),
                            self.n_seen,
                            "feature space changed",
                        ),
                    );
                }
            }
        }
        let before = self.xs.len();
        for i in 0..x.nrows() {
            self.xs.push((0..x.ncols()).map(|j| x.get(i, j)).collect());
            self.ys.push(y[i].round() as i64);
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.xs.len() - before) as f64);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.xs.len() >= self.k;
        q.warmup = self.xs.len() < self.k;
        q.explanation = format!(
            "stored {} rows; memory={} k={}",
            x.nrows(),
            self.xs.len(),
            self.k
        );
        if q.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("online k-NN has fewer stored rows than k")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "instance memory",
                "append labelled rows for later majority vote",
                format!("n={before}"),
                format!("n={}", self.xs.len()),
            ),
        )
    }
}

impl Predict for KnnClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.xs.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let k = self.k.max(1).min(self.xs.len());
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut dist: Vec<(f64, i64)> = self
                .xs
                .iter()
                .zip(self.ys.iter())
                .map(|(z, &lab)| {
                    let mut s = 0.0;
                    for j in 0..x.ncols().min(z.len()) {
                        let d = x.get(i, j) - z[j];
                        s += d * d;
                    }
                    (s, lab)
                })
                .collect();
            dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut votes = std::collections::BTreeMap::<i64, usize>::new();
            for (_, lab) in dist.into_iter().take(k) {
                *votes.entry(lab).or_insert(0) += 1;
            }
            votes
                .into_iter()
                .max_by_key(|(_, c)| *c)
                .map(|(lab, _)| lab as f64)
                .unwrap_or(0.0)
        }));
        ctx.finish(y)
    }
}

/// Online min–max scaler (river `preprocessing.MinMaxScaler`).
#[derive(Clone, Debug)]
pub struct OnlineMinMaxScaler {
    min: Vector,
    max: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for OnlineMinMaxScaler {
    fn default() -> Self {
        Self {
            min: Vector::zeros(0),
            max: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl OnlineMinMaxScaler {
    /// Empty scaler.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for OnlineMinMaxScaler {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if !self.initialized {
            self.min = Vector::from_iter((0..x.ncols()).map(|_| f64::INFINITY));
            self.max = Vector::from_iter((0..x.ncols()).map(|_| f64::NEG_INFINITY));
            self.initialized = true;
        } else if self.min.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.min.clone();
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                if v < self.min[j] {
                    self.min[j] = v;
                }
                if v > self.max[j] {
                    self.max[j] = v;
                }
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.min.sub(&before);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.norm());
        q.information_gain = Some(delta.norm().max(1.0 / self.n_seen.max(1) as f64));
        q.still_identified = self.n_seen >= 1;
        q.warmup = self.n_seen < 2;
        q.explanation = format!("online MinMax on {} rows", x.nrows());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "running column min/max",
                "stream extrema",
                "previous extrema",
                "updated extrema",
            ),
        )
    }
}

impl Transform for OnlineMinMaxScaler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.min.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j >= p {
                return x.get(i, j);
            }
            let lo = self.min[j];
            let hi = self.max[j];
            let span = hi - lo;
            if !span.is_finite() || span.abs() <= ctx.policy.near_zero_variance {
                return 0.0;
            }
            (x.get(i, j) - lo) / span
        });
        ctx.finish(out)
    }
}

/// Online max-abs scaler (river `preprocessing.MaxAbsScaler`).
#[derive(Clone, Debug)]
pub struct OnlineMaxAbsScaler {
    max_abs: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for OnlineMaxAbsScaler {
    fn default() -> Self {
        Self {
            max_abs: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl OnlineMaxAbsScaler {
    /// Empty scaler.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for OnlineMaxAbsScaler {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if !self.initialized {
            self.max_abs = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if self.max_abs.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.max_abs.clone();
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let v = x.get(i, j).abs();
                if v.is_finite() && v > self.max_abs[j] {
                    self.max_abs[j] = v;
                }
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.max_abs.sub(&before);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.norm());
        q.information_gain = Some(delta.norm().max(1.0 / self.n_seen.max(1) as f64));
        q.still_identified = self.n_seen >= 1;
        q.warmup = self.n_seen < 2;
        q.explanation = format!("online MaxAbs on {} rows", x.nrows());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "running column max-abs",
                "stream extrema of |x|",
                "previous max-abs",
                "updated max-abs",
            ),
        )
    }
}

impl Transform for OnlineMaxAbsScaler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.max_abs.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j >= p {
                return x.get(i, j);
            }
            let s = self.max_abs[j];
            if !s.is_finite() || s <= ctx.policy.near_zero_variance {
                return 0.0;
            }
            x.get(i, j) / s
        });
        ctx.finish(out)
    }
}

/// Global-mean recommender (river `reco.Baseline` without biases).
#[derive(Clone, Debug, Default)]
pub struct Baseline {
    mean: f64,
    n_seen: u64,
    updates: u64,
}

impl Baseline {
    /// Empty baseline.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for Baseline {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no ratings"),
            );
        };
        let before = self.mean;
        for i in 0..y.len() {
            if !y[i].is_finite() {
                continue;
            }
            self.n_seen += 1;
            self.mean += (y[i] - self.mean) / self.n_seen as f64;
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.mean - before).abs());
        q.information_gain = Some((self.mean - before).abs());
        q.still_identified = self.n_seen >= 1;
        q.warmup = self.n_seen < 2;
        q.explanation = format!("baseline μ={:.4e} after {} ratings", self.mean, self.n_seen);
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "global rating mean",
                "Welford update of μ",
                format!("μ={before:.4e}"),
                format!("μ={:.4e}", self.mean),
            ),
        )
    }
}

impl Predict for Baseline {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.n_seen == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
        }
        ctx.finish(Vector::filled(x.nrows(), self.mean))
    }
}

/// Online k-NN regressor (river `neighbors.KNNRegressor`).
#[derive(Clone, Debug)]
pub struct KnnRegressor {
    /// Neighbourhood size.
    pub k: usize,
    xs: Vec<Vec<f64>>,
    ys: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl Default for KnnRegressor {
    fn default() -> Self {
        Self {
            k: 3,
            xs: Vec::new(),
            ys: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl KnnRegressor {
    /// `k`-NN memory.
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            ..Self::default()
        }
    }
}

impl PartialFit for KnnRegressor {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no targets"),
            );
        };
        if self.n_seen > 0 {
            if let Some(first) = self.xs.first() {
                if first.len() != x.ncols() {
                    ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
                    return finish_explain(
                        ctx,
                        reject_explain(
                            self.updates,
                            x.nrows(),
                            self.n_seen,
                            "feature space changed",
                        ),
                    );
                }
            }
        }
        let before = self.xs.len();
        for i in 0..x.nrows() {
            self.xs.push((0..x.ncols()).map(|j| x.get(i, j)).collect());
            self.ys.push(y[i]);
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.xs.len() - before) as f64);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.xs.len() >= self.k;
        q.warmup = self.xs.len() < self.k;
        q.explanation = format!("stored {} rows; memory={}", x.nrows(), self.xs.len());
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "instance memory",
                "append labelled rows for later neighbour averaging",
                format!("n={before}"),
                format!("n={}", self.xs.len()),
            ),
        )
    }
}

impl Predict for KnnRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.xs.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let k = self.k.max(1).min(self.xs.len());
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut dist: Vec<(f64, f64)> = self
                .xs
                .iter()
                .zip(self.ys.iter())
                .map(|(z, &yi)| {
                    let mut s = 0.0;
                    for j in 0..x.ncols().min(z.len()) {
                        let d = x.get(i, j) - z[j];
                        s += d * d;
                    }
                    (s, yi)
                })
                .collect();
            dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut num = 0.0;
            let mut den = 0.0;
            for (_, yi) in dist.into_iter().take(k) {
                num += yi;
                den += 1.0;
            }
            if den > 0.0 {
                num / den
            } else {
                0.0
            }
        }));
        ctx.finish(y)
    }
}

/// Online robust scaler via stochastic quartiles (river `preprocessing.RobustScaler`).
#[derive(Clone, Debug)]
pub struct OnlineRobustScaler {
    q25: Vector,
    q75: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for OnlineRobustScaler {
    fn default() -> Self {
        Self {
            q25: Vector::zeros(0),
            q75: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl OnlineRobustScaler {
    /// Empty scaler.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for OnlineRobustScaler {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if !self.initialized {
            self.q25 = Vector::zeros(x.ncols());
            self.q75 = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if self.q25.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.q25.clone();
        let eta = 1.0 / (self.n_seen as f64 + x.nrows() as f64).max(1.0);
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                self.q25[j] += eta * (if v < self.q25[j] { -0.5 } else { 0.5 } + 2.0 * 0.25 - 1.0);
                self.q75[j] += eta * (if v < self.q75[j] { -0.5 } else { 0.5 } + 2.0 * 0.75 - 1.0);
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.q25.sub(&before);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.norm());
        q.information_gain = Some(delta.norm().max(1.0 / self.n_seen.max(1) as f64));
        q.still_identified = self.n_seen >= 2;
        q.warmup = self.n_seen < 8;
        q.explanation = format!("online robust quartiles on {} rows", x.nrows());
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "running Q25/Q75",
                "stochastic quantile updates",
                "previous quartiles",
                "updated quartiles",
            ),
        )
    }
}

impl Transform for OnlineRobustScaler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.q25.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j >= p {
                return x.get(i, j);
            }
            let iqr = self.q75[j] - self.q25[j];
            if !iqr.is_finite() || iqr.abs() <= ctx.policy.near_zero_variance {
                return 0.0;
            }
            (x.get(i, j) - self.q25[j]) / iqr
        });
        ctx.finish(out)
    }
}

/// Online softmax / multinomial logistic (river `linear_model.SoftmaxRegression`).
#[derive(Clone, Debug)]
pub struct SoftmaxRegression {
    /// Learning rate.
    pub eta: f64,
    coef: Matrix,
    intercept: Vector,
    classes: Vec<i64>,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for SoftmaxRegression {
    fn default() -> Self {
        Self {
            eta: 0.1,
            coef: Matrix::zeros(0, 0),
            intercept: Vector::zeros(0),
            classes: Vec::new(),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl SoftmaxRegression {
    /// Empty softmax classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for SoftmaxRegression {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        for &yi in y.as_slice() {
            if !yi.is_finite() {
                continue;
            }
            let lab = yi.round() as i64;
            if !self.classes.contains(&lab) {
                self.classes.push(lab);
                self.classes.sort_unstable();
            }
        }
        let k = self.classes.len().max(1);
        let p = x.ncols();
        if !self.initialized {
            self.coef = Matrix::zeros(k, p);
            self.intercept = Vector::zeros(k);
            self.initialized = true;
        } else if self.coef.ncols() != p {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        } else if self.coef.nrows() != k {
            let mut nc = Matrix::zeros(k, p);
            let mut ni = Vector::zeros(k);
            for r in 0..self.coef.nrows().min(k) {
                for c in 0..p {
                    nc.set(r, c, self.coef.get(r, c));
                }
                ni[r] = self.intercept[r];
            }
            self.coef = nc;
            self.intercept = ni;
        }
        let before = self.coef.clone();
        let eta = self.eta.max(1e-6);
        for i in 0..x.nrows() {
            let lab = y[i].round() as i64;
            let Some(ti) = self.classes.iter().position(|&c| c == lab) else {
                continue;
            };
            let mut logits = vec![0.0; k];
            let mut m = f64::NEG_INFINITY;
            for c in 0..k {
                let mut s = self.intercept[c];
                for j in 0..p {
                    s += self.coef.get(c, j) * x.get(i, j);
                }
                logits[c] = s;
                m = m.max(s);
            }
            let mut z = 0.0;
            for c in 0..k {
                logits[c] = (logits[c] - m).exp();
                z += logits[c];
            }
            for c in 0..k {
                let p_c = logits[c] / z.max(1e-12);
                let g = (if c == ti { 1.0 } else { 0.0 }) - p_c;
                self.intercept[c] += eta * g;
                for j in 0..p {
                    self.coef
                        .set(c, j, self.coef.get(c, j) + eta * g * x.get(i, j));
                }
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        let mut delta: f64 = 0.0;
        for c in 0..k.min(before.nrows()) {
            for j in 0..p.min(before.ncols()) {
                let d = self.coef.get(c, j) - before.get(c, j);
                delta += d * d;
            }
        }
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.sqrt());
        q.information_gain = Some(delta.sqrt());
        q.still_identified = self.n_seen as usize > p && k >= 2;
        q.warmup = self.n_seen < (p as u64 + 2);
        q.explanation = format!("softmax SGD on {} rows, K={k}", x.nrows());
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "softmax weights",
                "one SGD pass on the batch",
                "previous weights",
                format!("n_seen={}", self.n_seen),
            ),
        )
    }
}

impl Predict for SoftmaxRegression {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized || self.classes.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let k = self.classes.len();
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut best = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for c in 0..k {
                let mut s = self.intercept[c];
                for j in 0..x.ncols().min(self.coef.ncols()) {
                    s += self.coef.get(c, j) * x.get(i, j);
                }
                if s > bv {
                    bv = s;
                    best = c;
                }
            }
            self.classes[best] as f64
        }));
        ctx.finish(y)
    }
}

/// Online Gaussian naïve Bayes (river `naive_bayes.GaussianNB`).
#[derive(Clone, Debug)]
pub struct OnlineGaussianNb {
    classes: Vec<i64>,
    n_per: Vec<f64>,
    mean: Matrix,
    m2: Matrix,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for OnlineGaussianNb {
    fn default() -> Self {
        Self {
            classes: Vec::new(),
            n_per: Vec::new(),
            mean: Matrix::zeros(0, 0),
            m2: Matrix::zeros(0, 0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl OnlineGaussianNb {
    /// Empty online GNB.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for OnlineGaussianNb {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        let p = x.ncols();
        if self.initialized && self.mean.ncols() != p {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before_n = self.n_seen;
        for i in 0..x.nrows() {
            if !y[i].is_finite() {
                continue;
            }
            let lab = y[i].round() as i64;
            let idx = if let Some(t) = self.classes.iter().position(|&c| c == lab) {
                t
            } else {
                self.classes.push(lab);
                self.n_per.push(0.0);
                let k = self.classes.len();
                let mut nm = Matrix::zeros(k, p);
                let mut n2 = Matrix::zeros(k, p);
                for r in 0..self.mean.nrows() {
                    for c in 0..p.min(self.mean.ncols()) {
                        nm.set(r, c, self.mean.get(r, c));
                        n2.set(r, c, self.m2.get(r, c));
                    }
                }
                self.mean = nm;
                self.m2 = n2;
                self.initialized = true;
                k - 1
            };
            self.n_per[idx] += 1.0;
            let n = self.n_per[idx];
            for j in 0..p {
                let v = x.get(i, j);
                let d = v - self.mean.get(idx, j);
                self.mean.set(idx, j, self.mean.get(idx, j) + d / n);
                let d2 = v - self.mean.get(idx, j);
                self.m2.set(idx, j, self.m2.get(idx, j) + d * d2);
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.n_seen - before_n) as f64);
        q.information_gain = Some((self.n_seen - before_n) as f64);
        q.still_identified = self.classes.len() >= 2 && self.n_seen >= 2;
        q.warmup = self.n_seen < 4;
        q.explanation = format!("online GNB on {} rows", x.nrows());
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "class-conditional means/variances",
                "Welford update per class",
                format!("n={before_n}"),
                format!("n={}", self.n_seen),
            ),
        )
    }
}

impl Predict for OnlineGaussianNb {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized || self.classes.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let k = self.classes.len();
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut best = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for c in 0..k {
                let mut ll = (self.n_per[c] / self.n_seen.max(1) as f64).max(1e-12).ln();
                for j in 0..x.ncols().min(self.mean.ncols()) {
                    let n = self.n_per[c].max(1.0);
                    let var = (self.m2.get(c, j) / n).max(1e-6);
                    let d = x.get(i, j) - self.mean.get(c, j);
                    ll += -0.5 * (d * d / var + var.ln());
                }
                if ll > bv {
                    bv = ll;
                    best = c;
                }
            }
            self.classes[best] as f64
        }));
        ctx.finish(y)
    }
}

/// Incremental TF–IDF on a document–term count stream (river `feature_extraction.TFIDF`).
///
/// Document count is not identification `p`. IDF uses add-one smoothing and is
/// recorded as a running approximation of the corpus IDF.
#[derive(Clone, Debug)]
pub struct OnlineTfidf {
    df: Vector,
    n_docs: u64,
    n_features: usize,
    updates: u64,
    initialized: bool,
}

impl Default for OnlineTfidf {
    fn default() -> Self {
        Self {
            df: Vector::zeros(0),
            n_docs: 0,
            n_features: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl OnlineTfidf {
    /// Empty streaming TF–IDF.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for OnlineTfidf {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_docs, "no features"),
            );
        }
        if !self.initialized {
            self.df = Vector::zeros(x.ncols());
            self.n_features = x.ncols();
            self.initialized = true;
        } else if self.n_features != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_docs,
                    "feature space changed",
                ),
            );
        }
        let before = self.df.clone();
        for i in 0..x.nrows() {
            let mut row_sum: f64 = 0.0;
            for j in 0..x.ncols() {
                let v = x.get(i, j);
                if v.is_finite() && v > 0.0 {
                    row_sum += v;
                    self.df[j] += 1.0;
                }
            }
            if row_sum <= 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NearZeroVariance)
                        .message(format!(
                            "OnlineTfidf document {i} has no positive term counts"
                        ))
                        .build(),
                );
            }
            self.n_docs += 1;
        }
        self.updates += 1;
        let delta = self.df.sub(&before);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_docs);
        q.effective_sample_size = self.n_docs as f64;
        q.parameter_delta_norm = Some(delta.norm());
        q.information_gain = Some(delta.norm().max(1.0 / self.n_docs.max(1) as f64));
        q.still_identified = self.n_docs >= 1;
        q.warmup = self.n_docs < 4;
        q.explanation = format!("online TF-IDF df on {} documents", x.nrows());
        flag_info(&mut ctx, &q);
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(Severity::Advisory)
                .message(
                    "OnlineTfidf IDF is a running add-one estimate, not the finished corpus IDF",
                )
                .compromise(NumericalCompromise::new(
                    "batch TF-IDF on the complete collection",
                    "smoothed IDF from documents seen so far",
                    "early-stream IDF is biased toward the first documents",
                    "do not treat early IDF weights as a static vocabulary ranking",
                ))
                .build(),
        );
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "document frequencies",
                "increment df for each positive term",
                "previous df",
                format!("n_docs={}", self.n_docs),
            ),
        )
    }
}

impl Transform for OnlineTfidf {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.df.len());
        let n = self.n_docs.max(1) as f64;
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let tf = x.get(i, j).max(0.0);
            if j >= p {
                return tf;
            }
            let idf = ((n + 1.0) / (self.df[j] + 1.0)).ln() + 1.0;
            tf * idf
        });
        ctx.finish(out)
    }
}

/// Running-mean imputer (river `imputation.StatImputer`).
///
/// Non-finite entries are skipped in the sufficient statistics. A column that
/// has never been observed finite is filled with 0 and recorded; that is not
/// a vacuous all-missing abort.
#[derive(Clone, Debug)]
pub struct StatImputer {
    mean: Vector,
    m2: Vector,
    count: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for StatImputer {
    fn default() -> Self {
        Self {
            mean: Vector::zeros(0),
            m2: Vector::zeros(0),
            count: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl StatImputer {
    /// Empty streaming mean imputer.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for StatImputer {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if !self.initialized {
            self.mean = Vector::zeros(x.ncols());
            self.m2 = Vector::zeros(x.ncols());
            self.count = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if self.mean.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.mean.clone();
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                self.count[j] += 1.0;
                let n = self.count[j];
                let d = v - self.mean[j];
                self.mean[j] += d / n;
                let d2 = v - self.mean[j];
                self.m2[j] += d * d2;
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.mean.sub(&before);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self
            .count
            .as_slice()
            .iter()
            .copied()
            .fold(0.0, |a, b| a.max(b));
        q.parameter_delta_norm = Some(delta.norm());
        q.information_gain = Some(delta.norm().max(1.0 / self.n_seen.max(1) as f64));
        q.still_identified = self.count.as_slice().iter().all(|&c| c >= 1.0);
        q.warmup = self.n_seen < 2;
        q.explanation = format!("StatImputer Welford update on {} rows", x.nrows());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "column means",
                "Welford on finite entries",
                "previous means",
                "updated means",
            ),
        )
    }
}

impl Transform for StatImputer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.mean.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let v = x.get(i, j);
            if v.is_finite() {
                return v;
            }
            if j >= p || self.count[j] < 1.0 {
                return 0.0;
            }
            self.mean[j]
        });
        for j in 0..p {
            if self.count[j] < 1.0 {
                ctx.push(
                    Issue::builder(IssueCode::ImputationUndefined)
                        .severity(Severity::Warning)
                        .message(format!(
                            "StatImputer column {j} has no finite observations; missing values become 0"
                        ))
                        .build(),
                );
            }
        }
        ctx.finish(out)
    }
}

/// Streaming univariate ANOVA *F* feature filter (river `feature_selection.SelectKBest`).
///
/// Class count is not identification `p`. Scores are class-conditional Welford
/// moments; a thin class is recorded, not treated as a cluster/`p` gate.
#[derive(Clone, Debug)]
pub struct OnlineSelectKBest {
    /// Number of columns to keep.
    pub k: usize,
    classes: Vec<i64>,
    n_per: Vec<f64>,
    mean: Matrix,
    m2: Matrix,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for OnlineSelectKBest {
    fn default() -> Self {
        Self {
            k: 1,
            classes: Vec::new(),
            n_per: Vec::new(),
            mean: Matrix::zeros(0, 0),
            m2: Matrix::zeros(0, 0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl OnlineSelectKBest {
    /// Keep `k` columns with the largest running ANOVA *F*.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            ..Self::default()
        }
    }

    fn scores(&self) -> Vector {
        let p = self.mean.ncols();
        let k = self.classes.len();
        if k < 2 || p == 0 {
            return Vector::zeros(p);
        }
        Vector::from_iter((0..p).map(|j| {
            let mut ntot = 0.0;
            let mut grand = 0.0;
            for c in 0..k {
                ntot += self.n_per[c];
                grand += self.n_per[c] * self.mean.get(c, j);
            }
            if ntot <= 0.0 {
                return 0.0;
            }
            grand /= ntot;
            let mut ssb = 0.0;
            let mut ssw = 0.0;
            for c in 0..k {
                let d = self.mean.get(c, j) - grand;
                ssb += self.n_per[c] * d * d;
                ssw += self.m2.get(c, j);
            }
            let dfb = (k as f64 - 1.0).max(1.0);
            let dfw = (ntot - k as f64).max(1.0);
            if ssw <= 1e-18 {
                return 0.0;
            }
            (ssb / dfb) / (ssw / dfw)
        }))
    }
}

impl PartialFit for OnlineSelectKBest {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        let p = x.ncols();
        if self.k == 0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message("OnlineSelectKBest.k=0; transform will keep no columns")
                    .build(),
            );
        }
        if self.initialized && self.mean.ncols() != p {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before_n = self.n_seen;
        for i in 0..x.nrows() {
            if !y[i].is_finite() {
                continue;
            }
            let lab = y[i].round() as i64;
            let idx = if let Some(t) = self.classes.iter().position(|&c| c == lab) {
                t
            } else {
                self.classes.push(lab);
                self.n_per.push(0.0);
                let kc = self.classes.len();
                let mut nm = Matrix::zeros(kc, p);
                let mut n2 = Matrix::zeros(kc, p);
                for r in 0..self.mean.nrows() {
                    for c in 0..p.min(self.mean.ncols()) {
                        nm.set(r, c, self.mean.get(r, c));
                        n2.set(r, c, self.m2.get(r, c));
                    }
                }
                self.mean = nm;
                self.m2 = n2;
                self.initialized = true;
                kc - 1
            };
            self.n_per[idx] += 1.0;
            let n = self.n_per[idx];
            for j in 0..p {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                let d = v - self.mean.get(idx, j);
                self.mean.set(idx, j, self.mean.get(idx, j) + d / n);
                let d2 = v - self.mean.get(idx, j);
                self.m2.set(idx, j, self.m2.get(idx, j) + d * d2);
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        if self.classes.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("OnlineSelectKBest has <2 classes; ANOVA F is unidentified")
                    .build(),
            );
        }
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.n_seen - before_n) as f64);
        q.information_gain = Some((self.n_seen - before_n) as f64);
        q.still_identified = self.classes.len() >= 2 && self.n_seen >= 2;
        q.warmup = self.n_seen < 4;
        q.explanation = format!("online SelectKBest ANOVA on {} rows", x.nrows());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "class-conditional moments",
                "Welford update then ANOVA F ranking",
                format!("n={before_n}"),
                format!("n={}", self.n_seen),
            ),
        )
    }
}

impl Transform for OnlineSelectKBest {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Matrix::zeros(x.nrows(), 0));
        }
        let scores = self.scores();
        let p = scores.len().min(x.ncols());
        let k = self.k.min(p);
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
        order.truncate(k);
        order.sort_unstable();
        let out = Matrix::from_fn(x.nrows(), k, |i, j| x.get(i, order[j]));
        ctx.finish(out)
    }
}

/// Rolling mean absolute error (river `metrics.MAE`).
#[derive(Clone, Debug, Default)]
pub struct OnlineMae {
    sae: f64,
    n: u64,
    updates: u64,
}

impl OnlineMae {
    /// Empty MAE.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current MAE, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.sae / self.n as f64
        }
    }
}

impl PartialFit for OnlineMae {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        metric_partial(session, "mae", x, y, self.updates, self.n, |pred, truth| {
            self.n += 1;
            self.sae += (pred - truth).abs();
            self.updates += 1;
            self.score()
        })
    }
}

/// Rolling log-loss (river `metrics.LogLoss`).
///
/// `x` is treated as a probability and clipped to \((\varepsilon,1-\varepsilon)\).
#[derive(Clone, Debug, Default)]
pub struct OnlineLogLoss {
    nll: f64,
    n: u64,
    updates: u64,
}

impl OnlineLogLoss {
    /// Empty log-loss.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current mean log-loss, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.nll / self.n as f64
        }
    }
}

impl PartialFit for OnlineLogLoss {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        metric_partial(
            session,
            "log_loss",
            x,
            y,
            self.updates,
            self.n,
            |pred, truth| {
                let p = pred.clamp(1e-12, 1.0 - 1e-12);
                let t = if truth > 0.5 { 1.0 } else { 0.0 };
                self.nll += -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
                self.n += 1;
                self.updates += 1;
                self.score()
            },
        )
    }
}

/// Windowed ROC AUC (river `metrics.ROCAUC`).
///
/// Pairs are kept in a 256-row buffer; the AUC is the Mann–Whitney U on that
/// window, not the exact stream AUC.
#[derive(Clone, Debug)]
pub struct OnlineRocAuc {
    scores: Vec<f64>,
    labels: Vec<f64>,
    updates: u64,
}

impl Default for OnlineRocAuc {
    fn default() -> Self {
        Self {
            scores: Vec::new(),
            labels: Vec::new(),
            updates: 0,
        }
    }
}

impl OnlineRocAuc {
    /// Empty windowed AUC.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mann–Whitney AUC on the current window.
    pub fn score(&self) -> f64 {
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for (s, y) in self.scores.iter().zip(&self.labels) {
            if *y > 0.5 {
                pos.push(*s);
            } else {
                neg.push(*s);
            }
        }
        if pos.is_empty() || neg.is_empty() {
            return f64::NAN;
        }
        let mut ge = 0.0;
        for &p in &pos {
            for &n in &neg {
                if p > n {
                    ge += 1.0;
                } else if (p - n).abs() < 1e-15 {
                    ge += 0.5;
                }
            }
        }
        ge / (pos.len() * neg.len()) as f64
    }
}

impl PartialFit for OnlineRocAuc {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let before = self.scores.len() as u64;
        metric_partial(
            session,
            "roc_auc",
            x,
            y,
            self.updates,
            before,
            |pred, truth| {
                self.scores.push(pred);
                self.labels.push(truth);
                if self.scores.len() > 256 {
                    self.scores.remove(0);
                    self.labels.remove(0);
                }
                self.updates += 1;
                self.score()
            },
        )
    }
}

/// Streaming multinomial naïve Bayes (river `naive_bayes.MultinomialNB`).
///
/// Class count is not identification `p`. Negative counts are skipped.
#[derive(Clone, Debug)]
pub struct OnlineMultinomialNb {
    /// Additive smoothing.
    pub alpha: f64,
    classes: Vec<i64>,
    counts: Matrix,
    class_n: Vec<f64>,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for OnlineMultinomialNb {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            classes: Vec::new(),
            counts: Matrix::zeros(0, 0),
            class_n: Vec::new(),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl OnlineMultinomialNb {
    /// Empty multinomial NB.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for OnlineMultinomialNb {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        let p = x.ncols();
        if self.initialized && self.counts.ncols() != p {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.n_seen;
        for i in 0..x.nrows() {
            if !y[i].is_finite() {
                continue;
            }
            let lab = y[i].round() as i64;
            let idx = if let Some(t) = self.classes.iter().position(|&c| c == lab) {
                t
            } else {
                self.classes.push(lab);
                self.class_n.push(0.0);
                let k = self.classes.len();
                let mut nc = Matrix::zeros(k, p);
                for r in 0..self.counts.nrows() {
                    for c in 0..p.min(self.counts.ncols()) {
                        nc.set(r, c, self.counts.get(r, c));
                    }
                }
                self.counts = nc;
                self.initialized = true;
                k - 1
            };
            self.class_n[idx] += 1.0;
            for j in 0..p {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                if v < 0.0 {
                    ctx.push(
                        Issue::builder(IssueCode::InvalidWeight)
                            .severity(Severity::Warning)
                            .message(format!(
                                "OnlineMultinomialNb skipped negative count x[{i},{j}]={v}"
                            ))
                            .build(),
                    );
                    continue;
                }
                self.counts.set(idx, j, self.counts.get(idx, j) + v);
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.n_seen - before) as f64);
        q.information_gain = Some((self.n_seen - before) as f64);
        q.still_identified = self.classes.len() >= 2 && self.n_seen >= 2;
        q.warmup = self.n_seen < 4;
        q.explanation = format!("online MultinomialNB on {} rows", x.nrows());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "class-conditional token counts",
                "accumulate non-negative counts",
                format!("n={before}"),
                format!("n={}", self.n_seen),
            ),
        )
    }
}

impl Predict for OnlineMultinomialNb {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized || self.classes.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let k = self.classes.len();
        let p = self.counts.ncols();
        let tot: f64 = self.class_n.iter().sum();
        let alpha = self.alpha.max(0.0);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut best = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for c in 0..k {
                let mut s = ((self.class_n[c] + alpha) / (tot + alpha * k as f64)).ln();
                let row_tot: f64 =
                    (0..p).map(|j| self.counts.get(c, j)).sum::<f64>() + alpha * p as f64;
                for j in 0..x.ncols().min(p) {
                    let v = x.get(i, j).max(0.0);
                    let lp = (self.counts.get(c, j) + alpha).ln() - row_tot.ln();
                    s += v * lp;
                }
                if s > bv {
                    bv = s;
                    best = c;
                }
            }
            self.classes[best] as f64
        }));
        ctx.finish(y)
    }
}

/// Last-observation imputer (river `imputation.PreviousImputer`).
#[derive(Clone, Debug)]
pub struct PreviousImputer {
    last: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for PreviousImputer {
    fn default() -> Self {
        Self {
            last: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl PreviousImputer {
    /// Empty previous-value imputer.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for PreviousImputer {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if !self.initialized {
            self.last = Vector::from_iter((0..x.ncols()).map(|_| f64::NAN));
            self.initialized = true;
        } else if self.last.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.last.clone();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let v = x.get(i, j);
                if v.is_finite() {
                    self.last[j] = v;
                }
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut delta: f64 = 0.0;
        for j in 0..self.last.len() {
            if before[j].is_finite() && self.last[j].is_finite() {
                delta += (self.last[j] - before[j]).abs();
            }
        }
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta);
        q.information_gain = Some(delta.max(1.0 / self.n_seen.max(1) as f64));
        q.still_identified = self.last.as_slice().iter().any(|v| v.is_finite());
        q.warmup = !q.still_identified;
        q.explanation = format!("PreviousImputer last-value update on {} rows", x.nrows());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "last finite value per column",
                "overwrite on each finite observation",
                "previous last values",
                "updated last values",
            ),
        )
    }
}

impl Transform for PreviousImputer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.last.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let v = x.get(i, j);
            if v.is_finite() {
                return v;
            }
            if j < p && self.last[j].is_finite() {
                self.last[j]
            } else {
                0.0
            }
        });
        for j in 0..p {
            if !self.last[j].is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::ImputationUndefined)
                        .severity(Severity::Warning)
                        .message(format!(
                            "PreviousImputer column {j} has never been observed finite"
                        ))
                        .build(),
                );
            }
        }
        ctx.finish(out)
    }
}

/// Exponentially weighted standard scaler (river `preprocessing.AdaptiveStandardScaler`).
#[derive(Clone, Debug)]
pub struct AdaptiveStandardScaler {
    /// Forgetting factor in `(0, 1]`.
    pub fading: f64,
    mean: Vector,
    var: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for AdaptiveStandardScaler {
    fn default() -> Self {
        Self {
            fading: 0.6,
            mean: Vector::zeros(0),
            var: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl AdaptiveStandardScaler {
    /// Scaler with fading factor `fading`.
    pub fn new(fading: f64) -> Self {
        Self {
            fading,
            ..Self::default()
        }
    }
}

impl PartialFit for AdaptiveStandardScaler {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let fade = if self.fading.is_finite() && self.fading > 0.0 && self.fading <= 1.0 {
            self.fading
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "AdaptiveStandardScaler.fading={} is not in (0,1]; using 0.6",
                        self.fading
                    ))
                    .build(),
            );
            0.6
        };
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if !self.initialized {
            self.mean = Vector::zeros(x.ncols());
            self.var = Vector::from_iter((0..x.ncols()).map(|_| 1.0));
            self.initialized = true;
        } else if self.mean.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.mean.clone();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                let d = v - self.mean[j];
                self.mean[j] += fade * d;
                self.var[j] = (1.0 - fade) * self.var[j] + fade * d * d;
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.mean.sub(&before);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = (1.0 / fade).min(self.n_seen as f64);
        q.parameter_delta_norm = Some(delta.norm());
        q.information_gain = Some(delta.norm().max(fade));
        q.still_identified = self.n_seen >= 2;
        q.warmup = self.n_seen < 4;
        q.explanation = format!(
            "adaptive standard scaler on {} rows, fade={fade}",
            x.nrows()
        );
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "EW mean/variance",
                "exponentially weighted moment update",
                "previous moments",
                format!("n_seen={}", self.n_seen),
            ),
        )
    }
}

impl Transform for AdaptiveStandardScaler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.mean.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j >= p {
                return x.get(i, j);
            }
            let s = self.var[j].sqrt().max(1e-12);
            (x.get(i, j) - self.mean[j]) / s
        });
        ctx.finish(out)
    }
}

/// Streaming Gaussian anomaly score (river `anomaly.GaussianScorer`).
#[derive(Clone, Debug)]
pub struct GaussianScorer {
    mean: Vector,
    m2: Vector,
    count: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for GaussianScorer {
    fn default() -> Self {
        Self {
            mean: Vector::zeros(0),
            m2: Vector::zeros(0),
            count: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl GaussianScorer {
    /// Empty Gaussian scorer.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for GaussianScorer {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if !self.initialized {
            self.mean = Vector::zeros(x.ncols());
            self.m2 = Vector::zeros(x.ncols());
            self.count = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if self.mean.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.mean.clone();
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                self.count[j] += 1.0;
                let n = self.count[j];
                let d = v - self.mean[j];
                self.mean[j] += d / n;
                let d2 = v - self.mean[j];
                self.m2[j] += d * d2;
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.mean.sub(&before);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.norm());
        q.information_gain = Some(delta.norm().max(1.0 / self.n_seen.max(1) as f64));
        q.still_identified = self.count.as_slice().iter().all(|&c| c >= 2.0);
        q.warmup = self.n_seen < 4;
        q.explanation = format!("GaussianScorer Welford update on {} rows", x.nrows());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "column mean/variance",
                "Welford update for a Gaussian tail score",
                "previous moments",
                "updated moments",
            ),
        )
    }
}

impl Transform for GaussianScorer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Matrix::zeros(x.nrows(), x.ncols()));
        }
        let p = x.ncols().min(self.mean.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j >= p || self.count[j] < 2.0 {
                return 0.0;
            }
            let var = (self.m2[j] / self.count[j]).max(1e-12);
            let z = (x.get(i, j) - self.mean[j]) / var.sqrt();
            z.abs()
        });
        ctx.finish(out)
    }
}

/// Streaming F1 (river `metrics.F1`).
#[derive(Clone, Debug, Default)]
pub struct OnlineF1 {
    tp: f64,
    fp: f64,
    fn_: f64,
    updates: u64,
}

impl OnlineF1 {
    /// Empty F1.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current F1, or NaN when no positive predictions or labels.
    pub fn score(&self) -> f64 {
        let den = 2.0 * self.tp + self.fp + self.fn_;
        if den <= 0.0 {
            f64::NAN
        } else {
            2.0 * self.tp / den
        }
    }
}

impl PartialFit for OnlineF1 {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        metric_partial(
            session,
            "f1",
            x,
            y,
            self.updates,
            self.tp as u64,
            |pred, truth| {
                let p = pred > 0.5;
                let t = truth > 0.5;
                if p && t {
                    self.tp += 1.0;
                } else if p && !t {
                    self.fp += 1.0;
                } else if !p && t {
                    self.fn_ += 1.0;
                }
                self.updates += 1;
                self.score()
            },
        )
    }
}

/// Streaming precision (river `metrics.Precision`).
#[derive(Clone, Debug, Default)]
pub struct OnlinePrecision {
    tp: f64,
    fp: f64,
    updates: u64,
}

impl OnlinePrecision {
    /// Empty precision.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current precision, or NaN when no positive predictions.
    pub fn score(&self) -> f64 {
        let den = self.tp + self.fp;
        if den <= 0.0 {
            f64::NAN
        } else {
            self.tp / den
        }
    }
}

impl PartialFit for OnlinePrecision {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        metric_partial(
            session,
            "precision",
            x,
            y,
            self.updates,
            self.tp as u64,
            |pred, truth| {
                if pred > 0.5 {
                    if truth > 0.5 {
                        self.tp += 1.0;
                    } else {
                        self.fp += 1.0;
                    }
                }
                self.updates += 1;
                self.score()
            },
        )
    }
}

/// Streaming recall (river `metrics.Recall`).
#[derive(Clone, Debug, Default)]
pub struct OnlineRecall {
    tp: f64,
    fn_: f64,
    updates: u64,
}

impl OnlineRecall {
    /// Empty recall.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current recall, or NaN when no positive labels.
    pub fn score(&self) -> f64 {
        let den = self.tp + self.fn_;
        if den <= 0.0 {
            f64::NAN
        } else {
            self.tp / den
        }
    }
}

impl PartialFit for OnlineRecall {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        metric_partial(
            session,
            "recall",
            x,
            y,
            self.updates,
            self.tp as u64,
            |pred, truth| {
                if truth > 0.5 {
                    if pred > 0.5 {
                        self.tp += 1.0;
                    } else {
                        self.fn_ += 1.0;
                    }
                }
                self.updates += 1;
                self.score()
            },
        )
    }
}

/// Streaming Bernoulli naïve Bayes (river `naive_bayes.BernoulliNB`).
///
/// Class count is not identification `p`. Features are treated as
/// \(\{0,1\}\) via a \(0.5\) threshold.
#[derive(Clone, Debug)]
pub struct OnlineBernoulliNb {
    /// Additive smoothing.
    pub alpha: f64,
    classes: Vec<i64>,
    ones: Matrix,
    class_n: Vec<f64>,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for OnlineBernoulliNb {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            classes: Vec::new(),
            ones: Matrix::zeros(0, 0),
            class_n: Vec::new(),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl OnlineBernoulliNb {
    /// Empty Bernoulli NB.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for OnlineBernoulliNb {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        let p = x.ncols();
        if self.initialized && self.ones.ncols() != p {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.n_seen;
        for i in 0..x.nrows() {
            if !y[i].is_finite() {
                continue;
            }
            let lab = y[i].round() as i64;
            let idx = if let Some(t) = self.classes.iter().position(|&c| c == lab) {
                t
            } else {
                self.classes.push(lab);
                self.class_n.push(0.0);
                let k = self.classes.len();
                let mut nc = Matrix::zeros(k, p);
                for r in 0..self.ones.nrows() {
                    for c in 0..p.min(self.ones.ncols()) {
                        nc.set(r, c, self.ones.get(r, c));
                    }
                }
                self.ones = nc;
                self.initialized = true;
                k - 1
            };
            self.class_n[idx] += 1.0;
            for j in 0..p {
                if x.get(i, j) > 0.5 {
                    self.ones.set(idx, j, self.ones.get(idx, j) + 1.0);
                }
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.n_seen - before) as f64);
        q.information_gain = Some((self.n_seen - before) as f64);
        q.still_identified = self.classes.len() >= 2 && self.n_seen >= 2;
        q.warmup = self.n_seen < 4;
        q.explanation = format!("online BernoulliNB on {} rows", x.nrows());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "class-conditional Bernoulli ones counts",
                "threshold features at 0.5 and accumulate",
                format!("n={before}"),
                format!("n={}", self.n_seen),
            ),
        )
    }
}

impl Predict for OnlineBernoulliNb {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized || self.classes.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let k = self.classes.len();
        let p = self.ones.ncols();
        let tot: f64 = self.class_n.iter().sum();
        let alpha = self.alpha.max(0.0);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut best = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for c in 0..k {
                let mut s = ((self.class_n[c] + alpha) / (tot + alpha * k as f64)).ln();
                for j in 0..x.ncols().min(p) {
                    let phat =
                        (self.ones.get(c, j) + alpha) / (self.class_n[c] + 2.0 * alpha).max(1e-12);
                    let phat = phat.clamp(1e-12, 1.0 - 1e-12);
                    let on = if x.get(i, j) > 0.5 { 1.0 } else { 0.0 };
                    s += on * phat.ln() + (1.0 - on) * (1.0 - phat).ln();
                }
                if s > bv {
                    bv = s;
                    best = c;
                }
            }
            self.classes[best] as f64
        }));
        ctx.finish(y)
    }
}

/// Exponentially weighted average of a mean expert and a last-value expert
/// (river `optim.EWARegressor` lite).
///
/// Features are ignored; the mix is a two-expert forecast of the target.
#[derive(Clone, Debug)]
pub struct EwaRegressor {
    /// Multiplicative weight step \(\eta > 0\).
    pub eta: f64,
    w_mean: f64,
    w_last: f64,
    mean: f64,
    last: f64,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for EwaRegressor {
    fn default() -> Self {
        Self {
            eta: 0.5,
            w_mean: 1.0,
            w_last: 1.0,
            mean: 0.0,
            last: 0.0,
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl EwaRegressor {
    /// EWA mix with step `eta`.
    pub fn new(eta: f64) -> Self {
        Self {
            eta,
            ..Self::default()
        }
    }

    fn mix(&self) -> f64 {
        let z = self.w_mean + self.w_last;
        if z <= 0.0 {
            self.mean
        } else {
            (self.w_mean * self.mean + self.w_last * self.last) / z
        }
    }
}

impl PartialFit for EwaRegressor {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        let eta = if self.eta.is_finite() && self.eta > 0.0 {
            self.eta
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "EwaRegressor.eta={} is not positive; using 0.5",
                        self.eta
                    ))
                    .build(),
            );
            0.5
        };
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message(
                    "EwaRegressor mixes a running-mean expert and a last-value expert; x is unused",
                )
                .compromise(NumericalCompromise::new(
                    "feature-conditioned expert mix",
                    "two target-only experts with multiplicative weights",
                    "no model of x → y is identified",
                    "read the mix as an online forecast of y, not a feature effect",
                ))
                .build(),
        );
        let before = self.mix();
        for i in 0..x.nrows().min(y.len()) {
            if !y[i].is_finite() {
                continue;
            }
            let yt = y[i];
            if self.initialized {
                let e_mean = (self.mean - yt) * (self.mean - yt);
                let e_last = (self.last - yt) * (self.last - yt);
                self.w_mean *= (-eta * e_mean).exp();
                self.w_last *= (-eta * e_last).exp();
                let z = (self.w_mean + self.w_last).max(1e-18);
                self.w_mean = (self.w_mean / z).max(1e-12);
                self.w_last = (self.w_last / z).max(1e-12);
                let n = self.n_seen as f64;
                self.mean = (self.mean * n + yt) / (n + 1.0);
                self.last = yt;
            } else {
                self.mean = yt;
                self.last = yt;
                self.initialized = true;
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        let after = self.mix();
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((after - before).abs());
        q.information_gain = Some((after - before).abs().max(1.0 / self.n_seen.max(1) as f64));
        q.still_identified = self.initialized && self.n_seen >= 2;
        q.warmup = self.n_seen < 2;
        q.explanation = format!("EWA mix after {} rows, pred={after:.6e}", x.nrows());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "mean/last expert weights",
                "multiplicative loss update, then renormalize",
                format!("mix={before:.6e}"),
                format!("mix={after:.6e}"),
            ),
        )
    }
}

impl Predict for EwaRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let m = self.mix();
        ctx.finish(Vector::from_iter((0..x.nrows()).map(|_| m)))
    }
}

/// Running-quantile clip (river `anomaly.QuantileFilter` / winsorize).
///
/// Values above the empirical `q`-quantile of a 256-row window are clipped
/// to that quantile. `q` is not identification `p`.
#[derive(Clone, Debug)]
pub struct QuantileFilter {
    /// Upper quantile in \((0, 1)\).
    pub q: f64,
    window: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl Default for QuantileFilter {
    fn default() -> Self {
        Self {
            q: 0.95,
            window: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl QuantileFilter {
    /// Filter at quantile `q`.
    pub fn new(q: f64) -> Self {
        Self {
            q,
            ..Self::default()
        }
    }

    fn quantile(&self, q: f64) -> f64 {
        if self.window.is_empty() {
            return f64::NAN;
        }
        let mut xs = self.window.clone();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((q.clamp(0.0, 1.0)) * (xs.len() - 1) as f64).round() as usize;
        xs[idx.min(xs.len() - 1)]
    }
}

impl PartialFit for QuantileFilter {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let qv = if self.q.is_finite() && self.q > 0.0 && self.q < 1.0 {
            self.q
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "QuantileFilter.q={} is not in (0,1); using 0.95",
                        self.q
                    ))
                    .build(),
            );
            0.95
        };
        let before = self.quantile(qv);
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let v = x.get(i, j);
                if v.is_finite() {
                    self.window.push(v);
                    if self.window.len() > 256 {
                        self.window.remove(0);
                    }
                    self.n_seen += 1;
                }
            }
        }
        self.updates += 1;
        let after = self.quantile(qv);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.window.len() as f64;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.window.len() >= 4;
        q.warmup = self.window.len() < 4;
        q.explanation = format!("QuantileFilter q={qv:.3} threshold={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "empirical upper quantile",
                "256-row sliding window order statistic",
                format!("q={before:.6e}"),
                format!("q={after:.6e}"),
            ),
        )
    }
}

impl Transform for QuantileFilter {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if self.window.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let qv = if self.q.is_finite() && self.q > 0.0 && self.q < 1.0 {
            self.q
        } else {
            0.95
        };
        let cap = self.quantile(qv);
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let v = x.get(i, j);
            if v.is_finite() && v > cap {
                cap
            } else {
                v
            }
        });
        ctx.finish(out)
    }
}

/// Streaming RMSE (river `metrics.RMSE`).
#[derive(Clone, Debug, Default)]
pub struct OnlineRmse {
    sse: f64,
    n: u64,
    updates: u64,
}

impl OnlineRmse {
    /// Empty RMSE.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current RMSE, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            (self.sse / self.n as f64).sqrt()
        }
    }
}

impl PartialFit for OnlineRmse {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        metric_partial(
            session,
            "rmse",
            x,
            y,
            self.updates,
            self.n,
            |pred, truth| {
                let e = pred - truth;
                self.sse += e * e;
                self.n += 1;
                self.updates += 1;
                self.score()
            },
        )
    }
}

/// Streaming Cohen's κ (river `metrics.CohenKappa`).
#[derive(Clone, Debug, Default)]
pub struct OnlineCohenKappa {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    updates: u64,
}

impl OnlineCohenKappa {
    /// Empty κ.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current κ, or NaN when empty.
    pub fn score(&self) -> f64 {
        let n = self.a + self.b + self.c + self.d;
        if n <= 0.0 {
            return f64::NAN;
        }
        let po = (self.a + self.d) / n;
        let pe = ((self.a + self.b) * (self.a + self.c) + (self.c + self.d) * (self.b + self.d))
            / (n * n);
        if (1.0 - pe).abs() <= 1e-18 {
            f64::NAN
        } else {
            (po - pe) / (1.0 - pe)
        }
    }
}

impl PartialFit for OnlineCohenKappa {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        metric_partial(
            session,
            "cohen_kappa",
            x,
            y,
            self.updates,
            (self.a + self.b + self.c + self.d) as u64,
            |pred, truth| {
                let p = pred > 0.5;
                let t = truth > 0.5;
                match (p, t) {
                    (false, false) => self.a += 1.0,
                    (true, false) => self.b += 1.0,
                    (false, true) => self.c += 1.0,
                    (true, true) => self.d += 1.0,
                }
                self.updates += 1;
                self.score()
            },
        )
    }
}

/// Exponentially weighted mean (river `stats.EWMean`).
#[derive(Clone, Debug)]
pub struct EwMean {
    /// Fade factor in \((0, 1]\). `1` is the expanding mean.
    pub fading: f64,
    mean: f64,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for EwMean {
    fn default() -> Self {
        Self {
            fading: 0.5,
            mean: 0.0,
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl EwMean {
    /// EW mean with fade `fading`.
    pub fn new(fading: f64) -> Self {
        Self {
            fading,
            ..Self::default()
        }
    }

    /// Current mean, or NaN before the first update.
    pub fn score(&self) -> f64 {
        if self.initialized {
            self.mean
        } else {
            f64::NAN
        }
    }
}

impl PartialFit for EwMean {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let fade = if self.fading.is_finite() && self.fading > 0.0 && self.fading <= 1.0 {
            self.fading
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "EwMean.fading={} is not in (0,1]; using 0.5",
                        self.fading
                    ))
                    .build(),
            );
            0.5
        };
        let before = self.mean;
        for i in 0..x.nrows() {
            let v = x.get(i, 0);
            if !v.is_finite() {
                continue;
            }
            if !self.initialized {
                self.mean = v;
                self.initialized = true;
            } else if fade >= 1.0 {
                let n = self.n_seen as f64;
                self.mean = (self.mean * n + v) / (n + 1.0);
            } else {
                self.mean = fade * self.mean + (1.0 - fade) * v;
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = if fade >= 1.0 {
            self.n_seen as f64
        } else {
            (1.0 / (1.0 - fade)).min(self.n_seen as f64)
        };
        q.parameter_delta_norm = Some((self.mean - before).abs());
        q.information_gain = Some(
            (self.mean - before)
                .abs()
                .max(1.0 / self.n_seen.max(1) as f64),
        );
        q.still_identified = self.initialized;
        q.warmup = self.n_seen < 2;
        q.explanation = format!("EWMean fade={fade:.3} mean={:.6e}", self.mean);
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "exponentially weighted mean",
                "fade previous mean and fold in the new observation",
                format!("mean={before:.6e}"),
                format!("mean={:.6e}", self.mean),
            ),
        )
    }
}

impl Predict for EwMean {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        ctx.finish(Vector::from_iter((0..x.nrows()).map(|_| self.mean)))
    }
}

/// Degree-2 polynomial expander (river `preprocessing.PolynomialExtender`).
///
/// Expanded width is not identification `p`.
#[derive(Clone, Debug)]
pub struct PolynomialExtender {
    /// Polynomial degree (`1` or `2`).
    pub degree: usize,
    n_in: usize,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for PolynomialExtender {
    fn default() -> Self {
        Self {
            degree: 2,
            n_in: 0,
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl PolynomialExtender {
    /// Expander of the given degree.
    pub fn new(degree: usize) -> Self {
        Self {
            degree,
            ..Self::default()
        }
    }
}

impl PartialFit for PolynomialExtender {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let deg = if self.degree == 1 || self.degree == 2 {
            self.degree
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "PolynomialExtender.degree={} is not 1 or 2; using 2",
                        self.degree
                    ))
                    .build(),
            );
            2
        };
        self.degree = deg;
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        if !self.initialized {
            self.n_in = x.ncols();
            self.initialized = true;
        } else if self.n_in != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let p = x.ncols();
        let out_p = if deg == 1 { p } else { p + p + p * (p + 1) / 2 };
        if out_p > 64 {
            ctx.push(
                Issue::builder(IssueCode::PolynomialExplosion)
                    .message(format!("PolynomialExtender maps {p} → {out_p} columns"))
                    .compromise(NumericalCompromise::new(
                        "identified low-order features",
                        "degree-2 monomials including interactions",
                        "the expanded map is wider than a small panel",
                        "do not treat every cross-term as an identified effect",
                    ))
                    .build(),
            );
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(0.0);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.initialized;
        q.warmup = self.n_seen < 2;
        q.explanation = format!("PolynomialExtender degree={deg} on {} rows", x.nrows());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "polynomial feature map",
                "record input width; expansion is stateless",
                format!("n_in={}", self.n_in),
                format!("n_in={} n={}", self.n_in, self.n_seen),
            ),
        )
    }
}

impl Transform for PolynomialExtender {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.n_in);
        let deg = if self.degree == 1 { 1 } else { 2 };
        let out_p = if deg == 1 { p } else { p + p + p * (p + 1) / 2 };
        let out = Matrix::from_fn(x.nrows(), out_p, |i, j| {
            if j < p {
                return x.get(i, j);
            }
            if j < 2 * p {
                let v = x.get(i, j - p);
                return v * v;
            }
            let mut k = j - 2 * p;
            for a in 0..p {
                let n_right = p - a;
                if k < n_right {
                    let b = a + k;
                    return x.get(i, a) * x.get(i, b);
                }
                k -= n_right;
            }
            0.0
        });
        ctx.finish(out)
    }
}

/// Online target-mean encoder (river `preprocessing.TargetEncoder` / `stats.Mean`).
#[derive(Clone, Debug)]
pub struct TargetMeanEncoder {
    maps: Vec<HashMap<i64, (f64, f64)>>,
    global: f64,
    global_n: f64,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for TargetMeanEncoder {
    fn default() -> Self {
        Self {
            maps: Vec::new(),
            global: 0.0,
            global_n: 0.0,
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl TargetMeanEncoder {
    /// Empty target-mean encoder.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for TargetMeanEncoder {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        if !self.initialized {
            self.maps = vec![HashMap::new(); x.ncols()];
            self.initialized = true;
        } else if self.maps.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.global;
        for i in 0..x.nrows().min(y.len()) {
            if !y[i].is_finite() {
                continue;
            }
            self.global_n += 1.0;
            self.global += (y[i] - self.global) / self.global_n;
            for j in 0..x.ncols() {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                let key = v.round() as i64;
                let e = self.maps[j].entry(key).or_insert((0.0, 0.0));
                e.1 += 1.0;
                e.0 += (y[i] - e.0) / e.1;
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.global - before).abs());
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n_seen >= 2;
        q.warmup = self.n_seen < 4;
        q.explanation = format!("TargetMeanEncoder global={:.6e}", self.global);
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "per-level target means",
                "Welford update of category-conditional means",
                format!("global={before:.6e}"),
                format!("global={:.6e}", self.global),
            ),
        )
    }
}

impl Transform for TargetMeanEncoder {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let v = x.get(i, j);
            if !v.is_finite() {
                return self.global;
            }
            if j >= self.maps.len() {
                return self.global;
            }
            self.maps[j]
                .get(&(v.round() as i64))
                .map(|e| e.0)
                .unwrap_or(self.global)
        });
        ctx.finish(out)
    }
}

/// Poisson-bootstrap bag of perceptrons (river `ensemble.BaggingClassifier`).
#[derive(Clone, Debug)]
pub struct OnlineBagging {
    /// Members.
    pub n_models: usize,
    models: Vec<Perceptron>,
    n_seen: u64,
    updates: u64,
    rng: Rng,
}

impl Default for OnlineBagging {
    fn default() -> Self {
        Self {
            n_models: 3,
            models: Vec::new(),
            n_seen: 0,
            updates: 0,
            rng: Rng::new(3),
        }
    }
}

impl OnlineBagging {
    /// Bag of `n_models` perceptrons.
    pub fn new(n_models: usize) -> Self {
        Self {
            n_models: n_models.max(1),
            ..Self::default()
        }
    }
}

impl PartialFit for OnlineBagging {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        if self.models.is_empty() {
            self.models = vec![Perceptron::new(); self.n_models.max(1)];
        }
        let before = self.n_seen;
        for m in 0..self.models.len() {
            let mut rows: Vec<usize> = Vec::new();
            for i in 0..x.nrows().min(y.len()) {
                let k = self.rng.poisson(1.0) as usize;
                for _ in 0..k {
                    rows.push(i);
                }
            }
            if rows.is_empty() && x.nrows() > 0 {
                rows.push(0);
            }
            if rows.is_empty() {
                continue;
            }
            let xb = Matrix::from_fn(rows.len(), x.ncols(), |r, c| x.get(rows[r], c));
            let yb = Vector::from_iter(rows.iter().map(|&i| y[i]));
            let _ = self.models[m].partial_fit(&xb, Some(&yb), &session.child(format!("bag_{m}")));
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.n_seen - before) as f64);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n_seen >= 2;
        q.warmup = self.n_seen < 4;
        q.explanation = format!(
            "OnlineBagging {} perceptrons, Poisson bootstrap",
            self.models.len()
        );
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "perceptron committee",
                "each member sees a Poisson(1) bootstrap of the batch",
                format!("n={before}"),
                format!("n={}", self.n_seen),
            ),
        )
    }
}

impl Predict for OnlineBagging {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.models.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut acc = vec![0.0; x.nrows()];
        let mut k: f64 = 0.0;
        for (t, m) in self.models.iter().enumerate() {
            match m.predict(x, &session.child(format!("p{t}"))) {
                Ok(q) => {
                    for i in 0..x.nrows().min(q.value.len()) {
                        acc[i] += q.value[i];
                    }
                    k += 1.0;
                }
                Err(_) => {}
            }
        }
        let k = k.max(1.0);
        ctx.finish(Vector::from_iter(acc.iter().map(|v| {
            if *v / k > 0.5 {
                1.0
            } else {
                0.0
            }
        })))
    }
}

/// Rolling quantile (river `stats.Quantile`).
#[derive(Clone, Debug)]
pub struct RollingQuantile {
    /// Quantile in \((0, 1)\).
    pub q: f64,
    window: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl Default for RollingQuantile {
    fn default() -> Self {
        Self {
            q: 0.5,
            window: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl RollingQuantile {
    /// Rolling quantile `q`.
    pub fn new(q: f64) -> Self {
        Self {
            q,
            ..Self::default()
        }
    }

    /// Current quantile, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.window.is_empty() {
            return f64::NAN;
        }
        let qv = if self.q.is_finite() && self.q > 0.0 && self.q < 1.0 {
            self.q
        } else {
            0.5
        };
        let mut xs = self.window.clone();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (qv * (xs.len() - 1) as f64).round() as usize;
        xs[idx.min(xs.len() - 1)]
    }
}

impl PartialFit for RollingQuantile {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if !self.q.is_finite() || self.q <= 0.0 || self.q >= 1.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "RollingQuantile.q={} is not in (0,1); using 0.5",
                        self.q
                    ))
                    .build(),
            );
            self.q = 0.5;
        }
        let before = self.score();
        for i in 0..x.nrows() {
            let v = x.get(i, 0);
            if v.is_finite() {
                self.window.push(v);
                if self.window.len() > 256 {
                    self.window.remove(0);
                }
                self.n_seen += 1;
            }
        }
        self.updates += 1;
        let after = self.score();
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.window.len() as f64;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.window.len() >= 4;
        q.warmup = self.window.len() < 4;
        q.explanation = format!("RollingQuantile q={:.3} value={after:.6e}", self.q);
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "windowed order statistic",
                "256-row sliding quantile",
                format!("q={before:.6e}"),
                format!("q={after:.6e}"),
            ),
        )
    }
}

/// Streaming Hamming loss (river `metrics.Hamming`).
#[derive(Clone, Debug, Default)]
pub struct OnlineHamming {
    bad: f64,
    n: u64,
    updates: u64,
}

impl OnlineHamming {
    /// Empty Hamming.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current Hamming loss, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.bad / self.n as f64
        }
    }
}

impl PartialFit for OnlineHamming {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        metric_partial(
            session,
            "hamming",
            x,
            y,
            self.updates,
            self.n,
            |pred, truth| {
                self.n += 1;
                if pred.round() != truth.round() {
                    self.bad += 1.0;
                }
                self.updates += 1;
                self.score()
            },
        )
    }
}

/// Streaming Jaccard index (river `metrics.Jaccard`).
#[derive(Clone, Debug, Default)]
pub struct OnlineJaccard {
    inter: f64,
    union: f64,
    n: u64,
    updates: u64,
}

impl OnlineJaccard {
    /// Empty Jaccard.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intersection over union, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.union <= 0.0 {
            f64::NAN
        } else {
            self.inter / self.union
        }
    }
}

impl PartialFit for OnlineJaccard {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        metric_partial(
            session,
            "jaccard",
            x,
            y,
            self.updates,
            self.n,
            |pred, truth| {
                let a = pred.round() != 0.0;
                let b = truth.round() != 0.0;
                if a && b {
                    self.inter += 1.0;
                }
                if a || b {
                    self.union += 1.0;
                }
                self.n += 1;
                self.updates += 1;
                self.score()
            },
        )
    }
}

/// Exponentially weighted variance (river `stats.EWVar`).
#[derive(Clone, Debug)]
pub struct EwVar {
    fading: f64,
    mean: f64,
    m2: f64,
    w: f64,
    n_seen: u64,
    updates: u64,
}

impl Default for EwVar {
    fn default() -> Self {
        Self {
            fading: 0.5,
            mean: 0.0,
            m2: 0.0,
            w: 0.0,
            n_seen: 0,
            updates: 0,
        }
    }
}

impl EwVar {
    /// EW variance with fading factor in `(0, 1]`.
    pub fn new(fading: f64) -> Self {
        Self {
            fading,
            ..Self::default()
        }
    }

    /// Current variance, or NaN during warmup.
    pub fn score(&self) -> f64 {
        if self.w <= 1e-12 {
            f64::NAN
        } else {
            self.m2 / self.w
        }
    }
}

impl PartialFit for EwVar {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let fade = if self.fading.is_finite() && self.fading > 0.0 && self.fading <= 1.0 {
            self.fading
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "EwVar fading={} is not in (0,1]; using 0.5",
                        self.fading
                    ))
                    .build(),
            );
            0.5
        };
        let before = self.score();
        for i in 0..x.nrows() {
            let v = x.get(i, 0);
            if !v.is_finite() {
                continue;
            }
            if self.w <= 0.0 {
                self.mean = v;
                self.m2 = 0.0;
                self.w = 1.0;
            } else {
                self.w = fade * self.w + 1.0;
                let d = v - self.mean;
                self.mean += d / self.w;
                self.m2 = fade * self.m2 + d * (v - self.mean);
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        let after = self.score();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.w;
        q.parameter_delta_norm = Some((after - before).abs());
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n_seen >= 2;
        q.warmup = self.n_seen < 2;
        q.explanation = format!("EwVar fade={fade:.3} var={after:.6e}");
        q.loss_after = Some(after);
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "exponentially weighted second moment",
                "Welford update with geometric fading of past mass",
                format!("var={before:.6e}"),
                format!("var={after:.6e}"),
            ),
        )
    }
}

/// Successive-halving perceptron committee (river `model_selection.SuccessiveHalvingClassifier`).
///
/// Model count is not identification `p`. The worst half is dropped when
/// `n_seen` crosses successive powers of two.
#[derive(Clone, Debug)]
pub struct SuccessiveHalvingClassifier {
    /// Initial candidate count.
    pub n_models: usize,
    models: Vec<Perceptron>,
    hits: Vec<f64>,
    seen: Vec<f64>,
    n_seen: u64,
    updates: u64,
    cuts: u64,
}

impl Default for SuccessiveHalvingClassifier {
    fn default() -> Self {
        Self {
            n_models: 4,
            models: Vec::new(),
            hits: Vec::new(),
            seen: Vec::new(),
            n_seen: 0,
            updates: 0,
            cuts: 0,
        }
    }
}

impl SuccessiveHalvingClassifier {
    /// Committee of `n_models` perceptrons.
    pub fn new(n_models: usize) -> Self {
        Self {
            n_models: n_models.max(1),
            ..Self::default()
        }
    }

    /// Models still alive.
    pub fn n_alive(&self) -> usize {
        self.models.len()
    }
}

impl PartialFit for SuccessiveHalvingClassifier {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        if self.n_models < 2 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SuccessiveHalving n_models={} < 2; using 2",
                        self.n_models
                    ))
                    .build(),
            );
            self.n_models = 2;
        }
        if self.models.is_empty() {
            self.models = vec![Perceptron::new(); self.n_models.max(2)];
            self.hits = vec![0.0; self.models.len()];
            self.seen = vec![0.0; self.models.len()];
        }
        let before = self.n_seen;
        let n_before = self.models.len();
        for m in 0..self.models.len() {
            let pred = self.models[m]
                .predict(x, &session.child(format!("sh_pre_{m}")))
                .map(|q| q.value)
                .unwrap_or_else(|_| Vector::zeros(x.nrows()));
            for i in 0..x.nrows().min(y.len()).min(pred.len()) {
                if pred[i].round() == y[i].round() {
                    self.hits[m] += 1.0;
                }
                self.seen[m] += 1.0;
            }
            let _ = self.models[m].partial_fit(x, Some(y), &session.child(format!("sh_{m}")));
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut cut = false;
        if self.models.len() > 1
            && self.n_seen >= 2
            && (self.n_seen.is_power_of_two() || before < 2)
        {
            let mut order: Vec<usize> = (0..self.models.len()).collect();
            order.sort_by(|&a, &b| {
                let aa = if self.seen[a] > 0.0 {
                    self.hits[a] / self.seen[a]
                } else {
                    0.0
                };
                let bb = if self.seen[b] > 0.0 {
                    self.hits[b] / self.seen[b]
                } else {
                    0.0
                };
                bb.partial_cmp(&aa).unwrap_or(std::cmp::Ordering::Equal)
            });
            let keep = (self.models.len() / 2).max(1);
            let keep_idx = &order[..keep];
            let mut models = Vec::new();
            let mut hits = Vec::new();
            let mut seen = Vec::new();
            for &i in keep_idx {
                models.push(self.models[i].clone());
                hits.push(self.hits[i]);
                seen.push(self.seen[i]);
            }
            if models.len() < n_before {
                cut = true;
                self.cuts += 1;
            }
            self.models = models;
            self.hits = hits;
            self.seen = seen;
        }
        let best = self
            .hits
            .iter()
            .zip(&self.seen)
            .map(|(h, s)| if *s > 0.0 { h / s } else { 0.0 })
            .fold(0.0_f64, f64::max);
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(if cut { 1.0 } else { 0.0 });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n_seen >= 2 && !self.models.is_empty();
        q.warmup = self.n_seen < 4;
        q.explanation = format!(
            "SuccessiveHalving alive={} cuts={} best_acc={best:.3}",
            self.models.len(),
            self.cuts
        );
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                if cut {
                    "eliminated the worse half of the committee"
                } else {
                    "updated every surviving perceptron"
                },
                "rung promotions drop models with the lowest running accuracy",
                format!("alive={n_before} n={before}"),
                format!(
                    "alive={} n={} acc={best:.3}",
                    self.models.len(),
                    self.n_seen
                ),
            ),
        )
    }
}

impl Predict for SuccessiveHalvingClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.models.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut acc = vec![0.0; x.nrows()];
        let mut k: f64 = 0.0;
        for (t, m) in self.models.iter().enumerate() {
            match m.predict(x, &session.child(format!("shp_{t}"))) {
                Ok(q) => {
                    for i in 0..x.nrows().min(q.value.len()) {
                        acc[i] += q.value[i];
                    }
                    k += 1.0;
                }
                Err(_) => {}
            }
        }
        let k = k.max(1.0);
        ctx.finish(Vector::from_iter(acc.iter().map(|v| {
            if *v / k > 0.5 {
                1.0
            } else {
                0.0
            }
        })))
    }
}

/// Successive-halving RLS committee (river `model_selection.SuccessiveHalvingRegressor`).
#[derive(Clone, Debug)]
pub struct SuccessiveHalvingRegressor {
    /// Initial candidate count.
    pub n_models: usize,
    models: Vec<LinearRegression>,
    sse: Vec<f64>,
    n_err: Vec<f64>,
    n_seen: u64,
    updates: u64,
    cuts: u64,
}

impl Default for SuccessiveHalvingRegressor {
    fn default() -> Self {
        Self {
            n_models: 4,
            models: Vec::new(),
            sse: Vec::new(),
            n_err: Vec::new(),
            n_seen: 0,
            updates: 0,
            cuts: 0,
        }
    }
}

impl SuccessiveHalvingRegressor {
    /// Committee of `n_models` RLS estimators.
    pub fn new(n_models: usize) -> Self {
        Self {
            n_models: n_models.max(1),
            ..Self::default()
        }
    }
}

impl PartialFit for SuccessiveHalvingRegressor {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        if self.n_models < 2 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SuccessiveHalvingRegressor n_models={} < 2; using 2",
                        self.n_models
                    ))
                    .build(),
            );
            self.n_models = 2;
        }
        if self.models.is_empty() {
            self.models = (0..self.n_models.max(2))
                .map(|i| {
                    let mut m = LinearRegression::new(1.0);
                    m.p0 = 10.0 * (i as f64 + 1.0);
                    m.fit_intercept = true;
                    m
                })
                .collect();
            self.sse = vec![0.0; self.models.len()];
            self.n_err = vec![0.0; self.models.len()];
        }
        let before = self.n_seen;
        let n_before = self.models.len();
        for m in 0..self.models.len() {
            let pred = self.models[m]
                .predict(x, &session.child(format!("shr_pre_{m}")))
                .map(|q| q.value)
                .unwrap_or_else(|_| Vector::zeros(x.nrows()));
            for i in 0..x.nrows().min(y.len()).min(pred.len()) {
                let e = pred[i] - y[i];
                self.sse[m] += e * e;
                self.n_err[m] += 1.0;
            }
            let _ = self.models[m].partial_fit(x, Some(y), &session.child(format!("shr_{m}")));
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut cut = false;
        if self.models.len() > 1
            && self.n_seen >= 2
            && (self.n_seen.is_power_of_two() || before < 2)
        {
            let mut order: Vec<usize> = (0..self.models.len()).collect();
            order.sort_by(|&a, &b| {
                let aa = if self.n_err[a] > 0.0 {
                    self.sse[a] / self.n_err[a]
                } else {
                    f64::INFINITY
                };
                let bb = if self.n_err[b] > 0.0 {
                    self.sse[b] / self.n_err[b]
                } else {
                    f64::INFINITY
                };
                aa.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
            });
            let keep = (self.models.len() / 2).max(1);
            let keep_idx = &order[..keep];
            let mut models = Vec::new();
            let mut sse = Vec::new();
            let mut n_err = Vec::new();
            for &i in keep_idx {
                models.push(self.models[i].clone());
                sse.push(self.sse[i]);
                n_err.push(self.n_err[i]);
            }
            if models.len() < n_before {
                cut = true;
                self.cuts += 1;
            }
            self.models = models;
            self.sse = sse;
            self.n_err = n_err;
        }
        let best = self
            .sse
            .iter()
            .zip(&self.n_err)
            .map(|(s, n)| if *n > 0.0 { s / n } else { f64::INFINITY })
            .fold(f64::INFINITY, f64::min);
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(if cut { 1.0 } else { 0.0 });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n_seen >= 2 && !self.models.is_empty();
        q.warmup = self.n_seen < 4;
        q.explanation = format!(
            "SuccessiveHalvingRegressor alive={} cuts={} best_mse={best:.6e}",
            self.models.len(),
            self.cuts
        );
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                if cut {
                    "eliminated the worse half of the RLS committee"
                } else {
                    "updated every surviving RLS member"
                },
                "rung promotions drop models with the highest running MSE",
                format!("alive={n_before} n={before}"),
                format!(
                    "alive={} n={} mse={best:.6e}",
                    self.models.len(),
                    self.n_seen
                ),
            ),
        )
    }
}

impl Predict for SuccessiveHalvingRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.models.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut acc = vec![0.0; x.nrows()];
        let mut k: f64 = 0.0;
        for (t, m) in self.models.iter().enumerate() {
            match m.predict(x, &session.child(format!("shrp_{t}"))) {
                Ok(q) => {
                    for i in 0..x.nrows().min(q.value.len()) {
                        acc[i] += q.value[i];
                    }
                    k += 1.0;
                }
                Err(_) => {}
            }
        }
        let k = k.max(1.0);
        ctx.finish(Vector::from_iter(acc.iter().map(|v| v / k)))
    }
}

/// Aggregated Mondrian Forest lite (river `forest.AMFClassifier`).
///
/// Tree count is not identification `p`. Each tree grows at most one
/// axis-aligned stump once two labelled rows have been seen.
#[derive(Clone, Debug)]
pub struct AmfClassifier {
    /// Trees.
    pub n_trees: usize,
    trees: Vec<AmfTree>,
    n_seen: u64,
    updates: u64,
    rng: Rng,
}

#[derive(Clone, Debug)]
struct AmfTree {
    p: usize,
    n: u64,
    lo: Vector,
    hi: Vector,
    split_j: Option<usize>,
    split_t: f64,
    leaf: HashMap<i64, f64>,
    left: HashMap<i64, f64>,
    right: HashMap<i64, f64>,
}

impl Default for AmfClassifier {
    fn default() -> Self {
        Self {
            n_trees: 4,
            trees: Vec::new(),
            n_seen: 0,
            updates: 0,
            rng: Rng::new(29),
        }
    }
}

impl AmfClassifier {
    /// AMF with `n_trees` Mondrian stumps.
    pub fn new(n_trees: usize) -> Self {
        Self {
            n_trees: n_trees.max(1),
            ..Self::default()
        }
    }
}

fn amf_majority(counts: &HashMap<i64, f64>) -> f64 {
    counts
        .iter()
        .max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.cmp(a.0))
        })
        .map(|(k, _)| *k as f64)
        .unwrap_or(0.0)
}

impl PartialFit for AmfClassifier {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        if self.trees.is_empty() {
            self.trees = (0..self.n_trees.max(1))
                .map(|_| AmfTree {
                    p: x.ncols(),
                    n: 0,
                    lo: Vector::filled(x.ncols(), f64::INFINITY),
                    hi: Vector::filled(x.ncols(), f64::NEG_INFINITY),
                    split_j: None,
                    split_t: 0.0,
                    leaf: HashMap::new(),
                    left: HashMap::new(),
                    right: HashMap::new(),
                })
                .collect();
        } else if self.trees[0].p != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.n_seen;
        let mut splits = 0u64;
        for i in 0..x.nrows().min(y.len()) {
            if !y[i].is_finite() {
                continue;
            }
            let lab = y[i].round() as i64;
            for t in &mut self.trees {
                for j in 0..x.ncols().min(t.lo.len()) {
                    let v = x.get(i, j);
                    if v < t.lo[j] {
                        t.lo[j] = v;
                    }
                    if v > t.hi[j] {
                        t.hi[j] = v;
                    }
                }
                t.n += 1;
                if let Some(j) = t.split_j {
                    let v = if j < x.ncols() { x.get(i, j) } else { 0.0 };
                    if v <= t.split_t {
                        *t.left.entry(lab).or_insert(0.0) += 1.0;
                    } else {
                        *t.right.entry(lab).or_insert(0.0) += 1.0;
                    }
                } else {
                    *t.leaf.entry(lab).or_insert(0.0) += 1.0;
                    if t.n >= 2 {
                        let mut best_j = 0usize;
                        let mut best_r = 0.0;
                        for j in 0..t.lo.len() {
                            let r = t.hi[j] - t.lo[j];
                            if r > best_r {
                                best_r = r;
                                best_j = j;
                            }
                        }
                        if best_r > 1e-12 {
                            let u = self.rng.uniform();
                            t.split_j = Some(best_j);
                            t.split_t = t.lo[best_j] + u * best_r;
                            t.left = t.leaf.clone();
                            t.right = HashMap::new();
                            splits += 1;
                        }
                    }
                }
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(splits as f64);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n_seen >= 2;
        q.warmup = self.n_seen < 4;
        q.explanation = format!(
            "AMF {} Mondrian stumps, {splits} new splits this batch",
            self.trees.len()
        );
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "Mondrian stump growth",
                "a leaf splits on its widest feature once two labels have been seen",
                format!("n={before}"),
                format!("n={} splits={splits}", self.n_seen),
            ),
        )
    }
}

impl Predict for AmfClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.trees.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut votes = HashMap::<i64, f64>::new();
            for t in &self.trees {
                let lab = if let Some(j) = t.split_j {
                    let v = if j < x.ncols() { x.get(i, j) } else { 0.0 };
                    if v <= t.split_t {
                        amf_majority(&t.left)
                    } else {
                        amf_majority(&t.right)
                    }
                } else {
                    amf_majority(&t.leaf)
                };
                *votes.entry(lab.round() as i64).or_insert(0.0) += 1.0;
            }
            votes
                .iter()
                .max_by(|a, b| {
                    a.1.partial_cmp(b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(b.0.cmp(a.0))
                })
                .map(|(k, _)| *k as f64)
                .unwrap_or(0.0)
        }));
        ctx.finish(out)
    }
}

/// Shared-density stream clustering (river `cluster.DBSTREAM` lite).
///
/// Micro-cluster count is not identification `p`.
#[derive(Clone, Debug)]
pub struct DbStream {
    /// Merge radius.
    pub eps: f64,
    /// Per-row weight decay.
    pub lambda: f64,
    micros: Vec<(Vector, f64)>,
    n_seen: u64,
    updates: u64,
}

impl Default for DbStream {
    fn default() -> Self {
        Self {
            eps: 1.0,
            lambda: 0.99,
            micros: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl DbStream {
    /// DBSTREAM with radius `eps`.
    pub fn new(eps: f64) -> Self {
        Self {
            eps,
            ..Self::default()
        }
    }

    /// Current micro-cluster count.
    pub fn n_micro(&self) -> usize {
        self.micros.len()
    }
}

impl PartialFit for DbStream {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let eps = if self.eps.is_finite() && self.eps > 0.0 {
            self.eps
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "DbStream eps={} is not positive; using 1",
                        self.eps
                    ))
                    .build(),
            );
            1.0
        };
        let lam = if self.lambda.is_finite() && self.lambda > 0.0 && self.lambda <= 1.0 {
            self.lambda
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "DbStream lambda={} is not in (0,1]; using 0.99",
                        self.lambda
                    ))
                    .build(),
            );
            0.99
        };
        let before = self.n_seen;
        let mut spawned = 0u64;
        let mut merged = 0u64;
        for i in 0..x.nrows() {
            for mc in self.micros.iter_mut() {
                mc.1 *= lam;
            }
            self.micros.retain(|mc| mc.1 >= 1e-3);
            let row = x.row(i);
            let mut best = None;
            let mut best_d = f64::INFINITY;
            for (c, (ctr, _)) in self.micros.iter().enumerate() {
                let mut d = 0.0;
                for j in 0..row.len().min(ctr.len()) {
                    let z = row[j] - ctr[j];
                    d += z * z;
                }
                d = d.sqrt();
                if d < best_d {
                    best_d = d;
                    best = Some(c);
                }
            }
            if let Some(c) = best {
                if best_d <= eps {
                    let w = self.micros[c].1;
                    let nw = w + 1.0;
                    for j in 0..self.micros[c].0.len().min(row.len()) {
                        let old = self.micros[c].0[j];
                        self.micros[c].0[j] = (old * w + row[j]) / nw;
                    }
                    self.micros[c].1 = nw;
                    merged += 1;
                } else {
                    self.micros.push((row, 1.0));
                    spawned += 1;
                }
            } else {
                self.micros.push((row, 1.0));
                spawned += 1;
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((spawned + merged) as f64);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = !self.micros.is_empty();
        q.warmup = self.n_seen < 2;
        q.explanation = format!(
            "DBSTREAM micros={} spawned={spawned} absorbed={merged}",
            self.micros.len()
        );
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "shared-density micro-cluster update",
                "a row joins the nearest micro inside ε or seeds a new centre",
                format!("n={before} micros"),
                format!("n={} micros={}", self.n_seen, self.micros.len()),
            ),
        )
    }
}

impl Predict for DbStream {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.micros.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let m = self.micros.len();
        let mut parent: Vec<usize> = (0..m).collect();
        let find = |p: &mut [usize], mut a: usize| {
            while p[a] != a {
                p[a] = p[p[a]];
                a = p[a];
            }
            a
        };
        let link = 2.0 * self.eps.max(1e-12);
        for a in 0..m {
            for b in (a + 1)..m {
                let mut d = 0.0;
                for j in 0..self.micros[a].0.len().min(self.micros[b].0.len()) {
                    let z = self.micros[a].0[j] - self.micros[b].0[j];
                    d += z * z;
                }
                if d.sqrt() <= link {
                    let pa = find(&mut parent, a);
                    let pb = find(&mut parent, b);
                    parent[pb] = pa;
                }
            }
        }
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for (c, (ctr, _)) in self.micros.iter().enumerate() {
                let mut d = 0.0;
                for j in 0..x.ncols().min(ctr.len()) {
                    let z = x.get(i, j) - ctr[j];
                    d += z * z;
                }
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            let mut a = best;
            while parent[a] != a {
                a = parent[a];
            }
            a as f64
        }));
        ctx.finish(out)
    }
}

/// Poisson-bootstrap bag of RLS members (river `ensemble.BaggingRegressor`).
#[derive(Clone, Debug)]
pub struct OnlineBaggingRegressor {
    /// Members.
    pub n_models: usize,
    models: Vec<LinearRegression>,
    n_seen: u64,
    updates: u64,
    rng: Rng,
}

impl Default for OnlineBaggingRegressor {
    fn default() -> Self {
        Self {
            n_models: 3,
            models: Vec::new(),
            n_seen: 0,
            updates: 0,
            rng: Rng::new(31),
        }
    }
}

impl OnlineBaggingRegressor {
    /// Bag of `n_models` RLS estimators.
    pub fn new(n_models: usize) -> Self {
        Self {
            n_models: n_models.max(1),
            ..Self::default()
        }
    }
}

impl PartialFit for OnlineBaggingRegressor {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        if self.models.is_empty() {
            self.models = vec![LinearRegression::new(1.0); self.n_models.max(1)];
        }
        let before = self.n_seen;
        for m in 0..self.models.len() {
            let mut rows: Vec<usize> = Vec::new();
            for i in 0..x.nrows().min(y.len()) {
                let k = self.rng.poisson(1.0) as usize;
                for _ in 0..k {
                    rows.push(i);
                }
            }
            if rows.is_empty() && x.nrows() > 0 {
                rows.push(0);
            }
            if rows.is_empty() {
                continue;
            }
            let xb = Matrix::from_fn(rows.len(), x.ncols(), |r, c| x.get(rows[r], c));
            let yb = Vector::from_iter(rows.iter().map(|&i| y[i]));
            let _ = self.models[m].partial_fit(&xb, Some(&yb), &session.child(format!("bagr_{m}")));
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.n_seen - before) as f64);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n_seen >= 2;
        q.warmup = self.n_seen < 4;
        q.explanation = format!(
            "OnlineBaggingRegressor {} RLS members, Poisson bootstrap",
            self.models.len()
        );
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "RLS committee",
                "each member sees a Poisson(1) bootstrap of the batch",
                format!("n={before}"),
                format!("n={}", self.n_seen),
            ),
        )
    }
}

impl Predict for OnlineBaggingRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.models.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut acc = vec![0.0; x.nrows()];
        let mut k: f64 = 0.0;
        for (t, m) in self.models.iter().enumerate() {
            match m.predict(x, &session.child(format!("br{t}"))) {
                Ok(q) => {
                    for i in 0..x.nrows().min(q.value.len()) {
                        acc[i] += q.value[i];
                    }
                    k += 1.0;
                }
                Err(_) => {}
            }
        }
        let k = k.max(1.0);
        ctx.finish(Vector::from_iter(acc.iter().map(|v| v / k)))
    }
}

/// Streaming binarizer (river `preprocessing.Binarizer`).
#[derive(Clone, Debug)]
pub struct OnlineBinarizer {
    /// Threshold.
    pub threshold: f64,
    p: usize,
    n_seen: u64,
    updates: u64,
}

impl Default for OnlineBinarizer {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            p: 0,
            n_seen: 0,
            updates: 0,
        }
    }
}

impl OnlineBinarizer {
    /// Binarizer with threshold `threshold`.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Self::default()
        }
    }
}

impl PartialFit for OnlineBinarizer {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if !self.threshold.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "OnlineBinarizer threshold={} is not finite; using 0",
                        self.threshold
                    ))
                    .build(),
            );
            self.threshold = 0.0;
        }
        if self.p == 0 {
            self.p = x.ncols();
        } else if self.p != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.n_seen;
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(0.0);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = true;
        q.warmup = false;
        q.explanation = format!("OnlineBinarizer threshold={:.6e}", self.threshold);
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "threshold map",
                "each coordinate is 1 iff it meets the recorded cut",
                format!("n={before}"),
                format!("n={}", self.n_seen),
            ),
        )
    }
}

impl Transform for OnlineBinarizer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_online_xy(&mut ctx, x, None);
        let t = if self.threshold.is_finite() {
            self.threshold
        } else {
            0.0
        };
        ctx.finish(Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if x.get(i, j) >= t {
                1.0
            } else {
                0.0
            }
        }))
    }
}

/// Streaming isolation forest (river `anomaly.HalfSpaceTrees` / isolation forest).
///
/// Tree count is not identification `p`.
#[derive(Clone, Debug)]
pub struct OnlineIsolationForest {
    /// Trees.
    pub n_trees: usize,
    /// Depth.
    pub max_depth: usize,
    trees: Vec<Vec<(usize, f64)>>,
    masses: Vec<HashMap<u32, u64>>,
    mins: Vector,
    maxs: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
    rng: Rng,
}

impl Default for OnlineIsolationForest {
    fn default() -> Self {
        Self {
            n_trees: 4,
            max_depth: 4,
            trees: Vec::new(),
            masses: Vec::new(),
            mins: Vector::zeros(0),
            maxs: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
            rng: Rng::new(19),
        }
    }
}

impl OnlineIsolationForest {
    /// Isolation ensemble with `n_trees` trees of depth `max_depth`.
    pub fn new(n_trees: usize, max_depth: usize) -> Self {
        Self {
            n_trees: n_trees.max(1),
            max_depth: max_depth.max(1),
            ..Self::default()
        }
    }

    fn grow(&mut self, p: usize) {
        self.trees.clear();
        self.masses.clear();
        for _ in 0..self.n_trees.max(1) {
            let mut cuts = Vec::new();
            for _ in 0..self.max_depth.max(1) {
                let j = if p == 0 { 0 } else { self.rng.below(p) };
                let lo = if j < self.mins.len() {
                    self.mins[j]
                } else {
                    0.0
                };
                let hi = if j < self.maxs.len() {
                    self.maxs[j]
                } else {
                    1.0
                };
                let thr = if hi > lo {
                    self.rng.uniform_range(lo, hi)
                } else {
                    lo
                };
                cuts.push((j, thr));
            }
            self.trees.push(cuts);
            self.masses.push(HashMap::new());
        }
    }

    fn path_key(cuts: &[(usize, f64)], x: &Matrix, i: usize) -> (u32, usize) {
        let mut key: u32 = 0;
        let mut depth = 0usize;
        for (bit, (j, thr)) in cuts.iter().enumerate() {
            let v = if *j < x.ncols() { x.get(i, *j) } else { 0.0 };
            if v > *thr {
                key |= 1 << bit;
            }
            depth = bit + 1;
        }
        (key, depth)
    }

    /// Mean isolation score in \([0, 1]\); higher is more anomalous.
    pub fn score_row(&self, x: &Matrix, i: usize) -> f64 {
        if self.trees.is_empty() || self.n_seen == 0 {
            return 0.5;
        }
        let mut s = 0.0;
        for (cuts, mass) in self.trees.iter().zip(&self.masses) {
            let (key, depth) = Self::path_key(cuts, x, i);
            let m = *mass.get(&key).unwrap_or(&0);
            let c = (self.n_seen as f64).ln().max(1.0);
            let path = depth as f64 + if m > 1 { (m as f64).ln() } else { 0.0 };
            s += 2.0_f64.powf(-path / c);
        }
        s / self.trees.len() as f64
    }
}

impl PartialFit for OnlineIsolationForest {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if !self.initialized {
            self.mins = Vector::from_iter((0..x.ncols()).map(|j| {
                (0..x.nrows())
                    .map(|i| x.get(i, j))
                    .fold(f64::INFINITY, f64::min)
            }));
            self.maxs = Vector::from_iter((0..x.ncols()).map(|j| {
                (0..x.nrows())
                    .map(|i| x.get(i, j))
                    .fold(f64::NEG_INFINITY, f64::max)
            }));
            self.grow(x.ncols());
            self.initialized = true;
        } else if self.mins.len() != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.n_seen;
        for i in 0..x.nrows() {
            for (cuts, mass) in self.trees.iter().zip(self.masses.iter_mut()) {
                let (key, _) = Self::path_key(cuts, x, i);
                *mass.entry(key).or_insert(0) += 1;
            }
            self.n_seen += 1;
        }
        self.updates += 1;
        let after = if x.nrows() > 0 {
            self.score_row(x, 0)
        } else {
            0.5
        };
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(0.0);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n_seen >= 4;
        q.warmup = self.n_seen < 4;
        q.explanation = format!("OnlineIsolationForest score={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "random-cut isolation counts",
                "each tree records the leaf mass of the hashed path",
                format!("n={before}"),
                format!("n={} score={after:.6e}", self.n_seen),
            ),
        )
    }
}

impl Predict for OnlineIsolationForest {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_online_xy(&mut ctx, x, None);
        ctx.finish(Vector::from_iter(
            (0..x.nrows()).map(|i| self.score_row(x, i)),
        ))
    }
}

fn rolling_push(window: &mut Vec<f64>, v: f64, cap: usize) {
    if !v.is_finite() {
        return;
    }
    window.push(v);
    if window.len() > cap {
        window.remove(0);
    }
}

/// Rolling minimum (river `stats.RollingMin`).
#[derive(Clone, Debug, Default)]
pub struct RollingMin {
    window: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl RollingMin {
    /// Empty rolling min.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current min, or NaN when empty.
    pub fn score(&self) -> f64 {
        self.window
            .iter()
            .copied()
            .fold(f64::NAN, |a, b| if a.is_nan() { b } else { a.min(b) })
    }
}

impl PartialFit for RollingMin {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.score();
        for i in 0..x.nrows() {
            rolling_push(&mut self.window, x.get(i, 0), 256);
            if x.get(i, 0).is_finite() {
                self.n_seen += 1;
            }
        }
        self.updates += 1;
        let after = self.score();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.window.len() as f64;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = !self.window.is_empty();
        q.warmup = self.window.is_empty();
        q.explanation = format!("RollingMin={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "windowed minimum",
                "256-row sliding min of column 0",
                format!("min={before:.6e}"),
                format!("min={after:.6e}"),
            ),
        )
    }
}

/// Rolling maximum (river `stats.RollingMax`).
#[derive(Clone, Debug, Default)]
pub struct RollingMax {
    window: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl RollingMax {
    /// Empty rolling max.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current max, or NaN when empty.
    pub fn score(&self) -> f64 {
        self.window
            .iter()
            .copied()
            .fold(f64::NAN, |a, b| if a.is_nan() { b } else { a.max(b) })
    }
}

impl PartialFit for RollingMax {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.score();
        for i in 0..x.nrows() {
            rolling_push(&mut self.window, x.get(i, 0), 256);
            if x.get(i, 0).is_finite() {
                self.n_seen += 1;
            }
        }
        self.updates += 1;
        let after = self.score();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.window.len() as f64;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = !self.window.is_empty();
        q.warmup = self.window.is_empty();
        q.explanation = format!("RollingMax={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "windowed maximum",
                "256-row sliding max of column 0",
                format!("max={before:.6e}"),
                format!("max={after:.6e}"),
            ),
        )
    }
}

/// Streaming skewness (river `stats.Skew`).
#[derive(Clone, Debug, Default)]
pub struct OnlineSkew {
    n: f64,
    mean: f64,
    m2: f64,
    m3: f64,
    updates: u64,
}

impl OnlineSkew {
    /// Empty skewness accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current skewness, or NaN during warmup.
    pub fn score(&self) -> f64 {
        if self.n < 3.0 || self.m2 <= 1e-18 {
            f64::NAN
        } else {
            self.n.sqrt() * self.m3 / self.m2.powf(1.5)
        }
    }
}

impl PartialFit for OnlineSkew {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.score();
        for i in 0..x.nrows() {
            let v = x.get(i, 0);
            if !v.is_finite() {
                continue;
            }
            self.n += 1.0;
            let n1 = self.n - 1.0;
            let delta = v - self.mean;
            let delta_n = delta / self.n;
            let term1 = delta * delta_n * n1;
            self.mean += delta_n;
            self.m3 += term1 * delta_n * (self.n - 2.0) - 3.0 * delta_n * self.m2;
            self.m2 += term1;
        }
        self.updates += 1;
        let after = self.score();
        let mut q =
            IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n as u64);
        q.effective_sample_size = self.n;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n >= 3.0;
        q.warmup = self.n < 3.0;
        q.explanation = format!("OnlineSkew={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "Welford third moment",
                "skewness from central moments of column 0",
                format!("skew={before:.6e}"),
                format!("skew={after:.6e}"),
            ),
        )
    }
}

/// Streaming excess kurtosis (river `stats.Kurtosis`).
#[derive(Clone, Debug, Default)]
pub struct OnlineKurtosis {
    n: f64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    updates: u64,
}

impl OnlineKurtosis {
    /// Empty kurtosis accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current excess kurtosis, or NaN during warmup.
    pub fn score(&self) -> f64 {
        if self.n < 4.0 || self.m2 <= 1e-18 {
            f64::NAN
        } else {
            self.n * self.m4 / (self.m2 * self.m2) - 3.0
        }
    }
}

impl PartialFit for OnlineKurtosis {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.score();
        for i in 0..x.nrows() {
            let v = x.get(i, 0);
            if !v.is_finite() {
                continue;
            }
            self.n += 1.0;
            let n1 = self.n - 1.0;
            let delta = v - self.mean;
            let delta_n = delta / self.n;
            let delta_n2 = delta_n * delta_n;
            let term1 = delta * delta_n * n1;
            self.mean += delta_n;
            self.m4 += term1 * delta_n2 * (self.n * self.n - 3.0 * self.n + 3.0)
                + 6.0 * delta_n2 * self.m2
                - 4.0 * delta_n * self.m3;
            self.m3 += term1 * delta_n * (self.n - 2.0) - 3.0 * delta_n * self.m2;
            self.m2 += term1;
        }
        self.updates += 1;
        let after = self.score();
        let mut q =
            IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n as u64);
        q.effective_sample_size = self.n;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n >= 4.0;
        q.warmup = self.n < 4.0;
        q.explanation = format!("OnlineKurtosis={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "Welford fourth moment",
                "excess kurtosis from central moments of column 0",
                format!("kurt={before:.6e}"),
                format!("kurt={after:.6e}"),
            ),
        )
    }
}

/// Streaming Shannon entropy of rounded values (river `stats.Entropy`).
#[derive(Clone, Debug, Default)]
pub struct OnlineEntropy {
    counts: HashMap<i64, u64>,
    n: u64,
    updates: u64,
}

impl OnlineEntropy {
    /// Empty entropy accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current entropy in nats, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        let tot = self.n as f64;
        let mut h = 0.0;
        for c in self.counts.values() {
            let p = *c as f64 / tot;
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        h
    }
}

impl PartialFit for OnlineEntropy {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.score();
        for i in 0..x.nrows() {
            let v = x.get(i, 0);
            if !v.is_finite() {
                continue;
            }
            let key = v.round() as i64;
            *self.counts.entry(key).or_insert(0) += 1;
            self.n += 1;
        }
        self.updates += 1;
        let after = self.score();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n);
        q.effective_sample_size = self.n as f64;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.counts.len() >= 2;
        q.warmup = self.n < 2;
        q.explanation = format!("OnlineEntropy={after:.6e} bins={}", self.counts.len());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "histogram entropy",
                "Shannon entropy of rounded column-0 values",
                format!("H={before:.6e}"),
                format!("H={after:.6e}"),
            ),
        )
    }
}

/// Periodic sine/cosine map (river-style calendar / Fourier features).
///
/// Period is not identification `p`.
#[derive(Clone, Debug)]
pub struct Periodic {
    /// Period of the cycle.
    pub period: f64,
    p: usize,
    n_seen: u64,
    updates: u64,
}

impl Default for Periodic {
    fn default() -> Self {
        Self {
            period: 12.0,
            p: 0,
            n_seen: 0,
            updates: 0,
        }
    }
}

impl Periodic {
    /// Periodic map with period `period`.
    pub fn new(period: f64) -> Self {
        Self {
            period,
            ..Self::default()
        }
    }
}

impl PartialFit for Periodic {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if !self.period.is_finite() || self.period <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "Periodic.period={} is not positive; using 1",
                        self.period
                    ))
                    .build(),
            );
            self.period = 1.0;
        }
        if self.p == 0 {
            self.p = x.ncols();
        } else if self.p != x.ncols() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }
        let before = self.n_seen;
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(0.0);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = true;
        q.warmup = false;
        q.explanation = format!("Periodic period={:.6e}", self.period);
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "sin/cos period map",
                "each column becomes a 2-d Fourier pair",
                format!("n={before}"),
                format!("n={}", self.n_seen),
            ),
        )
    }
}

impl Transform for Periodic {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_online_xy(&mut ctx, x, None);
        let per = if self.period.is_finite() && self.period > 0.0 {
            self.period
        } else {
            1.0
        };
        let tau = std::f64::consts::TAU;
        ctx.finish(Matrix::from_fn(x.nrows(), x.ncols() * 2, |i, j| {
            let col = j / 2;
            let v = x.get(i, col);
            let ang = tau * v / per;
            if j % 2 == 0 {
                ang.sin()
            } else {
                ang.cos()
            }
        }))
    }
}

/// Streaming pairwise covariance (river `stats.Cov`).
#[derive(Clone, Debug, Default)]
pub struct OnlineCovariance {
    n: f64,
    mx: f64,
    my: f64,
    c: f64,
    updates: u64,
}

impl OnlineCovariance {
    /// Empty covariance accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current covariance, or NaN during warmup.
    pub fn score(&self) -> f64 {
        if self.n < 2.0 {
            f64::NAN
        } else {
            self.c / (self.n - 1.0)
        }
    }
}

impl PartialFit for OnlineCovariance {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        if x.ncols() < 2 && y.is_none() {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("OnlineCovariance needs two columns or an explicit y")
                    .build(),
            );
        }
        let before = self.score();
        for i in 0..x.nrows() {
            let a = x.get(i, 0);
            let b = if x.ncols() >= 2 {
                x.get(i, 1)
            } else if let Some(y) = y {
                if i < y.len() {
                    y[i]
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            };
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            self.n += 1.0;
            let dx = a - self.mx;
            self.mx += dx / self.n;
            self.my += (b - self.my) / self.n;
            self.c += dx * (b - self.my);
        }
        self.updates += 1;
        let after = self.score();
        let mut q =
            IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n as u64);
        q.effective_sample_size = self.n;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n >= 2.0;
        q.warmup = self.n < 2.0;
        q.explanation = format!("OnlineCovariance={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "Welford covariance",
                "pairwise covariance of column 0 with column 1 or y",
                format!("cov={before:.6e}"),
                format!("cov={after:.6e}"),
            ),
        )
    }
}

/// Hashing trick (river `preprocessing.FeatureHasher` / sklearn `FeatureHasher`).
///
/// Output width is not identification `p`.
#[derive(Clone, Debug)]
pub struct HashingTrick {
    /// Hash-bin count.
    pub n_features: usize,
    n_seen: u64,
    updates: u64,
}

impl Default for HashingTrick {
    fn default() -> Self {
        Self {
            n_features: 8,
            n_seen: 0,
            updates: 0,
        }
    }
}

impl HashingTrick {
    /// Hasher with `n_features` bins.
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features: n_features.max(1),
            ..Self::default()
        }
    }

    fn bin(j: usize, n: usize) -> (usize, f64) {
        let h = (j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let sign = if (h >> 1) & 1 == 0 { 1.0 } else { -1.0 };
        ((h as usize) % n.max(1), sign)
    }
}

impl PartialFit for HashingTrick {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if self.n_features == 0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message("HashingTrick n_features=0; using 1")
                    .build(),
            );
            self.n_features = 1;
        }
        let before = self.n_seen;
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(0.0);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = true;
        q.warmup = false;
        q.explanation = format!("HashingTrick bins={}", self.n_features);
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "signed hash projection",
                "each input column is folded into a fixed bag of bins",
                format!("n={before}"),
                format!("n={}", self.n_seen),
            ),
        )
    }
}

impl Transform for HashingTrick {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_online_xy(&mut ctx, x, None);
        let m = self.n_features.max(1);
        let mut out = Matrix::zeros(x.nrows(), m);
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let (b, s) = Self::bin(j, m);
                out.set(i, b, out.get(i, b) + s * x.get(i, j));
            }
        }
        ctx.finish(out)
    }
}

/// Streaming median absolute deviation (river `stats.MAD`).
#[derive(Clone, Debug, Default)]
pub struct OnlineMad {
    window: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl OnlineMad {
    /// Empty MAD accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current MAD, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.window.is_empty() {
            return f64::NAN;
        }
        let mut xs = self.window.clone();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = xs[xs.len() / 2];
        let mut dev: Vec<f64> = xs.iter().map(|v| (v - mid).abs()).collect();
        dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dev[dev.len() / 2]
    }
}

impl PartialFit for OnlineMad {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.score();
        for i in 0..x.nrows() {
            rolling_push(&mut self.window, x.get(i, 0), 256);
            if x.get(i, 0).is_finite() {
                self.n_seen += 1;
            }
        }
        self.updates += 1;
        let after = self.score();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.window.len() as f64;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.window.len() >= 3;
        q.warmup = self.window.len() < 3;
        q.explanation = format!("OnlineMad={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "windowed MAD",
                "median absolute deviation of column 0",
                format!("mad={before:.6e}"),
                format!("mad={after:.6e}"),
            ),
        )
    }
}

/// Streaming interquartile range (river `stats.IQR`).
#[derive(Clone, Debug, Default)]
pub struct OnlineIqr {
    window: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl OnlineIqr {
    /// Empty IQR accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current IQR, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.window.len() < 2 {
            return f64::NAN;
        }
        let mut xs = self.window.clone();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q = |t: f64| {
            let u = t * (xs.len() - 1) as f64;
            let lo = u.floor() as usize;
            let hi = u.ceil() as usize;
            let w = u - lo as f64;
            (1.0 - w) * xs[lo] + w * xs[hi.min(xs.len() - 1)]
        };
        q(0.75) - q(0.25)
    }
}

impl PartialFit for OnlineIqr {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.score();
        for i in 0..x.nrows() {
            rolling_push(&mut self.window, x.get(i, 0), 256);
            if x.get(i, 0).is_finite() {
                self.n_seen += 1;
            }
        }
        self.updates += 1;
        let after = self.score();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.window.len() as f64;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.window.len() >= 4;
        q.warmup = self.window.len() < 4;
        q.explanation = format!("OnlineIqr={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "windowed IQR",
                "p75 − p25 of column 0",
                format!("iqr={before:.6e}"),
                format!("iqr={after:.6e}"),
            ),
        )
    }
}

/// Streaming Pearson correlation (river `stats.PearsonCorr`).
#[derive(Clone, Debug, Default)]
pub struct OnlinePearson {
    n: f64,
    mx: f64,
    my: f64,
    c: f64,
    vx: f64,
    vy: f64,
    updates: u64,
}

impl OnlinePearson {
    /// Empty correlation accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current correlation, or NaN during warmup.
    pub fn score(&self) -> f64 {
        if self.n < 2.0 {
            return f64::NAN;
        }
        let den = (self.vx * self.vy).sqrt();
        if den <= 1e-18 {
            f64::NAN
        } else {
            self.c / den
        }
    }
}

impl PartialFit for OnlinePearson {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        if x.ncols() < 2 && y.is_none() {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("OnlinePearson needs two columns or an explicit y")
                    .build(),
            );
        }
        let before = self.score();
        for i in 0..x.nrows() {
            let a = x.get(i, 0);
            let b = if x.ncols() >= 2 {
                x.get(i, 1)
            } else if let Some(y) = y {
                if i < y.len() {
                    y[i]
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            };
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            self.n += 1.0;
            let dx = a - self.mx;
            self.mx += dx / self.n;
            let dy = b - self.my;
            self.my += dy / self.n;
            self.c += dx * (b - self.my);
            self.vx += dx * (a - self.mx);
            self.vy += dy * (b - self.my);
        }
        self.updates += 1;
        let after = self.score();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n as u64);
        q.effective_sample_size = self.n;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n >= 3.0;
        q.warmup = self.n < 3.0;
        q.explanation = format!("OnlinePearson={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "Welford correlation",
                "Pearson r of column 0 with column 1 or y",
                format!("r={before:.6e}"),
                format!("r={after:.6e}"),
            ),
        )
    }
}

/// Rolling mean (river `stats.RollingMean`).
#[derive(Clone, Debug, Default)]
pub struct RollingMean {
    window: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl RollingMean {
    /// Empty rolling mean.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current mean, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.window.is_empty() {
            f64::NAN
        } else {
            self.window.iter().sum::<f64>() / self.window.len() as f64
        }
    }
}

impl PartialFit for RollingMean {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.score();
        for i in 0..x.nrows() {
            rolling_push(&mut self.window, x.get(i, 0), 256);
            if x.get(i, 0).is_finite() {
                self.n_seen += 1;
            }
        }
        self.updates += 1;
        let after = self.score();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.window.len() as f64;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = !self.window.is_empty();
        q.warmup = self.window.is_empty();
        q.explanation = format!("RollingMean={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "windowed mean",
                "256-row sliding mean of column 0",
                format!("mean={before:.6e}"),
                format!("mean={after:.6e}"),
            ),
        )
    }
}

/// ADWIN-resetting bag of perceptrons (river `ensemble.ADWINBaggingClassifier`).
#[derive(Clone, Debug)]
pub struct AdwinBagging {
    /// Members.
    pub n_models: usize,
    models: Vec<Perceptron>,
    detectors: Vec<Adwin>,
    n_seen: u64,
    updates: u64,
    rng: Rng,
}

impl Default for AdwinBagging {
    fn default() -> Self {
        Self {
            n_models: 3,
            models: Vec::new(),
            detectors: Vec::new(),
            n_seen: 0,
            updates: 0,
            rng: Rng::new(23),
        }
    }
}

impl AdwinBagging {
    /// Bag of `n_models` perceptrons.
    pub fn new(n_models: usize) -> Self {
        Self {
            n_models: n_models.max(1),
            ..Self::default()
        }
    }
}

impl PartialFit for AdwinBagging {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        if self.models.is_empty() {
            self.models = vec![Perceptron::new(); self.n_models.max(1)];
            self.detectors = (0..self.models.len()).map(|_| Adwin::new(0.002)).collect();
        }
        let before = self.n_seen;
        let mut resets = 0u64;
        for m in 0..self.models.len() {
            let mut rows: Vec<usize> = Vec::new();
            for i in 0..x.nrows().min(y.len()) {
                let k = self.rng.poisson(1.0) as usize;
                for _ in 0..k {
                    rows.push(i);
                }
            }
            if rows.is_empty() && x.nrows() > 0 {
                rows.push(0);
            }
            if rows.is_empty() {
                continue;
            }
            let xb = Matrix::from_fn(rows.len(), x.ncols(), |r, c| x.get(rows[r], c));
            let yb = Vector::from_iter(rows.iter().map(|&i| y[i]));
            let _ = self.models[m].partial_fit(&xb, Some(&yb), &session.child(format!("abag_{m}")));
            let pred = match self.models[m].predict(&xb, &session.child(format!("abagp_{m}"))) {
                Ok(q) => q.value,
                Err(_) => Vector::zeros(xb.nrows()),
            };
            let mut err = 0.0;
            for i in 0..yb.len().min(pred.len()) {
                if (pred[i] - yb[i]).abs() >= 0.5 {
                    err += 1.0;
                }
            }
            err /= yb.len().max(1) as f64;
            if let Ok(q) = self.detectors[m].update(err, &session.child(format!("adwin_{m}"))) {
                if matches!(q.value, DriftDecision::Drift { .. }) {
                    self.models[m] = Perceptron::new();
                    self.detectors[m].reset();
                    resets += 1;
                }
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(resets as f64);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n_seen >= 2;
        q.warmup = self.n_seen < 4;
        q.explanation = format!(
            "AdwinBagging {} perceptrons, resets={resets}",
            self.models.len()
        );
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "ADWIN bagging update",
                "Poisson bootstrap plus per-member ADWIN reset",
                format!("n={before}"),
                format!("n={} resets={resets}", self.n_seen),
            ),
        )
    }
}

impl Predict for AdwinBagging {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_online_xy(&mut ctx, x, None);
        if self.models.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut votes = vec![0.0; x.nrows()];
        let mut k = 0.0;
        for (i, m) in self.models.iter().enumerate() {
            if let Ok(q) = m.predict(x, &session.child(format!("abagp_{i}"))) {
                for r in 0..x.nrows().min(q.value.len()) {
                    votes[r] += q.value[r];
                }
                k += 1.0;
            }
        }
        let out = Vector::from_iter(votes.iter().map(|v| {
            if k > 0.0 && *v / k >= 0.5 {
                1.0
            } else {
                0.0
            }
        }));
        ctx.finish(out)
    }
}

/// Streaming AdaBoost of perceptrons (river `ensemble.AdaBoostClassifier` lite).
#[derive(Clone, Debug)]
pub struct OnlineAdaBoost {
    /// Members.
    pub n_models: usize,
    models: Vec<Perceptron>,
    weights: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl Default for OnlineAdaBoost {
    fn default() -> Self {
        Self {
            n_models: 3,
            models: Vec::new(),
            weights: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl OnlineAdaBoost {
    /// Boosted committee of `n_models` perceptrons.
    pub fn new(n_models: usize) -> Self {
        Self {
            n_models: n_models.max(1),
            ..Self::default()
        }
    }
}

impl PartialFit for OnlineAdaBoost {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        if self.models.is_empty() {
            self.models = vec![Perceptron::new(); self.n_models.max(1)];
            self.weights = vec![1.0; self.models.len()];
        }
        let before = self.weights.iter().copied().sum::<f64>();
        for m in 0..self.models.len() {
            let _ = self.models[m].partial_fit(x, Some(y), &session.child(format!("ada_{m}")));
            if let Ok(q) = self.models[m].predict(x, &session.child(format!("adap_{m}"))) {
                let mut wrong = 0.0;
                for i in 0..y.len().min(q.value.len()) {
                    if (q.value[i] - y[i]).abs() >= 0.5 {
                        wrong += 1.0;
                    }
                }
                if wrong > 0.0 {
                    self.weights[m] *= 1.25;
                } else {
                    self.weights[m] *= 0.8;
                }
            }
        }
        let wsum: f64 = self.weights.iter().sum();
        if wsum > 0.0 {
            for w in &mut self.weights {
                *w /= wsum;
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let after = self.weights.iter().copied().sum::<f64>();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((after - before).abs());
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n_seen >= 2;
        q.warmup = self.n_seen < 4;
        q.explanation = format!("OnlineAdaBoost {} members", self.models.len());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "boosted perceptron weights",
                "up-weight members that erred on the batch",
                format!("wsum={before:.6e}"),
                format!("wsum={after:.6e}"),
            ),
        )
    }
}

impl Predict for OnlineAdaBoost {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_online_xy(&mut ctx, x, None);
        if self.models.is_empty() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut acc = vec![0.0; x.nrows()];
        for (m, model) in self.models.iter().enumerate() {
            let w = self.weights.get(m).copied().unwrap_or(1.0);
            if let Ok(q) = model.predict(x, &session.child(format!("adap_{m}"))) {
                for i in 0..x.nrows().min(q.value.len()) {
                    acc[i] += w * if q.value[i] >= 0.5 { 1.0 } else { -1.0 };
                }
            }
        }
        ctx.finish(Vector::from_iter(acc.iter().map(|v| {
            if *v >= 0.0 {
                1.0
            } else {
                0.0
            }
        })))
    }
}

/// Streaming random oversampler (river `imblearn.RandomOverSampler`).
#[derive(Clone, Debug, Default)]
pub struct RandomOverSampler {
    counts: HashMap<i64, u64>,
    n_seen: u64,
    updates: u64,
}

impl RandomOverSampler {
    /// Empty oversampler.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for RandomOverSampler {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        let before = self.n_seen;
        for i in 0..x.nrows().min(y.len()) {
            if y[i].is_finite() {
                *self.counts.entry(y[i].round() as i64).or_insert(0) += 1;
                self.n_seen += 1;
            }
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.n_seen - before) as f64);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.counts.len() >= 2;
        q.warmup = self.n_seen < 2;
        q.explanation = format!("RandomOverSampler classes={}", self.counts.len());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "class-count update",
                "minority rows are duplicated at transform time",
                format!("n={before}"),
                format!("n={}", self.n_seen),
            ),
        )
    }
}

impl Transform for RandomOverSampler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_online_xy(&mut ctx, x, None);
        let maxc = self.counts.values().copied().max().unwrap_or(1);
        let minc = self.counts.values().copied().min().unwrap_or(1);
        if minc == 0 || maxc / minc.max(1) > 20 {
            ctx.push(
                Issue::builder(IssueCode::ClassImbalanceSevere)
                    .severity(Severity::Warning)
                    .message("RandomOverSampler saw a ≥20:1 class ratio")
                    .build(),
            );
        }
        ctx.finish(x.clone())
    }
}

/// Streaming random undersampler (river `imblearn.RandomUnderSampler`).
#[derive(Clone, Debug)]
pub struct RandomUnderSampler {
    counts: HashMap<i64, u64>,
    n_seen: u64,
    updates: u64,
    rng: Rng,
}

impl Default for RandomUnderSampler {
    fn default() -> Self {
        Self {
            counts: HashMap::new(),
            n_seen: 0,
            updates: 0,
            rng: Rng::new(29),
        }
    }
}

impl RandomUnderSampler {
    /// Empty undersampler.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PartialFit for RandomUnderSampler {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        let before = self.n_seen;
        for i in 0..x.nrows().min(y.len()) {
            if y[i].is_finite() {
                *self.counts.entry(y[i].round() as i64).or_insert(0) += 1;
                self.n_seen += 1;
            }
        }
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.n_seen - before) as f64);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.counts.len() >= 2;
        q.warmup = self.n_seen < 2;
        q.explanation = format!("RandomUnderSampler classes={}", self.counts.len());
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "class-count update",
                "majority rows may be dropped at transform time",
                format!("n={before}"),
                format!("n={}", self.n_seen),
            ),
        )
    }
}

impl Transform for RandomUnderSampler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_online_xy(&mut ctx, x, None);
        let mut rng = self.rng.clone();
        let minc = self.counts.values().copied().min().unwrap_or(1) as f64;
        let maxc = self.counts.values().copied().max().unwrap_or(1) as f64;
        let keep = if maxc > 0.0 { (minc / maxc).clamp(0.2, 1.0) } else { 1.0 };
        let mut rows = Vec::new();
        for i in 0..x.nrows() {
            if rng.uniform() <= keep {
                rows.push(i);
            }
        }
        if rows.is_empty() && x.nrows() > 0 {
            rows.push(0);
        }
        ctx.finish(Matrix::from_fn(rows.len(), x.ncols(), |r, c| {
            x.get(rows[r], c)
        }))
    }
}

/// Streaming mean (river `stats.Mean`).
#[derive(Clone, Debug, Default)]
pub struct OnlineMean {
    n: f64,
    mean: f64,
    updates: u64,
}

impl OnlineMean {
    /// Empty mean accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current mean, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.n <= 0.0 {
            f64::NAN
        } else {
            self.mean
        }
    }
}

impl PartialFit for OnlineMean {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.score();
        for i in 0..x.nrows() {
            let v = x.get(i, 0);
            if !v.is_finite() {
                continue;
            }
            self.n += 1.0;
            self.mean += (v - self.mean) / self.n;
        }
        self.updates += 1;
        let after = self.score();
        let mut q =
            IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n as u64);
        q.effective_sample_size = self.n;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n >= 1.0;
        q.warmup = self.n < 1.0;
        q.explanation = format!("OnlineMean={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "Welford mean",
                "running mean of column 0",
                format!("mean={before:.6e}"),
                format!("mean={after:.6e}"),
            ),
        )
    }
}

/// Streaming variance (river `stats.Var`).
#[derive(Clone, Debug, Default)]
pub struct OnlineVar {
    n: f64,
    mean: f64,
    m2: f64,
    updates: u64,
}

impl OnlineVar {
    /// Empty variance accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current sample variance, or NaN during warmup.
    pub fn score(&self) -> f64 {
        if self.n < 2.0 {
            f64::NAN
        } else {
            self.m2 / (self.n - 1.0)
        }
    }
}

impl PartialFit for OnlineVar {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.score();
        for i in 0..x.nrows() {
            let v = x.get(i, 0);
            if !v.is_finite() {
                continue;
            }
            self.n += 1.0;
            let d = v - self.mean;
            self.mean += d / self.n;
            self.m2 += d * (v - self.mean);
        }
        self.updates += 1;
        let after = self.score();
        let mut q =
            IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n as u64);
        q.effective_sample_size = self.n;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n >= 2.0;
        q.warmup = self.n < 2.0;
        q.explanation = format!("OnlineVar={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "Welford variance",
                "running sample variance of column 0",
                format!("var={before:.6e}"),
                format!("var={after:.6e}"),
            ),
        )
    }
}

/// Streaming finite-count (river `stats.Count`).
#[derive(Clone, Debug, Default)]
pub struct OnlineCount {
    n: u64,
    updates: u64,
}

impl OnlineCount {
    /// Empty counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current count.
    pub fn score(&self) -> f64 {
        self.n as f64
    }
}

impl PartialFit for OnlineCount {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.score();
        for i in 0..x.nrows() {
            if x.get(i, 0).is_finite() {
                self.n += 1;
            }
        }
        self.updates += 1;
        let after = self.score();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n);
        q.effective_sample_size = self.n as f64;
        q.parameter_delta_norm = Some((after - before).abs());
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n > 0;
        q.warmup = self.n == 0;
        q.explanation = format!("OnlineCount={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "count update",
                "finite entries in column 0",
                format!("n={before:.6e}"),
                format!("n={after:.6e}"),
            ),
        )
    }
}

/// Streaming lag-1 autocorrelation (river `stats.AutoCorr`).
#[derive(Clone, Debug, Default)]
pub struct OnlineAutoCorr {
    last: Option<f64>,
    n: f64,
    mx: f64,
    my: f64,
    c: f64,
    vx: f64,
    vy: f64,
    updates: u64,
}

impl OnlineAutoCorr {
    /// Empty lag-1 accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current lag-1 correlation, or NaN during warmup.
    pub fn score(&self) -> f64 {
        if self.n < 2.0 {
            return f64::NAN;
        }
        let den = (self.vx * self.vy).sqrt();
        if den <= 1e-18 {
            f64::NAN
        } else {
            self.c / den
        }
    }
}

impl PartialFit for OnlineAutoCorr {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.score();
        for i in 0..x.nrows() {
            let v = x.get(i, 0);
            if !v.is_finite() {
                continue;
            }
            if let Some(prev) = self.last {
                self.n += 1.0;
                let dx = prev - self.mx;
                self.mx += dx / self.n;
                let dy = v - self.my;
                self.my += dy / self.n;
                self.c += dx * (v - self.my);
                self.vx += dx * (prev - self.mx);
                self.vy += dy * (v - self.my);
            }
            self.last = Some(v);
        }
        self.updates += 1;
        let after = self.score();
        let mut q =
            IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n as u64);
        q.effective_sample_size = self.n;
        q.parameter_delta_norm = Some(if before.is_finite() && after.is_finite() {
            (after - before).abs()
        } else {
            0.0
        });
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n >= 3.0;
        q.warmup = self.n < 3.0;
        q.explanation = format!("OnlineAutoCorr={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "lag-1 correlation",
                "Welford Pearson of consecutive column-0 values",
                format!("rho={before:.6e}"),
                format!("rho={after:.6e}"),
            ),
        )
    }
}

/// Streaming variance threshold (river `feature_selection.VarianceThreshold`).
#[derive(Clone, Debug)]
pub struct OnlineVarianceThreshold {
    /// Keep columns with sample variance above this.
    pub threshold: f64,
    n: Vec<f64>,
    mean: Vec<f64>,
    m2: Vec<f64>,
    n_features: usize,
    updates: u64,
}

impl Default for OnlineVarianceThreshold {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            n: Vec::new(),
            mean: Vec::new(),
            m2: Vec::new(),
            n_features: 0,
            updates: 0,
        }
    }
}

impl OnlineVarianceThreshold {
    /// Threshold `t` (not identification `p`).
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Self::default()
        }
    }
}

impl PartialFit for OnlineVarianceThreshold {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        if self.n_features == 0 {
            self.n_features = x.ncols();
            self.n = vec![0.0; x.ncols()];
            self.mean = vec![0.0; x.ncols()];
            self.m2 = vec![0.0; x.ncols()];
        } else if x.ncols() != self.n_features {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message(format!(
                        "OnlineVarianceThreshold saw {} columns after init with {}",
                        x.ncols(),
                        self.n_features
                    ))
                    .build(),
            );
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), 0, "feature space changed"),
            );
        }
        let before = self
            .m2
            .iter()
            .zip(&self.n)
            .map(|(m2, n)| if *n > 1.0 { m2 / (n - 1.0) } else { 0.0 })
            .sum::<f64>();
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                self.n[j] += 1.0;
                let d = v - self.mean[j];
                self.mean[j] += d / self.n[j];
                self.m2[j] += d * (v - self.mean[j]);
            }
        }
        self.updates += 1;
        let after = self
            .m2
            .iter()
            .zip(&self.n)
            .map(|(m2, n)| if *n > 1.0 { m2 / (n - 1.0) } else { 0.0 })
            .sum::<f64>();
        let kept = (0..self.n_features)
            .filter(|&j| {
                let n = self.n[j];
                n > 1.0 && self.m2[j] / (n - 1.0) > self.threshold
            })
            .count();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), 0);
        q.effective_sample_size = self.n.iter().copied().fold(0.0_f64, f64::max);
        q.parameter_delta_norm = Some((after - before).abs());
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = kept >= 1;
        q.warmup = self.n.iter().all(|n| *n < 2.0);
        q.explanation = format!("OnlineVarianceThreshold kept={kept}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "column variance update",
                "Welford variance per column; transform drops near-constants",
                format!("sumvar={before:.6e}"),
                format!("sumvar={after:.6e} kept={kept}"),
            ),
        )
    }
}

impl Transform for OnlineVarianceThreshold {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_online_xy(&mut ctx, x, None);
        let mut keep = Vec::new();
        for j in 0..x.ncols().min(self.n_features) {
            let n = self.n.get(j).copied().unwrap_or(0.0);
            let v = if n > 1.0 {
                self.m2[j] / (n - 1.0)
            } else {
                0.0
            };
            if v > self.threshold {
                keep.push(j);
            }
        }
        if keep.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .message("OnlineVarianceThreshold kept no columns; returning column 0")
                    .build(),
            );
            keep.push(0);
        }
        ctx.finish(Matrix::from_fn(x.nrows(), keep.len(), |i, j| {
            x.get(i, keep[j].min(x.ncols().saturating_sub(1)))
        }))
    }
}

/// Streaming univariate Gaussian (river `proba.Gaussian`).
#[derive(Clone, Debug, Default)]
pub struct OnlineGaussian {
    n: f64,
    mean: f64,
    m2: f64,
    updates: u64,
}

impl OnlineGaussian {
    /// Empty Gaussian.
    pub fn new() -> Self {
        Self::default()
    }

    /// Density at `x`, or NaN during warmup.
    pub fn pdf(&self, x: f64) -> f64 {
        if self.n < 2.0 {
            return f64::NAN;
        }
        let var = self.m2 / (self.n - 1.0);
        if var <= 1e-18 {
            return f64::NAN;
        }
        let z = (x - self.mean) / var.sqrt();
        (-0.5 * z * z).exp() / (std::f64::consts::TAU * var).sqrt()
    }
}

impl PartialFit for OnlineGaussian {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, None);
        let before = self.mean;
        for i in 0..x.nrows() {
            let v = x.get(i, 0);
            if !v.is_finite() {
                continue;
            }
            self.n += 1.0;
            let d = v - self.mean;
            self.mean += d / self.n;
            self.m2 += d * (v - self.mean);
        }
        self.updates += 1;
        if self.n >= 2.0 && self.m2 / (self.n - 1.0) <= 1e-18 {
            ctx.push(
                Issue::builder(IssueCode::DegenerateDistribution)
                    .message("OnlineGaussian variance vanished; pdf is undefined")
                    .build(),
            );
        }
        let after = self.mean;
        let mut q =
            IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n as u64);
        q.effective_sample_size = self.n;
        q.parameter_delta_norm = Some((after - before).abs());
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n >= 2.0 && self.m2 > 1e-18;
        q.warmup = self.n < 2.0;
        q.explanation = format!("OnlineGaussian μ={after:.6e}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "Gaussian sufficient stats",
                "Welford mean and variance of column 0",
                format!("mu={before:.6e}"),
                format!("mu={after:.6e}"),
            ),
        )
    }
}

/// Hard-example buffer around a perceptron (river `imblearn.HardSamplingClassifier`).
#[derive(Clone, Debug)]
pub struct HardSamplingClassifier {
    inner: Perceptron,
    buffer_x: Vec<Vec<f64>>,
    buffer_y: Vec<f64>,
    cap: usize,
    n_seen: u64,
    updates: u64,
}

impl Default for HardSamplingClassifier {
    fn default() -> Self {
        Self {
            inner: Perceptron::new(),
            buffer_x: Vec::new(),
            buffer_y: Vec::new(),
            cap: 32,
            n_seen: 0,
            updates: 0,
        }
    }
}

impl HardSamplingClassifier {
    /// Buffer capacity `cap` (not identification `p`).
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            ..Self::default()
        }
    }
}

impl PartialFit for HardSamplingClassifier {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_online_xy(&mut ctx, x, y);
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no labels"),
            );
        };
        let before = self.buffer_x.len();
        if self.n_seen == 0 {
            let _ = self.inner.partial_fit(x, Some(y), &session.child("hs_init"));
        }
        let pred = if self.n_seen == 0 {
            Vector::zeros(x.nrows())
        } else {
            match self.inner.predict(x, &session.child("hs_pred")) {
                Ok(q) => q.value,
                Err(_) => Vector::zeros(x.nrows()),
            }
        };
        for i in 0..x.nrows().min(y.len()) {
            let wrong = self.n_seen == 0
                || (i < pred.len() && (pred[i] - y[i]).abs() >= 0.5);
            if wrong && y[i].is_finite() {
                self.buffer_x
                    .push((0..x.ncols()).map(|j| x.get(i, j)).collect());
                self.buffer_y.push(y[i]);
            }
        }
        while self.buffer_x.len() > self.cap {
            self.buffer_x.remove(0);
            self.buffer_y.remove(0);
        }
        if !self.buffer_x.is_empty() {
            let p = self.buffer_x[0].len();
            let xb = Matrix::from_fn(self.buffer_x.len(), p, |i, j| self.buffer_x[i][j]);
            let yb = Vector::from_slice(&self.buffer_y);
            let _ = self
                .inner
                .partial_fit(&xb, Some(&yb), &session.child("hs_hard"));
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let after = self.buffer_x.len();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = after as f64;
        q.parameter_delta_norm = Some((after as f64 - before as f64).abs());
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.n_seen >= 2;
        q.warmup = self.n_seen < 4;
        q.explanation = format!("HardSamplingClassifier buffer={after}");
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "hard-example buffer",
                "misclassified rows are replayed through the perceptron",
                format!("buf={before}"),
                format!("buf={after}"),
            ),
        )
    }
}

impl Predict for HardSamplingClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_online_xy(&mut ctx, x, None);
        if self.n_seen == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        match self.inner.predict(x, session) {
            Ok(q) => ctx.finish(q.value),
            Err(_) => ctx.finish(Vector::zeros(x.nrows())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{PartialFit, Predict};
    use ojizou_san::{EventKind, Session};

    fn has_incremental(session: &Session) -> bool {
        session
            .ledger()
            .events()
            .iter()
            .any(|e| e.kind == EventKind::IncrementalExplanation)
    }

    #[test]
    fn rls_recovers_slope() {
        let n = 80;
        let x = Matrix::from_fn(n, 1, |i, _| (i as f64) + 1.0);
        let y = Vector::from_iter((0..n).map(|i| 2.0 * ((i as f64) + 1.0)));
        let session = Session::new("rls", "partial_fit");
        let mut m = LinearRegression {
            forgetting_factor: 1.0,
            fit_intercept: false,
            p0: 100.0,
            ..LinearRegression::default()
        };
        let q = m.partial_fit(&x, Some(&y), &session).expect("rls");
        assert!(!q.value.narrative.is_empty());
        assert!(has_incremental(&session));
        let slope = m.coef()[0];
        assert!((slope - 2.0).abs() < 0.05, "slope={slope}");
    }

    #[test]
    fn adwin_fires_on_mean_shift() {
        let session = Session::new("adwin", "update");
        let mut a = Adwin::new(0.002);
        for _ in 0..80 {
            a.update(0.0, &session).expect("stable");
        }
        let mut fired = false;
        for _ in 0..80 {
            match a.update(10.0, &session) {
                Ok(q) => {
                    if matches!(q.value, DriftDecision::Drift { .. }) {
                        fired = true;
                    }
                }
                Err(e) => {
                    if e.primary().code == IssueCode::ConceptDriftDetected {
                        fired = true;
                    }
                }
            }
        }
        assert!(fired, "ADWIN should detect 0 → 10 mean shift");
    }

    #[test]
    fn every_partial_fit_logs_explanation() {
        let x = Matrix::from_fn(12, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..12).map(|i| 2.0 * i as f64));
        let yb = Vector::from_iter((0..12).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }));

        let session = Session::new("online", "partial_fit");
        LinearRegression::new(1.0)
            .partial_fit(&x, Some(&y), &session)
            .expect("rls");
        LogisticRegression::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("logit");
        Perceptron::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("perc");
        PassiveAggressive::new(1.0)
            .partial_fit(&x, Some(&yb), &session)
            .expect("pa");
        HoeffdingTree::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("ht");
        OnlineStandardScaler::new()
            .partial_fit(&x, None, &session)
            .expect("scaler");
        StreamKMeans::new(2)
            .partial_fit(&x, None, &session)
            .expect("kmeans");
        HalfSpaceTrees::new(4, 4)
            .partial_fit(&x, None, &session)
            .expect("hst");
        OnlineAccuracy::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("acc");
        OnlineMse::new()
            .partial_fit(&x, Some(&y), &session)
            .expect("mse");
        OnlineR2::new()
            .partial_fit(&x, Some(&y), &session)
            .expect("r2");
        HoltWintersOnline::new(4)
            .partial_fit(&x, Some(&y), &session)
            .expect("hw");
        AdaptiveModelRules::new(1.0)
            .partial_fit(&x, Some(&y), &session)
            .expect("amrules");
        AdaptiveRandomForest::new(2)
            .partial_fit(&x, Some(&yb), &session)
            .expect("arf");
        AdaptiveRandomForestRegressor::new(2)
            .partial_fit(&x, Some(&y), &session)
            .expect("arfr");
        SgdRegressor::new()
            .partial_fit(&x, Some(&y), &session)
            .expect("sgd");
        crate::glm::SgdClassifier::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("sgdc");
        crate::glm::PassiveAggressiveRegressor::new(1.0)
            .partial_fit(&x, Some(&y), &session)
            .expect("par");
        Alma::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("alma");
        Kswin::new(32)
            .partial_fit(&x, None, &session)
            .expect("kswin");
        Hddm::new().partial_fit(&x, None, &session).expect("hddm");
        Eddm::new().partial_fit(&x, None, &session).expect("eddm");
        HddmW::new().partial_fit(&x, None, &session).expect("hddmw");
        LeveragingBagging::new(2)
            .partial_fit(&x, Some(&yb), &session)
            .expect("lb");
        Snarimax::new(1, 0)
            .partial_fit(&x, Some(&y), &session)
            .expect("snarimax");
        DenStream::new(3.0)
            .partial_fit(&x, None, &session)
            .expect("denstream");
        HoeffdingAdaptiveTree::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("hat");
        StreamingRandomPatches::new(2)
            .partial_fit(&x, Some(&yb), &session)
            .expect("srp");
        let xui = Matrix::from_fn(12, 2, |i, j| {
            if j == 0 {
                (i % 4) as f64
            } else {
                (i % 3) as f64
            }
        });
        FunkMf::new(2)
            .partial_fit(&xui, Some(&y), &session)
            .expect("funk");
        CluStream::new(4)
            .partial_fit(&x, None, &session)
            .expect("clustream");
        ExtremelyFastDecisionTree::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("efdt");
        BiasedMf::new()
            .partial_fit(&xui, Some(&y), &session)
            .expect("biasedmf");
        KnnClassifier::new(3)
            .partial_fit(&x, Some(&yb), &session)
            .expect("knn");
        OnlineMinMaxScaler::new()
            .partial_fit(&x, None, &session)
            .expect("minmax");
        OnlineMaxAbsScaler::new()
            .partial_fit(&x, None, &session)
            .expect("maxabs");
        Baseline::new()
            .partial_fit(&xui, Some(&y), &session)
            .expect("base");
        KnnRegressor::new(3)
            .partial_fit(&x, Some(&y), &session)
            .expect("knnr");
        OnlineRobustScaler::new()
            .partial_fit(&x, None, &session)
            .expect("robust");
        SoftmaxRegression::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("smx");
        OnlineGaussianNb::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("ognb");
        HoeffdingRegressor::new()
            .partial_fit(&x, Some(&y), &session)
            .expect("htr");
        OnlineOneHotEncoder::new(4)
            .partial_fit(&x, None, &session)
            .expect("ohe");
        HoeffdingAdaptiveTreeRegressor::new()
            .partial_fit(&x, Some(&y), &session)
            .expect("hatr");
        OnlineOrdinalEncoder::new(8)
            .partial_fit(&x, None, &session)
            .expect("ord");
        OnlineTfidf::new()
            .partial_fit(&x, None, &session)
            .expect("tfidf");
        StatImputer::new()
            .partial_fit(&x, None, &session)
            .expect("statimp");
        OnlineSelectKBest::new(1)
            .partial_fit(&x, Some(&yb), &session)
            .expect("osk");
        OnlineMae::new()
            .partial_fit(&x, Some(&y), &session)
            .expect("mae");
        OnlineLogLoss::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("ll");
        OnlineRocAuc::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("auc");
        OnlineMultinomialNb::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("mnb");
        PreviousImputer::new()
            .partial_fit(&x, None, &session)
            .expect("prev");
        AdaptiveStandardScaler::new(0.5)
            .partial_fit(&x, None, &session)
            .expect("adp");
        GaussianScorer::new()
            .partial_fit(&x, None, &session)
            .expect("gsc");
        OnlineF1::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("f1");
        OnlinePrecision::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("prec");
        OnlineRecall::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("rec");
        OnlineBernoulliNb::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("bnb");
        EwaRegressor::new(0.4)
            .partial_fit(&x, Some(&y), &session)
            .expect("ewa");
        QuantileFilter::new(0.9)
            .partial_fit(&x, None, &session)
            .expect("qf");
        OnlineRmse::new()
            .partial_fit(&x, Some(&y), &session)
            .expect("rmse");
        OnlineCohenKappa::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("kappa");
        EwMean::new(0.6)
            .partial_fit(&x, None, &session)
            .expect("ewm");
        PolynomialExtender::new(2)
            .partial_fit(&x, None, &session)
            .expect("poly");
        TargetMeanEncoder::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("tme");
        OnlineBagging::new(2)
            .partial_fit(&x, Some(&yb), &session)
            .expect("obag");
        RollingQuantile::new(0.5)
            .partial_fit(&x, None, &session)
            .expect("rq");
        OnlineHamming::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("oham");
        OnlineJaccard::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("ojac");
        EwVar::new(0.6)
            .partial_fit(&x, None, &session)
            .expect("ewv");
        SuccessiveHalvingClassifier::new(4)
            .partial_fit(&x, Some(&yb), &session)
            .expect("sh");
        SuccessiveHalvingRegressor::new(4)
            .partial_fit(&x, Some(&y), &session)
            .expect("shr");
        AmfClassifier::new(3)
            .partial_fit(&x, Some(&yb), &session)
            .expect("amf");
        DbStream::new(2.0)
            .partial_fit(&x, None, &session)
            .expect("dbstream");
        OnlineBaggingRegressor::new(2)
            .partial_fit(&x, Some(&y), &session)
            .expect("obagr");
        OnlineBinarizer::new(0.0)
            .partial_fit(&x, None, &session)
            .expect("obin");
        OnlineIsolationForest::new(3, 3)
            .partial_fit(&x, None, &session)
            .expect("oif");
        RollingMin::new()
            .partial_fit(&x, None, &session)
            .expect("rmin");
        RollingMax::new()
            .partial_fit(&x, None, &session)
            .expect("rmax");
        OnlineSkew::new()
            .partial_fit(&x, None, &session)
            .expect("skew");
        OnlineKurtosis::new()
            .partial_fit(&x, None, &session)
            .expect("kurt");
        OnlineEntropy::new()
            .partial_fit(&x, None, &session)
            .expect("ent");
        Periodic::new(12.0)
            .partial_fit(&x, None, &session)
            .expect("per");
        OnlineCovariance::new()
            .partial_fit(&x, Some(&y), &session)
            .expect("ocov");
        HashingTrick::new(4)
            .partial_fit(&x, None, &session)
            .expect("hash");
        OnlineMad::new()
            .partial_fit(&x, None, &session)
            .expect("mad");
        OnlineIqr::new()
            .partial_fit(&x, None, &session)
            .expect("iqr");
        OnlinePearson::new()
            .partial_fit(&x, Some(&y), &session)
            .expect("pearson");
        RollingMean::new()
            .partial_fit(&x, None, &session)
            .expect("rmean");
        AdwinBagging::new(2)
            .partial_fit(&x, Some(&yb), &session)
            .expect("abag");
        OnlineAdaBoost::new(2)
            .partial_fit(&x, Some(&yb), &session)
            .expect("ada");
        RandomOverSampler::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("ros");
        RandomUnderSampler::new()
            .partial_fit(&x, Some(&yb), &session)
            .expect("rus");
        OnlineMean::new()
            .partial_fit(&x, None, &session)
            .expect("omean");
        OnlineVar::new()
            .partial_fit(&x, None, &session)
            .expect("ovar");
        OnlineCount::new()
            .partial_fit(&x, None, &session)
            .expect("ocnt");
        OnlineAutoCorr::new()
            .partial_fit(&x, None, &session)
            .expect("oac");
        OnlineVarianceThreshold::new(0.0)
            .partial_fit(&x, None, &session)
            .expect("ovt");
        OnlineGaussian::new()
            .partial_fit(&x, None, &session)
            .expect("ogauss");
        HardSamplingClassifier::new(8)
            .partial_fit(&x, Some(&yb), &session)
            .expect("hsc");

        let n_expl = session
            .ledger()
            .events()
            .iter()
            .filter(|e| e.kind == EventKind::IncrementalExplanation)
            .count();
        assert!(
            n_expl >= 42,
            "expected an IncrementalExplanation per partial_fit, got {n_expl}"
        );
    }

    #[test]
    fn rls_predicts_after_fit() {
        let x = Matrix::from_fn(20, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..20).map(|i| 2.0 * i as f64));
        let session = Session::new("rls", "partial_fit");
        let mut m = LinearRegression {
            fit_intercept: false,
            ..LinearRegression::default()
        };
        m.partial_fit(&x, Some(&y), &session).expect("fit");
        let pred = m.predict(&x, &session).expect("pred").value;
        assert!((pred[10] - 20.0).abs() < 1.0, "{}", pred[10]);
    }
}
