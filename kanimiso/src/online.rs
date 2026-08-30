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
use signlred::{IncrementalQuality, Issue, IssueCode, Qualified, Result, Severity};

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
        AdaptiveRandomForest::new(2)
            .partial_fit(&x, Some(&yb), &session)
            .expect("arf");
        SgdRegressor::new()
            .partial_fit(&x, Some(&y), &session)
            .expect("sgd");

        let n_expl = session
            .ledger()
            .events()
            .iter()
            .filter(|e| e.kind == EventKind::IncrementalExplanation)
            .count();
        assert!(
            n_expl >= 14,
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
