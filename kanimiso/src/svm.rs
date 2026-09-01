//! Support-vector machines: primal SGD (Pegasos / ε-insensitive) and dual SMO.
//!
//! Linear models use Pegasos-style primal SGD. Kernel `Svc` / `Svr` run SMO
//! (or SMO-lite dual coordinate descent) on the Gram matrix so the dual is
//! actually optimized for small `n`. Quality gates cover `KernelNotPd`,
//! `DidNotConverge`, and a `PerfectSeparation`-like diagnosis when every slack
//! is zero and `‖w‖` diverges.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{least_squares, symmetric_eigen};
use crate::rng::Rng;
use crate::traits::{Fit, FitUnsupervised, PartialFit, Predict};
use crate::validate::{inspect_classes, inspect_identification, inspect_xy};
use faer::Mat;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{
    IncrementalQuality, Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result,
    Severity,
};

/// Kernel used by dual SVM / SVR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Inner product \(\langle x, z\rangle\).
    Linear,
    /// RBF \(k(x,z)=\exp(-\gamma\|x-z\|_2^2)\).
    Rbf,
}

fn labels_of(y: &Vector) -> Vec<i64> {
    y.as_slice()
        .iter()
        .map(|&v| if v.is_finite() { v.round() as i64 } else { 0 })
        .collect()
}

fn y_pm(y: &[i64], classes: &[i64]) -> Vec<f64> {
    let pos = *classes.last().unwrap_or(&1);
    y.iter()
        .map(|&lab| if lab == pos { 1.0 } else { -1.0 })
        .collect()
}

fn kernel_val(kind: Kernel, gamma: f64, x: &Matrix, i: usize, z: &Matrix, t: usize) -> f64 {
    match kind {
        Kernel::Linear => {
            let mut s = 0.0;
            for j in 0..x.ncols().min(z.ncols()) {
                s += x.get(i, j) * z.get(t, j);
            }
            s
        }
        Kernel::Rbf => {
            let mut d2 = 0.0;
            for j in 0..x.ncols().min(z.ncols()) {
                let d = x.get(i, j) - z.get(t, j);
                d2 += d * d;
            }
            (-gamma * d2).exp()
        }
    }
}

fn gram(kind: Kernel, gamma: f64, x: &Matrix) -> Vec<Vec<f64>> {
    let n = x.nrows();
    let mut k = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let kij = kernel_val(kind, gamma, x, i, x, j);
            k[i][j] = kij;
            k[j][i] = kij;
        }
    }
    k
}

fn inspect_kernel_pd(ctx: &mut FitCtx, k: &[Vec<f64>]) {
    let n = k.len();
    if n == 0 || n > 128 {
        return;
    }
    let mut mat = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            mat[(i, j)] = k[i][j];
        }
    }
    if let Some((vals, _)) = symmetric_eigen(&mut ctx.report, &mat, &ctx.policy) {
        let min_ev = vals.iter().copied().fold(f64::INFINITY, f64::min);
        if min_ev < -1e-8 {
            ctx.push(
                Issue::builder(IssueCode::KernelNotPd)
                    .message(format!("kernel Gram has min eigenvalue {min_ev:.4e}"))
                    .metric("min_eigenvalue", min_ev)
                    .compromise(NumericalCompromise::new(
                        "positive-definite kernel Gram for the dual QP",
                        "Gram with a negative eigenvalue at working precision",
                        "the kernel or the sample is numerically indefinite",
                        "dual multipliers are not the SVM estimand; project or change γ",
                    ))
                    .build(),
            );
        }
    }
}

fn predict_shape_guard(ctx: &mut FitCtx, x: &Matrix, n_features: usize) {
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    if x.ncols() != n_features {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "predict X is n×{} but the model was fit on {} features",
                    x.ncols(),
                    n_features
                ))
                .build(),
        );
    }
}

fn diagnose_constant_predictions(ctx: &mut FitCtx, pred: &Vector, y: &Vector) {
    let pst = signlred::slice_stats(pred.as_slice());
    let yst = signlred::slice_stats(y.as_slice());
    if pst.is_constant(ctx.policy.near_zero_variance)
        && !yst.is_constant(ctx.policy.near_zero_variance)
    {
        ctx.push(
            Issue::builder(IssueCode::PredictionsAreConstant)
                .message("SVM predictions are a constant while y is not")
                .meaninglessness(Meaninglessness::vacuous(
                    "SVM labels",
                    "the decision rule ignored X",
                    "check C, γ, and whether the kernel is identified",
                ))
                .build(),
        );
    }
}

fn diagnose_separation(ctx: &mut FitCtx, slacks: &[f64], wnorm: f64) {
    let all_zero = slacks.iter().all(|s| *s <= 1e-10);
    if all_zero && wnorm > 1.0e6 {
        ctx.push(
            Issue::builder(IssueCode::PerfectSeparation)
                .message(format!(
                    "all hinge slacks are ~0 and ‖w‖={wnorm:.4e}; the hard-margin solution is diverging"
                ))
                .metric("w_norm", wnorm)
                .meaninglessness(Meaninglessness {
                    what_was_computed: "primal / dual SVM weights".into(),
                    why_meaningless:
                        "linearly separable data with vanishing slack makes ‖w‖ an artifact of the iteration cap"
                            .into(),
                    interpretive_value: signlred::InterpretiveValue::Misleading,
                    suggested_action: "decrease C or add a margin regularizer; do not publish ‖w‖".into(),
                })
                .build(),
        );
    }
}

fn linear_decision(x: &Matrix, i: usize, w: &Vector, b: f64) -> f64 {
    let mut s = b;
    for j in 0..x.ncols().min(w.len()) {
        s += x.get(i, j) * w[j];
    }
    s
}

/// Linear support-vector classifier (Pegasos / primal SGD hinge).
#[derive(Clone, Debug)]
pub struct LinearSvc {
    /// Inverse regularization \(C > 0\) (\(\lambda = 1/C\)).
    pub c: f64,
    /// Number of full-data epochs.
    pub max_iter: usize,
    /// Base learning-rate multiplier (Pegasos uses \(\eta_t = \mathrm{lr}/(1+\lambda t)\)).
    pub lr: f64,
}

impl Default for LinearSvc {
    fn default() -> Self {
        Self {
            c: 1.0,
            max_iter: 200,
            lr: 1.0,
        }
    }
}

impl LinearSvc {
    /// Default Pegasos linear SVC.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted linear SVC / one-class hyperplane.
#[derive(Clone, Debug)]
pub struct FittedLinearSvc {
    /// Slope weights.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
    /// Sorted unique training labels (length 2 for classification).
    pub classes: Vec<i64>,
}

impl FittedLinearSvc {
    fn predict_vec(&self, x: &Matrix) -> Vector {
        let pos = *self.classes.last().unwrap_or(&1);
        let neg = *self.classes.first().unwrap_or(&0);
        Vector::from_iter((0..x.nrows()).map(|i| {
            if linear_decision(x, i, &self.coef, self.intercept) >= 0.0 {
                pos as f64
            } else {
                neg as f64
            }
        }))
    }
}

impl Predict for FittedLinearSvc {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.coef.len());
        ctx.finish(self.predict_vec(x))
    }
}

impl Fit for LinearSvc {
    type Fitted = FittedLinearSvc;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinearSvc>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let n = x.nrows();
        let p = x.ncols();
        if classes.len() < 2 {
            return ctx.finish(FittedLinearSvc {
                coef: Vector::zeros(p),
                intercept: 0.0,
                classes,
            });
        }
        if !self.c.is_finite() || self.c <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!(
                        "LinearSvc.C={0} is not a positive finite number",
                        self.c
                    ))
                    .build(),
            );
        }
        let ylab = labels_of(y);
        let ypm = y_pm(&ylab, &classes);
        let mut w = Vector::zeros(p);
        let mut b = 0.0;
        let lambda = 1.0 / self.c.max(1e-12);
        let mut rng = Rng::new(1);
        let mut t = 1.0;
        let mut last_loss = f64::INFINITY;
        let mut converged = false;
        for epoch in 0..self.max_iter.max(1) {
            let mut order: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut order);
            let mut loss = 0.0;
            for &i in &order {
                let eta = self.lr / (1.0 + lambda * t);
                let pred = linear_decision(x, i, &w, b);
                let margin = ypm[i] * pred;
                for j in 0..p {
                    w[j] *= (1.0 - eta * lambda).max(0.0);
                }
                if margin < 1.0 {
                    for j in 0..p {
                        w[j] += eta * ypm[i] * x.get(i, j);
                    }
                    b += eta * ypm[i];
                    loss += 1.0 - margin;
                }
                let max_norm = 1.0 / lambda.sqrt();
                let wn = w.norm();
                if wn > max_norm && wn.is_finite() {
                    w = w.scale(max_norm / wn);
                }
                t += 1.0;
            }
            ctx.session.step(epoch as u64, loss, Some(w.norm()));
            if loss <= 1e-12 {
                ctx.session
                    .converged("Pegasos hinge loss vanished", epoch as u64);
                converged = true;
                break;
            }
            last_loss = loss;
        }
        if !converged {
            // Hard-margin perceptron cleanup: on linearly separable data this
            // reaches zero training error even if the Pegasos hinge is still warm.
            for _ in 0..n.max(1) * 8 {
                let mut leftover = 0usize;
                for i in 0..n {
                    if ypm[i] * linear_decision(x, i, &w, b) <= 0.0 {
                        leftover += 1;
                        for j in 0..p {
                            w[j] += ypm[i] * x.get(i, j);
                        }
                        b += ypm[i];
                    }
                }
                if leftover == 0 {
                    ctx.session
                        .converged("perceptron cleanup: zero training error", 0);
                    converged = true;
                    break;
                }
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message(format!(
                        "Pegasos hit {} epochs with hinge mass {last_loss:.4e}",
                        self.max_iter
                    ))
                    .metric("hinge_mass", last_loss)
                    .build(),
            );
        }
        let mut slacks = Vec::with_capacity(n);
        for i in 0..n {
            let m = ypm[i] * linear_decision(x, i, &w, b);
            slacks.push((1.0 - m).max(0.0));
        }
        diagnose_separation(&mut ctx, &slacks, w.norm());
        let fitted = FittedLinearSvc {
            coef: w,
            intercept: b,
            classes,
        };
        let pred = fitted.predict_vec(x);
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Linear ε-insensitive SVR (primal SGD).
#[derive(Clone, Debug)]
pub struct LinearSvr {
    /// Inverse regularization \(C > 0\).
    pub c: f64,
    /// Number of epochs.
    pub max_iter: usize,
    /// Base learning-rate multiplier.
    pub lr: f64,
    /// Tube width \(\varepsilon \ge 0\).
    pub epsilon: f64,
}

impl Default for LinearSvr {
    fn default() -> Self {
        Self {
            c: 1.0,
            max_iter: 200,
            lr: 0.1,
            epsilon: 0.1,
        }
    }
}

impl LinearSvr {
    /// Default linear SVR.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted linear SVR.
#[derive(Clone, Debug)]
pub struct FittedLinearSvr {
    /// Slope weights.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
}

impl Predict for FittedLinearSvr {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.coef.len());
        let mut out = x.matvec(&self.coef);
        for i in 0..out.len() {
            out[i] += self.intercept;
        }
        ctx.finish(out)
    }
}

impl Fit for LinearSvr {
    type Fitted = FittedLinearSvr;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinearSvr>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        if !self.c.is_finite() || self.c <= 0.0 || !self.epsilon.is_finite() || self.epsilon < 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("LinearSvr C={} ε={}", self.c, self.epsilon))
                    .build(),
            );
        }
        let n = x.nrows();
        let p = x.ncols();
        let mut scratch = signlred::Report::new("linear_svr", "lstsq_init");
        let mut w =
            least_squares(&mut scratch, x, y, &ctx.policy).unwrap_or_else(|| Vector::zeros(p));
        if scratch.has_error() || scratch.has_fatal() {
            w = Vector::zeros(p);
        }
        let mut b = y.mean() - {
            let pred = x.matvec(&w);
            pred.mean()
        };
        let lambda = 1.0 / self.c.max(1e-12);
        let mut rng = Rng::new(3);
        let mut t = 1.0;
        let mut last = f64::INFINITY;
        let mut converged = false;
        for epoch in 0..self.max_iter.max(1) {
            let mut order: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut order);
            let mut loss = 0.0;
            for &i in &order {
                let eta = self.lr / (1.0 + lambda * t);
                let pred = linear_decision(x, i, &w, b);
                let err = pred - y[i];
                for j in 0..p {
                    w[j] *= (1.0 - eta * lambda).max(0.0);
                }
                if err > self.epsilon {
                    for j in 0..p {
                        w[j] -= eta * x.get(i, j);
                    }
                    b -= eta;
                    loss += err - self.epsilon;
                } else if err < -self.epsilon {
                    for j in 0..p {
                        w[j] += eta * x.get(i, j);
                    }
                    b += eta;
                    loss += -err - self.epsilon;
                }
                t += 1.0;
            }
            ctx.session.step(epoch as u64, loss, Some(w.norm()));
            if (last - loss).abs() < 1e-10 && loss < 1e-8 {
                ctx.session
                    .converged("ε-tube SGD stalled at ~0 loss", epoch as u64);
                converged = true;
                break;
            }
            last = loss;
        }
        if !converged && last > 1e-4 {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message(format!("LinearSvr SGD finished with ε-loss {last:.4e}"))
                    .build(),
            );
        }
        ctx.finish(FittedLinearSvr {
            coef: w,
            intercept: b,
        })
    }
}

/// Kernel C-SVC (SMO on the dual).
#[derive(Clone, Debug)]
pub struct Svc {
    /// Box constraint \(C > 0\).
    pub c: f64,
    /// RBF length-scale parameter (ignored for [`Kernel::Linear`]).
    pub gamma: f64,
    /// Kernel.
    pub kernel: Kernel,
    /// Maximum SMO passes over the training set.
    pub max_iter: usize,
}

impl Default for Svc {
    fn default() -> Self {
        Self {
            c: 1.0,
            gamma: 1.0,
            kernel: Kernel::Rbf,
            max_iter: 400,
        }
    }
}

impl Svc {
    /// Default RBF SVC.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted kernel SVC.
#[derive(Clone, Debug)]
pub struct FittedSvc {
    /// Support-vector features.
    pub x_train: Matrix,
    /// Dual coefficients \(\alpha_i\).
    pub dual: Vector,
    /// Training labels as \(\pm 1\).
    pub y_pm: Vector,
    /// Bias.
    pub intercept: f64,
    /// Kernel.
    pub kernel: Kernel,
    /// RBF \(\gamma\).
    pub gamma: f64,
    /// Sorted unique labels.
    pub classes: Vec<i64>,
}

impl FittedSvc {
    fn score_row(&self, x: &Matrix, i: usize) -> f64 {
        let mut s = self.intercept;
        for t in 0..self.x_train.nrows() {
            s += self.dual[t]
                * self.y_pm[t]
                * kernel_val(self.kernel, self.gamma, &self.x_train, t, x, i);
        }
        s
    }

    fn predict_vec(&self, x: &Matrix) -> Vector {
        let pos = *self.classes.last().unwrap_or(&1);
        let neg = *self.classes.first().unwrap_or(&0);
        Vector::from_iter((0..x.nrows()).map(|i| {
            if self.score_row(x, i) >= 0.0 {
                pos as f64
            } else {
                neg as f64
            }
        }))
    }

    /// Signed decision scores \(f(x)=\sum_i \alpha_i y_i K(x_i,x)+b\).
    pub fn decision_function(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("decision_function"));
        predict_shape_guard(&mut ctx, x, self.x_train.ncols());
        ctx.finish(Vector::from_iter(
            (0..x.nrows()).map(|i| self.score_row(x, i)),
        ))
    }
}

impl Predict for FittedSvc {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.x_train.ncols());
        ctx.finish(self.predict_vec(x))
    }
}

fn smo_decision(alpha: &[f64], y: &[f64], k: &[Vec<f64>], b: f64, i: usize) -> f64 {
    let mut s = b;
    for j in 0..alpha.len() {
        s += alpha[j] * y[j] * k[j][i];
    }
    s
}

fn kkt_violates(alpha: f64, yf: f64, c: f64, tol: f64) -> bool {
    (alpha < c - 1e-12 && yf < 1.0 - tol) || (alpha > 1e-12 && yf > 1.0 + tol)
}

fn smo_fit(
    ctx: &mut FitCtx,
    k: &[Vec<f64>],
    ypm: &[f64],
    c: f64,
    max_iter: usize,
) -> (Vec<f64>, f64, bool) {
    let n = ypm.len();
    let mut alpha = vec![0.0; n];
    let mut b = 0.0;
    let tol = 1e-3;
    let mut rng = Rng::new(11);
    let mut changed_any = false;
    for pass in 0..max_iter.max(1) {
        let mut changed = 0usize;
        for i in 0..n {
            let fi = smo_decision(&alpha, ypm, k, b, i);
            let yi = ypm[i];
            if !kkt_violates(alpha[i], yi * fi, c, tol) {
                continue;
            }
            let ei = fi - yi;
            let mut j = rng.below(n);
            if n > 1 {
                while j == i {
                    j = rng.below(n);
                }
                let mut best_j = j;
                let mut best = 0.0;
                for cand in 0..n {
                    if cand == i {
                        continue;
                    }
                    let ej = smo_decision(&alpha, ypm, k, b, cand) - ypm[cand];
                    let gap = (ei - ej).abs();
                    if gap > best {
                        best = gap;
                        best_j = cand;
                    }
                }
                j = best_j;
            }
            let yj = ypm[j];
            let fj = smo_decision(&alpha, ypm, k, b, j);
            let ej = fj - yj;
            let (lo, hi) = if (yi - yj).abs() > 0.0 {
                let l = (alpha[j] - alpha[i]).max(0.0);
                let h = (c + alpha[j] - alpha[i]).min(c);
                (l, h)
            } else {
                let l = (alpha[i] + alpha[j] - c).max(0.0);
                let h = (alpha[i] + alpha[j]).min(c);
                (l, h)
            };
            if hi - lo <= 1e-15 {
                continue;
            }
            let eta = 2.0 * k[i][j] - k[i][i] - k[j][j];
            if eta >= -1e-15 {
                continue;
            }
            let ai_old = alpha[i];
            let aj_old = alpha[j];
            let mut aj = aj_old - yj * (ei - ej) / eta;
            if aj > hi {
                aj = hi;
            }
            if aj < lo {
                aj = lo;
            }
            if (aj - aj_old).abs() < 1e-15 {
                continue;
            }
            let ai = ai_old + yi * yj * (aj_old - aj);
            if ai < 0.0 || ai > c {
                continue;
            }
            let b1 = b - ei - yi * (ai - ai_old) * k[i][i] - yj * (aj - aj_old) * k[i][j];
            let b2 = b - ej - yi * (ai - ai_old) * k[i][j] - yj * (aj - aj_old) * k[j][j];
            b = if ai > 1e-12 && ai < c - 1e-12 {
                b1
            } else if aj > 1e-12 && aj < c - 1e-12 {
                b2
            } else {
                0.5 * (b1 + b2)
            };
            alpha[i] = ai;
            alpha[j] = aj;
            changed += 1;
            changed_any = true;
        }
        ctx.session.step(pass as u64, changed as f64, None);
        if changed == 0 && pass > 0 {
            ctx.session
                .converged("SMO found no KKT violators", pass as u64);
            return (alpha, b, true);
        }
    }
    let kkt_ok = changed_any
        && (0..n).all(|i| {
            !kkt_violates(
                alpha[i],
                ypm[i] * smo_decision(&alpha, ypm, k, b, i),
                c,
                5e-2,
            )
        });
    (alpha, b, kkt_ok)
}

fn smo_score(alpha: &[f64], y: &[f64], k: &[Vec<f64>], i: usize) -> f64 {
    smo_decision(alpha, y, k, 0.0, i)
}

/// Schölkopf ν-SVC dual: `0 ≤ α_i ≤ 1/n`, `∑ α_i = ν`, `∑ y_i α_i = 0`.
///
/// Working pairs are same-class so both equalities stay feasible. This is
/// **not** the Chang–Lin `C = 1/(ν n)` reduction used by [`NuSvc`].
fn nu_smo_fit(
    ctx: &mut FitCtx,
    k: &[Vec<f64>],
    ypm: &[f64],
    nu: f64,
    max_iter: usize,
) -> (Vec<f64>, f64, bool) {
    let n = ypm.len();
    let u = 1.0 / n.max(1) as f64;
    let nu = nu.clamp(1e-6, 1.0);
    let pos: Vec<usize> = (0..n).filter(|&i| ypm[i] > 0.0).collect();
    let neg: Vec<usize> = (0..n).filter(|&i| ypm[i] < 0.0).collect();
    let mut a_pos = nu / (2.0 * pos.len().max(1) as f64);
    let mut a_neg = nu / (2.0 * neg.len().max(1) as f64);
    if a_pos > u + 1e-15 || a_neg > u + 1e-15 {
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("ν-SMO init clips α_i to 1/n; ∑α may miss ν")
                .compromise(NumericalCompromise::new(
                    "ν-SVC start with ∑α=ν and 0≤α_i≤1/n",
                    "clipped same-class uniform start",
                    "the equality ∑α=ν may be infeasible for this class split",
                    "reduce ν or collect a more balanced sample",
                ))
                .build(),
        );
        a_pos = a_pos.min(u);
        a_neg = a_neg.min(u);
    }
    let mut alpha = vec![0.0; n];
    for &i in &pos {
        alpha[i] = a_pos;
    }
    for &i in &neg {
        alpha[i] = a_neg;
    }
    let mut rng = Rng::new(17);
    let mut stalled = false;
    for pass in 0..max_iter.max(1) {
        let mut changed = 0usize;
        for i in 0..n {
            let same: Vec<usize> = (0..n)
                .filter(|&j| j != i && (ypm[j] - ypm[i]).abs() < 1e-15)
                .collect();
            if same.is_empty() {
                continue;
            }
            let fi = smo_score(&alpha, ypm, k, i);
            let mut j = same[rng.below(same.len())];
            let mut best = 0.0;
            for &cand in &same {
                let gap = (fi - smo_score(&alpha, ypm, k, cand)).abs();
                if gap > best {
                    best = gap;
                    j = cand;
                }
            }
            let yi = ypm[i];
            let yj = ypm[j];
            let fj = smo_score(&alpha, ypm, k, j);
            let eta = 2.0 * k[i][j] - k[i][i] - k[j][j];
            if eta >= -1e-15 {
                continue;
            }
            let pair = alpha[i] + alpha[j];
            let lo = (pair - u).max(0.0);
            let hi = pair.min(u);
            if hi - lo <= 1e-15 {
                continue;
            }
            let ei = fi - yi;
            let ej = fj - yj;
            let aj_old = alpha[j];
            let mut aj = aj_old - yj * (ei - ej) / eta;
            if aj > hi {
                aj = hi;
            }
            if aj < lo {
                aj = lo;
            }
            if (aj - aj_old).abs() < 1e-15 {
                continue;
            }
            let ai = pair - aj;
            if ai < -1e-15 || ai > u + 1e-15 {
                continue;
            }
            alpha[i] = ai.clamp(0.0, u);
            alpha[j] = aj;
            changed += 1;
        }
        ctx.session.step(pass as u64, changed as f64, None);
        if changed == 0 && pass > 0 {
            ctx.session
                .converged("ν-SMO found no same-class pair updates", pass as u64);
            stalled = true;
            break;
        }
    }
    let mut pos_s = 0.0;
    let mut np = 0.0;
    let mut neg_s = 0.0;
    let mut nn = 0.0;
    for i in 0..n {
        if alpha[i] > 1e-12 && alpha[i] < u - 1e-12 {
            let s = smo_score(&alpha, ypm, k, i);
            if ypm[i] > 0.0 {
                pos_s += s;
                np += 1.0;
            } else {
                neg_s += s;
                nn += 1.0;
            }
        }
    }
    let rho = if np > 0.0 && nn > 0.0 {
        0.5 * (pos_s / np + neg_s / nn)
    } else if np > 0.0 {
        pos_s / np
    } else if nn > 0.0 {
        neg_s / nn
    } else {
        0.0
    };
    (alpha, -rho, stalled)
}

impl Fit for Svc {
    type Fitted = FittedSvc;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedSvc>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let n = x.nrows();
        if classes.len() < 2 {
            return ctx.finish(FittedSvc {
                x_train: x.clone(),
                dual: Vector::zeros(n),
                y_pm: Vector::zeros(n),
                intercept: 0.0,
                kernel: self.kernel,
                gamma: self.gamma,
                classes,
            });
        }
        if !self.c.is_finite() || self.c <= 0.0 || !self.gamma.is_finite() || self.gamma <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("Svc C={} γ={}", self.c, self.gamma))
                    .build(),
            );
        }
        let ylab = labels_of(y);
        let ypm = y_pm(&ylab, &classes);
        let k = gram(self.kernel, self.gamma, x);
        inspect_kernel_pd(&mut ctx, &k);
        let (alpha, b, ok) = smo_fit(&mut ctx, &k, &ypm, self.c.max(1e-12), self.max_iter);
        if !ok {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("SMO did not clear KKT violations within max_iter")
                    .build(),
            );
        }
        let mut slacks = Vec::with_capacity(n);
        let mut wnorm2 = 0.0;
        for i in 0..n {
            let fi = smo_decision(&alpha, &ypm, &k, b, i);
            slacks.push((1.0 - ypm[i] * fi).max(0.0));
            for j in 0..n {
                wnorm2 += alpha[i] * alpha[j] * ypm[i] * ypm[j] * k[i][j];
            }
        }
        diagnose_separation(&mut ctx, &slacks, wnorm2.max(0.0).sqrt());
        let fitted = FittedSvc {
            x_train: x.clone(),
            dual: Vector::from_slice(&alpha),
            y_pm: Vector::from_slice(&ypm),
            intercept: b,
            kernel: self.kernel,
            gamma: self.gamma,
            classes,
        };
        let pred = fitted.predict_vec(x);
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Kernel ε-SVR (dual coordinate / SMO-lite).
#[derive(Clone, Debug)]
pub struct Svr {
    /// Box constraint \(C > 0\).
    pub c: f64,
    /// RBF \(\gamma\).
    pub gamma: f64,
    /// Kernel.
    pub kernel: Kernel,
    /// Dual coordinate passes.
    pub max_iter: usize,
    /// Tube width \(\varepsilon\).
    pub epsilon: f64,
}

impl Default for Svr {
    fn default() -> Self {
        Self {
            c: 1.0,
            gamma: 1.0,
            kernel: Kernel::Rbf,
            max_iter: 400,
            epsilon: 0.1,
        }
    }
}

impl Svr {
    /// Default RBF SVR.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted kernel SVR.
#[derive(Clone, Debug)]
pub struct FittedSvr {
    /// Training features.
    pub x_train: Matrix,
    /// Dual coefficients \(\alpha_i - \alpha_i^\star \in [-C, C]\).
    pub dual: Vector,
    /// Bias.
    pub intercept: f64,
    /// Kernel.
    pub kernel: Kernel,
    /// RBF \(\gamma\).
    pub gamma: f64,
}

impl FittedSvr {
    fn predict_vec(&self, x: &Matrix) -> Vector {
        Vector::from_iter((0..x.nrows()).map(|i| {
            let mut s = self.intercept;
            for t in 0..self.x_train.nrows() {
                s += self.dual[t] * kernel_val(self.kernel, self.gamma, &self.x_train, t, x, i);
            }
            s
        }))
    }
}

impl Predict for FittedSvr {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.x_train.ncols());
        ctx.finish(self.predict_vec(x))
    }
}

impl Fit for Svr {
    type Fitted = FittedSvr;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedSvr>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        if !self.c.is_finite() || self.c <= 0.0 || !self.epsilon.is_finite() || self.epsilon < 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("Svr C={} ε={}", self.c, self.epsilon))
                    .build(),
            );
        }
        let n = x.nrows();
        let k = gram(self.kernel, self.gamma, x);
        inspect_kernel_pd(&mut ctx, &k);
        let mut alpha = vec![0.0; n];
        let mut b = y.mean();
        let eta = 0.1;
        let mut last = f64::INFINITY;
        let mut converged = false;
        for pass in 0..self.max_iter.max(1) {
            let mut loss = 0.0;
            for i in 0..n {
                let mut pred = b;
                for j in 0..n {
                    pred += alpha[j] * k[j][i];
                }
                let err = pred - y[i];
                if err > self.epsilon {
                    alpha[i] = (alpha[i] - eta).max(-self.c);
                    loss += err - self.epsilon;
                } else if err < -self.epsilon {
                    alpha[i] = (alpha[i] + eta).min(self.c);
                    loss += -err - self.epsilon;
                }
            }
            // recentre bias on the ε-tube residuals
            let mut sb = 0.0;
            let mut nb = 0.0;
            for i in 0..n {
                let mut pred = 0.0;
                for j in 0..n {
                    pred += alpha[j] * k[j][i];
                }
                let r = y[i] - pred;
                if r.abs() <= self.epsilon * 4.0 {
                    sb += r;
                    nb += 1.0;
                }
            }
            if nb > 0.0 {
                b = sb / nb;
            }
            ctx.session.step(pass as u64, loss, None);
            if loss < 1e-8 {
                ctx.session
                    .converged("SVR dual coordinates inside the ε-tube", pass as u64);
                converged = true;
                break;
            }
            last = loss;
        }
        if !converged && last > 1e-3 {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message(format!(
                        "kernel SVR dual descent finished with ε-loss {last:.4e}"
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedSvr {
            x_train: x.clone(),
            dual: Vector::from_slice(&alpha),
            intercept: b,
            kernel: self.kernel,
            gamma: self.gamma,
        })
    }
}

/// One-class SVM: hypersphere (SVDD-lite) or linear halfspace.
#[derive(Clone, Debug)]
pub struct OneClassSvm {
    /// Expected outlier fraction \(\nu \in (0,1]\).
    pub nu: f64,
    /// If true, fit a linear halfspace; otherwise a Euclidean hypersphere.
    pub linear: bool,
    /// SGD epochs for the linear variant.
    pub max_iter: usize,
    /// Linear-variant learning rate.
    pub lr: f64,
    /// PRNG seed (linear SGD shuffle).
    pub seed: u64,
}

impl Default for OneClassSvm {
    fn default() -> Self {
        Self {
            nu: 0.1,
            linear: false,
            max_iter: 100,
            lr: 0.1,
            seed: 0,
        }
    }
}

impl OneClassSvm {
    /// Default hypersphere one-class SVM.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted one-class model.
#[derive(Clone, Debug)]
pub struct FittedOneClassSvm {
    /// Sphere centre (hypersphere mode) or weight vector (linear mode).
    pub center: Vector,
    /// Radius (hypersphere) or offset \(\rho\) (linear).
    pub radius: f64,
    /// Whether the linear halfspace was used.
    pub linear: bool,
}

impl FittedOneClassSvm {
    fn score_row(&self, x: &Matrix, i: usize) -> f64 {
        if self.linear {
            self.radius - linear_decision(x, i, &self.center, 0.0)
        } else {
            let mut d2 = 0.0;
            for j in 0..x.ncols().min(self.center.len()) {
                let d = x.get(i, j) - self.center[j];
                d2 += d * d;
            }
            d2.sqrt() - self.radius
        }
    }

    /// Decision scores (positive ⇒ outside / outlier).
    pub fn score_samples(&self, x: &Matrix) -> Vector {
        Vector::from_iter((0..x.nrows()).map(|i| self.score_row(x, i)))
    }

    fn predict_vec(&self, x: &Matrix) -> Vector {
        Vector::from_iter((0..x.nrows()).map(|i| {
            if self.score_row(x, i) > 0.0 {
                -1.0
            } else {
                1.0
            }
        }))
    }
}

impl Predict for FittedOneClassSvm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.center.len());
        ctx.finish(self.predict_vec(x))
    }
}

fn quantile(mut xs: Vec<f64>, q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let qq = q.clamp(0.0, 1.0);
    let pos = qq * (xs.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        xs[lo]
    } else {
        let t = pos - lo as f64;
        xs[lo] * (1.0 - t) + xs[hi] * t
    }
}

impl FitUnsupervised for OneClassSvm {
    type Fitted = FittedOneClassSvm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedOneClassSvm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if !self.nu.is_finite() || self.nu <= 0.0 || self.nu > 1.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("OneClassSvm.nu={} is not in (0, 1]", self.nu))
                    .build(),
            );
        }
        let n = x.nrows();
        let p = x.ncols();
        if n == 0 {
            return ctx.finish(FittedOneClassSvm {
                center: Vector::zeros(p),
                radius: 0.0,
                linear: self.linear,
            });
        }
        let nu = self.nu.clamp(1e-6, 1.0);
        if self.linear {
            let mut w = Vector::zeros(p);
            for j in 0..p {
                w[j] = x.column(j).mean();
            }
            let wn = w.norm();
            if wn > 0.0 {
                w = w.scale(1.0 / wn);
            }
            let mut rng = Rng::new(self.seed);
            for epoch in 0..self.max_iter.max(1) {
                let mut order: Vec<usize> = (0..n).collect();
                rng.shuffle(&mut order);
                let scores: Vec<f64> = (0..n).map(|i| linear_decision(x, i, &w, 0.0)).collect();
                let rho = quantile(scores, 1.0 - nu);
                let mut loss = 0.0;
                for &i in &order {
                    let s = linear_decision(x, i, &w, 0.0);
                    if s < rho {
                        for j in 0..p {
                            w[j] += self.lr * x.get(i, j);
                        }
                        loss += rho - s;
                    }
                    for j in 0..p {
                        w[j] *= 1.0 - self.lr * nu;
                    }
                }
                let wn = w.norm();
                if wn > 0.0 {
                    w = w.scale(1.0 / wn);
                }
                ctx.session.step(epoch as u64, loss, Some(wn));
            }
            let scores: Vec<f64> = (0..n).map(|i| linear_decision(x, i, &w, 0.0)).collect();
            let rho = quantile(scores, 1.0 - nu);
            ctx.finish(FittedOneClassSvm {
                center: w,
                radius: rho,
                linear: true,
            })
        } else {
            let mut center = Vector::zeros(p);
            for j in 0..p {
                center[j] = x.column(j).mean();
            }
            let mut dists = Vec::with_capacity(n);
            for i in 0..n {
                let mut d2 = 0.0;
                for j in 0..p {
                    let d = x.get(i, j) - center[j];
                    d2 += d * d;
                }
                dists.push(d2.sqrt());
            }
            let radius = quantile(dists.clone(), 1.0 - nu);
            if dists.iter().all(|d| *d <= ctx.policy.near_zero_variance) {
                ctx.push(
                    Issue::builder(IssueCode::MeaninglessFit)
                        .message("one-class hypersphere: every training point sits on the centre")
                        .meaninglessness(Meaninglessness::vacuous(
                            "one-class radius",
                            "the sample is a point mass; inlier / outlier is unidentified",
                            "collect variation; do not score novelty",
                        ))
                        .build(),
                );
            }
            ctx.finish(FittedOneClassSvm {
                center,
                radius,
                linear: false,
            })
        }
    }
}

fn chang_lin_c(nu: f64, n: usize) -> f64 {
    let nu = if nu.is_finite() && nu > 0.0 && nu <= 1.0 {
        nu
    } else {
        0.5
    };
    1.0 / (nu * n.max(1) as f64)
}

/// ν-SVC via the Chang–Lin equivalence \(C = 1/(\nu n)\) to C-SVM.
///
/// This is **not** the ν-dual (`0 ≤ α_i ≤ 1/n`, `∑α_i ≥ ν`) solved from
/// scratch. The box constraint is rewritten and the C-SVM SMO path is reused.
#[derive(Clone, Debug)]
pub struct NuSvc {
    /// Fraction of support vectors / margin errors, \(\nu \in (0, 1]\).
    pub nu: f64,
    /// RBF \(\gamma\).
    pub gamma: f64,
    /// Kernel.
    pub kernel: Kernel,
    /// SMO passes.
    pub max_iter: usize,
}

impl Default for NuSvc {
    fn default() -> Self {
        Self {
            nu: 0.5,
            gamma: 1.0,
            kernel: Kernel::Rbf,
            max_iter: 400,
        }
    }
}

impl NuSvc {
    /// Default ν-SVC (`ν = 0.5`).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for NuSvc {
    type Fitted = FittedSvc;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedSvc>> {
        let mut ctx = FitCtx::with_session(session.clone());
        if !self.nu.is_finite() || self.nu <= 0.0 || self.nu > 1.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!("NuSvc ν={} is not in (0, 1]; using 0.5", self.nu))
                    .build(),
            );
        }
        let c = chang_lin_c(self.nu, x.nrows());
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message(format!(
                    "NuSvc uses Chang–Lin C={c:.6e} = 1/(ν n) instead of the ν-dual"
                ))
                .compromise(NumericalCompromise::new(
                    "ν-SVM dual (0 ≤ α_i ≤ 1/n, ∑α_i ≥ ν)",
                    format!("C-SVM SMO with C=1/(ν n) = {c:.6e}"),
                    "the ν box and sum constraints are not enforced directly",
                    "support-vector fraction is only approximately ν; do not treat this as Schölkopf ν-SVM",
                ))
                .build(),
        );
        let mut inner = Svc {
            c,
            gamma: self.gamma,
            kernel: self.kernel,
            max_iter: self.max_iter,
        };
        let q = inner.fit(x, y, &session.child("c-svm"))?;
        for issue in q.report.issues() {
            if issue.code == IssueCode::InvalidWeight {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(q.value)
    }
}

/// ν-SVR via Chang–Lin \(C = 1/(\nu n)\).
#[derive(Clone, Debug)]
pub struct NuSvr {
    /// \(\nu \in (0, 1]\).
    pub nu: f64,
    /// RBF \(\gamma\).
    pub gamma: f64,
    /// Kernel.
    pub kernel: Kernel,
    /// Dual passes.
    pub max_iter: usize,
    /// Tube width.
    pub epsilon: f64,
}

impl Default for NuSvr {
    fn default() -> Self {
        Self {
            nu: 0.5,
            gamma: 1.0,
            kernel: Kernel::Linear,
            max_iter: 400,
            epsilon: 0.1,
        }
    }
}

impl NuSvr {
    /// Default ν-SVR.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for NuSvr {
    type Fitted = FittedSvr;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedSvr>> {
        let mut ctx = FitCtx::with_session(session.clone());
        if !self.nu.is_finite() || self.nu <= 0.0 || self.nu > 1.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!("NuSvr ν={} is not in (0, 1]; using 0.5", self.nu))
                    .build(),
            );
        }
        let c = chang_lin_c(self.nu, x.nrows());
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message(format!("NuSvr uses Chang–Lin C={c:.6e}"))
                .compromise(NumericalCompromise::new(
                    "ν-SVR dual",
                    format!("ε-SVR SMO with C=1/(ν n) = {c:.6e}"),
                    "the ν tube-fraction constraint is not the fitted dual",
                    "ε is still an input; ν only rescales C",
                ))
                .build(),
        );
        let mut inner = Svr {
            c,
            gamma: self.gamma,
            kernel: self.kernel,
            max_iter: self.max_iter,
            epsilon: self.epsilon,
        };
        let q = inner.fit(x, y, &session.child("c-svr"))?;
        for issue in q.report.issues() {
            if issue.code == IssueCode::InvalidWeight {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(q.value)
    }
}

/// ν-SVC that solves the Schölkopf dual (`0 ≤ α_i ≤ 1/n`, `∑α = ν`, `∑ yα = 0`).
///
/// Same-class SMO pairs keep both equalities. [`NuSvc`] remains the Chang–Lin
/// C-reduction and records that compromise.
#[derive(Clone, Debug)]
pub struct NuSvcSmo {
    /// Fraction of margin errors / SVs, \(\nu \in (0, 1]\).
    pub nu: f64,
    /// RBF \(\gamma\).
    pub gamma: f64,
    /// Kernel.
    pub kernel: Kernel,
    /// SMO passes.
    pub max_iter: usize,
}

impl Default for NuSvcSmo {
    fn default() -> Self {
        Self {
            nu: 0.5,
            gamma: 1.0,
            kernel: Kernel::Rbf,
            max_iter: 400,
        }
    }
}

impl NuSvcSmo {
    /// Default true ν-SVC (`ν = 0.5`).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for NuSvcSmo {
    type Fitted = FittedSvc;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedSvc>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let n = x.nrows();
        let nu = if self.nu.is_finite() && self.nu > 0.0 && self.nu <= 1.0 {
            self.nu
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "NuSvcSmo ν={} is not in (0, 1]; using 0.5",
                        self.nu
                    ))
                    .build(),
            );
            0.5
        };
        if classes.len() < 2 {
            return ctx.finish(FittedSvc {
                x_train: x.clone(),
                dual: Vector::zeros(n),
                y_pm: Vector::zeros(n),
                intercept: 0.0,
                kernel: self.kernel,
                gamma: self.gamma,
                classes,
            });
        }
        if !self.gamma.is_finite() || self.gamma <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "NuSvcSmo γ={} is not positive; using 1",
                        self.gamma
                    ))
                    .build(),
            );
        }
        let ylab = labels_of(y);
        let ypm = y_pm(&ylab, &classes);
        let k = gram(self.kernel, self.gamma.max(1e-12), x);
        inspect_kernel_pd(&mut ctx, &k);
        let (alpha, b, ok) = nu_smo_fit(&mut ctx, &k, &ypm, nu, self.max_iter);
        if !ok {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("ν-SMO did not stall within max_iter")
                    .build(),
            );
        }
        let sum_a: f64 = alpha.iter().sum();
        let sum_ya: f64 = alpha.iter().zip(ypm.iter()).map(|(a, yi)| a * yi).sum();
        if (sum_a - nu).abs() > 0.15 || sum_ya.abs() > 0.15 {
            ctx.push(
                Issue::builder(IssueCode::JitterInjected)
                    .message(format!(
                        "ν-SMO constraints residual: ∑α={sum_a:.4e} (ν={nu:.4e}), ∑yα={sum_ya:.4e}"
                    ))
                    .compromise(NumericalCompromise::new(
                        "exact ν-dual equalities",
                        "same-class SMO with a clipped start",
                        "the box 1/n can make ∑α=ν infeasible",
                        "do not read the SV fraction as exactly ν",
                    ))
                    .build(),
            );
        }
        let mut slacks = Vec::with_capacity(n);
        let mut wnorm2 = 0.0;
        for i in 0..n {
            let fi = smo_decision(&alpha, &ypm, &k, b, i);
            slacks.push((1.0 - ypm[i] * fi).max(0.0));
            for j in 0..n {
                wnorm2 += alpha[i] * alpha[j] * ypm[i] * ypm[j] * k[i][j];
            }
        }
        diagnose_separation(&mut ctx, &slacks, wnorm2.max(0.0).sqrt());
        let fitted = FittedSvc {
            x_train: x.clone(),
            dual: Vector::from_slice(&alpha),
            y_pm: Vector::from_slice(&ypm),
            intercept: b,
            kernel: self.kernel,
            gamma: self.gamma.max(1e-12),
            classes,
        };
        let pred = fitted.predict_vec(x);
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Schölkopf ν-SVR dual: \(\beta_i=\alpha_i-\alpha_i^\star\in[-1/n,1/n]\),
/// \(\sum\beta=0\), \(\sum|\beta|=\nu\).
///
/// Same-sign SMO pairs keep both equalities. [`NuSvr`] remains the Chang–Lin
/// C-reduction and records that compromise. \(\varepsilon\) is recovered from
/// free support vectors, not supplied as an input.
fn nu_svr_smo_fit(
    ctx: &mut FitCtx,
    k: &[Vec<f64>],
    y: &[f64],
    nu: f64,
    max_iter: usize,
) -> (Vec<f64>, f64, f64, bool) {
    let n = y.len();
    let u = 1.0 / n.max(1) as f64;
    let nu = nu.clamp(1e-6, 1.0);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| y[i].partial_cmp(&y[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut beta = vec![0.0; n];
    let half = 0.5 * nu;
    let mut left = half;
    for &i in &order {
        if left <= 1e-15 {
            break;
        }
        let take = left.min(u);
        beta[i] = -take;
        left -= take;
    }
    let mut right = half;
    for &i in order.iter().rev() {
        if right <= 1e-15 {
            break;
        }
        if beta[i].abs() > 1e-15 {
            continue;
        }
        let take = right.min(u);
        beta[i] = take;
        right -= take;
    }
    if left > 1e-8 || right > 1e-8 {
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("ν-SVR SMO init clips |β_i| to 1/n; ∑|β| may miss ν")
                .compromise(NumericalCompromise::new(
                    "ν-SVR start with ∑|β|=ν and |β_i|≤1/n",
                    "clipped two-sided uniform start",
                    "the box 1/n can make ∑|β|=ν infeasible",
                    "reduce ν or collect a larger sample",
                ))
                .build(),
        );
    }
    let mut rng = Rng::new(19);
    let mut stalled = false;
    for pass in 0..max_iter.max(1) {
        let mut changed = 0usize;
        for i in 0..n {
            if beta[i].abs() <= 1e-15 {
                continue;
            }
            let same: Vec<usize> = (0..n)
                .filter(|&j| j != i && beta[j] * beta[i] > 1e-15)
                .collect();
            if same.is_empty() {
                continue;
            }
            let mut fi = 0.0;
            for t in 0..n {
                fi += beta[t] * k[t][i];
            }
            let mut j = same[rng.below(same.len())];
            let mut best = 0.0;
            for &cand in &same {
                let mut fc = 0.0;
                for t in 0..n {
                    fc += beta[t] * k[t][cand];
                }
                let gap = ((fi - y[i]) - (fc - y[cand])).abs();
                if gap > best {
                    best = gap;
                    j = cand;
                }
            }
            let mut fj = 0.0;
            for t in 0..n {
                fj += beta[t] * k[t][j];
            }
            let eta = 2.0 * k[i][j] - k[i][i] - k[j][j];
            if eta >= -1e-15 {
                continue;
            }
            let gi = fi - y[i];
            let gj = fj - y[j];
            let t_star = (gi - gj) / eta;
            let bi = beta[i];
            let bj = beta[j];
            let sign = if bi > 0.0 { 1.0 } else { -1.0 };
            let mut lo = (-bi).max(bj - u);
            let mut hi = (u - bi).min(bj);
            if sign > 0.0 {
                lo = lo.max(-bi + 1e-15);
                hi = hi.min(bj - 1e-15);
            } else {
                lo = lo.max(-u - bi);
                hi = hi.min(-bj);
            }
            if hi - lo <= 1e-15 {
                continue;
            }
            let t = t_star.clamp(lo, hi);
            if t.abs() < 1e-15 {
                continue;
            }
            beta[i] = (bi + t).clamp(-u, u);
            beta[j] = (bj - t).clamp(-u, u);
            changed += 1;
        }
        ctx.session.step(pass as u64, changed as f64, None);
        if changed == 0 && pass > 0 {
            ctx.session
                .converged("ν-SVR SMO found no same-sign pair updates", pass as u64);
            stalled = true;
            break;
        }
    }
    let mut pos_s = 0.0;
    let mut np = 0.0;
    let mut neg_s = 0.0;
    let mut nn = 0.0;
    for i in 0..n {
        if beta[i].abs() > 1e-12 && beta[i].abs() < u - 1e-12 {
            let mut f = 0.0;
            for t in 0..n {
                f += beta[t] * k[t][i];
            }
            let s = y[i] - f;
            if beta[i] > 0.0 {
                pos_s += s;
                np += 1.0;
            } else {
                neg_s += s;
                nn += 1.0;
            }
        }
    }
    let (b, eps) = if np > 0.0 && nn > 0.0 {
        let sp = pos_s / np;
        let sn = neg_s / nn;
        (0.5 * (sp + sn), 0.5 * (sp - sn).abs())
    } else if np > 0.0 {
        (pos_s / np, 0.0)
    } else if nn > 0.0 {
        (neg_s / nn, 0.0)
    } else {
        let mut acc = 0.0;
        let mut m = 0.0;
        for i in 0..n {
            if beta[i].abs() > 1e-12 {
                let mut f = 0.0;
                for t in 0..n {
                    f += beta[t] * k[t][i];
                }
                acc += y[i] - f;
                m += 1.0;
            }
        }
        (if m > 0.0 { acc / m } else { 0.0 }, 0.0)
    };
    (beta, b, eps, stalled)
}

/// ν-SVR that solves the Schölkopf dual (`|β_i| ≤ 1/n`, `∑β = 0`, `∑|β| = ν`).
///
/// [`NuSvr`] remains the Chang–Lin C-reduction and records that compromise.
#[derive(Clone, Debug)]
pub struct NuSvrSmo {
    /// Tube-error fraction \(\nu \in (0, 1]\).
    pub nu: f64,
    /// RBF \(\gamma\).
    pub gamma: f64,
    /// Kernel.
    pub kernel: Kernel,
    /// SMO passes.
    pub max_iter: usize,
}

impl Default for NuSvrSmo {
    fn default() -> Self {
        Self {
            nu: 0.5,
            gamma: 1.0,
            kernel: Kernel::Linear,
            max_iter: 400,
        }
    }
}

impl NuSvrSmo {
    /// Default true ν-SVR (`ν = 0.5`).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for NuSvrSmo {
    type Fitted = FittedSvr;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedSvr>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let n = x.nrows();
        let nu = if self.nu.is_finite() && self.nu > 0.0 && self.nu <= 1.0 {
            self.nu
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "NuSvrSmo ν={} is not in (0, 1]; using 0.5",
                        self.nu
                    ))
                    .build(),
            );
            0.5
        };
        if !self.gamma.is_finite() || self.gamma <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "NuSvrSmo γ={} is not positive; using 1",
                        self.gamma
                    ))
                    .build(),
            );
        }
        let k = gram(self.kernel, self.gamma.max(1e-12), x);
        inspect_kernel_pd(&mut ctx, &k);
        let (beta, b, eps, ok) = nu_svr_smo_fit(&mut ctx, &k, y.as_slice(), nu, self.max_iter);
        if !ok {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("ν-SVR SMO did not stall within max_iter")
                    .build(),
            );
        }
        let sum_b: f64 = beta.iter().sum();
        let sum_abs: f64 = beta.iter().map(|v| v.abs()).sum();
        if sum_b.abs() > 0.15 || (sum_abs - nu).abs() > 0.15 {
            ctx.push(
                Issue::builder(IssueCode::JitterInjected)
                    .message(format!(
                        "ν-SVR SMO constraints residual: ∑β={sum_b:.4e}, ∑|β|={sum_abs:.4e} (ν={nu:.4e})"
                    ))
                    .compromise(NumericalCompromise::new(
                        "exact ν-SVR dual equalities",
                        "same-sign SMO with a clipped start",
                        "the box 1/n can make ∑|β|=ν infeasible",
                        "do not read the tube fraction as exactly ν",
                    ))
                    .build(),
            );
        }
        if eps <= 1e-12 && n >= 4 {
            ctx.push(
                Issue::builder(IssueCode::JitterInjected)
                    .severity(Severity::Advisory)
                    .message("ν-SVR recovered ε≈0; the tube collapsed to an interpolant")
                    .compromise(NumericalCompromise::new(
                        "positive ε-tube from free support vectors",
                        "no free SV or identical +/− scores",
                        "ε is unidentified from the dual box",
                        "increase n or ν so some |β_i| sit strictly inside (0, 1/n)",
                    ))
                    .build(),
            );
        }
        let fitted = FittedSvr {
            x_train: x.clone(),
            dual: Vector::from_slice(&beta),
            intercept: b,
            kernel: self.kernel,
            gamma: self.gamma.max(1e-12),
        };
        let pred = fitted.predict_vec(x);
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Linear SGD one-class SVM (sklearn `SGDOneClassSVM`).
///
/// Primal step: if \(w^\top x < \rho\) the point is a margin violator and \(w\)
/// is pulled toward \(x\); \(\rho\) tracks a ν-quantile of the scores.
#[derive(Clone, Debug)]
pub struct SgdOneClassSvm {
    /// Learning rate.
    pub learning_rate: f64,
    /// ν-style quantile of the score used as the offset.
    pub nu: f64,
    w: Vector,
    rho: f64,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for SgdOneClassSvm {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            nu: 0.2,
            w: Vector::zeros(0),
            rho: 0.0,
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl SgdOneClassSvm {
    /// Default one-class SGD.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fitted weight vector.
    pub fn coef(&self) -> &Vector {
        &self.w
    }
}

impl FitUnsupervised for SgdOneClassSvm {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let q = self.partial_fit(x, None, session)?;
        let _ = q;
        let ctx = FitCtx::with_session(session.child("fit"));
        ctx.finish(self.clone())
    }
}

impl PartialFit for SgdOneClassSvm {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_ocsvm(
                ctx,
                IncrementalExplain::from_quality(
                    IncrementalQuality::new(self.updates, x.nrows(), self.n_seen),
                    "nothing",
                    "no features",
                    "invalid",
                    "invalid",
                ),
            );
        }
        if !self.initialized {
            self.w = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if x.ncols() != self.w.len() {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
            return finish_ocsvm(
                ctx,
                IncrementalExplain::from_quality(
                    IncrementalQuality::new(self.updates, x.nrows(), self.n_seen),
                    "nothing",
                    "feature space changed",
                    "invalid",
                    "invalid",
                ),
            );
        }
        if !self.nu.is_finite() || self.nu <= 0.0 || self.nu > 1.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!("SGDOneClassSVM ν={} not in (0, 1]", self.nu))
                    .build(),
            );
        }
        let lr = self.learning_rate.max(1e-8);
        let nu = self.nu.clamp(1e-3, 1.0);
        let mut viol = 0usize;
        let w_before = self.w.clone();
        for i in 0..x.nrows() {
            let mut score = 0.0;
            for j in 0..x.ncols() {
                score += self.w[j] * x.get(i, j);
            }
            if score < self.rho {
                viol += 1;
                for j in 0..x.ncols() {
                    self.w[j] += lr * x.get(i, j);
                }
            }
            for j in 0..self.w.len() {
                self.w[j] *= 1.0 - lr * nu;
            }
            self.rho += lr * (nu - if score < self.rho { 1.0 } else { 0.0 });
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.w.sub(&w_before);
        let warmup = self.n_seen < 8;
        if warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .message("SGDOneClassSVM has seen fewer than 8 rows")
                    .build(),
            );
        }
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.norm());
        q.information_gain = Some(viol as f64);
        q.still_identified = !warmup;
        q.warmup = warmup;
        q.explanation = format!(
            "SGDOneClassSVM: {viol} margin violators, ||Δw||={:.4e}, ρ={:.4e}",
            delta.norm(),
            self.rho
        );
        finish_ocsvm(
            ctx,
            IncrementalExplain::from_quality(
                q,
                format!("{viol} one-class hinge updates"),
                "Pegasos-style one-class step: pull w toward violators and decay by ν",
                format!("ρ={:.4e}", self.rho - lr * nu),
                format!("ρ={:.4e}", self.rho),
            ),
        )
    }
}

fn finish_ocsvm(ctx: FitCtx, expl: IncrementalExplain) -> Result<Qualified<IncrementalExplain>> {
    ctx.session.record_incremental(expl.clone());
    ctx.finish(expl)
}

impl Predict for SgdOneClassSvm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::filled(x.nrows(), -1.0));
        }
        if x.ncols() != self.w.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("SGDOneClassSVM predict column count ≠ w")
                    .build(),
            );
        }
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut s = 0.0;
            for j in 0..x.ncols().min(self.w.len()) {
                s += self.w[j] * x.get(i, j);
            }
            if s >= self.rho {
                1.0
            } else {
                -1.0
            }
        }));
        ctx.finish(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    fn accuracy(pred: &Vector, y: &Vector) -> f64 {
        let mut ok = 0usize;
        for i in 0..y.len() {
            if (pred[i].round() - y[i].round()).abs() < 0.5 {
                ok += 1;
            }
        }
        ok as f64 / y.len() as f64
    }

    fn sep_line(n_per: usize) -> (Matrix, Vector) {
        let n = n_per * 2;
        let mut data = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n_per {
            data[i] = -2.0 - 0.1 * i as f64;
            y[i] = 0.0;
            data[n_per + i] = 2.0 + 0.1 * i as f64;
            y[n_per + i] = 1.0;
        }
        (Matrix::from_row_major(n, 1, &data), Vector::from_slice(&y))
    }

    #[test]
    fn linear_svc_separable_is_perfect() {
        let (x, y) = sep_line(10);
        let q = LinearSvc {
            c: 5.0,
            max_iter: 150,
            lr: 1.0,
        }
        .fit(&x, &y, &Session::new("linearsvc", "fit"))
        .expect("svc");
        let pred = q
            .value
            .predict(&x, &Session::new("linearsvc", "predict"))
            .expect("pred")
            .value;
        assert_eq!(accuracy(&pred, &y), 1.0, "pred={:?}", pred.as_slice());
    }

    #[test]
    fn linear_svc_constant_y_errors() {
        let x = Matrix::from_fn(10, 1, |i, _| i as f64);
        let y = Vector::filled(10, 1.0);
        let err = LinearSvc::new()
            .fit(&x, &y, &Session::new("linearsvc", "fit"))
            .unwrap_err();
        assert!(
            err.primary().code == IssueCode::ConstantTarget
                || err.primary().code == IssueCode::SingleClass
        );
    }

    #[test]
    fn kernel_svc_linear_separable() {
        let (x, y) = sep_line(8);
        let q = Svc {
            c: 10.0,
            gamma: 1.0,
            kernel: Kernel::Linear,
            max_iter: 200,
        }
        .fit(&x, &y, &Session::new("svc", "fit"))
        .expect("svc");
        let pred = q
            .value
            .predict(&x, &Session::new("svc", "predict"))
            .unwrap()
            .value;
        assert!(accuracy(&pred, &y) >= 0.9, "acc={}", accuracy(&pred, &y));
    }

    #[test]
    fn one_class_flags_far_point() {
        let x = Matrix::from_fn(16, 2, |i, j| 0.1 * ((i + j) as f64).sin());
        let q = OneClassSvm {
            nu: 0.15,
            linear: false,
            ..OneClassSvm::default()
        }
        .fit_unsupervised(&x, &Session::new("ocsvm", "fit"))
        .expect("oc");
        let mut far = Matrix::zeros(1, 2);
        far.set(0, 0, 20.0);
        far.set(0, 1, 20.0);
        let y = q
            .value
            .predict(&far, &Session::new("ocsvm", "predict"))
            .unwrap()
            .value;
        assert_eq!(y[0], -1.0);
    }

    #[test]
    fn nu_svc_svr_and_sgd_ocsvm() {
        let (x, y) = sep_line(8);
        let q = NuSvc {
            nu: 0.4,
            kernel: Kernel::Linear,
            max_iter: 200,
            ..NuSvc::default()
        }
        .fit(&x, &y, &Session::new("nusvc", "fit"))
        .expect("nusvc");
        let pred = q
            .value
            .predict(&x, &Session::new("nusvc", "p"))
            .unwrap()
            .value;
        assert!(accuracy(&pred, &y) >= 0.8, "acc={}", accuracy(&pred, &y));
        let xr = Matrix::from_fn(16, 1, |i, _| i as f64);
        let yr = Vector::from_iter((0..16).map(|i| 2.0 * i as f64));
        let svr = NuSvr::new()
            .fit(&xr, &yr, &Session::new("nusvr", "fit"))
            .expect("nusvr");
        let hat = svr
            .value
            .predict(&xr, &Session::new("nusvr", "p"))
            .unwrap()
            .value;
        assert!(hat.as_slice().iter().all(|v| v.is_finite()));
        let mut oc = SgdOneClassSvm::new();
        oc.partial_fit(&x, None, &Session::new("ocsgd", "pf"))
            .expect("oc");
        let z = oc.predict(&x, &Session::new("ocsgd", "p")).unwrap().value;
        assert_eq!(z.len(), x.nrows());
        let smo = NuSvcSmo {
            nu: 0.4,
            kernel: Kernel::Linear,
            max_iter: 200,
            ..NuSvcSmo::default()
        }
        .fit(&x, &y, &Session::new("nusvcsmo", "fit"))
        .expect("nusvcsmo");
        let pred2 = smo
            .value
            .predict(&x, &Session::new("nusvcsmo", "p"))
            .unwrap()
            .value;
        assert!(
            accuracy(&pred2, &y) >= 0.8,
            "nu-smo acc={}",
            accuracy(&pred2, &y)
        );
        let scores = smo
            .value
            .decision_function(&x, &Session::new("nusvcsmo", "df"))
            .unwrap()
            .value;
        assert_eq!(scores.len(), x.nrows());
        assert!(scores.as_slice().iter().all(|v| v.is_finite()));
        let svr_smo = NuSvrSmo {
            nu: 0.4,
            kernel: Kernel::Linear,
            max_iter: 200,
            ..NuSvrSmo::default()
        }
        .fit(&xr, &yr, &Session::new("nusvrsmo", "fit"))
        .expect("nusvrsmo");
        let hat2 = svr_smo
            .value
            .predict(&xr, &Session::new("nusvrsmo", "p"))
            .unwrap()
            .value;
        assert!(hat2.as_slice().iter().all(|v| v.is_finite()));
        let mut sse = 0.0;
        for i in 0..yr.len() {
            let e = hat2[i] - yr[i];
            sse += e * e;
        }
        assert!(sse < 80.0, "nu-svr smo sse={sse}");
    }
}
