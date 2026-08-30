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
use crate::traits::{Fit, FitUnsupervised, Predict};
use crate::validate::{inspect_classes, inspect_identification, inspect_xy};
use faer::Mat;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result};

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
    if pst.is_constant(ctx.policy.near_zero_variance) && !yst.is_constant(ctx.policy.near_zero_variance)
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
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedLinearSvc>> {
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
                    .message(format!("LinearSvc.C={0} is not a positive finite number", self.c))
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
                ctx.session.converged("Pegasos hinge loss vanished", epoch as u64);
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
                    ctx.session.converged("perceptron cleanup: zero training error", 0);
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
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedLinearSvr>> {
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
        let mut w = least_squares(&mut scratch, x, y, &ctx.policy).unwrap_or_else(|| Vector::zeros(p));
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
                ctx.session.converged("ε-tube SGD stalled at ~0 loss", epoch as u64);
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
        ctx.finish(FittedLinearSvr { coef: w, intercept: b })
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
            s += self.dual[t] * self.y_pm[t] * kernel_val(self.kernel, self.gamma, &self.x_train, t, x, i);
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
            ctx.session.converged("SMO found no KKT violators", pass as u64);
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
                ctx.session.converged("SVR dual coordinates inside the ε-tube", pass as u64);
                converged = true;
                break;
            }
            last = loss;
        }
        if !converged && last > 1e-3 {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message(format!("kernel SVR dual descent finished with ε-loss {last:.4e}"))
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
        let pred = q.value.predict(&x, &Session::new("svc", "predict")).unwrap().value;
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
}
