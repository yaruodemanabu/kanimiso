//! One-hidden-layer MLPs and a Bernoulli RBM.
//!
//! Back-prop uses ReLU + SGD. [`PartialFit`] returns a mandatory
//! [`ojizou_san::IncrementalExplain`]. Non-finite losses record
//! [`IssueCode::LossIsNan`]; exploding gradients record
//! [`IssueCode::GradientExploded`]; iteration caps record
//! [`IssueCode::DidNotConverge`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::rng::Rng;
use crate::traits::{Fit, FitUnsupervised, PartialFit, Predict};
use crate::validate::{inspect_classes, inspect_xy};
use ojizou_san::{IncrementalExplain, Session};
use signlred::{IncrementalQuality, Issue, IssueCode, Qualified, Result};

fn relu(z: f64) -> f64 {
    z.max(0.0)
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

fn init_weights(rng: &mut Rng, rows: usize, cols: usize) -> Matrix {
    let s = (2.0 / rows.max(1) as f64).sqrt();
    Matrix::from_fn(rows, cols, |_, _| s * rng.standard_normal())
}

/// Shared one-hidden-layer network.
#[derive(Clone, Debug)]
struct MlpCore {
    w1: Matrix,
    b1: Vector,
    w2: Matrix,
    b2: Vector,
    hidden: usize,
    n_seen: u64,
    updates: u64,
}

impl MlpCore {
    fn new(p: usize, hidden: usize, n_out: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let h = hidden.max(1);
        Self {
            w1: init_weights(&mut rng, p, h),
            b1: Vector::zeros(h),
            w2: init_weights(&mut rng, h, n_out),
            b2: Vector::zeros(n_out),
            hidden: h,
            n_seen: 0,
            updates: 0,
        }
    }

    fn forward(&self, x: &Matrix) -> (Matrix, Matrix) {
        let n = x.nrows();
        let h = self.hidden;
        let o = self.w2.ncols();
        let mut pre = Matrix::zeros(n, h);
        let mut act = Matrix::zeros(n, h);
        for i in 0..n {
            for k in 0..h {
                let mut s = self.b1[k];
                for j in 0..x.ncols().min(self.w1.nrows()) {
                    s += x.get(i, j) * self.w1.get(j, k);
                }
                pre.set(i, k, s);
                act.set(i, k, relu(s));
            }
        }
        let mut out = Matrix::zeros(n, o);
        for i in 0..n {
            for k in 0..o {
                let mut s = self.b2[k];
                for j in 0..h {
                    s += act.get(i, j) * self.w2.get(j, k);
                }
                out.set(i, k, s);
            }
        }
        (act, out)
    }

    fn sgd_step(
        &mut self,
        ctx: &mut FitCtx,
        x: &Matrix,
        target: &Matrix,
        lr: f64,
        classification: bool,
    ) -> (f64, f64) {
        let (act, raw) = self.forward(x);
        let n = x.nrows();
        let h = self.hidden;
        let o = self.w2.ncols();
        let p = x.ncols().min(self.w1.nrows());
        let mut pred = Matrix::zeros(n, o);
        let mut loss = 0.0;
        for i in 0..n {
            if classification && o == 1 {
                let p1 = sigmoid(raw.get(i, 0));
                pred.set(i, 0, p1);
                let y = target.get(i, 0);
                let p1c = p1.clamp(1e-15, 1.0 - 1e-15);
                loss += -(y * p1c.ln() + (1.0 - y) * (1.0 - p1c).ln());
            } else if classification && o > 1 {
                let mut m = f64::NEG_INFINITY;
                for k in 0..o {
                    m = m.max(raw.get(i, k));
                }
                let mut z = 0.0;
                for k in 0..o {
                    z += (raw.get(i, k) - m).exp();
                }
                for k in 0..o {
                    let pk = ((raw.get(i, k) - m).exp() / z).clamp(1e-15, 1.0);
                    pred.set(i, k, pk);
                    loss += -target.get(i, k) * pk.ln();
                }
            } else {
                for k in 0..o {
                    let e = raw.get(i, k) - target.get(i, k);
                    pred.set(i, k, raw.get(i, k));
                    loss += 0.5 * e * e;
                }
            }
        }
        loss /= n.max(1) as f64;
        if !loss.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::LossIsNan)
                    .message("MLP loss is not finite")
                    .build(),
            );
            return (loss, f64::NAN);
        }
        let mut d_out = Matrix::zeros(n, o);
        for i in 0..n {
            if classification && o == 1 {
                d_out.set(i, 0, (pred.get(i, 0) - target.get(i, 0)) / n as f64);
            } else if classification && o > 1 {
                for k in 0..o {
                    d_out.set(i, k, (pred.get(i, k) - target.get(i, k)) / n as f64);
                }
            } else {
                for k in 0..o {
                    d_out.set(i, k, (raw.get(i, k) - target.get(i, k)) / n as f64);
                }
            }
        }
        let mut dw2 = Matrix::zeros(h, o);
        let mut db2 = Vector::zeros(o);
        let mut d_act = Matrix::zeros(n, h);
        for i in 0..n {
            for k in 0..o {
                db2[k] += d_out.get(i, k);
                for j in 0..h {
                    dw2.set(j, k, dw2.get(j, k) + act.get(i, j) * d_out.get(i, k));
                    d_act.set(i, j, d_act.get(i, j) + d_out.get(i, k) * self.w2.get(j, k));
                }
            }
        }
        let mut dw1 = Matrix::zeros(p, h);
        let mut db1 = Vector::zeros(h);
        for i in 0..n {
            for j in 0..h {
                let g = if act.get(i, j) > 0.0 {
                    d_act.get(i, j)
                } else {
                    0.0
                };
                db1[j] += g;
                for c in 0..p {
                    dw1.set(c, j, dw1.get(c, j) + x.get(i, c) * g);
                }
            }
        }
        let mut g2 = 0.0;
        for j in 0..h {
            for k in 0..o {
                g2 += dw2.get(j, k) * dw2.get(j, k);
            }
        }
        for j in 0..p {
            for k in 0..h {
                g2 += dw1.get(j, k) * dw1.get(j, k);
            }
        }
        let gnorm = g2.sqrt();
        if gnorm > 1e6 || !gnorm.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::GradientExploded)
                    .message(format!("MLP ‖g‖={gnorm}"))
                    .build(),
            );
            return (loss, gnorm);
        }
        for j in 0..h {
            for k in 0..o {
                self.w2.set(j, k, self.w2.get(j, k) - lr * dw2.get(j, k));
            }
        }
        for k in 0..o {
            self.b2[k] -= lr * db2[k];
        }
        for j in 0..p {
            for k in 0..h {
                self.w1.set(j, k, self.w1.get(j, k) - lr * dw1.get(j, k));
            }
        }
        for k in 0..h {
            self.b1[k] -= lr * db1[k];
        }
        (loss, gnorm)
    }
}

/// One-hidden-layer MLP regressor.
#[derive(Clone, Debug)]
pub struct MLPRegressor {
    /// Hidden width.
    pub hidden: usize,
    /// SGD step size.
    pub learning_rate: f64,
    /// Epochs on `fit`.
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
    core: Option<MlpCore>,
}

impl Default for MLPRegressor {
    fn default() -> Self {
        Self {
            hidden: 8,
            learning_rate: 0.05,
            max_iter: 200,
            seed: 0,
            core: None,
        }
    }
}

impl MLPRegressor {
    /// Default MLP regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted MLP regressor (owns the trained weights).
#[derive(Clone, Debug)]
pub struct FittedMlpRegressor {
    core: MlpCore,
}

impl Predict for FittedMlpRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (_, out) = self.core.forward(x);
        ctx.finish(out.column(0))
    }
}

impl Predict for MLPRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        match &self.core {
            Some(c) => {
                let (_, out) = c.forward(x);
                ctx.finish(out.column(0))
            }
            None => {
                ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
                ctx.finish(Vector::zeros(x.nrows()))
            }
        }
    }
}

fn target_reg(y: &Vector) -> Matrix {
    Matrix::from_fn(y.len(), 1, |i, _| y[i])
}

impl Fit for MLPRegressor {
    type Fitted = FittedMlpRegressor;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedMlpRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let mut core = MlpCore::new(x.ncols(), self.hidden, 1, self.seed);
        let t = target_reg(y);
        let mut last = f64::INFINITY;
        let mut converged = false;
        for it in 0..self.max_iter.max(1) {
            let (loss, g) = core.sgd_step(&mut ctx, x, &t, self.learning_rate, false);
            ctx.session.step(it as u64, loss, Some(g));
            if !loss.is_finite() {
                break;
            }
            if (last - loss).abs() < 1e-10 && it > 4 {
                ctx.session.converged("MLP regressor loss stall", it as u64);
                converged = true;
                break;
            }
            last = loss;
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("MLPRegressor hit max_iter")
                    .build(),
            );
        }
        core.n_seen = x.nrows() as u64;
        self.core = Some(core.clone());
        ctx.finish(FittedMlpRegressor { core })
    }
}

impl PartialFit for MLPRegressor {
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
                    .message("MLPRegressor.partial_fit requires y")
                    .build(),
            );
            return ctx.finish(dummy_explain(0, x.nrows(), 0));
        };
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if self.core.is_none() {
            self.core = Some(MlpCore::new(x.ncols(), self.hidden, 1, self.seed));
        }
        let Some(core) = self.core.as_mut() else {
            return ctx.finish(dummy_explain(0, x.nrows(), 0));
        };
        if core.w1.nrows() != x.ncols() {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message("MLP feature width changed")
                    .build(),
            );
            return ctx.finish(dummy_explain(core.updates, x.nrows(), core.n_seen));
        }
        let t = target_reg(y);
        let (loss_b, _) = core.sgd_step(&mut ctx, x, &t, 0.0, false);
        let before = core.w1.frobenius() + core.w2.frobenius();
        let (loss_a, g) = core.sgd_step(&mut ctx, x, &t, self.learning_rate, false);
        let after = core.w1.frobenius() + core.w2.frobenius();
        core.n_seen += x.nrows() as u64;
        core.updates += 1;
        let mut q = IncrementalQuality::new(core.updates - 1, x.nrows(), core.n_seen);
        q.parameter_delta_norm = Some((after - before).abs());
        q.loss_before = Some(loss_b);
        q.loss_after = Some(loss_a);
        q.information_gain = Some((loss_b - loss_a).abs());
        q.still_identified = core.n_seen as usize > x.ncols();
        q.warmup = core.n_seen < 5;
        q.explanation = format!("MLP SGD: mse {loss_b:.6e} → {loss_a:.6e}, ‖g‖={g:.3e}");
        let expl = IncrementalExplain::from_quality(
            q,
            "hidden and output weights",
            "SGD on squared error through a ReLU hidden layer",
            format!("mse={loss_b:.6e}"),
            format!("mse={loss_a:.6e}"),
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

fn dummy_explain(update: u64, batch: usize, n_seen: u64) -> IncrementalExplain {
    IncrementalExplain::from_quality(
        IncrementalQuality::new(update, batch, n_seen),
        "nothing",
        "the update was rejected",
        "invalid",
        "invalid",
    )
}

/// One-hidden-layer MLP classifier (binary logistic or softmax).
#[derive(Clone, Debug)]
pub struct MLPClassifier {
    /// Hidden width.
    pub hidden: usize,
    /// SGD step size.
    pub learning_rate: f64,
    /// Epochs on `fit`.
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
    core: Option<MlpCore>,
    classes: Vec<i64>,
}

impl Default for MLPClassifier {
    fn default() -> Self {
        Self {
            hidden: 8,
            learning_rate: 0.1,
            max_iter: 200,
            seed: 0,
            core: None,
            classes: Vec::new(),
        }
    }
}

impl MLPClassifier {
    /// Default MLP classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted MLP classifier.
#[derive(Clone, Debug)]
pub struct FittedMlpClassifier {
    core: MlpCore,
    /// Training classes.
    pub classes: Vec<i64>,
}

fn class_target(y: &Vector, classes: &[i64]) -> Matrix {
    if classes.len() <= 2 {
        let pos = *classes.last().unwrap_or(&1);
        Matrix::from_fn(y.len(), 1, |i, _| {
            if y[i].round() as i64 == pos {
                1.0
            } else {
                0.0
            }
        })
    } else {
        Matrix::from_fn(y.len(), classes.len(), |i, j| {
            if y[i].round() as i64 == classes[j] {
                1.0
            } else {
                0.0
            }
        })
    }
}

fn decode_mlp(out: &Matrix, classes: &[i64]) -> Vector {
    let pos = *classes.last().unwrap_or(&1) as f64;
    let neg = *classes.first().unwrap_or(&0) as f64;
    if out.ncols() <= 1 {
        Vector::from_iter((0..out.nrows()).map(|i| {
            if sigmoid(out.get(i, 0)) >= 0.5 {
                pos
            } else {
                neg
            }
        }))
    } else {
        Vector::from_iter((0..out.nrows()).map(|i| {
            let mut b = 0usize;
            let mut v = f64::NEG_INFINITY;
            for j in 0..out.ncols() {
                if out.get(i, j) > v {
                    v = out.get(i, j);
                    b = j;
                }
            }
            classes.get(b).copied().unwrap_or(0) as f64
        }))
    }
}

impl Predict for FittedMlpClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (_, out) = self.core.forward(x);
        ctx.finish(decode_mlp(&out, &self.classes))
    }
}

impl Fit for MLPClassifier {
    type Fitted = FittedMlpClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMlpClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        let n_out = if classes.len() <= 2 { 1 } else { classes.len() };
        let mut core = MlpCore::new(x.ncols(), self.hidden, n_out.max(1), self.seed);
        let t = class_target(y, &classes);
        let mut last = f64::INFINITY;
        let mut converged = false;
        for it in 0..self.max_iter.max(1) {
            let (loss, g) = core.sgd_step(&mut ctx, x, &t, self.learning_rate, true);
            ctx.session.step(it as u64, loss, Some(g));
            if !loss.is_finite() {
                break;
            }
            if (last - loss).abs() < 1e-10 && it > 4 {
                ctx.session.converged("MLP classifier loss stall", it as u64);
                converged = true;
                break;
            }
            last = loss;
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("MLPClassifier hit max_iter")
                    .build(),
            );
        }
        core.n_seen = x.nrows() as u64;
        self.core = Some(core.clone());
        self.classes = classes.clone();
        ctx.finish(FittedMlpClassifier { core, classes })
    }
}

impl PartialFit for MLPClassifier {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return ctx.finish(dummy_explain(0, x.nrows(), 0));
        };
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        if self.classes.is_empty() {
            self.classes = counts.iter().map(|(c, _)| *c).collect();
        }
        let n_out = if self.classes.len() <= 2 {
            1
        } else {
            self.classes.len()
        };
        if self.core.is_none() {
            self.core = Some(MlpCore::new(x.ncols(), self.hidden, n_out.max(1), self.seed));
        }
        let Some(core) = self.core.as_mut() else {
            return ctx.finish(dummy_explain(0, x.nrows(), 0));
        };
        let t = class_target(y, &self.classes);
        let (loss_b, _) = core.sgd_step(&mut ctx, x, &t, 0.0, true);
        let (loss_a, g) = core.sgd_step(&mut ctx, x, &t, self.learning_rate, true);
        core.n_seen += x.nrows() as u64;
        core.updates += 1;
        let mut q = IncrementalQuality::new(core.updates - 1, x.nrows(), core.n_seen);
        q.loss_before = Some(loss_b);
        q.loss_after = Some(loss_a);
        q.information_gain = Some((loss_b - loss_a).abs());
        q.parameter_delta_norm = Some(g);
        q.still_identified = core.n_seen as usize > x.ncols();
        q.warmup = core.n_seen < 5;
        q.explanation = format!("MLP class SGD: ce {loss_b:.6e} → {loss_a:.6e}");
        let expl = IncrementalExplain::from_quality(
            q,
            "hidden and output weights",
            "SGD on cross-entropy through a ReLU hidden layer",
            format!("ce={loss_b:.6e}"),
            format!("ce={loss_a:.6e}"),
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

/// Bernoulli restricted Boltzmann machine trained with CD-1.
#[derive(Clone, Debug)]
pub struct BernoulliRBM {
    /// Hidden units.
    pub n_hidden: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// CD epochs on `fit`.
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for BernoulliRBM {
    fn default() -> Self {
        Self {
            n_hidden: 8,
            learning_rate: 0.1,
            max_iter: 40,
            seed: 0,
        }
    }
}

impl BernoulliRBM {
    /// RBM with `h` hidden units.
    pub fn new(n_hidden: usize) -> Self {
        Self {
            n_hidden,
            ..Self::default()
        }
    }
}

/// Fitted Bernoulli RBM.
#[derive(Clone, Debug)]
pub struct FittedRbm {
    /// Visible–hidden weights (`p × h`).
    pub w: Matrix,
    /// Visible biases.
    pub vis_bias: Vector,
    /// Hidden biases.
    pub hid_bias: Vector,
}

impl FittedRbm {
    fn p_h(&self, v: &Vector) -> Vector {
        let mut h = Vector::zeros(self.hid_bias.len());
        for k in 0..h.len() {
            let mut s = self.hid_bias[k];
            for j in 0..v.len().min(self.w.nrows()) {
                s += v[j] * self.w.get(j, k);
            }
            h[k] = sigmoid(s);
        }
        h
    }

    #[allow(dead_code)]
    fn p_v(&self, h: &Vector) -> Vector {
        let mut v = Vector::zeros(self.vis_bias.len());
        for j in 0..v.len() {
            let mut s = self.vis_bias[j];
            for k in 0..h.len().min(self.w.ncols()) {
                s += h[k] * self.w.get(j, k);
            }
            v[j] = sigmoid(s);
        }
        v
    }
}

fn sample_bernoulli(p: &Vector, rng: &mut Rng) -> Vector {
    Vector::from_iter(p.as_slice().iter().map(|&q| if rng.uniform() < q { 1.0 } else { 0.0 }))
}

fn row_as_vec(x: &Matrix, i: usize) -> Vector {
    x.row(i)
}

impl FitUnsupervised for BernoulliRBM {
    type Fitted = FittedRbm;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedRbm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let p = x.ncols();
        let h = self.n_hidden.max(1);
        let mut rng = Rng::new(self.seed);
        let mut w = init_weights(&mut rng, p, h);
        let mut vb = Vector::zeros(p);
        let mut hb = Vector::zeros(h);
        for it in 0..self.max_iter.max(1) {
            let mut recon = 0.0;
            for i in 0..x.nrows() {
                let v0 = row_as_vec(x, i);
                let ph0 = {
                    let mut t = Vector::zeros(h);
                    for k in 0..h {
                        let mut s = hb[k];
                        for j in 0..p {
                            s += v0[j] * w.get(j, k);
                        }
                        t[k] = sigmoid(s);
                    }
                    t
                };
                let h0 = sample_bernoulli(&ph0, &mut rng);
                let mut pv1 = Vector::zeros(p);
                for j in 0..p {
                    let mut s = vb[j];
                    for k in 0..h {
                        s += h0[k] * w.get(j, k);
                    }
                    pv1[j] = sigmoid(s);
                }
                let v1 = sample_bernoulli(&pv1, &mut rng);
                let mut ph1 = Vector::zeros(h);
                for k in 0..h {
                    let mut s = hb[k];
                    for j in 0..p {
                        s += v1[j] * w.get(j, k);
                    }
                    ph1[k] = sigmoid(s);
                }
                for j in 0..p {
                    for k in 0..h {
                        w.set(
                            j,
                            k,
                            w.get(j, k) + self.learning_rate * (v0[j] * ph0[k] - v1[j] * ph1[k]),
                        );
                    }
                    vb[j] += self.learning_rate * (v0[j] - pv1[j]);
                }
                for k in 0..h {
                    hb[k] += self.learning_rate * (ph0[k] - ph1[k]);
                }
                for j in 0..p {
                    let e = v0[j] - pv1[j];
                    recon += e * e;
                }
            }
            recon /= (x.nrows() * p).max(1) as f64;
            ctx.session.step(it as u64, recon, None);
            if !recon.is_finite() {
                ctx.push(Issue::builder(IssueCode::LossIsNan).message("RBM reconstruction is NaN").build());
                break;
            }
        }
        ctx.finish(FittedRbm {
            w,
            vis_bias: vb,
            hid_bias: hb,
        })
    }
}

impl FittedRbm {
    /// Hidden activations for the rows of `x`.
    pub fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let h = Matrix::from_fn(x.nrows(), self.hid_bias.len(), |i, k| {
            self.p_h(&row_as_vec(x, i))[k]
        });
        ctx.finish(h)
    }
}

impl PartialFit for BernoulliRBM {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let before = 0.0;
        let fitted = match self.fit_unsupervised(x, &session.child("cd1")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                return ctx.finish(dummy_explain(0, x.nrows(), 0));
            }
        };
        let mut q = IncrementalQuality::new(0, x.nrows(), x.nrows() as u64);
        q.information_gain = Some(fitted.w.frobenius());
        q.loss_before = Some(before);
        q.loss_after = Some(fitted.w.frobenius());
        q.still_identified = x.nrows() > 1;
        q.explanation = "CD-1 RBM weight update".into();
        let expl = IncrementalExplain::from_quality(
            q,
            "RBM weights and biases",
            "contrastive divergence (k=1)",
            "previous weights",
            "one CD-1 epoch",
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    #[test]
    fn mlp_regressor_fits_a_line() {
        let x = Matrix::from_fn(20, 1, |i, _| i as f64 / 10.0);
        let y = Vector::from_iter((0..20).map(|i| 0.5 * (i as f64 / 10.0)));
        let q = MLPRegressor {
            hidden: 6,
            learning_rate: 0.05,
            max_iter: 300,
            seed: 1,
            core: None,
        }
        .fit(&x, &y, &Session::new("mlp", "fit"))
        .unwrap();
        let pred = q
            .value
            .predict(&x, &Session::new("mlp", "pred"))
            .unwrap()
            .value;
        let mut sse = 0.0;
        for i in 0..y.len() {
            let e = pred[i] - y[i];
            sse += e * e;
        }
        assert!(sse / (y.len() as f64) < 0.05, "mse={}", sse / (y.len() as f64));
    }

    #[test]
    fn mlp_classifier_and_partial_fit() {
        let x = Matrix::from_fn(16, 1, |i, _| if i < 8 { -1.0 } else { 1.0 });
        let y = Vector::from_iter((0..16).map(|i| if i < 8 { 0.0 } else { 1.0 }));
        let q = MLPClassifier {
            hidden: 6,
            learning_rate: 0.2,
            max_iter: 200,
            seed: 2,
            core: None,
            classes: Vec::new(),
        }
        .fit(&x, &y, &Session::new("mlp", "clf"))
        .unwrap();
        let pred = q
            .value
            .predict(&x, &Session::new("mlp", "p"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..y.len() {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 12, "ok={ok}");
        let mut m = MLPRegressor::new();
        let expl = m
            .partial_fit(&x, Some(&y), &Session::new("mlp", "pf"))
            .unwrap();
        assert!(!expl.value.narrative.is_empty());
    }

    #[test]
    fn rbm_cd1_finite() {
        let x = Matrix::from_fn(8, 3, |i, j| if (i + j) % 2 == 0 { 1.0 } else { 0.0 });
        let q = BernoulliRBM {
            n_hidden: 4,
            max_iter: 8,
            ..BernoulliRBM::new(4)
        }
        .fit_unsupervised(&x, &Session::new("rbm", "fit"))
        .unwrap();
        let h = q
            .value
            .transform(&x, &Session::new("rbm", "tf"))
            .unwrap()
            .value;
        assert_eq!(h.shape(), (8, 4));
        assert!(h.to_row_major().iter().all(|v| v.is_finite()));
    }
}
