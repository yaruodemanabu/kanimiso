//! Naive Bayes: Gaussian, multinomial, Bernoulli, and complement.
//!
//! [`GaussianNB`] supports [`crate::traits::PartialFit`] with a mandatory
//! [`ojizou_san::IncrementalExplain`]: class-conditional mean / variance
//! Welford updates, effective sample size, and
//! [`signlred::IssueCode::LabelSpaceExpandedOnline`] when a new class appears.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::{Fit, PartialFit, Predict};
use crate::validate::{inspect_classes, inspect_identification, inspect_xy};
use ojizou_san::{IncrementalExplain, Session};
use signlred::{IncrementalQuality, Issue, IssueCode, Meaninglessness, Qualified, Result};

fn labels_of(y: &Vector) -> Vec<i64> {
    y.as_slice()
        .iter()
        .map(|&v| if v.is_finite() { v.round() as i64 } else { 0 })
        .collect()
}

fn class_index(lab: i64, classes: &[i64]) -> Option<usize> {
    classes.iter().position(|&c| c == lab)
}

#[allow(dead_code)]
fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0;
    for &v in xs {
        s += (v - m).exp();
    }
    m + s.ln()
}

fn argmax_label(classes: &[i64], scores: &[f64]) -> i64 {
    let mut best = 0usize;
    for i in 1..scores.len() {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    classes.get(best).copied().unwrap_or(0)
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
                .message("Naive Bayes predictions are a constant while y is not")
                .meaninglessness(Meaninglessness::vacuous(
                    "class-conditional Naive Bayes",
                    "every in-sample label collapsed to one class",
                    "check class-conditional variance and priors",
                ))
                .build(),
        );
    }
}

fn log_normal(x: f64, mean: f64, var: f64) -> f64 {
    let v = var.max(1e-15);
    -0.5 * ((x - mean) * (x - mean) / v + v.ln() + (2.0 * std::f64::consts::PI).ln())
}

/// Gaussian Naive Bayes with online (Welford) class-conditional updates.
#[derive(Clone, Debug)]
pub struct GaussianNB {
    /// Fraction of the global feature variance added to every class variance.
    pub var_smoothing: f64,
    n_features: Option<usize>,
    classes: Vec<i64>,
    class_n: Vec<f64>,
    mean: Vec<Vec<f64>>,
    m2: Vec<Vec<f64>>,
    feat_n: f64,
    feat_mean: Vec<f64>,
    feat_m2: Vec<f64>,
    n_seen: u64,
    updates: u64,
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self {
            var_smoothing: 1e-9,
            n_features: None,
            classes: Vec::new(),
            class_n: Vec::new(),
            mean: Vec::new(),
            m2: Vec::new(),
            feat_n: 0.0,
            feat_mean: Vec::new(),
            feat_m2: Vec::new(),
            n_seen: 0,
            updates: 0,
        }
    }
}

impl GaussianNB {
    /// Default Gaussian NB (`var_smoothing = 1e-9`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Effective sample size after all `partial_fit` / `fit` calls.
    pub fn n_eff(&self) -> f64 {
        self.n_seen as f64
    }

    /// Sorted labels seen so far.
    pub fn classes(&self) -> &[i64] {
        &self.classes
    }

    /// Class-conditional means (`K × p`), or `None` before the first update.
    pub fn class_means(&self) -> Option<Matrix> {
        let p = self.n_features?;
        let k = self.classes.len();
        Some(Matrix::from_fn(k, p, |c, j| self.mean[c][j]))
    }

    /// Smoothed class-conditional variances (`K × p`).
    pub fn class_vars(&self) -> Option<Matrix> {
        let p = self.n_features?;
        let k = self.classes.len();
        let smooth = self.smoothing();
        Some(Matrix::from_fn(k, p, |c, j| self.raw_var(c, j) + smooth))
    }

    fn smoothing(&self) -> f64 {
        let p = self.feat_mean.len();
        let mut vmax = 0.0;
        for j in 0..p {
            let v = if self.feat_n > 1.0 {
                self.feat_m2[j] / self.feat_n
            } else {
                0.0
            };
            if v > vmax {
                vmax = v;
            }
        }
        self.var_smoothing * vmax.max(1e-12)
    }

    fn raw_var(&self, c: usize, j: usize) -> f64 {
        let n = self.class_n[c];
        if n <= 0.0 {
            0.0
        } else {
            (self.m2[c][j] / n).max(0.0)
        }
    }

    fn ensure_class(&mut self, lab: i64, p: usize) -> (usize, bool) {
        if let Some(i) = class_index(lab, &self.classes) {
            return (i, false);
        }
        let pos = self
            .classes
            .iter()
            .position(|&c| c > lab)
            .unwrap_or(self.classes.len());
        self.classes.insert(pos, lab);
        self.class_n.insert(pos, 0.0);
        self.mean.insert(pos, vec![0.0; p]);
        self.m2.insert(pos, vec![0.0; p]);
        (pos, true)
    }

    fn log_scores(&self, x: &Matrix, i: usize) -> Vec<f64> {
        let p = self.n_features.unwrap_or(0);
        let ntot: f64 = self.class_n.iter().sum::<f64>().max(1e-15);
        let smooth = self.smoothing();
        self.classes
            .iter()
            .enumerate()
            .map(|(c, _)| {
                let prior = (self.class_n[c] / ntot).max(1e-15).ln();
                let mut s = prior;
                for j in 0..p.min(x.ncols()) {
                    s += log_normal(x.get(i, j), self.mean[c][j], self.raw_var(c, j) + smooth);
                }
                s
            })
            .collect()
    }

    fn predict_vec(&self, x: &Matrix) -> Vector {
        Vector::from_iter((0..x.nrows()).map(|i| {
            let s = self.log_scores(x, i);
            argmax_label(&self.classes, &s) as f64
        }))
    }

    fn snapshot_state(&self) -> (Vec<i64>, Vec<Vec<f64>>) {
        (self.classes.clone(), self.mean.clone())
    }

    fn to_fitted(&self) -> FittedGaussianNB {
        let p = self.n_features.unwrap_or(0);
        let k = self.classes.len();
        let ntot: f64 = self.class_n.iter().sum::<f64>().max(1e-15);
        let prior = Vector::from_iter(self.class_n.iter().map(|n| n / ntot));
        let smooth = self.smoothing();
        let theta = Matrix::from_fn(k.max(1), p, |c, j| {
            if c < self.mean.len() {
                self.mean[c][j]
            } else {
                0.0
            }
        });
        let var = Matrix::from_fn(k.max(1), p, |c, j| {
            if c < self.mean.len() {
                self.raw_var(c, j) + smooth
            } else {
                smooth
            }
        });
        FittedGaussianNB {
            classes: self.classes.clone(),
            class_prior: prior,
            theta,
            var,
            n_eff: self.n_eff(),
        }
    }
}

/// Batch-fitted Gaussian NB (also produced by [`GaussianNB::fit`]).
#[derive(Clone, Debug)]
pub struct FittedGaussianNB {
    /// Sorted unique labels.
    pub classes: Vec<i64>,
    /// Class priors.
    pub class_prior: Vector,
    /// Class-conditional means (`K × p`).
    pub theta: Matrix,
    /// Smoothed class-conditional variances (`K × p`).
    pub var: Matrix,
    /// Effective sample size at fit time.
    pub n_eff: f64,
}

impl FittedGaussianNB {
    fn predict_vec(&self, x: &Matrix) -> Vector {
        Vector::from_iter((0..x.nrows()).map(|i| {
            let mut scores = Vec::with_capacity(self.classes.len());
            for c in 0..self.classes.len() {
                let mut s = self.class_prior[c].max(1e-15).ln();
                for j in 0..x.ncols().min(self.theta.ncols()) {
                    s += log_normal(x.get(i, j), self.theta.get(c, j), self.var.get(c, j));
                }
                scores.push(s);
            }
            argmax_label(&self.classes, &scores) as f64
        }))
    }
}

impl Predict for FittedGaussianNB {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.theta.ncols());
        ctx.finish(self.predict_vec(x))
    }
}

impl Predict for GaussianNB {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.n_features.is_none() {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        predict_shape_guard(&mut ctx, x, self.n_features.unwrap_or(0));
        ctx.finish(self.predict_vec(x))
    }
}

impl Fit for GaussianNB {
    type Fitted = FittedGaussianNB;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedGaussianNB>> {
        *self = GaussianNB {
            var_smoothing: self.var_smoothing,
            ..GaussianNB::default()
        };
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        if counts.len() < 2 {
            return ctx.finish(self.to_fitted());
        }
        // Drive the same Welford path as partial_fit so n_eff / means stay consistent.
        let expl = apply_welford(self, x, y, &mut ctx, false);
        let _ = expl;
        for (c, n) in self.class_n.iter().enumerate() {
            if *n < 2.0 {
                ctx.push(
                    Issue::builder(IssueCode::DegenerateDistribution)
                        .message(format!(
                            "class {} has n_eff={n}; class-conditional variance is unidentified before smoothing",
                            self.classes[c]
                        ))
                        .build(),
                );
            }
        }
        let fitted = self.to_fitted();
        let pred = fitted.predict_vec(x);
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

fn apply_welford(
    model: &mut GaussianNB,
    x: &Matrix,
    y: &Vector,
    ctx: &mut FitCtx,
    online: bool,
) -> IncrementalExplain {
    let p = x.ncols();
    let (before_classes, before_means) = model.snapshot_state();
    let n_before = model.n_seen;
    let k_before = before_classes.len();
    let mut new_classes = 0usize;
    if model.n_features.is_none() {
        model.n_features = Some(p);
        model.feat_mean = vec![0.0; p];
        model.feat_m2 = vec![0.0; p];
    } else if model.n_features != Some(p) {
        ctx.push(
            Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                .message(format!(
                    "partial_fit X has {p} columns; model has {:?}",
                    model.n_features
                ))
                .build(),
        );
        let q = IncrementalQuality::new(model.updates, x.nrows(), model.n_seen);
        return IncrementalExplain::from_quality(
            q,
            "nothing",
            "feature space changed; the batch was rejected",
            format!("n_eff={}", n_before),
            format!("n_eff={}", model.n_seen),
        );
    }
    let ylab = labels_of(y);
    for i in 0..x.nrows() {
        if !y[i].is_finite() {
            continue;
        }
        let lab = ylab[i];
        let (c, is_new) = model.ensure_class(lab, p);
        if is_new {
            new_classes += 1;
        }
        model.class_n[c] += 1.0;
        let n_c = model.class_n[c];
        model.feat_n += 1.0;
        let feat_n = model.feat_n;
        for j in 0..p {
            let v = x.get(i, j);
            let d = v - model.mean[c][j];
            model.mean[c][j] += d / n_c;
            let d2 = v - model.mean[c][j];
            model.m2[c][j] += d * d2;
            let fd = v - model.feat_mean[j];
            model.feat_mean[j] += fd / feat_n;
            model.feat_m2[j] += fd * (v - model.feat_mean[j]);
        }
        model.n_seen += 1;
    }
    model.updates += 1;
    if new_classes > 0 && online && k_before > 0 {
        ctx.push(
            Issue::builder(IssueCode::LabelSpaceExpandedOnline)
                .message(format!(
                    "{new_classes} new class(es) appeared; earlier probability vectors are incomparable"
                ))
                .metric("n_new_classes", new_classes as f64)
                .metric("n_classes_after", model.classes.len() as f64)
                .build(),
        );
    }
    let mut delta = 0.0_f64;
    let mut max_d = 0.0_f64;
    let mut top = Vec::new();
    for (c, lab) in model.classes.iter().enumerate() {
        let prev = class_index(*lab, &before_classes).and_then(|old| before_means.get(old));
        let mut nrm = 0.0;
        match prev {
            Some(old) => {
                for j in 0..p.min(old.len()) {
                    let d = model.mean[c][j] - old[j];
                    nrm += d * d;
                }
            }
            None => {
                for j in 0..p {
                    nrm += model.mean[c][j] * model.mean[c][j];
                }
            }
        }
        nrm = nrm.sqrt();
        delta += nrm;
        max_d = max_d.max(nrm);
        top.push((format!("class{lab}_mean"), nrm));
    }
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top.truncate(4);
    let n_eff = model.n_seen as f64;
    let identified = model.class_n.iter().all(|n| *n >= 2.0) && !model.class_n.is_empty();
    let mut q = IncrementalQuality::new(model.updates.saturating_sub(1), x.nrows(), model.n_seen);
    q.effective_sample_size = n_eff;
    q.parameter_delta_norm = Some(delta);
    q.parameter_delta_max = Some(max_d);
    q.top_moved_parameters = top;
    q.information_gain = Some(delta);
    q.still_identified = identified;
    q.warmup = model.n_seen < 5;
    q.explanation = format!(
        "Welford class-conditional Gaussian update: Δμ_ℓ2={delta:.4e}, n_eff={n_eff}, classes={:?}",
        model.classes
    );
    IncrementalExplain::from_quality(
        q,
        format!(
            "class-conditional means and variances for {} classes (n_eff={n_eff})",
            model.classes.len()
        ),
        "Welford online update of per-class feature mean / M2; variance = M2 / n_c plus var_smoothing",
        format!("{} classes, n_seen={n_before}", k_before),
        format!("{} classes, n_seen={}", model.classes.len(), model.n_seen),
    )
}

impl PartialFit for GaussianNB {
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
                    .message("GaussianNB.partial_fit requires y")
                    .build(),
            );
            return ctx.finish(IncrementalExplain::from_quality(
                IncrementalQuality::new(self.updates, 0, self.n_seen),
                "nothing",
                "the update was rejected (missing y)",
                "invalid",
                "invalid",
            ));
        };
        // Scan X only: a single-class online batch is valid and must not abort as ConstantTarget.
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if y.len() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!("y.len()={} but X has {} rows", y.len(), x.nrows()))
                    .build(),
            );
        }
        if let Some(issue) = signlred::scan_finite(y.as_slice()).to_issue("y") {
            ctx.push(issue);
        }
        if self.n_features.is_none() {
            inspect_classes(&mut ctx.report, y, &ctx.policy);
        }
        let expl = apply_welford(self, x, y, &mut ctx, true);
        if expl.quality.is_uninformative(ctx.policy.uninformative_info_eps) {
            ctx.push(
                Issue::builder(IssueCode::UpdateWithZeroInformation)
                    .incremental(expl.quality.clone())
                    .message("this GaussianNB batch did not move class means")
                    .build(),
            );
        }
        if !expl.quality.still_identified {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(expl.quality.clone())
                    .message("at least one class has n_eff < 2; variances are unidentified")
                    .build(),
            );
        }
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

/// Multinomial Naive Bayes (count features, Laplace / Lidstone smoothing).
#[derive(Clone, Debug)]
pub struct MultinomialNB {
    /// Additive smoothing \(\alpha \ge 0\).
    pub alpha: f64,
}

impl Default for MultinomialNB {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl MultinomialNB {
    /// Multinomial NB with the given smoothing.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

/// Fitted multinomial / complement / Bernoulli NB.
#[derive(Clone, Debug)]
pub struct FittedDiscreteNB {
    /// Sorted unique labels.
    pub classes: Vec<i64>,
    /// Log class priors.
    pub log_prior: Vector,
    /// Log feature parameters (`K × p`). For complement NB this is the
    /// complement weight matrix (predict uses a min inner product).
    pub feature_log_prob: Matrix,
    /// If true, predict \(\arg\min_c w_c^\top x\) (complement).
    pub complement: bool,
}

impl FittedDiscreteNB {
    fn predict_vec(&self, x: &Matrix) -> Vector {
        Vector::from_iter((0..x.nrows()).map(|i| {
            let mut scores = Vec::with_capacity(self.classes.len());
            for c in 0..self.classes.len() {
                let mut s = self.log_prior[c];
                for j in 0..x.ncols().min(self.feature_log_prob.ncols()) {
                    s += x.get(i, j) * self.feature_log_prob.get(c, j);
                }
                if self.complement {
                    s = -s;
                }
                scores.push(s);
            }
            argmax_label(&self.classes, &scores) as f64
        }))
    }
}

impl Predict for FittedDiscreteNB {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.feature_log_prob.ncols());
        ctx.finish(self.predict_vec(x))
    }
}

fn fit_count_nb(
    ctx: &mut FitCtx,
    x: &Matrix,
    y: &Vector,
    alpha: f64,
    complement: bool,
) -> FittedDiscreteNB {
    let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
    inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
    let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
    let ylab = labels_of(y);
    let k = classes.len();
    let p = x.ncols();
    let mut fc = Matrix::zeros(k.max(1), p);
    let mut class_n = vec![0.0; k.max(1)];
    let mut neg = false;
    for i in 0..x.nrows() {
        let Some(c) = class_index(ylab[i], &classes) else {
            continue;
        };
        class_n[c] += 1.0;
        for j in 0..p {
            let v = x.get(i, j);
            if v < 0.0 {
                neg = true;
            }
            fc.set(c, j, fc.get(c, j) + v);
        }
    }
    if neg {
        ctx.push(
            Issue::builder(IssueCode::NonPositiveSeries)
                .message("count / Bernoulli Naive Bayes saw a negative feature")
                .build(),
        );
    }
    if !alpha.is_finite() || alpha < 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("smoothing α={alpha} is not a finite non-negative number"))
                .build(),
        );
    }
    let ntot: f64 = class_n.iter().sum::<f64>().max(1e-15);
    let mut log_prior = Vector::zeros(k.max(1));
    for c in 0..k {
        log_prior[c] = if complement {
            0.0
        } else {
            (class_n[c] / ntot).max(1e-15).ln()
        };
    }
    let a = alpha.max(0.0);
    let mut flp = Matrix::zeros(k.max(1), p);
    if complement && k > 0 {
        let mut total = vec![0.0; p];
        for j in 0..p {
            for c in 0..k {
                total[j] += fc.get(c, j);
            }
        }
        for c in 0..k {
            let mut den = 0.0;
            let mut comp = vec![0.0; p];
            for j in 0..p {
                comp[j] = (total[j] - fc.get(c, j) + a).max(0.0);
                den += comp[j];
            }
            den = den.max(1e-15);
            let mut nrm = 0.0;
            for j in 0..p {
                let w = (comp[j] / den).max(1e-15).ln();
                flp.set(c, j, w);
                nrm += w * w;
            }
            nrm = nrm.sqrt().max(1e-15);
            for j in 0..p {
                flp.set(c, j, flp.get(c, j) / nrm);
            }
        }
    } else {
        for c in 0..k {
            let mut den = 0.0;
            for j in 0..p {
                den += fc.get(c, j) + a;
            }
            den = den.max(1e-15);
            for j in 0..p {
                flp.set(c, j, ((fc.get(c, j) + a) / den).max(1e-15).ln());
            }
        }
    }
    FittedDiscreteNB {
        classes,
        log_prior,
        feature_log_prob: flp,
        complement,
    }
}

impl Fit for MultinomialNB {
    type Fitted = FittedDiscreteNB;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedDiscreteNB>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let fitted = fit_count_nb(&mut ctx, x, y, self.alpha, false);
        if !fitted.classes.is_empty() {
            let pred = fitted.predict_vec(x);
            diagnose_constant_predictions(&mut ctx, &pred, y);
        }
        ctx.finish(fitted)
    }
}

/// Bernoulli Naive Bayes (binary features).
#[derive(Clone, Debug)]
pub struct BernoulliNB {
    /// Additive smoothing.
    pub alpha: f64,
    /// Values strictly above this threshold are treated as 1.
    pub binarize: f64,
}

impl Default for BernoulliNB {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            binarize: 0.0,
        }
    }
}

impl BernoulliNB {
    /// Default Bernoulli NB.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Bernoulli NB (stores log \(\theta\) and log \(1-\theta\)).
#[derive(Clone, Debug)]
pub struct FittedBernoulliNB {
    /// Sorted unique labels.
    pub classes: Vec<i64>,
    /// Log class priors.
    pub log_prior: Vector,
    /// \(\log\theta_{c,j}\).
    pub feature_log_prob: Matrix,
    /// \(\log(1-\theta_{c,j})\).
    pub feature_log_neg: Matrix,
    /// Binarization threshold used at fit time.
    pub binarize: f64,
}

impl FittedBernoulliNB {
    fn predict_vec(&self, x: &Matrix) -> Vector {
        Vector::from_iter((0..x.nrows()).map(|i| {
            let mut scores = Vec::with_capacity(self.classes.len());
            for c in 0..self.classes.len() {
                let mut s = self.log_prior[c];
                for j in 0..x.ncols().min(self.feature_log_prob.ncols()) {
                    let bit = if x.get(i, j) > self.binarize { 1.0 } else { 0.0 };
                    s += bit * self.feature_log_prob.get(c, j)
                        + (1.0 - bit) * self.feature_log_neg.get(c, j);
                }
                scores.push(s);
            }
            argmax_label(&self.classes, &scores) as f64
        }))
    }
}

impl Predict for FittedBernoulliNB {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.feature_log_prob.ncols());
        ctx.finish(self.predict_vec(x))
    }
}

impl Fit for BernoulliNB {
    type Fitted = FittedBernoulliNB;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedBernoulliNB>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let ylab = labels_of(y);
        let k = classes.len();
        let p = x.ncols();
        let a = self.alpha.max(0.0);
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("BernoulliNB α={}", self.alpha))
                    .build(),
            );
        }
        let mut ones = Matrix::zeros(k.max(1), p);
        let mut class_n = vec![0.0; k.max(1)];
        for i in 0..x.nrows() {
            let Some(c) = class_index(ylab[i], &classes) else {
                continue;
            };
            class_n[c] += 1.0;
            for j in 0..p {
                if x.get(i, j) > self.binarize {
                    ones.set(c, j, ones.get(c, j) + 1.0);
                }
            }
        }
        let ntot: f64 = class_n.iter().sum::<f64>().max(1e-15);
        let mut log_prior = Vector::zeros(k.max(1));
        let mut flp = Matrix::zeros(k.max(1), p);
        let mut fln = Matrix::zeros(k.max(1), p);
        for c in 0..k {
            log_prior[c] = (class_n[c] / ntot).max(1e-15).ln();
            let den = class_n[c] + 2.0 * a;
            for j in 0..p {
                let th = ((ones.get(c, j) + a) / den.max(1e-15)).clamp(1e-15, 1.0 - 1e-15);
                flp.set(c, j, th.ln());
                fln.set(c, j, (1.0 - th).ln());
            }
        }
        let fitted = FittedBernoulliNB {
            classes,
            log_prior,
            feature_log_prob: flp,
            feature_log_neg: fln,
            binarize: self.binarize,
        };
        if !fitted.classes.is_empty() {
            let pred = fitted.predict_vec(x);
            diagnose_constant_predictions(&mut ctx, &pred, y);
        }
        ctx.finish(fitted)
    }
}

/// Complement Naive Bayes (Rennie et al.) for skewed class priors.
#[derive(Clone, Debug)]
pub struct ComplementNB {
    /// Additive smoothing.
    pub alpha: f64,
}

impl Default for ComplementNB {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl ComplementNB {
    /// Complement NB with the given smoothing.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Fit for ComplementNB {
    type Fitted = FittedDiscreteNB;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedDiscreteNB>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let fitted = fit_count_nb(&mut ctx, x, y, self.alpha, true);
        if !fitted.classes.is_empty() {
            let pred = fitted.predict_vec(x);
            diagnose_constant_predictions(&mut ctx, &pred, y);
        }
        ctx.finish(fitted)
    }
}

// silence unused helper in non-test builds that still document log-sum-exp
#[allow(dead_code)]
fn _log_prob_norm(scores: &[f64]) -> Vec<f64> {
    let z = logsumexp(scores);
    scores.iter().map(|s| s - z).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use ojizou_san::Session;
    use signlred::IssueCode;

    fn accuracy(pred: &Vector, y: &Vector) -> f64 {
        let mut ok = 0usize;
        for i in 0..y.len() {
            if (pred[i].round() - y[i].round()).abs() < 0.5 {
                ok += 1;
            }
        }
        ok as f64 / y.len() as f64
    }

    fn two_gaussians_1d(n_per: usize) -> (Matrix, Vector) {
        let mut rng = Rng::new(21);
        let n = n_per * 2;
        let mut data = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n_per {
            data[i] = 0.25 * rng.standard_normal();
            y[i] = 0.0;
            data[n_per + i] = 5.0 + 0.25 * rng.standard_normal();
            y[n_per + i] = 1.0;
        }
        (Matrix::from_row_major(n, 1, &data), Vector::from_slice(&y))
    }

    #[test]
    fn gaussian_nb_separated_1d() {
        let (x, y) = two_gaussians_1d(20);
        let q = GaussianNB::new()
            .fit(&x, &y, &Session::new("gnb", "fit"))
            .expect("gnb");
        let pred = q
            .value
            .predict(&x, &Session::new("gnb", "predict"))
            .expect("pred")
            .value;
        assert!(accuracy(&pred, &y) > 0.9, "acc={}", accuracy(&pred, &y));
        assert!(q.value.n_eff >= 40.0);
    }

    #[test]
    fn gaussian_nb_constant_y_errors() {
        let x = Matrix::from_fn(12, 1, |i, _| i as f64);
        let y = Vector::filled(12, 0.0);
        let err = GaussianNB::new()
            .fit(&x, &y, &Session::new("gnb", "fit"))
            .unwrap_err();
        assert!(
            err.primary().code == IssueCode::ConstantTarget
                || err.primary().code == IssueCode::SingleClass
        );
    }

    #[test]
    fn gaussian_nb_partial_fit_explains_and_expands() {
        let (x, y) = two_gaussians_1d(12);
        let mut nb = GaussianNB::new();
        let session = Session::new("gnb", "partial_fit");
        let q = nb.partial_fit(&x, Some(&y), &session).expect("pf1");
        assert!(!q.value.narrative.is_empty());
        assert!(q.value.what_changed.contains("means"));
        assert!(q.value.quality.effective_sample_size >= 24.0);
        assert!(session
            .ledger()
            .events()
            .iter()
            .any(|e| e.kind == ojizou_san::EventKind::IncrementalExplanation));

        let mut data = vec![0.0; 10];
        let mut y2 = vec![0.0; 10];
        let mut rng = Rng::new(3);
        for i in 0..10 {
            data[i] = 12.0 + 0.2 * rng.standard_normal();
            y2[i] = 2.0;
        }
        let x2 = Matrix::from_row_major(10, 1, &data);
        let y2 = Vector::from_slice(&y2);
        let session2 = Session::new("gnb", "partial_fit");
        let q2 = nb.partial_fit(&x2, Some(&y2), &session2).expect("pf2");
        assert!(
            q2.report.contains(IssueCode::LabelSpaceExpandedOnline),
            "expected LabelSpaceExpandedOnline, report={}",
            q2.report
        );
        assert_eq!(nb.classes().len(), 3);
    }

    #[test]
    fn multinomial_and_bernoulli_two_class() {
        let x = Matrix::from_fn(10, 2, |i, j| {
            if i < 5 {
                if j == 0 {
                    4.0
                } else {
                    0.0
                }
            } else if j == 1 {
                4.0
            } else {
                0.0
            }
        });
        let y = Vector::from_iter((0..10).map(|i| if i < 5 { 0.0 } else { 1.0 }));
        let q = MultinomialNB::new(1.0)
            .fit(&x, &y, &Session::new("mnb", "fit"))
            .expect("mnb");
        let pred = q.value.predict(&x, &Session::new("mnb", "predict")).unwrap().value;
        assert!(accuracy(&pred, &y) > 0.8);

        let q = BernoulliNB::new()
            .fit(&x, &y, &Session::new("bnb", "fit"))
            .expect("bnb");
        let pred = q.value.predict(&x, &Session::new("bnb", "predict")).unwrap().value;
        assert!(accuracy(&pred, &y) > 0.8);

        let q = ComplementNB::new(1.0)
            .fit(&x, &y, &Session::new("cnb", "fit"))
            .expect("cnb");
        let pred = q.value.predict(&x, &Session::new("cnb", "predict")).unwrap().value;
        assert!(accuracy(&pred, &y) > 0.8);
    }
}
