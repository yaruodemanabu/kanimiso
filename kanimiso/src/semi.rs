//! Graph-based semi-supervised learning: label propagation and label spreading.
//!
//! Unlabeled rows are encoded as `NaN` or `−1` in `y`. Labeled rows are
//! clamped after every iteration of propagation.

use crate::classification::RidgeClassifier;
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::{Fit, Predict};
use crate::validate::{inspect_classes, inspect_xy};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, Qualified, Result};

fn is_unlabeled(v: f64) -> bool {
    !v.is_finite() || (v + 1.0).abs() < 1e-12
}

fn rbf_row(x: &Matrix, i: usize, j: usize, gamma: f64) -> f64 {
    let mut d2 = 0.0;
    for c in 0..x.ncols() {
        let d = x.get(i, c) - x.get(j, c);
        d2 += d * d;
    }
    (-gamma * d2).exp()
}

fn labeled_classes(y: &Vector) -> Vec<i64> {
    let mut c: Vec<i64> = y
        .as_slice()
        .iter()
        .filter(|v| !is_unlabeled(**v))
        .map(|v| v.round() as i64)
        .collect();
    c.sort_unstable();
    c.dedup();
    c
}

fn one_hot(y: &Vector, classes: &[i64]) -> Matrix {
    let k = classes.len();
    Matrix::from_fn(y.len(), k, |i, j| {
        if is_unlabeled(y[i]) {
            0.0
        } else if y[i].round() as i64 == classes[j] {
            1.0
        } else {
            0.0
        }
    })
}

fn normalize_rows(m: &mut Matrix) {
    for i in 0..m.nrows() {
        let mut s = 0.0;
        for j in 0..m.ncols() {
            s += m.get(i, j).max(0.0);
        }
        if s > 0.0 {
            for j in 0..m.ncols() {
                m.set(i, j, m.get(i, j).max(0.0) / s);
            }
        }
    }
}

fn affinity(x: &Matrix, gamma: f64) -> Matrix {
    let n = x.nrows();
    Matrix::from_fn(
        n,
        n,
        |i, j| {
            if i == j {
                0.0
            } else {
                rbf_row(x, i, j, gamma)
            }
        },
    )
}

fn row_stochastic(w: &Matrix) -> Matrix {
    let n = w.nrows();
    Matrix::from_fn(n, n, |i, j| {
        let mut s = 0.0;
        for t in 0..n {
            s += w.get(i, t);
        }
        if s > 0.0 {
            w.get(i, j) / s
        } else {
            0.0
        }
    })
}

fn symmetric_normalized(w: &Matrix) -> Matrix {
    let n = w.nrows();
    let mut deg = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            deg[i] += w.get(i, j);
        }
    }
    Matrix::from_fn(n, n, |i, j| {
        let di = deg[i].max(1e-15).sqrt();
        let dj = deg[j].max(1e-15).sqrt();
        w.get(i, j) / (di * dj)
    })
}

fn clamp_labeled(f: &mut Matrix, y0: &Matrix, labeled: &[bool]) {
    for i in 0..f.nrows() {
        if labeled[i] {
            for j in 0..f.ncols() {
                f.set(i, j, y0.get(i, j));
            }
        }
    }
}

fn argmax_labels(f: &Matrix, classes: &[i64]) -> Vector {
    Vector::from_iter((0..f.nrows()).map(|i| {
        let mut best = 0usize;
        let mut val = f64::NEG_INFINITY;
        for j in 0..f.ncols() {
            if f.get(i, j) > val {
                val = f.get(i, j);
                best = j;
            }
        }
        classes.get(best).copied().unwrap_or(-1) as f64
    }))
}

/// Hard-clamp label propagation on a row-stochastic RBF graph.
#[derive(Clone, Debug)]
pub struct LabelPropagation {
    /// RBF `γ`.
    pub gamma: f64,
    /// Propagation iterations.
    pub max_iter: usize,
    /// Max row change for convergence.
    pub tol: f64,
}

impl Default for LabelPropagation {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            max_iter: 50,
            tol: 1e-6,
        }
    }
}

impl LabelPropagation {
    /// Default label propagation.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted graph label model.
#[derive(Clone, Debug)]
pub struct FittedLabelGraph {
    /// Soft labels (`n × K`).
    pub distributions: Matrix,
    /// Hard labels (labeled rows stay clamped).
    pub labels: Vector,
    /// Training features (needed for inductive predict via 1-NN in label space).
    pub x_train: Matrix,
    /// Class ids.
    pub classes: Vec<i64>,
}

impl Predict for FittedLabelGraph {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        // Inductive: nearest training row in Euclidean feature space.
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for t in 0..self.x_train.nrows() {
                let mut d = 0.0;
                for j in 0..x.ncols().min(self.x_train.ncols()) {
                    let e = x.get(i, j) - self.x_train.get(t, j);
                    d += e * e;
                }
                if d < bd {
                    bd = d;
                    best = t;
                }
            }
            if best < self.labels.len() {
                self.labels[best]
            } else {
                f64::NAN
            }
        }));
        ctx.finish(y)
    }
}

fn fit_graph(
    ctx: &mut FitCtx,
    x: &Matrix,
    y: &Vector,
    gamma: f64,
    max_iter: usize,
    tol: f64,
    spreading: bool,
    alpha: f64,
) -> FittedLabelGraph {
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let classes = labeled_classes(y);
    let labeled: Vec<bool> = y.as_slice().iter().map(|&v| !is_unlabeled(v)).collect();
    let n_lab = labeled.iter().filter(|b| **b).count();
    if classes.is_empty() || n_lab == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyClass)
                .message("no labeled rows (every y is NaN or −1)")
                .meaninglessness(Meaninglessness::vacuous(
                    "label propagation",
                    "there is no seed label to clamp",
                    "label at least one row per class",
                ))
                .build(),
        );
        return FittedLabelGraph {
            distributions: Matrix::zeros(x.nrows(), 0),
            labels: Vector::filled(x.nrows(), -1.0),
            x_train: x.clone(),
            classes,
        };
    }
    if classes.len() == 1 {
        ctx.push(
            Issue::builder(IssueCode::SingleClass)
                .message("only one labeled class is present")
                .build(),
        );
    }
    let y_lab = Vector::from_iter(y.as_slice().iter().copied().filter(|v| !is_unlabeled(*v)));
    inspect_classes(&mut ctx.report, &y_lab, &ctx.policy);
    let y0 = one_hot(y, &classes);
    let w = affinity(x, gamma.max(0.0));
    let tmat = if spreading {
        symmetric_normalized(&w)
    } else {
        row_stochastic(&w)
    };
    let mut f = y0.clone();
    let mut converged = false;
    for it in 0..max_iter.max(1) {
        let mut nxt = Matrix::zeros(f.nrows(), f.ncols());
        for i in 0..f.nrows() {
            for k in 0..f.ncols() {
                let mut s = 0.0;
                for j in 0..f.nrows() {
                    s += tmat.get(i, j) * f.get(j, k);
                }
                if spreading {
                    nxt.set(i, k, alpha * s + (1.0 - alpha) * y0.get(i, k));
                } else {
                    nxt.set(i, k, s);
                }
            }
        }
        if !spreading {
            clamp_labeled(&mut nxt, &y0, &labeled);
        }
        normalize_rows(&mut nxt);
        let mut delta: f64 = 0.0;
        for i in 0..f.nrows() {
            for k in 0..f.ncols() {
                delta = delta.max((nxt.get(i, k) - f.get(i, k)).abs());
            }
        }
        f = nxt;
        ctx.session.step(it as u64, delta, Some(delta));
        if delta < tol {
            ctx.session.converged("label graph clamp", it as u64);
            converged = true;
            break;
        }
    }
    if !converged {
        ctx.push(
            Issue::builder(IssueCode::DidNotConverge)
                .message("label graph iteration hit max_iter")
                .build(),
        );
    }
    let labels = argmax_labels(&f, &classes);
    FittedLabelGraph {
        distributions: f,
        labels,
        x_train: x.clone(),
        classes,
    }
}

impl Fit for LabelPropagation {
    type Fitted = FittedLabelGraph;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLabelGraph>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let fitted = fit_graph(
            &mut ctx,
            x,
            y,
            self.gamma,
            self.max_iter,
            self.tol,
            false,
            0.0,
        );
        ctx.finish(fitted)
    }
}

/// Soft label spreading: `F ← α S F + (1−α) Y` (Zhou et al.).
#[derive(Clone, Debug)]
pub struct LabelSpreading {
    /// RBF `γ`.
    pub gamma: f64,
    /// Clamping factor `α ∈ [0, 1)`.
    pub alpha: f64,
    /// Iterations.
    pub max_iter: usize,
    /// Tolerance.
    pub tol: f64,
}

impl Default for LabelSpreading {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            alpha: 0.2,
            max_iter: 50,
            tol: 1e-6,
        }
    }
}

impl LabelSpreading {
    /// Default label spreading.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Self-training: iteratively promote high-confidence unlabeled rows.
///
/// Unlabeled entries are `NaN` or `−1`. The base learner is a ridge
/// classifier so perfect separation does not abort the outer loop.
#[derive(Clone, Debug)]
pub struct SelfTrainingClassifier {
    /// Promote a row when `|decision|` exceeds this margin.
    pub threshold: f64,
    /// Maximum promotion rounds.
    pub max_iter: usize,
    /// Ridge penalty of the base classifier.
    pub alpha: f64,
}

impl Default for SelfTrainingClassifier {
    fn default() -> Self {
        Self {
            threshold: 0.75,
            max_iter: 10,
            alpha: 1.0,
        }
    }
}

impl SelfTrainingClassifier {
    /// Default self-trainer.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted self-training classifier.
#[derive(Clone, Debug)]
pub struct FittedSelfTraining {
    inner: crate::classification::FittedRidgeClassifier,
    /// How many originally unlabeled rows were promoted.
    pub n_promoted: usize,
    /// Rounds actually run.
    pub n_iter: usize,
}

impl Predict for FittedSelfTraining {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.inner.predict(x, session)
    }
}

impl Fit for SelfTrainingClassifier {
    type Fitted = FittedSelfTraining;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSelfTraining>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if y.len() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("self-training y length ≠ n")
                    .build(),
            );
        }
        let mut labeled: Vec<bool> = y.as_slice().iter().map(|&v| !is_unlabeled(v)).collect();
        let mut y_cur = y.clone();
        let n_lab0 = labeled.iter().filter(|b| **b).count();
        if n_lab0 == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyClass)
                    .message("self-training has no labeled seed rows")
                    .meaninglessness(Meaninglessness::vacuous(
                        "self-training classifier",
                        "every label is NaN or −1",
                        "label at least two classes before iterating",
                    ))
                    .build(),
            );
            return ctx.finish(FittedSelfTraining {
                inner: crate::classification::FittedRidgeClassifier::from_penalized(
                    crate::linear_model::FittedPenalized {
                        coef: Vector::zeros(x.ncols()),
                        intercept: 0.0,
                        alpha: self.alpha,
                        l1_ratio: 0.0,
                    },
                    Vec::new(),
                ),
                n_promoted: 0,
                n_iter: 0,
            });
        }
        let y_lab0 = Vector::from_iter(y.as_slice().iter().copied().filter(|v| !is_unlabeled(*v)));
        inspect_classes(&mut ctx.report, &y_lab0, &ctx.policy);
        let mut n_promoted = 0usize;
        let mut n_iter = 0usize;
        let mut last = None;
        for it in 0..self.max_iter.max(1) {
            n_iter = it + 1;
            let idx: Vec<usize> = labeled
                .iter()
                .enumerate()
                .filter_map(|(i, b)| if *b { Some(i) } else { None })
                .collect();
            let xt = Matrix::from_fn(idx.len(), x.ncols(), |r, c| x.get(idx[r], c));
            let yt = Vector::from_iter(idx.iter().map(|&i| y_cur[i]));
            let fitted = match RidgeClassifier::new(self.alpha).fit(
                &xt,
                &yt,
                &session.child(format!("st_{it}")),
            ) {
                Ok(q) => q.value,
                Err(e) => {
                    ctx.push(e.primary);
                    break;
                }
            };
            last = Some(fitted.clone());
            let mut added = 0usize;
            if let Ok(s) = fitted.decision_function(x, &session.child("score")) {
                for i in 0..x.nrows() {
                    if labeled[i] {
                        continue;
                    }
                    if s.value[i].abs() >= self.threshold {
                        y_cur[i] = crate::classification::from_score(s.value[i], &fitted.classes);
                        labeled[i] = true;
                        added += 1;
                        n_promoted += 1;
                    }
                }
            }
            ctx.session.step(it as u64, added as f64, None);
            if added == 0 {
                break;
            }
        }
        let still_unlab = labeled.iter().filter(|b| !**b).count();
        if still_unlab > 0 {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .message(format!(
                        "{still_unlab} rows stayed unlabeled after {n_iter} self-training rounds"
                    ))
                    .meaninglessness(Meaninglessness::new(
                        "pseudo-labels",
                        "those rows never crossed the confidence threshold",
                        signlred::InterpretiveValue::Misleading,
                        "the fitted classifier is identified on the promoted set only",
                    ))
                    .build(),
            );
        }
        let inner = last.unwrap_or_else(|| {
            crate::classification::FittedRidgeClassifier::from_penalized(
                crate::linear_model::FittedPenalized {
                    coef: Vector::zeros(x.ncols()),
                    intercept: 0.0,
                    alpha: self.alpha,
                    l1_ratio: 0.0,
                },
                Vec::new(),
            )
        });
        ctx.finish(FittedSelfTraining {
            inner,
            n_promoted,
            n_iter,
        })
    }
}

impl Fit for LabelSpreading {
    type Fitted = FittedLabelGraph;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLabelGraph>> {
        let mut ctx = FitCtx::with_session(session.clone());
        if !(0.0..1.0).contains(&self.alpha) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("LabelSpreading.alpha={} not in [0, 1)", self.alpha))
                    .build(),
            );
        }
        let fitted = fit_graph(
            &mut ctx,
            x,
            y,
            self.gamma,
            self.max_iter,
            self.tol,
            true,
            self.alpha.clamp(0.0, 0.99),
        );
        ctx.finish(fitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    #[test]
    fn propagation_fills_unlabeled() {
        let x = Matrix::from_fn(6, 1, |i, _| i as f64);
        let y = Vector::from_slice(&[0.0, f64::NAN, f64::NAN, 1.0, -1.0, 1.0]);
        let q = LabelPropagation {
            gamma: 0.5,
            max_iter: 40,
            tol: 1e-8,
        }
        .fit(&x, &y, &Session::new("semi", "lp"))
        .unwrap();
        assert_eq!(q.value.labels.len(), 6);
        assert!((q.value.labels[0] - 0.0).abs() < 1e-12);
        assert!((q.value.labels[3] - 1.0).abs() < 1e-12);
        assert!(q.value.labels[1].is_finite());
    }

    #[test]
    fn spreading_runs() {
        let x = Matrix::from_fn(5, 1, |i, _| (i as f64) * 2.0);
        let y = Vector::from_slice(&[0.0, -1.0, -1.0, -1.0, 1.0]);
        let q = LabelSpreading::new()
            .fit(&x, &y, &Session::new("semi", "ls"))
            .unwrap();
        let pred = q
            .value
            .predict(&x, &Session::new("semi", "pr"))
            .unwrap()
            .value;
        assert_eq!(pred.len(), 5);
    }

    #[test]
    fn self_training_promotes_unlabeled() {
        let x = Matrix::from_fn(16, 1, |i, _| if i < 8 { -1.2 } else { 1.2 });
        let y = Vector::from_iter((0..16).map(|i| {
            if i == 3 || i == 12 {
                f64::NAN
            } else if i < 8 {
                0.0
            } else {
                1.0
            }
        }));
        let q = SelfTrainingClassifier {
            threshold: 0.2,
            max_iter: 5,
            alpha: 0.5,
        }
        .fit(&x, &y, &Session::new("st", "fit"))
        .expect("st");
        assert!(q.value.n_iter >= 1);
        let pred = q.value.predict(&x, &Session::new("st", "p")).unwrap().value;
        assert!((pred[0] - 0.0).abs() < 0.5);
        assert!((pred[15] - 1.0).abs() < 0.5);
    }

    #[test]
    fn all_unlabeled_is_empty_class() {
        let x = Matrix::from_fn(4, 1, |i, _| i as f64);
        let y = Vector::filled(4, f64::NAN);
        let err = LabelPropagation::new()
            .fit(&x, &y, &Session::new("semi", "empty"))
            .unwrap_err();
        assert!(
            err.report.contains(IssueCode::EmptyClass)
                || err.primary().code == IssueCode::EmptyClass
        );
    }
}
