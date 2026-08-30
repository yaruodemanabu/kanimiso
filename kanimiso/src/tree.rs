//! CART trees, bagged forests, boosting, and isolation forests.
//!
//! Every `fit` / `predict` opens a [`crate::context::FitCtx`] so `signlred`
//! diagnoses (`SingleClass`, `EmptyClass`, `ConstantTarget`, constant
//! predictors, …) and `ojizou-san` records the session. A silent successful
//! fit is a bug.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::rng::Rng;
use crate::traits::{Fit, FitUnsupervised, Predict};
use crate::validate::{inspect_classes, inspect_xy};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, Qualified, Result};

/// Euler–Mascheroni constant used by the Isolation Forest path-length offset.
const EULER_GAMMA: f64 = 0.5772156649015329;

/// Average unsuccessful BST path length \(c(n)\) (Liu et al., Isolation Forest).
///
/// The anomaly module reuses this offset when it scores path lengths.
pub fn isolation_c_factor(n: f64) -> f64 {
    if n <= 1.0 {
        0.0
    } else if n <= 2.0 {
        1.0
    } else {
        let nm1 = n - 1.0;
        2.0 * (nm1.ln() + EULER_GAMMA) - 2.0 * nm1 / n
    }
}

fn class_index(lab: i64, classes: &[i64]) -> Option<usize> {
    classes.iter().position(|&c| c == lab)
}

fn majority(classes: &[i64], counts: &[f64]) -> i64 {
    let mut best_i = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (i, &c) in counts.iter().enumerate() {
        if c > best + 1e-15 || ((c - best).abs() <= 1e-15 && classes[i] < classes[best_i]) {
            best = c;
            best_i = i;
        }
    }
    classes[best_i]
}

fn gini(counts: &[f64]) -> f64 {
    let tot: f64 = counts.iter().sum();
    if tot <= 0.0 {
        return 0.0;
    }
    let mut s = 0.0;
    for &c in counts {
        let p = c / tot;
        s += p * p;
    }
    1.0 - s
}

fn weighted_counts(y: &[i64], classes: &[i64], idx: &[usize], weights: &[f64]) -> Vec<f64> {
    let mut counts = vec![0.0; classes.len()];
    for &i in idx {
        if let Some(k) = class_index(y[i], classes) {
            counts[k] += weights[i];
        }
    }
    counts
}

fn split_index(
    x: &Matrix,
    idx: &[usize],
    feature: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in idx {
        if x.get(i, feature) <= threshold {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    (left, right)
}

fn feature_subset(
    p: usize,
    max_features: Option<usize>,
    rng: &mut Rng,
    sqrt_default: bool,
) -> Vec<usize> {
    if p == 0 {
        return Vec::new();
    }
    let k = match max_features {
        Some(m) => m.max(1).min(p),
        None if sqrt_default => ((p as f64).sqrt().ceil() as usize).max(1).min(p),
        None => p,
    };
    rng.sample_indices(p, k)
}

fn diagnose_constant_predictions(ctx: &mut FitCtx, pred: &Vector, y: &Vector) {
    let pst = signlred::slice_stats(pred.as_slice());
    let yst = signlred::slice_stats(y.as_slice());
    if pst.is_constant(ctx.policy.near_zero_variance)
        && !yst.is_constant(ctx.policy.near_zero_variance)
    {
        ctx.push(
            Issue::builder(IssueCode::PredictionsAreConstant)
                .message("in-sample predictions are a constant while y is not")
                .meaninglessness(Meaninglessness::vacuous(
                    "tree / ensemble predictor",
                    "the fitted rule collapsed to a single label or value",
                    "increase depth, lower min_samples_split, or collect a separable sample",
                ))
                .build(),
        );
    }
}

fn labels_of(y: &Vector) -> Vec<i64> {
    y.as_slice()
        .iter()
        .map(|&v| if v.is_finite() { v.round() as i64 } else { 0 })
        .collect()
}

fn unit_weights(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

fn weighted_bootstrap(w: &[f64], rng: &mut Rng) -> Vec<usize> {
    let n = w.len();
    let mut cdf = vec![0.0; n];
    let mut acc = 0.0;
    for i in 0..n {
        acc += w[i].max(0.0);
        cdf[i] = acc;
    }
    if acc <= 0.0 || n == 0 {
        return (0..n).collect();
    }
    (0..n)
        .map(|_| {
            let u = rng.uniform() * acc;
            cdf.iter().position(|&c| c >= u).unwrap_or(n - 1)
        })
        .collect()
}

/// Classification tree node.
#[derive(Clone, Debug)]
enum ClassNode {
    Leaf {
        class: i64,
        counts: Vec<f64>,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<ClassNode>,
        right: Box<ClassNode>,
    },
}

/// Regression tree node.
#[derive(Clone, Debug)]
enum RegNode {
    Leaf {
        value: f64,
        n: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<RegNode>,
        right: Box<RegNode>,
    },
}

/// Isolation tree node.
#[derive(Clone, Debug)]
enum IsoNode {
    External {
        size: usize,
        depth: usize,
    },
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<IsoNode>,
        right: Box<IsoNode>,
    },
}

fn class_leaf(classes: &[i64], counts: &[f64]) -> ClassNode {
    ClassNode::Leaf {
        class: majority(classes, counts),
        counts: counts.to_vec(),
    }
}

fn class_gain(
    x: &Matrix,
    y: &[i64],
    classes: &[i64],
    idx: &[usize],
    weights: &[f64],
    feature: usize,
    threshold: f64,
    parent_g: f64,
    parent_w: f64,
) -> f64 {
    let k = classes.len();
    let mut left = vec![0.0; k];
    let mut left_w = 0.0;
    let mut right = vec![0.0; k];
    let mut right_w = 0.0;
    for &i in idx {
        let Some(c) = class_index(y[i], classes) else {
            continue;
        };
        if x.get(i, feature) <= threshold {
            left[c] += weights[i];
            left_w += weights[i];
        } else {
            right[c] += weights[i];
            right_w += weights[i];
        }
    }
    if left_w <= 0.0 || right_w <= 0.0 || parent_w <= 0.0 {
        return 0.0;
    }
    parent_g - (left_w / parent_w) * gini(&left) - (right_w / parent_w) * gini(&right)
}

fn best_class_split(
    x: &Matrix,
    y: &[i64],
    classes: &[i64],
    idx: &[usize],
    weights: &[f64],
    feats: &[usize],
    extra: bool,
    rng: &mut Rng,
    eps: f64,
) -> Option<(usize, f64)> {
    let parent = weighted_counts(y, classes, idx, weights);
    let parent_w: f64 = parent.iter().sum();
    if parent_w <= 0.0 {
        return None;
    }
    let parent_g = gini(&parent);
    let mut best_gain = 1e-15;
    let mut best = None;
    for &f in feats {
        if extra {
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for &i in idx {
                let v = x.get(i, f);
                mn = mn.min(v);
                mx = mx.max(v);
            }
            if !mn.is_finite() || mx - mn <= eps {
                continue;
            }
            let thr = rng.uniform_range(mn, mx);
            let gain = class_gain(x, y, classes, idx, weights, f, thr, parent_g, parent_w);
            if gain > best_gain {
                best_gain = gain;
                best = Some((f, thr));
            }
        } else {
            let mut pts: Vec<(f64, usize)> =
                idx.iter().copied().map(|i| (x.get(i, f), i)).collect();
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let k = classes.len();
            let mut left = vec![0.0; k];
            let mut left_w = 0.0;
            for s in 0..pts.len().saturating_sub(1) {
                let i = pts[s].1;
                if let Some(c) = class_index(y[i], classes) {
                    left[c] += weights[i];
                    left_w += weights[i];
                }
                if (pts[s + 1].0 - pts[s].0).abs() <= eps {
                    continue;
                }
                let right_w = parent_w - left_w;
                if left_w <= 0.0 || right_w <= 0.0 {
                    continue;
                }
                let mut right = vec![0.0; k];
                for c in 0..k {
                    right[c] = parent[c] - left[c];
                }
                let gain = parent_g
                    - (left_w / parent_w) * gini(&left)
                    - (right_w / parent_w) * gini(&right);
                if gain > best_gain {
                    best_gain = gain;
                    best = Some((f, 0.5 * (pts[s].0 + pts[s + 1].0)));
                }
            }
        }
    }
    best
}

fn grow_class(
    x: &Matrix,
    y: &[i64],
    classes: &[i64],
    idx: &[usize],
    weights: &[f64],
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
    max_features: Option<usize>,
    extra: bool,
    sqrt_features: bool,
    rng: &mut Rng,
    eps: f64,
) -> ClassNode {
    let counts = weighted_counts(y, classes, idx, weights);
    let n_eff = counts.iter().sum::<f64>();
    let pure = counts.iter().filter(|&&c| c > 0.0).count() <= 1;
    if depth >= max_depth || idx.len() < min_samples_split.max(2) || n_eff <= 0.0 || pure {
        return class_leaf(classes, &counts);
    }
    let feats = feature_subset(x.ncols(), max_features, rng, sqrt_features);
    let Some((feature, threshold)) =
        best_class_split(x, y, classes, idx, weights, &feats, extra, rng, eps)
    else {
        return class_leaf(classes, &counts);
    };
    let (left, right) = split_index(x, idx, feature, threshold);
    if left.is_empty() || right.is_empty() {
        return class_leaf(classes, &counts);
    }
    ClassNode::Split {
        feature,
        threshold,
        left: Box::new(grow_class(
            x,
            y,
            classes,
            &left,
            weights,
            depth + 1,
            max_depth,
            min_samples_split,
            max_features,
            extra,
            sqrt_features,
            rng,
            eps,
        )),
        right: Box::new(grow_class(
            x,
            y,
            classes,
            &right,
            weights,
            depth + 1,
            max_depth,
            min_samples_split,
            max_features,
            extra,
            sqrt_features,
            rng,
            eps,
        )),
    }
}

fn predict_class_one(node: &ClassNode, x: &Matrix, i: usize) -> i64 {
    match node {
        ClassNode::Leaf { class, .. } => *class,
        ClassNode::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            if x.get(i, *feature) <= *threshold {
                predict_class_one(left, x, i)
            } else {
                predict_class_one(right, x, i)
            }
        }
    }
}

fn predict_class_proba(node: &ClassNode, x: &Matrix, i: usize, k: usize) -> Vec<f64> {
    match node {
        ClassNode::Leaf { counts, .. } => {
            let tot: f64 = counts.iter().sum::<f64>().max(1e-15);
            let mut p = vec![0.0; k];
            for (j, &c) in counts.iter().enumerate().take(k) {
                p[j] = (c / tot).max(1e-15);
            }
            p
        }
        ClassNode::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            if x.get(i, *feature) <= *threshold {
                predict_class_proba(left, x, i, k)
            } else {
                predict_class_proba(right, x, i, k)
            }
        }
    }
}

fn predict_class_vec(node: &ClassNode, x: &Matrix) -> Vector {
    Vector::from_iter((0..x.nrows()).map(|i| predict_class_one(node, x, i) as f64))
}

fn is_class_stump(node: &ClassNode) -> bool {
    matches!(node, ClassNode::Leaf { .. })
}

fn mse_of(ys: &[f64], idx: &[usize], weights: &[f64]) -> (f64, f64, f64) {
    let mut wsum = 0.0;
    let mut s = 0.0;
    for &i in idx {
        wsum += weights[i];
        s += weights[i] * ys[i];
    }
    if wsum <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let mean = s / wsum;
    let mut sse = 0.0;
    for &i in idx {
        let d = ys[i] - mean;
        sse += weights[i] * d * d;
    }
    (mean, sse, wsum)
}

fn best_reg_split(
    x: &Matrix,
    ys: &[f64],
    idx: &[usize],
    weights: &[f64],
    feats: &[usize],
    extra: bool,
    rng: &mut Rng,
    eps: f64,
) -> Option<(usize, f64)> {
    let (parent_mean, parent_sse, parent_w) = mse_of(ys, idx, weights);
    let _ = parent_mean;
    if parent_w <= 0.0 || parent_sse <= 0.0 {
        return None;
    }
    let mut best_gain = 1e-15;
    let mut best = None;
    for &f in feats {
        if extra {
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for &i in idx {
                let v = x.get(i, f);
                mn = mn.min(v);
                mx = mx.max(v);
            }
            if mx - mn <= eps {
                continue;
            }
            let thr = rng.uniform_range(mn, mx);
            let (left, right) = split_index(x, idx, f, thr);
            let (_, ls, lw) = mse_of(ys, &left, weights);
            let (_, rs, rw) = mse_of(ys, &right, weights);
            if lw <= 0.0 || rw <= 0.0 {
                continue;
            }
            let gain = parent_sse - ls - rs;
            if gain > best_gain {
                best_gain = gain;
                best = Some((f, thr));
            }
        } else {
            let mut pts: Vec<(f64, usize)> =
                idx.iter().copied().map(|i| (x.get(i, f), i)).collect();
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut left_w = 0.0;
            let mut left_s = 0.0;
            for s in 0..pts.len().saturating_sub(1) {
                let i = pts[s].1;
                left_w += weights[i];
                left_s += weights[i] * ys[i];
                if (pts[s + 1].0 - pts[s].0).abs() <= eps {
                    continue;
                }
                let right_w = parent_w - left_w;
                if left_w <= 0.0 || right_w <= 0.0 {
                    continue;
                }
                let mut left_sse = 0.0;
                let lmean = left_s / left_w;
                let mut right_s = 0.0;
                let mut right_sse_w = 0.0;
                for &j in idx {
                    if x.get(j, f) <= 0.5 * (pts[s].0 + pts[s + 1].0) {
                        let d = ys[j] - lmean;
                        left_sse += weights[j] * d * d;
                    } else {
                        right_s += weights[j] * ys[j];
                        right_sse_w += weights[j];
                    }
                }
                if right_sse_w <= 0.0 {
                    continue;
                }
                let rmean = right_s / right_sse_w;
                let mut right_sse = 0.0;
                for &j in idx {
                    if x.get(j, f) > 0.5 * (pts[s].0 + pts[s + 1].0) {
                        let d = ys[j] - rmean;
                        right_sse += weights[j] * d * d;
                    }
                }
                let gain = parent_sse - left_sse - right_sse;
                if gain > best_gain {
                    best_gain = gain;
                    best = Some((f, 0.5 * (pts[s].0 + pts[s + 1].0)));
                }
            }
        }
    }
    best
}

fn grow_reg(
    x: &Matrix,
    ys: &[f64],
    idx: &[usize],
    weights: &[f64],
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
    max_features: Option<usize>,
    extra: bool,
    rng: &mut Rng,
    eps: f64,
) -> RegNode {
    let (mean, sse, wsum) = mse_of(ys, idx, weights);
    if depth >= max_depth || idx.len() < min_samples_split.max(2) || sse <= eps || wsum <= 0.0 {
        return RegNode::Leaf {
            value: mean,
            n: wsum,
        };
    }
    let feats = feature_subset(x.ncols(), max_features, rng, false);
    let Some((feature, threshold)) = best_reg_split(x, ys, idx, weights, &feats, extra, rng, eps)
    else {
        return RegNode::Leaf {
            value: mean,
            n: wsum,
        };
    };
    let (left, right) = split_index(x, idx, feature, threshold);
    if left.is_empty() || right.is_empty() {
        return RegNode::Leaf {
            value: mean,
            n: wsum,
        };
    }
    RegNode::Split {
        feature,
        threshold,
        left: Box::new(grow_reg(
            x,
            ys,
            &left,
            weights,
            depth + 1,
            max_depth,
            min_samples_split,
            max_features,
            extra,
            rng,
            eps,
        )),
        right: Box::new(grow_reg(
            x,
            ys,
            &right,
            weights,
            depth + 1,
            max_depth,
            min_samples_split,
            max_features,
            extra,
            rng,
            eps,
        )),
    }
}

fn predict_reg_one(node: &RegNode, x: &Matrix, i: usize) -> f64 {
    match node {
        RegNode::Leaf { value, .. } => *value,
        RegNode::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            if x.get(i, *feature) <= *threshold {
                predict_reg_one(left, x, i)
            } else {
                predict_reg_one(right, x, i)
            }
        }
    }
}

fn predict_reg_vec(node: &RegNode, x: &Matrix) -> Vector {
    Vector::from_iter((0..x.nrows()).map(|i| predict_reg_one(node, x, i)))
}

fn rewrite_logistic_leaves(node: &mut RegNode, x: &Matrix, r: &[f64], p: &[f64], idx: &[usize]) {
    match node {
        RegNode::Leaf { value, n } => {
            let mut num = 0.0;
            let mut den = 0.0;
            for &i in idx {
                num += r[i];
                den += p[i] * (1.0 - p[i]);
            }
            *value = num / den.max(1e-12);
            *n = idx.len() as f64;
        }
        RegNode::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            let (l, rg) = split_index(x, idx, *feature, *threshold);
            rewrite_logistic_leaves(left, x, r, p, &l);
            rewrite_logistic_leaves(right, x, r, p, &rg);
        }
    }
}

fn grow_iso(x: &Matrix, idx: &[usize], depth: usize, max_depth: usize, rng: &mut Rng) -> IsoNode {
    if depth >= max_depth || idx.len() <= 1 {
        return IsoNode::External {
            size: idx.len(),
            depth,
        };
    }
    let p = x.ncols();
    let mut order: Vec<usize> = (0..p).collect();
    rng.shuffle(&mut order);
    for &f in &order {
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for &i in idx {
            let v = x.get(i, f);
            mn = mn.min(v);
            mx = mx.max(v);
        }
        if mx - mn <= 1e-15 {
            continue;
        }
        let thr = rng.uniform_range(mn, mx);
        let (left, right) = split_index(x, idx, f, thr);
        if left.is_empty() || right.is_empty() {
            continue;
        }
        return IsoNode::Internal {
            feature: f,
            threshold: thr,
            left: Box::new(grow_iso(x, &left, depth + 1, max_depth, rng)),
            right: Box::new(grow_iso(x, &right, depth + 1, max_depth, rng)),
        };
    }
    IsoNode::External {
        size: idx.len(),
        depth,
    }
}

fn iso_path(node: &IsoNode, x: &Matrix, i: usize) -> f64 {
    match node {
        IsoNode::External { size, depth } => *depth as f64 + isolation_c_factor(*size as f64),
        IsoNode::Internal {
            feature,
            threshold,
            left,
            right,
        } => {
            if x.get(i, *feature) <= *threshold {
                iso_path(left, x, i)
            } else {
                iso_path(right, x, i)
            }
        }
    }
}

fn bootstrap_idx(rng: &mut Rng, n: usize) -> Vec<usize> {
    (0..n).map(|_| rng.below(n)).collect()
}

fn vote_labels(classes: &[i64], votes: &[f64]) -> i64 {
    majority(classes, votes)
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

/// CART classifier using Gini impurity.
#[derive(Clone, Debug)]
pub struct DecisionTreeClassifier {
    /// Maximum tree depth (root is depth 0; `0` yields a stump leaf).
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size at each node (`None` = all features).
    pub max_features: Option<usize>,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl DecisionTreeClassifier {
    /// Default Gini tree (`max_depth = 8`).
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted CART classifier.
#[derive(Clone, Debug)]
pub struct FittedTreeClassifier {
    root: ClassNode,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Training feature count.
    pub n_features: usize,
}

impl FittedTreeClassifier {
    /// Class-probability vector for row `i` of `x` (aligned with [`Self::classes`]).
    pub fn predict_proba_row(&self, x: &Matrix, i: usize) -> Vec<f64> {
        predict_class_proba(&self.root, x, i, self.classes.len())
    }
}

impl Predict for FittedTreeClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(predict_class_vec(&self.root, x))
    }
}

impl Fit for DecisionTreeClassifier {
    type Fitted = FittedTreeClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTreeClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let ylab = labels_of(y);
        let idx: Vec<usize> = (0..x.nrows()).collect();
        let w = unit_weights(x.nrows());
        let mut rng = Rng::new(self.seed);
        let root = if classes.is_empty() {
            ClassNode::Leaf {
                class: 0,
                counts: Vec::new(),
            }
        } else {
            grow_class(
                x,
                &ylab,
                &classes,
                &idx,
                &w,
                0,
                self.max_depth,
                self.min_samples_split,
                self.max_features,
                false,
                false,
                &mut rng,
                ctx.policy.near_zero_variance,
            )
        };
        let fitted = FittedTreeClassifier {
            root,
            classes,
            n_features: x.ncols(),
        };
        if !is_class_stump(&fitted.root) {
            ctx.session.converged("CART Gini split found", 0);
        } else if counts.len() > 1 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("Gini CART produced a single leaf on a multi-class target")
                    .meaninglessness(Meaninglessness::vacuous(
                        "decision tree",
                        "no split reduced Gini; the classifier is a constant",
                        "check feature variation and hyperparameters",
                    ))
                    .build(),
            );
        }
        let pred = predict_class_vec(&fitted.root, x);
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// CART regressor using mean-squared-error impurity.
#[derive(Clone, Debug)]
pub struct DecisionTreeRegressor {
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size at each node (`None` = all features).
    pub max_features: Option<usize>,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl DecisionTreeRegressor {
    /// Default MSE tree.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted CART regressor.
#[derive(Clone, Debug)]
pub struct FittedTreeRegressor {
    root: RegNode,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedTreeRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(predict_reg_vec(&self.root, x))
    }
}

impl Fit for DecisionTreeRegressor {
    type Fitted = FittedTreeRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTreeRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let ys = y.as_slice().to_vec();
        let idx: Vec<usize> = (0..x.nrows()).collect();
        let w = unit_weights(x.nrows());
        let mut rng = Rng::new(self.seed);
        let root = grow_reg(
            x,
            &ys,
            &idx,
            &w,
            0,
            self.max_depth,
            self.min_samples_split,
            self.max_features,
            false,
            &mut rng,
            ctx.policy.near_zero_variance,
        );
        let fitted = FittedTreeRegressor {
            root,
            n_features: x.ncols(),
        };
        let pred = predict_reg_vec(&fitted.root, x);
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Bootstrap-aggregated Gini trees with per-node feature subsample.
#[derive(Clone, Debug)]
pub struct RandomForestClassifier {
    /// Number of trees.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` ⇒ \(\sqrt{p}\)).
    pub max_features: Option<usize>,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for RandomForestClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 20,
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl RandomForestClassifier {
    /// Default random forest classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted classification forest (random forest or extra-trees).
#[derive(Clone, Debug)]
pub struct FittedForestClassifier {
    trees: Vec<ClassNode>,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedForestClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        let k = self.classes.len();
        let mut out = Vector::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut votes = vec![0.0; k];
            for t in &self.trees {
                let lab = predict_class_one(t, x, i);
                if let Some(j) = class_index(lab, &self.classes) {
                    votes[j] += 1.0;
                }
            }
            out[i] = vote_labels(&self.classes, &votes) as f64;
        }
        ctx.finish(out)
    }
}

fn grow_forest_class(
    ctx: &mut FitCtx,
    x: &Matrix,
    y: &Vector,
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    max_features: Option<usize>,
    seed: u64,
    extra: bool,
    bootstrap: bool,
) -> FittedForestClassifier {
    let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
    let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
    let ylab = labels_of(y);
    let w = unit_weights(x.nrows());
    let mut rng = Rng::new(seed);
    let mut trees = Vec::with_capacity(n_estimators);
    if classes.is_empty() {
        return FittedForestClassifier {
            trees,
            classes,
            n_features: x.ncols(),
        };
    }
    for t in 0..n_estimators {
        let mut trng = Rng::new(rng.next_u64());
        let idx = if bootstrap && x.nrows() > 0 {
            bootstrap_idx(&mut trng, x.nrows())
        } else {
            (0..x.nrows()).collect()
        };
        let root = grow_class(
            x,
            &ylab,
            &classes,
            &idx,
            &w,
            0,
            max_depth,
            min_samples_split,
            max_features,
            extra,
            !extra,
            &mut trng,
            ctx.policy.near_zero_variance,
        );
        ctx.session.step(t as u64, 0.0, None);
        trees.push(root);
    }
    if !trees.is_empty() {
        ctx.session
            .converged(format!("{n_estimators} trees grown"), n_estimators as u64);
    }
    FittedForestClassifier {
        trees,
        classes,
        n_features: x.ncols(),
    }
}

impl Fit for RandomForestClassifier {
    type Fitted = FittedForestClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForestClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let fitted = grow_forest_class(
            &mut ctx,
            x,
            y,
            self.n_estimators.max(1),
            self.max_depth,
            self.min_samples_split,
            self.max_features,
            self.seed,
            false,
            true,
        );
        let pred = match fitted.classes.is_empty() {
            true => Vector::zeros(x.nrows()),
            false => {
                let mut tmp = Vector::zeros(x.nrows());
                let k = fitted.classes.len();
                for i in 0..x.nrows() {
                    let mut votes = vec![0.0; k];
                    for t in &fitted.trees {
                        if let Some(j) = class_index(predict_class_one(t, x, i), &fitted.classes) {
                            votes[j] += 1.0;
                        }
                    }
                    tmp[i] = vote_labels(&fitted.classes, &votes) as f64;
                }
                tmp
            }
        };
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Bootstrap-aggregated MSE trees with per-node feature subsample.
#[derive(Clone, Debug)]
pub struct RandomForestRegressor {
    /// Number of trees.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` ⇒ all features).
    pub max_features: Option<usize>,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for RandomForestRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 20,
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl RandomForestRegressor {
    /// Default random forest regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted regression forest.
#[derive(Clone, Debug)]
pub struct FittedForestRegressor {
    trees: Vec<RegNode>,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedForestRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        let mut out = Vector::zeros(x.nrows());
        if self.trees.is_empty() {
            return ctx.finish(out);
        }
        let inv = 1.0 / self.trees.len() as f64;
        for i in 0..x.nrows() {
            let mut s = 0.0;
            for t in &self.trees {
                s += predict_reg_one(t, x, i);
            }
            out[i] = s * inv;
        }
        ctx.finish(out)
    }
}

impl Fit for RandomForestRegressor {
    type Fitted = FittedForestRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForestRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let ys = y.as_slice().to_vec();
        let w = unit_weights(x.nrows());
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let n_est = self.n_estimators.max(1);
        for t in 0..n_est {
            let mut trng = Rng::new(rng.next_u64());
            let idx = if x.nrows() > 0 {
                bootstrap_idx(&mut trng, x.nrows())
            } else {
                Vec::new()
            };
            trees.push(grow_reg(
                x,
                &ys,
                &idx,
                &w,
                0,
                self.max_depth,
                self.min_samples_split,
                self.max_features,
                false,
                &mut trng,
                ctx.policy.near_zero_variance,
            ));
            ctx.session.step(t as u64, 0.0, None);
        }
        let fitted = FittedForestRegressor {
            trees,
            n_features: x.ncols(),
        };
        let pred = {
            let mut out = Vector::zeros(x.nrows());
            if !fitted.trees.is_empty() {
                let inv = 1.0 / fitted.trees.len() as f64;
                for i in 0..x.nrows() {
                    let mut s = 0.0;
                    for t in &fitted.trees {
                        s += predict_reg_one(t, x, i);
                    }
                    out[i] = s * inv;
                }
            }
            out
        };
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Extremely randomized Gini trees (random thresholds, full sample).
#[derive(Clone, Debug)]
pub struct ExtraTreesClassifier {
    /// Number of trees.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` ⇒ \(\sqrt{p}\)).
    pub max_features: Option<usize>,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for ExtraTreesClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 20,
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl ExtraTreesClassifier {
    /// Default extra-trees classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for ExtraTreesClassifier {
    type Fitted = FittedForestClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForestClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let fitted = grow_forest_class(
            &mut ctx,
            x,
            y,
            self.n_estimators.max(1),
            self.max_depth,
            self.min_samples_split,
            self.max_features,
            self.seed,
            true,
            false,
        );
        let pred = {
            let mut tmp = Vector::zeros(x.nrows());
            let k = fitted.classes.len();
            if k > 0 {
                for i in 0..x.nrows() {
                    let mut votes = vec![0.0; k];
                    for t in &fitted.trees {
                        if let Some(j) = class_index(predict_class_one(t, x, i), &fitted.classes) {
                            votes[j] += 1.0;
                        }
                    }
                    tmp[i] = vote_labels(&fitted.classes, &votes) as f64;
                }
            }
            tmp
        };
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Single extremely randomized Gini tree (sklearn `ExtraTreeClassifier`).
///
/// This is [`ExtraTreesClassifier`] with one tree. Feature subsample size is
/// not identification `p`.
#[derive(Clone, Debug)]
pub struct ExtraTreeClassifier {
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` ⇒ \(\sqrt{p}\)).
    pub max_features: Option<usize>,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for ExtraTreeClassifier {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl ExtraTreeClassifier {
    /// Default single extra-tree classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for ExtraTreeClassifier {
    type Fitted = FittedForestClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForestClassifier>> {
        ExtraTreesClassifier {
            n_estimators: 1,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            max_features: self.max_features,
            seed: self.seed,
        }
        .fit(x, y, session)
    }
}

/// Extremely randomized MSE trees (random thresholds, full sample).
#[derive(Clone, Debug)]
pub struct ExtraTreesRegressor {
    /// Number of trees.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` ⇒ all features).
    pub max_features: Option<usize>,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for ExtraTreesRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 20,
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl ExtraTreesRegressor {
    /// Default extra-trees regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for ExtraTreesRegressor {
    type Fitted = FittedForestRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForestRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let ys = y.as_slice().to_vec();
        let w = unit_weights(x.nrows());
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let n_est = self.n_estimators.max(1);
        let idx: Vec<usize> = (0..x.nrows()).collect();
        for t in 0..n_est {
            let mut trng = Rng::new(rng.next_u64());
            trees.push(grow_reg(
                x,
                &ys,
                &idx,
                &w,
                0,
                self.max_depth,
                self.min_samples_split,
                self.max_features,
                true,
                &mut trng,
                ctx.policy.near_zero_variance,
            ));
            ctx.session.step(t as u64, 0.0, None);
        }
        let fitted = FittedForestRegressor {
            trees,
            n_features: x.ncols(),
        };
        let pred = {
            let mut out = Vector::zeros(x.nrows());
            if !fitted.trees.is_empty() {
                let inv = 1.0 / fitted.trees.len() as f64;
                for i in 0..x.nrows() {
                    let mut s = 0.0;
                    for t in &fitted.trees {
                        s += predict_reg_one(t, x, i);
                    }
                    out[i] = s * inv;
                }
            }
            out
        };
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Single extremely randomized MSE tree (sklearn `ExtraTreeRegressor`).
///
/// This is [`ExtraTreesRegressor`] with one tree. Feature subsample size is
/// not identification `p`.
#[derive(Clone, Debug)]
pub struct ExtraTreeRegressor {
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` ⇒ all features).
    pub max_features: Option<usize>,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for ExtraTreeRegressor {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl ExtraTreeRegressor {
    /// Default single extra-tree regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for ExtraTreeRegressor {
    type Fitted = FittedForestRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForestRegressor>> {
        ExtraTreesRegressor {
            n_estimators: 1,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            max_features: self.max_features,
            seed: self.seed,
        }
        .fit(x, y, session)
    }
}

/// Friedman gradient boosting for squared error.
#[derive(Clone, Debug)]
pub struct GradientBoostingRegressor {
    /// Number of sequential trees.
    pub n_estimators: usize,
    /// Shrinkage \(\nu\).
    pub learning_rate: f64,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for GradientBoostingRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 30,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            seed: 0,
        }
    }
}

impl GradientBoostingRegressor {
    /// Default squared-error gradient booster.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted squared-error gradient booster.
#[derive(Clone, Debug)]
pub struct FittedGbr {
    /// Initial constant (training mean).
    pub intercept: f64,
    trees: Vec<RegNode>,
    /// Shrinkage used at fit time.
    pub learning_rate: f64,
    /// Training feature count.
    pub n_features: usize,
}

impl FittedGbr {
    fn predict_vec(&self, x: &Matrix) -> Vector {
        let mut out = Vector::filled(x.nrows(), self.intercept);
        for t in &self.trees {
            for i in 0..x.nrows() {
                out[i] += self.learning_rate * predict_reg_one(t, x, i);
            }
        }
        out
    }
}

impl Predict for FittedGbr {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(self.predict_vec(x))
    }
}

impl Fit for GradientBoostingRegressor {
    type Fitted = FittedGbr;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedGbr>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let intercept = y.mean();
        let mut residual: Vec<f64> = y.as_slice().iter().map(|v| v - intercept).collect();
        let w = unit_weights(x.nrows());
        let idx: Vec<usize> = (0..x.nrows()).collect();
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let nu = self.learning_rate;
        if nu <= 0.0 || !nu.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!(
                        "learning_rate={nu} is not a positive finite number"
                    ))
                    .build(),
            );
        }
        for m in 0..self.n_estimators.max(1) {
            let mut trng = Rng::new(rng.next_u64());
            let tree = grow_reg(
                x,
                &residual,
                &idx,
                &w,
                0,
                self.max_depth,
                self.min_samples_split,
                None,
                false,
                &mut trng,
                ctx.policy.near_zero_variance,
            );
            let mut sse = 0.0;
            for i in 0..x.nrows() {
                let step = nu * predict_reg_one(&tree, x, i);
                residual[i] -= step;
                sse += residual[i] * residual[i];
            }
            ctx.session.step(m as u64, sse, None);
            trees.push(tree);
            if sse <= ctx.policy.near_zero_variance {
                ctx.session
                    .converged("boosting residuals vanished", m as u64);
                break;
            }
        }
        let fitted = FittedGbr {
            intercept,
            trees,
            learning_rate: nu,
            n_features: x.ncols(),
        };
        let pred = fitted.predict_vec(x);
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Friedman gradient boosting for binomial / multinomial log-loss.
#[derive(Clone, Debug)]
pub struct GradientBoostingClassifier {
    /// Number of sequential stages.
    pub n_estimators: usize,
    /// Shrinkage \(\nu\).
    pub learning_rate: f64,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for GradientBoostingClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 30,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            seed: 0,
        }
    }
}

impl GradientBoostingClassifier {
    /// Default log-loss gradient booster.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted log-loss gradient booster.
#[derive(Clone, Debug)]
pub struct FittedGbc {
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Per-class initial scores (log-odds / zero-mean log-prior).
    pub intercept: Vec<f64>,
    /// Stages; for binary, each stage has one tree (positive class).
    trees: Vec<Vec<RegNode>>,
    /// Shrinkage used at fit time.
    pub learning_rate: f64,
    /// Training feature count.
    pub n_features: usize,
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

fn softmax_row(scores: &[f64]) -> Vec<f64> {
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
    let z: f64 = e.iter().sum::<f64>().max(1e-15);
    for v in &mut e {
        *v /= z;
    }
    e
}

impl FittedGbc {
    fn scores_row(&self, x: &Matrix, i: usize) -> Vec<f64> {
        let k = self.classes.len();
        let mut f = self.intercept.clone();
        if f.len() != k {
            f.resize(k, 0.0);
        }
        let binary = k <= 2;
        for stage in &self.trees {
            if binary {
                if let Some(t) = stage.first() {
                    let step = self.learning_rate * predict_reg_one(t, x, i);
                    if k == 2 {
                        f[1] += step;
                    } else if k == 1 {
                        f[0] += step;
                    }
                }
            } else {
                for (c, t) in stage.iter().enumerate().take(k) {
                    f[c] += self.learning_rate * predict_reg_one(t, x, i);
                }
            }
        }
        f
    }

    fn predict_vec(&self, x: &Matrix) -> Vector {
        let k = self.classes.len();
        Vector::from_iter((0..x.nrows()).map(|i| {
            if k == 0 {
                return 0.0;
            }
            if k == 2 {
                let f = self.scores_row(x, i);
                let p =
                    sigmoid(f.get(1).copied().unwrap_or(0.0) - f.first().copied().unwrap_or(0.0));
                if p >= 0.5 {
                    self.classes[1] as f64
                } else {
                    self.classes[0] as f64
                }
            } else {
                let f = self.scores_row(x, i);
                let mut best = 0usize;
                for c in 1..f.len() {
                    if f[c] > f[best] {
                        best = c;
                    }
                }
                self.classes[best] as f64
            }
        }))
    }
}

impl Predict for FittedGbc {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(self.predict_vec(x))
    }
}

impl Fit for GradientBoostingClassifier {
    type Fitted = FittedGbc;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedGbc>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let ylab = labels_of(y);
        let n = x.nrows();
        let k = classes.len();
        let w = unit_weights(n);
        let idx: Vec<usize> = (0..n).collect();
        let mut rng = Rng::new(self.seed);
        if k < 2 {
            return ctx.finish(FittedGbc {
                classes,
                intercept: vec![0.0],
                trees: Vec::new(),
                learning_rate: self.learning_rate,
                n_features: x.ncols(),
            });
        }
        let ntot = counts.iter().map(|(_, c)| *c as f64).sum::<f64>().max(1.0);
        let mut intercept = vec![0.0; k];
        if k == 2 {
            let p1 = counts[1].1 as f64 / ntot;
            let p1 = p1.clamp(1e-15, 1.0 - 1e-15);
            intercept[1] = (p1 / (1.0 - p1)).ln();
        }
        let mut scores = vec![vec![0.0; k]; n];
        for i in 0..n {
            scores[i].clone_from(&intercept);
        }
        let mut trees: Vec<Vec<RegNode>> = Vec::new();
        let nu = self.learning_rate;
        for m in 0..self.n_estimators.max(1) {
            let mut stage = Vec::new();
            if k == 2 {
                let mut r = vec![0.0; n];
                let mut p = vec![0.0; n];
                let mut loss = 0.0;
                for i in 0..n {
                    let logit = scores[i][1] - scores[i][0];
                    p[i] = sigmoid(logit);
                    let yi = if ylab[i] == classes[1] { 1.0 } else { 0.0 };
                    r[i] = yi - p[i];
                    let pi = p[i].clamp(1e-15, 1.0 - 1e-15);
                    loss -= yi * pi.ln() + (1.0 - yi) * (1.0 - pi).ln();
                }
                let mut trng = Rng::new(rng.next_u64());
                let mut tree = grow_reg(
                    x,
                    &r,
                    &idx,
                    &w,
                    0,
                    self.max_depth,
                    self.min_samples_split,
                    None,
                    false,
                    &mut trng,
                    ctx.policy.near_zero_variance,
                );
                rewrite_logistic_leaves(&mut tree, x, &r, &p, &idx);
                for i in 0..n {
                    scores[i][1] += nu * predict_reg_one(&tree, x, i);
                }
                ctx.session.step(m as u64, loss, None);
                stage.push(tree);
            } else {
                let mut loss = 0.0;
                let mut probs = vec![vec![0.0; k]; n];
                for i in 0..n {
                    probs[i] = softmax_row(&scores[i]);
                    if let Some(c) = class_index(ylab[i], &classes) {
                        loss -= probs[i][c].max(1e-15).ln();
                    }
                }
                for c in 0..k {
                    let mut r = vec![0.0; n];
                    for i in 0..n {
                        let yi = if ylab[i] == classes[c] { 1.0 } else { 0.0 };
                        r[i] = yi - probs[i][c];
                    }
                    let mut trng = Rng::new(rng.next_u64());
                    let tree = grow_reg(
                        x,
                        &r,
                        &idx,
                        &w,
                        0,
                        self.max_depth,
                        self.min_samples_split,
                        None,
                        false,
                        &mut trng,
                        ctx.policy.near_zero_variance,
                    );
                    for i in 0..n {
                        scores[i][c] += nu * predict_reg_one(&tree, x, i);
                    }
                    stage.push(tree);
                }
                ctx.session.step(m as u64, loss, None);
            }
            trees.push(stage);
        }
        ctx.session
            .converged("finished log-loss boosting stages", trees.len() as u64);
        let fitted = FittedGbc {
            classes,
            intercept,
            trees,
            learning_rate: nu,
            n_features: x.ncols(),
        };
        let pred = fitted.predict_vec(x);
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// AdaBoost label-update scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaBoostAlgorithm {
    /// Discrete SAMME (Zhu et al.).
    Samme,
    /// Real SAMME.R using class probabilities.
    SammeR,
}

/// SAMME / SAMME.R AdaBoost classifier.
#[derive(Clone, Debug)]
pub struct AdaBoostClassifier {
    /// Number of weak learners.
    pub n_estimators: usize,
    /// Shrinkage on the additive model.
    pub learning_rate: f64,
    /// Weak-learner depth (SAMME needs ≥ 2 for XOR-like problems).
    pub max_depth: usize,
    /// `SAMME` or `SAMME.R`.
    pub algorithm: AdaBoostAlgorithm,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for AdaBoostClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 30,
            learning_rate: 1.0,
            max_depth: 2,
            algorithm: AdaBoostAlgorithm::SammeR,
            seed: 0,
        }
    }
}

impl AdaBoostClassifier {
    /// Default SAMME.R AdaBoost.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted AdaBoost model.
#[derive(Clone, Debug)]
pub struct FittedAdaBoost {
    trees: Vec<ClassNode>,
    /// SAMME weights (`α_m`); empty when using SAMME.R.
    pub alphas: Vec<f64>,
    /// Algorithm used at fit time.
    pub algorithm: AdaBoostAlgorithm,
    /// Shrinkage.
    pub learning_rate: f64,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Training feature count.
    pub n_features: usize,
}

impl FittedAdaBoost {
    fn predict_vec(&self, x: &Matrix) -> Vector {
        let k = self.classes.len();
        Vector::from_iter((0..x.nrows()).map(|i| {
            if k == 0 {
                return 0.0;
            }
            let mut scores = vec![0.0; k];
            match self.algorithm {
                AdaBoostAlgorithm::Samme => {
                    for (t, &alpha) in self.trees.iter().zip(&self.alphas) {
                        let lab = predict_class_one(t, x, i);
                        if let Some(j) = class_index(lab, &self.classes) {
                            scores[j] += alpha;
                        }
                    }
                }
                AdaBoostAlgorithm::SammeR => {
                    let km1 = (k as f64 - 1.0).max(1.0);
                    for t in &self.trees {
                        let p = predict_class_proba(t, x, i, k);
                        let mut lp: Vec<f64> = p.iter().map(|v| v.max(1e-15).ln()).collect();
                        let mean = lp.iter().sum::<f64>() / k as f64;
                        for c in 0..k {
                            lp[c] -= mean;
                            scores[c] += self.learning_rate * km1 * lp[c];
                        }
                    }
                }
            }
            vote_labels(&self.classes, &scores) as f64
        }))
    }
}

impl Predict for FittedAdaBoost {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(self.predict_vec(x))
    }
}

impl Fit for AdaBoostClassifier {
    type Fitted = FittedAdaBoost;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedAdaBoost>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let ylab = labels_of(y);
        let n = x.nrows();
        let k = classes.len();
        if k < 2 || n == 0 {
            return ctx.finish(FittedAdaBoost {
                trees: Vec::new(),
                alphas: Vec::new(),
                algorithm: self.algorithm,
                learning_rate: self.learning_rate,
                classes,
                n_features: x.ncols(),
            });
        }
        let mut w = vec![1.0 / n as f64; n];
        let idx: Vec<usize> = (0..n).collect();
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut alphas = Vec::new();
        let k_f = k as f64;
        for m in 0..self.n_estimators.max(1) {
            let mut trng = Rng::new(rng.next_u64());
            let tree = grow_class(
                x,
                &ylab,
                &classes,
                &idx,
                &w,
                0,
                self.max_depth,
                2,
                None,
                false,
                false,
                &mut trng,
                ctx.policy.near_zero_variance,
            );
            match self.algorithm {
                AdaBoostAlgorithm::Samme => {
                    let mut err = 0.0;
                    let mut wsum = 0.0;
                    for i in 0..n {
                        wsum += w[i];
                        if predict_class_one(&tree, x, i) != ylab[i] {
                            err += w[i];
                        }
                    }
                    err = if wsum > 0.0 { err / wsum } else { 1.0 };
                    if err >= 1.0 - 1.0 / k_f {
                        ctx.push(
                            Issue::builder(IssueCode::MeaninglessFit)
                                .message(format!("SAMME weak learner {m} is not better than random (err={err:.4})"))
                                .meaninglessness(Meaninglessness::vacuous(
                                    "AdaBoost SAMME stage",
                                    "the weighted error is at or worse than chance; α is undefined or non-positive",
                                    "use deeper trees or a linearly separable sample",
                                ))
                                .build(),
                        );
                        break;
                    }
                    let alpha = self.learning_rate
                        * (((1.0 - err) / err.max(1e-15)).ln() + (k_f - 1.0).ln());
                    for i in 0..n {
                        if predict_class_one(&tree, x, i) != ylab[i] {
                            w[i] *= alpha.exp();
                        }
                    }
                    let z: f64 = w.iter().sum::<f64>().max(1e-15);
                    for wi in &mut w {
                        *wi /= z;
                    }
                    ctx.session.step(m as u64, err, Some(alpha));
                    alphas.push(alpha);
                    trees.push(tree);
                }
                AdaBoostAlgorithm::SammeR => {
                    let factor = (k_f - 1.0) / k_f;
                    let mut loss = 0.0;
                    for i in 0..n {
                        let p = predict_class_proba(&tree, x, i, k);
                        if let Some(c) = class_index(ylab[i], &classes) {
                            let lp = p[c].max(1e-15).ln();
                            loss -= lp;
                            w[i] *= (-self.learning_rate * factor * lp).exp();
                        }
                    }
                    let z: f64 = w.iter().sum::<f64>().max(1e-15);
                    for wi in &mut w {
                        *wi /= z;
                    }
                    ctx.session.step(m as u64, loss, None);
                    trees.push(tree);
                }
            }
        }
        if trees.is_empty() && k >= 2 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("AdaBoost produced no usable weak learners")
                    .build(),
            );
        } else {
            ctx.session.converged(
                format!("{} AdaBoost stages", trees.len()),
                trees.len() as u64,
            );
        }
        let fitted = FittedAdaBoost {
            trees,
            alphas,
            algorithm: self.algorithm,
            learning_rate: self.learning_rate,
            classes,
            n_features: x.ncols(),
        };
        let pred = fitted.predict_vec(x);
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// AdaBoost.R2 regressor (Drucker 1997).
#[derive(Clone, Debug)]
pub struct AdaBoostRegressor {
    /// Number of weak learners.
    pub n_estimators: usize,
    /// Shrinkage on \(\ln(1/\beta)\).
    pub learning_rate: f64,
    /// Weak-learner depth.
    pub max_depth: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for AdaBoostRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 30,
            learning_rate: 1.0,
            max_depth: 3,
            seed: 0,
        }
    }
}

impl AdaBoostRegressor {
    /// Default AdaBoost.R2.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted AdaBoost.R2 model.
#[derive(Clone, Debug)]
pub struct FittedAdaBoostRegressor {
    trees: Vec<RegNode>,
    /// Stage weights \(\ln(1/\beta_m)\).
    pub alphas: Vec<f64>,
    /// Training feature count.
    pub n_features: usize,
}

impl FittedAdaBoostRegressor {
    fn predict_vec(&self, x: &Matrix) -> Vector {
        Vector::from_iter((0..x.nrows()).map(|i| {
            let mut pairs: Vec<(f64, f64)> = self
                .trees
                .iter()
                .zip(&self.alphas)
                .map(|(t, a)| (predict_reg_one(t, x, i), *a))
                .collect();
            if pairs.is_empty() {
                return 0.0;
            }
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let tot: f64 = pairs.iter().map(|(_, a)| *a).sum();
            let mut acc = 0.0;
            for (v, a) in pairs {
                acc += a;
                if acc >= 0.5 * tot {
                    return v;
                }
            }
            0.0
        }))
    }
}

impl Predict for FittedAdaBoostRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(self.predict_vec(x))
    }
}

impl Fit for AdaBoostRegressor {
    type Fitted = FittedAdaBoostRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedAdaBoostRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let n = x.nrows();
        if n == 0 || ctx.report.contains(IssueCode::ConstantTarget) {
            return ctx.finish(FittedAdaBoostRegressor {
                trees: Vec::new(),
                alphas: Vec::new(),
                n_features: x.ncols(),
            });
        }
        let ys = y.as_slice().to_vec();
        let mut w = vec![1.0 / n as f64; n];
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut alphas = Vec::new();
        for m in 0..self.n_estimators.max(1) {
            let mut trng = Rng::new(rng.next_u64());
            let sample = weighted_bootstrap(&w, &mut trng);
            let unit = vec![1.0; n];
            let tree = grow_reg(
                x,
                &ys,
                &sample,
                &unit,
                0,
                self.max_depth,
                2,
                None,
                false,
                &mut trng,
                ctx.policy.near_zero_variance,
            );
            let pred = predict_reg_vec(&tree, x);
            let mut max_e = 0.0;
            let mut err = vec![0.0; n];
            for i in 0..n {
                err[i] = (ys[i] - pred[i]).abs();
                if err[i] > max_e {
                    max_e = err[i];
                }
            }
            if max_e <= ctx.policy.near_zero_variance {
                ctx.session.step(m as u64, 0.0, Some(f64::INFINITY));
                trees.push(tree);
                alphas.push(1.0);
                break;
            }
            let mut lbar = 0.0;
            for i in 0..n {
                lbar += w[i] * (err[i] / max_e);
            }
            if lbar >= 0.5 {
                if trees.is_empty() {
                    ctx.push(
                        Issue::builder(IssueCode::MeaninglessFit)
                            .message(format!(
                                "AdaBoost.R2 stage {m} has weighted loss {lbar:.4} ≥ 1/2; β is undefined"
                            ))
                            .meaninglessness(Meaninglessness::vacuous(
                                "AdaBoost.R2 stage weight",
                                "the weak learner is not better than the median-absolute-error null",
                                "use deeper trees or a smoother target",
                            ))
                            .build(),
                    );
                } else {
                    ctx.push(
                        Issue::builder(IssueCode::DidNotConverge)
                            .message(format!(
                                "AdaBoost.R2 stopped at stage {m}: weighted loss {lbar:.4} ≥ 1/2"
                            ))
                            .build(),
                    );
                }
                break;
            }
            let beta = (lbar / (1.0 - lbar).max(1e-15)).max(1e-12);
            let alpha = self.learning_rate * (1.0 / beta).ln();
            for i in 0..n {
                w[i] *= beta.powf(1.0 - err[i] / max_e);
            }
            let z: f64 = w.iter().sum::<f64>().max(1e-15);
            for wi in &mut w {
                *wi /= z;
            }
            ctx.session.step(m as u64, lbar, Some(alpha));
            trees.push(tree);
            alphas.push(alpha);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("AdaBoost.R2 produced no usable weak learners")
                    .build(),
            );
        }
        let fitted = FittedAdaBoostRegressor {
            trees,
            alphas,
            n_features: x.ncols(),
        };
        let pred = fitted.predict_vec(x);
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Isolation Forest (Liu, Ting, Zhou): random-split path-length anomaly scores.
///
/// The later `anomaly` module reuses [`FittedIsolationForest`] scores.
#[derive(Clone, Debug)]
pub struct IsolationForest {
    /// Number of isolation trees.
    pub n_trees: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self {
            n_trees: 50,
            seed: 0,
        }
    }
}

impl IsolationForest {
    /// Isolation forest with `n_trees` trees.
    pub fn new(n_trees: usize) -> Self {
        Self { n_trees, seed: 0 }
    }
}

/// Fitted isolation forest.
#[derive(Clone, Debug)]
pub struct FittedIsolationForest {
    trees: Vec<IsoNode>,
    /// Subsample size used to grow each tree (and in \(c(n)\)).
    pub max_samples: usize,
    /// Training feature count.
    pub n_features: usize,
    /// \(c(\texttt{max_samples})\).
    pub c_norm: f64,
}

impl FittedIsolationForest {
    /// Mean path length of row `i`.
    pub fn average_path_length(&self, x: &Matrix, i: usize) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }
        let mut s = 0.0;
        for t in &self.trees {
            s += iso_path(t, x, i);
        }
        s / self.trees.len() as f64
    }

    /// Liu et al. anomaly score \(s(x,n)=2^{-E(h)/c(n)}\) (higher = more anomalous).
    pub fn score_samples(&self, x: &Matrix) -> Vector {
        let c = if self.c_norm > 0.0 { self.c_norm } else { 1.0 };
        Vector::from_iter((0..x.nrows()).map(|i| {
            let eh = self.average_path_length(x, i);
            2.0_f64.powf(-eh / c)
        }))
    }
}

impl Predict for FittedIsolationForest {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(self.score_samples(x))
    }
}

impl FitUnsupervised for IsolationForest {
    type Fitted = FittedIsolationForest;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedIsolationForest>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        if n == 0 {
            return ctx.finish(FittedIsolationForest {
                trees: Vec::new(),
                max_samples: 0,
                n_features: x.ncols(),
                c_norm: 0.0,
            });
        }
        let all_constant = (0..x.ncols()).all(|j| {
            let col = x.column(j);
            signlred::slice_stats(col.as_slice()).is_constant(ctx.policy.near_zero_variance)
        });
        if all_constant {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message(
                        "Isolation Forest on a constant design: every path length is identical",
                    )
                    .meaninglessness(Meaninglessness::vacuous(
                        "isolation path lengths",
                        "no feature has variation, so random splits cannot isolate",
                        "do not rank anomalies on a point mass",
                    ))
                    .build(),
            );
        }
        let max_samples = n.min(256);
        let max_depth = (max_samples as f64).log2().ceil().max(1.0) as usize;
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let n_trees = self.n_trees.max(1);
        for t in 0..n_trees {
            let mut trng = Rng::new(rng.next_u64());
            let idx = if n > max_samples {
                trng.sample_indices(n, max_samples)
            } else {
                (0..n).collect()
            };
            trees.push(grow_iso(x, &idx, 0, max_depth, &mut trng));
            ctx.session.step(t as u64, 0.0, None);
        }
        ctx.session
            .converged(format!("{n_trees} isolation trees"), n_trees as u64);
        ctx.finish(FittedIsolationForest {
            trees,
            max_samples,
            n_features: x.ncols(),
            c_norm: isolation_c_factor(max_samples as f64),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    fn accuracy(pred: &Vector, y: &Vector) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        let mut ok = 0usize;
        for i in 0..y.len() {
            if (pred[i].round() - y[i].round()).abs() < 0.5 {
                ok += 1;
            }
        }
        ok as f64 / y.len() as f64
    }

    fn xor_like(n: usize, seed: u64) -> (Matrix, Vector) {
        let mut rng = Rng::new(seed);
        let mut data = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let a = if rng.uniform() < 0.5 { 0.0 } else { 1.0 };
            let b = if rng.uniform() < 0.5 { 0.0 } else { 1.0 };
            data[i * 2] = a + 0.02 * rng.standard_normal();
            data[i * 2 + 1] = b + 0.02 * rng.standard_normal();
            y[i] = if (a > 0.5) ^ (b > 0.5) { 1.0 } else { 0.0 };
        }
        (Matrix::from_row_major(n, 2, &data), Vector::from_slice(&y))
    }

    fn two_blobs(n_per: usize, seed: u64) -> (Matrix, Vector) {
        let mut rng = Rng::new(seed);
        let n = n_per * 2;
        let mut data = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let c = if i < n_per { 0.0 } else { 4.0 };
            data[i * 2] = c + 0.2 * rng.standard_normal();
            data[i * 2 + 1] = c + 0.2 * rng.standard_normal();
            y[i] = if i < n_per { 0.0 } else { 1.0 };
        }
        (Matrix::from_row_major(n, 2, &data), Vector::from_slice(&y))
    }

    #[test]
    fn cart_xor_accuracy() {
        let (x, y) = xor_like(48, 3);
        let session = Session::new("decision_tree_classifier", "fit");
        let q = DecisionTreeClassifier {
            max_depth: 4,
            ..DecisionTreeClassifier::default()
        }
        .fit(&x, &y, &session)
        .expect("tree");
        let pred = q
            .value
            .predict(&x, &Session::new("decision_tree_classifier", "predict"))
            .expect("pred")
            .value;
        assert!(accuracy(&pred, &y) > 0.8, "acc={}", accuracy(&pred, &y));
    }

    #[test]
    fn forest_two_blob_accuracy() {
        let (x, y) = two_blobs(20, 9);
        let session = Session::new("random_forest", "fit");
        let q = RandomForestClassifier {
            n_estimators: 12,
            seed: 2,
            ..RandomForestClassifier::default()
        }
        .fit(&x, &y, &session)
        .expect("rf");
        let pred = q
            .value
            .predict(&x, &Session::new("random_forest", "predict"))
            .expect("pred")
            .value;
        assert!(accuracy(&pred, &y) > 0.8, "acc={}", accuracy(&pred, &y));
    }

    #[test]
    fn extra_trees_and_boosters_two_blob() {
        let (x, y) = two_blobs(18, 4);
        let et = ExtraTreesClassifier {
            n_estimators: 16,
            seed: 5,
            ..ExtraTreesClassifier::default()
        }
        .fit(&x, &y, &Session::new("extra_trees", "fit"))
        .expect("et");
        let p = et
            .value
            .predict(&x, &Session::new("extra_trees", "predict"))
            .unwrap()
            .value;
        assert!(accuracy(&p, &y) > 0.8);

        let etc = ExtraTreeClassifier {
            seed: 5,
            ..ExtraTreeClassifier::default()
        }
        .fit(&x, &y, &Session::new("extra_tree", "fit"))
        .expect("etc");
        let p1 = etc
            .value
            .predict(&x, &Session::new("extra_tree", "predict"))
            .unwrap()
            .value;
        assert_eq!(p1.len(), y.len());

        let gbc = GradientBoostingClassifier {
            n_estimators: 20,
            learning_rate: 0.2,
            max_depth: 2,
            seed: 1,
            ..GradientBoostingClassifier::default()
        }
        .fit(&x, &y, &Session::new("gbc", "fit"))
        .expect("gbc");
        let p = gbc
            .value
            .predict(&x, &Session::new("gbc", "predict"))
            .unwrap()
            .value;
        assert!(accuracy(&p, &y) > 0.8);

        let ada = AdaBoostClassifier {
            n_estimators: 20,
            algorithm: AdaBoostAlgorithm::Samme,
            max_depth: 2,
            seed: 1,
            ..AdaBoostClassifier::default()
        }
        .fit(&x, &y, &Session::new("adaboost", "fit"))
        .expect("ada");
        let p = ada
            .value
            .predict(&x, &Session::new("adaboost", "predict"))
            .unwrap()
            .value;
        assert!(accuracy(&p, &y) > 0.8);
    }

    #[test]
    fn constant_y_errors() {
        let x = Matrix::from_fn(8, 2, |i, j| (i + j) as f64);
        let y = Vector::filled(8, 3.0);
        let err = DecisionTreeClassifier::new()
            .fit(&x, &y, &Session::new("tree", "fit"))
            .unwrap_err();
        assert!(
            err.primary().code == IssueCode::ConstantTarget
                || err.primary().code == IssueCode::SingleClass
        );
        assert!(
            err.report.contains(IssueCode::ConstantTarget)
                || err.report.contains(IssueCode::SingleClass)
        );
        let err = DecisionTreeRegressor::new()
            .fit(&x, &y, &Session::new("tree_reg", "fit"))
            .unwrap_err();
        assert_eq!(err.primary().code, IssueCode::ConstantTarget);
    }

    #[test]
    fn isolation_scores_far_point() {
        let (x, _) = two_blobs(16, 1);
        let session = Session::new("iforest", "fit");
        let q = IsolationForest {
            n_trees: 20,
            seed: 7,
        }
        .fit_unsupervised(&x, &session)
        .expect("if");
        let mut far = Matrix::zeros(1, 2);
        far.set(0, 0, 40.0);
        far.set(0, 1, -40.0);
        let s_in = q
            .value
            .predict(&x, &Session::new("iforest", "predict"))
            .unwrap()
            .value;
        let s_out = q
            .value
            .predict(&far, &Session::new("iforest", "predict"))
            .unwrap()
            .value;
        let mean_in = s_in.mean();
        assert!(
            s_out[0] > mean_in,
            "outlier score {} vs inlier mean {}",
            s_out[0],
            mean_in
        );
    }

    #[test]
    fn gbr_fits_line() {
        let x = Matrix::from_fn(12, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..12).map(|i| 0.5 * i as f64));
        let q = GradientBoostingRegressor {
            n_estimators: 25,
            learning_rate: 0.2,
            max_depth: 2,
            ..GradientBoostingRegressor::default()
        }
        .fit(&x, &y, &Session::new("gbr", "fit"))
        .expect("gbr");
        let etr = ExtraTreeRegressor {
            seed: 3,
            ..ExtraTreeRegressor::default()
        }
        .fit(&x, &y, &Session::new("etr", "fit"))
        .expect("etr");
        let etp = etr
            .value
            .predict(&x, &Session::new("etr", "predict"))
            .unwrap()
            .value;
        assert_eq!(etp.len(), y.len());
        let pred = q
            .value
            .predict(&x, &Session::new("gbr", "predict"))
            .unwrap()
            .value;
        let mut sse = 0.0;
        for i in 0..y.len() {
            let e = pred[i] - y[i];
            sse += e * e;
        }
        assert!(
            sse / (y.len() as f64) < 0.5,
            "mse={}",
            sse / (y.len() as f64)
        );
    }

    #[test]
    fn adaboost_r2_fits_a_line() {
        let x = Matrix::from_fn(16, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..16).map(|i| 0.4 * i as f64 + 0.1 * ((i % 3) as f64)));
        let q = AdaBoostRegressor {
            n_estimators: 25,
            max_depth: 4,
            ..AdaBoostRegressor::default()
        }
        .fit(&x, &y, &Session::new("abr", "fit"))
        .expect("abr");
        let pred = q
            .value
            .predict(&x, &Session::new("abr", "p"))
            .unwrap()
            .value;
        let mut sse = 0.0;
        for i in 0..y.len() {
            let e = pred[i] - y[i];
            sse += e * e;
        }
        assert!(
            sse / (y.len() as f64) < 1.0,
            "mse={}",
            sse / (y.len() as f64)
        );
    }
}
