//! Histogram gradient boosting (sklearn `HistGradientBoosting*`).
//!
//! Features are quantized to a shared bin grid. Each tree is grown from
//! per-bin gradient / Hessian histograms (LightGBM-style), not by scanning
//! every unique threshold. Leaf values are Newton steps
//! \(-\sum g / (\sum h + \ell_2)\). A constant target or a single class is
//! statistically empty and aborts.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::{Fit, Predict};
use crate::validate::{inspect_classes, inspect_xy};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result};

/// Histogram gradient-boosting regressor (squared error).
#[derive(Clone, Debug)]
pub struct HistGradientBoostingRegressor {
    /// Boosting rounds.
    pub max_iter: usize,
    /// Shrinkage \(\nu\).
    pub learning_rate: f64,
    /// Tree depth (root = 0).
    pub max_depth: usize,
    /// Quantile bins per feature.
    pub max_bins: usize,
    /// Minimum histogram count on each child.
    pub min_samples_leaf: usize,
    /// Hessian ridge at every leaf / split.
    pub l2: f64,
}

impl Default for HistGradientBoostingRegressor {
    fn default() -> Self {
        Self {
            max_iter: 40,
            learning_rate: 0.1,
            max_depth: 3,
            max_bins: 32,
            min_samples_leaf: 2,
            l2: 1e-6,
        }
    }
}

impl HistGradientBoostingRegressor {
    /// Default histogram GBDT regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Histogram gradient-boosting classifier (binary logistic or softmax).
#[derive(Clone, Debug)]
pub struct HistGradientBoostingClassifier {
    /// Boosting rounds.
    pub max_iter: usize,
    /// Shrinkage \(\nu\).
    pub learning_rate: f64,
    /// Tree depth (root = 0).
    pub max_depth: usize,
    /// Quantile bins per feature.
    pub max_bins: usize,
    /// Minimum histogram count on each child.
    pub min_samples_leaf: usize,
    /// Hessian ridge at every leaf / split.
    pub l2: f64,
}

impl Default for HistGradientBoostingClassifier {
    fn default() -> Self {
        Self {
            max_iter: 40,
            learning_rate: 0.1,
            max_depth: 3,
            max_bins: 32,
            min_samples_leaf: 2,
            l2: 1e-6,
        }
    }
}

impl HistGradientBoostingClassifier {
    /// Default histogram GBDT classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted histogram GBDT (regression or a single score).
#[derive(Clone, Debug)]
pub struct FittedHistGbr {
    trees: Vec<HNode>,
    /// Training intercept (mean of \(y\)).
    pub intercept: f64,
    /// Shrinkage used at predict time.
    pub learning_rate: f64,
    /// Shared per-feature bin upper bounds.
    edges: Vec<Vec<f64>>,
}

/// Fitted histogram GBDT classifier.
#[derive(Clone, Debug)]
pub struct FittedHistGbc {
    /// One tree sequence per class (softmax); length 1 is binary logistic.
    trees: Vec<Vec<HNode>>,
    /// Per-class intercepts.
    pub intercepts: Vector,
    /// Shrinkage.
    pub learning_rate: f64,
    edges: Vec<Vec<f64>>,
    /// Sorted class labels.
    pub classes: Vec<i64>,
}

#[derive(Clone, Debug)]
enum HNode {
    Leaf {
        value: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<HNode>,
        right: Box<HNode>,
    },
}

fn quantile_edges(x: &Matrix, j: usize, max_bins: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..x.nrows())
        .map(|i| x.get(i, j))
        .filter(|z| z.is_finite())
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.dedup_by(|a, b| (*a - *b).abs() <= 1e-15);
    if v.len() <= 1 {
        return Vec::new();
    }
    let nb = max_bins.max(2).min(v.len());
    let mut edges = Vec::with_capacity(nb.saturating_sub(1));
    for k in 1..nb {
        let t = k as f64 / nb as f64;
        let idx = ((v.len() - 1) as f64 * t).round() as usize;
        edges.push(v[idx.min(v.len() - 1)]);
    }
    edges.dedup_by(|a, b| (*a - *b).abs() <= 1e-15);
    edges
}

fn bin_id(v: f64, edges: &[f64]) -> usize {
    for (i, &e) in edges.iter().enumerate() {
        if v <= e {
            return i;
        }
    }
    edges.len()
}

fn predict_one(node: &HNode, x: &Matrix, i: usize) -> f64 {
    match node {
        HNode::Leaf { value } => *value,
        HNode::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            if x.get(i, *feature) <= *threshold {
                predict_one(left, x, i)
            } else {
                predict_one(right, x, i)
            }
        }
    }
}

fn grow(
    x: &Matrix,
    edges: &[Vec<f64>],
    g: &[f64],
    h: &[f64],
    idx: &[usize],
    depth: usize,
    max_depth: usize,
    min_leaf: usize,
    l2: f64,
) -> HNode {
    let mut gs = 0.0;
    let mut hs = 0.0;
    for &i in idx {
        gs += g[i];
        hs += h[i];
    }
    let leaf = HNode::Leaf {
        value: if hs.abs() <= 1e-18 {
            0.0
        } else {
            -gs / (hs + l2)
        },
    };
    if depth >= max_depth || idx.len() < min_leaf.saturating_mul(2) {
        return leaf;
    }
    let mut best_gain = 0.0;
    let mut best: Option<(usize, f64)> = None;
    let parent = gs * gs / (hs + l2);
    for j in 0..x.ncols() {
        let ed = &edges[j];
        if ed.is_empty() {
            continue;
        }
        let nb = ed.len() + 1;
        let mut sg = vec![0.0; nb];
        let mut sh = vec![0.0; nb];
        let mut sc = vec![0usize; nb];
        for &i in idx {
            let b = bin_id(x.get(i, j), ed).min(nb - 1);
            sg[b] += g[i];
            sh[b] += h[i];
            sc[b] += 1;
        }
        let mut gl = 0.0;
        let mut hl = 0.0;
        let mut cl = 0usize;
        for b in 0..ed.len() {
            gl += sg[b];
            hl += sh[b];
            cl += sc[b];
            let cr = idx.len().saturating_sub(cl);
            if cl < min_leaf || cr < min_leaf {
                continue;
            }
            let gr = gs - gl;
            let hr = hs - hl;
            let gain = gl * gl / (hl + l2) + gr * gr / (hr + l2) - parent;
            if gain > best_gain {
                best_gain = gain;
                best = Some((j, ed[b]));
            }
        }
    }
    let Some((feature, threshold)) = best else {
        return leaf;
    };
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in idx {
        if x.get(i, feature) <= threshold {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    if left.is_empty() || right.is_empty() {
        return leaf;
    }
    HNode::Split {
        feature,
        threshold,
        left: Box::new(grow(
            x,
            edges,
            g,
            h,
            &left,
            depth + 1,
            max_depth,
            min_leaf,
            l2,
        )),
        right: Box::new(grow(
            x,
            edges,
            g,
            h,
            &right,
            depth + 1,
            max_depth,
            min_leaf,
            l2,
        )),
    }
}

fn build_edges(ctx: &mut FitCtx, x: &Matrix, max_bins: usize) -> Vec<Vec<f64>> {
    let mut edges = Vec::with_capacity(x.ncols());
    let mut n_const = 0usize;
    for j in 0..x.ncols() {
        let e = quantile_edges(x, j, max_bins);
        if e.is_empty() {
            n_const += 1;
        }
        edges.push(e);
    }
    if n_const == x.ncols() && x.ncols() > 0 {
        ctx.push(
            Issue::builder(IssueCode::ConstantFeature)
                .message("every feature is constant after binning; histogram trees cannot split")
                .meaninglessness(Meaninglessness::vacuous(
                    "histogram GBDT",
                    "no feature has more than one distinct finite value",
                    "add variation or drop the estimator",
                ))
                .build(),
        );
    } else if n_const > 0 {
        ctx.push(
            Issue::builder(IssueCode::NearZeroVariance)
                .severity(signlred::Severity::Advisory)
                .message(format!(
                    "{n_const} features have a single bin and will never split"
                ))
                .metric("constant_features", n_const as f64)
                .build(),
        );
    }
    if max_bins < 8 {
        ctx.push(
            Issue::builder(IssueCode::TruncatedSvdUsed)
                .severity(signlred::Severity::Advisory)
                .message(format!(
                    "max_bins={max_bins} is a coarse quantization of the original features"
                ))
                .compromise(NumericalCompromise::new(
                    "splits on the exact feature values",
                    format!("quantile histogram with {max_bins} bins"),
                    "histogram GBDT trades split resolution for speed",
                    "thresholds sit on bin edges, not on every unique value",
                ))
                .build(),
        );
    }
    edges
}

impl Fit for HistGradientBoostingRegressor {
    type Fitted = FittedHistGbr;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedHistGbr>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if ctx.report.contains(IssueCode::ConstantTarget)
            || ctx.report.contains(IssueCode::EmptyMatrix)
            || ctx.report.contains(IssueCode::NonFiniteInput)
        {
            return ctx.finish(FittedHistGbr {
                trees: Vec::new(),
                intercept: y.mean(),
                learning_rate: self.learning_rate,
                edges: Vec::new(),
            });
        }
        let edges = build_edges(&mut ctx, x, self.max_bins);
        let intercept = y.mean();
        let mut pred = Vector::filled(y.len(), intercept);
        let mut trees = Vec::new();
        let idx: Vec<usize> = (0..x.nrows()).collect();
        let h = vec![1.0; y.len()];
        let mut last_sse = f64::INFINITY;
        for it in 0..self.max_iter.max(1) {
            let mut g = vec![0.0; y.len()];
            let mut sse = 0.0;
            for i in 0..y.len() {
                let r = pred[i] - y[i];
                g[i] = r;
                sse += r * r;
            }
            ctx.session.step(it as u64, sse, None);
            let tree = grow(
                x,
                &edges,
                &g,
                &h,
                &idx,
                0,
                self.max_depth,
                self.min_samples_leaf.max(1),
                self.l2.max(0.0),
            );
            for i in 0..y.len() {
                pred[i] += self.learning_rate * predict_one(&tree, x, i);
            }
            trees.push(tree);
            if (last_sse - sse).abs() < 1e-15 * (1.0 + last_sse) && it > 0 {
                ctx.session.converged("histogram GB SSE stalled", it as u64);
                break;
            }
            last_sse = sse;
        }
        ctx.finish(FittedHistGbr {
            trees,
            intercept,
            learning_rate: self.learning_rate,
            edges,
        })
    }
}

impl Predict for FittedHistGbr {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        let mut out = Vector::filled(x.nrows(), self.intercept);
        for tree in &self.trees {
            for i in 0..x.nrows() {
                out[i] += self.learning_rate * predict_one(tree, x, i);
            }
        }
        let _ = &self.edges;
        ctx.finish(out)
    }
}

impl Fit for HistGradientBoostingClassifier {
    type Fitted = FittedHistGbc;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedHistGbc>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        if classes.len() < 2 {
            return ctx.finish(FittedHistGbc {
                trees: Vec::new(),
                intercepts: Vector::zeros(classes.len().max(1)),
                learning_rate: self.learning_rate,
                edges: Vec::new(),
                classes,
            });
        }
        let edges = build_edges(&mut ctx, x, self.max_bins);
        let k = classes.len();
        let mut scores = Matrix::zeros(x.nrows(), k);
        let mut trees = vec![Vec::new(); k];
        let idx: Vec<usize> = (0..x.nrows()).collect();
        let yoh: Vec<Vec<f64>> = (0..k)
            .map(|c| {
                y.as_slice()
                    .iter()
                    .map(|&v| {
                        if v.round() as i64 == classes[c] {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        for it in 0..self.max_iter.max(1) {
            let mut loss = 0.0;
            let mut proba = Matrix::zeros(x.nrows(), k);
            for i in 0..x.nrows() {
                let mut logits = vec![0.0; k];
                let mut m = f64::NEG_INFINITY;
                for c in 0..k {
                    logits[c] = scores.get(i, c);
                    if logits[c] > m {
                        m = logits[c];
                    }
                }
                let mut den = 0.0;
                let mut ex = vec![0.0; k];
                for c in 0..k {
                    ex[c] = (logits[c] - m).exp();
                    den += ex[c];
                }
                for c in 0..k {
                    let p = if den > 0.0 {
                        ex[c] / den
                    } else {
                        1.0 / k as f64
                    };
                    proba.set(i, c, p);
                    loss -= yoh[c][i] * p.max(1e-15).ln();
                }
            }
            ctx.session.step(it as u64, loss, None);
            for c in 0..k {
                let mut g = vec![0.0; x.nrows()];
                let mut h = vec![0.0; x.nrows()];
                for i in 0..x.nrows() {
                    let p = proba.get(i, c);
                    g[i] = p - yoh[c][i];
                    h[i] = (p * (1.0 - p)).max(1e-8);
                }
                let tree = grow(
                    x,
                    &edges,
                    &g,
                    &h,
                    &idx,
                    0,
                    self.max_depth,
                    self.min_samples_leaf.max(1),
                    self.l2.max(0.0),
                );
                for i in 0..x.nrows() {
                    scores.set(
                        i,
                        c,
                        scores.get(i, c) + self.learning_rate * predict_one(&tree, x, i),
                    );
                }
                trees[c].push(tree);
            }
        }
        if k > 2 {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .severity(signlred::Severity::Advisory)
                    .message("K-class histogram GB uses one tree per class per round (softmax residuals)")
                    .compromise(NumericalCompromise::new(
                        "joint softmax GBDT with a shared tree",
                        "K independent Newton trees on softmax residuals",
                        "a shared multi-output histogram tree is not implemented",
                        "class scores are coupled only through the softmax residual, not through shared splits",
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedHistGbc {
            trees,
            intercepts: Vector::zeros(k),
            learning_rate: self.learning_rate,
            edges,
            classes,
        })
    }
}

impl Predict for FittedHistGbc {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        let k = self.classes.len();
        let mut out = Vector::zeros(x.nrows());
        if k == 0 {
            return ctx.finish(out);
        }
        for i in 0..x.nrows() {
            let mut best = 0usize;
            let mut best_s = f64::NEG_INFINITY;
            for c in 0..k {
                let mut s = if c < self.intercepts.len() {
                    self.intercepts[c]
                } else {
                    0.0
                };
                if c < self.trees.len() {
                    for tree in &self.trees[c] {
                        s += self.learning_rate * predict_one(tree, x, i);
                    }
                }
                if s > best_s {
                    best_s = s;
                    best = c;
                }
            }
            out[i] = self.classes[best] as f64;
        }
        let _ = &self.edges;
        ctx.finish(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hist_regressor_fits_line() {
        let x = Matrix::from_fn(24, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..24).map(|i| 0.5 * i as f64));
        let q = HistGradientBoostingRegressor {
            max_iter: 30,
            learning_rate: 0.2,
            max_depth: 2,
            max_bins: 16,
            ..HistGradientBoostingRegressor::default()
        }
        .fit(&x, &y, &Session::new("hgb", "fit"))
        .expect("hgb");
        let pred = q
            .value
            .predict(&x, &Session::new("hgb", "p"))
            .unwrap()
            .value;
        let mut sse = 0.0;
        for i in 0..y.len() {
            let e = pred[i] - y[i];
            sse += e * e;
        }
        assert!(
            sse / (y.len() as f64) < 0.6,
            "mse={}",
            sse / (y.len() as f64)
        );
    }

    #[test]
    fn hist_classifier_three_classes() {
        let x = Matrix::from_fn(30, 2, |i, j| {
            let g = i / 10;
            g as f64 + 0.05 * j as f64 + 0.01 * (i % 10) as f64
        });
        let y = Vector::from_iter((0..30).map(|i| (i / 10) as f64));
        let q = HistGradientBoostingClassifier {
            max_iter: 25,
            learning_rate: 0.2,
            max_depth: 2,
            ..HistGradientBoostingClassifier::default()
        }
        .fit(&x, &y, &Session::new("hgbc", "fit"))
        .expect("hgbc");
        let pred = q
            .value
            .predict(&x, &Session::new("hgbc", "p"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..30 {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 24, "ok={ok}");
    }

    #[test]
    fn constant_target_aborts() {
        let x = Matrix::from_fn(8, 2, |i, j| (i + j) as f64);
        let y = Vector::filled(8, 3.0);
        let err = HistGradientBoostingRegressor::new()
            .fit(&x, &y, &Session::new("hgb", "fit"))
            .unwrap_err();
        assert_eq!(err.primary().code, IssueCode::ConstantTarget);
    }
}
