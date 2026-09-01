//! Recursive CART / isolation-tree growth on a [`faer::Mat<f64>`].

use crate::impurity::{mse_of, weighted_counts};
use crate::node::{class_leaf, ClassNode, IsoNode, RegNode};
use crate::rng::Rng;
use crate::split::{best_class_split, best_reg_split, feature_subset, split_index};
use faer::Mat;

/// Hyperparameters for a single CART grow.
#[derive(Clone, Copy, Debug)]
pub struct GrowSpec {
    /// Maximum depth (root is 0).
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` = all, or √p when [`Self::sqrt_features`]).
    pub max_features: Option<usize>,
    /// Extra-trees random thresholds.
    pub extra: bool,
    /// When `max_features` is `None`, subsample √p features.
    pub sqrt_features: bool,
    /// Minimum feature span treated as a distinct threshold.
    pub eps: f64,
}

impl Default for GrowSpec {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            extra: false,
            sqrt_features: false,
            eps: 1e-15,
        }
    }
}

/// Grow a Gini classification tree.
pub fn grow_class(
    x: &Mat<f64>,
    y: &[i64],
    classes: &[i64],
    idx: &[usize],
    weights: &[f64],
    spec: &GrowSpec,
    rng: &mut Rng,
) -> ClassNode {
    grow_class_at(x, y, classes, idx, weights, 0, spec, rng)
}

fn grow_class_at(
    x: &Mat<f64>,
    y: &[i64],
    classes: &[i64],
    idx: &[usize],
    weights: &[f64],
    depth: usize,
    spec: &GrowSpec,
    rng: &mut Rng,
) -> ClassNode {
    let counts = weighted_counts(y, classes, idx, weights);
    let n_eff = counts.iter().sum::<f64>();
    let pure = counts.iter().filter(|&&c| c > 0.0).count() <= 1;
    if depth >= spec.max_depth || idx.len() < spec.min_samples_split.max(2) || n_eff <= 0.0 || pure
    {
        return class_leaf(classes, &counts);
    }
    let feats = feature_subset(x.ncols(), spec.max_features, rng, spec.sqrt_features);
    let Some((feature, threshold)) = best_class_split(
        x, y, classes, idx, weights, &feats, spec.extra, rng, spec.eps,
    ) else {
        return class_leaf(classes, &counts);
    };
    let (left, right) = split_index(x, idx, feature, threshold);
    if left.is_empty() || right.is_empty() {
        return class_leaf(classes, &counts);
    }
    ClassNode::Split {
        feature,
        threshold,
        left: Box::new(grow_class_at(
            x,
            y,
            classes,
            &left,
            weights,
            depth + 1,
            spec,
            rng,
        )),
        right: Box::new(grow_class_at(
            x,
            y,
            classes,
            &right,
            weights,
            depth + 1,
            spec,
            rng,
        )),
    }
}

/// Grow an MSE regression tree.
pub fn grow_reg(
    x: &Mat<f64>,
    ys: &[f64],
    idx: &[usize],
    weights: &[f64],
    spec: &GrowSpec,
    rng: &mut Rng,
) -> RegNode {
    grow_reg_at(x, ys, idx, weights, 0, spec, rng)
}

fn grow_reg_at(
    x: &Mat<f64>,
    ys: &[f64],
    idx: &[usize],
    weights: &[f64],
    depth: usize,
    spec: &GrowSpec,
    rng: &mut Rng,
) -> RegNode {
    let (mean, sse, wsum) = mse_of(ys, idx, weights);
    if depth >= spec.max_depth
        || idx.len() < spec.min_samples_split.max(2)
        || sse <= spec.eps
        || wsum <= 0.0
    {
        return RegNode::Leaf {
            value: mean,
            n: wsum,
        };
    }
    let feats = feature_subset(x.ncols(), spec.max_features, rng, false);
    let Some((feature, threshold)) =
        best_reg_split(x, ys, idx, weights, &feats, spec.extra, rng, spec.eps)
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
        left: Box::new(grow_reg_at(x, ys, &left, weights, depth + 1, spec, rng)),
        right: Box::new(grow_reg_at(x, ys, &right, weights, depth + 1, spec, rng)),
    }
}

/// Rewrite logistic-boosting leaves to Newton steps \(∑ r / ∑ p(1-p)\).
pub fn rewrite_logistic_leaves(
    node: &mut RegNode,
    x: &Mat<f64>,
    r: &[f64],
    p: &[f64],
    idx: &[usize],
) {
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

/// Grow one isolation tree.
pub fn grow_iso(
    x: &Mat<f64>,
    idx: &[usize],
    depth: usize,
    max_depth: usize,
    rng: &mut Rng,
) -> IsoNode {
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
            let v = x[(i, f)];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predict::predict_class_one;
    use faer::Mat;

    #[test]
    fn cart_separates_two_blobs() {
        let x = Mat::<f64>::from_fn(4, 1, |i, _| if i < 2 { 0.0 } else { 1.0 });
        let y = [0_i64, 0, 1, 1];
        let classes = [0_i64, 1];
        let idx = [0usize, 1, 2, 3];
        let w = [1.0; 4];
        let spec = GrowSpec {
            max_depth: 2,
            min_samples_split: 2,
            ..GrowSpec::default()
        };
        let mut rng = Rng::new(1);
        let root = grow_class(&x, &y, &classes, &idx, &w, &spec, &mut rng);
        assert_eq!(predict_class_one(&root, &x, 0), 0);
        assert_eq!(predict_class_one(&root, &x, 3), 1);
    }
}
