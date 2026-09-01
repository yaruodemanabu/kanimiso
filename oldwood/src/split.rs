//! Axis-aligned split search on a [`faer::Mat<f64>`].

use crate::impurity::{class_index, gini, mse_of, weighted_counts};
use crate::rng::Rng;
use faer::Mat;

#[inline]
fn at(x: &Mat<f64>, i: usize, j: usize) -> f64 {
    x[(i, j)]
}

/// Partition `idx` by `x[i, feature] <= threshold`.
pub fn split_index(
    x: &Mat<f64>,
    idx: &[usize],
    feature: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in idx {
        if at(x, i, feature) <= threshold {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    (left, right)
}

/// Feature subsample at a node.
pub fn feature_subset(
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

fn class_gain(
    x: &Mat<f64>,
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
        if at(x, i, feature) <= threshold {
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

/// Best Gini split over `feats`. `extra` draws a random threshold per feature.
pub fn best_class_split(
    x: &Mat<f64>,
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
                let v = at(x, i, f);
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
                idx.iter().copied().map(|i| (at(x, i, f), i)).collect();
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

/// Best SSE split over `feats`.
pub fn best_reg_split(
    x: &Mat<f64>,
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
                let v = at(x, i, f);
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
                idx.iter().copied().map(|i| (at(x, i, f), i)).collect();
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
                let thr = 0.5 * (pts[s].0 + pts[s + 1].0);
                for &j in idx {
                    if at(x, j, f) <= thr {
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
                    if at(x, j, f) > thr {
                        let d = ys[j] - rmean;
                        right_sse += weights[j] * d * d;
                    }
                }
                let gain = parent_sse - left_sse - right_sse;
                if gain > best_gain {
                    best_gain = gain;
                    best = Some((f, thr));
                }
            }
        }
    }
    best
}
