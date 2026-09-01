//! Walk a grown tree on a [`faer::Mat<f64>`] row.

use crate::node::{isolation_c_factor, ClassNode, IsoNode, RegNode};
use faer::Mat;

/// Predicted class for row `i`.
pub fn predict_class_one(node: &ClassNode, x: &Mat<f64>, i: usize) -> i64 {
    match node {
        ClassNode::Leaf { class, .. } => *class,
        ClassNode::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            if x[(i, *feature)] <= *threshold {
                predict_class_one(left, x, i)
            } else {
                predict_class_one(right, x, i)
            }
        }
    }
}

/// Class-probability vector for row `i` (length `k`).
pub fn predict_class_proba(node: &ClassNode, x: &Mat<f64>, i: usize, k: usize) -> Vec<f64> {
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
            if x[(i, *feature)] <= *threshold {
                predict_class_proba(left, x, i, k)
            } else {
                predict_class_proba(right, x, i, k)
            }
        }
    }
}

/// Predicted labels for every row.
pub fn predict_class_labels(node: &ClassNode, x: &Mat<f64>) -> Vec<i64> {
    (0..x.nrows())
        .map(|i| predict_class_one(node, x, i))
        .collect()
}

/// Predicted value for row `i`.
pub fn predict_reg_one(node: &RegNode, x: &Mat<f64>, i: usize) -> f64 {
    match node {
        RegNode::Leaf { value, .. } => *value,
        RegNode::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            if x[(i, *feature)] <= *threshold {
                predict_reg_one(left, x, i)
            } else {
                predict_reg_one(right, x, i)
            }
        }
    }
}

/// Predicted values for every row.
pub fn predict_reg(node: &RegNode, x: &Mat<f64>) -> Vec<f64> {
    (0..x.nrows())
        .map(|i| predict_reg_one(node, x, i))
        .collect()
}

/// Isolation path length of row `i` (including \(c(size)\) at the leaf).
pub fn iso_path(node: &IsoNode, x: &Mat<f64>, i: usize) -> f64 {
    match node {
        IsoNode::External { size, depth } => *depth as f64 + isolation_c_factor(*size as f64),
        IsoNode::Internal {
            feature,
            threshold,
            left,
            right,
        } => {
            if x[(i, *feature)] <= *threshold {
                iso_path(left, x, i)
            } else {
                iso_path(right, x, i)
            }
        }
    }
}

/// Stable leaf code used by random-trees embedding.
pub fn iso_leaf_code(node: &IsoNode, x: &Mat<f64>, i: usize) -> u64 {
    match node {
        IsoNode::External { size, depth } => (*depth as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(*size as u64),
        IsoNode::Internal {
            feature,
            threshold,
            left,
            right,
        } => {
            let bit = if x[(i, *feature)] <= *threshold { 1 } else { 2 };
            iso_leaf_code(if bit == 1 { left } else { right }, x, i)
                .wrapping_mul(3)
                .wrapping_add(*feature as u64 + bit)
        }
    }
}
