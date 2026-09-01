//! Tree node types and the Isolation Forest path-length offset.

/// Euler–Mascheroni constant used by the Isolation Forest path-length offset.
const EULER_GAMMA: f64 = 0.5772156649015329;

/// Average unsuccessful BST path length \(c(n)\) (Liu et al., Isolation Forest).
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

/// Classification tree node.
#[derive(Clone, Debug)]
pub enum ClassNode {
    /// Terminal vote.
    Leaf {
        /// Majority class.
        class: i64,
        /// Weighted class counts (aligned with the training `classes` slice).
        counts: Vec<f64>,
    },
    /// Axis-aligned split.
    Split {
        /// Feature index.
        feature: usize,
        /// Left is `x[feature] <= threshold`.
        threshold: f64,
        /// Left child.
        left: Box<ClassNode>,
        /// Right child.
        right: Box<ClassNode>,
    },
}

/// Regression tree node.
#[derive(Clone, Debug)]
pub enum RegNode {
    /// Terminal mean (or Newton leaf after logistic rewrite).
    Leaf {
        /// Predicted value.
        value: f64,
        /// Effective sample weight at the leaf.
        n: f64,
    },
    /// Axis-aligned split.
    Split {
        /// Feature index.
        feature: usize,
        /// Left is `x[feature] <= threshold`.
        threshold: f64,
        /// Left child.
        left: Box<RegNode>,
        /// Right child.
        right: Box<RegNode>,
    },
}

/// Isolation tree node.
#[derive(Clone, Debug)]
pub enum IsoNode {
    /// External node (isolated or depth-capped).
    External {
        /// Subsample size that reached this node.
        size: usize,
        /// Depth of this node.
        depth: usize,
    },
    /// Random-split internal node.
    Internal {
        /// Feature index.
        feature: usize,
        /// Split threshold.
        threshold: f64,
        /// Left child.
        left: Box<IsoNode>,
        /// Right child.
        right: Box<IsoNode>,
    },
}

/// True when the classifier is a single leaf.
pub fn is_class_stump(node: &ClassNode) -> bool {
    matches!(node, ClassNode::Leaf { .. })
}

/// Majority-class leaf from a histogram.
pub fn class_leaf(classes: &[i64], counts: &[f64]) -> ClassNode {
    ClassNode::Leaf {
        class: super::impurity::majority(classes, counts),
        counts: counts.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolation_c_small_n() {
        assert_eq!(isolation_c_factor(0.0), 0.0);
        assert_eq!(isolation_c_factor(1.0), 0.0);
        assert_eq!(isolation_c_factor(2.0), 1.0);
        let c3 = isolation_c_factor(3.0);
        let expect = 2.0 * (2.0_f64.ln() + EULER_GAMMA) - 4.0 / 3.0;
        assert!((c3 - expect).abs() < 1e-15);
    }
}
