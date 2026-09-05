/// Arena node shape.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    /// Terminal node.
    Leaf,
    /// Binary split. Rows with `x[feature] <= threshold` visit `left`.
    Split {
        /// Feature column index.
        feature: usize,
        /// Finite split threshold.
        threshold: f64,
        /// Left child arena index.
        left: usize,
        /// Right child arena index.
        right: usize,
    },
}

/// Immutable-by-API arena node with a typed prediction value.
#[derive(Clone, Debug, PartialEq)]
pub struct ArenaNode<T> {
    pub(crate) kind: NodeKind,
    pub(crate) sample_count: usize,
    pub(crate) weighted_sample_count: f64,
    pub(crate) impurity: f64,
    pub(crate) impurity_decrease: f64,
    pub(crate) value: T,
}

impl<T> ArenaNode<T> {
    /// Leaf or split description.
    #[must_use]
    pub fn kind(&self) -> &NodeKind {
        &self.kind
    }

    /// Number of positive-weight training rows in this node.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Sum of training weights in this node.
    #[must_use]
    pub fn weighted_sample_count(&self) -> f64 {
        self.weighted_sample_count
    }

    /// Criterion impurity before splitting.
    #[must_use]
    pub fn impurity(&self) -> f64 {
        self.impurity
    }

    /// Weighted child impurity reduction; zero for a leaf.
    #[must_use]
    pub fn impurity_decrease(&self) -> f64 {
        self.impurity_decrease
    }

    /// Node prediction payload.
    #[must_use]
    pub fn value(&self) -> &T {
        &self.value
    }
}

/// Classification prediction and weighted class histogram stored in a node.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationValue {
    pub(crate) predicted_class: usize,
    pub(crate) class_weights: Vec<f64>,
}

impl ClassificationValue {
    /// Smallest class label attaining the largest weight.
    #[must_use]
    pub fn predicted_class(&self) -> usize {
        self.predicted_class
    }

    /// Weighted class histogram aligned with [`FittedClassifier::classes`](crate::FittedClassifier::classes).
    #[must_use]
    pub fn class_weights(&self) -> &[f64] {
        &self.class_weights
    }
}
