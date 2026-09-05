use crate::{Error, Result};

/// Shared stopping and feature-selection options for CART.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeOptions {
    /// Maximum root-to-leaf split depth. `None` means no depth limit.
    pub max_depth: Option<usize>,
    /// Minimum positive-weight rows required before attempting a split.
    pub min_samples_split: usize,
    /// Minimum positive-weight rows required in each child.
    pub min_samples_leaf: usize,
    /// Minimum sum of sample weights required in each child.
    pub min_weight_leaf: f64,
    /// Minimum root-weighted impurity decrease, matching CART's
    /// `(node_weight / root_weight) * local_decrease` convention.
    pub min_impurity_decrease: f64,
    /// Number of leading feature columns considered at every node. `None`
    /// considers all columns. Feature selection is deterministic and uses no
    /// random number generator.
    pub max_features: Option<usize>,
}

impl Default for TreeOptions {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_weight_leaf: 0.0,
            min_impurity_decrease: 0.0,
            max_features: None,
        }
    }
}

impl TreeOptions {
    /// Validates every option against an input feature count.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidOption`] when a count or floating-point
    /// threshold is outside its documented domain.
    pub fn validate(&self, columns: usize) -> Result<()> {
        if self.min_samples_split < 2 {
            return Err(Error::InvalidOption {
                name: "min_samples_split",
                requirement: "at least 2",
            });
        }
        if self.min_samples_leaf == 0 {
            return Err(Error::InvalidOption {
                name: "min_samples_leaf",
                requirement: "at least 1",
            });
        }
        if !self.min_weight_leaf.is_finite() || self.min_weight_leaf < 0.0 {
            return Err(Error::InvalidOption {
                name: "min_weight_leaf",
                requirement: "finite and non-negative",
            });
        }
        if !self.min_impurity_decrease.is_finite() || self.min_impurity_decrease < 0.0 {
            return Err(Error::InvalidOption {
                name: "min_impurity_decrease",
                requirement: "finite and non-negative",
            });
        }
        if matches!(self.max_features, Some(0)) {
            return Err(Error::InvalidOption {
                name: "max_features",
                requirement: "at least 1 when specified",
            });
        }
        if self.max_features.is_some_and(|features| features > columns) {
            return Err(Error::InvalidOption {
                name: "max_features",
                requirement: "no greater than the matrix column count",
            });
        }
        Ok(())
    }

    pub(crate) fn feature_count(&self, columns: usize) -> usize {
        self.max_features.unwrap_or(columns)
    }
}
