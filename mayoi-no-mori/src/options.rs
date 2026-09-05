//! Runtime-only ensemble configuration.

use crate::{Error, Result};
use oldwood::TreeOptions;

/// Number of candidate features proposed independently at every tree node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeatureSampling {
    /// Propose every feature.
    All,
    /// Propose `max(1, floor(sqrt(n_features)))` features.
    SquareRoot,
    /// Propose `ceil(log2(n_features))`, with a minimum of one.
    Logarithm,
    /// Propose exactly this many features.
    Count(usize),
    /// Propose `ceil(fraction * n_features)` features.
    Fraction(f64),
}

impl FeatureSampling {
    pub(crate) fn count(self, features: usize) -> Result<usize> {
        let count = match self {
            Self::All => features,
            Self::SquareRoot => floor_square_root(features),
            Self::Logarithm => ceil_logarithm(features),
            Self::Count(value) => value,
            Self::Fraction(value) => {
                if !(value.is_finite() && value > 0.0 && value <= 1.0) {
                    return Err(Error::InvalidOption {
                        name: "feature_sampling fraction",
                        requirement: "finite and in (0, 1]",
                    });
                }
                ceil_fraction_count(features, value)
            }
        };
        if count == 0 || count > features {
            return Err(Error::InvalidOption {
                name: "feature_sampling",
                requirement: "select between 1 and n_features columns",
            });
        }
        Ok(count)
    }
}

fn floor_square_root(value: usize) -> usize {
    value.isqrt().max(1)
}

fn ceil_logarithm(value: usize) -> usize {
    let mut remainder = value.saturating_sub(1);
    let mut exponent = 0;
    while remainder > 0 {
        remainder >>= 1;
        exponent += 1;
    }
    exponent.max(1)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn ceil_fraction_count(total: usize, fraction: f64) -> usize {
    // The public option contract defines this count as ceil(total * fraction).
    // Callers validate `fraction` in (0, 1], and allocated collections bound
    // `total`; these are the only lossy conversions in that contract boundary.
    if fraction >= 1.0 {
        total
    } else {
        (count_as_f64(total) * fraction).ceil() as usize
    }
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn count_as_f64(count: usize) -> f64 {
    // Ensemble sizes are loop- and allocation-bounded, while means and
    // proportions are defined in f64. Keep that conversion at one boundary.
    count as f64
}

/// Resampling and tree-policy options shared by forest estimators.
#[derive(Clone, Debug, PartialEq)]
pub struct ForestOptions {
    /// Number of fitted trees.
    pub trees: usize,
    /// Whether rows are sampled with replacement.
    pub bootstrap: bool,
    /// Fraction of rows drawn for each tree, in `(0, 1]`.
    pub sample_fraction: f64,
    /// Node-local feature sampling policy.
    pub feature_sampling: FeatureSampling,
    /// Reproducible `ChaCha8` seed.
    pub seed: u64,
    /// Compute out-of-bag predictions when bootstrap sampling is enabled.
    pub out_of_bag: bool,
    /// Stopping options passed to the shared CART kernel.
    pub tree: TreeOptions,
}

impl Default for ForestOptions {
    fn default() -> Self {
        Self {
            trees: 100,
            bootstrap: true,
            sample_fraction: 1.0,
            feature_sampling: FeatureSampling::SquareRoot,
            seed: 0,
            out_of_bag: false,
            tree: TreeOptions::default(),
        }
    }
}

impl ForestOptions {
    pub(crate) fn validate(&self, rows: usize, features: usize) -> Result<(usize, usize)> {
        if self.trees == 0 {
            return Err(Error::InvalidOption {
                name: "trees",
                requirement: "at least 1",
            });
        }
        if !(self.sample_fraction.is_finite()
            && self.sample_fraction > 0.0
            && self.sample_fraction <= 1.0)
        {
            return Err(Error::InvalidOption {
                name: "sample_fraction",
                requirement: "finite and in (0, 1]",
            });
        }
        if self.out_of_bag && !self.bootstrap {
            return Err(Error::InvalidOption {
                name: "out_of_bag",
                requirement: "false unless bootstrap is enabled",
            });
        }
        if self.tree.max_features.is_some() {
            return Err(Error::InvalidOption {
                name: "tree.max_features",
                requirement: "None because feature_sampling owns node-level sampling",
            });
        }
        self.tree.validate(features)?;
        let sampled = ceil_fraction_count(rows, self.sample_fraction);
        Ok((sampled.max(1), self.feature_sampling.count(features)?))
    }
}

/// Options for first-order stochastic gradient boosting.
#[derive(Clone, Debug, PartialEq)]
pub struct BoostingOptions {
    /// Number of additive boosting stages.
    pub iterations: usize,
    /// Shrinkage applied to every fitted tree.
    pub learning_rate: f64,
    /// Fraction of positive-weight rows sampled without replacement per stage.
    pub sample_fraction: f64,
    /// Reproducible `ChaCha8` seed.
    pub seed: u64,
    /// Stopping options passed to the shared CART kernel.
    pub tree: TreeOptions,
}

impl Default for BoostingOptions {
    fn default() -> Self {
        let tree = TreeOptions {
            max_depth: Some(3),
            ..TreeOptions::default()
        };
        Self {
            iterations: 100,
            learning_rate: 0.1,
            sample_fraction: 1.0,
            seed: 0,
            tree,
        }
    }
}

impl BoostingOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.iterations == 0 {
            return Err(Error::InvalidOption {
                name: "iterations",
                requirement: "at least 1",
            });
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(Error::InvalidOption {
                name: "learning_rate",
                requirement: "finite and positive",
            });
        }
        if !(self.sample_fraction.is_finite()
            && self.sample_fraction > 0.0
            && self.sample_fraction <= 1.0)
        {
            return Err(Error::InvalidOption {
                name: "sample_fraction",
                requirement: "finite and in (0, 1]",
            });
        }
        Ok(())
    }
}

/// Histogram/Newton boosting options used by the LightGBM-style estimators.
#[derive(Clone, Debug, PartialEq)]
pub struct LightGbmOptions {
    /// Shared additive-stage options.
    pub boosting: BoostingOptions,
    /// Maximum number of non-missing bins per feature.
    pub max_bins: usize,
    /// Node-local feature sampling policy for each histogram tree.
    pub feature_sampling: FeatureSampling,
    /// L1 regularization applied to each leaf gradient sum.
    pub l1_regularization: f64,
    /// L2 regularization added to each leaf Hessian sum.
    pub l2_regularization: f64,
    /// Minimum Hessian sum in each leaf.
    pub min_hessian_leaf: f64,
}

impl Default for LightGbmOptions {
    fn default() -> Self {
        Self {
            boosting: BoostingOptions::default(),
            max_bins: 255,
            feature_sampling: FeatureSampling::All,
            l1_regularization: 0.0,
            l2_regularization: 0.0,
            min_hessian_leaf: 0.0,
        }
    }
}

impl LightGbmOptions {
    pub(crate) fn validate(&self, features: usize) -> Result<()> {
        self.boosting.validate()?;
        if self.max_bins < 2 {
            return Err(Error::InvalidOption {
                name: "max_bins",
                requirement: "at least 2",
            });
        }
        if self.boosting.tree.max_features.is_some() {
            return Err(Error::InvalidOption {
                name: "boosting.tree.max_features",
                requirement: "None because feature_sampling owns node-level sampling",
            });
        }
        self.boosting.tree.validate(features)?;
        for (name, value) in [
            ("l1_regularization", self.l1_regularization),
            ("l2_regularization", self.l2_regularization),
            ("min_hessian_leaf", self.min_hessian_leaf),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::InvalidOption {
                    name,
                    requirement: "finite and non-negative",
                });
            }
        }
        Ok(())
    }
}

/// Ordered-target-statistic options used by the CatBoost-style estimators.
#[derive(Clone, Debug, PartialEq)]
pub struct CatBoostOptions {
    /// Shared additive-stage options.
    pub boosting: BoostingOptions,
    /// Zero-based columns treated as categorical codes.
    pub categorical_features: Vec<usize>,
    /// Prior pseudo-count in every ordered target statistic.
    pub prior_strength: f64,
    /// Fixed target prior. `None` selects 0 for regression and 1/2 for binary classification.
    pub target_prior: Option<f64>,
    /// Number of deterministic random permutations averaged during training.
    pub permutations: usize,
}

impl Default for CatBoostOptions {
    fn default() -> Self {
        Self {
            boosting: BoostingOptions::default(),
            categorical_features: Vec::new(),
            prior_strength: 1.0,
            target_prior: None,
            permutations: 4,
        }
    }
}

impl CatBoostOptions {
    pub(crate) fn validate(&self, features: usize) -> Result<()> {
        self.boosting.validate()?;
        self.boosting.tree.validate(features)?;
        if !self.prior_strength.is_finite() || self.prior_strength <= 0.0 {
            return Err(Error::InvalidOption {
                name: "prior_strength",
                requirement: "finite and positive",
            });
        }
        if self.target_prior.is_some_and(|prior| !prior.is_finite()) {
            return Err(Error::InvalidOption {
                name: "target_prior",
                requirement: "finite when supplied",
            });
        }
        if self.permutations == 0 {
            return Err(Error::InvalidOption {
                name: "permutations",
                requirement: "at least 1",
            });
        }
        let mut seen = vec![false; features];
        for &feature in &self.categorical_features {
            if feature >= features || seen[feature] {
                return Err(Error::InvalidCategoricalFeature { feature });
            }
            seen[feature] = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_root_feature_count_uses_the_conventional_floor() {
        assert_eq!(FeatureSampling::SquareRoot.count(1), Ok(1));
        assert_eq!(FeatureSampling::SquareRoot.count(2), Ok(1));
        assert_eq!(FeatureSampling::SquareRoot.count(8), Ok(2));
        assert_eq!(FeatureSampling::SquareRoot.count(9), Ok(3));
    }
}
