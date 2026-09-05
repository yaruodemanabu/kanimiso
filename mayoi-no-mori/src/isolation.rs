//! Isolation Forest with an independent random-partition arena.
//!
//! Isolation trees optimize no CART impurity or gain. Each internal node
//! chooses a varying feature and a threshold uniformly at random, so this
//! implementation intentionally does not route through [`oldwood`] splitting.

use amatsuki::{seed_rng, Rng};
use oldwood::MatrixView;

use crate::data::validate_predict;
use crate::random::{below, sample_rows};
use crate::{Error, Result};

/// Runtime configuration for an Isolation Forest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IsolationForestOptions {
    /// Number of independent isolation trees.
    pub trees: usize,
    /// Maximum rows sampled without replacement for each tree.
    ///
    /// Training with fewer rows uses all available rows.
    pub max_samples: usize,
    /// Reproducible `ChaCha8` seed.
    pub seed: u64,
}

impl Default for IsolationForestOptions {
    fn default() -> Self {
        Self {
            trees: 100,
            max_samples: 256,
            seed: 0,
        }
    }
}

impl IsolationForestOptions {
    fn validate(&self) -> Result<()> {
        if self.trees == 0 {
            return Err(Error::InvalidOption {
                name: "trees",
                requirement: "at least 1",
            });
        }
        if self.max_samples == 0 {
            return Err(Error::InvalidOption {
                name: "max_samples",
                requirement: "at least 1",
            });
        }
        Ok(())
    }
}

/// Random-partition Isolation Forest anomaly scorer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IsolationForest {
    /// Tree-count, runtime subsample-size, and seed configuration.
    pub options: IsolationForestOptions,
}

impl IsolationForest {
    /// Creates an Isolation Forest with explicit runtime options.
    #[must_use]
    pub fn new(options: IsolationForestOptions) -> Self {
        Self { options }
    }

    /// Fits independent random isolation trees.
    ///
    /// Each tree samples `min(max_samples, n_rows)` rows without replacement
    /// and stops at `ceil(log2(sample_count))` or when no split is possible.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] for empty or non-finite input or invalid options.
    pub fn fit<M: MatrixView + ?Sized>(&self, x: &M) -> Result<FittedIsolationForest> {
        validate_training_matrix(x)?;
        self.options.validate()?;

        let max_samples = self.options.max_samples.min(x.nrows());
        let max_depth = isolation_depth_limit(max_samples);
        let mut rng = seed_rng(self.options.seed);
        let mut trees = Vec::with_capacity(self.options.trees);
        for _ in 0..self.options.trees {
            let rows = sample_rows(&mut rng, x.nrows(), max_samples, false);
            trees.push(IsolationTree::fit(x, rows, max_depth, &mut rng));
        }
        let path_adjustments = path_adjustments(max_samples);
        let normalizer = path_adjustments[max_samples];
        Ok(FittedIsolationForest {
            trees,
            max_samples,
            features: x.ncols(),
            normalizer,
            path_adjustments,
        })
    }
}

/// Fitted Isolation Forest with immutable random-partition trees.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedIsolationForest {
    trees: Vec<IsolationTree>,
    max_samples: usize,
    features: usize,
    normalizer: f64,
    path_adjustments: Vec<f64>,
}

impl FittedIsolationForest {
    /// Number of fitted isolation trees.
    #[must_use]
    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }

    /// Actual rows sampled by every tree.
    #[must_use]
    pub fn max_samples(&self) -> usize {
        self.max_samples
    }

    /// Number of feature columns required for prediction.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.features
    }

    /// Isolation normalization constant `c(max_samples)`.
    #[must_use]
    pub fn normalizer(&self) -> f64 {
        self.normalizer
    }

    /// Returns the mean adjusted path length for one matrix row.
    ///
    /// An external leaf reached at depth `d` with `n` training rows contributes
    /// `d + c(n)`, as in the Isolation Forest definition.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] for an incompatible or non-finite matrix or when
    /// `row` is outside the matrix.
    pub fn average_path_length<M: MatrixView + ?Sized>(&self, x: &M, row: usize) -> Result<f64> {
        validate_predict(x, self.features)?;
        if row >= x.nrows() {
            return Err(Error::InvalidOption {
                name: "row",
                requirement: "less than the prediction matrix row count",
            });
        }
        validate_finite_row(x, row)?;
        self.average_path_length_unchecked(x, row)
    }

    fn average_path_length_unchecked<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        row: usize,
    ) -> Result<f64> {
        let total: f64 = self
            .trees
            .iter()
            .map(|tree| tree.path_length(x, row, &self.path_adjustments))
            .sum();
        let average = total / count_as_f64(self.trees.len());
        if average.is_finite() {
            Ok(average)
        } else {
            Err(Error::NumericalOverflow {
                operation: "isolation path-length average",
            })
        }
    }

    /// Returns the mean adjusted path length for every matrix row.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] for an incompatible or non-finite matrix or
    /// non-representable path accumulation.
    pub fn average_path_lengths<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<f64>> {
        validate_prediction_matrix(x, self.features)?;
        (0..x.nrows())
            .map(|row| self.average_path_length_unchecked(x, row))
            .collect()
    }

    /// Returns Liu–Ting–Zhou scores `2^(-E[h(x)] / c(max_samples))`.
    ///
    /// Larger values indicate observations isolated by shorter paths. With a
    /// one-row training sample, the zero normalizer is defined to yield `1.0`.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] for an incompatible or non-finite matrix or
    /// non-representable score arithmetic.
    pub fn score_samples<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<f64>> {
        let paths = self.average_path_lengths(x)?;
        if self.normalizer == 0.0 {
            return Ok(vec![1.0; x.nrows()]);
        }
        paths
            .into_iter()
            .map(|path| {
                let score = 2.0_f64.powf(-path / self.normalizer);
                if score.is_finite() {
                    Ok(score)
                } else {
                    Err(Error::NumericalOverflow {
                        operation: "isolation anomaly score",
                    })
                }
            })
            .collect()
    }

    /// Predicts anomaly scores; this is an alias of [`Self::score_samples`].
    ///
    /// The estimator does not guess a contamination threshold or convert
    /// scores into inlier/outlier labels.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::score_samples`].
    pub fn predict<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<f64>> {
        self.score_samples(x)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct IsolationTree {
    nodes: Vec<IsolationNode>,
}

impl IsolationTree {
    fn fit<M: MatrixView + ?Sized, R: Rng + ?Sized>(
        x: &M,
        rows: Vec<usize>,
        max_depth: usize,
        rng: &mut R,
    ) -> Self {
        let mut tree = Self { nodes: Vec::new() };
        tree.grow(x, rows, 0, max_depth, rng);
        tree
    }

    fn grow<M: MatrixView + ?Sized, R: Rng + ?Sized>(
        &mut self,
        x: &M,
        rows: Vec<usize>,
        depth: usize,
        max_depth: usize,
        rng: &mut R,
    ) -> usize {
        let node = self.nodes.len();
        self.nodes
            .push(IsolationNode::External { size: rows.len() });
        if depth >= max_depth || rows.len() <= 1 {
            return node;
        }
        let varying = varying_features(x, &rows);
        if varying.is_empty() {
            return node;
        }
        let (feature, minimum, maximum) = varying[below(rng, varying.len())];
        let threshold = uniform_threshold(minimum, maximum, rng);
        let mut left_rows = Vec::with_capacity(rows.len());
        let mut right_rows = Vec::with_capacity(rows.len());
        for row in rows {
            if x.get(row, feature) <= threshold {
                left_rows.push(row);
            } else {
                right_rows.push(row);
            }
        }
        debug_assert!(!left_rows.is_empty() && !right_rows.is_empty());
        let left = self.grow(x, left_rows, depth + 1, max_depth, rng);
        let right = self.grow(x, right_rows, depth + 1, max_depth, rng);
        self.nodes[node] = IsolationNode::Internal {
            feature,
            threshold,
            left,
            right,
        };
        node
    }

    fn path_length<M: MatrixView + ?Sized>(&self, x: &M, row: usize, adjustments: &[f64]) -> f64 {
        let mut node = 0;
        let mut depth = 0_u32;
        loop {
            let current = self
                .nodes
                .get(node)
                .copied()
                .expect("fitted isolation-tree child indices stay inside their arena");
            match current {
                IsolationNode::External { size } => return f64::from(depth) + adjustments[size],
                IsolationNode::Internal {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    node = if x.get(row, feature) <= threshold {
                        left
                    } else {
                        right
                    };
                    depth += 1;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum IsolationNode {
    External {
        size: usize,
    },
    Internal {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
}

fn validate_training_matrix<M: MatrixView + ?Sized>(x: &M) -> Result<()> {
    if x.nrows() == 0 {
        return Err(Error::EmptyTrainingData);
    }
    if x.ncols() == 0 {
        return Err(Error::EmptyFeatures);
    }
    validate_finite_features(x)
}

fn validate_prediction_matrix<M: MatrixView + ?Sized>(x: &M, features: usize) -> Result<()> {
    validate_predict(x, features)?;
    validate_finite_features(x)
}

fn validate_finite_features<M: MatrixView + ?Sized>(x: &M) -> Result<()> {
    for row in 0..x.nrows() {
        validate_finite_row(x, row)?;
    }
    Ok(())
}

fn validate_finite_row<M: MatrixView + ?Sized>(x: &M, row: usize) -> Result<()> {
    for column in 0..x.ncols() {
        if !x.get(row, column).is_finite() {
            return Err(Error::NonFinite {
                name: "feature",
                index: row.saturating_mul(x.ncols()).saturating_add(column),
            });
        }
    }
    Ok(())
}

fn varying_features<M: MatrixView + ?Sized>(x: &M, rows: &[usize]) -> Vec<(usize, f64, f64)> {
    let mut varying = Vec::new();
    for feature in 0..x.ncols() {
        let mut minimum = x.get(rows[0], feature);
        let mut maximum = minimum;
        for &row in &rows[1..] {
            let value = x.get(row, feature);
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
        if minimum < maximum {
            varying.push((feature, minimum, maximum));
        }
    }
    varying
}

fn uniform_threshold<R: Rng + ?Sized>(minimum: f64, maximum: f64, rng: &mut R) -> f64 {
    let unit = rng.next_f64();
    let span = maximum - minimum;
    let candidate = if span.is_finite() {
        minimum + unit * span
    } else {
        (1.0 - unit) * minimum + unit * maximum
    };
    if candidate.is_finite() && candidate >= minimum && candidate < maximum {
        return candidate;
    }
    let midpoint = 0.5 * minimum + 0.5 * maximum;
    if midpoint.is_finite() && midpoint >= minimum && midpoint < maximum {
        midpoint
    } else {
        minimum
    }
}

fn isolation_depth_limit(samples: usize) -> usize {
    if samples <= 1 {
        0
    } else {
        (usize::BITS - (samples - 1).leading_zeros()) as usize
    }
}

fn path_adjustments(max_samples: usize) -> Vec<f64> {
    let mut adjustments = vec![0.0; max_samples + 1];
    let mut harmonic = 0.0;
    for (samples, adjustment) in adjustments.iter_mut().enumerate().skip(2) {
        harmonic += 1.0 / count_as_f64(samples - 1);
        *adjustment = 2.0 * harmonic - 2.0 * count_as_f64(samples - 1) / count_as_f64(samples);
    }
    adjustments
}

#[allow(clippy::cast_precision_loss)]
fn count_as_f64(value: usize) -> f64 {
    // Every caller passes an allocated tree or sample count. Counts above
    // f64's exact-integer range cannot be represented by this process anyway.
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use oldwood::DenseMatrix;

    fn matrix(rows: usize, columns: usize, values: &[f64]) -> DenseMatrix {
        DenseMatrix::from_row_major(rows, columns, values.to_vec()).expect("valid fixture")
    }

    #[test]
    fn two_points_have_the_closed_form_path_and_score() {
        let x = matrix(2, 1, &[0.0, 1.0]);
        let fitted = IsolationForest::new(IsolationForestOptions {
            trees: 7,
            max_samples: 2,
            seed: 31,
        })
        .fit(&x)
        .expect("fit");

        assert!((fitted.normalizer() - 1.0).abs() <= f64::EPSILON);
        assert!((fitted.average_path_length(&x, 0).expect("path") - 1.0).abs() <= f64::EPSILON);
        assert!((fitted.average_path_length(&x, 1).expect("path") - 1.0).abs() <= f64::EPSILON);
        assert_eq!(fitted.score_samples(&x).expect("score"), vec![0.5, 0.5]);
        assert_eq!(fitted.predict(&x).expect("predict"), vec![0.5, 0.5]);
    }

    #[test]
    fn equal_seed_replays_the_complete_arena() {
        let x = matrix(
            8,
            2,
            &[
                0.0, 7.0, 1.0, 6.0, 2.0, 5.0, 3.0, 4.0, 4.0, 3.0, 5.0, 2.0, 6.0, 1.0, 7.0, 0.0,
            ],
        );
        let model = IsolationForest::new(IsolationForestOptions {
            trees: 11,
            max_samples: 5,
            seed: 91,
        });
        let first = model.fit(&x).expect("first fit");
        let second = model.fit(&x).expect("second fit");
        assert_eq!(first, second);
        assert_eq!(
            first.score_samples(&x).expect("first scores"),
            second.score_samples(&x).expect("second scores")
        );
    }

    #[test]
    fn runtime_max_samples_is_capped_by_available_rows() {
        let x = matrix(3, 1, &[0.0, 1.0, 2.0]);
        let fitted = IsolationForest::new(IsolationForestOptions {
            trees: 1,
            max_samples: 99,
            seed: 0,
        })
        .fit(&x)
        .expect("fit");
        assert_eq!(fitted.max_samples(), 3);
    }

    #[test]
    fn invalid_training_inputs_return_typed_errors() {
        let empty = matrix(0, 1, &[]);
        assert_eq!(
            IsolationForest::default().fit(&empty),
            Err(Error::EmptyTrainingData)
        );

        let non_finite = matrix(1, 1, &[f64::NAN]);
        assert!(matches!(
            IsolationForest::default().fit(&non_finite),
            Err(Error::NonFinite {
                name: "feature",
                index: 0
            })
        ));

        let x = matrix(1, 1, &[0.0]);
        let invalid = IsolationForest::new(IsolationForestOptions {
            trees: 0,
            ..IsolationForestOptions::default()
        });
        assert!(matches!(
            invalid.fit(&x),
            Err(Error::InvalidOption { name: "trees", .. })
        ));
        let invalid = IsolationForest::new(IsolationForestOptions {
            max_samples: 0,
            ..IsolationForestOptions::default()
        });
        assert!(matches!(
            invalid.fit(&x),
            Err(Error::InvalidOption {
                name: "max_samples",
                ..
            })
        ));
    }

    #[test]
    fn prediction_validates_shape_row_and_finiteness() {
        let x = matrix(2, 1, &[0.0, 1.0]);
        let fitted = IsolationForest::default().fit(&x).expect("fit");
        let wrong_shape = matrix(1, 2, &[0.0, 1.0]);
        assert!(matches!(
            fitted.predict(&wrong_shape),
            Err(Error::FeatureCount {
                expected: 1,
                actual: 2
            })
        ));
        assert!(matches!(
            fitted.average_path_length(&x, 2),
            Err(Error::InvalidOption { name: "row", .. })
        ));
        let non_finite = matrix(1, 1, &[f64::INFINITY]);
        assert!(matches!(
            fitted.predict(&non_finite),
            Err(Error::NonFinite {
                name: "feature",
                ..
            })
        ));
    }
}
