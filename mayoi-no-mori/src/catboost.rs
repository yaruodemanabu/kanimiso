//! Leakage-resistant ordered target statistics followed by gradient boosting.

use std::collections::BTreeMap;

use amatsuki::seed_rng;
use oldwood::{DenseMatrix, MatrixView};

use crate::data::{
    checked_weights, validate_predict, validate_regression_target, validate_training,
};
use crate::gradient::positive_weight_classes;
use crate::numeric::{ScaledSum, StableWeightedMean};
use crate::random::shuffle;
use crate::{
    CatBoostOptions, Error, FittedGradientBoostingClassifier, FittedGradientBoostingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, Result,
};

/// Ordered-target-statistic regressor inspired by `CatBoost`.
///
/// Categories are supplied as numeric codes in the configured columns. NaN is
/// accepted there as one missing-category code. Non-categorical columns must
/// be finite. This implementation does not read or write `CatBoost` model files
/// and does not implement symmetric-tree or GPU training.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CatBoostRegressor {
    /// Ordered-statistic and additive-stage options.
    pub options: CatBoostOptions,
}

impl CatBoostRegressor {
    /// Creates a regressor with explicit ordered-statistic options.
    #[must_use]
    pub fn new(options: CatBoostOptions) -> Self {
        Self { options }
    }

    /// Fits ordered statistics without exposing a row's own target to its encoding.
    ///
    /// # Errors
    ///
    /// Returns an error when matrix dimensions, targets, weights, categorical
    /// columns, or options are invalid; when the CART kernel rejects a stage;
    /// or when an intermediate statistic is not representable.
    pub fn fit<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        target: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedCatBoostRegressor> {
        validate_training(x, target.len(), sample_weight)?;
        validate_regression_target(target)?;
        self.options.validate(x.ncols())?;
        let weights = checked_weights(x.nrows(), sample_weight);
        let prior = self.options.target_prior.unwrap_or(0.0);
        let (encoder, training) =
            OrderedEncoder::fit_transform(x, target, &weights, prior, &self.options)?;
        let model = GradientBoostingRegressor::new(self.options.boosting.clone()).fit(
            &training,
            target,
            Some(&weights),
        )?;
        Ok(FittedCatBoostRegressor {
            encoder,
            model,
            features: x.ncols(),
        })
    }
}

/// Fitted ordered-statistic regressor.
#[derive(Clone, Debug)]
pub struct FittedCatBoostRegressor {
    encoder: OrderedEncoder,
    model: FittedGradientBoostingRegressor,
    features: usize,
}

impl FittedCatBoostRegressor {
    /// Predicts using full-training posterior statistics for known categories.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature count differs from training, a feature
    /// value is invalid for its column, the CART kernel rejects prediction, or
    /// an additive prediction is not representable.
    pub fn predict<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<f64>> {
        validate_predict(x, self.features)?;
        self.model.predict(&self.encoder.transform(x)?)
    }

    /// Configured categorical feature indices.
    #[must_use]
    pub fn categorical_features(&self) -> &[usize] {
        &self.encoder.categorical_features
    }

    /// Normalized impurity-decrease importance in the encoded feature space.
    #[must_use]
    pub fn feature_importances(&self) -> Vec<f64> {
        self.model.feature_importances()
    }
}

/// Binary ordered-target-statistic classifier inspired by `CatBoost`.
///
/// Binary classification is explicit: fitting more or fewer than two
/// positive-weight classes returns [`Error::BinaryClasses`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CatBoostClassifier {
    /// Ordered-statistic and additive-stage options.
    pub options: CatBoostOptions,
}

impl CatBoostClassifier {
    /// Creates a binary classifier with explicit ordered-statistic options.
    #[must_use]
    pub fn new(options: CatBoostOptions) -> Self {
        Self { options }
    }

    /// Fits ordered binary-target statistics and softmax boosting.
    ///
    /// # Errors
    ///
    /// Returns an error when matrix dimensions, weights, categorical columns,
    /// options, or the binary-class requirement are invalid; when the CART
    /// kernel rejects a stage; or when an intermediate value is not representable.
    pub fn fit<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        target: &[usize],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedCatBoostClassifier> {
        validate_training(x, target.len(), sample_weight)?;
        self.options.validate(x.ncols())?;
        let prior = self.options.target_prior.unwrap_or(0.5);
        if !(0.0..=1.0).contains(&prior) {
            return Err(Error::InvalidOption {
                name: "target_prior",
                requirement: "in [0, 1] for binary classification",
            });
        }
        let weights = checked_weights(x.nrows(), sample_weight);
        let classes = positive_weight_classes(target, &weights);
        if classes.len() != 2 {
            return Err(Error::BinaryClasses {
                actual: classes.len(),
            });
        }
        let signal: Vec<f64> = target
            .iter()
            .map(|value| f64::from(*value == classes[1]))
            .collect();
        let (encoder, training) =
            OrderedEncoder::fit_transform(x, &signal, &weights, prior, &self.options)?;
        let model = GradientBoostingClassifier::new(self.options.boosting.clone()).fit(
            &training,
            target,
            Some(&weights),
        )?;
        Ok(FittedCatBoostClassifier {
            encoder,
            model,
            features: x.ncols(),
        })
    }
}

/// Fitted binary ordered-statistic classifier.
#[derive(Clone, Debug)]
pub struct FittedCatBoostClassifier {
    encoder: OrderedEncoder,
    model: FittedGradientBoostingClassifier,
    features: usize,
}

impl FittedCatBoostClassifier {
    /// Sorted binary target labels defining probability-column order.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        self.model.classes()
    }

    /// Predicts posterior class probabilities.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature count differs from training, a feature
    /// value is invalid for its column, the CART kernel rejects prediction, or
    /// probability normalization cannot be represented.
    pub fn predict_proba<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<Vec<f64>>> {
        validate_predict(x, self.features)?;
        self.model.predict_proba(&self.encoder.transform(x)?)
    }

    /// Predicts original target labels rather than zero-based indices.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::predict_proba`].
    pub fn predict<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<usize>> {
        validate_predict(x, self.features)?;
        self.model.predict(&self.encoder.transform(x)?)
    }

    /// Configured categorical feature indices.
    #[must_use]
    pub fn categorical_features(&self) -> &[usize] {
        &self.encoder.categorical_features
    }

    /// Normalized impurity-decrease importance in the encoded feature space.
    #[must_use]
    pub fn feature_importances(&self) -> Vec<f64> {
        self.model.feature_importances()
    }
}

#[derive(Clone, Debug)]
struct OrderedEncoder {
    categorical_features: Vec<usize>,
    categorical: Vec<bool>,
    prior: f64,
    prior_strength: f64,
    statistics: Vec<BTreeMap<u64, CategoryStats>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct CategoryStats {
    values: StableWeightedMean,
}

impl CategoryStats {
    fn add(&mut self, signal: f64, weight: f64) -> Result<()> {
        self.values
            .add(signal, weight, "ordered category statistic")
    }
}

impl OrderedEncoder {
    fn fit_transform<M: MatrixView + ?Sized>(
        x: &M,
        signal: &[f64],
        weights: &[f64],
        prior: f64,
        options: &CatBoostOptions,
    ) -> Result<(Self, DenseMatrix)> {
        let mut categorical = vec![false; x.ncols()];
        for &feature in &options.categorical_features {
            categorical[feature] = true;
        }
        validate_categorical_matrix(x, &categorical)?;
        let mut training_values = vec![0.0; x.nrows() * x.ncols()];
        let mut ordered_sums = vec![ScaledSum::default(); x.nrows() * x.ncols()];
        for row in 0..x.nrows() {
            for column in 0..x.ncols() {
                if !categorical[column] {
                    training_values[row * x.ncols() + column] = x.get(row, column);
                }
            }
        }
        let mut rng = seed_rng(options.boosting.seed);
        for _ in 0..options.permutations {
            let mut order: Vec<usize> = (0..x.nrows()).collect();
            shuffle(&mut rng, &mut order);
            let mut running = vec![BTreeMap::<u64, CategoryStats>::new(); x.ncols()];
            for row in order {
                for &column in &options.categorical_features {
                    let key = category_key(x.get(row, column));
                    let stats = running[column].get(&key).copied().unwrap_or_default();
                    let encoded = posterior(stats, prior, options.prior_strength)?;
                    let index = row * x.ncols() + column;
                    ordered_sums[index].add(encoded, "ordered permutation average")?;
                    if weights[row] > 0.0 {
                        let entry = running[column].entry(key).or_default();
                        entry.add(signal[row], weights[row])?;
                    }
                }
            }
        }
        for row in 0..x.nrows() {
            for &column in &options.categorical_features {
                let index = row * x.ncols() + column;
                training_values[index] = ordered_sums[index]
                    .mean(options.permutations, "ordered permutation average")?;
            }
        }
        let mut statistics = vec![BTreeMap::<u64, CategoryStats>::new(); x.ncols()];
        for row in 0..x.nrows() {
            if weights[row] <= 0.0 {
                continue;
            }
            for &column in &options.categorical_features {
                let entry = statistics[column]
                    .entry(category_key(x.get(row, column)))
                    .or_default();
                entry.add(signal[row], weights[row])?;
            }
        }
        let encoder = Self {
            categorical_features: options.categorical_features.clone(),
            categorical,
            prior,
            prior_strength: options.prior_strength,
            statistics,
        };
        let training = DenseMatrix::from_row_major(x.nrows(), x.ncols(), training_values)?;
        Ok((encoder, training))
    }

    fn transform<M: MatrixView + ?Sized>(&self, x: &M) -> Result<DenseMatrix> {
        validate_categorical_matrix(x, &self.categorical)?;
        let mut values = Vec::with_capacity(x.nrows().saturating_mul(x.ncols()));
        for row in 0..x.nrows() {
            for column in 0..x.ncols() {
                if self.categorical[column] {
                    let stats = self.statistics[column]
                        .get(&category_key(x.get(row, column)))
                        .copied()
                        .unwrap_or_default();
                    values.push(posterior(stats, self.prior, self.prior_strength)?);
                } else {
                    values.push(x.get(row, column));
                }
            }
        }
        DenseMatrix::from_row_major(x.nrows(), x.ncols(), values).map_err(Into::into)
    }
}

fn validate_categorical_matrix<M: MatrixView + ?Sized>(x: &M, categorical: &[bool]) -> Result<()> {
    if x.ncols() != categorical.len() {
        return Err(Error::FeatureCount {
            expected: categorical.len(),
            actual: x.ncols(),
        });
    }
    for row in 0..x.nrows() {
        for (column, &is_categorical) in categorical.iter().enumerate() {
            let value = x.get(row, column);
            let valid = if is_categorical {
                !value.is_infinite()
            } else {
                value.is_finite()
            };
            if !valid {
                return Err(Error::NonFinite {
                    name: "feature",
                    index: row * x.ncols() + column,
                });
            }
        }
    }
    Ok(())
}

fn category_key(value: f64) -> u64 {
    let bits = value.to_bits();
    if value.is_nan() {
        f64::NAN.to_bits()
    } else if bits << 1 == 0 {
        0.0f64.to_bits()
    } else {
        bits
    }
}

fn posterior(stats: CategoryStats, prior: f64, prior_strength: f64) -> Result<f64> {
    if stats.values.mean("ordered category statistic")?.is_none() {
        return Ok(prior);
    }
    let mut combined = stats.values;
    combined.add(prior, prior_strength, "ordered category posterior")?;
    combined
        .mean("ordered category posterior")?
        .ok_or(Error::NumericalOverflow {
            operation: "ordered category posterior",
        })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;

    fn matrix(rows: usize, columns: usize, values: &[f64]) -> DenseMatrix {
        DenseMatrix::from_row_major(rows, columns, values.to_vec()).expect("valid fixture")
    }

    #[test]
    fn ordered_encoding_cannot_see_the_current_or_future_target() {
        let x = matrix(4, 1, &[10.0, 10.0, 10.0, 10.0]);
        let target = [0.0, 0.0, 1.0, 1.0];
        let options = CatBoostOptions {
            categorical_features: vec![0],
            permutations: 1,
            target_prior: Some(0.25),
            ..CatBoostOptions::default()
        };
        let mut order: Vec<usize> = (0..x.nrows()).collect();
        shuffle(&mut seed_rng(options.boosting.seed), &mut order);
        let changed_row = *order.last().expect("non-empty order");
        let mut changed_target = target;
        changed_target[changed_row] = 9.0;

        let (_, original) = OrderedEncoder::fit_transform(
            &x,
            &target,
            &[1.0; 4],
            options.target_prior.expect("test prior"),
            &options,
        )
        .expect("encode");
        let (_, changed) = OrderedEncoder::fit_transform(
            &x,
            &changed_target,
            &[1.0; 4],
            options.target_prior.expect("test prior"),
            &options,
        )
        .expect("encode changed target");
        assert!(original
            .as_slice()
            .iter()
            .zip(changed.as_slice())
            .all(|(left, right)| left.to_bits() == right.to_bits()));
    }

    #[test]
    fn regression_is_seed_reproducible_with_missing_category() {
        let x = matrix(
            8,
            2,
            &[
                0.0,
                0.0,
                0.1,
                0.0,
                0.2,
                1.0,
                0.3,
                1.0,
                0.7,
                f64::NAN,
                0.8,
                f64::NAN,
                0.9,
                2.0,
                1.0,
                2.0,
            ],
        );
        let target = [0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0];
        let estimator = CatBoostRegressor::new(CatBoostOptions {
            categorical_features: vec![1],
            boosting: crate::BoostingOptions {
                iterations: 12,
                seed: 17,
                ..crate::BoostingOptions::default()
            },
            ..CatBoostOptions::default()
        });
        let first = estimator.fit(&x, &target, None).expect("first");
        let second = estimator.fit(&x, &target, None).expect("second");
        assert_eq!(
            first.predict(&x).expect("first prediction"),
            second.predict(&x).expect("second prediction")
        );
    }

    #[test]
    fn classifier_rejects_multiclass_instead_of_silently_collapsing_it() {
        let x = matrix(3, 1, &[0.0, 1.0, 2.0]);
        let error = CatBoostClassifier::default()
            .fit(&x, &[1, 2, 3], None)
            .expect_err("multiclass is outside the documented subset");
        assert_eq!(error, Error::BinaryClasses { actual: 3 });
    }

    #[test]
    fn classifier_rejects_a_prior_outside_the_binary_signal_range() {
        let x = matrix(2, 1, &[0.0, 1.0]);
        let error = CatBoostClassifier::new(CatBoostOptions {
            categorical_features: vec![0],
            target_prior: Some(1.5),
            ..CatBoostOptions::default()
        })
        .fit(&x, &[2, 9], None)
        .expect_err("binary statistic prior must be a probability");
        assert!(matches!(
            error,
            Error::InvalidOption {
                name: "target_prior",
                ..
            }
        ));
    }

    #[test]
    fn posterior_preserves_extreme_prior_strengths_and_weights() {
        let tiny = f64::from_bits(1);
        let empty = posterior(CategoryStats::default(), 0.5, tiny).expect("empty posterior");
        assert_eq!(empty.to_bits(), 0.5_f64.to_bits());

        let mut balanced = CategoryStats::default();
        balanced.add(1.0, tiny).expect("tiny observation");
        let balanced_posterior = posterior(balanced, 0.5, tiny).expect("balanced tiny posterior");
        assert!(
            (balanced_posterior - 0.75).abs() <= f64::EPSILON,
            "actual={balanced_posterior:?}"
        );

        let mut huge = CategoryStats::default();
        huge.add(1.0, f64::MAX).expect("first huge weight");
        huge.add(1.0, f64::MAX).expect("second huge weight");
        let posterior = posterior(huge, 0.5, 1.0).expect("huge posterior");
        assert_eq!(posterior.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn posterior_preserves_a_small_signed_cancellation_residual() {
        let mut stats = CategoryStats::default();
        for signal in [1.0e16, 1.0, -1.0e16] {
            stats.add(signal, 1.0).expect("add signal");
        }
        let value = posterior(stats, 0.0, 1.0).expect("posterior");
        assert_eq!(value, 0.25);
    }

    #[test]
    fn permutation_average_preserves_an_extreme_fixed_prior() {
        let x = matrix(3, 1, &[1.0, 2.0, 3.0]);
        let options = CatBoostOptions {
            categorical_features: vec![0],
            target_prior: Some(f64::MAX),
            permutations: 3,
            ..CatBoostOptions::default()
        };
        let (_, encoded) =
            OrderedEncoder::fit_transform(&x, &[0.0; 3], &[1.0; 3], f64::MAX, &options)
                .expect("extreme prior average remains representable");
        assert!(encoded
            .as_slice()
            .iter()
            .all(|value| value.to_bits() == f64::MAX.to_bits()));
    }
}
