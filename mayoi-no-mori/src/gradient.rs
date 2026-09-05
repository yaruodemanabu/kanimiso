//! First-order stochastic gradient boosting over `oldwood` regression trees.

use amatsuki::{seed_rng, ChaCha8Rng};
use oldwood::{DecisionTreeRegressor, FittedRegressor, MatrixView, RegressionCriterion};

use crate::data::{
    checked_weights, validate_finite_features, validate_predict, validate_regression_target,
    validate_training, IndexedRows,
};
use crate::options::{ceil_fraction_count, count_as_f64};
use crate::random::shuffle;
use crate::{BoostingOptions, Error, Result};

/// Squared-error gradient-boosted regression trees.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GradientBoostingRegressor {
    /// Additive-stage and CART configuration.
    pub options: BoostingOptions,
}

impl GradientBoostingRegressor {
    /// Creates a regressor with explicit runtime options.
    #[must_use]
    pub fn new(options: BoostingOptions) -> Self {
        Self { options }
    }

    /// Fits squared-error residuals, optionally using non-negative sample weights.
    ///
    /// # Errors
    ///
    /// Returns an error when matrix dimensions, targets, weights, or options
    /// are invalid; when the CART kernel rejects a stage; or when an additive
    /// calculation is not representable.
    pub fn fit<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        target: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedGradientBoostingRegressor> {
        validate_training(x, target.len(), sample_weight)?;
        validate_finite_features(x)?;
        validate_regression_target(target)?;
        self.options.validate()?;
        self.options.tree.validate(x.ncols())?;
        let weights = checked_weights(x.nrows(), sample_weight);
        let base = weighted_mean(target, &weights)?;
        let mut prediction = vec![base; x.nrows()];
        let mut trees = Vec::with_capacity(self.options.iterations);
        let mut rng = seed_rng(self.options.seed);
        for _ in 0..self.options.iterations {
            let residual: Vec<f64> = target
                .iter()
                .zip(&prediction)
                .zip(&weights)
                .map(|((&actual, &fitted), &weight)| {
                    if weight <= 0.0 {
                        return Ok(0.0);
                    }
                    let value = actual - fitted;
                    if value.is_finite() {
                        Ok(value)
                    } else {
                        Err(Error::NumericalOverflow {
                            operation: "gradient-boosting residual",
                        })
                    }
                })
                .collect::<Result<_>>()?;
            let rows = stage_rows(&weights, self.options.sample_fraction, &mut rng);
            let sampled_x = IndexedRows::new(x, &rows);
            let sampled_y: Vec<f64> = rows.iter().map(|&row| residual[row]).collect();
            let stage_weights: Vec<f64> = rows.iter().map(|&row| weights[row]).collect();
            let tree = DecisionTreeRegressor::new(
                RegressionCriterion::SquaredError,
                self.options.tree.clone(),
            )
            .fit(&sampled_x, &sampled_y, Some(&stage_weights))?;
            let update = tree.predict(&x)?;
            add_scaled(&mut prediction, &update, self.options.learning_rate)?;
            trees.push(tree);
        }
        Ok(FittedGradientBoostingRegressor {
            base,
            learning_rate: self.options.learning_rate,
            trees,
            features: x.ncols(),
        })
    }
}

/// Fitted squared-error gradient-boosted regressor.
#[derive(Clone, Debug)]
pub struct FittedGradientBoostingRegressor {
    base: f64,
    learning_rate: f64,
    trees: Vec<FittedRegressor>,
    features: usize,
}

impl FittedGradientBoostingRegressor {
    /// Predicts one value per row.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature count differs from training, the CART
    /// kernel rejects prediction, or an additive prediction is not representable.
    pub fn predict<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<f64>> {
        validate_predict(x, self.features)?;
        let mut output = vec![self.base; x.nrows()];
        for tree in &self.trees {
            add_scaled(&mut output, &tree.predict(&x)?, self.learning_rate)?;
        }
        Ok(output)
    }

    /// Weighted training-target mean used before the first stage.
    #[must_use]
    pub fn base_value(&self) -> f64 {
        self.base
    }

    /// Fitted stage trees in additive order.
    #[must_use]
    pub fn trees(&self) -> &[FittedRegressor] {
        &self.trees
    }

    /// Normalized mean impurity-decrease importance across all trees.
    #[must_use]
    pub fn feature_importances(&self) -> Vec<f64> {
        average_importances(
            self.trees.iter().map(FittedRegressor::feature_importances),
            self.features,
        )
    }
}

/// Multiclass softmax gradient-boosted classification trees.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GradientBoostingClassifier {
    /// Additive-stage and CART configuration.
    pub options: BoostingOptions,
}

impl GradientBoostingClassifier {
    /// Creates a classifier with explicit runtime options.
    #[must_use]
    pub fn new(options: BoostingOptions) -> Self {
        Self { options }
    }

    /// Fits multinomial deviance residuals for arbitrary `usize` class labels.
    ///
    /// # Errors
    ///
    /// Returns an error when matrix dimensions, weights, classes, or options
    /// are invalid; when the CART kernel rejects a stage; or when logits or
    /// probabilities cannot be represented.
    pub fn fit<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        target: &[usize],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedGradientBoostingClassifier> {
        validate_training(x, target.len(), sample_weight)?;
        validate_finite_features(x)?;
        self.options.validate()?;
        self.options.tree.validate(x.ncols())?;
        let weights = checked_weights(x.nrows(), sample_weight);
        let classes = positive_weight_classes(target, &weights);
        if classes.len() < 2 {
            return Err(Error::InvalidOption {
                name: "target classes",
                requirement: "at least two classes with positive weight",
            });
        }
        let base_logits = class_log_priors(target, &weights, &classes)?;
        let mut logits = vec![base_logits.clone(); x.nrows()];
        let mut stages = Vec::with_capacity(self.options.iterations);
        let mut rng = seed_rng(self.options.seed);
        for _ in 0..self.options.iterations {
            let probabilities: Vec<Vec<f64>> = logits
                .iter()
                .map(|row| softmax(row))
                .collect::<Result<_>>()?;
            let rows = stage_rows(&weights, self.options.sample_fraction, &mut rng);
            let sampled_x = IndexedRows::new(x, &rows);
            let stage_weights: Vec<f64> = rows.iter().map(|&row| weights[row]).collect();
            let mut stage = Vec::with_capacity(classes.len());
            for (class_index, &class) in classes.iter().enumerate() {
                let residual: Vec<f64> = rows
                    .iter()
                    .map(|&row| f64::from(target[row] == class) - probabilities[row][class_index])
                    .collect();
                let tree = DecisionTreeRegressor::new(
                    RegressionCriterion::SquaredError,
                    self.options.tree.clone(),
                )
                .fit(&sampled_x, &residual, Some(&stage_weights))?;
                let update = tree.predict(&x)?;
                for (row, value) in update.into_iter().enumerate() {
                    logits[row][class_index] = checked_add(
                        logits[row][class_index],
                        self.options.learning_rate * value,
                        "classification stage update",
                    )?;
                }
                stage.push(tree);
            }
            stages.push(stage);
        }
        Ok(FittedGradientBoostingClassifier {
            classes,
            base_logits,
            learning_rate: self.options.learning_rate,
            stages,
            features: x.ncols(),
        })
    }
}

/// Fitted multiclass softmax gradient-boosted classifier.
#[derive(Clone, Debug)]
pub struct FittedGradientBoostingClassifier {
    classes: Vec<usize>,
    base_logits: Vec<f64>,
    learning_rate: f64,
    stages: Vec<Vec<FittedRegressor>>,
    features: usize,
}

impl FittedGradientBoostingClassifier {
    /// Sorted target labels defining probability-column order.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Predicts one probability row per observation.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature count differs from training, the CART
    /// kernel rejects prediction, or logits cannot be accumulated or normalized.
    pub fn predict_proba<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<Vec<f64>>> {
        validate_predict(x, self.features)?;
        let mut logits = vec![self.base_logits.clone(); x.nrows()];
        for stage in &self.stages {
            for (class_index, tree) in stage.iter().enumerate() {
                for (row, value) in tree.predict(&x)?.into_iter().enumerate() {
                    logits[row][class_index] = checked_add(
                        logits[row][class_index],
                        self.learning_rate * value,
                        "classification prediction update",
                    )?;
                }
            }
        }
        logits.iter().map(|row| softmax(row)).collect()
    }

    /// Predicts the smallest class label attaining the largest probability.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::predict_proba`].
    pub fn predict<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<usize>> {
        Ok(self
            .predict_proba(x)?
            .iter()
            .map(|row| {
                let best = row
                    .iter()
                    .enumerate()
                    .max_by(|left, right| {
                        left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0))
                    })
                    .map_or(0, |(index, _)| index);
                self.classes[best]
            })
            .collect())
    }

    /// Fitted additive stages; each inner slice follows [`Self::classes`].
    #[must_use]
    pub fn stages(&self) -> &[Vec<FittedRegressor>] {
        &self.stages
    }

    /// Normalized mean impurity-decrease importance across all stage trees.
    #[must_use]
    pub fn feature_importances(&self) -> Vec<f64> {
        average_importances(
            self.stages
                .iter()
                .flat_map(|stage| stage.iter())
                .map(FittedRegressor::feature_importances),
            self.features,
        )
    }
}

pub(crate) fn weighted_mean(values: &[f64], weights: &[f64]) -> Result<f64> {
    let mut accumulator = crate::numeric::StableWeightedMean::default();
    for (&value, &weight) in values.iter().zip(weights) {
        accumulator.add(value, weight, "weighted mean")?;
    }
    accumulator
        .mean("weighted mean")?
        .ok_or(Error::NoPositiveWeight)
}

pub(crate) fn add_scaled(output: &mut [f64], update: &[f64], scale: f64) -> Result<()> {
    for (current, &increment) in output.iter_mut().zip(update) {
        *current = checked_add(*current, scale * increment, "additive prediction")?;
    }
    Ok(())
}

pub(crate) fn checked_add(left: f64, right: f64, operation: &'static str) -> Result<f64> {
    let value = left + right;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::NumericalOverflow { operation })
    }
}

pub(crate) fn softmax(logits: &[f64]) -> Result<Vec<f64>> {
    let maximum =
        logits
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .ok_or(Error::NumericalOverflow {
                operation: "empty softmax",
            })?;
    let exponentials: Vec<f64> = logits.iter().map(|value| (value - maximum).exp()).collect();
    let denominator: f64 = exponentials.iter().sum();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(Error::NumericalOverflow {
            operation: "softmax normalization",
        });
    }
    Ok(exponentials
        .into_iter()
        .map(|value| value / denominator)
        .collect())
}

pub(crate) fn positive_weight_classes(target: &[usize], weights: &[f64]) -> Vec<usize> {
    let mut classes: Vec<usize> = target
        .iter()
        .zip(weights)
        .filter_map(|(&class, &weight)| (weight > 0.0).then_some(class))
        .collect();
    classes.sort_unstable();
    classes.dedup();
    classes
}

pub(crate) fn class_log_priors(
    target: &[usize],
    weights: &[f64],
    classes: &[usize],
) -> Result<Vec<f64>> {
    let mut log_counts = vec![f64::NEG_INFINITY; classes.len()];
    for (&class, &weight) in target.iter().zip(weights) {
        if weight <= 0.0 {
            continue;
        }
        let index = classes
            .binary_search(&class)
            .expect("positive-weight classes include each positive-weight target");
        log_counts[index] = log_add_exp(log_counts[index], weight.ln());
    }
    let log_total = log_counts
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, log_add_exp);
    if !log_total.is_finite() {
        return Err(Error::NoPositiveWeight);
    }
    let priors: Vec<f64> = log_counts
        .into_iter()
        .map(|log_count| log_count - log_total)
        .collect();
    if priors.iter().all(|value| value.is_finite()) {
        Ok(priors)
    } else {
        Err(Error::NoPositiveWeight)
    }
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

pub(crate) fn stage_rows(weights: &[f64], fraction: f64, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let mut rows: Vec<usize> = weights
        .iter()
        .enumerate()
        .filter_map(|(row, &weight)| (weight > 0.0).then_some(row))
        .collect();
    if fraction < 1.0 {
        shuffle(rng, &mut rows);
        let count = ceil_fraction_count(rows.len(), fraction);
        rows.truncate(count.max(1));
    }
    rows
}

pub(crate) fn average_importances<'a>(
    importances: impl Iterator<Item = &'a [f64]>,
    features: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; features];
    let mut count = 0usize;
    for values in importances {
        for (output, value) in output.iter_mut().zip(values) {
            *output += value;
        }
        count += 1;
    }
    if count > 0 {
        for value in &mut output {
            *value /= count_as_f64(count);
        }
        let total: f64 = output.iter().sum();
        if total > 0.0 {
            for value in &mut output {
                *value /= total;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use oldwood::DenseMatrix;

    fn matrix(rows: usize, columns: usize, values: &[f64]) -> DenseMatrix {
        DenseMatrix::from_row_major(rows, columns, values.to_vec()).expect("valid fixture")
    }

    #[test]
    fn constant_regression_is_the_closed_form_weighted_mean() {
        let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let target = [2.0, 2.0, 2.0, 2.0];
        let model = GradientBoostingRegressor::new(BoostingOptions {
            iterations: 7,
            ..BoostingOptions::default()
        })
        .fit(&x, &target, None)
        .expect("fit");
        assert_eq!(model.predict(&x).expect("predict"), target);
    }

    #[test]
    fn weighted_mean_keeps_a_representable_extreme_mean() {
        let mean =
            weighted_mean(&[f64::MAX; 9], &[1.0; 9]).expect("the convex mean is representable");
        assert_eq!(mean.to_bits(), f64::MAX.to_bits());
    }

    #[test]
    fn weighted_mean_preserves_a_small_cancellation_residual_in_both_orders() {
        let large = 8.0e153;
        for values in [[large, 1.0, -large], [large, -large, 1.0]] {
            let mean = weighted_mean(&values, &[1.0; 3]).expect("representable mean");
            assert_eq!(mean, 1.0 / 3.0);
        }
    }

    #[test]
    fn class_log_priors_preserve_extreme_finite_weight_ratios() {
        let tiny = f64::from_bits(1);
        let priors = class_log_priors(&[0, 1], &[f64::MAX, tiny], &[0, 1])
            .expect("log-domain class weights");
        assert!(priors.iter().all(|value| value.is_finite()));
        assert!(priors[0].abs() <= f64::EPSILON);
        assert!(priors[1] < -1_000.0);
    }

    #[test]
    fn zero_weight_extreme_target_is_not_used_in_residual_arithmetic() {
        let x = matrix(2, 1, &[0.0, 1.0]);
        let fitted = GradientBoostingRegressor::new(BoostingOptions {
            iterations: 1,
            ..BoostingOptions::default()
        })
        .fit(&x, &[f64::MAX, -f64::MAX], Some(&[1.0, 0.0]))
        .expect("zero-weight target is outside the objective");
        assert!(fitted
            .predict(&x)
            .expect("predict")
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn one_stage_squared_error_matches_the_analytical_residual_step() {
        let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let target = [0.0, 0.0, 2.0, 2.0];
        let fitted = GradientBoostingRegressor::new(BoostingOptions {
            iterations: 1,
            learning_rate: 0.25,
            ..BoostingOptions::default()
        })
        .fit(&x, &target, None)
        .expect("fit");
        assert_eq!(
            fitted.predict(&x).expect("predict"),
            vec![0.75, 0.75, 1.25, 1.25]
        );
    }

    #[test]
    fn softmax_probabilities_are_normalized_and_labels_are_not_indices() {
        let x = matrix(6, 1, &[0.0, 0.1, 0.2, 0.8, 0.9, 1.0]);
        let target = [4, 4, 4, 9, 9, 9];
        let model = GradientBoostingClassifier::new(BoostingOptions {
            iterations: 20,
            learning_rate: 0.2,
            ..BoostingOptions::default()
        })
        .fit(&x, &target, None)
        .expect("fit");
        for row in model.predict_proba(&x).expect("probability") {
            let sum: f64 = row.iter().sum();
            // Measured 2.220446049250313e-16 on 2026-09-04; fourfold margin.
            assert!((sum - 1.0).abs() <= 9.0e-16);
            assert!(row.iter().all(|value| (0.0..=1.0).contains(value)));
        }
        assert_eq!(model.predict(&x).expect("predict"), target);
    }

    #[test]
    fn stage_subsampling_is_seed_reproducible() {
        let x = matrix(8, 1, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let target = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0];
        let estimator = GradientBoostingRegressor::new(BoostingOptions {
            iterations: 5,
            sample_fraction: 0.625,
            seed: 71,
            ..BoostingOptions::default()
        });
        let first = estimator.fit(&x, &target, None).expect("first");
        let second = estimator.fit(&x, &target, None).expect("second");
        assert_eq!(
            first.predict(&x).expect("first prediction"),
            second.predict(&x).expect("second prediction")
        );
    }

    #[test]
    fn stage_subsampling_cannot_hide_a_non_finite_training_feature() {
        let x = matrix(4, 1, &[0.0, f64::INFINITY, 2.0, 3.0]);
        let error = GradientBoostingRegressor::new(BoostingOptions {
            iterations: 1,
            sample_fraction: 0.25,
            seed: 11,
            ..BoostingOptions::default()
        })
        .fit(&x, &[0.0, 1.0, 2.0, 3.0], None)
        .expect_err("the complete training matrix must be validated");
        assert_eq!(
            error,
            Error::NonFinite {
                name: "feature",
                index: 1,
            }
        );
    }
}
