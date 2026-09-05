//! Discrete SAMME classification and AdaBoost.R2 regression.
//!
//! Both estimators use [`oldwood`] CART models as their only weak-learner
//! implementation. The classifier implements discrete SAMME only; it does
//! not implement or claim SAMME.R probability updates.

use amatsuki::{seed_rng, ChaCha8Rng, Rng};
use oldwood::{
    ClassificationCriterion, DecisionTreeClassifier, DecisionTreeRegressor, FittedClassifier,
    FittedRegressor, MatrixView, RegressionCriterion, TreeOptions,
};

use crate::data::{
    checked_weights, validate_finite_features, validate_predict, validate_regression_target,
    validate_training, IndexedRows,
};
use crate::gradient::{checked_add, positive_weight_classes};
use crate::numeric::CompensatedSum;
use crate::{Error, Result};

/// Runtime configuration for a discrete SAMME classifier.
#[derive(Clone, Debug, PartialEq)]
pub struct SammeOptions {
    /// Maximum number of weak learners.
    pub estimators: usize,
    /// Positive shrinkage applied to every SAMME stage weight.
    pub learning_rate: f64,
    /// Stopping policy for each `oldwood` CART weak learner.
    pub tree: TreeOptions,
}

impl Default for SammeOptions {
    fn default() -> Self {
        Self {
            estimators: 30,
            learning_rate: 1.0,
            tree: TreeOptions {
                max_depth: Some(1),
                ..TreeOptions::default()
            },
        }
    }
}

impl SammeOptions {
    fn validate(&self) -> Result<()> {
        if self.estimators == 0 {
            return Err(Error::InvalidOption {
                name: "estimators",
                requirement: "at least 1",
            });
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(Error::InvalidOption {
                name: "learning_rate",
                requirement: "finite and positive",
            });
        }
        Ok(())
    }
}

/// Discrete SAMME `AdaBoost` classifier over deterministic `oldwood` CART.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaBoostClassifier {
    /// Additive-stage and weak-tree configuration.
    pub options: SammeOptions,
    /// Classification impurity criterion used by every weak tree.
    pub criterion: ClassificationCriterion,
}

impl Default for AdaBoostClassifier {
    fn default() -> Self {
        Self {
            options: SammeOptions::default(),
            criterion: ClassificationCriterion::Gini,
        }
    }
}

impl AdaBoostClassifier {
    /// Creates a discrete SAMME classifier with explicit runtime options.
    #[must_use]
    pub fn new(options: SammeOptions, criterion: ClassificationCriterion) -> Self {
        Self { options, criterion }
    }

    /// Fits discrete SAMME while preserving arbitrary `usize` class labels.
    ///
    /// Initial sample weights are normalized, and zero-weight rows remain
    /// excluded. A weak learner must have weighted error strictly below
    /// `1 - 1 / n_classes`. An exact classifier stops the sequence early.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] for invalid data or options, an unusable first
    /// weak learner, an `oldwood` failure, or non-representable arithmetic.
    pub fn fit<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        target: &[usize],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedAdaBoostClassifier> {
        validate_training(x, target.len(), sample_weight)?;
        validate_finite_features(x)?;
        self.options.validate()?;
        self.options.tree.validate(x.ncols())?;

        let mut weights = normalized_weights(&checked_weights(x.nrows(), sample_weight))?;
        let classes = positive_weight_classes(target, &weights);
        if classes.len() < 2 {
            return Err(Error::InvalidOption {
                name: "target classes",
                requirement: "at least two classes with positive weight",
            });
        }

        let mut estimators = Vec::with_capacity(self.options.estimators);
        let mut estimator_weights = Vec::with_capacity(self.options.estimators);
        let mut estimator_errors = Vec::with_capacity(self.options.estimators);
        let mut perfect_fit = false;
        for _ in 0..self.options.estimators {
            let tree = DecisionTreeClassifier::new(self.criterion, self.options.tree.clone()).fit(
                &x,
                target,
                Some(&weights),
            )?;
            let prediction = tree.predict(&x)?;
            let error = classification_error(target, &prediction, &weights)?;
            if error == 0.0 {
                // The formal stage weight is +infinity. Keeping only this
                // exact learner is the finite representation of that limit.
                estimators.clear();
                estimator_weights.clear();
                estimator_errors.clear();
                estimators.push(tree);
                estimator_weights.push(self.options.learning_rate);
                estimator_errors.push(0.0);
                perfect_fit = true;
                break;
            }

            let chance_limit = 1.0 - 1.0 / count_as_f64(classes.len());
            if error >= chance_limit {
                if estimators.is_empty() {
                    return Err(Error::InvalidOption {
                        name: "weak learner",
                        requirement: "weighted error below the SAMME random-guess limit",
                    });
                }
                break;
            }

            let alpha = samme_alpha(error, classes.len(), self.options.learning_rate)?;
            reweight_classification(&mut weights, target, &prediction, alpha)?;
            estimators.push(tree);
            estimator_weights.push(alpha);
            estimator_errors.push(error);
        }

        Ok(FittedAdaBoostClassifier {
            estimators,
            estimator_weights,
            estimator_errors,
            classes,
            features: x.ncols(),
            perfect_fit,
        })
    }
}

/// Fitted discrete SAMME classifier.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedAdaBoostClassifier {
    estimators: Vec<FittedClassifier>,
    estimator_weights: Vec<f64>,
    estimator_errors: Vec<f64>,
    classes: Vec<usize>,
    features: usize,
    perfect_fit: bool,
}

impl FittedAdaBoostClassifier {
    /// Sorted positive-weight training labels used for deterministic voting.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Fitted weak trees in stage order.
    #[must_use]
    pub fn estimators(&self) -> &[FittedClassifier] {
        &self.estimators
    }

    /// Finite SAMME vote weights aligned with [`Self::estimators`].
    #[must_use]
    pub fn estimator_weights(&self) -> &[f64] {
        &self.estimator_weights
    }

    /// Weighted training errors aligned with [`Self::estimators`].
    #[must_use]
    pub fn estimator_errors(&self) -> &[f64] {
        &self.estimator_errors
    }

    /// Whether fitting stopped because a weak learner was exactly correct.
    #[must_use]
    pub fn stopped_on_perfect_fit(&self) -> bool {
        self.perfect_fit
    }

    /// Number of feature columns required for prediction.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.features
    }

    /// Predicts the smallest class label attaining the largest SAMME vote.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] for an incompatible or non-finite prediction
    /// matrix, an underlying tree failure, or vote accumulation overflow.
    pub fn predict<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<usize>> {
        validate_predict(x, self.features)?;
        let mut scores = vec![vec![0.0; self.classes.len()]; x.nrows()];
        for (tree, &alpha) in self.estimators.iter().zip(&self.estimator_weights) {
            for (row, prediction) in tree.predict(&x)?.into_iter().enumerate() {
                let class = self.classes.binary_search(&prediction).map_err(|_| {
                    Error::NumericalOverflow {
                        operation: "SAMME weak learner returned an unknown class",
                    }
                })?;
                scores[row][class] = checked_add(
                    scores[row][class],
                    alpha,
                    "SAMME prediction vote accumulation",
                )?;
            }
        }
        Ok(scores
            .iter()
            .map(|row| self.classes[argmax_smallest(row)])
            .collect())
    }

    /// Normalized stage-weighted impurity-decrease importance.
    #[must_use]
    pub fn feature_importances(&self) -> Vec<f64> {
        weighted_importances(
            self.estimators
                .iter()
                .map(FittedClassifier::feature_importances),
            &self.estimator_weights,
            self.features,
        )
    }
}

/// Runtime configuration for AdaBoost.R2 regression.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaBoostR2Options {
    /// Maximum number of weak learners.
    pub estimators: usize,
    /// Positive shrinkage applied to stage weights and weight updates.
    pub learning_rate: f64,
    /// Reproducible `ChaCha8` seed for weighted bootstrap samples.
    pub seed: u64,
    /// Stopping policy for each `oldwood` CART weak learner.
    pub tree: TreeOptions,
}

impl Default for AdaBoostR2Options {
    fn default() -> Self {
        Self {
            estimators: 30,
            learning_rate: 1.0,
            seed: 0,
            tree: TreeOptions {
                max_depth: Some(3),
                ..TreeOptions::default()
            },
        }
    }
}

impl AdaBoostR2Options {
    fn validate(&self) -> Result<()> {
        if self.estimators == 0 {
            return Err(Error::InvalidOption {
                name: "estimators",
                requirement: "at least 1",
            });
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(Error::InvalidOption {
                name: "learning_rate",
                requirement: "finite and positive",
            });
        }
        Ok(())
    }
}

/// AdaBoost.R2 regressor using weighted-bootstrap `oldwood` CART learners.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AdaBoostRegressor {
    /// Additive-stage, bootstrap, and weak-tree configuration.
    pub options: AdaBoostR2Options,
}

impl AdaBoostRegressor {
    /// Creates an AdaBoost.R2 regressor with explicit runtime options.
    #[must_use]
    pub fn new(options: AdaBoostR2Options) -> Self {
        Self { options }
    }

    /// Fits AdaBoost.R2 with normalized absolute loss.
    ///
    /// Supplied sample weights define the initial bootstrap distribution.
    /// Zero-weight rows remain excluded. Each bootstrap contains exactly the
    /// training row count and is deterministic for a fixed seed.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] for invalid data or options, an unusable first
    /// weak learner, an `oldwood` failure, or non-representable arithmetic.
    pub fn fit<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        target: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedAdaBoostRegressor> {
        validate_training(x, target.len(), sample_weight)?;
        validate_regression_target(target)?;
        validate_finite_features(x)?;
        self.options.validate()?;
        self.options.tree.validate(x.ncols())?;

        let mut weights = normalized_weights(&checked_weights(x.nrows(), sample_weight))?;
        let mut rng = seed_rng(self.options.seed);
        let mut estimators = Vec::with_capacity(self.options.estimators);
        let mut estimator_weights = Vec::with_capacity(self.options.estimators);
        let mut estimator_errors = Vec::with_capacity(self.options.estimators);
        let mut perfect_fit = false;

        for _ in 0..self.options.estimators {
            let rows = weighted_bootstrap(&weights, x.nrows(), &mut rng);
            let sampled_x = IndexedRows::new(x, &rows);
            let sampled_target: Vec<f64> = rows.iter().map(|&row| target[row]).collect();
            let tree = DecisionTreeRegressor::new(
                RegressionCriterion::SquaredError,
                self.options.tree.clone(),
            )
            .fit(&sampled_x, &sampled_target, None)?;
            let prediction = tree.predict(&x)?;
            let losses = normalized_absolute_losses(target, &prediction, &weights)?;
            let error = weighted_loss(&losses, &weights)?;

            if error == 0.0 {
                // The formal weighted-median weight is +infinity. Keeping
                // only this exact learner is the finite limiting model.
                estimators.clear();
                estimator_weights.clear();
                estimator_errors.clear();
                estimators.push(tree);
                estimator_weights.push(self.options.learning_rate);
                estimator_errors.push(0.0);
                perfect_fit = true;
                break;
            }
            if error >= 0.5 {
                if estimators.is_empty() {
                    return Err(Error::InvalidOption {
                        name: "weak learner",
                        requirement: "weighted normalized absolute loss below 1/2",
                    });
                }
                break;
            }

            let beta = error / (1.0 - error);
            let alpha = self.options.learning_rate * (-beta.ln());
            if !alpha.is_finite() || alpha <= 0.0 {
                return Err(Error::NumericalOverflow {
                    operation: "AdaBoost.R2 stage weight",
                });
            }
            reweight_regression(&mut weights, &losses, beta, self.options.learning_rate)?;
            estimators.push(tree);
            estimator_weights.push(alpha);
            estimator_errors.push(error);
        }

        Ok(FittedAdaBoostRegressor {
            estimators,
            estimator_weights,
            estimator_errors,
            features: x.ncols(),
            perfect_fit,
        })
    }
}

/// Fitted AdaBoost.R2 regressor.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedAdaBoostRegressor {
    estimators: Vec<FittedRegressor>,
    estimator_weights: Vec<f64>,
    estimator_errors: Vec<f64>,
    features: usize,
    perfect_fit: bool,
}

impl FittedAdaBoostRegressor {
    /// Fitted weak trees in stage order.
    #[must_use]
    pub fn estimators(&self) -> &[FittedRegressor] {
        &self.estimators
    }

    /// Positive weighted-median weights aligned with [`Self::estimators`].
    #[must_use]
    pub fn estimator_weights(&self) -> &[f64] {
        &self.estimator_weights
    }

    /// Weighted normalized absolute losses aligned with [`Self::estimators`].
    #[must_use]
    pub fn estimator_errors(&self) -> &[f64] {
        &self.estimator_errors
    }

    /// Whether fitting stopped because a weak learner was exactly correct.
    #[must_use]
    pub fn stopped_on_perfect_fit(&self) -> bool {
        self.perfect_fit
    }

    /// Number of feature columns required for prediction.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.features
    }

    /// Predicts the weighted median of weak-tree predictions for every row.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] for an incompatible or non-finite prediction
    /// matrix or an underlying tree failure.
    pub fn predict<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<f64>> {
        validate_predict(x, self.features)?;
        let predictions: Vec<Vec<f64>> = self
            .estimators
            .iter()
            .map(|tree| tree.predict(&x).map_err(Error::from))
            .collect::<Result<_>>()?;
        let mut output = Vec::with_capacity(x.nrows());
        for row in 0..x.nrows() {
            let values = predictions.iter().map(|prediction| prediction[row]);
            output.push(weighted_median(values, &self.estimator_weights)?);
        }
        Ok(output)
    }

    /// Normalized stage-weighted impurity-decrease importance.
    #[must_use]
    pub fn feature_importances(&self) -> Vec<f64> {
        weighted_importances(
            self.estimators
                .iter()
                .map(FittedRegressor::feature_importances),
            &self.estimator_weights,
            self.features,
        )
    }
}

fn normalized_weights(weights: &[f64]) -> Result<Vec<f64>> {
    let maximum = weights
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or(Error::NoPositiveWeight)?;
    if maximum <= 0.0 {
        return Err(Error::NoPositiveWeight);
    }
    let mut normalized: Vec<f64> = weights.iter().map(|weight| weight / maximum).collect();
    if weights
        .iter()
        .zip(&normalized)
        .any(|(&original, &scaled)| original > 0.0 && scaled == 0.0)
    {
        return Err(Error::NumericalOverflow {
            operation: "sample-weight dynamic range",
        });
    }
    let mut total = CompensatedSum::default();
    for &weight in &normalized {
        total.add(weight, "sample-weight normalization")?;
    }
    let total = total.value("sample-weight normalization")?;
    for (scaled, &original) in normalized.iter_mut().zip(weights) {
        *scaled /= total;
        if original > 0.0 && *scaled == 0.0 {
            return Err(Error::NumericalOverflow {
                operation: "sample-weight dynamic range",
            });
        }
    }
    Ok(normalized)
}

fn classification_error(target: &[usize], prediction: &[usize], weights: &[f64]) -> Result<f64> {
    let mut error = CompensatedSum::default();
    let mut total = CompensatedSum::default();
    for ((&actual, &predicted), &weight) in target.iter().zip(prediction).zip(weights) {
        total.add(weight, "SAMME total weight")?;
        if actual != predicted {
            error.add(weight, "SAMME weighted error")?;
        }
    }
    error.ratio(total, "SAMME weighted error ratio")
}

fn samme_alpha(error: f64, classes: usize, learning_rate: f64) -> Result<f64> {
    let multiclass_term = count_as_f64(classes - 1).ln();
    let alpha = learning_rate * ((-error).ln_1p() - error.ln() + multiclass_term);
    if alpha.is_finite() && alpha > 0.0 {
        Ok(alpha)
    } else {
        Err(Error::NumericalOverflow {
            operation: "SAMME stage weight",
        })
    }
}

fn reweight_classification(
    weights: &mut [f64],
    target: &[usize],
    prediction: &[usize],
    alpha: f64,
) -> Result<()> {
    let increments: Vec<f64> = target
        .iter()
        .zip(prediction)
        .map(|(&actual, &predicted)| if actual == predicted { 0.0 } else { alpha })
        .collect();
    apply_log_weight_update(weights, &increments)
}

fn normalized_absolute_losses(
    target: &[f64],
    prediction: &[f64],
    weights: &[f64],
) -> Result<Vec<f64>> {
    let value_scale = target
        .iter()
        .zip(prediction)
        .zip(weights)
        .filter_map(|((&actual, &predicted), &weight)| {
            (weight > 0.0).then_some(actual.abs().max(predicted.abs()))
        })
        .fold(0.0_f64, f64::max);
    if value_scale == 0.0 {
        return Ok(vec![0.0; target.len()]);
    }
    let mut errors = Vec::with_capacity(target.len());
    let mut maximum = 0.0_f64;
    for ((&actual, &predicted), &weight) in target.iter().zip(prediction).zip(weights) {
        if weight <= 0.0 {
            errors.push(0.0);
            continue;
        }
        let error = (actual / value_scale - predicted / value_scale).abs();
        if !error.is_finite() {
            return Err(Error::NumericalOverflow {
                operation: "AdaBoost.R2 absolute error",
            });
        }
        maximum = maximum.max(error);
        errors.push(error);
    }
    if maximum == 0.0 {
        return Ok(vec![0.0; target.len()]);
    }
    Ok(errors
        .into_iter()
        .zip(weights)
        .map(|(error, &weight)| if weight > 0.0 { error / maximum } else { 0.0 })
        .collect())
}

fn weighted_loss(losses: &[f64], weights: &[f64]) -> Result<f64> {
    crate::gradient::weighted_mean(losses, weights)
}

fn reweight_regression(
    weights: &mut [f64],
    losses: &[f64],
    beta: f64,
    learning_rate: f64,
) -> Result<()> {
    let log_beta = beta.ln();
    let increments: Vec<f64> = losses
        .iter()
        .map(|loss| learning_rate * (1.0 - loss) * log_beta)
        .collect();
    apply_log_weight_update(weights, &increments)
}

fn apply_log_weight_update(weights: &mut [f64], increments: &[f64]) -> Result<()> {
    let log_weights: Vec<f64> = weights
        .iter()
        .zip(increments)
        .map(|(&weight, &increment)| {
            if weight > 0.0 {
                weight.ln() + increment
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let maximum = log_weights
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or(Error::NoPositiveWeight)?;
    if !maximum.is_finite() {
        return Err(Error::NoPositiveWeight);
    }
    for (weight, &log_weight) in weights.iter_mut().zip(&log_weights) {
        let was_positive = *weight > 0.0;
        *weight = (log_weight - maximum).exp();
        if was_positive && *weight == 0.0 {
            return Err(Error::NumericalOverflow {
                operation: "updated sample-weight dynamic range",
            });
        }
    }
    let mut total = CompensatedSum::default();
    for &weight in weights.iter() {
        total.add(weight, "updated sample-weight sum")?;
    }
    let total = total.value("updated sample-weight sum")?;
    for weight in weights {
        *weight /= total;
    }
    Ok(())
}

fn weighted_bootstrap(weights: &[f64], count: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut total = 0.0;
    let mut last_positive = 0;
    for (row, &weight) in weights.iter().enumerate() {
        total += weight;
        cumulative.push(total);
        if weight > 0.0 {
            last_positive = row;
        }
    }
    (0..count)
        .map(|_| {
            let draw = rng.next_f64() * total;
            cumulative
                .partition_point(|&boundary| boundary <= draw)
                .min(last_positive.max(cumulative.len() - 1))
        })
        .map(|row| {
            if weights[row] > 0.0 {
                row
            } else {
                last_positive
            }
        })
        .collect()
}

fn weighted_median(values: impl Iterator<Item = f64>, weights: &[f64]) -> Result<f64> {
    let mut pairs: Vec<(f64, f64)> = values.zip(weights.iter().copied()).collect();
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let maximum =
        weights
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .ok_or(Error::NumericalOverflow {
                operation: "empty AdaBoost.R2 weighted median",
            })?;
    if !maximum.is_finite() || maximum <= 0.0 {
        return Err(Error::NumericalOverflow {
            operation: "AdaBoost.R2 weighted-median weights",
        });
    }
    let mut total = CompensatedSum::default();
    for &(_, weight) in &pairs {
        total.add(weight / maximum, "AdaBoost.R2 weighted-median total")?;
    }
    let mut threshold = total;
    threshold.scale(0.5, "AdaBoost.R2 weighted-median threshold")?;
    let mut cumulative = CompensatedSum::default();
    for (value, weight) in pairs {
        cumulative.add(weight / maximum, "AdaBoost.R2 weighted-median prefix")?;
        if cumulative.cmp(threshold).is_ge() {
            return Ok(value);
        }
    }
    Err(Error::NumericalOverflow {
        operation: "AdaBoost.R2 weighted median",
    })
}

fn argmax_smallest(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map_or(0, |(index, _)| index)
}

fn weighted_importances<'a>(
    importances: impl Iterator<Item = &'a [f64]>,
    stage_weights: &[f64],
    features: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; features];
    let Some(maximum) = stage_weights.iter().copied().max_by(f64::total_cmp) else {
        return output;
    };
    let mut total_weight = 0.0;
    for (values, &stage_weight) in importances.zip(stage_weights) {
        let scaled_weight = stage_weight / maximum;
        total_weight += scaled_weight;
        for (output, &value) in output.iter_mut().zip(values) {
            *output += scaled_weight * value;
        }
    }
    if total_weight > 0.0 {
        for value in &mut output {
            *value /= total_weight;
        }
    }
    let total: f64 = output.iter().sum();
    if total > 0.0 {
        for value in &mut output {
            *value /= total;
        }
    }
    output
}

#[allow(clippy::cast_precision_loss)]
fn count_as_f64(value: usize) -> f64 {
    // Every caller passes the length of an allocated collection. Counts above
    // f64's exact-integer range cannot be represented by this process anyway.
    value as f64
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
    fn binary_quarter_error_has_closed_form_alpha_and_weights() {
        let x = matrix(4, 2, &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        let target = [11, 11, 11, 29];
        let fitted = AdaBoostClassifier::new(
            SammeOptions {
                estimators: 1,
                ..SammeOptions::default()
            },
            ClassificationCriterion::Gini,
        )
        .fit(&x, &target, Some(&[2.0, 2.0, 2.0, 2.0]))
        .expect("fit");
        let prediction = fitted.estimators()[0].predict(&x).expect("predict");
        let mut weights = vec![0.25; 4];
        let error = classification_error(&target, &prediction, &weights).expect("error");
        let alpha = samme_alpha(error, 2, 1.0).expect("alpha");
        reweight_classification(&mut weights, &target, &prediction, alpha).expect("update");

        assert!((error - 0.25).abs() <= f64::EPSILON);
        assert!((alpha - 3.0_f64.ln()).abs() <= 2.0 * f64::EPSILON);
        let expected = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5];
        for (&actual, expected) in weights.iter().zip(expected) {
            assert!((actual - expected).abs() <= 2.0 * f64::EPSILON);
        }
    }

    #[test]
    fn perfect_samme_learner_stops_after_one_tree() {
        let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let target = [7, 7, 41, 41];
        let fitted = AdaBoostClassifier::new(
            SammeOptions {
                estimators: 9,
                ..SammeOptions::default()
            },
            ClassificationCriterion::Gini,
        )
        .fit(&x, &target, None)
        .expect("fit");

        assert_eq!(fitted.estimators().len(), 1);
        assert!(fitted.stopped_on_perfect_fit());
        assert_eq!(fitted.predict(&x).expect("predict"), target);
        assert!(fitted.estimator_weights()[0].is_finite());
    }

    #[test]
    fn perfect_r2_learner_stops_after_one_tree() {
        let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let target = [2.5; 4];
        let fitted = AdaBoostRegressor::new(AdaBoostR2Options {
            estimators: 9,
            seed: 73,
            ..AdaBoostR2Options::default()
        })
        .fit(&x, &target, None)
        .expect("fit");

        assert_eq!(fitted.estimators().len(), 1);
        assert!(fitted.stopped_on_perfect_fit());
        assert_eq!(fitted.predict(&x).expect("predict"), target);
        assert!(fitted.estimator_weights()[0].is_finite());
    }

    #[test]
    fn r2_seed_replays_bootstrap_sequence() {
        let x = matrix(10, 1, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let target = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let model = AdaBoostRegressor::new(AdaBoostR2Options {
            estimators: 4,
            seed: 19,
            tree: TreeOptions {
                max_depth: Some(2),
                ..TreeOptions::default()
            },
            ..AdaBoostR2Options::default()
        });
        let first = model.fit(&x, &target, None).expect("first fit");
        let second = model.fit(&x, &target, None).expect("second fit");
        assert_eq!(first, second);
    }

    #[test]
    fn invalid_configuration_is_typed() {
        let x = matrix(2, 1, &[0.0, 1.0]);
        let error = AdaBoostClassifier::new(
            SammeOptions {
                estimators: 0,
                ..SammeOptions::default()
            },
            ClassificationCriterion::Gini,
        )
        .fit(&x, &[0, 1], None)
        .expect_err("zero stages are invalid");
        assert!(matches!(
            error,
            Error::InvalidOption {
                name: "estimators",
                ..
            }
        ));
    }

    #[test]
    fn r2_loss_does_not_evaluate_zero_weight_overflowing_rows() {
        let losses = normalized_absolute_losses(&[f64::MAX, 0.0], &[-f64::MAX, 0.0], &[0.0, 1.0])
            .expect("zero-weight row is outside the objective");
        assert_eq!(losses, vec![0.0, 0.0]);

        let scaled =
            normalized_absolute_losses(&[f64::MAX, -f64::MAX], &[-f64::MAX, f64::MAX], &[1.0, 1.0])
                .expect("normalization avoids overflowing representable loss ratios");
        assert_eq!(scaled, vec![1.0, 1.0]);
    }

    #[test]
    fn weighted_median_keeps_a_sub_ulp_tail_mass() {
        let tail = 2.0f64.powi(-53);
        let median = weighted_median([0.0, 1.0, 2.0].into_iter(), &[1.0, 1.0, tail])
            .expect("weighted median");
        assert_eq!(median, 1.0);
    }

    #[test]
    fn samme_accepts_a_stump_that_is_sub_ulp_better_than_chance() {
        let x = matrix(3, 1, &[0.0; 3]);
        let tail = 2.0f64.powi(-53);
        let fitted = AdaBoostClassifier::new(
            SammeOptions {
                estimators: 1,
                tree: TreeOptions {
                    max_depth: Some(0),
                    ..TreeOptions::default()
                },
                ..SammeOptions::default()
            },
            ClassificationCriterion::Gini,
        )
        .fit(&x, &[0, 1, 1], Some(&[1.0, 1.0, tail]))
        .expect("exact weight expansion is better than chance");
        assert_eq!(fitted.estimators().len(), 1);
        assert!(fitted.estimator_errors()[0] < 0.5);
        assert_eq!(fitted.predict(&x).expect("predict"), vec![1; 3]);
    }

    #[test]
    fn finite_positive_weight_cannot_silently_underflow_to_zero() {
        let x = matrix(2, 1, &[0.0, 1.0]);
        let error = AdaBoostClassifier::default()
            .fit(&x, &[0, 1], Some(&[f64::MAX, f64::from_bits(1)]))
            .expect_err("unsupported dynamic range must be explicit");
        assert!(matches!(error, Error::NumericalOverflow { .. }));
    }
}
