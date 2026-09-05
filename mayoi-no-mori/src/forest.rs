//! Random forests and extremely randomized trees.

use amatsuki::{seed_rng, Rng};
use oldwood::{
    ClassificationCriterion, DecisionTreeClassifier, DecisionTreeRegressor, FittedClassifier,
    FittedRegressor, MatrixView, RegressionCriterion,
};

use crate::data::{
    checked_weights, validate_finite_features, validate_predict, validate_regression_target,
    validate_training, IndexedRows,
};
use crate::gradient::{average_importances, checked_add, positive_weight_classes, weighted_mean};
use crate::numeric::ScaledSum;
use crate::options::count_as_f64;
use crate::random::{below, sample_rows};
use crate::strategy::{RandomSplitStrategy, ThresholdPolicy};
use crate::{Error, ForestOptions, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ForestFlavor {
    RandomForest,
    ExtraTrees,
}

impl ForestFlavor {
    fn threshold_policy(self) -> ThresholdPolicy {
        match self {
            Self::RandomForest => ThresholdPolicy::Exhaustive,
            Self::ExtraTrees => ThresholdPolicy::OneUniform,
        }
    }
}

/// Bootstrap-aggregated CART classifier with node-local feature sampling.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomForestClassifier {
    /// Resampling and CART options.
    pub options: ForestOptions,
    /// Classification impurity criterion evaluated by `oldwood`.
    pub criterion: ClassificationCriterion,
}

impl Default for RandomForestClassifier {
    fn default() -> Self {
        Self {
            options: ForestOptions::default(),
            criterion: ClassificationCriterion::Gini,
        }
    }
}

impl RandomForestClassifier {
    /// Creates a random forest with explicit options and criterion.
    #[must_use]
    pub fn new(options: ForestOptions, criterion: ClassificationCriterion) -> Self {
        Self { options, criterion }
    }

    /// Fits the forest, preserving arbitrary `usize` class labels.
    ///
    /// # Errors
    ///
    /// Returns an error when matrix dimensions, weights, options, or classes
    /// are invalid, when the CART kernel rejects a tree, or when ensemble
    /// probability accumulation is not representable.
    pub fn fit<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        target: &[usize],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedForestClassifier> {
        fit_classifier(
            x,
            target,
            sample_weight,
            &self.options,
            self.criterion,
            ForestFlavor::RandomForest,
        )
    }
}

/// CART classifier ensemble with random thresholds and no bootstrap by default.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtraTreesClassifier {
    /// Resampling and CART options.
    pub options: ForestOptions,
    /// Classification impurity criterion evaluated by `oldwood`.
    pub criterion: ClassificationCriterion,
}

impl Default for ExtraTreesClassifier {
    fn default() -> Self {
        let options = ForestOptions {
            bootstrap: false,
            ..ForestOptions::default()
        };
        Self {
            options,
            criterion: ClassificationCriterion::Gini,
        }
    }
}

impl ExtraTreesClassifier {
    /// Creates an extremely randomized classifier.
    #[must_use]
    pub fn new(options: ForestOptions, criterion: ClassificationCriterion) -> Self {
        Self { options, criterion }
    }

    /// Fits randomized threshold proposals through the shared CART evaluator.
    ///
    /// # Errors
    ///
    /// Returns an error when matrix dimensions, weights, options, or classes
    /// are invalid, when the CART kernel rejects a tree, or when ensemble
    /// probability accumulation is not representable.
    pub fn fit<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        target: &[usize],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedForestClassifier> {
        fit_classifier(
            x,
            target,
            sample_weight,
            &self.options,
            self.criterion,
            ForestFlavor::ExtraTrees,
        )
    }
}

/// Fitted random-forest or `ExtraTrees` classifier.
#[derive(Clone, Debug)]
pub struct FittedForestClassifier {
    trees: Vec<FittedClassifier>,
    classes: Vec<usize>,
    features: usize,
    oob_probabilities: Option<Vec<Option<Vec<f64>>>>,
    oob_score: Option<f64>,
}

impl FittedForestClassifier {
    /// Fitted CART members in generation order.
    #[must_use]
    pub fn trees(&self) -> &[FittedClassifier] {
        &self.trees
    }

    /// Sorted target labels defining probability-column order.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Predicts averaged class probabilities.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature count differs from training, the CART
    /// kernel rejects prediction, or probability accumulation overflows.
    pub fn predict_proba<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<Vec<f64>>> {
        validate_predict(x, self.features)?;
        let mut sums = vec![vec![0.0; self.classes.len()]; x.nrows()];
        for tree in &self.trees {
            add_tree_probabilities(tree, x, &self.classes, &mut sums)?;
        }
        let denominator = count_as_f64(self.trees.len());
        for row in &mut sums {
            for value in row {
                *value /= denominator;
            }
        }
        Ok(sums)
    }

    /// Predicts the smallest target label attaining the largest mean probability.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::predict_proba`].
    pub fn predict<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<usize>> {
        Ok(self
            .predict_proba(x)?
            .iter()
            .map(|row| self.classes[argmax(row)])
            .collect())
    }

    /// Per-row OOB probabilities, or `None` when OOB computation was disabled.
    #[must_use]
    pub fn oob_probabilities(&self) -> Option<&[Option<Vec<f64>>]> {
        self.oob_probabilities.as_deref()
    }

    /// Weighted OOB accuracy over rows receiving at least one OOB vote.
    #[must_use]
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score
    }

    /// Normalized mean impurity-decrease importance across the forest.
    #[must_use]
    pub fn feature_importances(&self) -> Vec<f64> {
        average_importances(
            self.trees.iter().map(FittedClassifier::feature_importances),
            self.features,
        )
    }
}

/// Bootstrap-aggregated CART regressor with node-local feature sampling.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomForestRegressor {
    /// Resampling and CART options.
    pub options: ForestOptions,
}

impl Default for RandomForestRegressor {
    fn default() -> Self {
        Self {
            options: ForestOptions {
                feature_sampling: crate::FeatureSampling::Fraction(1.0),
                ..ForestOptions::default()
            },
        }
    }
}

impl RandomForestRegressor {
    /// Creates a random-forest regressor with explicit options.
    #[must_use]
    pub fn new(options: ForestOptions) -> Self {
        Self { options }
    }

    /// Fits the forest using squared-error CART members.
    ///
    /// # Errors
    ///
    /// Returns an error when matrix dimensions, targets, weights, or options
    /// are invalid, when the CART kernel rejects a tree, or when an ensemble
    /// calculation is not representable.
    pub fn fit<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        target: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedForestRegressor> {
        fit_regressor(
            x,
            target,
            sample_weight,
            &self.options,
            ForestFlavor::RandomForest,
        )
    }
}

/// CART regression ensemble with random thresholds and no bootstrap by default.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtraTreesRegressor {
    /// Resampling and CART options.
    pub options: ForestOptions,
}

impl Default for ExtraTreesRegressor {
    fn default() -> Self {
        Self {
            options: ForestOptions {
                bootstrap: false,
                feature_sampling: crate::FeatureSampling::All,
                ..ForestOptions::default()
            },
        }
    }
}

impl ExtraTreesRegressor {
    /// Creates an extremely randomized regressor.
    #[must_use]
    pub fn new(options: ForestOptions) -> Self {
        Self { options }
    }

    /// Fits randomized threshold proposals through the shared CART evaluator.
    ///
    /// # Errors
    ///
    /// Returns an error when matrix dimensions, targets, weights, or options
    /// are invalid, when the CART kernel rejects a tree, or when an ensemble
    /// calculation is not representable.
    pub fn fit<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        target: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedForestRegressor> {
        fit_regressor(
            x,
            target,
            sample_weight,
            &self.options,
            ForestFlavor::ExtraTrees,
        )
    }
}

/// Fitted random-forest or `ExtraTrees` regressor.
#[derive(Clone, Debug)]
pub struct FittedForestRegressor {
    trees: Vec<FittedRegressor>,
    features: usize,
    oob_predictions: Option<Vec<Option<f64>>>,
    oob_score: Option<f64>,
}

impl FittedForestRegressor {
    /// Fitted CART members in generation order.
    #[must_use]
    pub fn trees(&self) -> &[FittedRegressor] {
        &self.trees
    }

    /// Predicts the arithmetic mean of member predictions.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature count differs from training, the CART
    /// kernel rejects prediction, or prediction accumulation overflows.
    pub fn predict<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<f64>> {
        validate_predict(x, self.features)?;
        let mut sums = vec![ScaledSum::default(); x.nrows()];
        for tree in &self.trees {
            for (sum, value) in sums.iter_mut().zip(tree.predict(&x)?) {
                sum.add(value, "forest prediction average")?;
            }
        }
        sums.into_iter()
            .map(|sum| sum.mean(self.trees.len(), "forest prediction average"))
            .collect()
    }

    /// Per-row OOB prediction, or `None` when OOB computation was disabled.
    #[must_use]
    pub fn oob_predictions(&self) -> Option<&[Option<f64>]> {
        self.oob_predictions.as_deref()
    }

    /// Weighted OOB coefficient of determination over covered rows.
    #[must_use]
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score
    }

    /// Normalized mean impurity-decrease importance across the forest.
    #[must_use]
    pub fn feature_importances(&self) -> Vec<f64> {
        average_importances(
            self.trees.iter().map(FittedRegressor::feature_importances),
            self.features,
        )
    }
}

fn fit_classifier<M: MatrixView + ?Sized>(
    x: &M,
    target: &[usize],
    sample_weight: Option<&[f64]>,
    options: &ForestOptions,
    criterion: ClassificationCriterion,
    flavor: ForestFlavor,
) -> Result<FittedForestClassifier> {
    validate_training(x, target.len(), sample_weight)?;
    validate_finite_features(x)?;
    let (sample_count, feature_count) = options.validate(x.nrows(), x.ncols())?;
    let weights = checked_weights(x.nrows(), sample_weight);
    let positive_rows = positive_rows(&weights);
    let classes = positive_weight_classes(target, &weights);
    if classes.is_empty() {
        return Err(Error::NoPositiveWeight);
    }
    let mut trees = Vec::with_capacity(options.trees);
    let mut oob_sums = options
        .out_of_bag
        .then(|| vec![vec![0.0; classes.len()]; x.nrows()]);
    let mut oob_counts = options.out_of_bag.then(|| vec![0usize; x.nrows()]);
    let mut root_rng = seed_rng(options.seed);
    for _ in 0..options.trees {
        let rows = sample_rows_with_positive_weight(
            &mut root_rng,
            x.nrows(),
            sample_count,
            options.bootstrap,
            &weights,
            &positive_rows,
        );
        let sampled_x = IndexedRows::new(x, &rows);
        let sampled_target: Vec<usize> = rows.iter().map(|&row| target[row]).collect();
        let tree_weights: Vec<f64> = rows.iter().map(|&row| weights[row]).collect();
        let strategy_seed = root_rng.next_u64();
        let mut strategy =
            RandomSplitStrategy::new(strategy_seed, feature_count, flavor.threshold_policy());
        let tree = DecisionTreeClassifier::new(criterion, options.tree.clone()).fit_with_strategy(
            &sampled_x,
            &sampled_target,
            Some(&tree_weights),
            &mut strategy,
        )?;
        if let (Some(sums), Some(counts)) = (&mut oob_sums, &mut oob_counts) {
            let in_bag = membership(x.nrows(), &rows);
            add_oob_classification(&tree, x, &classes, &in_bag, sums, counts)?;
        }
        trees.push(tree);
    }
    let (oob_probabilities, oob_score) = match (oob_sums, oob_counts) {
        (Some(sums), Some(counts)) => {
            let probabilities: Vec<Option<Vec<f64>>> = sums
                .into_iter()
                .zip(&counts)
                .map(|(mut row, &count)| {
                    (count > 0).then(|| {
                        for value in &mut row {
                            *value /= count_as_f64(count);
                        }
                        row
                    })
                })
                .collect();
            let score = oob_accuracy(&probabilities, target, &classes, &weights)?;
            (Some(probabilities), score)
        }
        _ => (None, None),
    };
    Ok(FittedForestClassifier {
        trees,
        classes,
        features: x.ncols(),
        oob_probabilities,
        oob_score,
    })
}

fn fit_regressor<M: MatrixView + ?Sized>(
    x: &M,
    target: &[f64],
    sample_weight: Option<&[f64]>,
    options: &ForestOptions,
    flavor: ForestFlavor,
) -> Result<FittedForestRegressor> {
    validate_training(x, target.len(), sample_weight)?;
    validate_finite_features(x)?;
    validate_regression_target(target)?;
    let (sample_count, feature_count) = options.validate(x.nrows(), x.ncols())?;
    let weights = checked_weights(x.nrows(), sample_weight);
    let positive_rows = positive_rows(&weights);
    let mut trees = Vec::with_capacity(options.trees);
    let mut oob_sums = options
        .out_of_bag
        .then(|| vec![ScaledSum::default(); x.nrows()]);
    let mut oob_counts = options.out_of_bag.then(|| vec![0usize; x.nrows()]);
    let mut root_rng = seed_rng(options.seed);
    for _ in 0..options.trees {
        let rows = sample_rows_with_positive_weight(
            &mut root_rng,
            x.nrows(),
            sample_count,
            options.bootstrap,
            &weights,
            &positive_rows,
        );
        let sampled_x = IndexedRows::new(x, &rows);
        let sampled_target: Vec<f64> = rows.iter().map(|&row| target[row]).collect();
        let tree_weights: Vec<f64> = rows.iter().map(|&row| weights[row]).collect();
        let strategy_seed = root_rng.next_u64();
        let mut strategy =
            RandomSplitStrategy::new(strategy_seed, feature_count, flavor.threshold_policy());
        let tree =
            DecisionTreeRegressor::new(RegressionCriterion::SquaredError, options.tree.clone())
                .fit_with_strategy(
                    &sampled_x,
                    &sampled_target,
                    Some(&tree_weights),
                    &mut strategy,
                )?;
        if let (Some(sums), Some(counts)) = (&mut oob_sums, &mut oob_counts) {
            let in_bag = membership(x.nrows(), &rows);
            let prediction = tree.predict(&x)?;
            for row in 0..x.nrows() {
                if !in_bag[row] {
                    counts[row] += 1;
                    sums[row].add(prediction[row], "OOB prediction average")?;
                }
            }
        }
        trees.push(tree);
    }
    let (oob_predictions, oob_score) = match (oob_sums, oob_counts) {
        (Some(sums), Some(counts)) => {
            let predictions: Vec<Option<f64>> = sums
                .into_iter()
                .zip(&counts)
                .map(|(sum, &count)| {
                    if count == 0 {
                        Ok(None)
                    } else {
                        sum.mean(count, "OOB prediction average").map(Some)
                    }
                })
                .collect::<Result<_>>()?;
            let score = oob_r2(&predictions, target, &weights)?;
            (Some(predictions), score)
        }
        _ => (None, None),
    };
    Ok(FittedForestRegressor {
        trees,
        features: x.ncols(),
        oob_predictions,
        oob_score,
    })
}

fn positive_rows(weights: &[f64]) -> Vec<usize> {
    weights
        .iter()
        .enumerate()
        .filter_map(|(row, &weight)| (weight > 0.0).then_some(row))
        .collect()
}

fn sample_rows_with_positive_weight<R: amatsuki::Rng + ?Sized>(
    rng: &mut R,
    rows: usize,
    count: usize,
    replacement: bool,
    weights: &[f64],
    positive_rows: &[usize],
) -> Vec<usize> {
    let mut sampled = sample_rows(rng, rows, count, replacement);
    if sampled.iter().all(|&row| weights[row] <= 0.0) {
        sampled[0] = positive_rows[below(rng, positive_rows.len())];
    }
    sampled
}

fn add_tree_probabilities<M: MatrixView + ?Sized>(
    tree: &FittedClassifier,
    x: &M,
    classes: &[usize],
    sums: &mut [Vec<f64>],
) -> Result<()> {
    let probabilities = tree.predict_proba(&x)?;
    for (row, sum_row) in sums.iter_mut().enumerate().take(x.nrows()) {
        for (local, &class) in tree.classes().iter().enumerate() {
            let global = classes
                .binary_search(&class)
                .expect("forest classes contain every member-tree class");
            sum_row[global] = checked_add(
                sum_row[global],
                probabilities.get(row, local),
                "forest probability sum",
            )?;
        }
    }
    Ok(())
}

fn add_oob_classification<M: MatrixView + ?Sized>(
    tree: &FittedClassifier,
    x: &M,
    classes: &[usize],
    in_bag: &[bool],
    sums: &mut [Vec<f64>],
    counts: &mut [usize],
) -> Result<()> {
    let probabilities = tree.predict_proba(&x)?;
    for (row, ((sum_row, count), &was_sampled)) in sums
        .iter_mut()
        .zip(counts.iter_mut())
        .zip(in_bag)
        .enumerate()
        .take(x.nrows())
    {
        if was_sampled {
            continue;
        }
        for (local, &class) in tree.classes().iter().enumerate() {
            let global = classes
                .binary_search(&class)
                .expect("forest classes contain every member-tree class");
            sum_row[global] = checked_add(
                sum_row[global],
                probabilities.get(row, local),
                "OOB probability sum",
            )?;
        }
        *count += 1;
    }
    Ok(())
}

fn membership(rows: usize, sampled: &[usize]) -> Vec<bool> {
    let mut output = vec![false; rows];
    for &row in sampled {
        output[row] = true;
    }
    output
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map_or(0, |(index, _)| index)
}

fn oob_accuracy(
    probabilities: &[Option<Vec<f64>>],
    target: &[usize],
    classes: &[usize],
    weights: &[f64],
) -> Result<Option<f64>> {
    let maximum_weight = probabilities
        .iter()
        .zip(weights)
        .filter_map(|(row, &weight)| row.as_ref().map(|_| weight))
        .fold(0.0_f64, f64::max);
    if maximum_weight <= 0.0 {
        return Ok(None);
    }
    let mut correct = 0.0;
    let mut total = 0.0;
    for ((row, &target), &weight) in probabilities.iter().zip(target).zip(weights) {
        if let Some(row) = row.as_ref().filter(|_| weight > 0.0) {
            let scaled_weight = weight / maximum_weight;
            total = checked_add(total, scaled_weight, "OOB accuracy denominator")?;
            if classes[argmax(row)] == target {
                correct = checked_add(correct, scaled_weight, "OOB accuracy numerator")?;
            }
        }
    }
    Ok((total > 0.0).then(|| correct / total))
}

fn oob_r2(predictions: &[Option<f64>], target: &[f64], weights: &[f64]) -> Result<Option<f64>> {
    let covered_values: Vec<f64> = predictions
        .iter()
        .zip(target)
        .zip(weights)
        .filter_map(|((prediction, &target), &weight)| {
            (prediction.is_some() && weight > 0.0).then_some(target)
        })
        .collect();
    let covered_weights: Vec<f64> = predictions
        .iter()
        .zip(weights)
        .filter_map(|(prediction, &weight)| {
            (prediction.is_some() && weight > 0.0).then_some(weight)
        })
        .collect();
    if covered_values.is_empty() {
        return Ok(None);
    }
    let mean = weighted_mean(&covered_values, &covered_weights)?;
    let value_scale = predictions
        .iter()
        .zip(target)
        .zip(weights)
        .filter_map(|((prediction, &actual), &weight)| {
            prediction
                .as_ref()
                .filter(|_| weight > 0.0)
                .map(|prediction| actual.abs().max(prediction.abs()))
        })
        .fold(mean.abs(), f64::max);
    if value_scale == 0.0 {
        return Ok(None);
    }
    let weight_scale = covered_weights.iter().copied().fold(0.0_f64, f64::max);
    let mut residual = 0.0;
    let mut total = 0.0;
    for ((prediction, &actual), &weight) in predictions.iter().zip(target).zip(weights) {
        if let Some(prediction) = prediction.as_ref().filter(|_| weight > 0.0) {
            let scaled_weight = weight / weight_scale;
            let residual_delta = actual / value_scale - prediction / value_scale;
            let total_delta = actual / value_scale - mean / value_scale;
            residual = checked_add(
                residual,
                scaled_weight * residual_delta * residual_delta,
                "OOB residual sum",
            )?;
            total = checked_add(
                total,
                scaled_weight * total_delta * total_delta,
                "OOB total sum",
            )?;
        }
    }
    if total <= 0.0 {
        return Ok(None);
    }
    let score = 1.0 - residual / total;
    if score.is_finite() {
        Ok(Some(score))
    } else {
        Err(Error::NumericalOverflow {
            operation: "OOB coefficient of determination",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oldwood::DenseMatrix;

    fn matrix(rows: usize, columns: usize, values: &[f64]) -> DenseMatrix {
        DenseMatrix::from_row_major(rows, columns, values.to_vec()).expect("valid fixture")
    }

    #[test]
    fn forest_is_reproducible_and_probability_rows_sum_to_one() {
        let x = matrix(
            8,
            2,
            &[
                0.0, 1.0, 0.1, 0.8, 0.2, 1.1, 0.3, 0.9, 1.0, 0.1, 1.1, 0.2, 0.9, 0.0, 1.2, 0.3,
            ],
        );
        let target = [2, 2, 2, 2, 7, 7, 7, 7];
        let estimator = RandomForestClassifier::new(
            ForestOptions {
                trees: 31,
                out_of_bag: true,
                seed: 91,
                ..ForestOptions::default()
            },
            ClassificationCriterion::Gini,
        );
        let first = estimator.fit(&x, &target, None).expect("first fit");
        let second = estimator.fit(&x, &target, None).expect("second fit");
        assert_eq!(
            first.predict(&x).expect("first prediction"),
            second.predict(&x).expect("second prediction")
        );
        assert_eq!(first.trees(), second.trees());
        assert_eq!(first.feature_importances(), second.feature_importances());
        assert_eq!(first.oob_probabilities(), second.oob_probabilities());
        assert_eq!(first.oob_score(), second.oob_score());
        for row in first.predict_proba(&x).expect("probabilities") {
            // Measured 2.220446049250313e-16 on 2026-09-04; fourfold margin.
            assert!((row.iter().sum::<f64>() - 1.0).abs() <= 9.0e-16);
        }
        assert!(first.oob_score().is_some());
    }

    #[test]
    fn extra_trees_classifier_defaults_to_sqrt_features_without_bootstrap() {
        let estimator = ExtraTreesClassifier::default();
        assert!(!estimator.options.bootstrap);
        assert_eq!(
            estimator.options.feature_sampling,
            crate::FeatureSampling::SquareRoot
        );
    }

    #[test]
    fn member_probability_columns_map_to_the_forest_class_order() {
        let x = matrix(4, 1, &[0.0, 0.1, 0.9, 1.0]);
        let tree = oldwood::DecisionTreeClassifier::default()
            .fit(&x, &[2, 2, 7, 7], None)
            .expect("two-class member");
        let local = tree.predict_proba(&x).expect("local probabilities");
        let mut global = vec![vec![0.0; 3]; x.nrows()];
        add_tree_probabilities(&tree, &x, &[2, 5, 7], &mut global).expect("map columns");
        for (row, global_row) in global.iter().enumerate() {
            assert_eq!(global_row[0].to_bits(), local.get(row, 0).to_bits());
            assert_eq!(global_row[1].to_bits(), 0.0_f64.to_bits());
            assert_eq!(global_row[2].to_bits(), local.get(row, 1).to_bits());
        }
    }

    #[test]
    fn extra_trees_uses_the_same_cart_node_arena() {
        let x = matrix(6, 1, &[0.0, 0.1, 0.2, 0.8, 0.9, 1.0]);
        let target = [0.0, 0.1, 0.2, 0.8, 0.9, 1.0];
        let fitted = ExtraTreesRegressor::new(ForestOptions {
            trees: 17,
            bootstrap: false,
            feature_sampling: crate::FeatureSampling::All,
            seed: 4,
            ..ForestOptions::default()
        })
        .fit(&x, &target, None)
        .expect("fit");
        assert!(fitted.trees().iter().all(|tree| !tree.nodes().is_empty()));
        assert!(fitted
            .predict(&x)
            .expect("predict")
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn oob_requires_bootstrap() {
        let x = matrix(2, 1, &[0.0, 1.0]);
        let error = RandomForestRegressor::new(ForestOptions {
            bootstrap: false,
            out_of_bag: true,
            ..ForestOptions::default()
        })
        .fit(&x, &[0.0, 1.0], None)
        .expect_err("invalid OOB configuration");
        assert!(matches!(
            error,
            Error::InvalidOption {
                name: "out_of_bag",
                ..
            }
        ));
    }

    #[test]
    fn sampling_never_builds_a_tree_from_zero_weight_rows_only() {
        let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let fitted = RandomForestRegressor::new(ForestOptions {
            trees: 32,
            sample_fraction: 0.25,
            seed: 17,
            ..ForestOptions::default()
        })
        .fit(&x, &[0.0, 1.0, 2.0, 3.0], Some(&[0.0, 0.0, 0.0, 1.0]))
        .expect("every sampled tree receives positive weight");
        assert_eq!(fitted.trees().len(), 32);
    }

    #[test]
    fn forest_mean_preserves_representable_extreme_predictions() {
        let x = matrix(2, 1, &[0.0, 1.0]);
        let fitted = RandomForestRegressor::new(ForestOptions {
            trees: 2,
            bootstrap: false,
            feature_sampling: crate::FeatureSampling::All,
            ..ForestOptions::default()
        })
        .fit(&x, &[f64::MAX, f64::MAX], None)
        .expect("fit constant extreme target");
        assert_eq!(
            fitted.predict(&x).expect("finite ensemble mean"),
            vec![f64::MAX, f64::MAX]
        );
    }

    #[test]
    fn oob_r2_skips_zero_weight_arithmetic_and_scales_extremes() {
        let ignored = oob_r2(&[Some(-f64::MAX), Some(0.0)], &[f64::MAX, 0.0], &[0.0, 1.0])
            .expect("zero-weight overflow is outside the metric");
        assert_eq!(ignored, None);

        let exact = oob_r2(
            &[Some(-f64::MAX), Some(f64::MAX)],
            &[-f64::MAX, f64::MAX],
            &[1.0, 1.0],
        )
        .expect("scaled sums of squares")
        .expect("non-constant target");
        assert_eq!(exact.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn subsampling_cannot_hide_a_non_finite_training_feature() {
        let x = matrix(4, 1, &[0.0, 1.0, f64::NAN, 3.0]);
        let error = RandomForestRegressor::new(ForestOptions {
            trees: 1,
            sample_fraction: 0.25,
            seed: 3,
            ..ForestOptions::default()
        })
        .fit(&x, &[0.0, 1.0, 2.0, 3.0], None)
        .expect_err("the complete training matrix must be validated");
        assert_eq!(
            error,
            Error::NonFinite {
                name: "feature",
                index: 2,
            }
        );
    }
}
