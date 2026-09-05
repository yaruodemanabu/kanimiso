//! Histogram-binned, second-order gradient boosting.

use amatsuki::{seed_rng, Rng};
use oldwood::{DenseMatrix, MatrixView, NodeKind, SplitContext, SplitStrategy, TreeOptions};

use crate::data::{
    checked_weights, validate_predict, validate_regression_target, validate_training,
};
use crate::gradient::{
    average_importances, checked_add, class_log_priors, positive_weight_classes, softmax,
    stage_rows, weighted_mean,
};
use crate::numeric::{CompensatedSum, ScaledSum};
use crate::options::count_as_f64;
use crate::random::soft_threshold;
use crate::strategy::{RandomSplitStrategy, ThresholdPolicy};
use crate::{Error, LightGbmOptions, Result};

/// Histogram/Newton boosted regressor inspired by `LightGBM`'s core objective.
///
/// This implementation is depth-wise and uses global quantile bins. It does
/// not claim `LightGBM` model-file, distributed-training, EFB, GOSS, or GPU
/// compatibility.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LightGbmRegressor {
    /// Binning, regularization, sampling, and CART options.
    pub options: LightGbmOptions,
}

impl LightGbmRegressor {
    /// Creates a histogram/Newton regressor with explicit options.
    #[must_use]
    pub fn new(options: LightGbmOptions) -> Self {
        Self { options }
    }

    /// Fits the squared-error Newton objective.
    ///
    /// # Errors
    ///
    /// Returns an error when matrix dimensions, targets, weights, binning, or
    /// options are invalid; when the CART kernel rejects a tree; or when a
    /// Newton or additive calculation is not representable.
    pub fn fit<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        target: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedLightGbmRegressor> {
        validate_training(x, target.len(), sample_weight)?;
        validate_regression_target(target)?;
        self.options.validate(x.ncols())?;
        let feature_count = self.options.feature_sampling.count(x.ncols())?;
        let weights = checked_weights(x.nrows(), sample_weight);
        let binning = HistogramBinner::fit(x, self.options.max_bins, &weights)?;
        let encoded = binning.transform(x)?;
        let base = weighted_mean(target, &weights)?;
        let mut prediction = vec![base; x.nrows()];
        let mut trees = Vec::with_capacity(self.options.boosting.iterations);
        let mut rng = seed_rng(self.options.boosting.seed);
        for _ in 0..self.options.boosting.iterations {
            let gradients: Vec<f64> = prediction
                .iter()
                .zip(target)
                .zip(&weights)
                .map(|((&fitted, &actual), &weight)| {
                    if weight <= 0.0 {
                        return Ok(0.0);
                    }
                    let gradient = (fitted - actual) * weight;
                    if gradient.is_finite() {
                        Ok(gradient)
                    } else {
                        Err(Error::NumericalOverflow {
                            operation: "LightGBM-style regression gradient",
                        })
                    }
                })
                .collect::<Result<_>>()?;
            let hessians = weights.clone();
            let rows = stage_rows(&weights, self.options.boosting.sample_fraction, &mut rng);
            let Some(tree) = fit_newton_tree(
                &encoded,
                &gradients,
                &hessians,
                &rows,
                &self.options,
                feature_count,
                rng.next_u64(),
            )?
            else {
                break;
            };
            let update = tree.predict(&encoded)?;
            for (fitted, value) in prediction.iter_mut().zip(update) {
                *fitted = checked_add(
                    *fitted,
                    self.options.boosting.learning_rate * value,
                    "LightGBM-style regression update",
                )?;
            }
            trees.push(tree);
        }
        Ok(FittedLightGbmRegressor {
            binner: binning,
            base,
            learning_rate: self.options.boosting.learning_rate,
            trees,
            features: x.ncols(),
        })
    }
}

/// Fitted histogram/Newton regressor.
#[derive(Clone, Debug)]
pub struct FittedLightGbmRegressor {
    binner: HistogramBinner,
    base: f64,
    learning_rate: f64,
    trees: Vec<NewtonTree>,
    features: usize,
}

impl FittedLightGbmRegressor {
    /// Weighted target mean used before the first additive stage.
    #[must_use]
    pub fn base_value(&self) -> f64 {
        self.base
    }

    /// Predicts one value per row after applying the stored bin edges.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature count differs from training, a
    /// non-missing feature is non-finite, the CART kernel rejects prediction,
    /// or an additive prediction is not representable.
    pub fn predict<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<f64>> {
        validate_predict(x, self.features)?;
        let binned = self.binner.transform(x)?;
        let mut output = vec![self.base; x.nrows()];
        for tree in &self.trees {
            for (current, value) in output.iter_mut().zip(tree.predict(&binned)?) {
                *current = checked_add(
                    *current,
                    self.learning_rate * value,
                    "LightGBM-style regression prediction",
                )?;
            }
        }
        Ok(output)
    }

    /// Number of fitted stages; this can be smaller than requested after zero Hessian.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.trees.len()
    }

    /// Normalized mean impurity-decrease importance across binned trees.
    #[must_use]
    pub fn feature_importances(&self) -> Vec<f64> {
        average_importances(
            self.trees.iter().map(NewtonTree::feature_importances),
            self.features,
        )
    }
}

/// Multiclass histogram/Newton boosted classifier inspired by `LightGBM`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LightGbmClassifier {
    /// Binning, regularization, sampling, and CART options.
    pub options: LightGbmOptions,
}

impl LightGbmClassifier {
    /// Creates a histogram/Newton classifier with explicit options.
    #[must_use]
    pub fn new(options: LightGbmOptions) -> Self {
        Self { options }
    }

    /// Fits one Newton tree per class and iteration under multinomial softmax.
    ///
    /// # Errors
    ///
    /// Returns an error when matrix dimensions, weights, classes, binning, or
    /// options are invalid; when the CART kernel rejects a tree; or when a
    /// Newton, logit, or probability calculation is not representable.
    pub fn fit<M: MatrixView + ?Sized>(
        &self,
        x: &M,
        target: &[usize],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedLightGbmClassifier> {
        validate_training(x, target.len(), sample_weight)?;
        self.options.validate(x.ncols())?;
        let feature_count = self.options.feature_sampling.count(x.ncols())?;
        let weights = checked_weights(x.nrows(), sample_weight);
        let classes = positive_weight_classes(target, &weights);
        if classes.len() < 2 {
            return Err(Error::InvalidOption {
                name: "target classes",
                requirement: "at least two classes with positive weight",
            });
        }
        let binning = HistogramBinner::fit(x, self.options.max_bins, &weights)?;
        let encoded = binning.transform(x)?;
        let base_logits = class_log_priors(target, &weights, &classes)?;
        let mut logits = vec![base_logits.clone(); x.nrows()];
        let mut stages = Vec::with_capacity(self.options.boosting.iterations);
        let mut rng = seed_rng(self.options.boosting.seed);
        let multiclass_hessian_correction = if classes.len() > 2 {
            count_as_f64(classes.len()) / count_as_f64(classes.len() - 1)
        } else {
            1.0
        };
        for _ in 0..self.options.boosting.iterations {
            let probabilities: Vec<Vec<f64>> = logits
                .iter()
                .map(|row| softmax(row))
                .collect::<Result<_>>()?;
            let rows = stage_rows(&weights, self.options.boosting.sample_fraction, &mut rng);
            let class_indices: Vec<usize> = if classes.len() == 2 {
                vec![1]
            } else {
                (0..classes.len()).collect()
            };
            let mut stage = Vec::with_capacity(class_indices.len());
            for class_index in class_indices {
                let class = classes[class_index];
                let gradients: Vec<f64> = target
                    .iter()
                    .enumerate()
                    .map(|(row, &actual)| {
                        (probabilities[row][class_index] - f64::from(actual == class))
                            * weights[row]
                    })
                    .collect();
                let hessians: Vec<f64> = probabilities
                    .iter()
                    .enumerate()
                    .map(|(row, probability)| {
                        probability[class_index]
                            * (1.0 - probability[class_index])
                            * weights[row]
                            * multiclass_hessian_correction
                    })
                    .collect();
                let Some(tree) = fit_newton_tree(
                    &encoded,
                    &gradients,
                    &hessians,
                    &rows,
                    &self.options,
                    feature_count,
                    rng.next_u64(),
                )?
                else {
                    continue;
                };
                for (row, value) in tree.predict(&encoded)?.into_iter().enumerate() {
                    logits[row][class_index] = checked_add(
                        logits[row][class_index],
                        self.options.boosting.learning_rate * value,
                        "LightGBM-style classification update",
                    )?;
                }
                stage.push((class_index, tree));
            }
            if stage.is_empty() {
                break;
            }
            stages.push(stage);
        }
        Ok(FittedLightGbmClassifier {
            binner: binning,
            classes,
            base_logits,
            learning_rate: self.options.boosting.learning_rate,
            stages,
            features: x.ncols(),
        })
    }
}

/// Fitted multiclass histogram/Newton classifier.
#[derive(Clone, Debug)]
pub struct FittedLightGbmClassifier {
    binner: HistogramBinner,
    classes: Vec<usize>,
    base_logits: Vec<f64>,
    learning_rate: f64,
    stages: Vec<Vec<(usize, NewtonTree)>>,
    features: usize,
}

impl FittedLightGbmClassifier {
    /// Sorted target labels defining probability-column order.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Per-class log priors used before the first additive stage.
    #[must_use]
    pub fn base_logits(&self) -> &[f64] {
        &self.base_logits
    }

    /// Predicts one probability row per observation.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature count differs from training, a
    /// non-missing feature is non-finite, the CART kernel rejects prediction,
    /// or logits cannot be accumulated or normalized.
    pub fn predict_proba<M: MatrixView + ?Sized>(&self, x: &M) -> Result<Vec<Vec<f64>>> {
        validate_predict(x, self.features)?;
        let binned = self.binner.transform(x)?;
        let mut logits = vec![self.base_logits.clone(); x.nrows()];
        for stage in &self.stages {
            for (class_index, tree) in stage {
                for (row, value) in tree.predict(&binned)?.into_iter().enumerate() {
                    logits[row][*class_index] = checked_add(
                        logits[row][*class_index],
                        self.learning_rate * value,
                        "LightGBM-style classification prediction",
                    )?;
                }
            }
        }
        logits.iter().map(|row| softmax(row)).collect()
    }

    /// Predicts the smallest target label attaining the largest probability.
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

    /// Number of non-empty boosting stages.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.stages.len()
    }

    /// Normalized mean impurity-decrease importance across all class trees.
    #[must_use]
    pub fn feature_importances(&self) -> Vec<f64> {
        average_importances(
            self.stages
                .iter()
                .flat_map(|stage| stage.iter().map(|(_, tree)| tree.feature_importances())),
            self.features,
        )
    }
}

#[derive(Clone, Debug)]
struct HistogramBinner {
    edges: Vec<Vec<f64>>,
}

impl HistogramBinner {
    fn fit<M: MatrixView + ?Sized>(x: &M, max_bins: usize, weights: &[f64]) -> Result<Self> {
        let mut all_edges = Vec::with_capacity(x.ncols());
        for column in 0..x.ncols() {
            let mut values = Vec::with_capacity(x.nrows());
            for (row, &weight) in weights.iter().enumerate().take(x.nrows()) {
                let value = x.get(row, column);
                if value.is_infinite() {
                    return Err(Error::NonFinite {
                        name: "feature",
                        index: row * x.ncols() + column,
                    });
                }
                if weight > 0.0 && !value.is_nan() {
                    values.push(value);
                }
            }
            values.sort_by(f64::total_cmp);
            let mut edges = Vec::new();
            if values.len() > 1 {
                let bins = max_bins.min(values.len());
                for bin in 1..bins {
                    let index = (bin * values.len()) / bins;
                    let edge = values[index.saturating_sub(1)];
                    if edges
                        .last()
                        .is_none_or(|last: &f64| !last.total_cmp(&edge).is_eq())
                    {
                        edges.push(edge);
                    }
                }
            }
            all_edges.push(edges);
        }
        Ok(Self { edges: all_edges })
    }

    fn transform<M: MatrixView + ?Sized>(&self, x: &M) -> Result<DenseMatrix> {
        if x.ncols() != self.edges.len() {
            return Err(Error::FeatureCount {
                expected: self.edges.len(),
                actual: x.ncols(),
            });
        }
        let mut values = Vec::with_capacity(x.nrows().saturating_mul(x.ncols()));
        for row in 0..x.nrows() {
            for column in 0..x.ncols() {
                let value = x.get(row, column);
                let bin = if value.is_nan() {
                    self.edges[column].len() + 1
                } else if value.is_infinite() {
                    return Err(Error::NonFinite {
                        name: "feature",
                        index: row * x.ncols() + column,
                    });
                } else {
                    self.edges[column].partition_point(|edge| value > *edge)
                };
                values.push(bin_as_f64(bin));
            }
        }
        DenseMatrix::from_row_major(x.nrows(), x.ncols(), values).map_err(Into::into)
    }
}

#[allow(clippy::cast_precision_loss)]
fn bin_as_f64(bin: usize) -> f64 {
    // `oldwood::DenseMatrix` stores all feature values as f64, so histogram
    // indices cross into that representation at this single conversion point.
    bin as f64
}

#[derive(Clone, Debug)]
struct NewtonTree {
    nodes: Vec<NewtonNode>,
    feature_importances: Vec<f64>,
    features: usize,
}

#[derive(Clone, Debug)]
struct NewtonNode {
    kind: NodeKind,
    value: f64,
}

impl NewtonTree {
    fn predict<M: MatrixView>(&self, x: &M) -> Result<Vec<f64>> {
        if x.ncols() != self.features {
            return Err(Error::FeatureCount {
                expected: self.features,
                actual: x.ncols(),
            });
        }
        Ok((0..x.nrows())
            .map(|row| {
                let mut node = 0;
                loop {
                    match self.nodes[node].kind {
                        NodeKind::Leaf => return self.nodes[node].value,
                        NodeKind::Split {
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
                        }
                    }
                }
            })
            .collect())
    }

    fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }
}

struct NewtonBuildTask {
    node_id: usize,
    depth: usize,
    rows: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
struct NewtonStats {
    gradient: ScaledSum,
    hessian: ScaledSum,
}

#[derive(Debug)]
struct NewtonSplit {
    feature: usize,
    threshold: f64,
    gain: f64,
    left_rows: Vec<usize>,
    right_rows: Vec<usize>,
}

fn fit_newton_tree(
    x: &DenseMatrix,
    gradients: &[f64],
    hessians: &[f64],
    rows: &[usize],
    options: &LightGbmOptions,
    feature_count: usize,
    seed: u64,
) -> Result<Option<NewtonTree>> {
    let positive_rows: Vec<usize> = rows
        .iter()
        .copied()
        .filter(|&row| hessians[row] > 0.0)
        .collect();
    if positive_rows.is_empty() {
        return Ok(None);
    }
    if options.min_hessian_leaf > 0.0 {
        let mut root_hessian_ratio = CompensatedSum::default();
        let mut one = CompensatedSum::default();
        one.add(1.0, "root Hessian threshold")?;
        let mut reaches_minimum = false;
        for &row in &positive_rows {
            if hessians[row] >= options.min_hessian_leaf {
                reaches_minimum = true;
                break;
            }
            root_hessian_ratio.add(
                hessians[row] / options.min_hessian_leaf,
                "root Hessian threshold ratio",
            )?;
            if root_hessian_ratio.cmp(one).is_ge() {
                reaches_minimum = true;
                break;
            }
        }
        if !reaches_minimum {
            return Ok(None);
        }
    }
    let mut strategy = RandomSplitStrategy::new(seed, feature_count, ThresholdPolicy::Exhaustive);
    let mut nodes = vec![NewtonNode {
        kind: NodeKind::Leaf,
        value: 0.0,
    }];
    let mut raw_importances = vec![ScaledSum::default(); x.ncols()];
    let mut stack = vec![NewtonBuildTask {
        node_id: 0,
        depth: 0,
        rows: positive_rows,
    }];
    while let Some(task) = stack.pop() {
        let stats = newton_stats(gradients, hessians, &task.rows)?;
        nodes[task.node_id].value = regularized_leaf_value(stats, options)?;
        if should_stop_newton(&options.boosting.tree, task.depth, task.rows.len()) {
            continue;
        }
        let context = SplitContext {
            node_id: task.node_id,
            depth: task.depth,
            sample_count: task.rows.len(),
        };
        let Some(split) = best_newton_split(
            x,
            gradients,
            hessians,
            &task.rows,
            stats,
            options,
            feature_count,
            context,
            &mut strategy,
        )?
        else {
            continue;
        };
        let left = nodes.len();
        nodes.push(NewtonNode {
            kind: NodeKind::Leaf,
            value: 0.0,
        });
        let right = nodes.len();
        nodes.push(NewtonNode {
            kind: NodeKind::Leaf,
            value: 0.0,
        });
        nodes[task.node_id].kind = NodeKind::Split {
            feature: split.feature,
            threshold: split.threshold,
            left,
            right,
        };
        raw_importances[split.feature].add(split.gain, "Newton feature importance")?;
        stack.push(NewtonBuildTask {
            node_id: right,
            depth: task.depth + 1,
            rows: split.right_rows,
        });
        stack.push(NewtonBuildTask {
            node_id: left,
            depth: task.depth + 1,
            rows: split.left_rows,
        });
    }
    Ok(Some(NewtonTree {
        nodes,
        feature_importances: normalize_newton_importances(&raw_importances)?,
        features: x.ncols(),
    }))
}

fn should_stop_newton(options: &TreeOptions, depth: usize, samples: usize) -> bool {
    options.max_depth.is_some_and(|limit| depth >= limit) || samples < options.min_samples_split
}

fn newton_stats(gradients: &[f64], hessians: &[f64], rows: &[usize]) -> Result<NewtonStats> {
    let mut gradient = ScaledSum::default();
    let mut hessian = ScaledSum::default();
    for &row in rows {
        gradient.add(gradients[row], "node gradient sum")?;
        hessian.add(hessians[row], "node Hessian sum")?;
    }
    Ok(NewtonStats { gradient, hessian })
}

fn regularized_components(
    stats: NewtonStats,
    options: &LightGbmOptions,
    operation: &'static str,
) -> Result<Option<(f64, f64, f64, f64)>> {
    let (gradient_normalized, gradient_sum_scale) = stats.gradient.components(operation)?;
    let (hessian_normalized, hessian_sum_scale) = stats.hessian.components(operation)?;
    let gradient_scale = gradient_sum_scale.max(options.l1_regularization);
    let denominator_scale = hessian_sum_scale.max(options.l2_regularization);
    if gradient_scale == 0.0 || denominator_scale == 0.0 {
        return Ok(None);
    }
    let normalized_gradient = gradient_normalized * (gradient_sum_scale / gradient_scale);
    let normalized_l1 = options.l1_regularization / gradient_scale;
    let thresholded = soft_threshold(normalized_gradient, normalized_l1);
    if thresholded == 0.0 {
        return Ok(None);
    }
    let normalized_denominator = hessian_normalized * (hessian_sum_scale / denominator_scale)
        + options.l2_regularization / denominator_scale;
    if !(normalized_gradient.is_finite()
        && thresholded.is_finite()
        && normalized_denominator.is_finite()
        && normalized_denominator > 0.0)
    {
        return Err(Error::NumericalOverflow { operation });
    }
    Ok(Some((
        thresholded,
        gradient_scale,
        normalized_denominator,
        denominator_scale,
    )))
}

fn regularized_leaf_value(stats: NewtonStats, options: &LightGbmOptions) -> Result<f64> {
    let Some((thresholded, gradient_scale, denominator, denominator_scale)) =
        regularized_components(stats, options, "regularized Newton leaf value")?
    else {
        return Ok(0.0);
    };
    crate::numeric::scaled_product_ratio(
        &[-thresholded, gradient_scale],
        &[denominator, denominator_scale],
        "regularized Newton leaf value",
    )
}

fn regularized_score(stats: NewtonStats, options: &LightGbmOptions) -> Result<f64> {
    let Some((thresholded, gradient_scale, denominator, denominator_scale)) =
        regularized_components(stats, options, "regularized Newton split score")?
    else {
        return Ok(0.0);
    };
    crate::numeric::scaled_product_ratio(
        &[thresholded, thresholded, gradient_scale, gradient_scale],
        &[denominator, denominator_scale],
        "regularized Newton split score",
    )
}

#[allow(clippy::too_many_arguments)]
fn best_newton_split<S: SplitStrategy + ?Sized>(
    x: &DenseMatrix,
    gradients: &[f64],
    hessians: &[f64],
    rows: &[usize],
    parent: NewtonStats,
    options: &LightGbmOptions,
    feature_count: usize,
    context: SplitContext,
    strategy: &mut S,
) -> Result<Option<NewtonSplit>> {
    let features = newton_candidate_features(x.ncols(), feature_count, context, strategy)?;
    let parent_score = regularized_score(parent, options)?;
    let minimum_hessian = options
        .min_hessian_leaf
        .max(options.boosting.tree.min_weight_leaf);
    let mut best: Option<NewtonSplit> = None;
    for feature in features {
        let mut unique_values: Vec<f64> = rows.iter().map(|&row| x.get(row, feature)).collect();
        unique_values.sort_by(f64::total_cmp);
        unique_values.dedup_by(|left, right| left.total_cmp(right).is_eq());
        if unique_values.len() < 2 {
            continue;
        }
        // Reduce once per bin. Prefix/suffix scans avoid subtracting nearly
        // equal totals and avoid a row scan for every candidate threshold.
        let mut bins = vec![NewtonStats::default(); unique_values.len()];
        let mut counts = vec![0; bins.len()];
        for &row in rows {
            let bin = unique_values
                .binary_search_by(|v| v.total_cmp(&x.get(row, feature)))
                .expect("observed bin");
            bins[bin].gradient.add(gradients[row], "bin gradient")?;
            bins[bin].hessian.add(hessians[row], "bin Hessian")?;
            counts[bin] += 1;
        }
        let mut prefix = bins.clone();
        for i in 1..bins.len() {
            prefix[i] = merge_newton_stats(prefix[i - 1], bins[i])?;
            counts[i] += counts[i - 1];
        }
        let mut suffix = bins.clone();
        for i in (0..bins.len() - 1).rev() {
            suffix[i] = merge_newton_stats(bins[i], suffix[i + 1])?;
        }
        let thresholds = newton_candidate_thresholds(context, feature, &unique_values, strategy)?;
        for threshold in thresholds {
            let boundary = unique_values.partition_point(|&value| value <= threshold);
            if boundary == 0 || boundary == bins.len() {
                continue;
            }
            let left = prefix[boundary - 1];
            let right = suffix[boundary];
            if counts[boundary - 1] < options.boosting.tree.min_samples_leaf
                || rows.len() - counts[boundary - 1] < options.boosting.tree.min_samples_leaf
                || !stats_hessian_reaches(left, minimum_hessian)?
                || !stats_hessian_reaches(right, minimum_hessian)?
            {
                continue;
            }
            let left_score = regularized_score(left, options)?;
            let right_score = regularized_score(right, options)?;
            let mut gain_sum = ScaledSum::default();
            gain_sum.add(left_score, "regularized Newton split gain")?;
            gain_sum.add(right_score, "regularized Newton split gain")?;
            gain_sum.add(-parent_score, "regularized Newton split gain")?;
            let gain = gain_sum.value("regularized Newton split gain")?;
            if gain <= 0.0 || gain < options.boosting.tree.min_impurity_decrease {
                continue;
            }
            let replace = best.as_ref().is_none_or(|current| {
                gain.total_cmp(&current.gain).is_gt()
                    || (gain.total_cmp(&current.gain).is_eq()
                        && (feature, threshold) < (current.feature, current.threshold))
            });
            if replace {
                best = Some(NewtonSplit {
                    feature,
                    threshold,
                    gain,
                    left_rows: Vec::new(),
                    right_rows: Vec::new(),
                });
            }
        }
    }
    if let Some(split) = &mut best {
        (split.left_rows, split.right_rows) = rows
            .iter()
            .copied()
            .partition(|&row| x.get(row, split.feature) <= split.threshold);
    }
    Ok(best)
}

fn merge_newton_stats(mut left: NewtonStats, right: NewtonStats) -> Result<NewtonStats> {
    left.gradient.merge(right.gradient, "bin gradient scan")?;
    left.hessian.merge(right.hessian, "bin Hessian scan")?;
    Ok(left)
}

fn stats_hessian_reaches(stats: NewtonStats, minimum: f64) -> Result<bool> {
    if minimum == 0.0 {
        return Ok(true);
    }
    let (normalized, scale) = stats.hessian.components("bin Hessian threshold")?;
    Ok(scale > 0.0 && normalized >= minimum / scale)
}

fn newton_candidate_features<S: SplitStrategy + ?Sized>(
    columns: usize,
    feature_count: usize,
    context: SplitContext,
    strategy: &mut S,
) -> Result<Vec<usize>> {
    let mut features = Vec::new();
    strategy.features(context, columns, &mut features);
    if let Some(&feature) = features.iter().find(|&&feature| feature >= columns) {
        return Err(oldwood::Error::InvalidStrategyFeature { feature, columns }.into());
    }
    features.sort_unstable();
    features.dedup();
    features.truncate(feature_count);
    Ok(features)
}

fn newton_candidate_thresholds<S: SplitStrategy + ?Sized>(
    context: SplitContext,
    feature: usize,
    unique_values: &[f64],
    strategy: &mut S,
) -> Result<Vec<f64>> {
    let lower = unique_values[0];
    let upper = *unique_values.last().expect("at least two unique values");
    let mut thresholds = Vec::new();
    strategy.thresholds(context, feature, unique_values, &mut thresholds);
    if thresholds
        .iter()
        .any(|threshold| !(threshold.is_finite() && *threshold >= lower && *threshold < upper))
    {
        return Err(oldwood::Error::InvalidStrategyThreshold { feature }.into());
    }
    thresholds.sort_by(f64::total_cmp);
    thresholds.dedup_by(|left, right| left.total_cmp(right).is_eq());
    Ok(thresholds)
}

fn normalize_newton_importances(importances: &[ScaledSum]) -> Result<Vec<f64>> {
    let mut components = Vec::with_capacity(importances.len());
    let mut common_scale: f64 = 0.0;
    for &importance in importances {
        let component = importance.components("Newton feature importance")?;
        common_scale = common_scale.max(component.1);
        components.push(component);
    }
    if common_scale == 0.0 {
        return Ok(vec![0.0; importances.len()]);
    }
    let mut normalized = Vec::with_capacity(importances.len());
    let mut total = CompensatedSum::default();
    for (coefficient, scale) in components {
        let value = coefficient * (scale / common_scale);
        total.add(value, "Newton feature importance normalization")?;
        normalized.push(value);
    }
    let total = total.value("Newton feature importance normalization")?;
    for value in &mut normalized {
        *value /= total;
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, columns: usize, values: &[f64]) -> DenseMatrix {
        DenseMatrix::from_row_major(rows, columns, values.to_vec()).expect("valid fixture")
    }

    #[test]
    fn quantile_bins_are_monotone_and_missing_is_a_distinct_high_bin() {
        let x = matrix(6, 1, &[3.0, f64::NAN, 1.0, 5.0, 2.0, 4.0]);
        let binning = HistogramBinner::fit(&x, 3, &[1.0; 6]).expect("fit bins");
        let encoded = binning.transform(&x).expect("transform");
        assert!(encoded.get(2, 0) <= encoded.get(4, 0));
        assert!(encoded.get(4, 0) <= encoded.get(0, 0));
        assert!(encoded.get(0, 0) <= encoded.get(5, 0));
        assert!(encoded.get(5, 0) <= encoded.get(3, 0));
        assert!(encoded.get(1, 0) > encoded.get(3, 0));
    }

    #[test]
    fn quantile_edges_respect_observation_multiplicity() {
        let x = matrix(8, 1, &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        let binning = HistogramBinner::fit(&x, 4, &[1.0; 8]).expect("fit bins");
        assert_eq!(binning.edges, vec![vec![0.0, 2.0]]);
    }

    #[test]
    fn zero_weight_feature_values_do_not_change_quantile_edges() {
        let base = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let augmented = matrix(6, 1, &[0.0, 1.0, 2.0, 3.0, -1.0e300, 1.0e300]);
        let base_bins = HistogramBinner::fit(&base, 3, &[1.0; 4]).expect("base bins");
        let augmented_bins = HistogramBinner::fit(&augmented, 3, &[1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
            .expect("augmented bins");
        assert_eq!(base_bins.edges, augmented_bins.edges);
    }

    #[test]
    fn constant_regression_matches_the_closed_form_mean_exactly() {
        let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let fitted = LightGbmRegressor::new(LightGbmOptions {
            boosting: crate::BoostingOptions {
                iterations: 5,
                ..crate::BoostingOptions::default()
            },
            ..LightGbmOptions::default()
        })
        .fit(&x, &[7.0, 7.0, 7.0, 7.0], None)
        .expect("fit");
        assert_eq!(fitted.predict(&x).expect("predict"), vec![7.0; 4]);
    }

    #[test]
    fn zero_weight_extreme_target_is_not_used_in_gradient_arithmetic() {
        let x = matrix(2, 1, &[0.0, 1.0]);
        let fitted = LightGbmRegressor::new(LightGbmOptions {
            boosting: crate::BoostingOptions {
                iterations: 1,
                ..crate::BoostingOptions::default()
            },
            ..LightGbmOptions::default()
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
    fn one_newton_regression_stage_matches_the_closed_form_leaf_values() {
        let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let fitted = LightGbmRegressor::new(LightGbmOptions {
            boosting: crate::BoostingOptions {
                iterations: 1,
                learning_rate: 0.25,
                ..crate::BoostingOptions::default()
            },
            max_bins: 4,
            ..LightGbmOptions::default()
        })
        .fit(&x, &[0.0, 0.0, 2.0, 2.0], None)
        .expect("fit");
        assert_eq!(
            fitted.predict(&x).expect("predict"),
            vec![0.75, 0.75, 1.25, 1.25]
        );
    }

    #[test]
    fn l1_and_l2_regularized_newton_leaves_match_closed_form() {
        let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let fitted = LightGbmRegressor::new(LightGbmOptions {
            boosting: crate::BoostingOptions {
                iterations: 1,
                learning_rate: 1.0,
                ..crate::BoostingOptions::default()
            },
            max_bins: 4,
            l1_regularization: 1.0,
            l2_regularization: 2.0,
            ..LightGbmOptions::default()
        })
        .fit(&x, &[0.0, 0.0, 2.0, 2.0], None)
        .expect("fit");
        assert_eq!(
            fitted.predict(&x).expect("predict"),
            vec![0.75, 0.75, 1.25, 1.25]
        );
    }

    #[test]
    fn regularized_newton_gain_selects_the_best_aggregate_split() {
        let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let fitted = LightGbmRegressor::new(LightGbmOptions {
            boosting: crate::BoostingOptions {
                iterations: 1,
                learning_rate: 1.0,
                tree: TreeOptions {
                    max_depth: Some(1),
                    ..TreeOptions::default()
                },
                ..crate::BoostingOptions::default()
            },
            max_bins: 4,
            l2_regularization: 10.0,
            ..LightGbmOptions::default()
        })
        .fit(&x, &[-2.0, -1.0, 0.0, 2.0], None)
        .expect("fit");
        assert_eq!(fitted.trees.len(), 1);
        assert!(matches!(
            fitted.trees[0].nodes[0].kind,
            NodeKind::Split {
                feature: 0,
                threshold: 1.5,
                ..
            }
        ));
    }

    #[test]
    fn regularized_root_leaf_avoids_pseudo_response_overflow() {
        let x = matrix(1, 1, &[0.0]);
        let options = LightGbmOptions {
            l2_regularization: 1.0,
            ..LightGbmOptions::default()
        };
        let fitted = fit_newton_tree(&x, &[1.0], &[f64::from_bits(1)], &[0], &options, 1, 0)
            .expect("aggregate Newton fit")
            .expect("one leaf");
        assert_eq!(fitted.predict(&x).expect("predict"), vec![-1.0]);
    }

    #[test]
    #[allow(clippy::float_cmp)] // Deliberate exact agreement of independently reduced scores.
    fn binned_newton_splits_match_independent_candidate_reductions() {
        let options = LightGbmOptions {
            l1_regularization: 0.5,
            l2_regularization: 1.25,
            boosting: crate::BoostingOptions {
                tree: TreeOptions {
                    max_depth: Some(1),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut worst: f64 = 0.0;
        for seed in 0..48 {
            let x = DenseMatrix::from_row_major(
                15,
                2,
                (0..30)
                    .map(|k| f64::from((k / 2 * (seed + 3) + (k % 2) * 5) % 7))
                    .collect(),
            )
            .unwrap();
            let g: Vec<_> = (0..15)
                .map(|i| f64::from((i * 7 + seed * 3) % 13) - 6.0)
                .collect();
            let h: Vec<_> = (0..15).map(|i| 1.0 + f64::from(i % 3)).collect();
            let rows: Vec<_> = (0..15).collect();
            let score = |selected: &[usize]| {
                let gradient: f64 = selected.iter().map(|&i| g[i]).sum();
                let hessian: f64 = selected.iter().map(|&i| h[i]).sum();
                (gradient.abs() - options.l1_regularization)
                    .max(0.0)
                    .powi(2)
                    / (hessian + options.l2_regularization)
            };
            let parent = score(&rows);
            let gain = |feature, threshold| {
                let (left, right): (Vec<_>, Vec<_>) = rows
                    .iter()
                    .copied()
                    .partition(|&i| x.get(i, feature) <= threshold);
                if left.is_empty() || right.is_empty() {
                    0.0
                } else {
                    score(&left) + score(&right) - parent
                }
            };
            let expected = (0..2)
                .flat_map(|feature| (0..6).map(move |bin| (feature, f64::from(bin) + 0.5)))
                .map(|(feature, threshold)| gain(feature, threshold))
                .fold(0.0, f64::max);
            let fit = fit_newton_tree(&x, &g, &h, &rows, &options, 2, 0)
                .unwrap()
                .unwrap();
            let actual = match fit.nodes[0].kind {
                NodeKind::Leaf => 0.0,
                NodeKind::Split {
                    feature, threshold, ..
                } => gain(feature, threshold),
            };
            worst = worst.max((actual - expected).abs() / (1.0 + expected.abs()));
        }
        eprintln!("Newton split brute-force scaled error: {worst:e}");
        // The winning independently reduced score agrees exactly on these 48 fixtures.
        assert_eq!(worst, 0.0);
    }

    #[test]
    fn newton_leaf_ratio_stays_finite_when_raw_denominator_overflows() {
        let x = matrix(1, 1, &[0.0]);
        let options = LightGbmOptions {
            l2_regularization: f64::MAX,
            ..LightGbmOptions::default()
        };
        let fitted = fit_newton_tree(&x, &[f64::MAX], &[f64::MAX], &[0], &options, 1, 0)
            .expect("scaled regularized ratio")
            .expect("one leaf");
        assert_eq!(fitted.predict(&x).expect("predict"), vec![-0.5]);
    }

    #[test]
    fn minimum_hessian_applies_to_the_root_leaf_and_stops_the_stage() {
        let x = matrix(2, 1, &[0.0, 1.0]);
        let fitted = LightGbmRegressor::new(LightGbmOptions {
            boosting: crate::BoostingOptions {
                iterations: 1,
                ..crate::BoostingOptions::default()
            },
            min_hessian_leaf: 3.0,
            ..LightGbmOptions::default()
        })
        .fit(&x, &[0.0, 2.0], None)
        .expect("fit");
        assert_eq!(fitted.iterations(), 0);
        assert_eq!(fitted.predict(&x).expect("predict"), vec![1.0, 1.0]);
    }

    #[test]
    fn zero_hessian_input_produces_no_newton_tree() {
        let x = matrix(2, 1, &[0.0, 1.0]);
        let rows = [0, 1];
        let fitted = fit_newton_tree(
            &x,
            &[1.0, -1.0],
            &[0.0, 0.0],
            &rows,
            &LightGbmOptions::default(),
            1,
            0,
        )
        .expect("zero Hessian is a defined stop");
        assert!(fitted.is_none());
    }

    #[test]
    fn root_hessian_accepts_one_value_above_a_subnormal_minimum() {
        let x = matrix(1, 1, &[0.0]);
        let options = LightGbmOptions {
            min_hessian_leaf: f64::from_bits(1),
            ..LightGbmOptions::default()
        };
        let fitted = fit_newton_tree(&x, &[0.0], &[1.0], &[0], &options, 1, 0)
            .expect("finite threshold check");
        assert!(fitted.is_some());
    }

    #[test]
    fn root_hessian_keeps_an_exact_boundary_hidden_below_one_ulp() {
        let x = matrix(5, 1, &[0.0; 5]);
        let quarter_ulp = 2.0f64.powi(-54);
        let hessians = [
            1.0 - 2.0f64.powi(-52),
            quarter_ulp,
            quarter_ulp,
            quarter_ulp,
            quarter_ulp,
        ];
        let options = LightGbmOptions {
            min_hessian_leaf: 1.0,
            ..LightGbmOptions::default()
        };
        let fitted = fit_newton_tree(&x, &[0.0; 5], &hessians, &[0, 1, 2, 3, 4], &options, 1, 0)
            .expect("compensated boundary check");
        assert!(fitted.is_some());
    }

    #[test]
    fn leaf_gradient_sum_preserves_a_small_cancellation_residual() {
        let x = matrix(3, 1, &[0.0; 3]);
        let fitted = fit_newton_tree(
            &x,
            &[1.0e16, 1.0, -1.0e16],
            &[1.0; 3],
            &[0, 1, 2],
            &LightGbmOptions::default(),
            1,
            0,
        )
        .expect("fit")
        .expect("one leaf");
        assert_eq!(fitted.predict(&x).expect("predict"), vec![-1.0 / 3.0; 3]);
    }

    #[test]
    fn binary_newton_stage_uses_one_logit_tree_and_matches_closed_form() {
        let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let fitted = LightGbmClassifier::new(LightGbmOptions {
            boosting: crate::BoostingOptions {
                iterations: 1,
                learning_rate: 0.25,
                ..crate::BoostingOptions::default()
            },
            max_bins: 4,
            ..LightGbmOptions::default()
        })
        .fit(&x, &[0, 0, 1, 1], None)
        .expect("fit");
        let probabilities = fitted.predict_proba(&x).expect("probability");
        let expected_low: f64 = 0.377_540_668_798_145_4;
        let expected_high: f64 = 0.622_459_331_201_854_6;
        for row in &probabilities[..2] {
            // Closed-form sigmoid(-0.5); measured 5.551115123125783e-17.
            assert!((row[1] - expected_low).abs() <= 2.3e-16);
        }
        for row in &probabilities[2..] {
            // Closed-form sigmoid(0.5); measured 0.0 on 2026-09-04.
            assert_eq!(row[1].to_bits(), expected_high.to_bits());
        }
    }

    #[test]
    fn multiclass_hessian_uses_the_redundant_logit_correction() {
        let x = matrix(6, 1, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let fitted = LightGbmClassifier::new(LightGbmOptions {
            boosting: crate::BoostingOptions {
                iterations: 1,
                learning_rate: 1.0,
                tree: oldwood::TreeOptions {
                    max_depth: Some(2),
                    ..oldwood::TreeOptions::default()
                },
                ..crate::BoostingOptions::default()
            },
            max_bins: 6,
            ..LightGbmOptions::default()
        })
        .fit(&x, &[0, 0, 1, 1, 2, 2], None)
        .expect("fit");
        let probabilities = fitted.predict_proba(&x).expect("probability");
        let expected = 2.0_f64.exp() / (2.0_f64.exp() + 2.0 * (-1.0_f64).exp());
        for (row, probability) in probabilities.iter().enumerate() {
            // Closed-form logits are (+2, -1, -1) up to class permutation;
            // Measured absolute error was one ULP (1.11e-16) on 2026-09-05;
            // four ULPs retain a narrow architecture-independent margin.
            assert!((probability[row / 2] - expected).abs() <= 4.0 * f64::EPSILON);
        }
    }

    #[test]
    fn invalid_nested_tree_options_fail_before_a_zero_stage_exit() {
        let x = matrix(2, 1, &[0.0, 1.0]);
        for tree in [
            oldwood::TreeOptions {
                min_samples_split: 1,
                ..oldwood::TreeOptions::default()
            },
            oldwood::TreeOptions {
                min_samples_leaf: 0,
                ..oldwood::TreeOptions::default()
            },
            oldwood::TreeOptions {
                min_weight_leaf: -1.0,
                ..oldwood::TreeOptions::default()
            },
            oldwood::TreeOptions {
                min_impurity_decrease: -1.0,
                ..oldwood::TreeOptions::default()
            },
        ] {
            let error = LightGbmRegressor::new(LightGbmOptions {
                boosting: crate::BoostingOptions {
                    tree,
                    ..crate::BoostingOptions::default()
                },
                min_hessian_leaf: f64::MAX,
                ..LightGbmOptions::default()
            })
            .fit(&x, &[0.0, 1.0], None)
            .expect_err("invalid CART option must be validated eagerly");
            assert!(matches!(
                error,
                Error::Tree(oldwood::Error::InvalidOption { .. })
            ));
        }
    }

    #[test]
    fn multiclass_probabilities_are_finite_and_normalized() {
        let x = matrix(9, 1, &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2]);
        let target = [3, 3, 3, 5, 5, 5, 8, 8, 8];
        let fitted = LightGbmClassifier::new(LightGbmOptions {
            boosting: crate::BoostingOptions {
                iterations: 12,
                learning_rate: 0.2,
                ..crate::BoostingOptions::default()
            },
            max_bins: 4,
            ..LightGbmOptions::default()
        })
        .fit(&x, &target, None)
        .expect("fit");
        for row in fitted.predict_proba(&x).expect("probabilities") {
            let sum: f64 = row.iter().sum();
            // Measured 2.220446049250313e-16 on 2026-09-04; fourfold margin.
            assert!((sum - 1.0).abs() <= 9.0e-16);
            assert!(row.iter().all(|value| value.is_finite()));
        }
    }
}
