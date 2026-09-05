use crate::numeric::CompensatedSum;
use crate::split::{
    best_class_split, best_regression_split, class_summary, partition, regression_summary,
    ClassSplitRequest, RegressionSplitRequest,
};
use crate::validation::{
    validate_prediction_matrix, validate_target_length, validate_targets, validate_training_matrix,
    weights,
};
use crate::{
    ArenaNode, ClassificationCriterion, ClassificationValue, DenseMatrix, Error, Exhaustive,
    MatrixView, NodeKind, RegressionCriterion, Result, SplitContext, SplitStrategy, TreeOptions,
};

/// Deterministic CART classifier.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionTreeClassifier {
    criterion: ClassificationCriterion,
    options: TreeOptions,
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new(ClassificationCriterion::default(), TreeOptions::default())
    }
}

impl DecisionTreeClassifier {
    /// Creates a classifier with a runtime criterion and shared tree options.
    #[must_use]
    pub fn new(criterion: ClassificationCriterion, options: TreeOptions) -> Self {
        Self { criterion, options }
    }

    /// Configured criterion.
    #[must_use]
    pub fn criterion(&self) -> ClassificationCriterion {
        self.criterion
    }

    /// Configured stopping and feature-selection options.
    #[must_use]
    pub fn options(&self) -> &TreeOptions {
        &self.options
    }

    /// Fits exhaustive deterministic CART.
    ///
    /// Zero-weight rows are ignored completely. Positive weights affect leaf
    /// probabilities, predictions, impurity, and split selection.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] for invalid shapes, options, feature values,
    /// weights, strategy output, or non-representable numerical results.
    pub fn fit<M: MatrixView>(
        &self,
        matrix: &M,
        targets: &[usize],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedClassifier> {
        self.fit_with_strategy(matrix, targets, sample_weight, &mut Exhaustive)
    }

    /// Fits with externally supplied feature and threshold candidates.
    ///
    /// The strategy cannot replace CART scoring. Candidate validation,
    /// stable ordering, deduplication, gain evaluation, and tie-breaking stay
    /// inside this crate.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] for invalid shapes, options, feature values,
    /// weights, strategy output, or non-representable numerical results.
    pub fn fit_with_strategy<M: MatrixView, S: SplitStrategy + ?Sized>(
        &self,
        matrix: &M,
        targets: &[usize],
        sample_weight: Option<&[f64]>,
        strategy: &mut S,
    ) -> Result<FittedClassifier> {
        let (sample_weight, indices, classes) =
            prepare_classification(matrix, targets, sample_weight, &self.options)?;
        let root_weight = selected_weight(&sample_weight, &indices)?;

        let mut nodes = vec![classification_placeholder(classes[0], classes.len())];
        let mut raw_importances = vec![0.0; matrix.ncols()];
        let mut stack = vec![BuildTask {
            node_id: 0,
            depth: 0,
            indices,
        }];
        while let Some(task) = stack.pop() {
            let summary = class_summary(
                &task.indices,
                targets,
                &sample_weight,
                &classes,
                self.criterion,
            )?;
            nodes[task.node_id] = ArenaNode {
                kind: NodeKind::Leaf,
                sample_count: task.indices.len(),
                weighted_sample_count: summary.weight,
                impurity: summary.impurity,
                impurity_decrease: 0.0,
                value: ClassificationValue {
                    predicted_class: summary.prediction,
                    class_weights: summary.counts.clone(),
                },
            };
            if should_stop(
                &self.options,
                task.depth,
                task.indices.len(),
                summary.impurity,
            ) {
                continue;
            }
            let context = SplitContext {
                node_id: task.node_id,
                depth: task.depth,
                sample_count: task.indices.len(),
            };
            let Some(split) = best_class_split(
                &ClassSplitRequest {
                    matrix,
                    targets,
                    weights: &sample_weight,
                    classes: &classes,
                    indices: &task.indices,
                    parent: &summary,
                    root_weight,
                    criterion: self.criterion,
                    options: &self.options,
                    context,
                },
                strategy,
            )?
            else {
                continue;
            };
            let (left_indices, right_indices) =
                partition(matrix, &task.indices, split.feature, split.threshold);
            let left = nodes.len();
            nodes.push(classification_placeholder(classes[0], classes.len()));
            let right = nodes.len();
            nodes.push(classification_placeholder(classes[0], classes.len()));
            nodes[task.node_id].kind = NodeKind::Split {
                feature: split.feature,
                threshold: split.threshold,
                left,
                right,
            };
            nodes[task.node_id].impurity_decrease = split.gain;
            raw_importances[split.feature] =
                add_importance(raw_importances[split.feature], summary.weight, split.gain)?;
            stack.push(BuildTask {
                node_id: right,
                depth: task.depth + 1,
                indices: right_indices,
            });
            stack.push(BuildTask {
                node_id: left,
                depth: task.depth + 1,
                indices: left_indices,
            });
        }
        normalize_importances(&mut raw_importances)?;
        Ok(FittedClassifier {
            nodes,
            classes,
            feature_importances: raw_importances,
            n_features: matrix.ncols(),
        })
    }
}

/// Fitted classification tree.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedClassifier {
    nodes: Vec<ArenaNode<ClassificationValue>>,
    classes: Vec<usize>,
    feature_importances: Vec<f64>,
    n_features: usize,
}

impl FittedClassifier {
    /// Sorted positive-weight training labels.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Immutable arena. Root index is zero.
    #[must_use]
    pub fn nodes(&self) -> &[ArenaNode<ClassificationValue>] {
        &self.nodes
    }

    /// Number of input features used during fitting.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Normalized weighted impurity decrease for each feature.
    #[must_use]
    pub fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }

    /// Predicts one class label per row.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] when the matrix has the wrong feature count or a
    /// non-finite value.
    pub fn predict<M: MatrixView>(&self, matrix: &M) -> Result<Vec<usize>> {
        let leaves = self.apply(matrix)?;
        Ok(leaves
            .iter()
            .map(|&leaf| self.nodes[leaf].value.predicted_class)
            .collect())
    }

    /// Returns row-major class probabilities aligned with [`Self::classes`].
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] when the matrix has the wrong feature count or a
    /// non-finite value, or when probability construction overflows.
    pub fn predict_proba<M: MatrixView>(&self, matrix: &M) -> Result<DenseMatrix> {
        let leaves = self.apply(matrix)?;
        let mut probabilities = Vec::with_capacity(matrix.nrows() * self.classes.len());
        for leaf in leaves {
            let value = &self.nodes[leaf].value;
            let counts = &value.class_weights;
            let mut total = CompensatedSum::default();
            for &count in counts {
                total.add(count, "leaf class-weight summation")?;
            }
            let total = total.total("leaf class-weight summation")?;
            let mut row: Vec<f64> = counts.iter().map(|count| count / total).collect();
            preserve_exact_winner(&mut row, value.predicted_class, &self.classes);
            probabilities.extend(row);
        }
        DenseMatrix::from_row_major(matrix.nrows(), self.classes.len(), probabilities)
    }

    /// Returns the leaf arena index reached by every row.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] when the matrix has the wrong feature count or a
    /// non-finite value.
    pub fn apply<M: MatrixView>(&self, matrix: &M) -> Result<Vec<usize>> {
        validate_prediction_matrix(matrix, self.n_features)?;
        Ok((0..matrix.nrows())
            .map(|row| leaf_index(&self.nodes, matrix, row))
            .collect())
    }
}

#[allow(clippy::float_cmp)] // Only bit-identical rounded ties require the one-ULP projection.
fn preserve_exact_winner(probabilities: &mut [f64], predicted_class: usize, classes: &[usize]) {
    let predicted = classes
        .binary_search(&predicted_class)
        .expect("node prediction belongs to fitted classes");
    let rounded_winner = probabilities
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map_or(0, |(index, _)| index);
    if rounded_winner == predicted || probabilities[rounded_winner] != probabilities[predicted] {
        return;
    }
    let increased = next_up(probabilities[predicted]);
    let transfer = increased - probabilities[predicted];
    let decreased = probabilities[rounded_winner] - transfer;
    if increased <= 1.0 && decreased >= 0.0 {
        probabilities[predicted] = increased;
        probabilities[rounded_winner] = decreased;
    }
}

fn next_up(value: f64) -> f64 {
    if value == f64::INFINITY {
        value
    } else if value == -0.0 {
        f64::from_bits(1)
    } else if value >= 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

/// Deterministic CART regressor.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionTreeRegressor {
    criterion: RegressionCriterion,
    options: TreeOptions,
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self::new(RegressionCriterion::default(), TreeOptions::default())
    }
}

impl DecisionTreeRegressor {
    /// Creates a regressor with a runtime criterion and shared tree options.
    #[must_use]
    pub fn new(criterion: RegressionCriterion, options: TreeOptions) -> Self {
        Self { criterion, options }
    }

    /// Configured criterion.
    #[must_use]
    pub fn criterion(&self) -> RegressionCriterion {
        self.criterion
    }

    /// Configured stopping and feature-selection options.
    #[must_use]
    pub fn options(&self) -> &TreeOptions {
        &self.options
    }

    /// Fits exhaustive deterministic CART.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] for invalid shapes, options, targets, weights, or
    /// non-representable numerical results.
    pub fn fit<M: MatrixView>(
        &self,
        matrix: &M,
        targets: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<FittedRegressor> {
        self.fit_with_strategy(matrix, targets, sample_weight, &mut Exhaustive)
    }

    /// Fits with externally supplied feature and threshold candidates.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] for invalid shapes, options, targets, weights,
    /// strategy output, or non-representable numerical results.
    pub fn fit_with_strategy<M: MatrixView, S: SplitStrategy + ?Sized>(
        &self,
        matrix: &M,
        targets: &[f64],
        sample_weight: Option<&[f64]>,
        strategy: &mut S,
    ) -> Result<FittedRegressor> {
        let (sample_weight, indices) =
            prepare_regression(matrix, targets, sample_weight, &self.options)?;
        let root_weight = selected_weight(&sample_weight, &indices)?;

        let mut nodes = vec![regression_placeholder()];
        let mut raw_importances = vec![0.0; matrix.ncols()];
        let mut stack = vec![BuildTask {
            node_id: 0,
            depth: 0,
            indices,
        }];
        while let Some(task) = stack.pop() {
            let summary =
                regression_summary(&task.indices, targets, &sample_weight, self.criterion)?;
            nodes[task.node_id] = ArenaNode {
                kind: NodeKind::Leaf,
                sample_count: task.indices.len(),
                weighted_sample_count: summary.moments.weight,
                impurity: summary.impurity,
                impurity_decrease: 0.0,
                value: summary.moments.mean,
            };
            if should_stop(
                &self.options,
                task.depth,
                task.indices.len(),
                summary.impurity,
            ) {
                continue;
            }
            let context = SplitContext {
                node_id: task.node_id,
                depth: task.depth,
                sample_count: task.indices.len(),
            };
            let Some(split) = best_regression_split(
                &RegressionSplitRequest {
                    matrix,
                    targets,
                    weights: &sample_weight,
                    indices: &task.indices,
                    parent: summary,
                    root_weight,
                    criterion: self.criterion,
                    options: &self.options,
                    context,
                },
                strategy,
            )?
            else {
                continue;
            };
            let (left_indices, right_indices) =
                partition(matrix, &task.indices, split.feature, split.threshold);
            let left = nodes.len();
            nodes.push(regression_placeholder());
            let right = nodes.len();
            nodes.push(regression_placeholder());
            nodes[task.node_id].kind = NodeKind::Split {
                feature: split.feature,
                threshold: split.threshold,
                left,
                right,
            };
            nodes[task.node_id].impurity_decrease = split.gain;
            raw_importances[split.feature] = add_importance(
                raw_importances[split.feature],
                summary.moments.weight,
                split.gain,
            )?;
            stack.push(BuildTask {
                node_id: right,
                depth: task.depth + 1,
                indices: right_indices,
            });
            stack.push(BuildTask {
                node_id: left,
                depth: task.depth + 1,
                indices: left_indices,
            });
        }
        normalize_importances(&mut raw_importances)?;
        Ok(FittedRegressor {
            nodes,
            feature_importances: raw_importances,
            n_features: matrix.ncols(),
        })
    }
}

/// Fitted regression tree.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedRegressor {
    nodes: Vec<ArenaNode<f64>>,
    feature_importances: Vec<f64>,
    n_features: usize,
}

impl FittedRegressor {
    /// Immutable arena. Root index is zero.
    #[must_use]
    pub fn nodes(&self) -> &[ArenaNode<f64>] {
        &self.nodes
    }

    /// Number of input features used during fitting.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Normalized weighted impurity decrease for each feature.
    #[must_use]
    pub fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }

    /// Predicts one weighted leaf mean per row.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] when the matrix has the wrong feature count or a
    /// non-finite value.
    pub fn predict<M: MatrixView>(&self, matrix: &M) -> Result<Vec<f64>> {
        let leaves = self.apply(matrix)?;
        Ok(leaves.iter().map(|&leaf| self.nodes[leaf].value).collect())
    }

    /// Returns the leaf arena index reached by every row.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] when the matrix has the wrong feature count or a
    /// non-finite value.
    pub fn apply<M: MatrixView>(&self, matrix: &M) -> Result<Vec<usize>> {
        validate_prediction_matrix(matrix, self.n_features)?;
        Ok((0..matrix.nrows())
            .map(|row| leaf_index(&self.nodes, matrix, row))
            .collect())
    }
}

struct BuildTask {
    node_id: usize,
    depth: usize,
    indices: Vec<usize>,
}

fn classification_placeholder(class: usize, classes: usize) -> ArenaNode<ClassificationValue> {
    ArenaNode {
        kind: NodeKind::Leaf,
        sample_count: 0,
        weighted_sample_count: 0.0,
        impurity: 0.0,
        impurity_decrease: 0.0,
        value: ClassificationValue {
            predicted_class: class,
            class_weights: vec![0.0; classes],
        },
    }
}

fn regression_placeholder() -> ArenaNode<f64> {
    ArenaNode {
        kind: NodeKind::Leaf,
        sample_count: 0,
        weighted_sample_count: 0.0,
        impurity: 0.0,
        impurity_decrease: 0.0,
        value: 0.0,
    }
}

fn should_stop(options: &TreeOptions, depth: usize, samples: usize, impurity: f64) -> bool {
    options.max_depth.is_some_and(|limit| depth >= limit)
        || samples < options.min_samples_split
        || samples < options.min_samples_leaf.saturating_mul(2)
        || impurity == 0.0
}

fn add_importance(current: f64, node_weight: f64, gain: f64) -> Result<f64> {
    let importance = current + node_weight * gain;
    if importance.is_finite() {
        Ok(importance)
    } else {
        Err(Error::NumericalOverflow {
            operation: "feature importance accumulation",
        })
    }
}

fn normalize_importances(importances: &mut [f64]) -> Result<()> {
    let mut total = CompensatedSum::default();
    for &importance in importances.iter() {
        total.add(importance, "feature importance summation")?;
    }
    let total = total.total("feature importance summation")?;
    if total > 0.0 {
        for importance in importances {
            *importance /= total;
        }
    }
    Ok(())
}

fn leaf_index<T, M: MatrixView>(nodes: &[ArenaNode<T>], matrix: &M, row: usize) -> usize {
    let mut node_id = 0;
    loop {
        match nodes[node_id].kind {
            NodeKind::Leaf => return node_id,
            NodeKind::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                node_id = if matrix.get(row, feature) <= threshold {
                    left
                } else {
                    right
                };
            }
        }
    }
}

fn prepare_classification<M: MatrixView>(
    matrix: &M,
    targets: &[usize],
    supplied_weights: Option<&[f64]>,
    options: &TreeOptions,
) -> Result<(Vec<f64>, Vec<usize>, Vec<usize>)> {
    validate_training_matrix(matrix)?;
    validate_target_length(matrix.nrows(), targets.len())?;
    options.validate(matrix.ncols())?;
    let sample_weight = weights(matrix.nrows(), supplied_weights)?;
    let indices = positive_weight_indices(&sample_weight);
    let mut classes: Vec<usize> = indices.iter().map(|&row| targets[row]).collect();
    classes.sort_unstable();
    classes.dedup();
    Ok((sample_weight, indices, classes))
}

fn prepare_regression<M: MatrixView>(
    matrix: &M,
    targets: &[f64],
    supplied_weights: Option<&[f64]>,
    options: &TreeOptions,
) -> Result<(Vec<f64>, Vec<usize>)> {
    validate_training_matrix(matrix)?;
    validate_targets(matrix.nrows(), targets)?;
    options.validate(matrix.ncols())?;
    let sample_weight = weights(matrix.nrows(), supplied_weights)?;
    let indices = positive_weight_indices(&sample_weight);
    Ok((sample_weight, indices))
}

fn positive_weight_indices(sample_weight: &[f64]) -> Vec<usize> {
    sample_weight
        .iter()
        .enumerate()
        .filter_map(|(row, &weight)| (weight > 0.0).then_some(row))
        .collect()
}

fn selected_weight(sample_weight: &[f64], indices: &[usize]) -> Result<f64> {
    let mut total = CompensatedSum::default();
    for &row in indices {
        total.add(sample_weight[row], "root sample-weight summation")?;
    }
    total.total("root sample-weight summation")
}
