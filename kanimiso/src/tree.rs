//! Kanimiso adapters for the standalone tree crates.
//!
//! [`oldwood`] owns the deterministic CART kernel. [`mayoi_no_mori`] owns
//! resampling, random feature proposals, boosting, and isolation trees. This
//! module only translates Kanimiso's [`Fit`], [`Predict`], quality-report, and
//! floating-label conventions; it contains no split-search or tree-growth
//! implementation.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::{Fit, FitUnsupervised, Predict};
use crate::validate::{inspect_classes, inspect_xy};
use mayoi_no_mori::{
    AdaBoostR2Options, BoostingOptions, FeatureSampling, ForestOptions, IsolationForestOptions,
    SammeOptions,
};
use ojizou_san::Session;
use signlred::{slice_stats, Issue, IssueCode, Meaninglessness, Qualified, Result};

fn oldwood_issue(error: &oldwood::Error) -> IssueCode {
    match error {
        oldwood::Error::EmptyTrainingData | oldwood::Error::EmptyFeatures => IssueCode::EmptyMatrix,
        oldwood::Error::TargetLength { .. }
        | oldwood::Error::WeightLength { .. }
        | oldwood::Error::FeatureCount { .. }
        | oldwood::Error::InvalidMatrixStorage { .. } => IssueCode::DimensionMismatch,
        oldwood::Error::NonFiniteFeature { .. } | oldwood::Error::NonFiniteTarget { .. } => {
            IssueCode::NonFiniteInput
        }
        oldwood::Error::InvalidWeight { .. } | oldwood::Error::NoPositiveWeight => {
            IssueCode::InvalidWeight
        }
        oldwood::Error::NumericalOverflow { .. } => IssueCode::NumericalOverflow,
        _ => IssueCode::InvalidParameter,
    }
}

fn mayoi_issue(error: &mayoi_no_mori::Error) -> IssueCode {
    match error {
        mayoi_no_mori::Error::Tree(error) => oldwood_issue(error),
        mayoi_no_mori::Error::EmptyTrainingData | mayoi_no_mori::Error::EmptyFeatures => {
            IssueCode::EmptyMatrix
        }
        mayoi_no_mori::Error::Length { .. } | mayoi_no_mori::Error::FeatureCount { .. } => {
            IssueCode::DimensionMismatch
        }
        mayoi_no_mori::Error::NonFinite { .. } => IssueCode::NonFiniteInput,
        mayoi_no_mori::Error::NegativeWeight { .. } | mayoi_no_mori::Error::NoPositiveWeight => {
            IssueCode::InvalidWeight
        }
        mayoi_no_mori::Error::NumericalOverflow { .. } => IssueCode::NumericalOverflow,
        _ => IssueCode::InvalidParameter,
    }
}

#[allow(clippy::result_large_err)] // Kanimiso's quality-preserving Result owns a complete Report.
fn fail<T>(mut ctx: FitCtx, code: IssueCode, message: impl Into<String>) -> Result<Qualified<T>> {
    ctx.push(Issue::builder(code).message(message).build());
    Err(ctx.finish_failure())
}

#[allow(clippy::result_large_err)] // Adapter must retain Kanimiso's public failure contract.
fn fail_oldwood<T>(ctx: FitCtx, operation: &str, error: oldwood::Error) -> Result<Qualified<T>> {
    let code = oldwood_issue(&error);
    fail(ctx, code, format!("{operation}: {error}"))
}

#[allow(clippy::result_large_err)] // Adapter must retain Kanimiso's public failure contract.
pub(crate) fn fail_mayoi<T>(
    ctx: FitCtx,
    operation: &str,
    error: mayoi_no_mori::Error,
) -> Result<Qualified<T>> {
    let code = mayoi_issue(&error);
    fail(ctx, code, format!("{operation}: {error}"))
}

fn tree_options(
    max_depth: usize,
    min_samples_split: usize,
    max_features: Option<usize>,
) -> oldwood::TreeOptions {
    oldwood::TreeOptions {
        max_depth: Some(max_depth),
        min_samples_split,
        max_features,
        ..oldwood::TreeOptions::default()
    }
}

fn ensemble_tree_options(max_depth: usize, min_samples_split: usize) -> oldwood::TreeOptions {
    tree_options(max_depth, min_samples_split, None)
}

pub(crate) fn encoded_labels(ctx: &mut FitCtx, y: &Vector) -> (Vec<i64>, Vec<usize>) {
    let labels: Vec<Option<i64>> = y
        .as_slice()
        .iter()
        .enumerate()
        .map(|(row, &value)| checked_label(ctx, row, value))
        .collect();
    let valid_labels = Vector::from_iter(labels.iter().flatten().map(|&label| label as f64));
    let classes: Vec<i64> = inspect_classes(&mut ctx.report, &valid_labels, &ctx.policy)
        .into_iter()
        .map(|(label, _)| label)
        .collect();
    let encoded = labels
        .iter()
        .map(|label| {
            label.map_or(0, |label| {
                classes
                    .binary_search(&label)
                    .expect("classes were constructed from every valid label")
            })
        })
        .collect();
    (classes, encoded)
}

fn checked_label(ctx: &mut FitCtx, row: usize, value: f64) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    let minimum = i64::MIN as f64;
    let maximum_exclusive = -minimum;
    if value != rounded || rounded < minimum || rounded >= maximum_exclusive {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message(format!(
                    "classification label at row {row} is {value}; labels must be integer-valued and representable as i64"
                ))
                .build(),
        );
        None
    } else {
        Some(rounded as i64)
    }
}

fn decoded(labels: &[usize], classes: &[i64]) -> core::result::Result<Vector, usize> {
    labels
        .iter()
        .map(|&label| {
            classes
                .get(label)
                .copied()
                .map(|class| class as f64)
                .ok_or(label)
        })
        .collect::<core::result::Result<Vec<_>, _>>()
        .map(Vector::from_iter)
}

pub(crate) fn finish_decoded(
    ctx: FitCtx,
    labels: &[usize],
    classes: &[i64],
    operation: &str,
) -> Result<Qualified<Vector>> {
    match decoded(labels, classes) {
        Ok(values) => ctx.finish(values),
        Err(label) => fail(
            ctx,
            IssueCode::NonFiniteOutput,
            format!("{operation}: estimator returned unknown class index {label}"),
        ),
    }
}

fn diagnose_constant_predictions(ctx: &mut FitCtx, prediction: &Vector, target: &Vector) {
    let prediction_stats = slice_stats(prediction.as_slice());
    let target_stats = slice_stats(target.as_slice());
    if prediction_stats.is_constant(ctx.policy.near_zero_variance)
        && !target_stats.is_constant(ctx.policy.near_zero_variance)
    {
        ctx.push(
            Issue::builder(IssueCode::PredictionsAreConstant)
                .message("in-sample predictions are constant while the target is not")
                .meaninglessness(Meaninglessness::vacuous(
                    "tree / ensemble predictor",
                    "the fitted rule collapsed to one label or value",
                    "increase depth, relax leaf constraints, or collect separable features",
                ))
                .build(),
        );
    }
}

#[allow(clippy::result_large_err)] // A failed nested prediction retains its complete report.
pub(crate) fn finish_with_prediction_diagnostic<F>(
    mut ctx: FitCtx,
    fitted: F,
    x: &Matrix,
    target: &Vector,
) -> Result<Qualified<F>>
where
    F: Predict<Output = Vector>,
{
    match fitted.predict(x, &ctx.session.child("in_sample_diagnostic")) {
        Ok(prediction) => {
            ctx.report.merge(prediction.report);
            diagnose_constant_predictions(&mut ctx, &prediction.value, target);
            ctx.finish(fitted)
        }
        Err(failure) => Err(ctx.merge_failure(failure)),
    }
}

fn feature_sampling(value: Option<usize>, fallback: FeatureSampling) -> FeatureSampling {
    value.map_or(fallback, FeatureSampling::Count)
}

/// Deterministic CART classifier backed by [`oldwood`].
#[derive(Clone, Debug)]
pub struct DecisionTreeClassifier {
    /// Maximum tree depth (root is depth zero).
    pub max_depth: usize,
    /// Minimum positive-weight rows required before attempting a split.
    pub min_samples_split: usize,
    /// Number of leading features considered by deterministic CART.
    pub max_features: Option<usize>,
    /// Compatibility field; exhaustive CART does not consume randomness.
    pub seed: u64,
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl DecisionTreeClassifier {
    /// Returns the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Kanimiso classifier delegating traversal to [`oldwood`].
#[derive(Clone, Debug)]
pub struct FittedTreeClassifier {
    inner: oldwood::FittedClassifier,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Training feature count.
    pub n_features: usize,
}

impl FittedTreeClassifier {
    /// Returns the class-probability vector for one row.
    ///
    /// # Errors
    ///
    /// Returns a quality failure for an invalid matrix, an out-of-range row,
    /// or an inconsistent class-probability shape.
    pub fn predict_proba_row(
        &self,
        x: &Matrix,
        row: usize,
        session: &Session,
    ) -> Result<Qualified<Vec<f64>>> {
        let ctx = FitCtx::with_session(session.child("predict_proba_row"));
        if row >= x.nrows() {
            return fail(
                ctx,
                IssueCode::DimensionMismatch,
                format!("probability row {row} is outside {} input rows", x.nrows()),
            );
        }
        match self.inner.predict_proba(x) {
            Ok(probabilities) => {
                let width = self.classes.len();
                let start = row.saturating_mul(width);
                match probabilities
                    .as_slice()
                    .get(start..start.saturating_add(width))
                {
                    Some(values) => ctx.finish(values.to_vec()),
                    None => fail(
                        ctx,
                        IssueCode::NonFiniteOutput,
                        "CART probability output has an inconsistent shape",
                    ),
                }
            }
            Err(error) => fail_oldwood(ctx, "CART probability prediction", error),
        }
    }
}

impl Predict for FittedTreeClassifier {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        match self.inner.predict(x) {
            Ok(labels) => finish_decoded(ctx, &labels, &self.classes, "CART prediction"),
            Err(error) => fail_oldwood(ctx, "CART classification prediction", error),
        }
    }
}

impl Fit for DecisionTreeClassifier {
    type Fitted = FittedTreeClassifier;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let (classes, target) = encoded_labels(&mut ctx, y);
        let model = oldwood::DecisionTreeClassifier::new(
            oldwood::ClassificationCriterion::Gini,
            tree_options(self.max_depth, self.min_samples_split, self.max_features),
        );
        match model.fit(x, &target, None) {
            Ok(inner) => finish_with_prediction_diagnostic(
                ctx,
                FittedTreeClassifier {
                    inner,
                    classes,
                    n_features: x.ncols(),
                },
                x,
                y,
            ),
            Err(error) => fail_oldwood(ctx, "CART classification fit", error),
        }
    }
}

/// Deterministic squared-error CART regressor backed by [`oldwood`].
#[derive(Clone, Debug)]
pub struct DecisionTreeRegressor {
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum positive-weight rows required before attempting a split.
    pub min_samples_split: usize,
    /// Number of leading features considered by deterministic CART.
    pub max_features: Option<usize>,
    /// Compatibility field; exhaustive CART does not consume randomness.
    pub seed: u64,
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl DecisionTreeRegressor {
    /// Returns the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Kanimiso regressor delegating traversal to [`oldwood`].
#[derive(Clone, Debug)]
pub struct FittedTreeRegressor {
    inner: oldwood::FittedRegressor,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedTreeRegressor {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        match self.inner.predict(x) {
            Ok(values) => ctx.finish(Vector::from_iter(values)),
            Err(error) => fail_oldwood(ctx, "CART regression prediction", error),
        }
    }
}

impl Fit for DecisionTreeRegressor {
    type Fitted = FittedTreeRegressor;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let model = oldwood::DecisionTreeRegressor::new(
            oldwood::RegressionCriterion::SquaredError,
            tree_options(self.max_depth, self.min_samples_split, self.max_features),
        );
        match model.fit(x, y.as_slice(), None) {
            Ok(inner) => finish_with_prediction_diagnostic(
                ctx,
                FittedTreeRegressor {
                    inner,
                    n_features: x.ncols(),
                },
                x,
                y,
            ),
            Err(error) => fail_oldwood(ctx, "CART regression fit", error),
        }
    }
}

/// Bootstrap random-forest classifier backed by [`mayoi_no_mori`].
#[derive(Clone, Debug)]
pub struct RandomForestClassifier {
    /// Number of trees.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum rows required to split.
    pub min_samples_split: usize,
    /// Node-local feature sample size; `None` means square root.
    pub max_features: Option<usize>,
    /// Reproducible ChaCha8 seed.
    pub seed: u64,
}

impl Default for RandomForestClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 20,
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl RandomForestClassifier {
    /// Returns the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted random-forest or ExtraTrees classifier.
#[derive(Clone, Debug)]
pub struct FittedForestClassifier {
    inner: mayoi_no_mori::FittedForestClassifier,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedForestClassifier {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        match self.inner.predict(x) {
            Ok(labels) => finish_decoded(ctx, &labels, &self.classes, "forest prediction"),
            Err(error) => fail_mayoi(ctx, "forest classification prediction", error),
        }
    }
}

fn classifier_forest_options(model: &RandomForestClassifier, bootstrap: bool) -> ForestOptions {
    ForestOptions {
        trees: model.n_estimators,
        bootstrap,
        sample_fraction: 1.0,
        feature_sampling: feature_sampling(model.max_features, FeatureSampling::SquareRoot),
        seed: model.seed,
        out_of_bag: false,
        tree: ensemble_tree_options(model.max_depth, model.min_samples_split),
    }
}

impl Fit for RandomForestClassifier {
    type Fitted = FittedForestClassifier;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let (classes, target) = encoded_labels(&mut ctx, y);
        let model = mayoi_no_mori::RandomForestClassifier::new(
            classifier_forest_options(self, true),
            oldwood::ClassificationCriterion::Gini,
        );
        match model.fit(x, &target, None) {
            Ok(inner) => finish_with_prediction_diagnostic(
                ctx,
                FittedForestClassifier {
                    inner,
                    classes,
                    n_features: x.ncols(),
                },
                x,
                y,
            ),
            Err(error) => fail_mayoi(ctx, "random-forest classification fit", error),
        }
    }
}

/// Bootstrap random-forest regressor backed by [`mayoi_no_mori`].
#[derive(Clone, Debug)]
pub struct RandomForestRegressor {
    /// Number of trees.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum rows required to split.
    pub min_samples_split: usize,
    /// Node-local feature sample size; `None` means all features.
    pub max_features: Option<usize>,
    /// Reproducible ChaCha8 seed.
    pub seed: u64,
}

impl Default for RandomForestRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 20,
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl RandomForestRegressor {
    /// Returns the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted random-forest or ExtraTrees regressor.
#[derive(Clone, Debug)]
pub struct FittedForestRegressor {
    inner: mayoi_no_mori::FittedForestRegressor,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedForestRegressor {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        match self.inner.predict(x) {
            Ok(values) => ctx.finish(Vector::from_iter(values)),
            Err(error) => fail_mayoi(ctx, "forest regression prediction", error),
        }
    }
}

fn regressor_forest_options(model: &RandomForestRegressor, bootstrap: bool) -> ForestOptions {
    ForestOptions {
        trees: model.n_estimators,
        bootstrap,
        sample_fraction: 1.0,
        feature_sampling: feature_sampling(model.max_features, FeatureSampling::All),
        seed: model.seed,
        out_of_bag: false,
        tree: ensemble_tree_options(model.max_depth, model.min_samples_split),
    }
}

impl Fit for RandomForestRegressor {
    type Fitted = FittedForestRegressor;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let model = mayoi_no_mori::RandomForestRegressor::new(regressor_forest_options(self, true));
        match model.fit(x, y.as_slice(), None) {
            Ok(inner) => finish_with_prediction_diagnostic(
                ctx,
                FittedForestRegressor {
                    inner,
                    n_features: x.ncols(),
                },
                x,
                y,
            ),
            Err(error) => fail_mayoi(ctx, "random-forest regression fit", error),
        }
    }
}

/// Extremely randomized classifier using one random threshold per feature.
#[derive(Clone, Debug)]
pub struct ExtraTreesClassifier {
    /// Number of trees.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum rows required to split.
    pub min_samples_split: usize,
    /// Node-local feature sample size; `None` means square root.
    pub max_features: Option<usize>,
    /// Reproducible ChaCha8 seed.
    pub seed: u64,
}

impl Default for ExtraTreesClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 20,
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl ExtraTreesClassifier {
    /// Returns the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for ExtraTreesClassifier {
    type Fitted = FittedForestClassifier;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let (classes, target) = encoded_labels(&mut ctx, y);
        let compatibility = RandomForestClassifier {
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            max_features: self.max_features,
            seed: self.seed,
        };
        let model = mayoi_no_mori::ExtraTreesClassifier::new(
            classifier_forest_options(&compatibility, false),
            oldwood::ClassificationCriterion::Gini,
        );
        match model.fit(x, &target, None) {
            Ok(inner) => finish_with_prediction_diagnostic(
                ctx,
                FittedForestClassifier {
                    inner,
                    classes,
                    n_features: x.ncols(),
                },
                x,
                y,
            ),
            Err(error) => fail_mayoi(ctx, "ExtraTrees classification fit", error),
        }
    }
}

/// One extremely randomized classification tree.
#[derive(Clone, Debug)]
pub struct ExtraTreeClassifier {
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum rows required to split.
    pub min_samples_split: usize,
    /// Node-local feature sample size; `None` means square root.
    pub max_features: Option<usize>,
    /// Reproducible ChaCha8 seed.
    pub seed: u64,
}

impl Default for ExtraTreeClassifier {
    fn default() -> Self {
        let forest = ExtraTreesClassifier::default();
        Self {
            max_depth: forest.max_depth,
            min_samples_split: forest.min_samples_split,
            max_features: forest.max_features,
            seed: forest.seed,
        }
    }
}

impl Fit for ExtraTreeClassifier {
    type Fitted = FittedForestClassifier;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        ExtraTreesClassifier {
            n_estimators: 1,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            max_features: self.max_features,
            seed: self.seed,
        }
        .fit(x, y, session)
    }
}

/// Extremely randomized regressor using one random threshold per feature.
#[derive(Clone, Debug)]
pub struct ExtraTreesRegressor {
    /// Number of trees.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum rows required to split.
    pub min_samples_split: usize,
    /// Node-local feature sample size; `None` means all features.
    pub max_features: Option<usize>,
    /// Reproducible ChaCha8 seed.
    pub seed: u64,
}

impl Default for ExtraTreesRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 20,
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl ExtraTreesRegressor {
    /// Returns the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for ExtraTreesRegressor {
    type Fitted = FittedForestRegressor;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let compatibility = RandomForestRegressor {
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            max_features: self.max_features,
            seed: self.seed,
        };
        let model = mayoi_no_mori::ExtraTreesRegressor::new(regressor_forest_options(
            &compatibility,
            false,
        ));
        match model.fit(x, y.as_slice(), None) {
            Ok(inner) => finish_with_prediction_diagnostic(
                ctx,
                FittedForestRegressor {
                    inner,
                    n_features: x.ncols(),
                },
                x,
                y,
            ),
            Err(error) => fail_mayoi(ctx, "ExtraTrees regression fit", error),
        }
    }
}

/// One extremely randomized regression tree.
#[derive(Clone, Debug)]
pub struct ExtraTreeRegressor {
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum rows required to split.
    pub min_samples_split: usize,
    /// Node-local feature sample size; `None` means all features.
    pub max_features: Option<usize>,
    /// Reproducible ChaCha8 seed.
    pub seed: u64,
}

impl Default for ExtraTreeRegressor {
    fn default() -> Self {
        let forest = ExtraTreesRegressor::default();
        Self {
            max_depth: forest.max_depth,
            min_samples_split: forest.min_samples_split,
            max_features: forest.max_features,
            seed: forest.seed,
        }
    }
}

impl Fit for ExtraTreeRegressor {
    type Fitted = FittedForestRegressor;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        ExtraTreesRegressor {
            n_estimators: 1,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            max_features: self.max_features,
            seed: self.seed,
        }
        .fit(x, y, session)
    }
}

/// Squared-error gradient-boosted regression trees.
#[derive(Clone, Debug)]
pub struct GradientBoostingRegressor {
    /// Number of additive stages.
    pub n_estimators: usize,
    /// Positive shrinkage.
    pub learning_rate: f64,
    /// Maximum CART depth.
    pub max_depth: usize,
    /// Minimum rows required to split.
    pub min_samples_split: usize,
    /// Reproducible ChaCha8 seed.
    pub seed: u64,
}

impl Default for GradientBoostingRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 30,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            seed: 0,
        }
    }
}

impl GradientBoostingRegressor {
    /// Returns the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted first-order gradient-boosting regressor.
#[derive(Clone, Debug)]
pub struct FittedGbr {
    inner: mayoi_no_mori::FittedGradientBoostingRegressor,
    /// Weighted training-target mean.
    pub intercept: f64,
    /// Shrinkage used at fit time.
    pub learning_rate: f64,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedGbr {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        match self.inner.predict(x) {
            Ok(values) => ctx.finish(Vector::from_iter(values)),
            Err(error) => fail_mayoi(ctx, "gradient-boosting regression prediction", error),
        }
    }
}

fn boosting_options(
    iterations: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_split: usize,
    seed: u64,
) -> BoostingOptions {
    BoostingOptions {
        iterations,
        learning_rate,
        sample_fraction: 1.0,
        seed,
        tree: ensemble_tree_options(max_depth, min_samples_split),
    }
}

impl Fit for GradientBoostingRegressor {
    type Fitted = FittedGbr;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let model = mayoi_no_mori::GradientBoostingRegressor::new(boosting_options(
            self.n_estimators,
            self.learning_rate,
            self.max_depth,
            self.min_samples_split,
            self.seed,
        ));
        match model.fit(x, y.as_slice(), None) {
            Ok(inner) => {
                let intercept = inner.base_value();
                finish_with_prediction_diagnostic(
                    ctx,
                    FittedGbr {
                        inner,
                        intercept,
                        learning_rate: self.learning_rate,
                        n_features: x.ncols(),
                    },
                    x,
                    y,
                )
            }
            Err(error) => fail_mayoi(ctx, "gradient-boosting regression fit", error),
        }
    }
}

/// Multiclass softmax gradient-boosted classifier.
#[derive(Clone, Debug)]
pub struct GradientBoostingClassifier {
    /// Number of additive stages.
    pub n_estimators: usize,
    /// Positive shrinkage.
    pub learning_rate: f64,
    /// Maximum CART depth.
    pub max_depth: usize,
    /// Minimum rows required to split.
    pub min_samples_split: usize,
    /// Reproducible ChaCha8 seed.
    pub seed: u64,
}

impl Default for GradientBoostingClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 30,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            seed: 0,
        }
    }
}

impl GradientBoostingClassifier {
    /// Returns the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted multiclass gradient-boosting classifier.
#[derive(Clone, Debug)]
pub struct FittedGbc {
    inner: mayoi_no_mori::FittedGradientBoostingClassifier,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Per-class log-prior scores.
    pub intercept: Vec<f64>,
    /// Shrinkage used at fit time.
    pub learning_rate: f64,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedGbc {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        match self.inner.predict(x) {
            Ok(labels) => finish_decoded(
                ctx,
                &labels,
                &self.classes,
                "gradient-boosting classification prediction",
            ),
            Err(error) => fail_mayoi(ctx, "gradient-boosting classification prediction", error),
        }
    }
}

impl Fit for GradientBoostingClassifier {
    type Fitted = FittedGbc;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let (classes, target) = encoded_labels(&mut ctx, y);
        let counts = classes
            .iter()
            .map(|class| {
                y.as_slice()
                    .iter()
                    .filter(|value| value.is_finite() && value.round() as i64 == *class)
                    .count()
            })
            .collect::<Vec<_>>();
        let total = counts.iter().sum::<usize>().max(1) as f64;
        let intercept = counts
            .iter()
            .map(|&count| ((count.max(1) as f64) / total).ln())
            .collect();
        let model = mayoi_no_mori::GradientBoostingClassifier::new(boosting_options(
            self.n_estimators,
            self.learning_rate,
            self.max_depth,
            self.min_samples_split,
            self.seed,
        ));
        match model.fit(x, &target, None) {
            Ok(inner) => finish_with_prediction_diagnostic(
                ctx,
                FittedGbc {
                    inner,
                    classes,
                    intercept,
                    learning_rate: self.learning_rate,
                    n_features: x.ncols(),
                },
                x,
                y,
            ),
            Err(error) => fail_mayoi(ctx, "gradient-boosting classification fit", error),
        }
    }
}

/// AdaBoost label-update scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaBoostAlgorithm {
    /// Discrete multiclass SAMME.
    Samme,
    /// Unsupported legacy selector; the former implementation was not conformant.
    SammeR,
}

/// Discrete SAMME AdaBoost classifier.
#[derive(Clone, Debug)]
pub struct AdaBoostClassifier {
    /// Maximum number of weak learners.
    pub n_estimators: usize,
    /// Positive shrinkage.
    pub learning_rate: f64,
    /// Maximum weak-tree depth.
    pub max_depth: usize,
    /// Only [`AdaBoostAlgorithm::Samme`] is accepted.
    pub algorithm: AdaBoostAlgorithm,
    /// Compatibility field; SAMME uses deterministic weighted CART.
    pub seed: u64,
}

impl Default for AdaBoostClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 30,
            learning_rate: 1.0,
            max_depth: 1,
            algorithm: AdaBoostAlgorithm::Samme,
            seed: 0,
        }
    }
}

impl AdaBoostClassifier {
    /// Returns the verified discrete-SAMME defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted discrete-SAMME classifier.
#[derive(Clone, Debug)]
pub struct FittedAdaBoost {
    inner: mayoi_no_mori::FittedAdaBoostClassifier,
    /// Finite SAMME stage weights.
    pub alphas: Vec<f64>,
    /// Algorithm used at fit time.
    pub algorithm: AdaBoostAlgorithm,
    /// Shrinkage used at fit time.
    pub learning_rate: f64,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedAdaBoost {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        match self.inner.predict(x) {
            Ok(labels) => finish_decoded(ctx, &labels, &self.classes, "SAMME prediction"),
            Err(error) => fail_mayoi(ctx, "SAMME prediction", error),
        }
    }
}

impl Fit for AdaBoostClassifier {
    type Fitted = FittedAdaBoost;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if self.algorithm == AdaBoostAlgorithm::SammeR {
            return fail(
                ctx,
                IssueCode::InvalidParameter,
                "SAMME.R was removed because the legacy update was not specification-conformant; use discrete SAMME",
            );
        }
        let (classes, target) = encoded_labels(&mut ctx, y);
        let model = mayoi_no_mori::AdaBoostClassifier::new(
            SammeOptions {
                estimators: self.n_estimators,
                learning_rate: self.learning_rate,
                tree: tree_options(self.max_depth, 2, None),
            },
            oldwood::ClassificationCriterion::Gini,
        );
        match model.fit(x, &target, None) {
            Ok(inner) => {
                let alphas = inner.estimator_weights().to_vec();
                finish_with_prediction_diagnostic(
                    ctx,
                    FittedAdaBoost {
                        inner,
                        alphas,
                        algorithm: AdaBoostAlgorithm::Samme,
                        learning_rate: self.learning_rate,
                        classes,
                        n_features: x.ncols(),
                    },
                    x,
                    y,
                )
            }
            Err(error) => fail_mayoi(ctx, "SAMME fit", error),
        }
    }
}

/// AdaBoost.R2 regressor.
#[derive(Clone, Debug)]
pub struct AdaBoostRegressor {
    /// Maximum number of weak learners.
    pub n_estimators: usize,
    /// Positive shrinkage.
    pub learning_rate: f64,
    /// Maximum weak-tree depth.
    pub max_depth: usize,
    /// Reproducible ChaCha8 seed for weighted bootstrap sampling.
    pub seed: u64,
}

impl Default for AdaBoostRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 30,
            learning_rate: 1.0,
            max_depth: 3,
            seed: 0,
        }
    }
}

impl AdaBoostRegressor {
    /// Returns the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted AdaBoost.R2 model.
#[derive(Clone, Debug)]
pub struct FittedAdaBoostRegressor {
    inner: mayoi_no_mori::FittedAdaBoostRegressor,
    /// Positive weighted-median stage weights.
    pub alphas: Vec<f64>,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedAdaBoostRegressor {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        match self.inner.predict(x) {
            Ok(values) => ctx.finish(Vector::from_iter(values)),
            Err(error) => fail_mayoi(ctx, "AdaBoost.R2 prediction", error),
        }
    }
}

impl Fit for AdaBoostRegressor {
    type Fitted = FittedAdaBoostRegressor;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let model = mayoi_no_mori::AdaBoostRegressor::new(AdaBoostR2Options {
            estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            seed: self.seed,
            tree: tree_options(self.max_depth, 2, None),
        });
        match model.fit(x, y.as_slice(), None) {
            Ok(inner) => {
                let alphas = inner.estimator_weights().to_vec();
                finish_with_prediction_diagnostic(
                    ctx,
                    FittedAdaBoostRegressor {
                        inner,
                        alphas,
                        n_features: x.ncols(),
                    },
                    x,
                    y,
                )
            }
            Err(error) => fail_mayoi(ctx, "AdaBoost.R2 fit", error),
        }
    }
}

/// Isolation Forest anomaly scorer backed by [`mayoi_no_mori`].
#[derive(Clone, Debug)]
pub struct IsolationForest {
    /// Number of isolation trees.
    pub n_trees: usize,
    /// Reproducible ChaCha8 seed.
    pub seed: u64,
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self {
            n_trees: 50,
            seed: 0,
        }
    }
}

impl IsolationForest {
    /// Returns the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Isolation Forest.
#[derive(Clone, Debug)]
pub struct FittedIsolationForest {
    inner: mayoi_no_mori::FittedIsolationForest,
    /// Actual subsample size used by every tree.
    pub max_samples: usize,
    /// Training feature count.
    pub n_features: usize,
    /// Isolation normalizer `c(max_samples)`.
    pub c_norm: f64,
}

impl FittedIsolationForest {
    /// Returns the mean adjusted path length for one row.
    ///
    /// # Errors
    ///
    /// Returns a quality failure for an incompatible matrix, non-finite input,
    /// or an out-of-range row.
    pub fn average_path_length(
        &self,
        x: &Matrix,
        row: usize,
        session: &Session,
    ) -> Result<Qualified<f64>> {
        let ctx = FitCtx::with_session(session.child("average_path_length"));
        match self.inner.average_path_length(x, row) {
            Ok(value) => ctx.finish(value),
            Err(error) => fail_mayoi(ctx, "Isolation Forest path length", error),
        }
    }

    /// Returns one isolation score per row.
    ///
    /// # Errors
    ///
    /// Returns a quality failure for an incompatible matrix or non-finite input.
    pub fn score_samples(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("score_samples"));
        match self.inner.score_samples(x) {
            Ok(values) => ctx.finish(Vector::from_iter(values)),
            Err(error) => fail_mayoi(ctx, "Isolation Forest scoring", error),
        }
    }
}

impl Predict for FittedIsolationForest {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.score_samples(x, &session.child("predict"))
    }
}

impl FitUnsupervised for IsolationForest {
    type Fitted = FittedIsolationForest;

    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let model = mayoi_no_mori::IsolationForest::new(IsolationForestOptions {
            trees: self.n_trees,
            max_samples: 256,
            seed: self.seed,
        });
        match model.fit(x) {
            Ok(inner) => {
                let max_samples = inner.max_samples();
                let c_norm = inner.normalizer();
                ctx.finish(FittedIsolationForest {
                    inner,
                    max_samples,
                    n_features: x.ncols(),
                    c_norm,
                })
            }
            Err(error) => fail_mayoi(ctx, "Isolation Forest fit", error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(name: &str) -> Session {
        Session::new(name, "fit")
    }

    #[test]
    fn cart_adapter_matches_a_separable_case() {
        let x = Matrix::from_row_major(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y = Vector::from_iter([-4.0, -4.0, 9.0, 9.0]);
        let fitted = DecisionTreeClassifier::new()
            .fit(&x, &y, &session("cart_adapter"))
            .expect("fit")
            .value;
        assert_eq!(fitted.classes, vec![-4, 9]);
        assert_eq!(
            fitted
                .predict(&x, &session("cart_predict"))
                .expect("predict")
                .value,
            y
        );
    }

    #[test]
    fn depth_zero_cart_aborts_when_nonconstant_target_collapses() {
        let x = Matrix::from_row_major(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y = Vector::from_iter([-2.0, -1.0, 1.0, 2.0]);
        let failure = DecisionTreeRegressor {
            max_depth: 0,
            ..DecisionTreeRegressor::default()
        }
        .fit(&x, &y, &session("cart_depth_zero"))
        .expect_err("a depth-zero rule must not hide a vacuous fit");
        assert_eq!(failure.primary.code, IssueCode::PredictionsAreConstant);
        assert!(failure.report.contains(IssueCode::PredictionsAreConstant));
    }

    #[test]
    fn classifier_rejects_label_at_positive_i64_boundary() {
        let x = Matrix::from_row_major(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y = Vector::from_iter([0.0, 1.0, 9_223_372_036_854_775_808.0, 1.0]);
        let failure = DecisionTreeClassifier::new()
            .fit(&x, &y, &session("cart_i64_boundary"))
            .expect_err("2^63 is not representable as i64");
        assert_eq!(failure.primary.code, IssueCode::InvalidParameter);
        assert!(failure.report.contains(IssueCode::InvalidParameter));
    }

    #[test]
    fn forest_adapter_replays_a_seed() {
        let x = Matrix::from_row_major(6, 1, &[0.0, 0.5, 1.0, 2.0, 2.5, 3.0]);
        let y = Vector::from_iter([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let mut first = RandomForestClassifier {
            n_estimators: 12,
            seed: 42,
            ..RandomForestClassifier::default()
        };
        let mut second = first.clone();
        let a = first.fit(&x, &y, &session("rf_a")).expect("first").value;
        let b = second.fit(&x, &y, &session("rf_b")).expect("second").value;
        assert_eq!(
            a.predict(&x, &session("rf_ap")).expect("predict a").value,
            b.predict(&x, &session("rf_bp")).expect("predict b").value
        );
    }

    #[test]
    fn legacy_samme_r_selector_is_rejected_explicitly() {
        let x = Matrix::from_row_major(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y = Vector::from_iter([0.0, 0.0, 1.0, 1.0]);
        let failure = AdaBoostClassifier {
            algorithm: AdaBoostAlgorithm::SammeR,
            ..AdaBoostClassifier::default()
        }
        .fit(&x, &y, &session("samme_r"))
        .expect_err("unsupported legacy selector");
        assert_eq!(failure.primary.code, IssueCode::InvalidParameter);
    }

    #[test]
    fn isolation_scoring_rejects_shape_and_non_finite_inputs() {
        let training = Matrix::from_row_major(
            8,
            2,
            &[
                0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 1.5, 1.5, 2.0, 1.0, 2.5, 2.0, 3.0, 1.5, 3.5, 3.0,
            ],
        );
        let fitted = IsolationForest {
            n_trees: 8,
            seed: 17,
        }
        .fit_unsupervised(&training, &session("isolation_fit"))
        .expect("fit")
        .value;

        let wrong_shape = Matrix::from_row_major(2, 1, &[0.0, 1.0]);
        let shape_failure = fitted
            .score_samples(&wrong_shape, &session("isolation_shape"))
            .expect_err("feature-count mismatch must be a quality failure");
        assert_eq!(shape_failure.primary.code, IssueCode::DimensionMismatch);

        let non_finite = Matrix::from_row_major(1, 2, &[f64::NAN, 0.0]);
        let finite_failure = fitted
            .score_samples(&non_finite, &session("isolation_non_finite"))
            .expect_err("non-finite scoring input must not become a NaN score");
        assert_eq!(finite_failure.primary.code, IssueCode::NonFiniteInput);
    }
}
