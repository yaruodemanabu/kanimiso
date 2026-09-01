//! CART / forest / boosting wrappers around [`oldwood`] and [`mayoi_no_mori`].
//!
//! Numeric grow lives in those crates on [`faer`]. This module only opens a
//! [`crate::context::FitCtx`] so `signlred` diagnoses empty/constant targets
//! and `ojizou-san` records the session. A silent successful fit is a bug.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::{Fit, FitUnsupervised, Predict, Transform};
use crate::validate::{inspect_classes, inspect_xy};
use mayoi_no_mori::{
    fit_adaboost, fit_adaboost_r2, fit_embedding, fit_gbc, fit_gbr, fit_isolation,
    grow_forest_class, grow_forest_reg, AdaBoostR2Stop, AdaBoostSpec, AdaBoostStop, BoostStop,
    EmbeddingSpec, FittedAdaBoost as MoriAdaBoost, FittedAdaBoostR2, FittedEmbedding,
    FittedGbc as MoriGbc, FittedGbr as MoriGbr, FittedIsolation, ForestClassifier, ForestRegressor,
    ForestSpec, GbcSpec, GbrSpec, IsolationSpec,
};
use ojizou_san::Session;
use oldwood::{
    grow_class, grow_reg, is_class_stump, predict_class_labels, predict_class_proba, predict_reg,
    ClassNode, GrowSpec, RegNode, Rng,
};
use signlred::{Issue, IssueCode, Meaninglessness, Qualified, Result};

/// Average unsuccessful BST path length \(c(n)\) (Liu et al.).
///
/// Re-exported from [`oldwood`] so `anomaly` and `coverage` keep the same path.
pub(crate) use oldwood::isolation_c_factor;

/// AdaBoost label-update scheme.
pub(crate) use mayoi_no_mori::AdaBoostAlgorithm;

fn diagnose_constant_predictions(ctx: &mut FitCtx, pred: &Vector, y: &Vector) {
    let pst = signlred::slice_stats(pred.as_slice());
    let yst = signlred::slice_stats(y.as_slice());
    if pst.is_constant(ctx.policy.near_zero_variance)
        && !yst.is_constant(ctx.policy.near_zero_variance)
    {
        ctx.push(
            Issue::builder(IssueCode::PredictionsAreConstant)
                .message("in-sample predictions are a constant while y is not")
                .meaninglessness(Meaninglessness::vacuous(
                    "tree / ensemble predictor",
                    "the fitted rule collapsed to a single label or value",
                    "increase depth, lower min_samples_split, or collect a separable sample",
                ))
                .build(),
        );
    }
}

fn labels_of(y: &Vector) -> Vec<i64> {
    y.as_slice()
        .iter()
        .map(|&v| if v.is_finite() { v.round() as i64 } else { 0 })
        .collect()
}

fn unit_weights(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

fn cart_spec(
    max_depth: usize,
    min_samples_split: usize,
    max_features: Option<usize>,
    extra: bool,
    sqrt_features: bool,
    eps: f64,
) -> GrowSpec {
    GrowSpec {
        max_depth,
        min_samples_split,
        max_features,
        extra,
        sqrt_features,
        eps,
    }
}

fn predict_shape_guard(ctx: &mut FitCtx, x: &Matrix, n_features: usize) {
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    if x.ncols() != n_features {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "predict X is n×{} but the model was fit on {} features",
                    x.ncols(),
                    n_features
                ))
                .build(),
        );
    }
}

fn labels_to_vector(labs: &[i64]) -> Vector {
    Vector::from_iter(labs.iter().map(|&c| c as f64))
}

/// CART classifier using Gini impurity.
#[derive(Clone, Debug)]
pub(crate) struct DecisionTreeClassifier {
    /// Maximum tree depth (root is depth 0; `0` yields a stump leaf).
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size at each node (`None` = all features).
    pub max_features: Option<usize>,
    /// PRNG seed.
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
    /// Default Gini tree (`max_depth = 8`).
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Fitted CART classifier.
#[derive(Clone, Debug)]
pub(crate) struct FittedTreeClassifier {
    root: ClassNode,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Training feature count.
    pub n_features: usize,
}

impl FittedTreeClassifier {
    /// Class-probability vector for row `i` of `x` (aligned with [`Self::classes`]).
    pub(crate) fn predict_proba_row(&self, x: &Matrix, i: usize) -> Vec<f64> {
        predict_class_proba(&self.root, x.inner(), i, self.classes.len())
    }
}

impl Predict for FittedTreeClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(labels_to_vector(&predict_class_labels(
            &self.root,
            x.inner(),
        )))
    }
}

impl Fit for DecisionTreeClassifier {
    type Fitted = FittedTreeClassifier;
    fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTreeClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let ylab = labels_of(y);
        let idx: Vec<usize> = (0..x.nrows()).collect();
        let w = unit_weights(x.nrows());
        let mut rng = Rng::new(self.seed);
        let spec = cart_spec(
            self.max_depth,
            self.min_samples_split,
            self.max_features,
            false,
            false,
            ctx.policy.near_zero_variance,
        );
        let root = if classes.is_empty() {
            ClassNode::Leaf {
                class: 0,
                counts: Vec::new(),
            }
        } else {
            grow_class(x.inner(), &ylab, &classes, &idx, &w, &spec, &mut rng)
        };
        let fitted = FittedTreeClassifier {
            root,
            classes,
            n_features: x.ncols(),
        };
        if !is_class_stump(&fitted.root) {
            ctx.session.converged("CART Gini split found", 0);
        } else if counts.len() > 1 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("Gini CART produced a single leaf on a multi-class target")
                    .meaninglessness(Meaninglessness::vacuous(
                        "decision tree",
                        "no split reduced Gini; the classifier is a constant",
                        "check feature variation and hyperparameters",
                    ))
                    .build(),
            );
        }
        let pred = labels_to_vector(&predict_class_labels(&fitted.root, x.inner()));
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// CART regressor using mean-squared-error impurity.
#[derive(Clone, Debug)]
pub(crate) struct DecisionTreeRegressor {
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size at each node (`None` = all features).
    pub max_features: Option<usize>,
    /// PRNG seed.
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
    /// Default MSE tree.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Fitted CART regressor.
#[derive(Clone, Debug)]
pub(crate) struct FittedTreeRegressor {
    root: RegNode,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedTreeRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(Vector::from_iter(predict_reg(&self.root, x.inner())))
    }
}

impl Fit for DecisionTreeRegressor {
    type Fitted = FittedTreeRegressor;
    fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTreeRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let ys = y.as_slice().to_vec();
        let idx: Vec<usize> = (0..x.nrows()).collect();
        let w = unit_weights(x.nrows());
        let mut rng = Rng::new(self.seed);
        let spec = cart_spec(
            self.max_depth,
            self.min_samples_split,
            self.max_features,
            false,
            false,
            ctx.policy.near_zero_variance,
        );
        let root = grow_reg(x.inner(), &ys, &idx, &w, &spec, &mut rng);
        let fitted = FittedTreeRegressor {
            root,
            n_features: x.ncols(),
        };
        let pred = Vector::from_iter(predict_reg(&fitted.root, x.inner()));
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Bootstrap-aggregated Gini trees with per-node feature subsample.
#[derive(Clone, Debug)]
pub(crate) struct RandomForestClassifier {
    /// Number of trees.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` ⇒ \(\sqrt{p}\)).
    pub max_features: Option<usize>,
    /// PRNG seed.
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
    /// Default random forest classifier.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Fitted classification forest (random forest or extra-trees).
#[derive(Clone, Debug)]
pub(crate) struct FittedForestClassifier {
    inner: ForestClassifier,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedForestClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(labels_to_vector(&self.inner.predict_labels(x.inner())))
    }
}

fn forest_class_spec(
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    max_features: Option<usize>,
    seed: u64,
    extra: bool,
    bootstrap: bool,
    eps: f64,
) -> ForestSpec {
    ForestSpec {
        n_estimators,
        grow: cart_spec(
            max_depth,
            min_samples_split,
            max_features,
            extra,
            !extra,
            eps,
        ),
        bootstrap,
        seed,
    }
}

impl Fit for RandomForestClassifier {
    type Fitted = FittedForestClassifier;
    fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForestClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let ylab = labels_of(y);
        let spec = forest_class_spec(
            self.n_estimators.max(1),
            self.max_depth,
            self.min_samples_split,
            self.max_features,
            self.seed,
            false,
            true,
            ctx.policy.near_zero_variance,
        );
        let inner = grow_forest_class(x.inner(), &ylab, &classes, &spec);
        if !inner.trees.is_empty() {
            ctx.session.converged(
                format!("{} trees grown", inner.trees.len()),
                inner.trees.len() as u64,
            );
        }
        let pred = labels_to_vector(&inner.predict_labels(x.inner()));
        let fitted = FittedForestClassifier {
            classes: inner.classes.clone(),
            n_features: inner.n_features,
            inner,
        };
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Bootstrap-aggregated MSE trees with per-node feature subsample.
#[derive(Clone, Debug)]
pub(crate) struct RandomForestRegressor {
    /// Number of trees.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` ⇒ all features).
    pub max_features: Option<usize>,
    /// PRNG seed.
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
    /// Default random forest regressor.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Fitted regression forest.
#[derive(Clone, Debug)]
pub(crate) struct FittedForestRegressor {
    inner: ForestRegressor,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedForestRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(Vector::from_iter(self.inner.predict(x.inner())))
    }
}

impl Fit for RandomForestRegressor {
    type Fitted = FittedForestRegressor;
    fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForestRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let spec = ForestSpec {
            n_estimators: self.n_estimators.max(1),
            grow: cart_spec(
                self.max_depth,
                self.min_samples_split,
                self.max_features,
                false,
                false,
                ctx.policy.near_zero_variance,
            ),
            bootstrap: true,
            seed: self.seed,
        };
        let inner = grow_forest_reg(x.inner(), y.as_slice(), &spec);
        if !inner.trees.is_empty() {
            ctx.session.converged(
                format!("{} trees grown", inner.trees.len()),
                inner.trees.len() as u64,
            );
        }
        let pred = Vector::from_iter(inner.predict(x.inner()));
        let fitted = FittedForestRegressor {
            n_features: inner.n_features,
            inner,
        };
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Extremely randomized Gini trees (random thresholds, full sample).
#[derive(Clone, Debug)]
pub(crate) struct ExtraTreesClassifier {
    /// Number of trees.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` ⇒ \(\sqrt{p}\)).
    pub max_features: Option<usize>,
    /// PRNG seed.
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
    /// Default extra-trees classifier.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl Fit for ExtraTreesClassifier {
    type Fitted = FittedForestClassifier;
    fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForestClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let ylab = labels_of(y);
        let spec = forest_class_spec(
            self.n_estimators.max(1),
            self.max_depth,
            self.min_samples_split,
            self.max_features,
            self.seed,
            true,
            false,
            ctx.policy.near_zero_variance,
        );
        let inner = grow_forest_class(x.inner(), &ylab, &classes, &spec);
        if !inner.trees.is_empty() {
            ctx.session.converged(
                format!("{} extra-trees grown", inner.trees.len()),
                inner.trees.len() as u64,
            );
        }
        let pred = labels_to_vector(&inner.predict_labels(x.inner()));
        let fitted = FittedForestClassifier {
            classes: inner.classes.clone(),
            n_features: inner.n_features,
            inner,
        };
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Single extremely randomized Gini tree (sklearn `ExtraTreeClassifier`).
#[derive(Clone, Debug)]
pub(crate) struct ExtraTreeClassifier {
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` ⇒ \(\sqrt{p}\)).
    pub max_features: Option<usize>,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for ExtraTreeClassifier {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl ExtraTreeClassifier {
    /// Default single extra-tree classifier.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl Fit for ExtraTreeClassifier {
    type Fitted = FittedForestClassifier;
    fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForestClassifier>> {
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

/// Extremely randomized MSE trees (random thresholds, full sample).
#[derive(Clone, Debug)]
pub(crate) struct ExtraTreesRegressor {
    /// Number of trees.
    pub n_estimators: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` ⇒ all features).
    pub max_features: Option<usize>,
    /// PRNG seed.
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
    /// Default extra-trees regressor.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl Fit for ExtraTreesRegressor {
    type Fitted = FittedForestRegressor;
    fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForestRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let spec = ForestSpec {
            n_estimators: self.n_estimators.max(1),
            grow: cart_spec(
                self.max_depth,
                self.min_samples_split,
                self.max_features,
                true,
                false,
                ctx.policy.near_zero_variance,
            ),
            bootstrap: false,
            seed: self.seed,
        };
        let inner = grow_forest_reg(x.inner(), y.as_slice(), &spec);
        if !inner.trees.is_empty() {
            ctx.session.converged(
                format!("{} extra-trees grown", inner.trees.len()),
                inner.trees.len() as u64,
            );
        }
        let pred = Vector::from_iter(inner.predict(x.inner()));
        let fitted = FittedForestRegressor {
            n_features: inner.n_features,
            inner,
        };
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Single extremely randomized MSE tree (sklearn `ExtraTreeRegressor`).
#[derive(Clone, Debug)]
pub(crate) struct ExtraTreeRegressor {
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Feature subsample size (`None` ⇒ all features).
    pub max_features: Option<usize>,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for ExtraTreeRegressor {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_samples_split: 2,
            max_features: None,
            seed: 0,
        }
    }
}

impl ExtraTreeRegressor {
    /// Default single extra-tree regressor.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl Fit for ExtraTreeRegressor {
    type Fitted = FittedForestRegressor;
    fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForestRegressor>> {
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

/// Friedman gradient boosting for squared error.
#[derive(Clone, Debug)]
pub(crate) struct GradientBoostingRegressor {
    /// Number of sequential trees.
    pub n_estimators: usize,
    /// Shrinkage \(\nu\).
    pub learning_rate: f64,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// PRNG seed.
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
    /// Default squared-error gradient booster.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Fitted squared-error gradient booster.
#[derive(Clone, Debug)]
pub(crate) struct FittedGbr {
    inner: MoriGbr,
    /// Initial constant (training mean).
    pub intercept: f64,
    /// Shrinkage used at fit time.
    pub learning_rate: f64,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedGbr {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(Vector::from_iter(self.inner.predict(x.inner())))
    }
}

impl Fit for GradientBoostingRegressor {
    type Fitted = FittedGbr;
    fn fit(&self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedGbr>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let spec = GbrSpec {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            grow: cart_spec(
                self.max_depth,
                self.min_samples_split,
                None,
                false,
                false,
                ctx.policy.near_zero_variance,
            ),
            seed: self.seed,
        };
        let inner = fit_gbr(x.inner(), y.as_slice(), &spec);
        match &inner.stop {
            BoostStop::InvalidLearningRate => {
                ctx.push(
                    Issue::builder(IssueCode::InvalidWeight)
                        .message(format!(
                            "learning_rate={} is not a positive finite number",
                            self.learning_rate
                        ))
                        .build(),
                );
            }
            BoostStop::Finished {
                stages,
                residuals_vanished,
            } => {
                if *residuals_vanished {
                    ctx.session
                        .converged("boosting residuals vanished", *stages as u64);
                } else {
                    ctx.session
                        .converged("finished squared-error boosting stages", *stages as u64);
                }
            }
        }
        let pred = Vector::from_iter(inner.predict(x.inner()));
        let fitted = FittedGbr {
            intercept: inner.intercept,
            learning_rate: inner.learning_rate,
            n_features: inner.n_features,
            inner,
        };
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Friedman gradient boosting for binomial / multinomial log-loss.
#[derive(Clone, Debug)]
pub(crate) struct GradientBoostingClassifier {
    /// Number of sequential stages.
    pub n_estimators: usize,
    /// Shrinkage \(\nu\).
    pub learning_rate: f64,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// PRNG seed.
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
    /// Default log-loss gradient booster.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Fitted log-loss gradient booster.
#[derive(Clone, Debug)]
pub(crate) struct FittedGbc {
    inner: MoriGbc,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Per-class initial scores (log-odds / zero-mean log-prior).
    pub intercept: Vec<f64>,
    /// Shrinkage used at fit time.
    pub learning_rate: f64,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedGbc {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(labels_to_vector(&self.inner.predict_labels(x.inner())))
    }
}

impl Fit for GradientBoostingClassifier {
    type Fitted = FittedGbc;
    fn fit(&self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedGbc>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let ylab = labels_of(y);
        let spec = GbcSpec {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            grow: cart_spec(
                self.max_depth,
                self.min_samples_split,
                None,
                false,
                false,
                ctx.policy.near_zero_variance,
            ),
            seed: self.seed,
        };
        let inner = fit_gbc(x.inner(), &ylab, &classes, &spec);
        ctx.session.converged(
            "finished log-loss boosting stages",
            inner.trees.len() as u64,
        );
        let pred = labels_to_vector(&inner.predict_labels(x.inner()));
        let fitted = FittedGbc {
            classes: inner.classes.clone(),
            intercept: inner.intercept.clone(),
            learning_rate: inner.learning_rate,
            n_features: inner.n_features,
            inner,
        };
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// SAMME / SAMME.R AdaBoost classifier.
#[derive(Clone, Debug)]
pub(crate) struct AdaBoostClassifier {
    /// Number of weak learners.
    pub n_estimators: usize,
    /// Shrinkage on the additive model.
    pub learning_rate: f64,
    /// Weak-learner depth (SAMME needs ≥ 2 for XOR-like problems).
    pub max_depth: usize,
    /// `SAMME` or `SAMME.R`.
    pub algorithm: AdaBoostAlgorithm,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for AdaBoostClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 30,
            learning_rate: 1.0,
            max_depth: 2,
            algorithm: AdaBoostAlgorithm::SammeR,
            seed: 0,
        }
    }
}

impl AdaBoostClassifier {
    /// Default SAMME.R AdaBoost.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Fitted AdaBoost model.
#[derive(Clone, Debug)]
pub(crate) struct FittedAdaBoost {
    inner: MoriAdaBoost,
    /// SAMME weights (`α_m`); empty when using SAMME.R.
    pub alphas: Vec<f64>,
    /// Algorithm used at fit time.
    pub algorithm: AdaBoostAlgorithm,
    /// Shrinkage.
    pub learning_rate: f64,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedAdaBoost {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(labels_to_vector(&self.inner.predict_labels(x.inner())))
    }
}

impl Fit for AdaBoostClassifier {
    type Fitted = FittedAdaBoost;
    fn fit(&self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedAdaBoost>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let ylab = labels_of(y);
        let spec = AdaBoostSpec {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            grow: cart_spec(
                self.max_depth,
                2,
                None,
                false,
                false,
                ctx.policy.near_zero_variance,
            ),
            algorithm: self.algorithm,
            seed: self.seed,
        };
        let inner = fit_adaboost(x.inner(), &ylab, &classes, &spec);
        match &inner.stop {
            AdaBoostStop::WeakNotBetterThanChance { stage, err } => {
                ctx.push(
                    Issue::builder(IssueCode::MeaninglessFit)
                        .message(format!(
                            "SAMME weak learner {stage} is not better than random (err={err:.4})"
                        ))
                        .meaninglessness(Meaninglessness::vacuous(
                            "AdaBoost SAMME stage",
                            "the weighted error is at or worse than chance; α is undefined or non-positive",
                            "use deeper trees or a linearly separable sample",
                        ))
                        .build(),
                );
            }
            AdaBoostStop::Empty => {
                if classes.len() >= 2 {
                    ctx.push(
                        Issue::builder(IssueCode::UnidentifiedModel)
                            .message("AdaBoost produced no usable weak learners")
                            .build(),
                    );
                }
            }
            AdaBoostStop::Finished { stages } => {
                ctx.session
                    .converged(format!("{stages} AdaBoost stages"), *stages as u64);
            }
        }
        let pred = labels_to_vector(&inner.predict_labels(x.inner()));
        let fitted = FittedAdaBoost {
            alphas: inner.alphas.clone(),
            algorithm: inner.algorithm,
            learning_rate: inner.learning_rate,
            classes: inner.classes.clone(),
            n_features: inner.n_features,
            inner,
        };
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// AdaBoost.R2 regressor (Drucker 1997).
#[derive(Clone, Debug)]
pub(crate) struct AdaBoostRegressor {
    /// Number of weak learners.
    pub n_estimators: usize,
    /// Shrinkage on \(\ln(1/\beta)\).
    pub learning_rate: f64,
    /// Weak-learner depth.
    pub max_depth: usize,
    /// PRNG seed.
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
    /// Default AdaBoost.R2.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Fitted AdaBoost.R2 model.
#[derive(Clone, Debug)]
pub(crate) struct FittedAdaBoostRegressor {
    inner: FittedAdaBoostR2,
    /// Stage weights \(\ln(1/\beta_m)\).
    pub alphas: Vec<f64>,
    /// Training feature count.
    pub n_features: usize,
}

impl Predict for FittedAdaBoostRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(Vector::from_iter(self.inner.predict(x.inner())))
    }
}

impl Fit for AdaBoostRegressor {
    type Fitted = FittedAdaBoostRegressor;
    fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedAdaBoostRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if x.nrows() == 0 || ctx.report.contains(IssueCode::ConstantTarget) {
            return ctx.finish(FittedAdaBoostRegressor {
                inner: FittedAdaBoostR2 {
                    trees: Vec::new(),
                    alphas: Vec::new(),
                    n_features: x.ncols(),
                    stop: AdaBoostR2Stop::Empty,
                },
                alphas: Vec::new(),
                n_features: x.ncols(),
            });
        }
        let spec = mayoi_no_mori::AdaBoostR2Spec {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            grow: cart_spec(
                self.max_depth,
                2,
                None,
                false,
                false,
                ctx.policy.near_zero_variance,
            ),
            seed: self.seed,
        };
        let inner = fit_adaboost_r2(
            x.inner(),
            y.as_slice(),
            &spec,
            ctx.policy.near_zero_variance,
        );
        match &inner.stop {
            AdaBoostR2Stop::WeightedLossGeHalf {
                stage,
                loss,
                had_prior,
            } => {
                if *had_prior {
                    ctx.push(
                        Issue::builder(IssueCode::DidNotConverge)
                            .message(format!(
                                "AdaBoost.R2 stopped at stage {stage}: weighted loss {loss:.4} ≥ 1/2"
                            ))
                            .build(),
                    );
                } else {
                    ctx.push(
                        Issue::builder(IssueCode::MeaninglessFit)
                            .message(format!(
                                "AdaBoost.R2 stage {stage} has weighted loss {loss:.4} ≥ 1/2; β is undefined"
                            ))
                            .meaninglessness(Meaninglessness::vacuous(
                                "AdaBoost.R2 stage weight",
                                "the weak learner is not better than the median-absolute-error null",
                                "use deeper trees or a smoother target",
                            ))
                            .build(),
                    );
                }
            }
            AdaBoostR2Stop::Empty => {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("AdaBoost.R2 produced no usable weak learners")
                        .build(),
                );
            }
            AdaBoostR2Stop::Finished { stages } => {
                ctx.session
                    .converged(format!("{stages} AdaBoost.R2 stages"), *stages as u64);
            }
        }
        let pred = Vector::from_iter(inner.predict(x.inner()));
        let fitted = FittedAdaBoostRegressor {
            alphas: inner.alphas.clone(),
            n_features: inner.n_features,
            inner,
        };
        diagnose_constant_predictions(&mut ctx, &pred, y);
        ctx.finish(fitted)
    }
}

/// Isolation Forest (Liu, Ting, Zhou): random-split path-length anomaly scores.
///
/// The later `anomaly` module reuses [`FittedIsolationForest`] scores.
#[derive(Clone, Debug)]
pub(crate) struct IsolationForest {
    /// Number of isolation trees.
    pub n_trees: usize,
    /// PRNG seed.
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
    /// Isolation forest with `n_trees` trees.
    pub(crate) fn new(n_trees: usize) -> Self {
        Self { n_trees, seed: 0 }
    }
}

/// Fitted isolation forest.
#[derive(Clone, Debug)]
pub(crate) struct FittedIsolationForest {
    inner: FittedIsolation,
    /// Subsample size used to grow each tree (and in \(c(n)\)).
    pub max_samples: usize,
    /// Training feature count.
    pub n_features: usize,
    /// \(c(\texttt{max_samples})\).
    pub c_norm: f64,
}

impl FittedIsolationForest {
    /// Mean path length of row `i`.
    pub(crate) fn average_path_length(&self, x: &Matrix, i: usize) -> f64 {
        self.inner.average_path_length(x.inner(), i)
    }

    /// Liu et al. anomaly score \(s(x,n)=2^{-E(h)/c(n)}\) (higher = more anomalous).
    pub(crate) fn score_samples(&self, x: &Matrix) -> Vector {
        Vector::from_iter(self.inner.score_samples(x.inner()))
    }
}

impl Predict for FittedIsolationForest {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.n_features);
        ctx.finish(self.score_samples(x))
    }
}

impl FitUnsupervised for IsolationForest {
    type Fitted = FittedIsolationForest;
    fn fit_unsupervised(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedIsolationForest>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        if n == 0 {
            return ctx.finish(FittedIsolationForest {
                inner: FittedIsolation {
                    trees: Vec::new(),
                    max_samples: 0,
                    n_features: x.ncols(),
                    c_norm: 0.0,
                },
                max_samples: 0,
                n_features: x.ncols(),
                c_norm: 0.0,
            });
        }
        let all_constant = (0..x.ncols()).all(|j| {
            let col = x.column(j);
            signlred::slice_stats(col.as_slice()).is_constant(ctx.policy.near_zero_variance)
        });
        if all_constant {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message(
                        "Isolation Forest on a constant design: every path length is identical",
                    )
                    .meaninglessness(Meaninglessness::vacuous(
                        "isolation path lengths",
                        "no feature has variation, so random splits cannot isolate",
                        "do not rank anomalies on a point mass",
                    ))
                    .build(),
            );
        }
        let inner = fit_isolation(
            x.inner(),
            &IsolationSpec {
                n_trees: self.n_trees,
                seed: self.seed,
            },
        );
        ctx.session.converged(
            format!("{} isolation trees", inner.trees.len()),
            inner.trees.len() as u64,
        );
        ctx.finish(FittedIsolationForest {
            max_samples: inner.max_samples,
            n_features: inner.n_features,
            c_norm: inner.c_norm,
            inner,
        })
    }
}

/// Completely-random tree leaf embedding (sklearn `RandomTreesEmbedding`).
#[derive(Clone, Debug)]
pub(crate) struct RandomTreesEmbedding {
    /// Number of random trees.
    pub n_estimators: usize,
    /// Hashed leaf-code width.
    pub n_components: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for RandomTreesEmbedding {
    fn default() -> Self {
        Self {
            n_estimators: 8,
            n_components: 8,
            seed: 0,
        }
    }
}

impl RandomTreesEmbedding {
    /// Embedding with `n_estimators` trees and `n_components` hash bins.
    pub(crate) fn new(n_estimators: usize, n_components: usize) -> Self {
        Self {
            n_estimators,
            n_components,
            ..Self::default()
        }
    }
}

/// Fitted random-tree leaf embedding.
#[derive(Clone, Debug)]
pub(crate) struct FittedRandomTreesEmbedding {
    inner: FittedEmbedding,
    n_components: usize,
    n_features: usize,
}

impl FitUnsupervised for RandomTreesEmbedding {
    type Fitted = FittedRandomTreesEmbedding;
    fn fit_unsupervised(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedRandomTreesEmbedding>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n_comp = self.n_components.max(1);
        if x.nrows() == 0 || x.ncols() == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .severity(signlred::Severity::Warning)
                    .message("RandomTreesEmbedding received an empty design")
                    .build(),
            );
        }
        let inner = fit_embedding(
            x.inner(),
            &EmbeddingSpec {
                n_estimators: self.n_estimators,
                n_components: n_comp,
                seed: self.seed,
            },
        );
        ctx.finish(FittedRandomTreesEmbedding {
            n_components: inner.n_components,
            n_features: inner.n_features,
            inner,
        })
    }
}

impl Transform for FittedRandomTreesEmbedding {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.ncols() != self.n_features {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(signlred::Severity::Warning)
                    .message("RandomTreesEmbedding column count changed")
                    .build(),
            );
        }
        ctx.finish(Matrix::from_faer(self.inner.transform(x.inner())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use ojizou_san::Session;

    fn accuracy(pred: &Vector, y: &Vector) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        let mut ok = 0usize;
        for i in 0..y.len() {
            if (pred[i].round() - y[i].round()).abs() < 0.5 {
                ok += 1;
            }
        }
        ok as f64 / y.len() as f64
    }

    fn xor_like(n: usize, seed: u64) -> (Matrix, Vector) {
        let mut rng = Rng::new(seed);
        let mut data = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let a = if rng.uniform() < 0.5 { 0.0 } else { 1.0 };
            let b = if rng.uniform() < 0.5 { 0.0 } else { 1.0 };
            data[i * 2] = a + 0.02 * rng.standard_normal();
            data[i * 2 + 1] = b + 0.02 * rng.standard_normal();
            y[i] = if (a > 0.5) ^ (b > 0.5) { 1.0 } else { 0.0 };
        }
        (Matrix::from_row_major(n, 2, &data), Vector::from_slice(&y))
    }

    fn two_blobs(n_per: usize, seed: u64) -> (Matrix, Vector) {
        let mut rng = Rng::new(seed);
        let n = n_per * 2;
        let mut data = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let c = if i < n_per { 0.0 } else { 4.0 };
            data[i * 2] = c + 0.2 * rng.standard_normal();
            data[i * 2 + 1] = c + 0.2 * rng.standard_normal();
            y[i] = if i < n_per { 0.0 } else { 1.0 };
        }
        (Matrix::from_row_major(n, 2, &data), Vector::from_slice(&y))
    }

    #[test]
    fn cart_xor_accuracy() {
        let (x, y) = xor_like(48, 3);
        let session = Session::new("decision_tree_classifier", "fit");
        let q = DecisionTreeClassifier {
            max_depth: 4,
            ..DecisionTreeClassifier::default()
        }
        .fit(&x, &y, &session)
        .expect("tree");
        let pred = q
            .value
            .predict(&x, &Session::new("decision_tree_classifier", "predict"))
            .expect("pred")
            .value;
        assert!(accuracy(&pred, &y) > 0.8, "acc={}", accuracy(&pred, &y));
    }

    #[test]
    fn forest_two_blob_accuracy() {
        let (x, y) = two_blobs(20, 9);
        let session = Session::new("random_forest", "fit");
        let q = RandomForestClassifier {
            n_estimators: 12,
            seed: 2,
            ..RandomForestClassifier::default()
        }
        .fit(&x, &y, &session)
        .expect("rf");
        let pred = q
            .value
            .predict(&x, &Session::new("random_forest", "predict"))
            .expect("pred")
            .value;
        assert!(accuracy(&pred, &y) > 0.8, "acc={}", accuracy(&pred, &y));
    }

    #[test]
    fn extra_trees_and_boosters_two_blob() {
        let (x, y) = two_blobs(18, 4);
        let et = ExtraTreesClassifier {
            n_estimators: 16,
            seed: 5,
            ..ExtraTreesClassifier::default()
        }
        .fit(&x, &y, &Session::new("extra_trees", "fit"))
        .expect("et");
        let p = et
            .value
            .predict(&x, &Session::new("extra_trees", "predict"))
            .unwrap()
            .value;
        assert!(accuracy(&p, &y) > 0.8);

        let etc = ExtraTreeClassifier {
            seed: 5,
            ..ExtraTreeClassifier::default()
        }
        .fit(&x, &y, &Session::new("extra_tree", "fit"))
        .expect("etc");
        let p1 = etc
            .value
            .predict(&x, &Session::new("extra_tree", "predict"))
            .unwrap()
            .value;
        assert_eq!(p1.len(), y.len());

        let gbc = GradientBoostingClassifier {
            n_estimators: 20,
            learning_rate: 0.2,
            max_depth: 2,
            seed: 1,
            ..GradientBoostingClassifier::default()
        }
        .fit(&x, &y, &Session::new("gbc", "fit"))
        .expect("gbc");
        let p = gbc
            .value
            .predict(&x, &Session::new("gbc", "predict"))
            .unwrap()
            .value;
        assert!(accuracy(&p, &y) > 0.8);

        let ada = AdaBoostClassifier {
            n_estimators: 20,
            algorithm: AdaBoostAlgorithm::Samme,
            max_depth: 2,
            seed: 1,
            ..AdaBoostClassifier::default()
        }
        .fit(&x, &y, &Session::new("adaboost", "fit"))
        .expect("ada");
        let p = ada
            .value
            .predict(&x, &Session::new("adaboost", "predict"))
            .unwrap()
            .value;
        assert!(accuracy(&p, &y) > 0.8);
    }

    #[test]
    fn constant_y_errors() {
        let x = Matrix::from_fn(8, 2, |i, j| (i + j) as f64);
        let y = Vector::filled(8, 3.0);
        let err = DecisionTreeClassifier::new()
            .fit(&x, &y, &Session::new("tree", "fit"))
            .unwrap_err();
        assert!(
            err.primary().code == IssueCode::ConstantTarget
                || err.primary().code == IssueCode::SingleClass
        );
        assert!(
            err.report.contains(IssueCode::ConstantTarget)
                || err.report.contains(IssueCode::SingleClass)
        );
        let err = DecisionTreeRegressor::new()
            .fit(&x, &y, &Session::new("tree_reg", "fit"))
            .unwrap_err();
        assert_eq!(err.primary().code, IssueCode::ConstantTarget);
    }

    #[test]
    fn isolation_scores_far_point() {
        let (x, _) = two_blobs(16, 1);
        let session = Session::new("iforest", "fit");
        let q = IsolationForest {
            n_trees: 20,
            seed: 7,
        }
        .fit_unsupervised(&x, &session)
        .expect("if");
        let mut far = Matrix::zeros(1, 2);
        far.set(0, 0, 40.0);
        far.set(0, 1, -40.0);
        let s_in = q
            .value
            .predict(&x, &Session::new("iforest", "predict"))
            .unwrap()
            .value;
        let s_out = q
            .value
            .predict(&far, &Session::new("iforest", "predict"))
            .unwrap()
            .value;
        let mean_in = s_in.mean();
        assert!(
            s_out[0] > mean_in,
            "outlier score {} vs inlier mean {}",
            s_out[0],
            mean_in
        );
        let emb = RandomTreesEmbedding::new(4, 6)
            .fit_unsupervised(&x, &Session::new("rte", "fit"))
            .expect("rte");
        let z = emb
            .value
            .transform(&x, &Session::new("rte", "t"))
            .unwrap()
            .value;
        assert_eq!(z.nrows(), x.nrows());
        assert_eq!(z.ncols(), 6);
        assert!(z.get(0, 0).is_finite());
    }

    #[test]
    fn gbr_fits_line() {
        let x = Matrix::from_fn(12, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..12).map(|i| 0.5 * i as f64));
        let q = GradientBoostingRegressor {
            n_estimators: 25,
            learning_rate: 0.2,
            max_depth: 2,
            ..GradientBoostingRegressor::default()
        }
        .fit(&x, &y, &Session::new("gbr", "fit"))
        .expect("gbr");
        let etr = ExtraTreeRegressor {
            seed: 3,
            ..ExtraTreeRegressor::default()
        }
        .fit(&x, &y, &Session::new("etr", "fit"))
        .expect("etr");
        let etp = etr
            .value
            .predict(&x, &Session::new("etr", "predict"))
            .unwrap()
            .value;
        assert_eq!(etp.len(), y.len());
        let pred = q
            .value
            .predict(&x, &Session::new("gbr", "predict"))
            .unwrap()
            .value;
        let mut sse = 0.0;
        for i in 0..y.len() {
            let e = pred[i] - y[i];
            sse += e * e;
        }
        assert!(
            sse / (y.len() as f64) < 0.5,
            "mse={}",
            sse / (y.len() as f64)
        );
    }

    #[test]
    fn adaboost_r2_fits_a_line() {
        let x = Matrix::from_fn(16, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..16).map(|i| 0.4 * i as f64 + 0.1 * ((i % 3) as f64)));
        let q = AdaBoostRegressor {
            n_estimators: 25,
            max_depth: 4,
            ..AdaBoostRegressor::default()
        }
        .fit(&x, &y, &Session::new("abr", "fit"))
        .expect("abr");
        let pred = q
            .value
            .predict(&x, &Session::new("abr", "p"))
            .unwrap()
            .value;
        let mut sse = 0.0;
        for i in 0..y.len() {
            let e = pred[i] - y[i];
            sse += e * e;
        }
        assert!(
            sse / (y.len() as f64) < 1.0,
            "mse={}",
            sse / (y.len() as f64)
        );
    }
}
