//! Compatibility adapters for `mayoi-no-mori`'s histogram/Newton trees.
//!
//! The histogram builder and Newton leaf solver live exclusively in
//! [`mayoi_no_mori::LightGbmRegressor`] and
//! [`mayoi_no_mori::LightGbmClassifier`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::{Fit, Predict};
use crate::tree::{encoded_labels, fail_mayoi, finish_decoded, finish_with_prediction_diagnostic};
use crate::validate::inspect_xy_allow_missing_features;
use mayoi_no_mori::{BoostingOptions, FeatureSampling, LightGbmOptions};
use ojizou_san::Session;
use signlred::{Qualified, Result};

/// Histogram/Newton gradient-boosting regressor.
#[derive(Clone, Debug)]
pub struct HistGradientBoostingRegressor {
    /// Boosting rounds.
    pub max_iter: usize,
    /// Positive shrinkage.
    pub learning_rate: f64,
    /// Maximum depth (root is depth zero).
    pub max_depth: usize,
    /// Maximum non-missing bins per feature.
    pub max_bins: usize,
    /// Minimum rows in every CART leaf.
    pub min_samples_leaf: usize,
    /// Non-negative L2 penalty on Newton leaf values.
    pub l2: f64,
}

impl Default for HistGradientBoostingRegressor {
    fn default() -> Self {
        Self {
            max_iter: 40,
            learning_rate: 0.1,
            max_depth: 3,
            max_bins: 32,
            min_samples_leaf: 2,
            l2: 1e-6,
        }
    }
}

impl HistGradientBoostingRegressor {
    /// Returns the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Histogram/Newton gradient-boosting classifier.
#[derive(Clone, Debug)]
pub struct HistGradientBoostingClassifier {
    /// Boosting rounds.
    pub max_iter: usize,
    /// Positive shrinkage.
    pub learning_rate: f64,
    /// Maximum depth (root is depth zero).
    pub max_depth: usize,
    /// Maximum non-missing bins per feature.
    pub max_bins: usize,
    /// Minimum rows in every CART leaf.
    pub min_samples_leaf: usize,
    /// Non-negative L2 penalty on Newton leaf values.
    pub l2: f64,
}

impl Default for HistGradientBoostingClassifier {
    fn default() -> Self {
        let regression = HistGradientBoostingRegressor::default();
        Self {
            max_iter: regression.max_iter,
            learning_rate: regression.learning_rate,
            max_depth: regression.max_depth,
            max_bins: regression.max_bins,
            min_samples_leaf: regression.min_samples_leaf,
            l2: regression.l2,
        }
    }
}

impl HistGradientBoostingClassifier {
    /// Returns the documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

fn options(
    max_iter: usize,
    learning_rate: f64,
    max_depth: usize,
    max_bins: usize,
    min_samples_leaf: usize,
    l2: f64,
) -> LightGbmOptions {
    LightGbmOptions {
        boosting: BoostingOptions {
            iterations: max_iter,
            learning_rate,
            sample_fraction: 1.0,
            seed: 0,
            tree: oldwood::TreeOptions {
                max_depth: Some(max_depth),
                min_samples_leaf,
                ..oldwood::TreeOptions::default()
            },
        },
        max_bins,
        feature_sampling: FeatureSampling::All,
        l1_regularization: 0.0,
        l2_regularization: l2,
        min_hessian_leaf: 0.0,
    }
}

/// Fitted histogram/Newton regressor.
#[derive(Clone, Debug)]
pub struct FittedHistGbr {
    inner: mayoi_no_mori::FittedLightGbmRegressor,
    /// Training-target mean before the first stage.
    pub intercept: f64,
    /// Shrinkage used at prediction time.
    pub learning_rate: f64,
}

impl Predict for FittedHistGbr {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        match self.inner.predict(x) {
            Ok(values) => ctx.finish(Vector::from_iter(values)),
            Err(error) => fail_mayoi(ctx, "histogram regression prediction", error),
        }
    }
}

impl Fit for HistGradientBoostingRegressor {
    type Fitted = FittedHistGbr;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy_allow_missing_features(&mut ctx.report, x, Some(y), &ctx.policy);
        let model = mayoi_no_mori::LightGbmRegressor::new(options(
            self.max_iter,
            self.learning_rate,
            self.max_depth,
            self.max_bins,
            self.min_samples_leaf,
            self.l2,
        ));
        match model.fit(x, y.as_slice(), None) {
            Ok(inner) => {
                let intercept = inner.base_value();
                finish_with_prediction_diagnostic(
                    ctx,
                    FittedHistGbr {
                        inner,
                        intercept,
                        learning_rate: self.learning_rate,
                    },
                    x,
                    y,
                )
            }
            Err(error) => fail_mayoi(ctx, "histogram regression fit", error),
        }
    }
}

/// Fitted histogram/Newton classifier.
#[derive(Clone, Debug)]
pub struct FittedHistGbc {
    inner: mayoi_no_mori::FittedLightGbmClassifier,
    /// Per-class log-prior values before the first stage.
    pub intercepts: Vector,
    /// Shrinkage used at prediction time.
    pub learning_rate: f64,
    /// Sorted training labels.
    pub classes: Vec<i64>,
}

impl Predict for FittedHistGbc {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        match self.inner.predict(x) {
            Ok(labels) => finish_decoded(
                ctx,
                &labels,
                &self.classes,
                "histogram classification prediction",
            ),
            Err(error) => fail_mayoi(ctx, "histogram classification prediction", error),
        }
    }
}

impl Fit for HistGradientBoostingClassifier {
    type Fitted = FittedHistGbc;

    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy_allow_missing_features(&mut ctx.report, x, Some(y), &ctx.policy);
        let (classes, target) = encoded_labels(&mut ctx, y);
        let model = mayoi_no_mori::LightGbmClassifier::new(options(
            self.max_iter,
            self.learning_rate,
            self.max_depth,
            self.max_bins,
            self.min_samples_leaf,
            self.l2,
        ));
        match model.fit(x, &target, None) {
            Ok(inner) => {
                let intercepts = Vector::from_iter(inner.base_logits().iter().copied());
                finish_with_prediction_diagnostic(
                    ctx,
                    FittedHistGbc {
                        inner,
                        intercepts,
                        learning_rate: self.learning_rate,
                        classes,
                    },
                    x,
                    y,
                )
            }
            Err(error) => fail_mayoi(ctx, "histogram classification fit", error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use signlred::IssueCode;

    #[test]
    fn histogram_adapter_fits_a_three_class_problem() {
        let x = Matrix::from_fn(30, 2, |row, column| {
            (row / 10) as f64 + 0.05 * column as f64 + 0.01 * (row % 10) as f64
        });
        let y = Vector::from_iter((0..30).map(|row| (row / 10) as f64));
        let fitted = HistGradientBoostingClassifier {
            max_iter: 25,
            learning_rate: 0.2,
            max_depth: 2,
            ..HistGradientBoostingClassifier::default()
        }
        .fit(&x, &y, &Session::new("hist_adapter", "fit"))
        .expect("fit")
        .value;
        let prediction = fitted
            .predict(&x, &Session::new("hist_adapter", "predict"))
            .expect("predict")
            .value;
        let correct = prediction
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .filter(|(actual, expected)| (*actual - *expected).abs() < 0.5)
            .count();
        assert!(correct >= 24, "correct={correct}");
    }

    #[test]
    fn constant_regression_target_preserves_the_quality_abort() {
        let x = Matrix::from_fn(8, 2, |row, column| (row + column) as f64);
        let y = Vector::filled(8, 3.0);
        let failure = HistGradientBoostingRegressor::new()
            .fit(&x, &y, &Session::new("hist_constant", "fit"))
            .expect_err("constant response must abort");
        assert_eq!(failure.primary.code, IssueCode::ConstantTarget);
    }

    #[test]
    fn depth_zero_histogram_tree_reports_constant_predictions() {
        let x = Matrix::from_row_major(6, 1, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Vector::from_iter([0.0, 0.0, 1.0, 2.0, 3.0, 3.0]);
        let failure = HistGradientBoostingRegressor {
            max_iter: 2,
            max_depth: 0,
            min_samples_leaf: 1,
            ..HistGradientBoostingRegressor::default()
        }
        .fit(&x, &y, &Session::new("hist_depth_zero", "fit"))
        .expect_err("a depth-zero ensemble must expose its constant predictions");
        assert_eq!(failure.primary.code, IssueCode::PredictionsAreConstant);
        assert!(failure.report.contains(IssueCode::PredictionsAreConstant));
    }

    #[test]
    fn histogram_adapter_accepts_nan_missing_values_but_rejects_infinity() {
        let y = Vector::from_iter([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let with_missing =
            Matrix::from_row_major(8, 1, &[0.0, 0.5, 1.0, f64::NAN, 2.0, 2.5, 3.0, f64::NAN]);
        let accepted = HistGradientBoostingRegressor {
            max_iter: 4,
            max_depth: 1,
            min_samples_leaf: 1,
            ..HistGradientBoostingRegressor::default()
        }
        .fit(&with_missing, &y, &Session::new("hist_missing", "fit"))
        .expect("NaN is the supported missing-value marker");
        assert!(!accepted.report.contains(IssueCode::NonFiniteInput));

        let with_infinity =
            Matrix::from_row_major(8, 1, &[0.0, 0.5, 1.0, f64::INFINITY, 2.0, 2.5, 3.0, 3.5]);
        let failure = HistGradientBoostingRegressor {
            max_iter: 4,
            max_depth: 1,
            min_samples_leaf: 1,
            ..HistGradientBoostingRegressor::default()
        }
        .fit(&with_infinity, &y, &Session::new("hist_infinity", "fit"))
        .expect_err("infinity is not a missing-value marker");
        assert_eq!(failure.primary.code, IssueCode::NonFiniteInput);
        assert!(failure.report.contains(IssueCode::NonFiniteInput));
    }

    #[test]
    fn histogram_regression_intercept_matches_the_training_mean() {
        let x = Matrix::from_row_major(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y = Vector::from_iter([0.0, 0.0, 2.0, 2.0]);
        let fitted = HistGradientBoostingRegressor {
            max_iter: 1,
            learning_rate: 0.25,
            max_depth: 1,
            max_bins: 4,
            min_samples_leaf: 1,
            l2: 0.0,
        }
        .fit(&x, &y, &Session::new("hist_intercept", "fit"))
        .expect("fit")
        .value;
        assert_eq!(fitted.intercept, 1.0);
        assert_eq!(
            fitted
                .predict(&x, &Session::new("hist_intercept", "predict"))
                .expect("predict")
                .value,
            Vector::from_iter([0.75, 0.75, 1.25, 1.25])
        );
    }
}
