//! Outlier / novelty detectors with a common `+1` inlier / `−1` outlier label.
//!
//! Wrappers reuse [`crate::tree::IsolationForest`], [`crate::neighbors::LocalOutlierFactor`],
//! [`crate::svm::OneClassSvm`], and [`crate::covariance::EllipticEnvelope`]. Scores
//! are detector-native (higher = more anomalous); [`Predict`] always emits
//! the sklearn convention `+1 / −1`.

use crate::context::FitCtx;
use crate::covariance::{EllipticEnvelope as CovEnvelope, FittedEllipticEnvelope};
use crate::data::{Matrix, Vector};
use crate::metrics::{minkowski_distance, valid_minkowski_order};
use crate::neighbors::{FittedLof, LocalOutlierFactor as NeighLof};
use crate::online::{finish_explain, flag_info, inspect_online_xy, reject_explain};
use crate::svm::{FittedOneClassSvm, OneClassSvm};
use crate::traits::{FitUnsupervised, PartialFit, Predict};
use crate::tree::{FittedIsolationForest, IsolationForest as TreeIsolationForest};
use crate::validate::inspect_xy;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{Failure, IncrementalQuality, Issue, IssueCode, Policy, Qualified, Result};
use std::collections::VecDeque;
use std::num::NonZeroUsize;

fn quantile(mut xs: Vec<f64>, q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = q.clamp(0.0, 1.0) * (xs.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        xs[lo]
    } else {
        let t = pos - lo as f64;
        xs[lo] * (1.0 - t) + xs[hi] * t
    }
}

fn labels_from_scores(scores: &Vector, threshold: f64, higher_is_outlier: bool) -> Vector {
    Vector::from_iter(scores.as_slice().iter().map(|&s| {
        let out = if higher_is_outlier {
            s >= threshold
        } else {
            s > threshold
        };
        if out {
            -1.0
        } else {
            1.0
        }
    }))
}

/// Isolation Forest wrapper (Liu et al. path-length scores).
#[derive(Clone, Debug)]
pub struct IsolationForest {
    /// Number of isolation trees.
    pub n_trees: usize,
    /// PRNG seed.
    pub seed: u64,
    /// Expected outlier fraction used to set the decision threshold.
    pub contamination: f64,
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self {
            n_trees: 50,
            seed: 0,
            contamination: 0.1,
        }
    }
}

impl IsolationForest {
    /// Isolation forest with `n_trees` trees.
    pub fn new(n_trees: usize) -> Self {
        Self {
            n_trees,
            ..Self::default()
        }
    }
}

/// Fitted isolation forest with a contamination threshold.
#[derive(Clone, Debug)]
pub struct FittedAnomalyForest {
    inner: Option<FittedIsolationForest>,
    /// Score cutoff (higher score = more isolated).
    pub threshold: f64,
}

impl FittedAnomalyForest {
    #[allow(clippy::result_large_err)] // Preserve the full nested quality report.
    fn score_vec(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        match &self.inner {
            Some(fitted) => fitted.score_samples(x, session),
            None => {
                let mut ctx = FitCtx::with_session(session.clone());
                inspect_xy(&mut ctx.report, x, None, &ctx.policy);
                ctx.finish(Vector::zeros(x.nrows()))
            }
        }
    }

    /// Isolation scores `2^{-E(h)/c(n)}` (higher = more anomalous).
    pub fn scores(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.score_vec(x, &session.child("score"))
    }
}

impl Predict for FittedAnomalyForest {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.score_vec(x, &session.child("predict"))
            .map(|qualified| {
                qualified.map(|scores| labels_from_scores(&scores, self.threshold, true))
            })
    }
}

impl FitUnsupervised for IsolationForest {
    type Fitted = FittedAnomalyForest;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedAnomalyForest>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if !self.contamination.is_finite() || self.contamination <= 0.0 || self.contamination > 0.5
        {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!(
                        "contamination={} not in (0, 0.5]",
                        self.contamination
                    ))
                    .build(),
            );
        }
        let mut inner = TreeIsolationForest {
            n_trees: self.n_trees,
            seed: self.seed,
        };
        let fitted = match inner.fit_unsupervised(x, &session.child("iforest")) {
            Ok(q) => Some(q.value),
            Err(e) => {
                ctx.report.merge(e.report);
                None
            }
        };
        let scores = match &fitted {
            Some(fitted) => match fitted.score_samples(x, &session.child("training_scores")) {
                Ok(qualified) => {
                    ctx.report.merge(qualified.report);
                    qualified.value
                }
                Err(failure) => return Err(ctx.merge_failure(failure)),
            },
            None => Vector::zeros(x.nrows()),
        };
        let q = (1.0 - self.contamination.clamp(1e-6, 0.5)).clamp(0.5, 1.0);
        let threshold = quantile(scores.as_slice().to_vec(), q);
        ctx.finish(FittedAnomalyForest {
            inner: fitted,
            threshold,
        })
    }
}

/// Local outlier factor wrapper.
#[derive(Clone, Debug)]
pub struct LocalOutlierFactor {
    /// Neighborhood size.
    pub k: usize,
    /// Expected outlier fraction.
    pub contamination: f64,
}

impl Default for LocalOutlierFactor {
    fn default() -> Self {
        Self {
            k: 5,
            contamination: 0.1,
        }
    }
}

impl LocalOutlierFactor {
    /// LOF with `k` neighbors.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            ..Self::default()
        }
    }
}

/// Fitted LOF with a contamination threshold.
#[derive(Clone, Debug)]
pub struct FittedAnomalyLof {
    inner: FittedLof,
    /// LOF cutoff (higher = more outlying).
    pub threshold: f64,
}

impl FittedAnomalyLof {
    /// LOF scores (higher = more outlying).
    pub fn scores(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("score"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(self.inner.score_samples(x))
    }
}

impl Predict for FittedAnomalyLof {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let s = self.inner.score_samples(x);
        ctx.finish(labels_from_scores(&s, self.threshold, true))
    }
}

impl FitUnsupervised for LocalOutlierFactor {
    type Fitted = FittedAnomalyLof;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedAnomalyLof>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let mut lof = NeighLof { k: self.k };
        let fitted = match lof.fit_unsupervised(x, &session.child("lof")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedLof {
                    x_train: x.clone(),
                    k: self.k.max(1),
                    lrd: Vector::zeros(x.nrows()),
                    k_dist: Vector::zeros(x.nrows()),
                }
            }
        };
        let scores = fitted.score_samples(x);
        let q = (1.0 - self.contamination.clamp(1e-6, 0.5)).clamp(0.5, 1.0);
        let threshold = quantile(scores.as_slice().to_vec(), q);
        ctx.finish(FittedAnomalyLof {
            inner: fitted,
            threshold,
        })
    }
}

/// One-class hypersphere (SVDD-lite), wrapping [`OneClassSvm`].
#[derive(Clone, Debug)]
pub struct OneClassHypersphere {
    /// Expected outlier fraction `ν`.
    pub nu: f64,
}

impl Default for OneClassHypersphere {
    fn default() -> Self {
        Self { nu: 0.1 }
    }
}

impl OneClassHypersphere {
    /// Hypersphere with the given `ν`.
    pub fn new(nu: f64) -> Self {
        Self { nu }
    }
}

/// Fitted hypersphere.
#[derive(Clone, Debug)]
pub struct FittedHypersphere {
    inner: FittedOneClassSvm,
}

impl FittedHypersphere {
    /// Decision scores (positive ⇒ outside / outlier).
    pub fn scores(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("score"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(self.inner.score_samples(x))
    }
}

impl Predict for FittedHypersphere {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.inner.predict(x, session)
    }
}

impl FitUnsupervised for OneClassHypersphere {
    type Fitted = FittedHypersphere;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHypersphere>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let mut oc = OneClassSvm {
            nu: self.nu,
            linear: false,
            ..OneClassSvm::default()
        };
        match oc.fit_unsupervised(x, &session.child("ocsvm")) {
            Ok(q) => ctx.finish(FittedHypersphere { inner: q.value }),
            Err(e) => {
                ctx.push(e.primary);
                ctx.finish(FittedHypersphere {
                    inner: FittedOneClassSvm {
                        center: Vector::zeros(x.ncols()),
                        radius: 0.0,
                        linear: false,
                    },
                })
            }
        }
    }
}

/// Elliptic envelope (Mahalanobis from MinCovDet).
#[derive(Clone, Debug)]
pub struct EllipticEnvelope {
    inner: CovEnvelope,
}

impl Default for EllipticEnvelope {
    fn default() -> Self {
        Self {
            inner: CovEnvelope::default(),
        }
    }
}

impl EllipticEnvelope {
    /// Envelope with the given contamination.
    pub fn new(contamination: f64) -> Self {
        Self {
            inner: CovEnvelope::new(contamination),
        }
    }
}

/// Fitted elliptic envelope (re-export of the covariance fit).
pub type FittedAnomalyEnvelope = FittedEllipticEnvelope;

impl FitUnsupervised for EllipticEnvelope {
    type Fitted = FittedEllipticEnvelope;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedEllipticEnvelope>> {
        self.inner.fit_unsupervised(x, session)
    }
}

/// Bounded online k-nearest-distance anomaly scorer.
///
/// One runtime `p` replaces the former numbered Minkowski family. For finite
/// `p >= 1`, the score is the mean distance to exactly `k` retained rows under
/// the true Minkowski norm; `p = +∞` selects Chebyshev distance. When
/// `log_transform` is enabled, each coordinate is transformed with
/// `sign(x) * ln(1 + |x|)`, which is finite at zero and preserves sign.
///
/// The reference set is a FIFO window, not a random reservoir. Prediction is
/// read-only, and an invalid update leaves all state unchanged.
#[derive(Clone, Debug)]
pub struct KnnDistanceAnomaly {
    k: NonZeroUsize,
    p: f64,
    log_transform: bool,
    window: NonZeroUsize,
    rows: VecDeque<Vec<f64>>,
    n_features: Option<NonZeroUsize>,
    n_seen: u64,
    updates: u64,
    policy: Policy,
}

impl KnnDistanceAnomaly {
    /// Construct with [`Policy::default`].
    pub fn new(k: usize, p: f64, log_transform: bool, window: usize) -> Result<Self> {
        Self::with_policy(k, p, log_transform, window, Policy::default())
    }

    /// Construct with an explicit numerical-quality policy.
    pub fn with_policy(
        k: usize,
        p: f64,
        log_transform: bool,
        window: usize,
        policy: Policy,
    ) -> Result<Self> {
        let k = NonZeroUsize::new(k)
            .ok_or_else(|| Self::invalid_parameter("neighbor count k must be positive"))?;
        let window = NonZeroUsize::new(window)
            .ok_or_else(|| Self::invalid_parameter("reference window must be positive"))?;
        if window.get() < k.get() {
            return Err(Self::invalid_parameter(format!(
                "reference window {} is smaller than k={}",
                window.get(),
                k.get()
            )));
        }
        if !valid_minkowski_order(p) {
            return Err(Self::invalid_parameter(format!(
                "Minkowski order p={p}; expected finite p >= 1 or positive infinity"
            )));
        }
        Ok(Self {
            k,
            p,
            log_transform,
            window,
            // A user may choose a very large logical window. Allocate lazily so
            // construction validates parameters without attempting an enormous
            // reservation before the first observation arrives.
            rows: VecDeque::new(),
            n_features: None,
            n_seen: 0,
            updates: 0,
            policy,
        })
    }

    /// Number of nearest retained rows averaged into each score.
    pub fn k(&self) -> usize {
        self.k.get()
    }

    /// Minkowski order; positive infinity means Chebyshev distance.
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Whether the sign-preserving `ln(1 + |x|)` transform is enabled.
    pub fn log_transform(&self) -> bool {
        self.log_transform
    }

    /// Maximum number of retained reference rows.
    pub fn window(&self) -> usize {
        self.window.get()
    }

    /// Numerical-quality policy used by update and prediction.
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    fn invalid_parameter(message: impl Into<String>) -> Failure {
        Failure::from_issue(
            "KnnDistanceAnomaly",
            "new",
            Issue::builder(IssueCode::InvalidParameter)
                .message(message)
                .build(),
        )
    }

    fn transform_value(&self, value: f64) -> f64 {
        if self.log_transform {
            value.signum() * value.abs().ln_1p()
        } else {
            value
        }
    }

    fn transform_row(&self, x: &Matrix, row: usize) -> Vec<f64> {
        (0..x.ncols())
            .map(|column| self.transform_value(x.get(row, column)))
            .collect()
    }

    fn matrix_is_finite(x: &Matrix) -> bool {
        (0..x.nrows()).all(|row| (0..x.ncols()).all(|column| x.get(row, column).is_finite()))
    }

    fn score_row(&self, row: &[f64]) -> Option<f64> {
        let mut distances = self
            .rows
            .iter()
            .map(|reference| minkowski_distance(row, reference, self.p))
            .collect::<Option<Vec<_>>>()?;
        distances.sort_by(f64::total_cmp);
        let nearest = distances.get(..self.k.get())?;
        let scale = nearest.iter().copied().fold(0.0_f64, f64::max);
        if scale == 0.0 {
            return Some(0.0);
        }
        let scaled_mean =
            nearest.iter().map(|distance| distance / scale).sum::<f64>() / nearest.len() as f64;
        let score = scale * scaled_mean;
        score.is_finite().then_some(score)
    }
}

impl PartialFit for KnnDistanceAnomaly {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        ctx.policy = self.policy.clone();
        inspect_online_xy(&mut ctx, x, None);
        if x.nrows() == 0 || x.ncols() == 0 {
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "empty batch"),
            );
        }
        if !Self::matrix_is_finite(x) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteInput)
                    .message("KnnDistanceAnomaly requires a finite batch")
                    .build(),
            );
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "non-finite batch"),
            );
        }
        if self
            .n_features
            .is_some_and(|features| features.get() != x.ncols())
        {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message(format!(
                        "KnnDistanceAnomaly was initialized with {} features, received {}",
                        self.n_features.map(NonZeroUsize::get).unwrap_or_default(),
                        x.ncols()
                    ))
                    .build(),
            );
            return finish_explain(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "feature space changed",
                ),
            );
        }

        let before = self.rows.len();
        for row in 0..x.nrows() {
            if self.rows.len() == self.window.get() {
                self.rows.pop_front();
            }
            self.rows.push_back(self.transform_row(x, row));
        }
        self.n_features = NonZeroUsize::new(x.ncols());
        self.n_seen = self.n_seen.saturating_add(x.nrows() as u64);
        self.updates = self.updates.saturating_add(1);

        let identified = self.rows.len() >= self.k.get();
        let mut quality =
            IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        quality.effective_sample_size = self.rows.len() as f64;
        quality.parameter_delta_norm = None;
        quality.information_gain = Some(x.nrows() as f64);
        quality.still_identified = identified;
        quality.warmup = !identified;
        quality.explanation = format!(
            "retained {} -> {} rows in a FIFO window; k={} p={} log_transform={}",
            before,
            self.rows.len(),
            self.k,
            self.p,
            self.log_transform
        );
        flag_info(&mut ctx, &quality);
        if quality.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(quality.clone())
                    .message(format!(
                        "retained {} reference rows; k={} are required",
                        self.rows.len(),
                        self.k
                    ))
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                quality,
                format!(
                    "updated bounded k-nearest-distance reference window (k={})",
                    self.k
                ),
                "the complete finite batch passed dimensional validation",
                format!("{before} retained reference rows"),
                format!("{} retained reference rows", self.rows.len()),
            ),
        )
    }
}

impl Predict for KnnDistanceAnomaly {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        ctx.policy = self.policy.clone();
        inspect_online_xy(&mut ctx, x, None);
        if x.nrows() == 0 || x.ncols() == 0 {
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        if !Self::matrix_is_finite(x) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteInput)
                    .message("KnnDistanceAnomaly requires finite prediction input")
                    .build(),
            );
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let Some(features) = self.n_features else {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        };
        if features.get() != x.ncols() {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message(format!(
                        "KnnDistanceAnomaly expects {} features, received {}",
                        features,
                        x.ncols()
                    ))
                    .build(),
            );
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        if self.rows.len() < self.k.get() {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .message(format!(
                        "retained {} reference rows; k={} are required",
                        self.rows.len(),
                        self.k
                    ))
                    .build(),
            );
            return ctx.finish(Vector::zeros(x.nrows()));
        }

        let mut failed = false;
        let scores = Vector::from_iter((0..x.nrows()).map(|row| {
            let transformed = self.transform_row(x, row);
            self.score_row(&transformed).unwrap_or_else(|| {
                failed = true;
                0.0
            })
        }));
        if failed {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("k-nearest Minkowski scoring exceeded binary64 range")
                    .build(),
            );
        }
        ctx.finish(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    fn cloud_plus_outlier() -> Matrix {
        Matrix::from_fn(16, 2, |i, j| {
            if i == 0 {
                if j == 0 {
                    20.0
                } else {
                    -20.0
                }
            } else {
                0.1 * ((i + j) as f64)
            }
        })
    }

    #[test]
    fn iforest_scores_and_labels() {
        let x = cloud_plus_outlier();
        let q = IsolationForest {
            n_trees: 16,
            seed: 3,
            contamination: 0.12,
        }
        .fit_unsupervised(&x, &Session::new("anom", "if"))
        .unwrap();
        let s = q
            .value
            .scores(&x, &Session::new("anom", "sc"))
            .unwrap()
            .value;
        assert_eq!(s.len(), 16);
        let y = q
            .value
            .predict(&x, &Session::new("anom", "pr"))
            .unwrap()
            .value;
        assert!(y.as_slice().iter().all(|&v| v == 1.0 || v == -1.0));
    }

    #[test]
    fn lof_and_hypersphere_run() {
        let x = cloud_plus_outlier();
        let lof = LocalOutlierFactor::new(3)
            .fit_unsupervised(&x, &Session::new("anom", "lof"))
            .unwrap();
        let y = lof
            .value
            .predict(&x, &Session::new("anom", "pr"))
            .unwrap()
            .value;
        assert_eq!(y.len(), 16);
        let hs = OneClassHypersphere::new(0.15)
            .fit_unsupervised(&x, &Session::new("anom", "hs"))
            .unwrap();
        let z = hs
            .value
            .predict(&x, &Session::new("anom", "pr"))
            .unwrap()
            .value;
        assert_eq!(z[0], -1.0);
    }

    #[test]
    fn elliptic_envelope_flags_far_point() {
        let x = cloud_plus_outlier();
        let q = EllipticEnvelope::new(0.12)
            .fit_unsupervised(&x, &Session::new("anom", "ee"))
            .unwrap();
        let y = q
            .value
            .predict(&x, &Session::new("anom", "pr"))
            .unwrap()
            .value;
        assert_eq!(y[0], -1.0);
    }

    fn one_reference_score(p: f64, log_transform: bool, reference: &[f64], query: &[f64]) -> f64 {
        assert_eq!(reference.len(), query.len());
        let mut detector = KnnDistanceAnomaly::new(1, p, log_transform, 4).unwrap();
        let reference = Matrix::from_row_major(1, reference.len(), reference);
        let _update = detector
            .partial_fit(&reference, None, &Session::new("knn", "update"))
            .unwrap();
        detector
            .predict(
                &Matrix::from_row_major(1, query.len(), query),
                &Session::new("knn", "predict"),
            )
            .unwrap()
            .value[0]
    }

    #[test]
    fn knn_distance_constructor_enforces_metric_domain() {
        for (k, p, window) in [
            (0, 2.0, 4),
            (2, 2.0, 0),
            (3, 2.0, 2),
            (1, 0.5, 4),
            (1, f64::NAN, 4),
            (1, f64::NEG_INFINITY, 4),
        ] {
            let failure = KnnDistanceAnomaly::new(k, p, false, window).unwrap_err();
            assert_eq!(failure.primary.code, IssueCode::InvalidParameter);
        }
        for p in [1.0, 2.0, 5.0, f64::INFINITY] {
            let detector = KnnDistanceAnomaly::new(2, p, true, 8).unwrap();
            assert_eq!(detector.k(), 2);
            assert_eq!(detector.p(), p);
            assert!(detector.log_transform());
            assert_eq!(detector.window(), 8);
        }
        let huge_window = KnnDistanceAnomaly::new(1, 2.0, false, usize::MAX).unwrap();
        assert_eq!(huge_window.window(), usize::MAX);
        assert!(huge_window.rows.is_empty());
    }

    #[test]
    fn knn_distance_orders_match_closed_form_norms() {
        let query = [3.0, 4.0];
        let cases = [
            (1.0, 7.0),
            (2.0, 5.0),
            (5.0, 1267.0_f64.powf(0.2)),
            (f64::INFINITY, 4.0),
        ];
        // Measured max absolute error was 0.0 on 2026-09-02; 2e-14 allows
        // cross-platform libm rounding while remaining far below data scale.
        for (p, expected) in cases {
            let actual = one_reference_score(p, false, &[0.0, 0.0], &query);
            assert!(
                (actual - expected).abs() <= 2.0e-14,
                "p={p}: actual={actual:.17e}, expected={expected:.17e}"
            );
        }
        // The removed p=2 implementation returned 25 here (squared L2).
        assert_eq!(one_reference_score(2.0, false, &[0.0, 0.0], &query), 5.0);
    }

    #[test]
    fn knn_runtime_p3_matches_the_closed_form_and_ranks_distance() {
        let references =
            Matrix::from_row_major(5, 2, &[0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 4.0, -2.0, 1.0]);
        let mut detector = KnnDistanceAnomaly::new(5, 3.0, false, 64).unwrap();
        let _update = detector
            .partial_fit(&references, None, &Session::new("knn", "p3-update"))
            .unwrap();
        let scores = detector
            .predict(
                &Matrix::from_row_major(2, 2, &[1.0, 1.0, 10.0, 10.0]),
                &Session::new("knn", "p3-predict"),
            )
            .unwrap()
            .value;
        let expected = (2.0_f64.cbrt() + 1.0 + 2.0_f64.cbrt() + 35.0_f64.cbrt() + 3.0) / 5.0;
        // Measured absolute error was 0.0 on 2026-09-02; tol = 2e-14.
        assert!((scores[0] - expected).abs() <= 2.0e-14);
        assert!(scores[0] < scores[1]);
    }

    #[test]
    fn knn_log_transform_preserves_sign_zero_and_cross_unit_distance() {
        let mut detector = KnnDistanceAnomaly::new(1, 1.0, true, 4).unwrap();
        let _update = detector
            .partial_fit(
                &Matrix::from_row_major(1, 3, &[-1.0, 0.0, 0.5]),
                None,
                &Session::new("knn", "update"),
            )
            .unwrap();
        let score = detector
            .predict(
                &Matrix::from_row_major(1, 3, &[1.0, 1.0, 2.0]),
                &Session::new("knn", "predict"),
            )
            .unwrap()
            .value[0];
        let expected = 4.0 * 2.0_f64.ln();
        // Measured absolute error was 0.0 on 2026-09-02; tol = 2e-14.
        assert!((score - expected).abs() <= 2.0e-14);
        assert!(
            score > 0.0,
            "the removed |ln| transform aliased these values"
        );

        let cases = [
            (1.0, -1.0, 1.0, 2.0 * 2.0_f64.ln()),
            (1.0, 0.0, 1.0, 2.0_f64.ln()),
            (1.0, 0.5, 2.0, 2.0_f64.ln()),
            (2.0, 0.5, 2.0, 2.0_f64.ln()),
            (f64::INFINITY, 0.5, 2.0, 2.0_f64.ln()),
        ];
        for (p, reference, query, expected) in cases {
            let actual = one_reference_score(p, true, &[reference], &[query]);
            // Measured max absolute error was 0.0 on 2026-09-02; tol = 2e-14.
            assert!((actual - expected).abs() <= 2.0e-14);
        }
    }

    #[test]
    fn knn_warmup_and_duplicate_neighbors_have_explicit_semantics() {
        let mut detector = KnnDistanceAnomaly::new(2, 2.0, false, 3).unwrap();
        let _first_update = detector
            .partial_fit(&Matrix::zeros(1, 1), None, &Session::new("knn", "first"))
            .unwrap();
        let warmup = detector
            .predict(&Matrix::zeros(1, 1), &Session::new("knn", "warmup"))
            .unwrap();
        assert_eq!(warmup.value[0], 0.0);
        assert!(warmup
            .report
            .issues()
            .iter()
            .any(|issue| issue.code == IssueCode::WarmupIncomplete));

        let _second_update = detector
            .partial_fit(&Matrix::zeros(1, 1), None, &Session::new("knn", "second"))
            .unwrap();
        let ready = detector
            .predict(&Matrix::zeros(1, 1), &Session::new("knn", "ready"))
            .unwrap();
        assert_eq!(ready.value[0], 0.0);
        assert!(!ready
            .report
            .issues()
            .iter()
            .any(|issue| issue.code == IssueCode::WarmupIncomplete));
    }

    #[test]
    fn knn_stream_partition_eviction_and_prediction_are_deterministic() {
        let all = Matrix::from_row_major(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let first = Matrix::from_row_major(2, 1, &[0.0, 1.0]);
        let second = Matrix::from_row_major(2, 1, &[2.0, 3.0]);
        let mut whole = KnnDistanceAnomaly::new(2, 2.0, false, 3).unwrap();
        let mut split = KnnDistanceAnomaly::new(2, 2.0, false, 3).unwrap();
        let whole_session = Session::new("knn", "whole");
        let whole_update = whole.partial_fit(&all, None, &whole_session).unwrap();
        assert_eq!(whole_update.value.quality.effective_sample_size, 3.0);
        assert_eq!(whole_update.value.quality.n_seen, 4);
        assert!(whole_session
            .ledger()
            .events()
            .iter()
            .any(|event| event.incremental.is_some()));
        let _first_update = split
            .partial_fit(&first, None, &Session::new("knn", "first"))
            .unwrap();
        let _second_update = split
            .partial_fit(&second, None, &Session::new("knn", "second"))
            .unwrap();
        assert_eq!(whole.rows, split.rows);
        assert_eq!(whole.n_seen, split.n_seen);

        let rows_before = whole.rows.clone();
        let updates_before = whole.updates;
        let score = whole
            .predict(&Matrix::zeros(1, 1), &Session::new("knn", "predict"))
            .unwrap()
            .value[0];
        assert_eq!(score, 1.5);
        assert_eq!(whole.rows, rows_before);
        assert_eq!(whole.updates, updates_before);
    }

    #[test]
    fn knn_prediction_errors_and_empty_updates_preserve_state() {
        let mut detector = KnnDistanceAnomaly::new(1, 2.0, false, 4).unwrap();
        let failure = detector
            .predict(&Matrix::zeros(1, 2), &Session::new("knn", "before-init"))
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::PartialFitBeforeInit);

        let failure = detector
            .partial_fit(
                &Matrix::zeros(0, 2),
                None,
                &Session::new("knn", "empty-update"),
            )
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::EmptyMatrix);
        assert!(detector.rows.is_empty());
        assert_eq!(detector.n_seen, 0);
        assert_eq!(detector.updates, 0);

        let _initial_update = detector
            .partial_fit(
                &Matrix::zeros(1, 2),
                None,
                &Session::new("knn", "initialize"),
            )
            .unwrap();
        let rows_before = detector.rows.clone();
        let updates_before = detector.updates;
        let failure = detector
            .predict(&Matrix::zeros(1, 1), &Session::new("knn", "bad-shape"))
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::FeatureSpaceChangedOnline);
        let failure = detector
            .predict(
                &Matrix::from_row_major(1, 2, &[f64::INFINITY, 0.0]),
                &Session::new("knn", "bad-value"),
            )
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::NonFiniteInput);
        assert_eq!(detector.rows, rows_before);
        assert_eq!(detector.updates, updates_before);
    }

    #[test]
    fn knn_rejects_invalid_updates_transactionally_and_extreme_norms_stay_finite() {
        let mut detector = KnnDistanceAnomaly::new(1, 360.0, false, 3).unwrap();
        let _initial_update = detector
            .partial_fit(
                &Matrix::zeros(1, 2),
                None,
                &Session::new("knn", "initialize"),
            )
            .unwrap();
        let rows_before = detector.rows.clone();
        let n_seen_before = detector.n_seen;
        let non_finite = Matrix::from_row_major(1, 2, &[f64::NAN, 1.0]);
        let failure = detector
            .partial_fit(&non_finite, None, &Session::new("knn", "nan"))
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::NonFiniteInput);
        assert_eq!(detector.rows, rows_before);
        assert_eq!(detector.n_seen, n_seen_before);

        let wrong_width = Matrix::zeros(1, 1);
        let failure = detector
            .partial_fit(&wrong_width, None, &Session::new("knn", "shape"))
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::FeatureSpaceChangedOnline);
        assert_eq!(detector.rows, rows_before);
        assert_eq!(detector.n_seen, n_seen_before);

        let score = detector
            .predict(
                &Matrix::from_row_major(1, 2, &[3.0e200, 4.0e200]),
                &Session::new("knn", "stress"),
            )
            .unwrap()
            .value[0];
        assert!(score.is_finite());
        assert!(score >= 4.0e200 && score <= 4.01e200);

        let mut overflow = KnnDistanceAnomaly::new(1, 2.0, false, 2).unwrap();
        let _reference_update = overflow
            .partial_fit(
                &Matrix::from_row_major(1, 1, &[-f64::MAX]),
                None,
                &Session::new("knn", "overflow-reference"),
            )
            .unwrap();
        let failure = overflow
            .predict(
                &Matrix::from_row_major(1, 1, &[f64::MAX]),
                &Session::new("knn", "overflow-query"),
            )
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::NonFiniteOutput);
    }
}
