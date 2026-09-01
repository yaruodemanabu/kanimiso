//! Outlier / novelty detectors.
//!
//! Batch wrappers reuse [`crate::tree::IsolationForest`],
//! [`crate::neighbors::LocalOutlierFactor`], [`crate::svm::OneClassSvm`], and
//! [`crate::covariance::EllipticEnvelope`]. Their scores are detector-native
//! (higher = more anomalous); [`Predict`] emits the sklearn convention `+1 / −1`.
//!
//! [`KnnDistanceAnomaly`] is the streaming replacement for the v0.1
//! `LogMinkowski{3..359}Anomaly` family (AGENTS.md §3.4). It scores rows by
//! mean k-NN distance and does **not** emit the ±1 label convention.

use crate::bridge::failure_from_wormhole;
use crate::context::FitCtx;
use crate::covariance::{EllipticEnvelope as CovEnvelope, FittedEllipticEnvelope};
use crate::data::{Matrix, Vector};
use crate::neighbors::{FittedLof, LocalOutlierFactor as NeighLof};
use crate::svm::{FittedOneClassSvm, OneClassSvm};
use crate::traits::{FitUnsupervised, PartialFit, Predict};
use crate::tree::{FittedIsolationForest, IsolationForest as TreeIsolationForest};
use crate::validate::inspect_xy;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{IncrementalQuality, Issue, IssueCode, Qualified, Result};
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
    fn score_vec(&self, x: &Matrix) -> Vector {
        match &self.inner {
            Some(f) => f.score_samples(x),
            None => Vector::zeros(x.nrows()),
        }
    }

    /// Isolation scores `2^{-E(h)/c(n)}` (higher = more anomalous).
    pub fn scores(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("score"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(self.score_vec(x))
    }
}

impl Predict for FittedAnomalyForest {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let s = self.score_vec(x);
        ctx.finish(labels_from_scores(&s, self.threshold, true))
    }
}

impl FitUnsupervised for IsolationForest {
    type Fitted = FittedAnomalyForest;
    fn fit_unsupervised(
        &self,
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
                ctx.push(e.primary);
                None
            }
        };
        let scores = match &fitted {
            Some(f) => f.score_samples(x),
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
        &self,
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
        &self,
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

/// Fitted elliptic envelope.
#[derive(Clone, Debug)]
pub struct FittedAnomalyEnvelope {
    inner: FittedEllipticEnvelope,
}

impl FittedAnomalyEnvelope {
    /// Mahalanobis scores (higher = more outlying).
    pub fn scores(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.inner.scores(x, session)
    }
}

impl Predict for FittedAnomalyEnvelope {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.inner.predict(x, session)
    }
}

impl FitUnsupervised for EllipticEnvelope {
    type Fitted = FittedAnomalyEnvelope;
    fn fit_unsupervised(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedAnomalyEnvelope>> {
        self.inner
            .fit_unsupervised(x, session)
            .map(|q| q.map(|inner| FittedAnomalyEnvelope { inner }))
    }
}

/// v0.1 log-Minkowski coordinate floor. This is a chart for `|x| = 0`, not a
/// density floor (R7).
const LOG_COORD_FLOOR: f64 = 1e-18;

/// v0.1 near-duplicate neighbor drop. Distances at or below this are ignored
/// when averaging the k nearest; it is not a density floor (R7).
const NEIGHBOR_DISTANCE_FLOOR: f64 = 1e-15;

/// Reservoir length used by every v0.1 `LogMinkowski*Anomaly`.
const RESERVOIR_CAP: usize = 64;

/// Minimum stored rows before a score is defined (v0.1 warmup).
const MIN_RESERVOIR: usize = 3;

/// Streaming k-NN distance anomaly (AGENTS.md §3.4).
///
/// Replaces the v0.1 `LogMinkowski3..359Anomaly` family. The Minkowski
/// exponent is a [`wormhole::Metric`] runtime value, not a type name.
/// Distances go through [`wormhole::metrics::distance`]; `p < 1` is rejected
/// by wormhole and surfaced as a [`signlred::Failure`].
///
/// When [`Self::log_transform`] is set, each coordinate is mapped with the
/// v0.1 log-Minkowski chart `|ln max(|x|, ε)|` (floored at ε). The score is
/// the mean of the `k` smallest finite neighbor distances strictly above a
/// near-duplicate floor. The reservoir holds at most 64 rows.
#[derive(Clone, Debug)]
pub struct KnnDistanceAnomaly {
    /// Ground metric. [`wormhole::Metric::Minkowski`] requires `p >= 1`.
    pub metric: wormhole::Metric,
    /// Number of nearest neighbors averaged into the score.
    pub k: NonZeroUsize,
    /// Apply the v0.1 `|ln|x||` coordinate chart before the metric.
    pub log_transform: bool,
    rows: Vec<Vec<f64>>,
    ncols: usize,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl KnnDistanceAnomaly {
    /// Empty detector with the given metric, neighborhood size, and chart.
    pub fn new(metric: wormhole::Metric, k: NonZeroUsize, log_transform: bool) -> Self {
        Self {
            metric,
            k,
            log_transform,
            rows: Vec::new(),
            ncols: 0,
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }

    /// v0.1 log-Minkowski detector: `|ln|x||` coordinates, `k = 5`.
    ///
    /// `p = 1` is Manhattan, `p = 2` is Euclidean, `p = +∞` is Chebyshev,
    /// and any other finite `p` is [`wormhole::Metric::Minkowski`]. Invalid
    /// `p` (non-finite other than `+∞`, or `p < 1`) fails on the first
    /// [`PartialFit::partial_fit`] / [`Predict::predict`].
    pub fn log_minkowski(p: f64) -> Self {
        Self::new(metric_from_p(p), NonZeroUsize::new(5).expect("5 ≠ 0"), true)
    }

    fn ensure_metric(&self) -> Result<()> {
        wormhole::metrics::distance(&[0.0], &[1.0], self.metric)
            .map(|_| ())
            .map_err(|err| failure_from_wormhole("KnnDistanceAnomaly", "metric", err))
    }

    fn chart(&self, row: &[f64]) -> Vec<f64> {
        if self.log_transform {
            row.iter().copied().map(log_minkowski_coord).collect()
        } else {
            row.to_vec()
        }
    }

    fn pair_distance(&self, left: &[f64], right: &[f64]) -> Option<f64> {
        let d = wormhole::metrics::distance(left, right, self.metric).ok()?;
        if d.is_finite() && d > NEIGHBOR_DISTANCE_FLOOR {
            Some(d)
        } else {
            None
        }
    }

    fn score_row(&self, x: &Matrix, i: usize) -> f64 {
        if self.rows.len() < MIN_RESERVOIR || self.ncols == 0 {
            return 0.0;
        }
        let p = self.ncols.min(x.ncols());
        let mut qrow = Vec::with_capacity(p);
        for j in 0..p {
            let z = x.get(i, j);
            if !z.is_finite() {
                return 0.0;
            }
            qrow.push(z);
        }
        let query = self.chart(&qrow);
        let mut ds: Vec<f64> = self
            .rows
            .iter()
            .filter_map(|row| {
                let stored = if row.len() == p {
                    row.as_slice()
                } else {
                    return None;
                };
                self.pair_distance(&query, &self.chart(stored))
            })
            .collect();
        if ds.is_empty() {
            return 0.0;
        }
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let kk = self.k.get().min(ds.len());
        let mut s = 0.0_f64;
        for d in ds.iter().take(kk) {
            s += *d;
        }
        s / kk as f64
    }
}

impl Default for KnnDistanceAnomaly {
    fn default() -> Self {
        Self::new(
            wormhole::Metric::Euclidean,
            NonZeroUsize::new(5).expect("5 ≠ 0"),
            true,
        )
    }
}

fn metric_from_p(p: f64) -> wormhole::Metric {
    if p.is_infinite() && p.is_sign_positive() {
        wormhole::Metric::Chebyshev
    } else if p == 1.0 {
        wormhole::Metric::Manhattan
    } else if p == 2.0 {
        wormhole::Metric::Euclidean
    } else {
        wormhole::Metric::Minkowski(p)
    }
}

fn log_minkowski_coord(x: f64) -> f64 {
    let magnitude = x.abs();
    let floored = if magnitude > LOG_COORD_FLOOR {
        magnitude
    } else {
        LOG_COORD_FLOOR
    };
    let logged = floored.ln().abs();
    if logged > LOG_COORD_FLOOR {
        logged
    } else {
        LOG_COORD_FLOOR
    }
}

fn inspect_stream_xy(ctx: &mut FitCtx, x: &Matrix) {
    let (n, p) = x.shape();
    ctx.report.set_sample_shape(n, p);
    if n == 0 || p == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message(format!("online design is {n}×{p}"))
                .build(),
        );
    }
}

fn reject_explain(update: u64, batch: usize, n_seen: u64, why: &str) -> IncrementalExplain {
    IncrementalExplain::from_quality(
        IncrementalQuality::new(update, batch, n_seen),
        "nothing",
        why,
        "invalid",
        "invalid",
    )
}

fn flag_info(ctx: &mut FitCtx, q: &IncrementalQuality) {
    if q.is_uninformative(ctx.policy.uninformative_info_eps) {
        ctx.push(
            Issue::builder(IssueCode::UpdateWithZeroInformation)
                .incremental(q.clone())
                .message("this online update added no usable information")
                .build(),
        );
    }
}

fn finish_explain(ctx: FitCtx, expl: IncrementalExplain) -> Result<Qualified<IncrementalExplain>> {
    ctx.session.record_incremental(expl.clone());
    ctx.finish(expl)
}

impl PartialFit for KnnDistanceAnomaly {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        self.ensure_metric()?;
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_stream_xy(&mut ctx, x);
        if x.ncols() == 0 {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return finish_explain(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "no features"),
            );
        }
        let p = x.ncols();
        if !self.initialized {
            self.ncols = p;
            self.initialized = true;
        } else if self.ncols != p {
            ctx.push(Issue::builder(IssueCode::FeatureSpaceChangedOnline).build());
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
        let before_n = self.n_seen;
        for i in 0..x.nrows() {
            let mut row = Vec::with_capacity(p);
            let mut ok = true;
            for j in 0..p {
                let z = x.get(i, j);
                if !z.is_finite() {
                    ok = false;
                    break;
                }
                row.push(z);
            }
            if !ok {
                continue;
            }
            if self.rows.len() >= RESERVOIR_CAP {
                self.rows.remove(0);
            }
            self.rows.push(row);
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((self.n_seen - before_n) as f64);
        q.information_gain = Some(x.nrows() as f64);
        q.still_identified = self.rows.len() >= MIN_RESERVOIR;
        q.warmup = self.rows.len() < MIN_RESERVOIR;
        q.explanation = format!(
            "k-NN distance reservoir n={} k={} log_transform={}",
            self.rows.len(),
            self.k,
            self.log_transform
        );
        flag_info(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "k-NN distance anomaly reservoir",
                "KnnDistanceAnomaly stores raw rows and scores them with a wormhole metric; p is a runtime parameter",
                "previous reservoir",
                "updated reservoir",
            ),
        )
    }
}

impl Predict for KnnDistanceAnomaly {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.ensure_metric()?;
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_stream_xy(&mut ctx, x);
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        ctx.finish(Vector::from_iter(
            (0..x.nrows()).map(|i| self.score_row(x, i)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{PartialFit, Predict};
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

    fn v0_1_log_coord(x: f64) -> f64 {
        let magnitude = x.abs();
        let floored = if magnitude > LOG_COORD_FLOOR {
            magnitude
        } else {
            LOG_COORD_FLOOR
        };
        let logged = floored.ln().abs();
        if logged > LOG_COORD_FLOOR {
            logged
        } else {
            LOG_COORD_FLOOR
        }
    }

    fn v0_1_log_minkowski(a: &[f64], b: &[f64], p: f64) -> f64 {
        let n = a.len().min(b.len());
        if p.is_infinite() {
            let mut m = 0.0_f64;
            for j in 0..n {
                let d = (v0_1_log_coord(a[j]) - v0_1_log_coord(b[j])).abs();
                if d > m {
                    m = d;
                }
            }
            return m;
        }
        let mut s = 0.0_f64;
        for j in 0..n {
            s += (v0_1_log_coord(a[j]) - v0_1_log_coord(b[j])).abs().powf(p);
        }
        s.powf(1.0 / p)
    }

    fn v0_1_knn_score(rows: &[Vec<f64>], query: &[f64], p: f64, k: usize) -> f64 {
        if rows.len() < MIN_RESERVOIR {
            return 0.0;
        }
        let mut ds: Vec<f64> = rows
            .iter()
            .map(|row| v0_1_log_minkowski(query, row, p))
            .filter(|d| d.is_finite() && *d > NEIGHBOR_DISTANCE_FLOOR)
            .collect();
        if ds.is_empty() {
            return 0.0;
        }
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let kk = k.min(ds.len());
        ds.iter().take(kk).sum::<f64>() / kk as f64
    }

    #[test]
    fn knn_distance_log_minkowski_matches_v0_1_for_representative_p() {
        // Train includes values in (0, 1) and > 1 so |ln| differs from ln.
        let train = Matrix::from_fn(12, 2, |i, j| {
            let base = 0.25 * ((i as f64) + 1.0);
            if j == 0 {
                base
            } else {
                base * 1.7
            }
        });
        let query = Matrix::from_fn(1, 2, |_, j| if j == 0 { 4.0 } else { 0.4 });
        let session = Session::new("knn-dist", "score");
        let mut stored = Vec::new();
        for i in 0..train.nrows() {
            stored.push(vec![train.get(i, 0), train.get(i, 1)]);
        }
        let qrow = [query.get(0, 0), query.get(0, 1)];
        // measured 2026-09-01: |unified − v0.1 formula| = 0 on this design
        // (wormhole Minkowski uses powf; integer p matches powi here).
        for p in [1.0_f64, 2.0, 5.0, f64::INFINITY] {
            let mut det = KnnDistanceAnomaly::log_minkowski(p);
            det.partial_fit(&train, None, &session)
                .unwrap_or_else(|e| panic!("p={p}: {e}"));
            let got = det.predict(&query, &session).expect("pred").value[0];
            let expected = v0_1_knn_score(&stored, &qrow, p, 5);
            assert!(
                (got - expected).abs() <= 1e-12,
                "p={p}: got {got} expected {expected}"
            );
        }
    }

    #[test]
    fn knn_distance_p_below_one_is_failure() {
        let x = Matrix::from_fn(8, 1, |i, _| (i as f64) + 1.0);
        let session = Session::new("knn-dist", "bad-p");
        let err = KnnDistanceAnomaly::log_minkowski(0.5)
            .partial_fit(&x, None, &session)
            .expect_err("p < 1 is a wormhole InvalidParameter");
        assert_eq!(err.primary().code, IssueCode::MeaninglessFit);
    }
}
