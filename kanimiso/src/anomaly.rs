//! Outlier / novelty detectors with a common `+1` inlier / `−1` outlier label.
//!
//! Wrappers reuse [`crate::tree::IsolationForest`], [`crate::neighbors::LocalOutlierFactor`],
//! [`crate::svm::OneClassSvm`], and [`crate::covariance::EllipticEnvelope`]. Scores
//! are detector-native (higher = more anomalous); [`Predict`] always emits
//! the sklearn convention `+1 / −1`.

use crate::context::FitCtx;
use crate::covariance::{EllipticEnvelope as CovEnvelope, FittedEllipticEnvelope};
use crate::data::{Matrix, Vector};
use crate::neighbors::{FittedLof, LocalOutlierFactor as NeighLof};
use crate::svm::{FittedOneClassSvm, OneClassSvm};
use crate::traits::{FitUnsupervised, Predict};
use crate::tree::{FittedIsolationForest, IsolationForest as TreeIsolationForest};
use crate::validate::inspect_xy;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Qualified, Result};

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
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedAnomalyForest>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if !self.contamination.is_finite()
            || self.contamination <= 0.0
            || self.contamination > 0.5
        {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("contamination={} not in (0, 0.5]", self.contamination))
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
}
