//! Feature selection, random Fourier features, and simple time-series maps.
//!
//! Supervised selectors that score against the full `y` (no cross-validation
//! note) raise [`IssueCode::TargetLeakageSuspected`]. Constant columns are
//! dropped by [`VarianceThreshold`] and ignored by [`SelectKBest`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::least_squares;
use crate::rng::Rng;
use crate::traits::{Fit, FitUnsupervised, Transform};
use crate::validate::inspect_xy;
use faer::Mat;
use ojizou_san::Session;
use signlred::{
    slice_stats, Issue, IssueCode, Meaninglessness, Qualified, Result, Severity, SliceStats,
};

/// Re-export of [`crate::preprocess::PolynomialFeatures`] for a sklearn-like
/// `feature_extraction` surface.
pub use crate::preprocess::PolynomialFeatures;

/// Drop columns whose sample variance is at or below `threshold`.
#[derive(Clone, Debug)]
pub struct VarianceThreshold {
    /// Variance cutoff (0 drops exact constants).
    pub threshold: f64,
    variances: Vector,
    support: Vec<bool>,
    fitted: bool,
}

impl Default for VarianceThreshold {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            variances: Vector::zeros(0),
            support: Vec::new(),
            fitted: false,
        }
    }
}

impl VarianceThreshold {
    /// Drop columns with variance ≤ `threshold`.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Self::default()
        }
    }

    /// Fitted per-column variances.
    pub fn variances(&self) -> &Vector {
        &self.variances
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }
}

impl FitUnsupervised for VarianceThreshold {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        self.variances = Vector::zeros(p);
        self.support = vec![false; p];
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
            let st = slice_stats(&col);
            self.variances[j] = st.variance;
            self.support[j] = st.count >= 2 && st.variance > self.threshold;
            if st.count > 0 && st.is_constant(ctx.policy.near_zero_variance) {
                ctx.push(constant_col_issue(j, st));
            }
        }
        if self.support.iter().all(|s| !*s) {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("VarianceThreshold dropped every column")
                    .meaninglessness(Meaninglessness::vacuous(
                        "feature mask",
                        "no column exceeded the variance cutoff",
                        "lower the threshold or inspect the design",
                    ))
                    .build(),
            );
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for VarianceThreshold {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Keep the `k` features with largest f-regression scores (`corr²` with `y`).
#[derive(Clone, Debug)]
pub struct SelectKBest {
    /// Number of features to keep.
    pub k: usize,
    scores: Vector,
    support: Vec<bool>,
    fitted: bool,
}

impl Default for SelectKBest {
    fn default() -> Self {
        Self {
            k: 1,
            scores: Vector::zeros(0),
            support: Vec::new(),
            fitted: false,
        }
    }
}

impl SelectKBest {
    /// Keep `k` features.
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            ..Self::default()
        }
    }

    /// `corr²` scores (NaN for constant / unidentified columns).
    pub fn scores(&self) -> &Vector {
        &self.scores
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }
}

impl Fit for SelectKBest {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::TargetLeakageSuspected)
                .severity(Severity::Advisory)
                .message(
                    "SelectKBest scored every feature against the full y; without a CV note this can leak the target into the selected set",
                )
                .build(),
        );
        let (n, p) = x.shape();
        self.scores = Vector::zeros(p);
        let yst = slice_stats(y.as_slice());
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
            let xst = slice_stats(&col);
            self.scores[j] = pearson_sq(&col, xst, y.as_slice(), yst);
        }
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|a, b| {
            self.scores[*b]
                .partial_cmp(&self.scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.support = vec![false; p];
        for &j in order.iter().take(self.k.min(p)) {
            if self.scores[j].is_finite() {
                self.support[j] = true;
            }
        }
        if self.support.iter().all(|s| !*s) && p > 0 {
            self.support[order[0]] = true;
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for SelectKBest {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Recursive feature elimination wrapping a plain OLS score (`|β|`).
#[derive(Clone, Debug)]
pub struct Rfe {
    /// Features to keep.
    pub n_features_to_select: usize,
    support: Vec<bool>,
    ranking: Vec<usize>,
    fitted: bool,
}

impl Default for Rfe {
    fn default() -> Self {
        Self {
            n_features_to_select: 1,
            support: Vec::new(),
            ranking: Vec::new(),
            fitted: false,
        }
    }
}

impl Rfe {
    /// Eliminate down to `n_features_to_select` columns.
    pub fn new(n_features_to_select: usize) -> Self {
        Self {
            n_features_to_select: n_features_to_select.max(1),
            ..Self::default()
        }
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }

    /// Elimination rank (1 = kept in the final set).
    pub fn ranking(&self) -> &[usize] {
        &self.ranking
    }
}

impl Fit for Rfe {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::TargetLeakageSuspected)
                .severity(Severity::Advisory)
                .message(
                    "RFE refits OLS on the full y at every elimination step; this is not a CV-safe selector",
                )
                .build(),
        );
        let p = x.ncols();
        let keep = self.n_features_to_select.min(p.max(1));
        let mut active: Vec<usize> = (0..p).collect();
        self.ranking = vec![0; p];
        let mut rank = p;
        while active.len() > keep {
            let sub = take_columns(x, &active);
            let Some(beta) = least_squares(&mut ctx.report, &sub, y, &ctx.policy) else {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("OLS score inside RFE failed; stopping elimination")
                        .build(),
                );
                break;
            };
            let mut worst = 0usize;
            let mut worst_abs = f64::INFINITY;
            for (t, _) in active.iter().enumerate() {
                let a = beta[t].abs();
                if a < worst_abs {
                    worst_abs = a;
                    worst = t;
                }
            }
            let dropped = active.remove(worst);
            self.ranking[dropped] = rank;
            rank -= 1;
        }
        self.support = vec![false; p];
        for &j in &active {
            self.support[j] = true;
            self.ranking[j] = 1;
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for Rfe {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Random Fourier features for an RBF kernel (Rahimi & Recht).
#[derive(Clone, Debug)]
pub struct RbfSampler {
    /// Number of Monte-Carlo frequencies.
    pub n_components: usize,
    /// RBF `γ` in `exp(-γ‖x−x'‖²)`.
    pub gamma: f64,
    /// PRNG seed.
    pub seed: u64,
    weights: Matrix,
    offset: Vector,
    fitted: bool,
}

impl Default for RbfSampler {
    fn default() -> Self {
        Self {
            n_components: 100,
            gamma: 1.0,
            seed: 0,
            weights: Matrix::zeros(0, 0),
            offset: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl RbfSampler {
    /// Sampler with `n_components` features and RBF `γ`.
    pub fn new(n_components: usize, gamma: f64) -> Self {
        Self {
            n_components: n_components.max(1),
            gamma,
            ..Self::default()
        }
    }
}

impl FitUnsupervised for RbfSampler {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let p = x.ncols();
        let m = self.n_components;
        let mut rng = Rng::new(self.seed);
        let sd = (2.0 * self.gamma).sqrt();
        self.weights = Matrix::from_fn(p, m, |_, _| rng.standard_normal() * sd);
        self.offset =
            Vector::from_iter((0..m).map(|_| rng.uniform_range(0.0, 2.0 * std::f64::consts::PI)));
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for RbfSampler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        if x.ncols() != self.weights.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("RbfSampler transform p ≠ fitted p")
                    .build(),
            );
        }
        let scale = (2.0 / self.n_components as f64).sqrt();
        let m = self.offset.len();
        let out = Matrix::from_fn(x.nrows(), m, |i, k| {
            let mut s = self.offset[k];
            for j in 0..x.ncols().min(self.weights.nrows()) {
                s += x.get(i, j) * self.weights.get(j, k);
            }
            scale * s.cos()
        });
        ctx.finish(out)
    }
}

/// Nyström approximation to an RBF kernel map.
#[derive(Clone, Debug)]
pub struct Nystroem {
    /// Number of landmark rows.
    pub n_components: usize,
    /// RBF `γ`.
    pub gamma: f64,
    /// PRNG seed for landmark sampling.
    pub seed: u64,
    landmarks: Matrix,
    norm: Mat<f64>,
    fitted: bool,
}

impl Default for Nystroem {
    fn default() -> Self {
        Self {
            n_components: 10,
            gamma: 1.0,
            seed: 1,
            landmarks: Matrix::zeros(0, 0),
            norm: Mat::<f64>::zeros(0, 0),
            fitted: false,
        }
    }
}

impl Nystroem {
    /// Nyström map with `n_components` landmarks.
    pub fn new(n_components: usize, gamma: f64) -> Self {
        Self {
            n_components: n_components.max(1),
            gamma,
            ..Self::default()
        }
    }
}

impl FitUnsupervised for Nystroem {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let m = self.n_components.min(x.nrows()).max(1);
        let mut rng = Rng::new(self.seed);
        let idx = rng.sample_indices(x.nrows(), m);
        self.landmarks = Matrix::from_fn(m, x.ncols(), |r, c| x.get(idx[r], c));
        let mut k = Mat::<f64>::zeros(m, m);
        for i in 0..m {
            for j in 0..=i {
                let kij = rbf_entry(&self.landmarks, i, &self.landmarks, j, self.gamma);
                k[(i, j)] = kij;
                k[(j, i)] = kij;
            }
        }
        let Some((vals, u)) = crate::linalg::symmetric_eigen(&mut ctx.report, &k, &ctx.policy)
        else {
            ctx.push(
                Issue::builder(IssueCode::KernelNotPd)
                    .message("Nyström landmark kernel eigendecomposition failed")
                    .build(),
            );
            self.norm = Mat::<f64>::zeros(m, m);
            self.fitted = true;
            return ctx.finish(self.clone());
        };
        let mut kept = 0usize;
        for &λ in &vals {
            if λ > ctx.policy.rank_tol_relative {
                kept += 1;
            } else if λ < 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NegativeEigenvalueDropped)
                        .message(format!("Nyström kernel eigenvalue {λ:.3e} < 0"))
                        .build(),
                );
            }
        }
        if kept == 0 {
            ctx.push(
                Issue::builder(IssueCode::RankZero)
                    .message("Nyström kernel has no positive eigenvalue")
                    .build(),
            );
        }
        // W^{+}½ ≈ U diag(λ⁺½) — columns of `norm` are the embedding directions.
        let r = vals.len().min(u.ncols());
        self.norm = Mat::<f64>::zeros(m, r);
        for c in 0..r {
            let s = if vals[c] > ctx.policy.rank_tol_relative {
                1.0 / vals[c].sqrt()
            } else {
                0.0
            };
            for i in 0..m {
                self.norm[(i, c)] = u[(i, c)] * s;
            }
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for Nystroem {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        let m = self.landmarks.nrows();
        let r = self.norm.ncols();
        let out = Matrix::from_fn(x.nrows(), r, |i, c| {
            let mut s = 0.0;
            for t in 0..m {
                let kxt = rbf_entry(x, i, &self.landmarks, t, self.gamma);
                s += kxt * self.norm[(t, c)];
            }
            s
        });
        ctx.finish(out)
    }
}

/// Histogram mutual-information classifier feature scorer / selector.
#[derive(Clone, Debug)]
pub struct MutualInfoClassif {
    /// Histogram bins per feature.
    pub n_bins: usize,
    /// Keep the top `k` features (all if `k` exceeds `p`).
    pub k: usize,
    scores: Vector,
    support: Vec<bool>,
    fitted: bool,
}

impl Default for MutualInfoClassif {
    fn default() -> Self {
        Self {
            n_bins: 8,
            k: 1,
            scores: Vector::zeros(0),
            support: Vec::new(),
            fitted: false,
        }
    }
}

impl MutualInfoClassif {
    /// Selector with `n_bins` and top-`k`.
    pub fn new(n_bins: usize, k: usize) -> Self {
        Self {
            n_bins: n_bins.max(2),
            k: k.max(1),
            ..Self::default()
        }
    }

    /// Estimated I(X_j; Y) in nats.
    pub fn scores(&self) -> &Vector {
        &self.scores
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }
}

impl Fit for MutualInfoClassif {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::TargetLeakageSuspected)
                .severity(Severity::Advisory)
                .message(
                    "MutualInfoClassif estimates I(X;Y) on the same labels later used for training",
                )
                .build(),
        );
        let (n, p) = x.shape();
        self.scores = Vector::zeros(p);
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
            self.scores[j] = histogram_mi(&col, y.as_slice(), self.n_bins);
        }
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|a, b| {
            self.scores[*b]
                .partial_cmp(&self.scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.support = vec![false; p];
        for &j in order.iter().take(self.k.min(p)) {
            self.support[j] = true;
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for MutualInfoClassif {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Build a design whose column `j` is `y[t - lags[j]]` (leading rows are NaN).
pub fn lag_features(y: &Vector, lags: &[usize]) -> Matrix {
    let n = y.len();
    let p = lags.len();
    Matrix::from_fn(n, p, |i, j| {
        let lag = lags[j];
        if i >= lag {
            y[i - lag]
        } else {
            f64::NAN
        }
    })
}

/// Trailing rolling mean of window `window` (NaN until the window is full).
pub fn rolling_mean(y: &Vector, window: usize) -> Vector {
    let w = window.max(1);
    let mut out = Vector::zeros(y.len());
    let mut run = 0.0;
    let mut q: Vec<f64> = Vec::new();
    for i in 0..y.len() {
        let v = y[i];
        q.push(v);
        if v.is_finite() {
            run += v;
        }
        if q.len() > w {
            let old = q.remove(0);
            if old.is_finite() {
                run -= old;
            }
        }
        if q.len() == w {
            let k = q.iter().filter(|x| x.is_finite()).count() as f64;
            out[i] = if k > 0.0 { run / k } else { f64::NAN };
        } else {
            out[i] = f64::NAN;
        }
    }
    out
}

fn select_columns(x: &Matrix, support: &[bool]) -> Matrix {
    let cols: Vec<usize> = support
        .iter()
        .enumerate()
        .filter_map(|(j, s)| if *s { Some(j) } else { None })
        .collect();
    if cols.is_empty() {
        return Matrix::zeros(x.nrows(), 0);
    }
    Matrix::from_fn(x.nrows(), cols.len(), |i, t| x.get(i, cols[t]))
}

fn take_columns(x: &Matrix, cols: &[usize]) -> Matrix {
    Matrix::from_fn(x.nrows(), cols.len(), |i, t| x.get(i, cols[t]))
}

fn pearson_sq(a: &[f64], sa: signlred::SliceStats, b: &[f64], sb: signlred::SliceStats) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 || sa.std() == 0.0 || sb.std() == 0.0 {
        return 0.0;
    }
    let mut s = 0.0;
    let mut k = 0.0;
    for i in 0..n {
        if a[i].is_finite() && b[i].is_finite() {
            s += (a[i] - sa.mean) * (b[i] - sb.mean);
            k += 1.0;
        }
    }
    if k < 2.0 {
        return 0.0;
    }
    let r = s / ((k - 1.0) * sa.std() * sb.std());
    if r.is_finite() {
        r * r
    } else {
        0.0
    }
}

fn rbf_entry(a: &Matrix, ia: usize, b: &Matrix, ib: usize, gamma: f64) -> f64 {
    let mut d2 = 0.0;
    for c in 0..a.ncols().min(b.ncols()) {
        let d = a.get(ia, c) - b.get(ib, c);
        d2 += d * d;
    }
    (-gamma * d2).exp()
}

fn constant_col_issue(index: usize, stats: SliceStats) -> Issue {
    Issue::builder(IssueCode::ConstantFeature)
        .message(format!(
            "feature {index} is constant on [{:.6e}, {:.6e}]",
            stats.min, stats.max
        ))
        .metric("feature_index", index as f64)
        .metric("feature_std", stats.std())
        .build()
}

fn histogram_mi(x: &[f64], y: &[f64], n_bins: usize) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }
    let st = slice_stats(x);
    let mut classes: Vec<i64> = Vec::new();
    for &v in y.iter().take(n) {
        if v.is_finite() {
            let lab = v.round() as i64;
            if !classes.contains(&lab) {
                classes.push(lab);
            }
        }
    }
    classes.sort_unstable();
    if classes.is_empty() || st.count < 2 {
        return 0.0;
    }
    let nb = n_bins.max(2);
    let span = (st.max - st.min).max(1e-15);
    let mut joint = vec![0.0; nb * classes.len()];
    let mut mx = vec![0.0; nb];
    let mut my = vec![0.0; classes.len()];
    let mut tot: f64 = 0.0;
    for i in 0..n {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        let mut b = ((x[i] - st.min) / span * nb as f64).floor() as usize;
        if b >= nb {
            b = nb - 1;
        }
        let lab = y[i].round() as i64;
        let c = match classes.iter().position(|k| *k == lab) {
            Some(c) => c,
            None => continue,
        };
        joint[b * classes.len() + c] += 1.0;
        mx[b] += 1.0;
        my[c] += 1.0;
        tot += 1.0;
    }
    if tot <= 0.0 {
        return 0.0;
    }
    let mut mi: f64 = 0.0;
    for b in 0..nb {
        for c in 0..classes.len() {
            let pxy = joint[b * classes.len() + c] / tot;
            let px = mx[b] / tot;
            let py = my[c] / tot;
            if pxy > 0.0 && px > 0.0 && py > 0.0 {
                mi += pxy * (pxy / (px * py)).ln();
            }
        }
    }
    if !mi.is_finite() {
        0.0
    } else {
        mi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Fit, FitUnsupervised, Transform};
    use ojizou_san::Session;

    #[test]
    fn variance_threshold_drops_constant() {
        let x = Matrix::from_fn(10, 2, |i, j| if j == 0 { 5.0 } else { i as f64 });
        let session = Session::new("vt", "fit");
        let mut vt = VarianceThreshold::new(0.0);
        vt.fit_unsupervised(&x, &session).expect("fit");
        let z = vt.transform(&x, &session).expect("transform").value;
        assert_eq!(z.ncols(), 1);
        assert!((z.get(3, 0) - 3.0).abs() < 1e-12);
        assert!(!vt.support()[0]);
        assert!(vt.support()[1]);
    }

    #[test]
    fn select_k_best_keeps_signal() {
        let x = Matrix::from_fn(20, 3, |i, j| match j {
            0 => i as f64,
            1 => ((i * 17 + 5) % 7) as f64,
            _ => ((i * 3) % 2) as f64,
        });
        let y = Vector::from_iter((0..20).map(|i| i as f64));
        let session = Session::new("skb", "fit");
        let mut sel = SelectKBest::new(1);
        sel.fit(&x, &y, &session).expect("fit");
        assert!(sel.support()[0], "scores={:?}", sel.scores().as_slice());
        assert_eq!(sel.support().iter().filter(|s| **s).count(), 1);
        let z = sel.transform(&x, &session).expect("transform").value;
        assert_eq!(z.ncols(), 1);
        assert!((z.get(5, 0) - 5.0).abs() < 1e-12);
        assert!(session.ledger().len() > 0);
    }
}
