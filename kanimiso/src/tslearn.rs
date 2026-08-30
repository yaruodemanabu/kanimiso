//! Time-series distances, barycentres, clustering, SAX/PAA, and a DTW baseline SVM.
//!
//! Distances and estimators open a [`crate::context::FitCtx`]. DTW is the
//! classic dynamic program; soft-DTW uses the `γ`-softmin of Cuturi & Blondel.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linear_model::{FittedPenalized, Ridge};
use crate::rng::Rng;
use crate::special::norm_cdf;
use crate::traits::{Fit, Predict};
use crate::validate::{inspect_classes, inspect_xy};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Qualified, Result};

fn series_ok(a: &Vector) -> bool {
    a.as_slice().iter().any(|v| v.is_finite())
}

fn _use_series_ok(a: &Vector, ctx: &mut FitCtx) {
    if !series_ok(a) {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("series has no finite samples")
                .build(),
        );
    }
}

/// Classic DTW distance (absolute local cost, no window).
pub fn dtw(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("dtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("dtw.b") {
        ctx.push(issue);
    }
    _use_series_ok(a, &mut ctx);
    _use_series_ok(b, &mut ctx);
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("DTW on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(dtw_raw(a.as_slice(), b.as_slice()))
}

fn dtw_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    let inf: f64 = 1e300;
    let mut prev = vec![inf; m + 1];
    let mut cur = vec![inf; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        cur[0] = inf;
        for j in 1..=m {
            let cost: f64 = (a[i - 1] - b[j - 1]).abs();
            cur[j] = cost + prev[j].min(cur[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Pairwise DTW between rows of `a` and rows of `b` (each row is a series).
pub fn cdist_dtw(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        dtw_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn softmin(xs: &[f64], gamma: f64) -> f64 {
    let g = gamma.max(1e-12);
    let mut m = f64::INFINITY;
    for &v in xs {
        if v < m {
            m = v;
        }
    }
    if !m.is_finite() {
        return f64::INFINITY;
    }
    let mut s = 0.0;
    for &v in xs {
        s += (-(v - m) / g).exp();
    }
    m - g * s.ln()
}

/// Soft-DTW (Cuturi & Blondel) with smoothness `gamma`.
pub fn softdtw(a: &Vector, b: &Vector, gamma: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !gamma.is_finite() || gamma <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("softdtw gamma={gamma} is not positive"))
                .build(),
        );
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(Issue::builder(IssueCode::EmptyMatrix).build());
        return ctx.finish(f64::NAN);
    }
    let n = a.len();
    let m = b.len();
    let inf = 1e300;
    let mut r = vec![inf; (n + 2) * (m + 2)];
    let idx = |i: usize, j: usize| i * (m + 2) + j;
    r[idx(0, 0)] = 0.0;
    let g = gamma.max(1e-12);
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let v = softmin(&[r[idx(i - 1, j)], r[idx(i, j - 1)], r[idx(i - 1, j - 1)]], g);
            r[idx(i, j)] = cost + v;
        }
    }
    ctx.finish(r[idx(n, m)])
}

fn dtw_path(a: &[f64], b: &[f64]) -> Vec<(usize, usize)> {
    let n = a.len();
    let m = b.len();
    let inf: f64 = 1e300;
    let mut dp = vec![inf; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    dp[at(0, 0)] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            dp[at(i, j)] = cost
                + dp[at(i - 1, j)]
                    .min(dp[at(i, j - 1)])
                    .min(dp[at(i - 1, j - 1)]);
        }
    }
    let mut path = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let a1 = dp[at(i - 1, j)];
        let a2 = dp[at(i, j - 1)];
        let a3 = dp[at(i - 1, j - 1)];
        if a3 <= a1 && a3 <= a2 {
            i -= 1;
            j -= 1;
        } else if a1 <= a2 {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    path.reverse();
    path
}

/// DTW barycentre averaging (DBA) of the rows of `x`.
pub fn dtw_barycenter(x: &Matrix, max_iter: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    if x.nrows() == 0 || x.ncols() == 0 {
        return ctx.finish(Vector::zeros(0));
    }
    let t = x.ncols();
    let mut c = Vector::from_iter((0..t).map(|j| x.column(j).mean()));
    for it in 0..max_iter.max(1) {
        let mut acc = vec![0.0; t];
        let mut cnt = vec![0.0; t];
        for i in 0..x.nrows() {
            let s = x.row(i);
            let path = dtw_path(c.as_slice(), s.as_slice());
            for (ci, si) in path {
                acc[ci] += s[si];
                cnt[ci] += 1.0;
            }
        }
        let mut delta = 0.0;
        for j in 0..t {
            if cnt[j] > 0.0 {
                let v = acc[j] / cnt[j];
                delta += (v - c[j]).abs();
                c[j] = v;
            }
        }
        ctx.session.step(it as u64, delta, None);
        if delta < 1e-8 {
            ctx.session.converged("DBA", it as u64);
            break;
        }
    }
    ctx.finish(c)
}

/// k-means with DTW distance and DBA centroids.
#[derive(Clone, Debug)]
pub struct TimeSeriesKMeans {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Assignment / DBA iterations.
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for TimeSeriesKMeans {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            max_iter: 20,
            seed: 0,
        }
    }
}

impl TimeSeriesKMeans {
    /// DTW k-means with `k` clusters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Self::default()
        }
    }
}

/// Fitted DTW k-means.
#[derive(Clone, Debug)]
pub struct FittedTsKMeans {
    /// Centroids (`k × T`).
    pub centers: Matrix,
    /// Training assignments.
    pub labels: Vector,
}

impl Predict for FittedTsKMeans {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let s = x.row(i);
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for c in 0..self.centers.nrows() {
                let d = dtw_raw(s.as_slice(), self.centers.row(c).as_slice());
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            best as f64
        }));
        ctx.finish(y)
    }
}

impl Fit for TimeSeriesKMeans {
    type Fitted = FittedTsKMeans;
    fn fit(&mut self, x: &Matrix, _y: &Vector, session: &Session) -> Result<Qualified<FittedTsKMeans>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let k = self.n_clusters.max(1).min(n.max(1));
        if n == 0 {
            return ctx.finish(FittedTsKMeans {
                centers: Matrix::zeros(0, x.ncols()),
                labels: Vector::zeros(0),
            });
        }
        let mut rng = Rng::new(self.seed);
        let seeds = rng.sample_indices(n, k);
        let mut centers = Matrix::from_fn(k, x.ncols(), |c, j| x.get(seeds[c.min(seeds.len() - 1)], j));
        let mut labels = Vector::zeros(n);
        for it in 0..self.max_iter.max(1) {
            let mut changed = 0usize;
            for i in 0..n {
                let s = x.row(i);
                let mut best = 0usize;
                let mut bd = f64::INFINITY;
                for c in 0..k {
                    let d = dtw_raw(s.as_slice(), centers.row(c).as_slice());
                    if d < bd {
                        bd = d;
                        best = c;
                    }
                }
                if (labels[i] - best as f64).abs() > 0.5 {
                    changed += 1;
                }
                labels[i] = best as f64;
            }
            for c in 0..k {
                let members: Vec<usize> = (0..n).filter(|&i| labels[i] as usize == c).collect();
                if members.is_empty() {
                    ctx.push(
                        Issue::builder(IssueCode::DegenerateClusters)
                            .message(format!("DTW k-means cluster {c} emptied"))
                            .build(),
                    );
                    continue;
                }
                let sub = Matrix::from_fn(members.len(), x.ncols(), |i, j| x.get(members[i], j));
                if let Ok(q) = dtw_barycenter(&sub, 5, &session.child(format!("dba_{c}"))) {
                    for j in 0..x.ncols() {
                        centers.set(c, j, q.value[j]);
                    }
                }
            }
            ctx.session.step(it as u64, changed as f64, None);
            if changed == 0 && it > 0 {
                ctx.session.converged("DTW k-means assignment", it as u64);
                break;
            }
        }
        ctx.finish(FittedTsKMeans { centers, labels })
    }
}

fn znorm(s: &Vector) -> Vector {
    let m = s.mean();
    let sd = s.std().max(1e-12);
    Vector::from_iter(s.as_slice().iter().map(|v| (v - m) / sd))
}

fn ncc(a: &Vector, b: &Vector) -> (f64, isize) {
    let n = a.len();
    let m = b.len();
    let mut best = f64::NEG_INFINITY;
    let mut shift = 0isize;
    let max_sh = (n + m) as isize;
    for sh in -(max_sh / 2)..=(max_sh / 2) {
        let mut s = 0.0;
        let mut k = 0.0;
        for i in 0..n {
            let j = i as isize + sh;
            if j >= 0 && (j as usize) < m {
                s += a[i] * b[j as usize];
                k += 1.0;
            }
        }
        if k > 0.0 && s > best {
            best = s;
            shift = sh;
        }
    }
    (best, shift)
}

/// k-Shape: z-normalized series clustered by normalized cross-correlation.
#[derive(Clone, Debug)]
pub struct KShape {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Iterations.
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for KShape {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            max_iter: 20,
            seed: 0,
        }
    }
}

impl KShape {
    /// k-Shape with `k` clusters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Self::default()
        }
    }
}

/// Fitted k-Shape model.
#[derive(Clone, Debug)]
pub struct FittedKShape {
    /// Z-normalized centroids.
    pub centers: Matrix,
    /// Training assignments.
    pub labels: Vector,
}

impl Predict for FittedKShape {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let s = znorm(&x.row(i));
            let mut best = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for c in 0..self.centers.nrows() {
                let (v, _) = ncc(&s, &self.centers.row(c));
                if v > bv {
                    bv = v;
                    best = c;
                }
            }
            best as f64
        }));
        ctx.finish(y)
    }
}

impl Fit for KShape {
    type Fitted = FittedKShape;
    fn fit(&mut self, x: &Matrix, _y: &Vector, session: &Session) -> Result<Qualified<FittedKShape>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let t = x.ncols();
        let k = self.n_clusters.max(1).min(n.max(1));
        if n == 0 {
            return ctx.finish(FittedKShape {
                centers: Matrix::zeros(0, t),
                labels: Vector::zeros(0),
            });
        }
        let zn: Vec<Vector> = (0..n).map(|i| znorm(&x.row(i))).collect();
        let mut rng = Rng::new(self.seed);
        let seeds = rng.sample_indices(n, k);
        let mut centers = Matrix::from_fn(k, t, |c, j| zn[seeds[c.min(seeds.len() - 1)]][j]);
        let mut labels = Vector::zeros(n);
        for it in 0..self.max_iter.max(1) {
            for i in 0..n {
                let mut best = 0usize;
                let mut bv = f64::NEG_INFINITY;
                for c in 0..k {
                    let (v, _) = ncc(&zn[i], &centers.row(c));
                    if v > bv {
                        bv = v;
                        best = c;
                    }
                }
                labels[i] = best as f64;
            }
            for c in 0..k {
                let members: Vec<usize> = (0..n).filter(|&i| labels[i] as usize == c).collect();
                if members.is_empty() {
                    ctx.push(
                        Issue::builder(IssueCode::DegenerateClusters)
                            .message(format!("k-Shape cluster {c} emptied"))
                            .build(),
                    );
                    continue;
                }
                let mut acc = Vector::zeros(t);
                let centroid = centers.row(c);
                for &i in &members {
                    let (_, sh) = ncc(&zn[i], &centroid);
                    for j in 0..t {
                        let src = j as isize - sh;
                        if src >= 0 && (src as usize) < t {
                            acc[j] += zn[i][src as usize];
                        }
                    }
                }
                let mean = acc.scale(1.0 / members.len() as f64);
                let z = znorm(&mean);
                for j in 0..t {
                    centers.set(c, j, z[j]);
                }
            }
            ctx.session.step(it as u64, 0.0, None);
        }
        ctx.finish(FittedKShape { centers, labels })
    }
}

/// Piecewise aggregate approximation onto `n_pieces` windows.
pub fn paa(y: &Vector, n_pieces: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(y.as_slice()).to_issue("paa") {
        ctx.push(issue);
    }
    let w = n_pieces.max(1);
    if y.is_empty() {
        return ctx.finish(Vector::zeros(0));
    }
    let n = y.len();
    let out = Vector::from_iter((0..w).map(|k| {
        let lo = k * n / w;
        let hi = ((k + 1) * n / w).max(lo + 1).min(n);
        let mut s = 0.0;
        let mut c = 0.0;
        for i in lo..hi {
            if y[i].is_finite() {
                s += y[i];
                c += 1.0;
            }
        }
        if c > 0.0 {
            s / c
        } else {
            0.0
        }
    }));
    ctx.finish(out)
}

/// Symbolic aggregate approximation: PAA then Gaussian breakpoints.
pub fn sax(y: &Vector, n_pieces: usize, alphabet: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let z = znorm(y);
    let p = match paa(&z, n_pieces, &session.child("paa")) {
        Ok(q) => q.value,
        Err(e) => {
            ctx.push(e.primary);
            return ctx.finish(Vector::zeros(0));
        }
    };
    let a = alphabet.max(2);
    // Inverse-Φ breakpoints that split ℝ into `a` equal-mass bins.
    let mut cuts = Vec::with_capacity(a.saturating_sub(1));
    for k in 1..a {
        let q = k as f64 / a as f64;
        // Binary search Φ⁻¹(q).
        let mut lo = -8.0;
        let mut hi = 8.0;
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if norm_cdf(mid) < q {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        cuts.push(0.5 * (lo + hi));
    }
    let out = Vector::from_iter(p.as_slice().iter().map(|&v| {
        let mut sym = 0.0;
        for (i, &c) in cuts.iter().enumerate() {
            if v > c {
                sym = (i + 1) as f64;
            }
        }
        sym
    }));
    ctx.finish(out)
}

/// Minimum Euclidean distance between `shapelet` and any subsequence of `series`.
pub fn shapelet_distance(series: &Vector, shapelet: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if series.is_empty() || shapelet.is_empty() || shapelet.len() > series.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("shapelet longer than the series (or empty)")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let m = shapelet.len();
    let mut best = f64::INFINITY;
    for start in 0..=series.len() - m {
        let mut s = 0.0;
        for t in 0..m {
            let d = series[start + t] - shapelet[t];
            s += d * d;
        }
        if s < best {
            best = s;
        }
    }
    ctx.finish(best.sqrt())
}

/// Time-series classifier: linear model on PAA features + a DTW 1-NN baseline.
#[derive(Clone, Debug)]
pub struct TimeSeriesSvm {
    /// PAA length used as the linear feature map.
    pub n_pieces: usize,
    /// Ridge penalty on the PAA features.
    pub alpha: f64,
}

impl Default for TimeSeriesSvm {
    fn default() -> Self {
        Self {
            n_pieces: 8,
            alpha: 1.0,
        }
    }
}

impl TimeSeriesSvm {
    /// Default PAA-linear + DTW 1-NN classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted time-series SVM-style classifier.
#[derive(Clone, Debug)]
pub struct FittedTimeSeriesSvm {
    /// Training series (rows).
    pub x_train: Matrix,
    /// Training labels.
    pub y_train: Vector,
    /// Linear model on PAA features.
    pub linear: FittedPenalized,
    /// PAA length.
    pub n_pieces: usize,
    /// Classes.
    pub classes: Vec<i64>,
}

impl FittedTimeSeriesSvm {
    fn paa_matrix(&self, x: &Matrix, session: &Session) -> Matrix {
        let w = self.n_pieces.max(1);
        Matrix::from_fn(x.nrows(), w, |i, j| {
            match paa(&x.row(i), w, session) {
                Ok(q) if j < q.value.len() => q.value[j],
                _ => 0.0,
            }
        })
    }

    /// DTW 1-NN labels (the baseline).
    pub fn predict_dtw_nn(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("dtw_nn"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let s = x.row(i);
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for t in 0..self.x_train.nrows() {
                let d = dtw_raw(s.as_slice(), self.x_train.row(t).as_slice());
                if d < bd {
                    bd = d;
                    best = t;
                }
            }
            self.y_train[best]
        }));
        ctx.finish(y)
    }
}

impl Predict for FittedTimeSeriesSvm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let z = self.paa_matrix(x, &session.child("paa"));
        let raw = match self.linear.predict(&z, &session.child("linear")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vector::zeros(x.nrows())
            }
        };
        let pos = *self.classes.last().unwrap_or(&1) as f64;
        let neg = *self.classes.first().unwrap_or(&0) as f64;
        let y = Vector::from_iter(raw.as_slice().iter().map(|&s| if s >= 0.0 { pos } else { neg }));
        ctx.finish(y)
    }
}

impl Fit for TimeSeriesSvm {
    type Fitted = FittedTimeSeriesSvm;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTimeSeriesSvm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        let w = self.n_pieces.max(1);
        let z = Matrix::from_fn(x.nrows(), w, |i, j| {
            paa(&x.row(i), w, &session.child("paa"))
                .ok()
                .and_then(|q| {
                    if j < q.value.len() {
                        Some(q.value[j])
                    } else {
                        None
                    }
                })
                .unwrap_or(0.0)
        });
        let ypm = Vector::from_iter(y.as_slice().iter().map(|&v| {
            if classes.len() >= 2 && v.round() as i64 == classes[classes.len() - 1] {
                1.0
            } else {
                -1.0
            }
        }));
        let linear = match Ridge::new(self.alpha).fit(&z, &ypm, &session.child("ridge")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedPenalized {
                    coef: Vector::zeros(w),
                    intercept: 0.0,
                    alpha: self.alpha,
                    l1_ratio: 0.0,
                }
            }
        };
        ctx.finish(FittedTimeSeriesSvm {
            x_train: x.clone(),
            y_train: y.clone(),
            linear,
            n_pieces: w,
            classes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    #[test]
    fn dtw_identical_is_zero() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0, 2.0]);
        let d = dtw(&a, &a, &Session::new("ts", "dtw")).unwrap().value;
        assert!(d.abs() < 1e-12);
        let sd = softdtw(&a, &a, 0.1, &Session::new("ts", "sdtw"))
            .unwrap()
            .value;
        assert!(sd.is_finite());
    }

    #[test]
    fn paa_sax_shapelet() {
        let y = Vector::from_slice(&[0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0]);
        let p = paa(&y, 4, &Session::new("ts", "paa")).unwrap().value;
        assert_eq!(p.len(), 4);
        let s = sax(&y, 4, 4, &Session::new("ts", "sax")).unwrap().value;
        assert_eq!(s.len(), 4);
        let sh = Vector::from_slice(&[2.0, 3.0, 2.0]);
        let d = shapelet_distance(&y, &sh, &Session::new("ts", "sh"))
            .unwrap()
            .value;
        assert!(d < 0.2, "d={d}");
    }

    #[test]
    fn dtw_kmeans_and_svm() {
        let x = Matrix::from_fn(8, 6, |i, j| {
            if i < 4 {
                (j as f64).sin()
            } else {
                (j as f64).cos() + 2.0
            }
        });
        let y = Vector::from_iter((0..8).map(|i| if i < 4 { 0.0 } else { 1.0 }));
        let km = TimeSeriesKMeans::new(2)
            .fit(&x, &y, &Session::new("ts", "km"))
            .unwrap();
        assert_eq!(km.value.centers.nrows(), 2);
        let svm = TimeSeriesSvm {
            n_pieces: 4,
            alpha: 0.1,
        }
        .fit(&x, &y, &Session::new("ts", "svm"))
        .unwrap();
        let pred = svm
            .value
            .predict(&x, &Session::new("ts", "p"))
            .unwrap()
            .value;
        assert_eq!(pred.len(), 8);
        let nn = svm
            .value
            .predict_dtw_nn(&x, &Session::new("ts", "nn"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..8 {
            if (nn[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 6, "nn ok={ok}");
        let cd = cdist_dtw(&x, &x, &Session::new("ts", "cd")).unwrap().value;
        assert_eq!(cd.shape(), (8, 8));
        let ks = KShape::new(2)
            .fit(&x, &y, &Session::new("ts", "ks"))
            .unwrap();
        assert_eq!(ks.value.centers.nrows(), 2);
        let b = dtw_barycenter(&x, 4, &Session::new("ts", "dba"))
            .unwrap()
            .value;
        assert_eq!(b.len(), 6);
    }
}
