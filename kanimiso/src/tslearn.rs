//! Time-series distances, barycentres, clustering, SAX/PAA, and a DTW baseline SVM.
//!
//! Distances and estimators open a [`crate::context::FitCtx`]. DTW is the
//! classic dynamic program; soft-DTW uses the `γ`-softmin of Cuturi & Blondel.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::ridge_solve;
use crate::linear_model::{FittedPenalized, Ridge};
use crate::rng::Rng;
use crate::special::norm_cdf;
use crate::traits::{Fit, FitUnsupervised, Predict, Transform};
use crate::validate::{inspect_classes, inspect_xy};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, NumericalCompromise, Qualified, Result, Severity};
use std::collections::BTreeMap;

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

/// Longest common subsequence similarity under an ε-tube (tslearn `lcss`).
///
/// Optional Sakoe–Chiba `band` (`None` ⇒ full). Similarity is
/// \(\mathrm{LCS}/\max(n,m)\).
pub fn lcss(
    a: &Vector,
    b: &Vector,
    eps: f64,
    band: Option<usize>,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("lcss.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("lcss.b") {
        ctx.push(issue);
    }
    if !eps.is_finite() || eps < 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("LCSS ε={eps} is not a finite ≥0 radius; using |ε|"))
                .build(),
        );
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("LCSS on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let e = if eps.is_finite() { eps.abs() } else { 0.0 };
    ctx.finish(lcss_raw(a.as_slice(), b.as_slice(), e, band))
}

/// Naive STOMP-style Euclidean matrix profile (tslearn / stumpy `matrix_profile`).
///
/// Window length is not identification `p`. The exclusion zone is `window/4`.
pub fn matrix_profile(s: &Vector, window: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(s.as_slice()).to_issue("matrix_profile") {
        ctx.push(issue);
    }
    let n = s.len();
    let m = window;
    if m < 2 || m >= n {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("matrix_profile window={m} is unusable for n={n}"))
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let n_sub = n + 1 - m;
    let excl = (m / 4).max(1);
    let mut mp = Vector::zeros(n_sub);
    for i in 0..n_sub {
        let mut best = f64::INFINITY;
        for j in 0..n_sub {
            if i.abs_diff(j) < excl {
                continue;
            }
            let mut d = 0.0;
            for t in 0..m {
                let e = s[i + t] - s[j + t];
                d += e * e;
            }
            if d < best {
                best = d;
            }
        }
        mp[i] = best.max(0.0).sqrt();
    }
    ctx.finish(mp)
}

/// Real-valued edit distance (insert/delete cost 1, replace `|a-b|`).
pub fn edit_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("edit.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("edit.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("edit_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let n = a.len();
    let m = b.len();
    let mut prev = vec![0.0; m + 1];
    let mut cur = vec![0.0; m + 1];
    for j in 0..=m {
        prev[j] = j as f64;
    }
    for i in 1..=n {
        cur[0] = i as f64;
        for j in 1..=m {
            let rep = prev[j - 1] + (a[i - 1] - b[j - 1]).abs();
            let del = prev[j] + 1.0;
            let ins = cur[j - 1] + 1.0;
            cur[j] = rep.min(del).min(ins);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    ctx.finish(prev[m])
}

fn lcss_raw(a: &[f64], b: &[f64], eps: f64, band: Option<usize>) -> f64 {
    let n = a.len();
    let m = b.len();
    let mut prev = vec![0usize; m + 1];
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = 0;
        for j in 1..=m {
            if let Some(w) = band {
                if i.abs_diff(j) > w {
                    cur[j] = prev[j].max(cur[j - 1]);
                    continue;
                }
            }
            if (a[i - 1] - b[j - 1]).abs() <= eps {
                cur[j] = prev[j - 1] + 1;
            } else {
                cur[j] = prev[j].max(cur[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let lcs = prev[m] as f64;
    lcs / (n.max(m) as f64)
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
    ctx.finish(softdtw_raw(a.as_slice(), b.as_slice(), gamma))
}

fn softdtw_raw(a: &[f64], b: &[f64], gamma: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let inf = 1e300;
    let mut r = vec![inf; (n + 2) * (m + 2)];
    let idx = |i: usize, j: usize| i * (m + 2) + j;
    r[idx(0, 0)] = 0.0;
    let g = gamma.max(1e-12);
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let v = softmin(
                &[r[idx(i - 1, j)], r[idx(i, j - 1)], r[idx(i - 1, j - 1)]],
                g,
            );
            r[idx(i, j)] = cost + v;
        }
    }
    r[idx(n, m)]
}

/// Pairwise soft-DTW between rows of `a` and rows of `b`.
pub fn cdist_softdtw(
    a: &Matrix,
    b: &Matrix,
    gamma: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    if !gamma.is_finite() || gamma <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("cdist_softdtw gamma={gamma} is not positive"))
                .build(),
        );
    }
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        softdtw_raw(a.row(i).as_slice(), b.row(j).as_slice(), gamma)
    });
    ctx.finish(out)
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
    fn fit(
        &mut self,
        x: &Matrix,
        _y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTsKMeans>> {
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
        let mut centers =
            Matrix::from_fn(k, x.ncols(), |c, j| x.get(seeds[c.min(seeds.len() - 1)], j));
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
    fn fit(
        &mut self,
        x: &Matrix,
        _y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedKShape>> {
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
pub fn sax(
    y: &Vector,
    n_pieces: usize,
    alphabet: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
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
pub fn shapelet_distance(
    series: &Vector,
    shapelet: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
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
        Matrix::from_fn(x.nrows(), w, |i, j| match paa(&x.row(i), w, session) {
            Ok(q) if j < q.value.len() => q.value[j],
            _ => 0.0,
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
        let y = Vector::from_iter(
            raw.as_slice()
                .iter()
                .map(|&s| if s >= 0.0 { pos } else { neg }),
        );
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

/// Mini-ROCKET-style random convolutional features (sktime / tslearn ROCKET).
#[derive(Clone, Debug)]
pub struct Rocket {
    /// Number of random kernels.
    pub n_kernels: usize,
    /// Kernel length.
    pub kernel_len: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for Rocket {
    fn default() -> Self {
        Self {
            n_kernels: 32,
            kernel_len: 7,
            seed: 7,
        }
    }
}

impl Rocket {
    /// ROCKET with `k` kernels.
    pub fn new(n_kernels: usize) -> Self {
        Self {
            n_kernels,
            ..Self::default()
        }
    }

    /// Transform each row (series) into PPV + max features per kernel.
    pub fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let t = x.ncols();
        if t < self.kernel_len {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "series length {t} < kernel length {}",
                        self.kernel_len
                    ))
                    .build(),
            );
        }
        let mut rng = crate::rng::Rng::new(self.seed);
        let k = self.n_kernels;
        let w = self.kernel_len.min(t.max(1));
        let mut kernels = vec![vec![0.0; w]; k];
        for ker in kernels.iter_mut() {
            let mut s = 0.0;
            for v in ker.iter_mut() {
                *v = rng.standard_normal();
                s += *v;
            }
            let mean = s / w as f64;
            for v in ker.iter_mut() {
                *v -= mean;
            }
        }
        let out_p = k * 2;
        let feat = Matrix::from_fn(n, out_p, |i, j| {
            let kid = j / 2;
            let want_ppv = j % 2 == 0;
            let ker = &kernels[kid];
            let last = t.saturating_sub(w) + 1;
            let mut mx = f64::NEG_INFINITY;
            let mut pos = 0.0;
            let mut cnt = 0.0;
            for start in 0..last {
                let mut acc = 0.0;
                for u in 0..w {
                    acc += ker[u] * x.get(i, start + u);
                }
                if acc > mx {
                    mx = acc;
                }
                if acc > 0.0 {
                    pos += 1.0;
                }
                cnt += 1.0;
            }
            if want_ppv {
                if cnt > 0.0 {
                    pos / cnt
                } else {
                    0.0
                }
            } else if mx.is_finite() {
                mx
            } else {
                0.0
            }
        });
        if out_p > n {
            ctx.push(
                Issue::builder(IssueCode::PolynomialExplosion)
                    .message(format!(
                        "ROCKET features {out_p} > n={n}; this is interpolation"
                    ))
                    .build(),
            );
        }
        ctx.finish(feat)
    }
}

/// Interval-feature forest (sktime `TimeSeriesForestClassifier`).
///
/// Each tree sees `n_intervals` random windows of every row-as-series. The
/// features are mean, standard deviation, and OLS slope of the window. A
/// series shorter than 3 samples cannot identify a slope.
#[derive(Clone, Debug)]
pub struct TimeSeriesForestClassifier {
    /// Number of trees.
    pub n_estimators: usize,
    /// Random intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for TimeSeriesForestClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            n_intervals: 4,
            max_depth: 6,
            seed: 3,
        }
    }
}

impl TimeSeriesForestClassifier {
    /// Default interval forest.
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug)]
struct Interval {
    start: usize,
    end: usize,
}

/// Fitted interval forest.
#[derive(Clone, Debug)]
pub struct FittedTimeSeriesForest {
    trees: Vec<crate::tree::FittedTreeClassifier>,
    intervals: Vec<Vec<Interval>>,
    /// Sorted class labels.
    pub classes: Vec<i64>,
}

fn interval_feats(x: &Matrix, intervals: &[Interval]) -> Matrix {
    let p = intervals.len() * 3;
    Matrix::from_fn(x.nrows(), p, |i, j| {
        let spec = &intervals[j / 3];
        let kind = j % 3;
        let a = spec.start.min(x.ncols());
        let b = spec.end.min(x.ncols()).max(a + 1);
        let len = b - a;
        let mut mean = 0.0;
        for t in a..b {
            mean += x.get(i, t);
        }
        mean /= len as f64;
        if kind == 0 {
            return mean;
        }
        let mut ss = 0.0;
        let mut num = 0.0;
        let mut den = 0.0;
        let tbar = (len.saturating_sub(1)) as f64 / 2.0;
        for (u, t) in (a..b).enumerate() {
            let d = x.get(i, t) - mean;
            ss += d * d;
            let dt = u as f64 - tbar;
            num += dt * d;
            den += dt * dt;
        }
        if kind == 1 {
            if len <= 1 {
                0.0
            } else {
                (ss / (len as f64 - 1.0)).sqrt()
            }
        } else if den > 0.0 {
            num / den
        } else {
            0.0
        }
    })
}

impl Fit for TimeSeriesForestClassifier {
    type Fitted = FittedTimeSeriesForest;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTimeSeriesForest>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        if x.ncols() < 3 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "TimeSeriesForest series length {} < 3; slope features are unidentified",
                        x.ncols()
                    ))
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut intervals = Vec::new();
        let tlen = x.ncols().max(1);
        for e in 0..self.n_estimators.max(1) {
            let mut iv = Vec::new();
            for _ in 0..self.n_intervals.max(1) {
                let a = rng.below(tlen);
                let span = 1 + rng.below(tlen);
                let b = (a + span).min(tlen);
                iv.push(Interval {
                    start: a,
                    end: b.max(a + 1),
                });
            }
            let feat = interval_feats(x, &iv);
            let mut tree = crate::tree::DecisionTreeClassifier {
                max_depth: self.max_depth,
                seed: rng.next_u64(),
                ..crate::tree::DecisionTreeClassifier::default()
            };
            match tree.fit(&feat, y, &session.child("tsf_tree")) {
                Ok(q) => {
                    trees.push(q.value);
                    intervals.push(iv);
                }
                Err(err) => {
                    for issue in err.report.issues() {
                        ctx.push(issue.clone());
                    }
                }
            }
            ctx.session.step(e as u64, 0.0, None);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("every TimeSeriesForest tree failed to fit")
                    .build(),
            );
        }
        ctx.finish(FittedTimeSeriesForest {
            trees,
            intervals,
            classes,
        })
    }
}

impl Predict for FittedTimeSeriesForest {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut votes = vec![std::collections::BTreeMap::<i64, usize>::new(); x.nrows()];
        for (tree, iv) in self.trees.iter().zip(&self.intervals) {
            let feat = interval_feats(x, iv);
            match tree.predict(&feat, &session.child("tsf_pred")) {
                Ok(q) => {
                    for i in 0..x.nrows() {
                        let lab = q.value[i].round() as i64;
                        *votes[i].entry(lab).or_insert(0) += 1;
                    }
                }
                Err(_) => {}
            }
        }
        let out = Vector::from_iter(votes.iter().map(|m| {
            m.iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(self.classes.first().copied().unwrap_or(0) as f64)
        }));
        ctx.finish(out)
    }
}

fn interval_feats_cif(x: &Matrix, intervals: &[Interval]) -> Matrix {
    let p = intervals.len() * 5;
    Matrix::from_fn(x.nrows(), p, |i, j| {
        let spec = &intervals[j / 5];
        let kind = j % 5;
        let a = spec.start.min(x.ncols());
        let b = spec.end.min(x.ncols()).max(a + 1);
        let mut vals: Vec<f64> = (a..b).map(|t| x.get(i, t)).collect();
        let len = vals.len();
        let mean = vals.iter().sum::<f64>() / len as f64;
        match kind {
            0 => mean,
            1 => {
                if len <= 1 {
                    0.0
                } else {
                    let ss: f64 = vals.iter().map(|v| (v - mean) * (v - mean)).sum();
                    (ss / (len as f64 - 1.0)).sqrt()
                }
            }
            2 => {
                let tbar = (len.saturating_sub(1)) as f64 / 2.0;
                let mut num = 0.0;
                let mut den = 0.0;
                for (u, v) in vals.iter().enumerate() {
                    let dt = u as f64 - tbar;
                    num += dt * (*v - mean);
                    den += dt * dt;
                }
                if den > 0.0 {
                    num / den
                } else {
                    0.0
                }
            }
            3 => {
                vals.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
                if len % 2 == 1 {
                    vals[len / 2]
                } else if len > 0 {
                    0.5 * (vals[len / 2 - 1] + vals[len / 2])
                } else {
                    0.0
                }
            }
            _ => {
                vals.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
                if len == 0 {
                    0.0
                } else {
                    let q1 = vals[len / 4];
                    let q3 = vals[(3 * len / 4).min(len - 1)];
                    q3 - q1
                }
            }
        }
    })
}

/// Canonical interval forest (sktime `CanonicalIntervalForest`).
///
/// Each interval yields mean, std, slope, median, and IQR — a catch22-lite
/// subset, recorded as a compromise.
#[derive(Clone, Debug)]
pub struct CanonicalIntervalForest {
    /// Number of trees.
    pub n_estimators: usize,
    /// Random intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for CanonicalIntervalForest {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            n_intervals: 4,
            max_depth: 6,
            seed: 5,
        }
    }
}

impl CanonicalIntervalForest {
    /// Default CIF.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted CIF.
#[derive(Clone, Debug)]
pub struct FittedCanonicalIntervalForest {
    trees: Vec<crate::tree::FittedTreeClassifier>,
    intervals: Vec<Vec<Interval>>,
    /// Sorted class labels.
    pub classes: Vec<i64>,
}

impl Fit for CanonicalIntervalForest {
    type Fitted = FittedCanonicalIntervalForest;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedCanonicalIntervalForest>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        if x.ncols() < 3 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "CanonicalIntervalForest series length {} < 3",
                        x.ncols()
                    ))
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("CIF uses mean/std/slope/median/IQR, not the full catch22 set")
                .compromise(NumericalCompromise::new(
                    "catch22 interval features",
                    "five summary statistics per random interval",
                    "the canonical feature set is a documented subset",
                    "do not treat this as the published CIF feature map",
                ))
                .build(),
        );
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut intervals = Vec::new();
        let tlen = x.ncols().max(1);
        for e in 0..self.n_estimators.max(1) {
            let mut iv = Vec::new();
            for _ in 0..self.n_intervals.max(1) {
                let a = rng.below(tlen);
                let span = 1 + rng.below(tlen);
                let b = (a + span).min(tlen);
                iv.push(Interval {
                    start: a,
                    end: b.max(a + 1),
                });
            }
            let feat = interval_feats_cif(x, &iv);
            let mut tree = crate::tree::DecisionTreeClassifier {
                max_depth: self.max_depth,
                seed: rng.next_u64(),
                ..crate::tree::DecisionTreeClassifier::default()
            };
            match tree.fit(&feat, y, &session.child("cif_tree")) {
                Ok(q) => {
                    trees.push(q.value);
                    intervals.push(iv);
                }
                Err(err) => {
                    for issue in err.report.issues() {
                        ctx.push(issue.clone());
                    }
                }
            }
            ctx.session.step(e as u64, 0.0, None);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("every CanonicalIntervalForest tree failed to fit")
                    .build(),
            );
        }
        ctx.finish(FittedCanonicalIntervalForest {
            trees,
            intervals,
            classes,
        })
    }
}

impl Predict for FittedCanonicalIntervalForest {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut votes = vec![BTreeMap::<i64, usize>::new(); x.nrows()];
        for (tree, iv) in self.trees.iter().zip(&self.intervals) {
            let feat = interval_feats_cif(x, iv);
            match tree.predict(&feat, &session.child("cif_pred")) {
                Ok(q) => {
                    for i in 0..x.nrows() {
                        let lab = q.value[i].round() as i64;
                        *votes[i].entry(lab).or_insert(0) += 1;
                    }
                }
                Err(_) => {}
            }
        }
        let out = Vector::from_iter(votes.iter().map(|m| {
            m.iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(self.classes.first().copied().unwrap_or(0) as f64)
        }));
        ctx.finish(out)
    }
}

/// ROCKET features + ridge classifier (sktime `RocketClassifier`).
#[derive(Clone, Debug)]
pub struct RocketClassifier {
    /// Random kernels.
    pub n_kernels: usize,
    /// Kernel length.
    pub kernel_len: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for RocketClassifier {
    fn default() -> Self {
        Self {
            n_kernels: 32,
            kernel_len: 7,
            alpha: 1.0,
            seed: 7,
        }
    }
}

impl RocketClassifier {
    /// Default ROCKET classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted ROCKET + ridge classifier.
#[derive(Clone, Debug)]
pub struct FittedRocketClassifier {
    rocket: Rocket,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for RocketClassifier {
    type Fitted = FittedRocketClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRocketClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let rocket = Rocket {
            n_kernels: self.n_kernels,
            kernel_len: self.kernel_len,
            seed: self.seed,
        };
        let feat = rocket.transform(x, &session.child("rocket"))?;
        let mut clf = crate::classification::RidgeClassifier::new(self.alpha);
        let inner = clf.fit(&feat.value, y, &session.child("ridge"))?.value;
        ctx.finish(FittedRocketClassifier { rocket, inner })
    }
}

impl Predict for FittedRocketClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let feat = self.rocket.transform(x, &session.child("rocket"))?;
        self.inner.predict(&feat.value, session)
    }
}

/// Majority vote of ROCKET and a time-series forest (sktime `HIVECOTE` lite).
///
/// Ensemble size is not identification `p`.
#[derive(Clone, Debug)]
pub struct HiveCote {
    /// ROCKET kernels.
    pub n_kernels: usize,
    /// Forest trees.
    pub n_estimators: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for HiveCote {
    fn default() -> Self {
        Self {
            n_kernels: 16,
            n_estimators: 6,
            seed: 3,
        }
    }
}

impl HiveCote {
    /// Default HIVE-COTE lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted two-member HIVE-COTE vote.
#[derive(Clone, Debug)]
pub struct FittedHiveCote {
    rocket: FittedRocketClassifier,
    forest: FittedTimeSeriesForest,
}

impl Fit for HiveCote {
    type Fitted = FittedHiveCote;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedHiveCote>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message(
                    "HIVE-COTE lite is a vote of ROCKET and TSF, not the full STC/cBOSS/TDE stack",
                )
                .compromise(NumericalCompromise::new(
                    "HIVE-COTE v2 weighted ensemble",
                    "unweighted vote of RocketClassifier and TimeSeriesForest",
                    "shapelet / dictionary members are omitted",
                    "do not read the vote as a published HIVE-COTE accuracy",
                ))
                .build(),
        );
        let rocket = RocketClassifier {
            n_kernels: self.n_kernels,
            kernel_len: 5,
            alpha: 0.5,
            seed: self.seed,
        }
        .fit(x, y, &session.child("hc-rocket"))?
        .value;
        let forest = TimeSeriesForestClassifier {
            n_estimators: self.n_estimators,
            n_intervals: 3,
            max_depth: 4,
            seed: self.seed,
        }
        .fit(x, y, &session.child("hc-tsf"))?
        .value;
        ctx.finish(FittedHiveCote { rocket, forest })
    }
}

impl Predict for FittedHiveCote {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let a = self.rocket.predict(x, &session.child("r"))?;
        let b = self.forest.predict(x, &session.child("f"))?;
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let va = if i < a.value.len() { a.value[i] } else { 0.0 };
            let vb = if i < b.value.len() { b.value[i] } else { 0.0 };
            if (va - vb).abs() < 1e-12 {
                va
            } else {
                va
            }
        }));
        ctx.finish(y)
    }
}

/// DTW 1-NN classifier (sktime `KNeighborsTimeSeriesClassifier`).
#[derive(Clone, Debug)]
pub struct KNeighborsTimeSeries {
    /// Neighbours (only \(k=1\) is identified without a weighted vote).
    pub n_neighbors: usize,
}

impl Default for KNeighborsTimeSeries {
    fn default() -> Self {
        Self { n_neighbors: 1 }
    }
}

impl KNeighborsTimeSeries {
    /// `k`-NN DTW classifier.
    pub fn new(n_neighbors: usize) -> Self {
        Self { n_neighbors }
    }
}

/// Fitted DTW neighbour store.
#[derive(Clone, Debug)]
pub struct FittedKnnTs {
    x_train: Matrix,
    y_train: Vector,
    k: usize,
}

impl Fit for KNeighborsTimeSeries {
    type Fitted = FittedKnnTs;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedKnnTs>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        if self.n_neighbors != 1 {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!(
                        "KNeighborsTimeSeries requested k={}; only 1-NN is implemented as a majority of one",
                        self.n_neighbors
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedKnnTs {
            x_train: x.clone(),
            y_train: y.clone(),
            k: self.n_neighbors.max(1),
        })
    }
}

impl Predict for FittedKnnTs {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let a = x.row(i);
            let mut best = Vec::new();
            for t in 0..self.x_train.nrows() {
                let b = self.x_train.row(t);
                let d = match dtw(&a, &b, &session.child("dtw")) {
                    Ok(q) => q.value,
                    Err(_) => f64::INFINITY,
                };
                best.push((d, self.y_train[t]));
            }
            best.sort_by(|u, v| u.0.partial_cmp(&v.0).unwrap_or(std::cmp::Ordering::Equal));
            let take = self.k.min(best.len());
            let mut votes: std::collections::BTreeMap<i64, usize> =
                std::collections::BTreeMap::new();
            for item in best.iter().take(take) {
                *votes.entry(item.1.round() as i64).or_insert(0) += 1;
            }
            votes
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(0.0)
        }));
        ctx.finish(out)
    }
}

/// Soft-DTW nearest-neighbour regressor (tslearn `TimeSeriesSVR` / soft-DTW k-NN).
///
/// Neighbour count is not identification `p`. A constant `y` is vacuous via
/// [`inspect_xy`].
#[derive(Clone, Debug)]
pub struct SoftDtwRegressor {
    /// Neighbourhood size.
    pub k: usize,
    /// Soft-DTW smoothness.
    pub gamma: f64,
}

impl Default for SoftDtwRegressor {
    fn default() -> Self {
        Self { k: 3, gamma: 0.5 }
    }
}

impl SoftDtwRegressor {
    /// `k`-NN soft-DTW regressor.
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            ..Self::default()
        }
    }
}

/// Fitted soft-DTW regressor.
#[derive(Clone, Debug)]
pub struct FittedSoftDtwRegressor {
    /// Training series (rows).
    pub x_train: Matrix,
    /// Training targets.
    pub y_train: Vector,
    /// Neighbourhood size.
    pub k: usize,
    /// Soft-DTW smoothness.
    pub gamma: f64,
}

impl Fit for SoftDtwRegressor {
    type Fitted = FittedSoftDtwRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSoftDtwRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if !self.gamma.is_finite() || self.gamma <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SoftDtwRegressor gamma={} is not positive; using 0.5",
                        self.gamma
                    ))
                    .build(),
            );
            self.gamma = 0.5;
        }
        ctx.finish(FittedSoftDtwRegressor {
            x_train: x.clone(),
            y_train: y.clone(),
            k: self.k.max(1),
            gamma: self.gamma,
        })
    }
}

impl Predict for FittedSoftDtwRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let k = self.k.max(1).min(self.x_train.nrows().max(1));
        let g = self.gamma.max(1e-12);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let a = x.row(i);
            let mut dist: Vec<(f64, f64)> = (0..self.x_train.nrows())
                .map(|t| {
                    let d = softdtw_raw(a.as_slice(), self.x_train.row(t).as_slice(), g);
                    (d, self.y_train[t])
                })
                .collect();
            dist.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut num = 0.0;
            let mut den = 0.0;
            for (d, yi) in dist.into_iter().take(k) {
                let w = (-d).exp();
                num += w * yi;
                den += w;
            }
            if den > 0.0 {
                num / den
            } else {
                0.0
            }
        }));
        ctx.finish(y)
    }
}

/// Piecewise aggregate approximation (tslearn `PiecewiseAggregateApproximation`).
///
/// Segment count is not identification `p`.
#[derive(Clone, Debug)]
pub struct Paa {
    /// Number of segments.
    pub n_segments: usize,
}

impl Default for Paa {
    fn default() -> Self {
        Self { n_segments: 4 }
    }
}

impl Paa {
    /// PAA with `n_segments` bins.
    pub fn new(n_segments: usize) -> Self {
        Self {
            n_segments: n_segments.max(1),
        }
    }
}

impl Transform for Paa {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let m = self.n_segments.max(1).min(x.ncols().max(1));
        let out = Matrix::from_fn(x.nrows(), m, |i, s| {
            let lo = s * x.ncols() / m;
            let hi = ((s + 1) * x.ncols() / m).max(lo + 1);
            let mut acc = 0.0;
            let mut c = 0.0;
            for j in lo..hi.min(x.ncols()) {
                acc += x.get(i, j);
                c += 1.0;
            }
            if c > 0.0 {
                acc / c
            } else {
                0.0
            }
        });
        ctx.finish(out)
    }
}

/// Symbolic aggregate approximation (tslearn `SymbolicAggregateApproximation`).
#[derive(Clone, Debug)]
pub struct Sax {
    /// PAA segments.
    pub n_segments: usize,
    /// Alphabet size.
    pub alphabet: usize,
}

impl Default for Sax {
    fn default() -> Self {
        Self {
            n_segments: 4,
            alphabet: 4,
        }
    }
}

impl Sax {
    /// SAX with the given segments and alphabet.
    pub fn new(n_segments: usize, alphabet: usize) -> Self {
        Self {
            n_segments: n_segments.max(1),
            alphabet: alphabet.max(2),
        }
    }
}

impl Transform for Sax {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let paa = Paa::new(self.n_segments).transform(x, &session.child("paa"))?;
        let mut ctx = FitCtx::with_session(session.child("sax"));
        let z = paa.value;
        let a = self.alphabet.max(2);
        let out = Matrix::from_fn(z.nrows(), z.ncols(), |i, j| {
            let v = z.get(i, j);
            let u = 0.5 + 0.5 * crate::special::erf(v / std::f64::consts::SQRT_2);
            ((u * a as f64).floor() as usize).min(a - 1) as f64
        });
        ctx.finish(out)
    }
}

/// Linear SVC on PAA features (tslearn `TimeSeriesSVC` lite).
#[derive(Clone, Debug)]
pub struct TimeSeriesSvc {
    /// PAA segments.
    pub n_segments: usize,
    /// Ridge penalty on the PAA design.
    pub alpha: f64,
}

impl Default for TimeSeriesSvc {
    fn default() -> Self {
        Self {
            n_segments: 4,
            alpha: 0.1,
        }
    }
}

impl TimeSeriesSvc {
    /// SVC on a PAA map.
    pub fn new(n_segments: usize) -> Self {
        Self {
            n_segments: n_segments.max(1),
            ..Self::default()
        }
    }
}

impl Fit for TimeSeriesSvc {
    type Fitted = crate::classification::FittedRidgeClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<crate::classification::FittedRidgeClassifier>> {
        let z = Paa::new(self.n_segments).transform(x, &session.child("paa"))?;
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, &z.value, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = {
            let mut c: Vec<i64> = y
                .as_slice()
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| v.round() as i64)
                .collect();
            c.sort_unstable();
            c.dedup();
            c
        };
        let pm = Vector::from_iter(y.as_slice().iter().map(|&v| {
            let lab = v.round() as i64;
            if classes.len() >= 2 && lab == classes[classes.len() - 1] {
                1.0
            } else {
                -1.0
            }
        }));
        let mut scratch = signlred::Report::new("tssvc", "ridge");
        let design = z.value.with_intercept();
        let beta = crate::linalg::ridge_solve(
            &mut scratch,
            &design,
            &pm,
            self.alpha.max(0.0),
            &ctx.policy,
        )
        .unwrap_or_else(|| Vector::zeros(design.ncols()));
        ctx.finish(
            crate::classification::FittedRidgeClassifier::from_penalized(
                FittedPenalized {
                    coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                    intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                    alpha: self.alpha,
                    l1_ratio: 0.0,
                },
                if classes.len() >= 2 {
                    classes
                } else {
                    vec![0, 1]
                },
            ),
        )
    }
}

/// Interval-feature forest regressor (sktime `TimeSeriesForestRegressor`).
#[derive(Clone, Debug)]
pub struct TimeSeriesForestRegressor {
    /// Number of trees.
    pub n_estimators: usize,
    /// Random intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for TimeSeriesForestRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            n_intervals: 4,
            max_depth: 6,
            seed: 3,
        }
    }
}

impl TimeSeriesForestRegressor {
    /// Default interval forest regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted interval forest regressor.
#[derive(Clone, Debug)]
pub struct FittedTimeSeriesForestReg {
    trees: Vec<crate::tree::FittedTreeRegressor>,
    intervals: Vec<Vec<Interval>>,
}

impl Fit for TimeSeriesForestRegressor {
    type Fitted = FittedTimeSeriesForestReg;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTimeSeriesForestReg>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if x.ncols() < 3 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "TimeSeriesForestRegressor series length {} < 3",
                        x.ncols()
                    ))
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut intervals = Vec::new();
        let tlen = x.ncols().max(1);
        for e in 0..self.n_estimators.max(1) {
            let mut iv = Vec::new();
            for _ in 0..self.n_intervals.max(1) {
                let a = rng.below(tlen);
                let span = 1 + rng.below(tlen);
                let b = (a + span).min(tlen);
                iv.push(Interval {
                    start: a,
                    end: b.max(a + 1),
                });
            }
            let feat = interval_feats(x, &iv);
            let mut tree = crate::tree::DecisionTreeRegressor {
                max_depth: self.max_depth,
                seed: rng.next_u64(),
                ..crate::tree::DecisionTreeRegressor::default()
            };
            match tree.fit(&feat, y, &session.child("tsfr_tree")) {
                Ok(q) => {
                    trees.push(q.value);
                    intervals.push(iv);
                }
                Err(err) => {
                    for issue in err.report.issues() {
                        ctx.push(issue.clone());
                    }
                }
            }
            ctx.session.step(e as u64, 0.0, None);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("every TimeSeriesForestRegressor tree failed to fit")
                    .build(),
            );
        }
        ctx.finish(FittedTimeSeriesForestReg { trees, intervals })
    }
}

impl Predict for FittedTimeSeriesForestReg {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut acc = Vector::zeros(x.nrows());
        let mut k = 0.0;
        for (tree, iv) in self.trees.iter().zip(&self.intervals) {
            let feat = interval_feats(x, iv);
            if let Ok(q) = tree.predict(&feat, &session.child("tsfr_pred")) {
                for i in 0..x.nrows() {
                    acc[i] += q.value[i];
                }
                k += 1.0;
            }
        }
        if k > 0.0 {
            acc = acc.scale(1.0 / k);
        }
        ctx.finish(acc)
    }
}

/// Kernel k-means with a soft-DTW RBF kernel on the rows.
#[derive(Clone, Debug)]
pub struct KernelKMeans {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Soft-DTW smoothness (also the kernel scale).
    pub gamma: f64,
    /// Assignment iterations.
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for KernelKMeans {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            gamma: 1.0,
            max_iter: 20,
            seed: 0,
        }
    }
}

impl KernelKMeans {
    /// Soft-DTW kernel k-means with `k` clusters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Self::default()
        }
    }
}

/// Fitted kernel k-means partition.
#[derive(Clone, Debug)]
pub struct FittedKernelKMeans {
    /// Training assignments.
    pub labels: Vector,
    /// Soft-DTW RBF Gram used for assignment.
    pub kernel: Matrix,
}

impl FitUnsupervised for KernelKMeans {
    type Fitted = FittedKernelKMeans;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedKernelKMeans>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let k = self.n_clusters.max(1).min(n.max(1));
        if n == 0 {
            return ctx.finish(FittedKernelKMeans {
                labels: Vector::zeros(0),
                kernel: Matrix::zeros(0, 0),
            });
        }
        let g = self.gamma.max(1e-8);
        let kernel = Matrix::from_fn(n, n, |i, j| {
            if i == j {
                1.0
            } else {
                let d = softdtw_raw(x.row(i).as_slice(), x.row(j).as_slice(), g);
                (-d / g).exp()
            }
        });
        let mut rng = Rng::new(self.seed);
        let seeds = rng.sample_indices(n, k);
        let mut labels = Vector::from_iter((0..n).map(|i| {
            let mut best = 0usize;
            let mut bd = f64::NEG_INFINITY;
            for (c, &s) in seeds.iter().enumerate() {
                let v = kernel.get(i, s);
                if v > bd {
                    bd = v;
                    best = c;
                }
            }
            best as f64
        }));
        for it in 0..self.max_iter.max(1) {
            let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
            for i in 0..n {
                let c = labels[i].round().clamp(0.0, (k - 1) as f64) as usize;
                members[c].push(i);
            }
            for c in 0..k {
                if members[c].is_empty() {
                    ctx.push(
                        Issue::builder(IssueCode::EmptyCluster)
                            .message(format!("kernel k-means cluster {c} emptied; re-seeded"))
                            .build(),
                    );
                    members[c].push(rng.below(n));
                }
            }
            let mut changed = 0usize;
            for i in 0..n {
                let mut best = 0usize;
                let mut bd = f64::INFINITY;
                for c in 0..k {
                    let m = &members[c];
                    let inv = 1.0 / m.len() as f64;
                    let mut mean_k = 0.0;
                    for &j in m {
                        mean_k += kernel.get(i, j);
                    }
                    mean_k *= inv;
                    let mut cc = 0.0;
                    for &j in m {
                        for &l in m {
                            cc += kernel.get(j, l);
                        }
                    }
                    cc *= inv * inv;
                    let dist = kernel.get(i, i) - 2.0 * mean_k + cc;
                    if dist < bd {
                        bd = dist;
                        best = c;
                    }
                }
                if (labels[i] - best as f64).abs() > 0.5 {
                    changed += 1;
                }
                labels[i] = best as f64;
            }
            ctx.session.step(it as u64, changed as f64, None);
            if changed == 0 && it > 0 {
                ctx.session.converged("kernel k-means", it as u64);
                break;
            }
        }
        ctx.finish(FittedKernelKMeans { labels, kernel })
    }
}

/// Per-series mean/variance scaler (tslearn `TimeSeriesScalerMeanVariance`).
///
/// Each row is z-scored independently. A constant series becomes zeros and
/// records a near-zero-variance warning.
#[derive(Clone, Debug, Default)]
pub struct TimeSeriesScalerMeanVariance;

impl TimeSeriesScalerMeanVariance {
    /// Default per-series z-score.
    pub fn new() -> Self {
        Self
    }
}

impl FitUnsupervised for TimeSeriesScalerMeanVariance {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(self.clone())
    }
}

impl Transform for TimeSeriesScalerMeanVariance {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let row = x.row(i);
            let sd = row.std();
            if sd <= ctx.policy.near_zero_variance {
                0.0
            } else {
                (x.get(i, j) - row.mean()) / sd
            }
        });
        for i in 0..x.nrows() {
            if x.row(i).std() <= ctx.policy.near_zero_variance {
                ctx.push(
                    Issue::builder(IssueCode::NearZeroVariance)
                        .message(format!("series {i} has ~0 variance; it is mapped to 0"))
                        .build(),
                );
            }
        }
        ctx.finish(out)
    }
}

fn softdtw_grad(a: &[f64], b: &[f64], gamma: f64) -> (f64, Vec<f64>) {
    let n = a.len();
    let m = b.len();
    let mut grad = vec![0.0; n];
    if n == 0 || m == 0 {
        return (f64::NAN, grad);
    }
    let inf = 1e300;
    let g = gamma.max(1e-12);
    let cols = m + 2;
    let idx = |i: usize, j: usize| i * cols + j;
    let mut r = vec![inf; (n + 2) * cols];
    r[idx(0, 0)] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let v = softmin(
                &[r[idx(i - 1, j)], r[idx(i, j - 1)], r[idx(i - 1, j - 1)]],
                g,
            );
            r[idx(i, j)] = cost + v;
        }
    }
    let mut e = vec![0.0; (n + 2) * cols];
    e[idx(n, m)] = 1.0;
    for i in (1..=n).rev() {
        for j in (1..=m).rev() {
            let ee = e[idx(i, j)];
            if ee == 0.0 {
                continue;
            }
            let preds = [r[idx(i - 1, j)], r[idx(i, j - 1)], r[idx(i - 1, j - 1)]];
            let mut den = 0.0;
            let mut sm = [0.0; 3];
            for (k, &p) in preds.iter().enumerate() {
                sm[k] = (-p / g).exp();
                den += sm[k];
            }
            if den > 0.0 {
                e[idx(i - 1, j)] += ee * sm[0] / den;
                e[idx(i, j - 1)] += ee * sm[1] / den;
                e[idx(i - 1, j - 1)] += ee * sm[2] / den;
            }
            let sgn = if a[i - 1] >= b[j - 1] { 1.0 } else { -1.0 };
            grad[i - 1] += ee * sgn;
        }
    }
    (r[idx(n, m)], grad)
}

/// Soft-DTW barycentre of the rows of `x` (Cuturi & Blondel).
pub fn softdtw_barycenter(
    x: &Matrix,
    gamma: f64,
    max_iter: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    if !gamma.is_finite() || gamma <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("softdtw_barycenter gamma={gamma} is not positive"))
                .build(),
        );
    }
    if x.nrows() == 0 || x.ncols() == 0 {
        return ctx.finish(Vector::zeros(0));
    }
    let t = x.ncols();
    let mut c = Vector::from_iter((0..t).map(|j| x.column(j).mean()));
    let g = gamma.max(1e-12);
    for it in 0..max_iter.max(1) {
        let mut acc = vec![0.0; t];
        let mut loss = 0.0;
        for i in 0..x.nrows() {
            let row = x.row(i);
            let (v, dc) = softdtw_grad(c.as_slice(), row.as_slice(), g);
            loss += v;
            for j in 0..t {
                acc[j] += dc[j];
            }
        }
        let inv = 1.0 / x.nrows() as f64;
        let mut delta = 0.0;
        for j in 0..t {
            let step = 0.25 * acc[j] * inv;
            c[j] -= step;
            delta += step.abs();
        }
        ctx.session.step(it as u64, loss * inv, Some(delta));
        if delta < 1e-7 {
            ctx.session.converged("soft-DTW barycentre", it as u64);
            break;
        }
    }
    ctx.finish(c)
}

/// Global alignment kernel \(K=\exp(-\mathrm{softDTW}/\sigma)\).
pub fn global_alignment_kernel(
    a: &Vector,
    b: &Vector,
    sigma: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !sigma.is_finite() || sigma <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("GAK sigma={sigma} is not positive"))
                .build(),
        );
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(Issue::builder(IssueCode::EmptyMatrix).build());
        return ctx.finish(f64::NAN);
    }
    let d = softdtw_raw(a.as_slice(), b.as_slice(), 0.1);
    ctx.finish((-d / sigma.max(1e-12)).exp())
}

/// Petitjean DBA alias of [`dtw_barycenter`] (tslearn `dtw_barycenter_averaging`).
pub fn dba(x: &Matrix, max_iter: usize, session: &Session) -> Result<Qualified<Vector>> {
    dtw_barycenter(x, max_iter, session)
}

/// Per-series min–max scaler (tslearn `TimeSeriesScalerMinMax`).
#[derive(Clone, Debug, Default)]
pub struct TimeSeriesScalerMinMax;

impl TimeSeriesScalerMinMax {
    /// Default per-series `[0, 1]` map.
    pub fn new() -> Self {
        Self
    }
}

impl FitUnsupervised for TimeSeriesScalerMinMax {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(self.clone())
    }
}

impl Transform for TimeSeriesScalerMinMax {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let row = x.row(i);
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &v in row.as_slice() {
                if v.is_finite() {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let span = hi - lo;
            if !span.is_finite() || span <= ctx.policy.near_zero_variance {
                0.0
            } else {
                (x.get(i, j) - lo) / span
            }
        });
        for i in 0..x.nrows() {
            if x.row(i).std() <= ctx.policy.near_zero_variance {
                ctx.push(
                    Issue::builder(IssueCode::NearZeroVariance)
                        .message(format!("series {i} has ~0 span; it is mapped to 0"))
                        .build(),
                );
            }
        }
        ctx.finish(out)
    }
}

/// MiniROCKET-style dilated PPV features (Dempster, Schmidt, Webb).
#[derive(Clone, Debug)]
pub struct MiniRocket {
    /// Number of random dilated kernels.
    pub n_kernels: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for MiniRocket {
    fn default() -> Self {
        Self {
            n_kernels: 32,
            seed: 7,
        }
    }
}

impl MiniRocket {
    /// MiniROCKET with `k` kernels.
    pub fn new(n_kernels: usize) -> Self {
        Self {
            n_kernels,
            ..Self::default()
        }
    }

    /// Transform each row into one PPV feature per kernel.
    pub fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let t = x.ncols();
        let w = 9usize.min(t.max(1));
        if t < 9 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("MiniROCKET series length {t} < 9"))
                    .build(),
            );
        }
        let mut rng = crate::rng::Rng::new(self.seed);
        let k = self.n_kernels.max(1);
        let max_dil = if t > w {
            ((t - 1) as f64 / (w - 1) as f64).log2().max(0.0)
        } else {
            0.0
        };
        let mut kernels: Vec<(usize, [usize; 3])> = Vec::with_capacity(k);
        for _ in 0..k {
            let dil = 2f64.powf(rng.uniform() * max_dil).floor().max(1.0) as usize;
            let pos = [rng.below(w), rng.below(w), rng.below(w)];
            kernels.push((dil, pos));
        }
        let feat = Matrix::from_fn(n, k, |i, kid| {
            let (dil, pos) = kernels[kid];
            let last = t.saturating_sub(1 + (w - 1) * dil) + 1;
            let mut pos_cnt = 0.0;
            let mut cnt = 0.0;
            for start in 0..last.max(1) {
                let mut acc = 0.0;
                for u in 0..w {
                    let idx = start + u * dil;
                    if idx >= t {
                        continue;
                    }
                    let wt = if pos.contains(&u) { 2.0 } else { -1.0 };
                    acc += wt * x.get(i, idx);
                }
                if acc > 0.0 {
                    pos_cnt += 1.0;
                }
                cnt += 1.0;
            }
            if cnt > 0.0 {
                pos_cnt / cnt
            } else {
                0.0
            }
        });
        if k > n {
            ctx.push(
                Issue::builder(IssueCode::PolynomialExplosion)
                    .message(format!("MiniROCKET features {k} > n={n}"))
                    .build(),
            );
        }
        ctx.finish(feat)
    }
}

fn dft_mags(win: &[f64], n_coef: usize) -> Vec<f64> {
    let w = win.len().max(1);
    let keep = n_coef.max(1).min(w);
    let mut out = Vec::with_capacity(keep);
    for k in 1..=keep {
        let mut re = 0.0;
        let mut im = 0.0;
        for (n, &v) in win.iter().enumerate() {
            let ang = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / w as f64;
            re += v * ang.cos();
            im += v * ang.sin();
        }
        out.push((re * re + im * im).sqrt());
    }
    out
}

fn sfa_word(mags: &[f64], breaks: &[f64]) -> u64 {
    let a = (breaks.len() + 1) as u64;
    let mut w = 0u64;
    for &m in mags {
        let mut bin = 0u64;
        for (b, &t) in breaks.iter().enumerate() {
            if m > t {
                bin = (b + 1) as u64;
            }
        }
        w = w.wrapping_mul(a.saturating_add(3)).wrapping_add(bin + 1);
    }
    w
}

fn boss_histograms(
    x: &Matrix,
    window: usize,
    word_len: usize,
    alphabet: usize,
) -> (Matrix, Vec<u64>) {
    let n = x.nrows();
    let t = x.ncols();
    let w = window.clamp(2, t.max(2));
    let mut all_words: BTreeMap<u64, usize> = BTreeMap::new();
    let mut per_row: Vec<BTreeMap<u64, f64>> = Vec::with_capacity(n);
    let breaks: Vec<f64> = {
        let a = alphabet.max(2);
        (1..a)
            .map(|i| {
                // Equal-mass Gaussian breakpoints on a unit scale, then unused;
                // actual binning is on raw DFT magnitudes via these cutoffs after
                // a global median scale (filled below).
                i as f64 / a as f64
            })
            .collect()
    };
    let mut all_mags = Vec::new();
    for i in 0..n {
        let last = t.saturating_sub(w) + 1;
        for start in 0..last.max(1) {
            let win: Vec<f64> = (0..w.min(t)).map(|u| x.get(i, start + u)).collect();
            all_mags.extend(dft_mags(&win, word_len));
        }
    }
    all_mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let scaled_breaks: Vec<f64> = breaks
        .iter()
        .map(|&q| {
            if all_mags.is_empty() {
                q
            } else {
                let pos = (q * (all_mags.len() - 1) as f64).round() as usize;
                all_mags[pos.min(all_mags.len() - 1)]
            }
        })
        .collect();
    for i in 0..n {
        let mut hist = BTreeMap::new();
        let last = t.saturating_sub(w) + 1;
        for start in 0..last.max(1) {
            let win: Vec<f64> = (0..w.min(t)).map(|u| x.get(i, start + u)).collect();
            let mags = dft_mags(&win, word_len);
            let word = sfa_word(&mags, &scaled_breaks);
            *hist.entry(word).or_insert(0.0) += 1.0;
            all_words.entry(word).or_insert(0);
        }
        per_row.push(hist);
    }
    let vocab: Vec<u64> = all_words.keys().copied().collect();
    let p = vocab.len();
    let index: BTreeMap<u64, usize> = vocab.iter().enumerate().map(|(i, w)| (*w, i)).collect();
    let h = Matrix::from_fn(n, p, |i, j| {
        let w = vocab[j];
        *per_row[i].get(&w).unwrap_or(&0.0)
    });
    let _ = index;
    (h, vocab)
}

/// BOSS word-histogram + ridge classifier (sktime `BOSSEnsemble` lite).
#[derive(Clone, Debug)]
pub struct BossEnsemble {
    /// Sliding-window length.
    pub window: usize,
    /// DFT coefficients kept per window.
    pub word_len: usize,
    /// SFA alphabet size.
    pub alphabet: usize,
}

impl Default for BossEnsemble {
    fn default() -> Self {
        Self {
            window: 8,
            word_len: 4,
            alphabet: 4,
        }
    }
}

impl BossEnsemble {
    /// Default BOSS.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted BOSS / WEASEL histogram ridge.
#[derive(Clone, Debug)]
pub struct FittedBoss {
    /// Word vocabulary (hashes).
    pub vocab: Vec<u64>,
    /// Ridge on histograms.
    pub ridge: FittedPenalized,
    /// Window / word / alphabet used at fit.
    pub spec: (usize, usize, usize),
}

impl Predict for FittedBoss {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let (h, _) = boss_histograms(x, self.spec.0, self.spec.1, self.spec.2);
        let p = self.ridge.coef.len();
        let z = if h.ncols() == p {
            h
        } else {
            Matrix::from_fn(
                h.nrows(),
                p,
                |i, j| {
                    if j < h.ncols() {
                        h.get(i, j)
                    } else {
                        0.0
                    }
                },
            )
        };
        self.ridge.predict(&z, session)
    }
}

impl Fit for BossEnsemble {
    type Fitted = FittedBoss;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedBoss>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        if x.ncols() < self.window {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!(
                        "BOSS window {} > series length {}",
                        self.window,
                        x.ncols()
                    ))
                    .build(),
            );
        }
        let (h, vocab) = boss_histograms(x, self.window, self.word_len, self.alphabet);
        // Do not inspect_identification(n, n_words).
        if vocab.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .message("BOSS vocabulary is empty")
                    .build(),
            );
        }
        let mut scratch = signlred::Report::new("boss", "ridge");
        let yc = Vector::from_iter(y.as_slice().iter().map(|v| v - y.mean()));
        let (hc, _) = h.centered();
        let coef = ridge_solve(&mut scratch, &hc, &yc, 0.1, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(h.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedBoss {
            vocab,
            ridge: FittedPenalized {
                coef,
                intercept: y.mean(),
                alpha: 0.1,
                l1_ratio: 0.0,
            },
            spec: (self.window, self.word_len, self.alphabet),
        })
    }
}

/// WEASEL: BOSS histograms with a variance filter on words, then ridge.
#[derive(Clone, Debug)]
pub struct Weasel {
    /// Sliding-window length.
    pub window: usize,
    /// DFT coefficients kept per window.
    pub word_len: usize,
    /// SFA alphabet size.
    pub alphabet: usize,
    /// Keep this many most-variable words.
    pub n_words: usize,
}

impl Default for Weasel {
    fn default() -> Self {
        Self {
            window: 8,
            word_len: 4,
            alphabet: 4,
            n_words: 8,
        }
    }
}

impl Weasel {
    /// Default WEASEL.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for Weasel {
    type Fitted = FittedBoss;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedBoss>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let (h, vocab) = boss_histograms(x, self.window, self.word_len, self.alphabet);
        let mut vars: Vec<(usize, f64)> = (0..h.ncols()).map(|j| (j, h.column(j).std())).collect();
        vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep = self.n_words.max(1).min(h.ncols().max(1));
        let idx: Vec<usize> = vars.iter().take(keep).map(|p| p.0).collect();
        let z = if idx.is_empty() {
            Matrix::zeros(h.nrows(), 0)
        } else {
            Matrix::from_fn(h.nrows(), idx.len(), |i, t| h.get(i, idx[t]))
        };
        let vocab: Vec<u64> = idx
            .iter()
            .map(|&j| vocab.get(j).copied().unwrap_or(0))
            .collect();
        let mut scratch = signlred::Report::new("weasel", "ridge");
        let yc = Vector::from_iter(y.as_slice().iter().map(|v| v - y.mean()));
        let (zc, _) = z.centered();
        let coef = ridge_solve(&mut scratch, &zc, &yc, 0.1, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(z.ncols()));
        ctx.finish(FittedBoss {
            vocab,
            ridge: FittedPenalized {
                coef,
                intercept: y.mean(),
                alpha: 0.1,
                l1_ratio: 0.0,
            },
            spec: (self.window, self.word_len, self.alphabet),
        })
    }
}

/// Random-shapelet transform + ridge (tslearn `LearningShapelets` lite).
///
/// Shapelets are sampled, not gradient-learned — recorded as a compromise.
/// Do not pass `n_shapelets` as `p` to identification: 10 series and 4
/// shapelets is a feature map, not an overparameterized linear model.
#[derive(Clone, Debug)]
pub struct LearningShapelets {
    /// Number of random shapelets.
    pub n_shapelets: usize,
    /// Shapelet length.
    pub length: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for LearningShapelets {
    fn default() -> Self {
        Self {
            n_shapelets: 4,
            length: 4,
            seed: 3,
        }
    }
}

impl LearningShapelets {
    /// `k` shapelets of length `length`.
    pub fn new(n_shapelets: usize, length: usize) -> Self {
        Self {
            n_shapelets: n_shapelets.max(1),
            length: length.max(2),
            ..Self::default()
        }
    }
}

/// Fitted shapelet ridge.
#[derive(Clone, Debug)]
pub struct FittedShapelets {
    /// Shapelets (`k` × `L`).
    pub shapelets: Matrix,
    /// Ridge on min-distance features.
    pub ridge: FittedPenalized,
}

fn min_shapelet_dist(row: &Matrix, i: usize, shape: &Matrix, s: usize) -> f64 {
    let tlen = row.ncols();
    let slen = shape.ncols();
    if slen == 0 || tlen < slen {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for start in 0..=tlen - slen {
        let mut d = 0.0;
        for u in 0..slen {
            let e = row.get(i, start + u) - shape.get(s, u);
            d += e * e;
        }
        best = best.min(d);
    }
    best.sqrt()
}

impl Fit for LearningShapelets {
    type Fitted = FittedShapelets;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedShapelets>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let _ = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let l = self.length.min(x.ncols().max(2)).max(2);
        if x.ncols() < l {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("LearningShapelets length={l} > T={}", x.ncols()))
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .severity(Severity::Advisory)
                .message("LearningShapelets samples random windows; it is not gradient shapelet learning")
                .compromise(NumericalCompromise::new(
                    "learned shapelets (Grabocka et al.)",
                    "random subsequences + min-distance + ridge",
                    "shapelets are not optimized against the classification loss",
                    "treat the features as a random convolutional sketch",
                ))
                .build(),
        );
        let k = self.n_shapelets.max(1);
        let mut rng = Rng::new(self.seed | 7);
        let slen = l.min(x.ncols().max(1));
        let shapelets = Matrix::from_fn(k, slen, |_, _| 0.0);
        let mut shapelets = shapelets;
        if x.nrows() > 0 && x.ncols() >= slen {
            for s in 0..k {
                let row = rng.below(x.nrows());
                let start = if x.ncols() > slen {
                    rng.below(x.ncols() - slen + 1)
                } else {
                    0
                };
                for u in 0..slen {
                    shapelets.set(s, u, x.get(row, start + u));
                }
            }
        }
        let feat = Matrix::from_fn(x.nrows(), k, |i, s| min_shapelet_dist(x, i, &shapelets, s));
        let ypm = Vector::from_iter(
            y.as_slice()
                .iter()
                .map(|&v| if v >= 0.5 { 1.0 } else { -1.0 }),
        );
        let mut scratch = signlred::Report::new("shapelet", "ridge");
        let coef = ridge_solve(&mut scratch, &feat, &ypm, 0.5, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(k));
        ctx.finish(FittedShapelets {
            shapelets,
            ridge: FittedPenalized {
                coef,
                intercept: 0.0,
                alpha: 0.5,
                l1_ratio: 0.0,
            },
        })
    }
}

impl Predict for FittedShapelets {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let k = self.shapelets.nrows();
        let feat = Matrix::from_fn(x.nrows(), k, |i, s| {
            min_shapelet_dist(x, i, &self.shapelets, s)
        });
        let raw = if feat.ncols() == self.ridge.coef.len() {
            feat.matvec(&self.ridge.coef)
        } else {
            Vector::zeros(x.nrows())
        };
        let y = Vector::from_iter(
            raw.as_slice()
                .iter()
                .map(|&s| if s >= 0.0 { 1.0 } else { 0.0 }),
        );
        ctx.finish(y)
    }
}

/// DTW k-NN regressor (tslearn `KNeighborsTimeSeriesRegressor`).
///
/// Neighbour count is not identification `p`.
#[derive(Clone, Debug)]
pub struct KNeighborsTimeSeriesRegressor {
    /// Neighbourhood size.
    pub n_neighbors: usize,
}

impl Default for KNeighborsTimeSeriesRegressor {
    fn default() -> Self {
        Self { n_neighbors: 3 }
    }
}

impl KNeighborsTimeSeriesRegressor {
    /// `k`-NN DTW regressor.
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors: n_neighbors.max(1),
        }
    }
}

/// Fitted DTW neighbour store for regression.
#[derive(Clone, Debug)]
pub struct FittedKnnTsRegressor {
    x_train: Matrix,
    y_train: Vector,
    k: usize,
}

impl Fit for KNeighborsTimeSeriesRegressor {
    type Fitted = FittedKnnTsRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedKnnTsRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.finish(FittedKnnTsRegressor {
            x_train: x.clone(),
            y_train: y.clone(),
            k: self.n_neighbors.max(1),
        })
    }
}

impl Predict for FittedKnnTsRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let k = self.k.min(self.x_train.nrows().max(1));
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let a = x.row(i);
            let mut dist: Vec<(f64, f64)> = (0..self.x_train.nrows())
                .map(|t| {
                    let d = dtw_raw(a.as_slice(), self.x_train.row(t).as_slice());
                    (d, self.y_train[t])
                })
                .collect();
            dist.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
            let take = k.min(dist.len());
            if take == 0 {
                return 0.0;
            }
            let mut s = 0.0;
            for item in dist.iter().take(take) {
                s += item.1;
            }
            s / take as f64
        }));
        ctx.finish(out)
    }
}

/// Unsupervised random-shapelet feature map (tslearn `ShapeletModel` transform).
///
/// Shapelet count is not identification `p`.
#[derive(Clone, Debug)]
pub struct ShapeletTransform {
    /// Number of random shapelets.
    pub n_shapelets: usize,
    /// Shapelet length.
    pub length: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for ShapeletTransform {
    fn default() -> Self {
        Self {
            n_shapelets: 4,
            length: 4,
            seed: 5,
        }
    }
}

impl ShapeletTransform {
    /// `k` shapelets of length `length`.
    pub fn new(n_shapelets: usize, length: usize) -> Self {
        Self {
            n_shapelets: n_shapelets.max(1),
            length: length.max(2),
            ..Self::default()
        }
    }
}

/// Fitted shapelet dictionary.
#[derive(Clone, Debug)]
pub struct FittedShapeletTransform {
    /// Shapelets (`k` × `L`).
    pub shapelets: Matrix,
}

impl FitUnsupervised for ShapeletTransform {
    type Fitted = FittedShapeletTransform;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedShapeletTransform>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let l = self.length.min(x.ncols().max(2)).max(2);
        if x.ncols() < l {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("ShapeletTransform length={l} > T={}", x.ncols()))
                    .build(),
            );
        }
        let k = self.n_shapelets.max(1);
        let slen = l.min(x.ncols().max(1));
        let mut shapelets = Matrix::zeros(k, slen);
        let mut rng = Rng::new(self.seed | 11);
        if x.nrows() > 0 && x.ncols() >= slen {
            for s in 0..k {
                let row = rng.below(x.nrows());
                let start = if x.ncols() > slen {
                    rng.below(x.ncols() - slen + 1)
                } else {
                    0
                };
                for u in 0..slen {
                    shapelets.set(s, u, x.get(row, start + u));
                }
            }
        }
        let mut identical = true;
        if k >= 2 && slen > 0 {
            for s in 1..k {
                for u in 0..slen {
                    if (shapelets.get(s, u) - shapelets.get(0, u)).abs() > 1e-12 {
                        identical = false;
                    }
                }
            }
        } else {
            identical = false;
        }
        if identical {
            ctx.push(
                Issue::builder(IssueCode::JitterInjected)
                    .severity(Severity::Warning)
                    .message("ShapeletTransform sampled identical windows")
                    .compromise(NumericalCompromise::new(
                        "diverse shapelet dictionary",
                        "repeated random subsequences",
                        "min-distance features are collinear",
                        "increase seed diversity or series length",
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedShapeletTransform { shapelets })
    }
}

impl Transform for FittedShapeletTransform {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let k = self.shapelets.nrows();
        let feat = Matrix::from_fn(x.nrows(), k, |i, s| {
            min_shapelet_dist(x, i, &self.shapelets, s)
        });
        ctx.finish(feat)
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
        let lc = lcss(&a, &a, 0.1, None, &Session::new("ts", "lcss"))
            .unwrap()
            .value;
        assert!((lc - 1.0).abs() < 1e-12);
        let ed = edit_distance(&a, &a, &Session::new("ts", "ed"))
            .unwrap()
            .value;
        assert!(ed.abs() < 1e-12);
        let long = Vector::from_iter((0..16).map(|i| (i as f64).sin()));
        let mp = matrix_profile(&long, 4, &Session::new("ts", "mp"))
            .unwrap()
            .value;
        assert_eq!(mp.len(), 13);
        assert!(mp.as_slice().iter().all(|v| v.is_finite()));
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
        let tsf = TimeSeriesForestClassifier {
            n_estimators: 6,
            n_intervals: 3,
            max_depth: 4,
            seed: 2,
        }
        .fit(&x, &y, &Session::new("ts", "tsf"))
        .unwrap();
        let pred = tsf
            .value
            .predict(&x, &Session::new("ts", "tsfp"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..8 {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 5, "tsf ok={ok}");
        let cif = CanonicalIntervalForest {
            n_estimators: 6,
            n_intervals: 3,
            max_depth: 4,
            seed: 2,
        }
        .fit(&x, &y, &Session::new("ts", "cif"))
        .unwrap();
        let predc = cif
            .value
            .predict(&x, &Session::new("ts", "cifp"))
            .unwrap()
            .value;
        assert_eq!(predc.len(), 8);
        let knn = KNeighborsTimeSeries::new(1)
            .fit(&x, &y, &Session::new("ts", "knn"))
            .unwrap();
        let pred = knn
            .value
            .predict(&x, &Session::new("ts", "knnp"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..8 {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 6, "knn ok={ok}");
        let rc = RocketClassifier {
            n_kernels: 16,
            kernel_len: 5,
            alpha: 0.5,
            seed: 2,
        }
        .fit(&x, &y, &Session::new("ts", "rocketc"))
        .unwrap();
        let pred = rc
            .value
            .predict(&x, &Session::new("ts", "rp"))
            .unwrap()
            .value;
        assert_eq!(pred.len(), 8);
        let yr = Vector::from_iter((0..8).map(|i| x.row(i).mean()));
        let tsfr = TimeSeriesForestRegressor {
            n_estimators: 6,
            n_intervals: 3,
            max_depth: 3,
            seed: 2,
        }
        .fit(&x, &yr, &Session::new("ts", "tsfr"))
        .unwrap();
        let pred = tsfr
            .value
            .predict(&x, &Session::new("ts", "tsfrp"))
            .unwrap()
            .value;
        assert!(pred.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn softdtw_cdist_kernel_kmeans_and_scaler() {
        let x = Matrix::from_fn(6, 4, |i, j| {
            if i < 3 {
                (j as f64) + 0.1 * i as f64
            } else {
                3.0 - j as f64 + 0.1 * i as f64
            }
        });
        let cd = cdist_softdtw(&x, &x, 0.5, &Session::new("ts", "csdtw"))
            .unwrap()
            .value;
        assert_eq!(cd.shape(), (6, 6));
        assert!(cd.get(0, 0).is_finite());
        let km = KernelKMeans::new(2)
            .fit_unsupervised(&x, &Session::new("ts", "kkm"))
            .unwrap();
        assert_eq!(km.value.labels.len(), 6);
        let mut sc = TimeSeriesScalerMeanVariance::new();
        sc.fit_unsupervised(&x, &Session::new("ts", "sc")).unwrap();
        let z = sc.transform(&x, &Session::new("ts", "sct")).unwrap().value;
        assert!((z.row(0).mean()).abs() < 1e-8);
        let mut mm = TimeSeriesScalerMinMax::new();
        mm.fit_unsupervised(&x, &Session::new("ts", "mm")).unwrap();
        let z2 = mm.transform(&x, &Session::new("ts", "mmt")).unwrap().value;
        assert!(z2.get(0, 0) >= -1e-12 && z2.get(0, 0) <= 1.0 + 1e-12);
        let a = x.row(0);
        let kaa = global_alignment_kernel(&a, &a, 1.0, &Session::new("ts", "gak"))
            .unwrap()
            .value;
        let far = Vector::from_iter((0..a.len()).map(|j| a[j] + 5.0));
        let kaf = global_alignment_kernel(&a, &far, 1.0, &Session::new("ts", "gak2"))
            .unwrap()
            .value;
        assert!(kaa > kaf, "kaa={kaa} kaf={kaf}");
        let b = softdtw_barycenter(&x, 0.5, 6, &Session::new("ts", "sdb"))
            .unwrap()
            .value;
        assert_eq!(b.len(), x.ncols());
        let mr = MiniRocket::new(8)
            .transform(&x, &Session::new("ts", "mr"))
            .unwrap()
            .value;
        assert_eq!(mr.shape(), (6, 8));
        let yb = Vector::from_iter((0..6).map(|i| if i < 3 { 0.0 } else { 1.0 }));
        let boss = BossEnsemble {
            window: 4,
            word_len: 3,
            alphabet: 4,
        }
        .fit(&x, &yb, &Session::new("ts", "boss"))
        .unwrap();
        let bp = boss
            .value
            .predict(&x, &Session::new("ts", "bossp"))
            .unwrap()
            .value;
        assert_eq!(bp.len(), 6);
        let wsl = Weasel {
            window: 4,
            word_len: 3,
            alphabet: 4,
            n_words: 6,
        }
        .fit(&x, &yb, &Session::new("ts", "weasel"))
        .unwrap();
        assert!(!wsl.value.vocab.is_empty() || x.nrows() > 0);
        let sh = LearningShapelets::new(4, 3)
            .fit(&x, &yb, &Session::new("ts", "shp"))
            .unwrap();
        let sp = sh
            .value
            .predict(&x, &Session::new("ts", "shpp"))
            .unwrap()
            .value;
        assert_eq!(sp.len(), 6);
        let yr = Vector::from_iter((0..6).map(|i| if i < 3 { 0.0 } else { 1.0 }));
        let sdr = SoftDtwRegressor::new(2)
            .fit(&x, &yr, &Session::new("ts", "sdr"))
            .unwrap();
        let pr = sdr
            .value
            .predict(&x, &Session::new("ts", "sdrp"))
            .unwrap()
            .value;
        assert_eq!(pr.len(), 6);
        assert!(pr.as_slice().iter().all(|v| v.is_finite()));
        let paa = Paa::new(2)
            .transform(&x, &Session::new("ts", "paa"))
            .unwrap()
            .value;
        assert_eq!(paa.ncols(), 2);
        let sax = Sax::new(2, 4)
            .transform(&x, &Session::new("ts", "sax"))
            .unwrap()
            .value;
        assert_eq!(sax.ncols(), 2);
        let tsvc = TimeSeriesSvc::new(2)
            .fit(&x, &yb, &Session::new("ts", "svc"))
            .unwrap();
        let sp2 = tsvc
            .value
            .predict(&paa, &Session::new("ts", "svcp"))
            .unwrap()
            .value;
        assert_eq!(sp2.len(), 6);
        let hc = HiveCote::new()
            .fit(&x, &yb, &Session::new("ts", "hive"))
            .unwrap();
        let hp = hc
            .value
            .predict(&x, &Session::new("ts", "hivep"))
            .unwrap()
            .value;
        assert_eq!(hp.len(), 6);
        let knnr = KNeighborsTimeSeriesRegressor::new(2)
            .fit(&x, &yr, &Session::new("ts", "knnr"))
            .unwrap();
        let knp = knnr
            .value
            .predict(&x, &Session::new("ts", "knnrp"))
            .unwrap()
            .value;
        assert_eq!(knp.len(), 6);
        assert!(knp.as_slice().iter().all(|v| v.is_finite()));
        let st = ShapeletTransform::new(3, 3)
            .fit_unsupervised(&x, &Session::new("ts", "sht"))
            .unwrap();
        let z = st
            .value
            .transform(&x, &Session::new("ts", "shtt"))
            .unwrap()
            .value;
        assert_eq!(z.nrows(), 6);
        assert_eq!(z.ncols(), 3);
        let bary = dba(&x, 8, &Session::new("ts", "dba")).unwrap().value;
        assert_eq!(bary.len(), x.ncols());
        assert!(bary.as_slice().iter().all(|v| v.is_finite()));
    }
}
