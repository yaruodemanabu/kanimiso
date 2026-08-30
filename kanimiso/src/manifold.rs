//! Manifold embeddings: MDS, Isomap, spectral, LLE, and exact t-SNE.
//!
//! Classical MDS drops negative eigenvalues of the double-centred Gram
//! ([`IssueCode::NegativeEigenvalueDropped`]). High Kruskal stress records
//! [`IssueCode::EmbeddingUnstable`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{symmetric_eigen, thin_svd};
use crate::traits::{FitUnsupervised, Transform};
use crate::validate::inspect_xy;
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, NumericalCompromise, Qualified, Report, Result, Severity};

fn sq_dist(x: &Matrix, i: usize, j: usize) -> f64 {
    let mut s = 0.0;
    for c in 0..x.ncols() {
        let d = x.get(i, c) - x.get(j, c);
        s += d * d;
    }
    s
}

fn euclid_matrix(x: &Matrix) -> Matrix {
    let n = x.nrows();
    Matrix::from_fn(
        n,
        n,
        |i, j| {
            if i == j {
                0.0
            } else {
                sq_dist(x, i, j).sqrt()
            }
        },
    )
}

fn double_center_sq(dist: &Matrix) -> Mat<f64> {
    let n = dist.nrows();
    let mut d2 = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let d = dist.get(i, j);
            d2[i * n + j] = d * d;
        }
    }
    let mut row = vec![0.0; n];
    let mut col = vec![0.0; n];
    let mut grand = 0.0;
    for i in 0..n {
        for j in 0..n {
            let v = d2[i * n + j];
            row[i] += v;
            col[j] += v;
            grand += v;
        }
    }
    let nf = n.max(1) as f64;
    for i in 0..n {
        row[i] /= nf;
        col[i] /= nf;
    }
    grand /= nf * nf;
    Mat::<f64>::from_fn(n, n, |i, j| {
        -0.5 * (d2[i * n + j] - row[i] - col[j] + grand)
    })
}

fn embed_from_gram(ctx: &mut FitCtx, gram: &Mat<f64>, n_components: usize) -> Matrix {
    let n = gram.nrows();
    let k = n_components.min(n);
    let Some((vals, vecs)) = symmetric_eigen(&mut ctx.report, gram, &ctx.policy) else {
        ctx.push(Issue::builder(IssueCode::EigenDidNotConverge).build());
        return Matrix::zeros(n, k);
    };
    let mut pairs: Vec<(f64, usize)> = vals
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut dropped = 0usize;
    for &(v, _) in &pairs {
        if v < -ctx.policy.rank_tol_relative {
            dropped += 1;
        }
    }
    if dropped > 0 {
        ctx.push(
            Issue::builder(IssueCode::NegativeEigenvalueDropped)
                .message(format!(
                    "{dropped} negative eigenvalues dropped from the Gram"
                ))
                .metric("n_negative", dropped as f64)
                .compromise(NumericalCompromise::new(
                    "PSD double-centred Gram",
                    "classical MDS using only nonnegative eigenpairs",
                    "the distance matrix is not Euclidean at working precision",
                    "embedded distances are a projection, not an isometry",
                ))
                .build(),
        );
    }
    let mut out = Matrix::zeros(n, k);
    let mut used = 0usize;
    for &(v, idx) in &pairs {
        if used >= k {
            break;
        }
        if v <= ctx.policy.rank_tol_relative {
            continue;
        }
        let s = v.sqrt();
        for i in 0..n.min(vecs.nrows()) {
            out.set(i, used, s * vecs[(i, idx)]);
        }
        used += 1;
    }
    if used < k {
        ctx.push(
            Issue::builder(IssueCode::ComponentsExceedRank)
                .message(format!("requested {k} MDS components, recovered {used}"))
                .build(),
        );
    }
    out
}

fn kruskal_stress(dist: &Matrix, y: &Matrix) -> f64 {
    let n = dist.nrows().min(y.nrows());
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist.get(i, j);
            let mut e = 0.0;
            for c in 0..y.ncols() {
                let t = y.get(i, c) - y.get(j, c);
                e += t * t;
            }
            let delta = e.sqrt();
            let r = d - delta;
            num += r * r;
            den += d * d;
        }
    }
    if den <= 0.0 {
        return f64::NAN;
    }
    (num / den).sqrt()
}

fn warn_stress(ctx: &mut FitCtx, stress: f64) {
    if stress.is_finite() && stress > 0.2 {
        ctx.push(
            Issue::builder(IssueCode::EmbeddingUnstable)
                .message(format!("Kruskal stress={stress:.4} > 0.2"))
                .metric("stress", stress)
                .build(),
        );
    }
}

/// Classical multidimensional scaling from a distance matrix or from Euclidean `X`.
#[derive(Clone, Debug)]
pub struct MDS {
    /// Embedding dimension.
    pub n_components: usize,
}

impl Default for MDS {
    fn default() -> Self {
        Self { n_components: 2 }
    }
}

impl MDS {
    /// MDS onto `k` coordinates.
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }

    /// Embed a symmetric distance matrix.
    pub fn fit_distances(
        &self,
        dist: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedEmbedding>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, dist, None, &ctx.policy);
        if dist.nrows() != dist.ncols() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("MDS distance matrix is not square")
                    .build(),
            );
        }
        let gram = double_center_sq(dist);
        let y = embed_from_gram(&mut ctx, &gram, self.n_components.max(1));
        let stress = kruskal_stress(dist, &y);
        warn_stress(&mut ctx, stress);
        ctx.finish(FittedEmbedding {
            embedding: y,
            stress,
        })
    }
}

/// Fitted manifold coordinates plus Kruskal stress.
#[derive(Clone, Debug)]
pub struct FittedEmbedding {
    /// Embedded coordinates (`n × k`).
    pub embedding: Matrix,
    /// Kruskal stress (NaN when undefined).
    pub stress: f64,
}

impl FitUnsupervised for MDS {
    type Fitted = FittedEmbedding;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedEmbedding>> {
        let dist = euclid_matrix(x);
        self.fit_distances(&dist, session)
    }
}

fn knn_graph(x: &Matrix, k: usize) -> Vec<Vec<usize>> {
    let n = x.nrows();
    let kk = k.max(1).min(n.saturating_sub(1).max(1));
    let mut g = vec![Vec::new(); n];
    for i in 0..n {
        let mut d: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (sq_dist(x, i, j), j))
            .collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        g[i] = d.into_iter().take(kk).map(|(_, j)| j).collect();
    }
    g
}

fn floyd(n: usize, edges: &[(usize, usize, f64)]) -> Matrix {
    let inf = 1e12;
    let mut d = Matrix::from_fn(n, n, |i, j| if i == j { 0.0 } else { inf });
    for &(i, j, w) in edges {
        if w < d.get(i, j) {
            d.set(i, j, w);
            d.set(j, i, w);
        }
    }
    for k in 0..n {
        for i in 0..n {
            let dik = d.get(i, k);
            if dik >= inf {
                continue;
            }
            for j in 0..n {
                let v = dik + d.get(k, j);
                if v < d.get(i, j) {
                    d.set(i, j, v);
                }
            }
        }
    }
    d
}

/// Isomap: k-NN geodesic distances (Floyd–Warshall) + classical MDS.
#[derive(Clone, Debug)]
pub struct Isomap {
    /// Neighbors per point.
    pub n_neighbors: usize,
    /// Embedding dimension.
    pub n_components: usize,
}

impl Default for Isomap {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            n_components: 2,
        }
    }
}

impl Isomap {
    /// Isomap with `k` neighbors onto `d` coordinates.
    pub fn new(n_neighbors: usize, n_components: usize) -> Self {
        Self {
            n_neighbors,
            n_components,
        }
    }
}

impl FitUnsupervised for Isomap {
    type Fitted = FittedEmbedding;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedEmbedding>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let g = knn_graph(x, self.n_neighbors);
        let mut edges = Vec::new();
        for i in 0..n {
            for &j in &g[i] {
                edges.push((i, j, sq_dist(x, i, j).sqrt()));
            }
        }
        let geo = floyd(n, &edges);
        let mut disconnected = 0usize;
        let mut finite_max = 0.0;
        for i in 0..n {
            for j in 0..n {
                let v = geo.get(i, j);
                if v >= 1e12 {
                    disconnected += 1;
                } else if v > finite_max {
                    finite_max = v;
                }
            }
        }
        if disconnected > 0 {
            ctx.push(
                Issue::builder(IssueCode::EmbeddingUnstable)
                    .message(format!(
                        "Isomap graph has {disconnected} disconnected pairs; filled with 2·max"
                    ))
                    .build(),
            );
        }
        let geo = {
            let fill = (2.0 * finite_max).max(1.0);
            Matrix::from_fn(n, n, |i, j| {
                let v = geo.get(i, j);
                if i == j {
                    0.0
                } else if v >= 1e12 {
                    fill
                } else {
                    v
                }
            })
        };
        let gram = double_center_sq(&geo);
        let y = embed_from_gram(&mut ctx, &gram, self.n_components.max(1));
        let stress = kruskal_stress(&geo, &y);
        warn_stress(&mut ctx, stress);
        ctx.finish(FittedEmbedding {
            embedding: y,
            stress,
        })
    }
}

fn rbf_affinity(x: &Matrix, gamma: f64) -> Mat<f64> {
    let n = x.nrows();
    Mat::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            0.0
        } else {
            (-gamma * sq_dist(x, i, j)).exp()
        }
    })
}

/// Laplacian eigenmaps (normalized spectral embedding).
#[derive(Clone, Debug)]
pub struct SpectralEmbedding {
    /// Embedding dimension (skips the constant eigenvector).
    pub n_components: usize,
    /// RBF `γ` in `exp(−γ ‖x−x'‖²)`.
    pub gamma: f64,
}

impl Default for SpectralEmbedding {
    fn default() -> Self {
        Self {
            n_components: 2,
            gamma: 1.0,
        }
    }
}

impl SpectralEmbedding {
    /// Spectral embedding onto `k` coordinates.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }
}

impl FitUnsupervised for SpectralEmbedding {
    type Fitted = FittedEmbedding;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedEmbedding>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let w = rbf_affinity(x, self.gamma.max(0.0));
        let mut deg = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += w[(i, j)];
            }
            deg[i] = s;
            if s <= ctx.policy.near_zero_variance {
                ctx.push(
                    Issue::builder(IssueCode::EmbeddingUnstable)
                        .message(format!("spectral degree[{i}]≈0"))
                        .build(),
                );
            }
        }
        let lap = Mat::<f64>::from_fn(n, n, |i, j| {
            let di = deg[i].max(1e-15).sqrt();
            let dj = deg[j].max(1e-15).sqrt();
            if i == j {
                1.0
            } else {
                -w[(i, j)] / (di * dj)
            }
        });
        let Some((vals, vecs)) = symmetric_eigen(&mut ctx.report, &lap, &ctx.policy) else {
            return ctx.finish(FittedEmbedding {
                embedding: Matrix::zeros(n, self.n_components.max(1)),
                stress: f64::NAN,
            });
        };
        let mut pairs: Vec<(f64, usize)> = vals
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.n_components.max(1);
        let mut out = Matrix::zeros(n, k);
        // Skip the smallest (constant) eigenvector when present.
        let start = if pairs.len() > 1 { 1 } else { 0 };
        for (c, &(v, idx)) in pairs.iter().skip(start).take(k).enumerate() {
            let _ = v;
            for i in 0..n.min(vecs.nrows()) {
                out.set(i, c, vecs[(i, idx)]);
            }
        }
        ctx.finish(FittedEmbedding {
            embedding: out,
            stress: f64::NAN,
        })
    }
}

/// Locally linear embedding.
#[derive(Clone, Debug)]
pub struct LocallyLinearEmbedding {
    /// Neighbors per point.
    pub n_neighbors: usize,
    /// Embedding dimension.
    pub n_components: usize,
}

impl Default for LocallyLinearEmbedding {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            n_components: 2,
        }
    }
}

impl LocallyLinearEmbedding {
    /// LLE with `k` neighbors onto `d` coordinates.
    pub fn new(n_neighbors: usize, n_components: usize) -> Self {
        Self {
            n_neighbors,
            n_components,
        }
    }
}

impl FitUnsupervised for LocallyLinearEmbedding {
    type Fitted = FittedEmbedding;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedEmbedding>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let p = x.ncols();
        let g = knn_graph(x, self.n_neighbors);
        let mut w = vec![vec![0.0; n]; n];
        for i in 0..n {
            let nbrs = &g[i];
            let k = nbrs.len();
            if k == 0 {
                continue;
            }
            let z = Matrix::from_fn(k, p, |a, c| x.get(nbrs[a], c) - x.get(i, c));
            let ones = Vector::filled(k, 1.0);
            // Local Gram is k×k (neighbors), not p×p.
            let mut gram = faer::Mat::<f64>::zeros(k, k);
            for a in 0..k {
                for b in 0..=a {
                    let mut s = 0.0;
                    for c in 0..p {
                        s += z.get(a, c) * z.get(b, c);
                    }
                    gram[(a, b)] = s;
                    gram[(b, a)] = s;
                }
                gram[(a, a)] += 1e-3;
            }
            let wt = match gram.llt(Side::Lower) {
                Ok(chol) => {
                    let rhs = ones.to_matrix();
                    let sol = chol.solve(rhs.inner());
                    Vector::from_iter((0..k).map(|i| sol[(i, 0)]))
                }
                Err(_) => Vector::filled(k, 1.0 / k as f64),
            };
            let s: f64 = wt.as_slice().iter().sum();
            let scale = if s.abs() > 0.0 {
                1.0 / s
            } else {
                1.0 / k as f64
            };
            for (a, &j) in nbrs.iter().enumerate() {
                w[i][j] = wt[a] * scale;
            }
        }
        // M = (I−W)ᵀ(I−W)
        let m = Mat::<f64>::from_fn(n, n, |a, b| {
            let mut s = 0.0;
            for i in 0..n {
                let ia = if a == i { 1.0 } else { 0.0 } - w[i][a];
                let ib = if b == i { 1.0 } else { 0.0 } - w[i][b];
                s += ia * ib;
            }
            s
        });
        let Some((vals, vecs)) = symmetric_eigen(&mut ctx.report, &m, &ctx.policy) else {
            return ctx.finish(FittedEmbedding {
                embedding: Matrix::zeros(n, self.n_components.max(1)),
                stress: f64::NAN,
            });
        };
        let mut pairs: Vec<(f64, usize)> = vals
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.n_components.max(1);
        let mut out = Matrix::zeros(n, k);
        let start = if pairs.len() > 1 { 1 } else { 0 };
        for (c, &(_, idx)) in pairs.iter().skip(start).take(k).enumerate() {
            for i in 0..n.min(vecs.nrows()) {
                out.set(i, c, vecs[(i, idx)]);
            }
        }
        ctx.finish(FittedEmbedding {
            embedding: out,
            stress: f64::NAN,
        })
    }
}

fn local_tangent_scores(
    x: &Matrix,
    nbrs: &[usize],
    d: usize,
    policy: &signlred::Policy,
) -> Option<Matrix> {
    let k = nbrs.len();
    let p = x.ncols();
    if k == 0 || p == 0 {
        return None;
    }
    let mut mean = vec![0.0; p];
    for &i in nbrs {
        for c in 0..p {
            mean[c] += x.get(i, c);
        }
    }
    let kf = k as f64;
    for c in 0..p {
        mean[c] /= kf;
    }
    let z = Matrix::from_fn(k, p, |a, c| x.get(nbrs[a], c) - mean[c]);
    let mut scratch = Report::new("lle", "tangent");
    let svd = thin_svd(&mut scratch, &z, policy)?;
    let r = d.max(1).min(svd.v.ncols()).min(svd.singular_values.len());
    Some(Matrix::from_fn(k, r, |a, c| {
        let mut s = 0.0;
        for j in 0..p.min(svd.v.nrows()) {
            s += z.get(a, j) * svd.v[(j, c)];
        }
        s
    }))
}

fn orthonormalize_columns(a: &mut Matrix) {
    let (k, q) = a.shape();
    for j in 0..q {
        for i in 0..j {
            let mut dot = 0.0;
            for r in 0..k {
                dot += a.get(r, j) * a.get(r, i);
            }
            for r in 0..k {
                a.set(r, j, a.get(r, j) - dot * a.get(r, i));
            }
        }
        let mut nrm = 0.0_f64;
        for r in 0..k {
            nrm += a.get(r, j) * a.get(r, j);
        }
        nrm = nrm.sqrt();
        if nrm < 1e-12 {
            continue;
        }
        for r in 0..k {
            a.set(r, j, a.get(r, j) / nrm);
        }
    }
}

fn embed_smallest_eigs(
    ctx: &mut FitCtx,
    m: &Mat<f64>,
    n: usize,
    n_components: usize,
) -> Matrix {
    let k = n_components.max(1);
    let Some((vals, vecs)) = symmetric_eigen(&mut ctx.report, m, &ctx.policy) else {
        return Matrix::zeros(n, k);
    };
    let mut pairs: Vec<(f64, usize)> = vals
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Matrix::zeros(n, k);
    let start = if pairs.len() > 1 { 1 } else { 0 };
    for (c, &(_, idx)) in pairs.iter().skip(start).take(k).enumerate() {
        for i in 0..n.min(vecs.nrows()) {
            out.set(i, c, vecs[(i, idx)]);
        }
    }
    out
}

/// Hessian eigenmaps (Donoho–Grimes; sklearn `LocallyLinearEmbedding(method="hessian")`).
///
/// Neighbor / Hessian-monomial counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct HessianLle {
    /// Neighbors per point (expanded to the quadratic design width when needed).
    pub n_neighbors: usize,
    /// Embedding dimension.
    pub n_components: usize,
}

impl Default for HessianLle {
    fn default() -> Self {
        Self {
            n_neighbors: 6,
            n_components: 2,
        }
    }
}

impl HessianLle {
    /// Hessian LLE with `k` neighbors onto `d` coordinates.
    pub fn new(n_neighbors: usize, n_components: usize) -> Self {
        Self {
            n_neighbors,
            n_components,
        }
    }
}

impl FitUnsupervised for HessianLle {
    type Fitted = FittedEmbedding;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedEmbedding>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let d = self.n_components.max(1);
        let n_quad = d * (d + 1) / 2;
        let need = (1 + d + n_quad).max(self.n_neighbors.max(1));
        if need > self.n_neighbors {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!(
                        "HessianLle expanded n_neighbors {} → {need} so the local Hessian design fits",
                        self.n_neighbors
                    ))
                    .build(),
            );
        }
        let g = knn_graph(x, need);
        let mut m = Mat::<f64>::zeros(n, n);
        let mut used = 0u64;
        for i in 0..n {
            let nbrs = &g[i];
            let k = nbrs.len();
            if k < 1 + d {
                continue;
            }
            let Some(u) = local_tangent_scores(x, nbrs, d, &ctx.policy) else {
                continue;
            };
            let dd = u.ncols();
            let nq = dd * (dd + 1) / 2;
            let n_lin = 1 + dd;
            let mut yi = Matrix::from_fn(k, n_lin + nq, |a, c| {
                if c == 0 {
                    1.0
                } else if c <= dd {
                    u.get(a, c - 1)
                } else {
                    let q = c - n_lin;
                    let mut acc = 0usize;
                    for p0 in 0..dd {
                        for p1 in p0..dd {
                            if acc == q {
                                return u.get(a, p0) * u.get(a, p1);
                            }
                            acc += 1;
                        }
                    }
                    0.0
                }
            });
            orthonormalize_columns(&mut yi);
            for a in 0..k {
                for b in 0..k {
                    let mut s = 0.0;
                    for q in 0..nq {
                        s += yi.get(a, n_lin + q) * yi.get(b, n_lin + q);
                    }
                    m[(nbrs[a], nbrs[b])] += s;
                }
            }
            used += 1;
        }
        if used == 0 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("HessianLle built no local Hessian operators")
                    .meaninglessness(signlred::Meaninglessness::vacuous(
                        "Hessian eigenmaps",
                        "every neighborhood failed the local SVD",
                        "use more neighbors or a less degenerate sample",
                    ))
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("HessianLle is a Gram–Schmidt Hessian sketch, not sklearn's exact eigenmaps")
                .compromise(NumericalCompromise::new(
                    "Donoho–Grimes Hessian eigenmaps",
                    "local PCA plus orthonormalized quadratic monomials",
                    "the published null-space / SVD Hessian estimator is omitted",
                    "read the map as a Hessian-alignment sketch",
                ))
                .build(),
        );
        let out = embed_smallest_eigs(&mut ctx, &m, n, d);
        ctx.finish(FittedEmbedding {
            embedding: out,
            stress: f64::NAN,
        })
    }
}

/// Modified locally linear embedding (Zhang–Wang; sklearn `method="modified"`).
///
/// Uses the local Gram null space rather than \(G^{-1}\mathbf{1}\). Neighbor
/// count is not identification `p`.
#[derive(Clone, Debug)]
pub struct ModifiedLle {
    /// Neighbors per point.
    pub n_neighbors: usize,
    /// Embedding dimension.
    pub n_components: usize,
}

impl Default for ModifiedLle {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            n_components: 2,
        }
    }
}

impl ModifiedLle {
    /// Modified LLE with `k` neighbors onto `d` coordinates.
    pub fn new(n_neighbors: usize, n_components: usize) -> Self {
        Self {
            n_neighbors,
            n_components,
        }
    }
}

impl FitUnsupervised for ModifiedLle {
    type Fitted = FittedEmbedding;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedEmbedding>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let p = x.ncols();
        let d = self.n_components.max(1);
        let g = knn_graph(x, self.n_neighbors);
        let mut w = vec![vec![0.0; n]; n];
        for i in 0..n {
            let nbrs = &g[i];
            let k = nbrs.len();
            if k == 0 {
                continue;
            }
            let z = Matrix::from_fn(k, p, |a, c| x.get(nbrs[a], c) - x.get(i, c));
            let mut gram = Mat::<f64>::zeros(k, k);
            for a in 0..k {
                for b in 0..=a {
                    let mut s = 0.0;
                    for c in 0..p {
                        s += z.get(a, c) * z.get(b, c);
                    }
                    gram[(a, b)] = s;
                    gram[(b, a)] = s;
                }
                gram[(a, a)] += 1e-8;
            }
            let mut scratch = Report::new("mlle", "gram");
            let wt = match symmetric_eigen(&mut scratch, &gram, &ctx.policy) {
                Some((vals, vecs)) => {
                    let mut pairs: Vec<(f64, usize)> = vals
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(i, v)| (v, i))
                        .collect();
                    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    let sdim = k.saturating_sub(d).max(1).min(pairs.len());
                    let mut raw = vec![0.0; k];
                    for &(_, idx) in pairs.iter().take(sdim) {
                        let mut ones = 0.0;
                        for a in 0..k.min(vecs.nrows()) {
                            ones += vecs[(a, idx)];
                        }
                        for a in 0..k.min(vecs.nrows()) {
                            raw[a] += vecs[(a, idx)] * ones;
                        }
                    }
                    Vector::from_iter(raw)
                }
                None => Vector::filled(k, 1.0 / k as f64),
            };
            let s: f64 = wt.as_slice().iter().sum();
            let scale = if s.abs() > 1e-15 {
                1.0 / s
            } else {
                1.0 / k as f64
            };
            for (a, &j) in nbrs.iter().enumerate() {
                w[i][j] = wt[a] * scale;
            }
        }
        let m = Mat::<f64>::from_fn(n, n, |a, b| {
            let mut s = 0.0;
            for i in 0..n {
                let ia = if a == i { 1.0 } else { 0.0 } - w[i][a];
                let ib = if b == i { 1.0 } else { 0.0 } - w[i][b];
                s += ia * ib;
            }
            s
        });
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("ModifiedLle projects ones onto the local Gram null space")
                .compromise(NumericalCompromise::new(
                    "Zhang–Wang modified LLE",
                    "null-space projection of the constant vector",
                    "Householder regularization of multiple weight vectors is omitted",
                    "read the map as a null-space LLE sketch",
                ))
                .build(),
        );
        let out = embed_smallest_eigs(&mut ctx, &m, n, d);
        ctx.finish(FittedEmbedding {
            embedding: out,
            stress: f64::NAN,
        })
    }
}

/// Local tangent space alignment (Zhang–Zha; sklearn `LocallyLinearEmbedding(method="ltsa")`).
///
/// Neighbor count is not identification `p`.
#[derive(Clone, Debug)]
pub struct Ltsa {
    /// Neighbors per point.
    pub n_neighbors: usize,
    /// Embedding dimension.
    pub n_components: usize,
}

impl Default for Ltsa {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            n_components: 2,
        }
    }
}

impl Ltsa {
    /// LTSA with `k` neighbors onto `d` coordinates.
    pub fn new(n_neighbors: usize, n_components: usize) -> Self {
        Self {
            n_neighbors,
            n_components,
        }
    }
}

impl FitUnsupervised for Ltsa {
    type Fitted = FittedEmbedding;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedEmbedding>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let d = self.n_components.max(1);
        let g = knn_graph(x, self.n_neighbors);
        let mut b = Mat::<f64>::zeros(n, n);
        let mut used = 0u64;
        for i in 0..n {
            let nbrs = &g[i];
            let k = nbrs.len();
            if k < 2 {
                continue;
            }
            let Some(u) = local_tangent_scores(x, nbrs, d, &ctx.policy) else {
                continue;
            };
            let dd = u.ncols();
            let kf = (k as f64).sqrt();
            let mut gi = Matrix::from_fn(k, 1 + dd, |a, c| {
                if c == 0 {
                    1.0 / kf
                } else {
                    u.get(a, c - 1)
                }
            });
            orthonormalize_columns(&mut gi);
            let q = gi.ncols();
            for a in 0..k {
                for c in 0..k {
                    let mut proj = 0.0;
                    for j in 0..q {
                        proj += gi.get(a, j) * gi.get(c, j);
                    }
                    let w = if a == c { 1.0 } else { 0.0 } - proj;
                    b[(nbrs[a], nbrs[c])] += w;
                }
            }
            used += 1;
        }
        if used == 0 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("Ltsa aligned no local tangent spaces")
                    .meaninglessness(signlred::Meaninglessness::vacuous(
                        "LTSA embedding",
                        "every neighborhood failed the local SVD",
                        "use more neighbors or a less degenerate sample",
                    ))
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Ltsa is local-PCA alignment, not the published Zhang–Zha solver")
                .compromise(NumericalCompromise::new(
                    "Zhang–Zha LTSA",
                    "I−GGᵀ assembly of local tangent frames",
                    "the published orthogonal Procrustes alignment is omitted",
                    "read the map as a tangent-alignment sketch",
                ))
                .build(),
        );
        let out = embed_smallest_eigs(&mut ctx, &b, n, d);
        ctx.finish(FittedEmbedding {
            embedding: out,
            stress: f64::NAN,
        })
    }
}

/// Exact t-SNE for small `n` (student-t affinities, early exaggeration).
#[derive(Clone, Debug)]
pub struct TSNE {
    /// Embedding dimension (2 or 3).
    pub n_components: usize,
    /// Perplexity target for the input Gaussians.
    pub perplexity: f64,
    /// Gradient steps.
    pub max_iter: usize,
    /// Early-exaggeration factor on `P` for the first `exaggeration_iter` steps.
    pub exaggeration: f64,
    /// Steps that use exaggeration.
    pub exaggeration_iter: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// PRNG seed for the initial embedding.
    pub seed: u64,
}

impl Default for TSNE {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 5.0,
            max_iter: 250,
            exaggeration: 4.0,
            exaggeration_iter: 50,
            learning_rate: 50.0,
            seed: 0,
        }
    }
}

impl TSNE {
    /// Small-n exact t-SNE.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }
}

fn binary_search_sigma(d2: &[f64], perplexity: f64) -> f64 {
    let target = perplexity.max(1.0).ln();
    let mut lo = 1e-8_f64;
    let mut hi = 1e8_f64;
    let mut sigma = 1.0;
    for _ in 0..40 {
        sigma = (lo * hi).sqrt();
        let mut sum = 0.0;
        let mut h = 0.0;
        let two_s2 = 2.0 * sigma * sigma;
        for &d in d2 {
            let p = (-d / two_s2).exp();
            sum += p;
        }
        if sum <= 0.0 {
            lo = sigma;
            continue;
        }
        for &d in d2 {
            let p = (-d / two_s2).exp() / sum;
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        if h > target {
            hi = sigma;
        } else {
            lo = sigma;
        }
    }
    sigma
}

impl FitUnsupervised for TSNE {
    type Fitted = FittedEmbedding;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedEmbedding>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let k = self.n_components.max(1).min(3);
        if n > 80 {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!(
                        "exact t-SNE is O(n²); n={n} is large for this path"
                    ))
                    .build(),
            );
        }
        if n < 3 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("t-SNE needs at least 3 points")
                    .build(),
            );
            return ctx.finish(FittedEmbedding {
                embedding: Matrix::zeros(n, k),
                stress: f64::NAN,
            });
        }
        let mut d2 = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                d2[i * n + j] = if i == j { 0.0 } else { sq_dist(x, i, j) };
            }
        }
        let mut pcond = vec![0.0; n * n];
        for i in 0..n {
            let row: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| d2[i * n + j]).collect();
            let sigma = binary_search_sigma(&row, self.perplexity);
            let two_s2 = 2.0 * sigma * sigma;
            let mut z = 0.0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let v = (-d2[i * n + j] / two_s2).exp();
                pcond[i * n + j] = v;
                z += v;
            }
            if z > 0.0 {
                for j in 0..n {
                    pcond[i * n + j] /= z;
                }
            }
        }
        let mut p = vec![0.0; n * n];
        let nf = (2 * n).max(1) as f64;
        for i in 0..n {
            for j in 0..n {
                p[i * n + j] = (pcond[i * n + j] + pcond[j * n + i]) / nf;
            }
        }
        let mut rng = crate::rng::Rng::new(self.seed);
        let mut y = Matrix::from_fn(n, k, |_, _| 0.01 * rng.standard_normal());
        let mut vel = Matrix::zeros(n, k);
        for it in 0..self.max_iter.max(1) {
            let ex = if it < self.exaggeration_iter {
                self.exaggeration.max(1.0)
            } else {
                1.0
            };
            let mut q = vec![0.0; n * n];
            let mut zq = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let mut d = 0.0;
                    for c in 0..k {
                        let t = y.get(i, c) - y.get(j, c);
                        d += t * t;
                    }
                    let v = 1.0 / (1.0 + d);
                    q[i * n + j] = v;
                    zq += v;
                }
            }
            if zq <= 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::LossIsNan)
                        .message("t-SNE Z(q)=0")
                        .build(),
                );
                break;
            }
            for v in q.iter_mut() {
                *v /= zq;
            }
            let mut grad = Matrix::zeros(n, k);
            let mut kl = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let pij = (ex * p[i * n + j]).max(1e-12);
                    let qij = q[i * n + j].max(1e-12);
                    kl += pij * (pij / qij).ln();
                    let mut d2y = 0.0;
                    for c in 0..k {
                        let t = y.get(i, c) - y.get(j, c);
                        d2y += t * t;
                    }
                    let coef = 4.0 * (pij - qij) / (1.0 + d2y);
                    for c in 0..k {
                        grad.set(i, c, grad.get(i, c) + coef * (y.get(i, c) - y.get(j, c)));
                    }
                }
            }
            if !kl.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::LossIsNan)
                        .message("t-SNE KL is NaN")
                        .build(),
                );
                break;
            }
            let gnorm = {
                let mut s = 0.0;
                for i in 0..n {
                    for c in 0..k {
                        s += grad.get(i, c) * grad.get(i, c);
                    }
                }
                s.sqrt()
            };
            if gnorm > 1e6 {
                ctx.push(
                    Issue::builder(IssueCode::GradientExploded)
                        .message(format!("t-SNE ‖g‖={gnorm:.3e}"))
                        .build(),
                );
                break;
            }
            let mom = if it < 20 { 0.5 } else { 0.8 };
            for i in 0..n {
                for c in 0..k {
                    let v = mom * vel.get(i, c) - self.learning_rate * grad.get(i, c);
                    vel.set(i, c, v);
                    y.set(i, c, y.get(i, c) + v);
                }
            }
            ctx.session.step(it as u64, kl, Some(gnorm));
        }
        let dist = euclid_matrix(x);
        let stress = kruskal_stress(&dist, &y);
        warn_stress(&mut ctx, stress);
        ctx.finish(FittedEmbedding {
            embedding: y,
            stress,
        })
    }
}

impl Transform for FittedEmbedding {
    fn transform(&self, _x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        ctx.push(
            Issue::builder(IssueCode::StaleState)
                .message(
                    "manifold embeddings do not out-of-sample transform; returning the fitted map",
                )
                .build(),
        );
        ctx.finish(self.embedding.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    fn triangle() -> Matrix {
        Matrix::from_fn(3, 2, |i, j| match (i, j) {
            (0, 0) => 0.0,
            (0, 1) => 0.0,
            (1, 0) => 1.0,
            (1, 1) => 0.0,
            (2, 0) => 0.5,
            (2, 1) => 0.866,
            _ => 0.0,
        })
    }

    #[test]
    fn mds_recovers_triangle_distances() {
        let x = triangle();
        let q = MDS::new(2)
            .fit_unsupervised(&x, &Session::new("man", "mds"))
            .unwrap();
        assert_eq!(q.value.embedding.shape(), (3, 2));
        assert!(q.value.stress.is_finite());
        assert!(q.value.stress < 0.05, "stress={}", q.value.stress);
    }

    #[test]
    fn isomap_and_spectral_shapes() {
        let x = Matrix::from_fn(8, 2, |i, j| if j == 0 { i as f64 } else { (i % 3) as f64 });
        let iso = Isomap::new(3, 2)
            .fit_unsupervised(&x, &Session::new("man", "iso"))
            .unwrap();
        assert_eq!(iso.value.embedding.shape(), (8, 2));
        let sp = SpectralEmbedding::new(2)
            .fit_unsupervised(&x, &Session::new("man", "sp"))
            .unwrap();
        assert_eq!(sp.value.embedding.shape(), (8, 2));
    }

    #[test]
    fn lle_and_tsne_run() {
        let x = Matrix::from_fn(10, 2, |i, j| {
            (i as f64) * 0.3 + (j as f64) * 0.7 + 0.05 * (i * j) as f64
        });
        let lle = LocallyLinearEmbedding::new(3, 2)
            .fit_unsupervised(&x, &Session::new("man", "lle"))
            .unwrap();
        assert_eq!(lle.value.embedding.shape(), (10, 2));
        let ts = TSNE {
            max_iter: 40,
            exaggeration_iter: 10,
            perplexity: 3.0,
            ..TSNE::new(2)
        }
        .fit_unsupervised(&x, &Session::new("man", "tsne"))
        .unwrap();
        assert_eq!(ts.value.embedding.shape(), (10, 2));
        assert!(ts
            .value
            .embedding
            .to_row_major()
            .iter()
            .all(|v| v.is_finite()));
        let hl = HessianLle::new(5, 2)
            .fit_unsupervised(&x, &Session::new("man", "hlle"))
            .unwrap();
        assert_eq!(hl.value.embedding.shape(), (10, 2));
        assert!(hl
            .value
            .embedding
            .to_row_major()
            .iter()
            .all(|v| v.is_finite()));
        let ml = ModifiedLle::new(3, 2)
            .fit_unsupervised(&x, &Session::new("man", "mlle"))
            .unwrap();
        assert_eq!(ml.value.embedding.shape(), (10, 2));
        let lt = Ltsa::new(3, 2)
            .fit_unsupervised(&x, &Session::new("man", "ltsa"))
            .unwrap();
        assert_eq!(lt.value.embedding.shape(), (10, 2));
    }
}
