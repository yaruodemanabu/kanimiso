//! Matrix decompositions and latent-factor models.
//!
//! PCA / truncated SVD go through [`thin_svd`] of (centered) `X`. Incremental
//! PCA maintains a running mean and a sequential Karhunen–Loève basis and
//! **must** emit [`IncrementalExplain`] on every `partial_fit`. NMF, FastICA,
//! factor analysis, CCA, sparse PCA, and dictionary learning are pure-Rust
//! iterative methods that still record rank, NaN, and meaningless-fit issues.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{chol_solve, ridge_solve, symmetric_eigen, thin_svd};
use crate::rng::Rng;
use crate::traits::{Fit, FitUnsupervised, PartialFit, Transform};
use crate::validate::{inspect_identification, inspect_xy};
use faer::Mat;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{
    IncrementalQuality, Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result,
};

fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let prod = a.inner() * b.inner();
    Matrix::from_fn(prod.nrows(), prod.ncols(), |i, j| prod[(i, j)])
}

fn matmul_nt(a: &Matrix, b: &Matrix) -> Matrix {
    // a * bᵀ
    let prod = a.inner() * b.inner().transpose();
    Matrix::from_fn(prod.nrows(), prod.ncols(), |i, j| prod[(i, j)])
}

fn matmul_tn(a: &Matrix, b: &Matrix) -> Matrix {
    // aᵀ * b
    let prod = a.inner().transpose() * b.inner();
    Matrix::from_fn(prod.nrows(), prod.ncols(), |i, j| prod[(i, j)])
}

fn push_nonfinite_mat(ctx: &mut FitCtx, m: &Matrix, what: &str) {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            if !m.get(i, j).is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message(format!("{what} contains NaN/Inf"))
                        .build(),
                );
                return;
            }
        }
    }
}

fn push_nonfinite_vec(ctx: &mut FitCtx, v: &Vector, what: &str) {
    if v.as_slice().iter().any(|z| !z.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteOutput)
                .message(format!("{what} contains NaN/Inf"))
                .build(),
        );
    }
}

fn components_exceed_rank(requested: usize, rank: usize) -> Issue {
    Issue::builder(IssueCode::ComponentsExceedRank)
        .message(format!(
            "requested {requested} components but numerical rank is {rank}"
        ))
        .metric("n_components", requested as f64)
        .metric("rank", rank as f64)
        .compromise(NumericalCompromise::new(
            format!("{requested} latent directions"),
            format!("{rank} directions retained"),
            "the design does not support more components at working precision",
            "later components are not estimated and must not be interpreted",
        ))
        .build()
}

fn dummy_explain(update: u64, batch: usize, n_seen: u64) -> IncrementalExplain {
    IncrementalExplain::from_quality(
        IncrementalQuality::new(update, batch, n_seen),
        "nothing",
        "the update was rejected",
        "invalid",
        "invalid",
    )
}

fn copy_svd_components(v: &Mat<f64>, k: usize) -> Matrix {
    // V is p × r; sklearn-style components are k × p (rows are principal axes).
    let p = v.nrows();
    let r = v.ncols();
    let kk = k.min(r);
    Matrix::from_fn(kk, p, |c, j| {
        if j < p && c < r {
            v[(j, c)]
        } else {
            0.0
        }
    })
}

fn explained_from_singular(s: &[f64], n: usize, k: usize) -> (Vector, Vector) {
    let df = (n.saturating_sub(1)).max(1) as f64;
    let ev = Vector::from_iter(s.iter().take(k).map(|si| (si * si) / df));
    let tot: f64 = s.iter().map(|si| (si * si) / df).sum();
    let ratio = if tot > 0.0 {
        Vector::from_iter((0..ev.len()).map(|i| ev[i] / tot))
    } else {
        Vector::zeros(ev.len())
    };
    (ev, ratio)
}

fn project_centered(xc: &Matrix, components: &Matrix) -> Matrix {
    // scores = Xc * componentsᵀ   (n × k)
    matmul_nt(xc, components)
}

/// Principal component analysis via a thin SVD of column-centered `X`.
#[derive(Clone, Debug)]
pub struct Pca {
    /// Number of components requested.
    pub n_components: usize,
}

impl Default for Pca {
    fn default() -> Self {
        Self { n_components: 2 }
    }
}

impl Pca {
    /// Keep `n_components` axes.
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedPca>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted PCA.
#[derive(Clone, Debug)]
pub struct FittedPca {
    /// Principal axes (`k` × `p`).
    pub components: Matrix,
    /// Column means of the training design.
    pub mean: Vector,
    /// Singular values of the centered matrix (length `k`).
    pub singular_values: Vector,
    /// Eigenvalues of the sample covariance (`σ² / (n−1)`).
    pub explained_variance: Vector,
    /// Fractions of total variance, summing to ≤ 1.
    pub explained_variance_ratio: Vector,
    /// Numerical rank of the centered design.
    pub rank: usize,
}

impl FittedPca {
    fn transform_mat(&self, x: &Matrix) -> Matrix {
        let (n, p) = x.shape();
        let pc = self.mean.len().min(p);
        let xc = Matrix::from_fn(n, pc, |i, j| x.get(i, j) - self.mean[j]);
        project_centered(&xc, &self.components)
    }
}

impl Transform for FittedPca {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.ncols() != self.components.ncols() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!(
                        "PCA transform X is n×{} but components are k×{}",
                        x.ncols(),
                        self.components.ncols()
                    ))
                    .build(),
            );
        }
        ctx.finish(self.transform_mat(x))
    }
}

impl FitUnsupervised for Pca {
    type Fitted = FittedPca;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedPca>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.n_components.max(1), &ctx.policy);
        let (n, p) = x.shape();
        if n == 0 || p == 0 {
            return ctx.finish(FittedPca {
                components: Matrix::zeros(self.n_components, p),
                mean: Vector::zeros(p),
                singular_values: Vector::zeros(0),
                explained_variance: Vector::zeros(0),
                explained_variance_ratio: Vector::zeros(0),
                rank: 0,
            });
        }
        let (xc, mean) = x.centered();
        let Some(svd) = thin_svd(&mut ctx.report, &xc, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::SvdDidNotConverge)
                    .message("PCA thin SVD failed")
                    .build(),
            );
            return ctx.finish(FittedPca {
                components: Matrix::zeros(self.n_components.min(p), p),
                mean,
                singular_values: Vector::zeros(0),
                explained_variance: Vector::zeros(0),
                explained_variance_ratio: Vector::zeros(0),
                rank: 0,
            });
        };
        let rank = svd.rank(ctx.policy.rank_tol_relative);
        if rank == 0 {
            ctx.push(
                Issue::builder(IssueCode::RankZero)
                    .message("centered X is the zero operator")
                    .meaninglessness(Meaninglessness::vacuous(
                        "principal components",
                        "a zero matrix has no direction of variation",
                        "do not interpret loadings",
                    ))
                    .build(),
            );
        }
        let kappa = svd.condition_number();
        if !kappa.is_finite() || kappa > ctx.policy.condition_number_warn {
            ctx.push(
                Issue::builder(IssueCode::IllConditioned)
                    .message(format!("PCA singular-value condition κ={kappa:.4e}"))
                    .metric("condition_number", kappa)
                    .build(),
            );
        }
        let k_req = self.n_components.max(1);
        let k = k_req.min(rank.max(1)).min(svd.singular_values.len()).min(p);
        if k_req > rank {
            ctx.push(components_exceed_rank(k_req, rank));
            ctx.push(
                Issue::builder(IssueCode::TruncatedSvdUsed)
                    .message(format!("PCA truncated to numerical rank {rank}"))
                    .compromise(NumericalCompromise::new(
                        format!("{k_req} principal components"),
                        format!("{k} components from a rank-{rank} SVD"),
                        "extra components lie in the numerical null space",
                        "explained-variance ratios of dropped components are identically zero",
                    ))
                    .build(),
            );
        }
        let components = copy_svd_components(&svd.v, k);
        let singular_values = Vector::from_iter(svd.singular_values.iter().take(k).copied());
        let (explained_variance, explained_variance_ratio) =
            explained_from_singular(&svd.singular_values, n, k);
        push_nonfinite_mat(&mut ctx, &components, "PCA components");
        ctx.finish(FittedPca {
            components,
            mean,
            singular_values,
            explained_variance,
            explained_variance_ratio,
            rank,
        })
    }
}

/// Incremental PCA: running mean + sequential Karhunen–Loève (SKL) update.
///
/// Each `partial_fit` concatenates the previous scaled basis with the newly
/// centered batch (and a mean-correction row) and re-factorizes, then emits
/// [`IncrementalExplain`].
#[derive(Clone, Debug)]
pub struct IncrementalPca {
    /// Number of components to retain.
    pub n_components: usize,
    n_seen: u64,
    updates: u64,
    mean: Option<Vector>,
    /// Current axes (`k` × `p`).
    components: Option<Matrix>,
    singular_values: Option<Vector>,
    initialized: bool,
}

impl Default for IncrementalPca {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_seen: 0,
            updates: 0,
            mean: None,
            components: None,
            singular_values: None,
            initialized: false,
        }
    }
}

impl IncrementalPca {
    /// Keep `n_components` online axes.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Current mean, if any batch has been seen.
    pub fn mean(&self) -> Option<&Vector> {
        self.mean.as_ref()
    }

    /// Current components (`k` × `p`).
    pub fn components(&self) -> Option<&Matrix> {
        self.components.as_ref()
    }

    /// Fit alias (one SKL step on the whole matrix).
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedPca>> {
        self.fit_unsupervised(x, session)
    }

    fn to_fitted(&self, p: usize) -> FittedPca {
        let k = self.n_components;
        let components = self
            .components
            .clone()
            .unwrap_or_else(|| Matrix::zeros(k, p));
        let mean = self.mean.clone().unwrap_or_else(|| Vector::zeros(p));
        let singular_values = self
            .singular_values
            .clone()
            .unwrap_or_else(|| Vector::zeros(0));
        let n = self.n_seen as usize;
        let s: Vec<f64> = singular_values.as_slice().to_vec();
        let (explained_variance, explained_variance_ratio) = explained_from_singular(&s, n.max(2), s.len());
        FittedPca {
            components,
            mean,
            singular_values,
            explained_variance,
            explained_variance_ratio,
            rank: s.iter().filter(|v| **v > 1e-12).count(),
        }
    }
}

impl FitUnsupervised for IncrementalPca {
    type Fitted = FittedPca;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedPca>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.n_components.max(1), &ctx.policy);
        if x.nrows() == 0 || x.ncols() == 0 {
            return ctx.finish(self.to_fitted(x.ncols()));
        }
        let _ = self.partial_fit(x, None, &session.child("ipca_init"));
        ctx.finish(self.to_fitted(x.ncols()))
    }
}

impl Transform for IncrementalPca {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let fitted = self.to_fitted(x.ncols());
        fitted.transform(x, session)
    }
}

impl PartialFit for IncrementalPca {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.n_components.max(1), &ctx.policy);
        let (m, p) = x.shape();
        if m == 0 || p == 0 {
            return ctx.finish(dummy_explain(self.updates, 0, self.n_seen));
        }
        if self.initialized {
            if self.mean.as_ref().map(|mu| mu.len()) != Some(p) {
                ctx.push(
                    Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                        .message("IncrementalPca feature dimension changed")
                        .build(),
                );
                return ctx.finish(dummy_explain(self.updates, m, self.n_seen));
            }
        }
        let batch_mean = {
            let mut mu = Vector::zeros(p);
            for j in 0..p {
                let mut s = 0.0;
                for i in 0..m {
                    s += x.get(i, j);
                }
                mu[j] = s / m as f64;
            }
            mu
        };
        let n_old = self.n_seen as f64;
        let n_new = n_old + m as f64;
        let old_mean = self.mean.clone().unwrap_or_else(|| Vector::zeros(p));
        let mut new_mean = Vector::zeros(p);
        for j in 0..p {
            new_mean[j] = if n_new > 0.0 {
                (n_old * old_mean[j] + (m as f64) * batch_mean[j]) / n_new
            } else {
                batch_mean[j]
            };
        }
        // Sequential Karhunen–Loève: stack [s * V; Xc; √(n_old*m/n_new) Δμ].
        let k = self.n_components.max(1).min(p);
        let mut rows: Vec<Vector> = Vec::new();
        if let (Some(comp), Some(svals)) = (&self.components, &self.singular_values) {
            for c in 0..comp.nrows() {
                let scale = if c < svals.len() { svals[c] } else { 0.0 };
                rows.push(Vector::from_iter((0..p).map(|j| scale * comp.get(c, j))));
            }
        }
        for i in 0..m {
            rows.push(Vector::from_iter((0..p).map(|j| x.get(i, j) - new_mean[j])));
        }
        if n_old > 0.0 && m > 0 {
            let corr = ((n_old * m as f64) / n_new).sqrt();
            rows.push(Vector::from_iter(
                (0..p).map(|j| corr * (old_mean[j] - batch_mean[j])),
            ));
        }
        let stacked = Matrix::from_fn(rows.len(), p, |i, j| rows[i][j]);
        let before_s = self
            .singular_values
            .clone()
            .unwrap_or_else(|| Vector::zeros(0));
        let Some(svd) = thin_svd(&mut ctx.report, &stacked, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::SvdDidNotConverge)
                    .message("incremental PCA SKL SVD failed")
                    .build(),
            );
            return ctx.finish(dummy_explain(self.updates, m, self.n_seen));
        };
        let rank = svd.rank(ctx.policy.rank_tol_relative);
        if k > rank {
            ctx.push(components_exceed_rank(k, rank));
        }
        let kk = k.min(svd.singular_values.len()).min(rank.max(1));
        let new_comp = copy_svd_components(&svd.v, kk);
        let new_s = Vector::from_iter(svd.singular_values.iter().take(kk).copied());
        let mut delta = 0.0;
        let mut moved = Vec::new();
        if let Some(old) = &self.components {
            let r = old.nrows().min(new_comp.nrows());
            let c = old.ncols().min(new_comp.ncols());
            for a in 0..r {
                let mut d2 = 0.0;
                for j in 0..c {
                    // components are defined up to sign
                    let d1 = old.get(a, j) - new_comp.get(a, j);
                    let d2s = old.get(a, j) + new_comp.get(a, j);
                    d2 += d1.min(d2s.abs()).abs().powi(2);
                }
                let d = d2.sqrt();
                delta += d * d;
                moved.push((format!("component[{a}]"), d));
            }
        } else {
            delta = new_s.norm();
            for a in 0..new_comp.nrows() {
                let sv = if a < new_s.len() { new_s[a] } else { 0.0 };
                moved.push((format!("component[{a}]"), sv));
            }
        }
        // Vector has Index but get on Vector - we use [a]
        let _ = before_s;
        self.mean = Some(new_mean);
        self.components = Some(new_comp);
        self.singular_values = Some(new_s.clone());
        self.n_seen = n_new as u64;
        self.updates += 1;
        self.initialized = true;
        let mut q = IncrementalQuality::new(self.updates - 1, m, self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.sqrt());
        q.parameter_delta_max = moved.iter().map(|(_, d)| *d).fold(None, |a, b| Some(a.unwrap_or(0.0).max(b)));
        q.top_moved_parameters = moved.clone();
        q.information_gain = Some(new_s.as_slice().first().copied().unwrap_or(0.0));
        q.still_identified = rank > 0 && self.n_seen as usize > p;
        q.warmup = self.n_seen < (p as u64 + 2);
        q.explanation = format!(
            "SKL incremental PCA: n_seen={}, rank={}, σ={:?}, component moves {:?}",
            self.n_seen,
            rank,
            new_s.as_slice(),
            moved
        );
        if rank == 0 {
            ctx.push(
                Issue::builder(IssueCode::UpdateWithZeroInformation)
                    .incremental(q.clone())
                    .message("incremental PCA batch added no variance")
                    .build(),
            );
        }
        if q.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("IncrementalPca has seen fewer rows than features+2")
                    .build(),
            );
        }
        let expl = IncrementalExplain::from_quality(
            q,
            format!("principal axes moved {moved:?}; singular values {:?}", new_s.as_slice()),
            "sequential Karhunen–Loève: SVD of [σV; X_centered; mean-correction]",
            format!("n_old={n_old}"),
            format!("n_seen={}, rank={rank}, n_eff={}", self.n_seen, self.n_seen),
        )
        .contribute("n_eff", self.n_seen as f64);
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

/// Thin SVD of the (optionally uncentered) design, truncated to `n_components`.
#[derive(Clone, Debug)]
pub struct TruncatedSvd {
    /// Number of singular triples to keep.
    pub n_components: usize,
}

impl Default for TruncatedSvd {
    fn default() -> Self {
        Self { n_components: 2 }
    }
}

impl TruncatedSvd {
    /// Keep `n_components` triples.
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedTruncatedSvd>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted truncated SVD.
#[derive(Clone, Debug)]
pub struct FittedTruncatedSvd {
    /// Right singular vectors as rows (`k` × `p`).
    pub components: Matrix,
    /// Singular values.
    pub singular_values: Vector,
    /// Explained variance of the *uncentered* Gram, `σ² / n`.
    pub explained_variance: Vector,
    /// Numerical rank before truncation.
    pub rank: usize,
}

impl Transform for FittedTruncatedSvd {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(project_centered(x, &self.components))
    }
}

impl FitUnsupervised for TruncatedSvd {
    type Fitted = FittedTruncatedSvd;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedTruncatedSvd>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.n_components.max(1), &ctx.policy);
        let (n, p) = x.shape();
        if n == 0 || p == 0 {
            return ctx.finish(FittedTruncatedSvd {
                components: Matrix::zeros(self.n_components, p),
                singular_values: Vector::zeros(0),
                explained_variance: Vector::zeros(0),
                rank: 0,
            });
        }
        let Some(svd) = thin_svd(&mut ctx.report, x, &ctx.policy) else {
            return ctx.finish(FittedTruncatedSvd {
                components: Matrix::zeros(self.n_components.min(p), p),
                singular_values: Vector::zeros(0),
                explained_variance: Vector::zeros(0),
                rank: 0,
            });
        };
        let rank = svd.rank(ctx.policy.rank_tol_relative);
        let k_req = self.n_components.max(1);
        let k = k_req.min(rank.max(1)).min(svd.singular_values.len());
        if k_req > rank {
            ctx.push(components_exceed_rank(k_req, rank));
        }
        ctx.push(
            Issue::builder(IssueCode::TruncatedSvdUsed)
                .message(format!("keeping {k} of {} singular triples", svd.singular_values.len()))
                .compromise(NumericalCompromise::new(
                    "full SVD",
                    format!("truncated SVD at k={k}"),
                    "only the leading spectrum is requested",
                    "reconstruction lives in a k-dimensional subspace",
                ))
                .build(),
        );
        let components = copy_svd_components(&svd.v, k);
        let singular_values = Vector::from_iter(svd.singular_values.iter().take(k).copied());
        let nn = n.max(1) as f64;
        let explained_variance =
            Vector::from_iter(svd.singular_values.iter().take(k).map(|s| (s * s) / nn));
        ctx.finish(FittedTruncatedSvd {
            components,
            singular_values,
            explained_variance,
            rank,
        })
    }
}

/// Non-negative matrix factorization by multiplicative updates.
#[derive(Clone, Debug)]
pub struct Nmf {
    /// Number of latent components.
    pub n_components: usize,
    /// Multiplicative-update iterations.
    pub max_iter: usize,
    /// Seed for the nonnegative random start.
    pub seed: u64,
}

impl Default for Nmf {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 200,
            seed: 0,
        }
    }
}

impl Nmf {
    /// `k` nonnegative factors.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedNmf>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted NMF: `X ≈ W H` with `W` (`n` × `k`) and `H` (`k` × `p`) nonnegative.
#[derive(Clone, Debug)]
pub struct FittedNmf {
    /// Left nonnegative factor.
    pub w: Matrix,
    /// Right nonnegative factor.
    pub h: Matrix,
    /// Frobenius reconstruction error.
    pub reconstruction_err: f64,
}

impl Transform for FittedNmf {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        // Nonnegative least-squares codes via a few multiplicative steps on W.
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let k = self.h.nrows();
        let n = x.nrows();
        let mut w = Matrix::from_fn(n, k, |_, _| 1.0);
        let hht = matmul_nt(&self.h, &self.h);
        let xht = matmul_nt(x, &self.h);
        for _ in 0..20 {
            let denom = matmul(&w, &hht);
            w = Matrix::from_fn(n, k, |i, c| {
                let d = denom.get(i, c).max(1e-12);
                (w.get(i, c) * xht.get(i, c) / d).max(0.0)
            });
        }
        ctx.finish(w)
    }
}

impl FitUnsupervised for Nmf {
    type Fitted = FittedNmf;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedNmf>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.n_components.max(1), &ctx.policy);
        let (n, p) = x.shape();
        let k = self.n_components.max(1).min(n.max(1)).min(p.max(1));
        if n == 0 || p == 0 {
            return ctx.finish(FittedNmf {
                w: Matrix::zeros(n, k),
                h: Matrix::zeros(k, p),
                reconstruction_err: f64::NAN,
            });
        }
        let mut clipped = 0usize;
        let xp = Matrix::from_fn(n, p, |i, j| {
            let v = x.get(i, j);
            if v < 0.0 {
                clipped += 1;
                0.0
            } else {
                v
            }
        });
        if clipped > 0 {
            ctx.push(
                Issue::builder(IssueCode::InconsistentSystem)
                    .message(format!(
                        "NMF projected {clipped} negative entries to 0 (nonnegativity check)"
                    ))
                    .metric("n_negative", clipped as f64)
                    .compromise(NumericalCompromise::new(
                        "NMF on a nonnegative design",
                        "X := max(X, 0) then Lee–Seung multiplicative updates",
                        "the input contained negative entries",
                        "factors explain the projected matrix, not the original signed values",
                    ))
                    .build(),
            );
        }
        if k < self.n_components {
            ctx.push(components_exceed_rank(self.n_components, k));
        }
        let mut rng = Rng::new(self.seed | 1);
        let mut w = Matrix::from_fn(n, k, |_, _| rng.uniform().abs() + 1e-3);
        let mut h = Matrix::from_fn(k, p, |_, _| rng.uniform().abs() + 1e-3);
        for it in 0..self.max_iter.max(1) {
            // H ← H ⊙ (WᵀX) ⊘ (WᵀW H)
            let wtx = matmul_tn(&w, &xp);
            let wtw = matmul_tn(&w, &w);
            let wtw_h = matmul(&wtw, &h);
            h = Matrix::from_fn(k, p, |c, j| {
                (h.get(c, j) * wtx.get(c, j) / wtw_h.get(c, j).max(1e-12)).max(0.0)
            });
            // W ← W ⊙ (X Hᵀ) ⊘ (W H Hᵀ)
            let xht = matmul_nt(&xp, &h);
            let hht = matmul_nt(&h, &h);
            let w_hht = matmul(&w, &hht);
            w = Matrix::from_fn(n, k, |i, c| {
                (w.get(i, c) * xht.get(i, c) / w_hht.get(i, c).max(1e-12)).max(0.0)
            });
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("NMF multiplicative updates hit max_iter")
                        .build(),
                );
            }
        }
        let recon = matmul(&w, &h);
        let mut err = 0.0;
        for j in 0..p {
            for i in 0..n {
                let d = xp.get(i, j) - recon.get(i, j);
                err += d * d;
            }
        }
        err = err.sqrt();
        push_nonfinite_mat(&mut ctx, &w, "NMF W");
        push_nonfinite_mat(&mut ctx, &h, "NMF H");
        ctx.finish(FittedNmf {
            w,
            h,
            reconstruction_err: err,
        })
    }
}

/// FastICA: SVD whitening plus one-unit deflation with `g = tanh`.
#[derive(Clone, Debug)]
pub struct FastIca {
    /// Number of independent components.
    pub n_components: usize,
    /// Fixed-point iterations per unit.
    pub max_iter: usize,
    /// Seed for the initial w.
    pub seed: u64,
}

impl Default for FastIca {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 200,
            seed: 0,
        }
    }
}

impl FastIca {
    /// `k` independent components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedFastIca>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted FastICA.
#[derive(Clone, Debug)]
pub struct FittedFastIca {
    /// Unmixing rows (`k` × `p`) in the original (centered) space.
    pub components: Matrix,
    /// Column means.
    pub mean: Vector,
    /// Whitening matrix (`k` × `p`).
    pub whitening: Matrix,
}

impl Transform for FittedFastIca {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        let pc = self.mean.len().min(p);
        let xc = Matrix::from_fn(n, pc, |i, j| x.get(i, j) - self.mean[j]);
        ctx.finish(project_centered(&xc, &self.components))
    }
}

impl FitUnsupervised for FastIca {
    type Fitted = FittedFastIca;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedFastIca>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.n_components.max(1), &ctx.policy);
        let (n, p) = x.shape();
        if n == 0 || p == 0 {
            return ctx.finish(FittedFastIca {
                components: Matrix::zeros(self.n_components, p),
                mean: Vector::zeros(p),
                whitening: Matrix::zeros(self.n_components, p),
            });
        }
        let (xc, mean) = x.centered();
        let Some(svd) = thin_svd(&mut ctx.report, &xc, &ctx.policy) else {
            return ctx.finish(FittedFastIca {
                components: Matrix::zeros(self.n_components.min(p), p),
                mean,
                whitening: Matrix::zeros(self.n_components.min(p), p),
            });
        };
        let rank = svd.rank(ctx.policy.rank_tol_relative);
        let k_req = self.n_components.max(1);
        let k = k_req.min(rank.max(1)).min(p);
        if k_req > rank {
            ctx.push(components_exceed_rank(k_req, rank));
        }
        // Whitening: Xw = √(n−1) U_k   and  K = √(n−1) S^{-1} Vᵀ  so Xw = Xc Kᵀ.
        let scale = ((n.saturating_sub(1)).max(1) as f64).sqrt();
        let mut whitening = Matrix::zeros(k, p);
        for c in 0..k {
            let sig = svd.singular_values[c];
            if sig <= ctx.policy.rank_tol_relative * svd.singular_values.first().copied().unwrap_or(1.0) {
                ctx.push(
                    Issue::builder(IssueCode::RankDeficient)
                        .message(format!("FastICA whitening skipped near-zero σ[{c}]={sig:.3e}"))
                        .build(),
                );
                continue;
            }
            let coeff = scale / sig;
            for j in 0..p {
                if j < svd.v.nrows() && c < svd.v.ncols() {
                    whitening.set(c, j, coeff * svd.v[(j, c)]);
                }
            }
        }
        let xw = project_centered(&xc, &whitening); // n × k, already whitened
        let mut rng = Rng::new(self.seed | 1);
        let mut wmat = Matrix::zeros(k, k);
        for c in 0..k {
            let mut w = Vector::from_iter((0..k).map(|_| rng.standard_normal()));
            // Orthogonalize against previous units (deflation).
            for prev in 0..c {
                let mut dot = 0.0;
                for j in 0..k {
                    dot += w[j] * wmat.get(prev, j);
                }
                for j in 0..k {
                    w[j] -= dot * wmat.get(prev, j);
                }
            }
            let nrm = w.norm();
            if nrm > 0.0 {
                w = w.scale(1.0 / nrm);
            }
            let mut converged = false;
            for it in 0..self.max_iter.max(1) {
                // w⁺ = E[x g(wᵀx)] − E[g'(wᵀx)] w,   g = tanh, g' = 1−tanh²
                let mut wx = Vector::zeros(n);
                for i in 0..n {
                    let mut s = 0.0;
                    for j in 0..k {
                        s += xw.get(i, j) * w[j];
                    }
                    wx[i] = s;
                }
                let mut gw_mean = Vector::zeros(k);
                let mut gp = 0.0;
                for i in 0..n {
                    let t = wx[i].tanh();
                    gp += 1.0 - t * t;
                    for j in 0..k {
                        gw_mean[j] += xw.get(i, j) * t;
                    }
                }
                let invn = 1.0 / n as f64;
                gp *= invn;
                for j in 0..k {
                    gw_mean[j] *= invn;
                }
                let mut wnew = Vector::from_iter((0..k).map(|j| gw_mean[j] - gp * w[j]));
                for prev in 0..c {
                    let mut dot = 0.0;
                    for j in 0..k {
                        dot += wnew[j] * wmat.get(prev, j);
                    }
                    for j in 0..k {
                        wnew[j] -= dot * wmat.get(prev, j);
                    }
                }
                let nn = wnew.norm();
                if nn <= ctx.policy.near_zero_variance {
                    ctx.push(
                        Issue::builder(IssueCode::StepSizeCollapsed)
                            .message(format!("FastICA unit {c} collapsed"))
                            .build(),
                    );
                    break;
                }
                wnew = wnew.scale(1.0 / nn);
                let lim = w.dot(&wnew).abs();
                w = wnew;
                if (1.0 - lim) < 1e-6 {
                    converged = true;
                    let _ = it;
                    break;
                }
            }
            if !converged {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message(format!("FastICA unit {c} did not converge"))
                        .build(),
                );
            }
            for j in 0..k {
                wmat.set(c, j, w[j]);
            }
        }
        // components in original space: W_ica * whitening
        let components = matmul(&wmat, &whitening);
        push_nonfinite_mat(&mut ctx, &components, "FastICA components");
        ctx.finish(FittedFastIca {
            components,
            mean,
            whitening,
        })
    }
}

/// Factor analysis: SVD / EM-style uniqueness model `cov ≈ ΛΛᵀ + diag(ψ)`.
#[derive(Clone, Debug)]
pub struct FactorAnalysis {
    /// Number of latent factors.
    pub n_components: usize,
    /// EM iterations (0 = SVD approximation only).
    pub max_iter: usize,
}

impl Default for FactorAnalysis {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 25,
        }
    }
}

impl FactorAnalysis {
    /// `k` factors.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedFactorAnalysis>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted factor-analysis model.
#[derive(Clone, Debug)]
pub struct FittedFactorAnalysis {
    /// Loadings `Λ` (`p` × `k`).
    pub loadings: Matrix,
    /// Uniqueness / specific variances (`p`).
    pub uniqueness: Vector,
    /// Column means.
    pub mean: Vector,
}

impl Transform for FittedFactorAnalysis {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        let k = self.loadings.ncols();
        let pc = self.mean.len().min(p).min(self.loadings.nrows());
        // Bartlett / ridge scores: (Λᵀ Ψ⁻¹ Λ)⁻¹ Λᵀ Ψ⁻¹ (x−μ)
        let mut out = Matrix::zeros(n, k);
        for i in 0..n {
            let mut rhs = Vector::zeros(k);
            for a in 0..k {
                let mut s = 0.0;
                for j in 0..pc {
                    let psi = self.uniqueness[j].max(1e-8);
                    s += self.loadings.get(j, a) * (x.get(i, j) - self.mean[j]) / psi;
                }
                rhs[a] = s;
            }
            let mut gram = Mat::<f64>::zeros(k, k);
            for a in 0..k {
                for b in 0..k {
                    let mut s = 0.0;
                    for j in 0..pc {
                        let psi = self.uniqueness[j].max(1e-8);
                        s += self.loadings.get(j, a) * self.loadings.get(j, b) / psi;
                    }
                    gram[(a, b)] = s;
                    if a == b {
                        gram[(a, b)] += 1e-8;
                    }
                }
            }
            if let Some(sol) = chol_solve(&mut ctx.report, &gram, &rhs, &ctx.policy) {
                for a in 0..k {
                    out.set(i, a, sol[a]);
                }
            }
        }
        ctx.finish(out)
    }
}

impl FitUnsupervised for FactorAnalysis {
    type Fitted = FittedFactorAnalysis;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedFactorAnalysis>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.n_components.max(1), &ctx.policy);
        let (n, p) = x.shape();
        if n == 0 || p == 0 {
            return ctx.finish(FittedFactorAnalysis {
                loadings: Matrix::zeros(p, self.n_components),
                uniqueness: Vector::zeros(p),
                mean: Vector::zeros(p),
            });
        }
        let (xc, mean) = x.centered();
        let mut cov = xc.gram();
        let invn = 1.0 / n.max(1) as f64;
        for i in 0..p {
            for j in 0..p {
                cov[(i, j)] *= invn;
            }
        }
        let Some((evals, evecs)) = symmetric_eigen(&mut ctx.report, &cov, &ctx.policy) else {
            return ctx.finish(FittedFactorAnalysis {
                loadings: Matrix::zeros(p, self.n_components.min(p)),
                uniqueness: Vector::filled(p, 1.0),
                mean,
            });
        };
        let mut order: Vec<usize> = (0..evals.len()).collect();
        order.sort_by(|&a, &b| evals[b].partial_cmp(&evals[a]).unwrap_or(std::cmp::Ordering::Equal));
        let rank = evals.iter().filter(|e| **e > ctx.policy.rank_tol_relative).count();
        let k_req = self.n_components.max(1);
        let k = k_req.min(p).min(rank.max(1));
        if k_req > rank {
            ctx.push(components_exceed_rank(k_req, rank));
        }
        let mut uniqueness = Vector::zeros(p);
        for j in 0..p {
            uniqueness[j] = cov[(j, j)].max(1e-8);
        }
        let mut loadings = Matrix::zeros(p, k);
        for iter in 0..self.max_iter.max(1) {
            // SVD-style loadings: Λ = U √max(λ−ψ̄, 0)
            let psi_bar = uniqueness.mean();
            for c in 0..k {
                let ev = evals.get(order[c]).copied().unwrap_or(0.0);
                let scale = (ev - psi_bar).max(0.0).sqrt();
                if ev < 0.0 {
                    ctx.push(
                        Issue::builder(IssueCode::NegativeEigenvalueDropped)
                            .message(format!("factor-analysis covariance eigenvalue {ev:.3e}"))
                            .metric("eigenvalue", ev)
                            .build(),
                    );
                }
                for j in 0..p {
                    let u = if j < evecs.nrows() && order[c] < evecs.ncols() {
                        evecs[(j, order[c])]
                    } else {
                        0.0
                    };
                    loadings.set(j, c, u * scale);
                }
            }
            for j in 0..p {
                let mut comm = 0.0;
                for c in 0..k {
                    comm += loadings.get(j, c) * loadings.get(j, c);
                }
                let psi = (cov[(j, j)] - comm).max(1e-8);
                uniqueness[j] = psi;
                if (cov[(j, j)] - comm).abs() <= ctx.policy.near_zero_variance {
                    ctx.push(
                        Issue::builder(IssueCode::DegenerateDistribution)
                            .message(format!("factor {j} uniqueness collapsed (Heywood case)"))
                            .metric("feature", j as f64)
                            .build(),
                    );
                }
            }
            let _ = iter;
        }
        push_nonfinite_mat(&mut ctx, &loadings, "FA loadings");
        ctx.finish(FittedFactorAnalysis {
            loadings,
            uniqueness,
            mean,
        })
    }
}

/// Canonical correlation of the columns of `X` with a second view `y`.
///
/// Implemented as the SVD of the whitened cross-covariance
/// `Σ_{xx}^{-1/2} Σ_{xy} Σ_{yy}^{-1/2}` (here `y` is univariate so the SVD
/// is a single right/left vector).
#[derive(Clone, Debug)]
pub struct Cca {
    /// Number of canonical pairs requested (at most 1 when `y` is a vector).
    pub n_components: usize,
}

impl Default for Cca {
    fn default() -> Self {
        Self { n_components: 1 }
    }
}

impl Cca {
    /// Request `n_components` canonical pairs.
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }
}

/// Fitted CCA.
#[derive(Clone, Debug)]
pub struct FittedCca {
    /// Weights for `X` (`p` × `k`).
    pub x_weights: Matrix,
    /// Weights for `y` (length `k`).
    pub y_weights: Vector,
    /// Canonical correlations.
    pub correlations: Vector,
    /// `X` column means.
    pub x_mean: Vector,
    /// `y` mean.
    pub y_mean: f64,
}

impl Transform for FittedCca {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        let k = self.x_weights.ncols();
        let pc = self.x_mean.len().min(p).min(self.x_weights.nrows());
        let out = Matrix::from_fn(n, k, |i, c| {
            let mut s = 0.0;
            for j in 0..pc {
                s += (x.get(i, j) - self.x_mean[j]) * self.x_weights.get(j, c);
            }
            s
        });
        ctx.finish(out)
    }
}

impl Fit for Cca {
    type Fitted = FittedCca;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedCca>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.n_components.max(1), &ctx.policy);
        let (n, p) = x.shape();
        if n == 0 || p == 0 {
            return ctx.finish(FittedCca {
                x_weights: Matrix::zeros(p, self.n_components),
                y_weights: Vector::zeros(self.n_components),
                correlations: Vector::zeros(self.n_components),
                x_mean: Vector::zeros(p),
                y_mean: 0.0,
            });
        }
        let (xc, x_mean) = x.centered();
        let y_mean = y.mean();
        let yc = Vector::from_iter(y.as_slice().iter().map(|v| v - y_mean));
        if yc.std() <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::ConstantTarget)
                    .message("CCA second view has zero variance")
                    .meaninglessness(Meaninglessness::vacuous(
                        "canonical correlation",
                        "a constant y has no correlation with any linear combination of X",
                        "do not interpret the canonical weights",
                    ))
                    .build(),
            );
        }
        let Some(svd) = thin_svd(&mut ctx.report, &xc, &ctx.policy) else {
            return ctx.finish(FittedCca {
                x_weights: Matrix::zeros(p, 1),
                y_weights: Vector::from_slice(&[1.0]),
                correlations: Vector::zeros(1),
                x_mean,
                y_mean,
            });
        };
        let rank = svd.rank(ctx.policy.rank_tol_relative);
        if self.n_components > 1 {
            ctx.push(components_exceed_rank(self.n_components, 1.min(rank)));
        }
        if rank == 0 {
            ctx.push(
                Issue::builder(IssueCode::RankZero)
                    .message("CCA X is rank-0 after centering")
                    .build(),
            );
        }
        // Whiten X: K = V S⁺ ,  Xw = Xc K (n × r)
        let r = rank.max(1).min(svd.singular_values.len());
        let scale = ((n.saturating_sub(1)).max(1) as f64).sqrt();
        let mut kmat = Matrix::zeros(p, r);
        for c in 0..r {
            let sig = svd.singular_values[c];
            if sig <= 0.0 {
                continue;
            }
            let coeff = scale / sig;
            for j in 0..p {
                if j < svd.v.nrows() && c < svd.v.ncols() {
                    kmat.set(j, c, coeff * svd.v[(j, c)]);
                }
            }
        }
        let xw = matmul(&xc, &kmat); // n × r
        // Cross-covariance in whitened space: c = Xwᵀ y / (n−1)  (r × 1).
        let df = (n.saturating_sub(1)).max(1) as f64;
        let mut cross = Matrix::zeros(r, 1);
        for c in 0..r {
            let mut s = 0.0;
            for i in 0..n {
                s += xw.get(i, c) * yc[i];
            }
            cross.set(c, 0, s / df);
        }
        let ystd = yc.std().max(1e-15);
        // Whitened y is yc / (s_y); SVD of the r×1 cross matrix.
        let Some(csvd) = thin_svd(&mut ctx.report, &cross, &ctx.policy) else {
            return ctx.finish(FittedCca {
                x_weights: Matrix::zeros(p, 1),
                y_weights: Vector::from_slice(&[1.0 / ystd]),
                correlations: Vector::zeros(1),
                x_mean,
                y_mean,
            });
        };
        let u0 = if csvd.u.nrows() > 0 && csvd.u.ncols() > 0 {
            Vector::from_iter((0..r).map(|i| {
                if i < csvd.u.nrows() {
                    csvd.u[(i, 0)]
                } else {
                    0.0
                }
            }))
        } else {
            Vector::filled(r, 0.0)
        };
        // x_weight = K u  (p)
        let mut xw_col = Vector::zeros(p);
        for j in 0..p {
            let mut s = 0.0;
            for c in 0..r {
                s += kmat.get(j, c) * u0[c];
            }
            xw_col[j] = s;
        }
        let corr = csvd.singular_values.first().copied().unwrap_or(0.0) / ystd;
        let mut x_weights = Matrix::zeros(p, 1);
        for j in 0..p {
            x_weights.set(j, 0, xw_col[j]);
        }
        ctx.finish(FittedCca {
            x_weights,
            y_weights: Vector::from_slice(&[1.0 / ystd]),
            correlations: Vector::from_slice(&[corr.clamp(-1.0, 1.0)]),
            x_mean,
            y_mean,
        })
    }
}

/// Sparse PCA by iterative soft-thresholding of the loadings (deflation).
#[derive(Clone, Debug)]
pub struct SparsePca {
    /// Number of sparse components.
    pub n_components: usize,
    /// Soft-threshold level applied to loadings.
    pub alpha: f64,
    /// Alternating iterations per component.
    pub max_iter: usize,
}

impl Default for SparsePca {
    fn default() -> Self {
        Self {
            n_components: 2,
            alpha: 0.1,
            max_iter: 50,
        }
    }
}

impl SparsePca {
    /// `k` sparse axes.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedSparsePca>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted sparse PCA.
#[derive(Clone, Debug)]
pub struct FittedSparsePca {
    /// Sparse principal axes (`k` × `p`).
    pub components: Matrix,
    /// Column means.
    pub mean: Vector,
}

impl Transform for FittedSparsePca {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        let pc = self.mean.len().min(p);
        let xc = Matrix::from_fn(n, pc, |i, j| x.get(i, j) - self.mean[j]);
        ctx.finish(project_centered(&xc, &self.components))
    }
}

fn soft_threshold(v: f64, alpha: f64) -> f64 {
    if v > alpha {
        v - alpha
    } else if v < -alpha {
        v + alpha
    } else {
        0.0
    }
}

impl FitUnsupervised for SparsePca {
    type Fitted = FittedSparsePca;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedSparsePca>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.n_components.max(1), &ctx.policy);
        let (n, p) = x.shape();
        if n == 0 || p == 0 {
            return ctx.finish(FittedSparsePca {
                components: Matrix::zeros(self.n_components, p),
                mean: Vector::zeros(p),
            });
        }
        let (mut residual, mean) = x.centered();
        let Some(svd) = thin_svd(&mut ctx.report, &residual, &ctx.policy) else {
            return ctx.finish(FittedSparsePca {
                components: Matrix::zeros(self.n_components.min(p), p),
                mean,
            });
        };
        let rank = svd.rank(ctx.policy.rank_tol_relative);
        let k_req = self.n_components.max(1);
        let k = k_req.min(p).min(rank.max(1));
        if k_req > rank {
            ctx.push(components_exceed_rank(k_req, rank));
        }
        let mut components = Matrix::zeros(k, p);
        for c in 0..k {
            let mut w = if c < svd.v.ncols() {
                Vector::from_iter((0..p).map(|j| {
                    if j < svd.v.nrows() {
                        svd.v[(j, c)]
                    } else {
                        0.0
                    }
                }))
            } else {
                Vector::filled(p, 1.0 / (p as f64).sqrt())
            };
            for _ in 0..self.max_iter.max(1) {
                let scores = residual.matvec(&w);
                let mut wnew = residual.matvec_t(&scores);
                for j in 0..p {
                    wnew[j] = soft_threshold(wnew[j], self.alpha);
                }
                let nn = wnew.norm();
                if nn <= ctx.policy.near_zero_variance {
                    ctx.push(
                        Issue::builder(IssueCode::StepSizeCollapsed)
                            .message(format!("sparse PCA component {c} thresholded to 0"))
                            .build(),
                    );
                    break;
                }
                w = wnew.scale(1.0 / nn);
            }
            for j in 0..p {
                components.set(c, j, w[j]);
            }
            // Deflate.
            let scores = residual.matvec(&w);
            residual = Matrix::from_fn(n, p, |i, j| residual.get(i, j) - scores[i] * w[j]);
            let nnz = (0..p).filter(|&j| components.get(c, j).abs() > 0.0).count();
            if nnz == 0 {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message(format!("sparse PCA component {c} is the zero vector"))
                        .build(),
                );
            }
        }
        push_nonfinite_mat(&mut ctx, &components, "sparse PCA components");
        ctx.finish(FittedSparsePca { components, mean })
    }
}

/// Dictionary learning (a few MOD / ISTA iterations).
#[derive(Clone, Debug)]
pub struct DictionaryLearning {
    /// Number of atoms.
    pub n_components: usize,
    /// Alternating iterations.
    pub max_iter: usize,
    /// Soft-threshold on codes (ISTA).
    pub alpha: f64,
    /// Seed for atom initialization.
    pub seed: u64,
}

impl Default for DictionaryLearning {
    fn default() -> Self {
        Self {
            n_components: 4,
            max_iter: 8,
            alpha: 0.1,
            seed: 0,
        }
    }
}

impl DictionaryLearning {
    /// `k` atoms.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedDictionary>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted dictionary: `X ≈ codes * dictionary` with unit-norm atoms as rows.
#[derive(Clone, Debug)]
pub struct FittedDictionary {
    /// Dictionary (`k` × `p`), rows are atoms.
    pub dictionary: Matrix,
    /// Sparse codes (`n` × `k`).
    pub codes: Matrix,
}

impl Transform for FittedDictionary {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let codes = ista_codes(x, &self.dictionary, 0.1, 30);
        ctx.finish(codes)
    }
}

fn ista_codes(x: &Matrix, dict: &Matrix, alpha: f64, iters: usize) -> Matrix {
    // codes ← soft(codes − η (codes D − X) D�    // codes ← soft(codes − η (codes D − X) Dᵀ, α)
    // D is k × p, X is n × p, codes n × k.
    let n = x.nrows();
    let k = dict.nrows();
    let p = x.ncols().min(dict.ncols());
    let mut codes = Matrix::zeros(n, k);
    let mut lip = 0.0;
    for c in 0..k {
        let mut n2 = 0.0;
        for j in 0..p {
            n2 += dict.get(c, j) * dict.get(c, j);
        }
        lip += n2;
    }
    let eta = 1.0 / lip.max(1.0);
    for _ in 0..iters {
        let recon = matmul_nt(&codes, dict); // n × p
        // grad = (recon − X) Dᵀ   → n × k
        let resid = Matrix::from_fn(n, p, |i, j| recon.get(i, j) - x.get(i, j));
        let grad = matmul_nt(&resid, dict);
        codes = Matrix::from_fn(n, k, |i, c| {
            soft_threshold(codes.get(i, c) - eta * grad.get(i, c), alpha)
        });
    }
    codes
}

impl FitUnsupervised for DictionaryLearning {
    type Fitted = FittedDictionary;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedDictionary>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.n_components.max(1), &ctx.policy);
        let (n, p) = x.shape();
        let k = self.n_components.max(1).min(n.max(1)).min(p.max(1) * 4);
        if n == 0 || p == 0 {
            return ctx.finish(FittedDictionary {
                dictionary: Matrix::zeros(k, p),
                codes: Matrix::zeros(n, k),
            });
        }
        if k < self.n_components {
            ctx.push(components_exceed_rank(self.n_components, k));
        }
        let mut rng = Rng::new(self.seed | 1);
        let idx = rng.sample_indices(n, k.min(n));
        let mut dict = Matrix::zeros(k, p);
        for (c, &i) in idx.iter().enumerate() {
            let mut n2 = 0.0;
            for j in 0..p {
                let v = x.get(i, j);
                dict.set(c, j, v);
                n2 += v * v;
            }
            let nn = n2.sqrt().max(1e-12);
            for j in 0..p {
                dict.set(c, j, dict.get(c, j) / nn);
            }
        }
        if idx.len() < k {
            for c in idx.len()..k {
                let mut n2 = 0.0;
                for j in 0..p {
                    let v = rng.standard_normal();
                    dict.set(c, j, v);
                    n2 += v * v;
                }
                let nn = n2.sqrt().max(1e-12);
                for j in 0..p {
                    dict.set(c, j, dict.get(c, j) / nn);
                }
            }
        }
        let mut codes = Matrix::zeros(n, k);
        for it in 0..self.max_iter.max(1) {
            codes = ista_codes(x, &dict, self.alpha, 15);
            // MOD: D ← (CᵀC + λI)⁻¹ Cᵀ X   written as ridge per column of X.
            // dictionary rows are atoms: solve for each feature column.
            let ct = Matrix::from_fn(k, n, |c, i| codes.get(i, c));
            for j in 0..p {
                let yj = x.column(j);
                // We need (CᵀC + λI) d = Cᵀ x_{:j}. C is n×k, so use ridge_solve on codes.
                if let Some(atom_col) = ridge_solve(&mut ctx.report, &codes, &yj, 1e-4, &ctx.policy)
                {
                    for c in 0..k.min(atom_col.len()) {
                        dict.set(c, j, atom_col[c]);
                    }
                }
            }
            let _ = ct;
            for c in 0..k {
                let mut n2 = 0.0;
                for j in 0..p {
                    n2 += dict.get(c, j) * dict.get(c, j);
                }
                let nn = n2.sqrt();
                if nn <= ctx.policy.near_zero_variance {
                    ctx.push(
                        Issue::builder(IssueCode::UnidentifiedModel)
                            .message(format!("dictionary atom {c} has vanishing norm"))
                            .build(),
                    );
                    continue;
                }
                for j in 0..p {
                    dict.set(c, j, dict.get(c, j) / nn);
                }
            }
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("dictionary learning hit max_iter")
                        .build(),
                );
            }
        }
        push_nonfinite_mat(&mut ctx, &dict, "dictionary");
        push_nonfinite_mat(&mut ctx, &codes, "codes");
        ctx.finish(FittedDictionary {
            dictionary: dict,
            codes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    fn rank1() -> Matrix {
        // X[i, j] = (i+1) * (j+1)  — exact rank 1, every column varies.
        // n ≥ 5 · n_components so inspect_identification does not abort the
        // n_components=6 warning test (min_samples_per_parameter = 5).
        Matrix::from_fn(40, 4, |i, j| (i as f64 + 1.0) * (j as f64 + 1.0))
    }

    #[test]
    fn pca_rank1_first_component_explains_all() {
        let x = rank1();
        let session = Session::new("pca", "fit");
        let q = Pca::new(2)
            .fit(&x, &session)
            .expect("pca");
        let ev = q.value.explained_variance_ratio.as_slice();
        assert!(!ev.is_empty());
        assert!(
            ev[0] > 0.99,
            "first component should own the rank-1 variance, got {ev:?}"
        );
        let z = q
            .value
            .transform(&x, &Session::new("pca", "transform"))
            .expect("transform");
        assert_eq!(z.value.ncols(), q.value.components.nrows());
    }

    #[test]
    fn pca_components_exceed_rank_warns() {
        let x = rank1();
        let session = Session::new("pca", "fit");
        let q = Pca::new(6).fit(&x, &session).expect("pca should warn, not abort");
        assert!(
            q.report.contains(IssueCode::ComponentsExceedRank),
            "expected ComponentsExceedRank, issues={:?}",
            q.report
                .issues()
                .iter()
                .map(|i| i.code)
                .collect::<Vec<_>>()
        );
        assert!(q.value.rank <= 1, "rank-1 data, got rank {}", q.value.rank);
    }

    #[test]
    fn incremental_pca_explains() {
        let x = rank1();
        let session = Session::new("ipca", "partial_fit");
        let mut m = IncrementalPca::new(1);
        let q = m.partial_fit(&x, None, &session).expect("ipca");
        assert!(!q.value.narrative.is_empty());
        assert!(q.value.quality.n_seen > 0);
    }
}
