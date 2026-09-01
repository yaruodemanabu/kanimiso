//! Matrix decompositions and latent-factor models.
//!
//! PCA / truncated SVD go through [`thin_svd`] of (centered) `X`. Incremental
//! PCA maintains a running mean and a sequential Karhunen–Loève basis and
//! **must** emit [`IncrementalExplain`] on every `partial_fit`. NMF, FastICA,
//! factor analysis, CCA, sparse PCA, and dictionary learning are pure-Rust
//! iterative methods that still record rank, NaN, and meaningless-fit issues.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{chol_solve, least_squares, ridge_solve, symmetric_eigen, thin_svd};
use crate::rng::Rng;
use crate::traits::{Fit, FitUnsupervised, PartialFit, Transform};
use crate::validate::{inspect_identification, inspect_xy};
use faer::Mat;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{
    IncrementalQuality, Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Report,
    Result, Severity,
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
    Matrix::from_fn(kk, p, |c, j| if j < p && c < r { v[(j, c)] } else { 0.0 })
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

pub(crate) use crate::kernel_pca::{FittedKernelPca, KernelPca};

/// Principal component analysis via a thin SVD of column-centered `X`.
#[derive(Clone, Debug)]
pub(crate) struct Pca {
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
    pub(crate) fn new(n_components: usize) -> Self {
        Self { n_components }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedPca>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted PCA.
#[derive(Clone, Debug)]
pub(crate) struct FittedPca {
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
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedPca>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_components.max(1),
            &ctx.policy,
        );
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
        let k_req = self.n_components.max(1);
        let k = k_req.min(rank.max(1)).min(svd.singular_values.len()).min(p);
        // κ of the *retained* spectrum only. The full thin SVD of a rank-deficient
        // matrix has σ_min = 0 ⇒ κ = ∞; Policy would rewrite that IllConditioned
        // metric into a fatal NearSingular and abort a perfectly valid PCA.
        if k >= 2 {
            let smax = svd.singular_values[0];
            let smin = svd.singular_values[k - 1];
            let kappa = if smin > 0.0 { smax / smin } else { f64::NAN };
            if kappa.is_finite() && kappa > ctx.policy.condition_number_warn {
                ctx.push(
                    Issue::builder(IssueCode::IllConditioned)
                        .message(format!("PCA retained-spectrum condition κ={kappa:.4e}"))
                        .metric("condition_number", kappa)
                        .build(),
                );
            }
        }
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
pub(crate) struct IncrementalPca {
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
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Current mean, if any batch has been seen.
    pub(crate) fn mean(&self) -> Option<&Vector> {
        self.mean.as_ref()
    }

    /// Current components (`k` × `p`).
    pub(crate) fn components(&self) -> Option<&Matrix> {
        self.components.as_ref()
    }

    /// Fit alias (one SKL step on the whole matrix).
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedPca>> {
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
        let (explained_variance, explained_variance_ratio) =
            explained_from_singular(&s, n.max(2), s.len());
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
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedPca>> {
        let mut this = self.clone();
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            this.n_components.max(1),
            &ctx.policy,
        );
        if x.nrows() == 0 || x.ncols() == 0 {
            return ctx.finish(this.to_fitted(x.ncols()));
        }
        let _ = this.partial_fit(x, None, &session.child("ipca_init"));
        ctx.finish(this.to_fitted(x.ncols()))
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
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_components.max(1),
            &ctx.policy,
        );
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
        q.parameter_delta_max = moved
            .iter()
            .map(|(_, d)| *d)
            .fold(None, |a, b| Some(a.unwrap_or(0.0).max(b)));
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
            format!(
                "principal axes moved {moved:?}; singular values {:?}",
                new_s.as_slice()
            ),
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
pub(crate) struct TruncatedSvd {
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
    pub(crate) fn new(n_components: usize) -> Self {
        Self { n_components }
    }

    /// Fit alias.
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedTruncatedSvd>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted truncated SVD.
#[derive(Clone, Debug)]
pub(crate) struct FittedTruncatedSvd {
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
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedTruncatedSvd>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_components.max(1),
            &ctx.policy,
        );
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
                .message(format!(
                    "keeping {k} of {} singular triples",
                    svd.singular_values.len()
                ))
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
pub(crate) struct Nmf {
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
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedNmf>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted NMF: `X ≈ W H` with `W` (`n` × `k`) and `H` (`k` × `p`) nonnegative.
#[derive(Clone, Debug)]
pub(crate) struct FittedNmf {
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
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedNmf>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_components.max(1),
            &ctx.policy,
        );
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

/// Mini-batch NMF: multiplicative updates of `H` on successive batches.
#[derive(Clone, Debug)]
pub(crate) struct MiniBatchNmf {
    /// Number of latent components.
    pub n_components: usize,
    /// Rows per update.
    pub batch_size: usize,
    /// Seed for the nonnegative start.
    pub seed: u64,
    h: Option<Matrix>,
    n_seen: u64,
    updates: u64,
}

impl Default for MiniBatchNmf {
    fn default() -> Self {
        Self {
            n_components: 2,
            batch_size: 16,
            seed: 0,
            h: None,
            n_seen: 0,
            updates: 0,
        }
    }
}

impl MiniBatchNmf {
    /// Mini-batch NMF with `k` components.
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Current right factor, if initialized.
    pub(crate) fn h(&self) -> Option<&Matrix> {
        self.h.as_ref()
    }
}

impl PartialFit for MiniBatchNmf {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        if n == 0 || p == 0 {
            if self.h.is_none() {
                ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            }
            ctx.session
                .record_incremental(dummy_explain(self.updates, n, self.n_seen));
            return ctx.finish(dummy_explain(self.updates, n, self.n_seen));
        }
        let k = self.n_components.max(1).min(p.max(1));
        if self.h.as_ref().map(|h| h.ncols()) != Some(p) {
            if self.h.is_some() {
                ctx.push(
                    Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                        .message("MiniBatchNMF feature dimension changed")
                        .build(),
                );
            }
            let mut rng = Rng::new(self.seed | 1);
            self.h = Some(Matrix::from_fn(k, p, |_, _| rng.uniform().abs() + 1e-3));
        }
        let xp = Matrix::from_fn(n, p, |i, j| x.get(i, j).max(0.0));
        let mut h = self.h.clone().unwrap();
        let before = h.frobenius();
        let mut rng = Rng::new(self.seed.wrapping_add(self.updates + 1));
        let mut w = Matrix::from_fn(n, k, |_, _| rng.uniform().abs() + 1e-3);
        for _ in 0..8 {
            let xht = matmul_nt(&xp, &h);
            let hht = matmul_nt(&h, &h);
            let w_hht = matmul(&w, &hht);
            w = Matrix::from_fn(n, k, |i, c| {
                (w.get(i, c) * xht.get(i, c) / w_hht.get(i, c).max(1e-12)).max(0.0)
            });
            let wtx = matmul_tn(&w, &xp);
            let wtw = matmul_tn(&w, &w);
            let wtw_h = matmul(&wtw, &h);
            h = Matrix::from_fn(k, p, |c, j| {
                (h.get(c, j) * wtx.get(c, j) / wtw_h.get(c, j).max(1e-12)).max(0.0)
            });
        }
        let after = h.frobenius();
        self.h = Some(h);
        self.n_seen += n as u64;
        self.updates += 1;
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), n, self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some((after - before).abs());
        q.information_gain = Some((after - before).abs() + n as f64);
        q.still_identified = self.n_seen >= k as u64;
        q.warmup = self.n_seen < (5 * k) as u64;
        q.explanation = format!("mini-batch NMF: ||H|| {before:.4e} → {after:.4e} on {n} rows");
        if q.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("MiniBatchNMF has seen fewer than 5k rows")
                    .build(),
            );
        }
        let expl = IncrementalExplain::from_quality(
            q,
            "multiplicative update of H",
            "Lee–Seung steps on the current batch",
            format!("||H||={before:.4e}"),
            format!("||H||={after:.4e}"),
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

impl FitUnsupervised for MiniBatchNmf {
    type Fitted = FittedNmf;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedNmf>> {
        let mut this = self.clone();
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let bs = this.batch_size.max(1);
        let n = x.nrows();
        if n == 0 {
            return ctx.finish(FittedNmf {
                w: Matrix::zeros(0, this.n_components),
                h: Matrix::zeros(this.n_components, x.ncols()),
                reconstruction_err: f64::NAN,
            });
        }
        let mut start = 0usize;
        while start < n {
            let end = (start + bs).min(n);
            let batch = Matrix::from_fn(end - start, x.ncols(), |i, j| x.get(start + i, j));
            match this.partial_fit(&batch, None, &session.child("mb")) {
                Ok(q) => {
                    for issue in q.report.issues() {
                        if issue.code == IssueCode::WarmupIncomplete {
                            continue;
                        }
                        ctx.push(issue.clone());
                    }
                }
                Err(e) => ctx.push(e.primary),
            }
            start = end;
        }
        let h = this
            .h
            .clone()
            .unwrap_or_else(|| Matrix::zeros(this.n_components.max(1), x.ncols()));
        let mut nmf = FittedNmf {
            w: Matrix::zeros(n, h.nrows()),
            h: h.clone(),
            reconstruction_err: f64::NAN,
        };
        match nmf.transform(x, &session.child("codes")) {
            Ok(q) => nmf.w = q.value,
            Err(e) => ctx.push(e.primary),
        }
        let recon = matmul(&nmf.w, &nmf.h);
        let mut err = 0.0;
        for j in 0..x.ncols() {
            for i in 0..n {
                let d = x.get(i, j).max(0.0) - recon.get(i, j);
                err += d * d;
            }
        }
        nmf.reconstruction_err = err.sqrt();
        ctx.finish(nmf)
    }
}

/// FastICA: SVD whitening plus one-unit deflation with `g = tanh`.
#[derive(Clone, Debug)]
pub(crate) struct FastIca {
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
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedFastIca>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted FastICA.
#[derive(Clone, Debug)]
pub(crate) struct FittedFastIca {
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
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedFastIca>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_components.max(1),
            &ctx.policy,
        );
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
            if sig
                <= ctx.policy.rank_tol_relative
                    * svd.singular_values.first().copied().unwrap_or(1.0)
            {
                ctx.push(
                    Issue::builder(IssueCode::RankDeficient)
                        .message(format!(
                            "FastICA whitening skipped near-zero σ[{c}]={sig:.3e}"
                        ))
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
pub(crate) struct FactorAnalysis {
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
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedFactorAnalysis>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted factor-analysis model.
#[derive(Clone, Debug)]
pub(crate) struct FittedFactorAnalysis {
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
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedFactorAnalysis>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_components.max(1),
            &ctx.policy,
        );
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
        order.sort_by(|&a, &b| {
            evals[b]
                .partial_cmp(&evals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let rank = evals
            .iter()
            .filter(|e| **e > ctx.policy.rank_tol_relative)
            .count();
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
pub(crate) struct Cca {
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
    pub(crate) fn new(n_components: usize) -> Self {
        Self { n_components }
    }
}

/// Fitted CCA.
#[derive(Clone, Debug)]
pub(crate) struct FittedCca {
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
    fn fit(&self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedCca>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_components.max(1),
            &ctx.policy,
        );
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
pub(crate) struct SparsePca {
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
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedSparsePca>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted sparse PCA.
#[derive(Clone, Debug)]
pub(crate) struct FittedSparsePca {
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
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedSparsePca>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_components.max(1),
            &ctx.policy,
        );
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

/// Mini-batch sparse PCA (sklearn `MiniBatchSparsePCA`).
///
/// Each `partial_fit` takes one ISTA step on the batch Gram and must emit
/// [`IncrementalExplain`]. Do not pass `n_components` as identification `p`.
#[derive(Clone, Debug)]
pub(crate) struct MiniBatchSparsePca {
    /// Number of sparse components.
    pub n_components: usize,
    /// Soft-threshold level.
    pub alpha: f64,
    /// Batch size used by the offline `fit` alias.
    pub batch_size: usize,
    components: Matrix,
    mean: Vector,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for MiniBatchSparsePca {
    fn default() -> Self {
        Self {
            n_components: 2,
            alpha: 0.1,
            batch_size: 16,
            components: Matrix::zeros(0, 0),
            mean: Vector::zeros(0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl MiniBatchSparsePca {
    /// `k` sparse axes.
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components: n_components.max(1),
            ..Self::default()
        }
    }

    /// Offline alias: one `partial_fit` on the full design.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedSparsePca>> {
        let mut this = self.clone();
        this.partial_fit(x, None, session)?;
        let mut ctx = FitCtx::with_session(session.child("finish"));
        ctx.finish(FittedSparsePca {
            components: this.components.clone(),
            mean: this.mean.clone(),
        })
    }
}

impl PartialFit for MiniBatchSparsePca {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        if n == 0 || p == 0 {
            return ctx.finish(dummy_explain(self.updates, n, self.n_seen));
        }
        if !self.initialized {
            let k = self.n_components.max(1).min(p);
            self.components = Matrix::from_fn(k, p, |i, j| if i == j { 1.0 } else { 0.0 });
            self.mean = Vector::zeros(p);
            self.initialized = true;
        } else if self.mean.len() != p {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message("MiniBatchSparsePCA feature dimension changed")
                    .build(),
            );
            return ctx.finish(dummy_explain(self.updates, n, self.n_seen));
        }
        let n_old = self.n_seen as f64;
        let before = self.components.clone();
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += x.get(i, j);
            }
            let bm = s / n as f64;
            self.mean[j] = if n_old + n as f64 > 0.0 {
                (n_old * self.mean[j] + n as f64 * bm) / (n_old + n as f64)
            } else {
                bm
            };
        }
        let k = self.components.nrows();
        for c in 0..k {
            let mut scores = Vector::zeros(n);
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..p {
                    s += (x.get(i, j) - self.mean[j]) * self.components.get(c, j);
                }
                scores[i] = s;
            }
            let mut w = Vector::zeros(p);
            for j in 0..p {
                let mut s = 0.0;
                for i in 0..n {
                    s += (x.get(i, j) - self.mean[j]) * scores[i];
                }
                w[j] = soft_threshold(s, self.alpha);
            }
            let nn = w.norm();
            if nn <= ctx.policy.near_zero_variance {
                ctx.push(
                    Issue::builder(IssueCode::UpdateWithZeroInformation)
                        .message(format!("mini-batch sparse PCA component {c} vanished"))
                        .build(),
                );
                continue;
            }
            w = w.scale(1.0 / nn);
            for j in 0..p {
                self.components.set(c, j, w[j]);
            }
        }
        self.n_seen += n as u64;
        self.updates += 1;
        let mut delta = 0.0;
        for c in 0..k.min(before.nrows()) {
            for j in 0..p.min(before.ncols()) {
                let d = self.components.get(c, j) - before.get(c, j);
                delta += d * d;
            }
        }
        let mut q = IncrementalQuality::new(self.updates - 1, n, self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.sqrt());
        q.information_gain = Some(delta.sqrt());
        q.still_identified = self.n_seen as usize > p;
        q.warmup = self.n_seen < (p as u64 + 2);
        q.explanation = format!("mini-batch sparse PCA ISTA on {n} rows");
        let expl = IncrementalExplain::from_quality(
            q,
            "sparse principal axes",
            "one ISTA / soft-threshold step on the batch covariance",
            "previous loadings",
            format!("n_seen={}", self.n_seen),
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

/// Dictionary learning (a few MOD / ISTA iterations).
#[derive(Clone, Debug)]
pub(crate) struct DictionaryLearning {
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
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedDictionary>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted dictionary: `X ≈ codes * dictionary` with unit-norm atoms as rows.
#[derive(Clone, Debug)]
pub(crate) struct FittedDictionary {
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
        let recon = matmul(&codes, dict); // n × k · k × p → n × p
        let resid = Matrix::from_fn(n, p, |i, j| recon.get(i, j) - x.get(i, j));
        let grad = matmul_nt(&resid, dict); // (recon − X) Dᵀ → n × k
        codes = Matrix::from_fn(n, k, |i, c| {
            soft_threshold(codes.get(i, c) - eta * grad.get(i, c), alpha)
        });
    }
    codes
}

impl FitUnsupervised for DictionaryLearning {
    type Fitted = FittedDictionary;
    fn fit_unsupervised(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedDictionary>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_components.max(1),
            &ctx.policy,
        );
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

/// Mini-batch dictionary learning (sklearn `MiniBatchDictionaryLearning`).
///
/// Each `partial_fit` takes one ISTA / MOD step and must emit
/// [`IncrementalExplain`]. Atom count is not identification `p`.
#[derive(Clone, Debug)]
pub(crate) struct MiniBatchDictionaryLearning {
    /// Number of atoms.
    pub n_components: usize,
    /// Soft-threshold on codes.
    pub alpha: f64,
    dictionary: Matrix,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for MiniBatchDictionaryLearning {
    fn default() -> Self {
        Self {
            n_components: 4,
            alpha: 0.1,
            dictionary: Matrix::zeros(0, 0),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl MiniBatchDictionaryLearning {
    /// `k` atoms.
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components: n_components.max(1),
            ..Self::default()
        }
    }

    /// Current dictionary (`k` × `p`), if initialized.
    pub(crate) fn dictionary(&self) -> Option<&Matrix> {
        if self.initialized {
            Some(&self.dictionary)
        } else {
            None
        }
    }
}

impl PartialFit for MiniBatchDictionaryLearning {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        if n == 0 || p == 0 {
            return ctx.finish(dummy_explain(self.updates, n, self.n_seen));
        }
        if !self.initialized {
            let k = self.n_components.max(1).min(p.max(1));
            self.dictionary = Matrix::from_fn(k, p, |i, j| if i == j { 1.0 } else { 0.05 });
            self.initialized = true;
        } else if self.dictionary.ncols() != p {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message("MiniBatchDictionaryLearning feature dimension changed")
                    .build(),
            );
            return ctx.finish(dummy_explain(self.updates, n, self.n_seen));
        }
        let before = self.dictionary.clone();
        let codes = ista_codes(x, &self.dictionary, self.alpha, 12);
        let mut scratch = signlred::Report::new("mbdl", "mod");
        for j in 0..p {
            let yj = x.column(j);
            if let Some(atom_col) = ridge_solve(&mut scratch, &codes, &yj, 1e-4, &ctx.policy) {
                for c in 0..self.dictionary.nrows().min(atom_col.len()) {
                    self.dictionary.set(c, j, atom_col[c]);
                }
            }
        }
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let k = self.dictionary.nrows();
        for c in 0..k {
            let mut n2: f64 = 0.0;
            for j in 0..p {
                n2 += self.dictionary.get(c, j) * self.dictionary.get(c, j);
            }
            let nn = n2.sqrt();
            if nn <= ctx.policy.near_zero_variance {
                ctx.push(
                    Issue::builder(IssueCode::UpdateWithZeroInformation)
                        .message(format!("mini-batch dictionary atom {c} vanished"))
                        .build(),
                );
                continue;
            }
            for j in 0..p {
                self.dictionary.set(c, j, self.dictionary.get(c, j) / nn);
            }
        }
        self.n_seen += n as u64;
        self.updates += 1;
        let mut delta: f64 = 0.0;
        for c in 0..k.min(before.nrows()) {
            for j in 0..p.min(before.ncols()) {
                let d = self.dictionary.get(c, j) - before.get(c, j);
                delta += d * d;
            }
        }
        let mut q = IncrementalQuality::new(self.updates - 1, n, self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.sqrt());
        q.information_gain = Some(delta.sqrt());
        q.still_identified = self.n_seen as usize > p;
        q.warmup = self.n_seen < (p as u64 + 2);
        q.explanation = format!("mini-batch dictionary MOD on {n} rows");
        let expl = IncrementalExplain::from_quality(
            q,
            "dictionary atoms",
            "one ISTA code step and MOD atom update on the batch",
            "previous atoms",
            format!("n_seen={}", self.n_seen),
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

/// Sparse coding against a fixed dictionary (sklearn `SparseCoder`).
///
/// Atom count is not identification `p`.
#[derive(Clone, Debug)]
pub(crate) struct SparseCoder {
    /// Soft-threshold on codes.
    pub alpha: f64,
    dictionary: Matrix,
    fitted: bool,
}

impl Default for SparseCoder {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            dictionary: Matrix::zeros(0, 0),
            fitted: false,
        }
    }
}

impl SparseCoder {
    /// Coder with the given atoms (`k` × `p`).
    pub(crate) fn new(dictionary: Matrix) -> Self {
        Self {
            dictionary,
            fitted: true,
            ..Self::default()
        }
    }
}

impl Transform for SparseCoder {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if !self.fitted || self.dictionary.nrows() == 0 {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(Matrix::zeros(x.nrows(), 0));
        }
        if x.ncols() != self.dictionary.ncols() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("SparseCoder X columns ≠ dictionary columns")
                    .build(),
            );
        }
        ctx.finish(ista_codes(x, &self.dictionary, self.alpha, 20))
    }
}

/// Kernel CCA (sklearn `KernelCCA` / Bach–Jordan regularized KCCA).
///
/// Dual size is not identification `p`. Distinct from [`Cca`] (linear
/// whitened cross-covariance).
#[derive(Clone, Debug)]
pub(crate) struct KernelCca {
    /// RBF \(\gamma\). Not identification `p`.
    pub gamma: f64,
    /// Dual ridge. Not identification `p`.
    pub ridge: f64,
}

impl Default for KernelCca {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            ridge: 0.1,
        }
    }
}

impl KernelCca {
    /// Kernel CCA with RBF `gamma`.
    pub(crate) fn new(gamma: f64) -> Self {
        Self {
            gamma,
            ..Self::default()
        }
    }
}

/// Fitted kernel CCA.
#[derive(Clone, Debug)]
pub(crate) struct FittedKernelCca {
    /// Dual coefficients on the centered training kernel.
    pub alpha: Vector,
    /// Canonical correlation.
    pub correlation: f64,
    /// Training rows (for out-of-sample kernels).
    train: Matrix,
    /// Kernel centering mean \(1^\top K / n\).
    k_mean: Vector,
    /// Grand mean of \(K\).
    k_grand: f64,
    gamma: f64,
    /// Training \(y\) mean.
    pub y_mean: f64,
    /// Training \(y\) scale.
    pub y_std: f64,
}

impl Transform for FittedKernelCca {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let m = self.train.nrows();
        let out = Matrix::from_fn(n, 1, |i, _| {
            let mut s = 0.0_f64;
            for j in 0..m.min(self.alpha.len()) {
                let kij = rbf_rows(x, i, &self.train, j, self.gamma);
                let kc = kij
                    - self.k_mean.as_slice().get(j).copied().unwrap_or(0.0)
                    - self.row_mean(x, i)
                    + self.k_grand;
                s += kc * self.alpha[j];
            }
            s
        });
        ctx.finish(out)
    }
}

impl FittedKernelCca {
    fn row_mean(&self, x: &Matrix, i: usize) -> f64 {
        let m = self.train.nrows();
        if m == 0 {
            return 0.0;
        }
        let mut s = 0.0_f64;
        for j in 0..m {
            s += rbf_rows(x, i, &self.train, j, self.gamma);
        }
        s / m as f64
    }
}

fn rbf_rows(x: &Matrix, i: usize, other: &Matrix, j: usize, gamma: f64) -> f64 {
    let g = if gamma.is_finite() && gamma > 0.0 {
        gamma
    } else {
        1.0
    };
    let mut s = 0.0_f64;
    let p = x.ncols().min(other.ncols());
    for c in 0..p {
        let d = x.get(i, c) - other.get(j, c);
        s += d * d;
    }
    (-g * s).exp()
}

impl Fit for KernelCca {
    type Fitted = FittedKernelCca;
    fn fit(&self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedKernelCca>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let n = x.nrows().min(y.len());
        if n == 0 || x.ncols() == 0 {
            return ctx.finish(FittedKernelCca {
                alpha: Vector::zeros(0),
                correlation: 0.0,
                train: x.clone(),
                k_mean: Vector::zeros(0),
                k_grand: 0.0,
                gamma: self.gamma,
                y_mean: 0.0,
                y_std: 1.0,
            });
        }
        let y_mean = y.mean();
        let yc = Vector::from_iter((0..n).map(|i| y[i] - y_mean));
        let y_std = yc.std().max(1e-15);
        if yc.std() <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::ConstantTarget)
                    .message("Kernel CCA second view has zero variance")
                    .meaninglessness(Meaninglessness::vacuous(
                        "kernel canonical correlation",
                        "a constant y has no correlation with any kernel combination of X",
                        "do not interpret the dual weights",
                    ))
                    .build(),
            );
        }
        let gamma = if self.gamma.is_finite() && self.gamma > 0.0 {
            self.gamma
        } else {
            0.1
        };
        let ridge = if self.ridge.is_finite() && self.ridge >= 0.0 {
            self.ridge
        } else {
            0.1
        };
        let mut kraw = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                kraw.set(i, j, rbf_rows(x, i, x, j, gamma));
            }
        }
        let k_mean = Vector::from_iter((0..n).map(|j| {
            let mut s = 0.0_f64;
            for i in 0..n {
                s += kraw.get(i, j);
            }
            s / n as f64
        }));
        let mut k_grand = 0.0_f64;
        for i in 0..n {
            k_grand += k_mean[i];
        }
        k_grand /= n as f64;
        let mut gram = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            let row_mean = {
                let mut s = 0.0_f64;
                for j in 0..n {
                    s += kraw.get(i, j);
                }
                s / n as f64
            };
            for j in 0..n {
                gram[(i, j)] = kraw.get(i, j) - row_mean - k_mean[j] + k_grand;
            }
            gram[(i, i)] += ridge;
        }
        let mut scratch = Report::new("kcca", "chol");
        let alpha = match chol_solve(&mut scratch, &gram, &yc, &ctx.policy) {
            Some(a) => a,
            None => {
                ctx.push(
                    Issue::builder(IssueCode::CholeskyFailed)
                        .severity(Severity::Warning)
                        .message("KernelCca dual Cholesky failed; using a zero dual")
                        .build(),
                );
                Vector::zeros(n)
            }
        };
        let mut pred = Vector::zeros(n);
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..n {
                let kc = kraw.get(i, j)
                    - {
                        let mut rm = 0.0_f64;
                        for t in 0..n {
                            rm += kraw.get(i, t);
                        }
                        rm / n as f64
                    }
                    - k_mean[j]
                    + k_grand;
                s += kc * alpha[j];
            }
            pred[i] = s;
        }
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for i in 0..n {
            num += pred[i] * yc[i];
            den += pred[i] * pred[i];
        }
        let corr = if den > 1e-15 && y_std > 0.0 {
            (num / (den.sqrt() * y_std * (n as f64).sqrt())).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        ctx.finish(FittedKernelCca {
            alpha,
            correlation: corr,
            train: Matrix::from_fn(n, x.ncols(), |i, j| x.get(i, j)),
            k_mean,
            k_grand,
            gamma,
            y_mean,
            y_std,
        })
    }
}

/// One-factor confirmatory factor analysis (statsmodels `Factor` / LISREL-lite).
///
/// MINRES communalities on the correlation matrix. Factor count is fixed at
/// one and is not identification `p`. Distinct from [`FactorAnalysis`]
/// (exploratory multi-factor SVD/EM).
#[derive(Clone, Debug)]
pub(crate) struct ConfirmatoryFactor {
    /// Communality iterations.
    pub max_iter: usize,
}

impl Default for ConfirmatoryFactor {
    fn default() -> Self {
        Self { max_iter: 25 }
    }
}

impl ConfirmatoryFactor {
    /// Default one-factor CFA.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Fit alias.
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedConfirmatoryFactor>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted one-factor CFA.
#[derive(Clone, Debug)]
pub(crate) struct FittedConfirmatoryFactor {
    /// Loadings \(\lambda\) (`p`).
    pub loadings: Vector,
    /// Uniqueness \(\psi_j=1-\lambda_j^2\).
    pub uniqueness: Vector,
    /// Column means.
    pub mean: Vector,
    /// Column scales used to form correlations.
    pub scale: Vector,
    /// Residual sum of squares of off-diagonal correlations.
    pub residual_ss: f64,
}

impl Transform for FittedConfirmatoryFactor {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        let pc = self.mean.len().min(p).min(self.loadings.len());
        let mut denom = 0.0_f64;
        for j in 0..pc {
            let psi = self
                .uniqueness
                .as_slice()
                .get(j)
                .copied()
                .unwrap_or(1.0)
                .max(1e-8);
            denom += self.loadings[j] * self.loadings[j] / psi;
        }
        let den = denom.max(1e-8);
        let out = Matrix::from_fn(n, 1, |i, _| {
            let mut s = 0.0_f64;
            for j in 0..pc {
                let psi = self
                    .uniqueness
                    .as_slice()
                    .get(j)
                    .copied()
                    .unwrap_or(1.0)
                    .max(1e-8);
                let z = (x.get(i, j) - self.mean[j])
                    / self
                        .scale
                        .as_slice()
                        .get(j)
                        .copied()
                        .unwrap_or(1.0)
                        .max(1e-8);
                s += self.loadings[j] * z / psi;
            }
            s / den
        });
        ctx.finish(out)
    }
}

impl FitUnsupervised for ConfirmatoryFactor {
    type Fitted = FittedConfirmatoryFactor;
    fn fit_unsupervised(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedConfirmatoryFactor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let (n, p) = x.shape();
        if n == 0 || p == 0 {
            return ctx.finish(FittedConfirmatoryFactor {
                loadings: Vector::zeros(p),
                uniqueness: Vector::filled(p.max(1), 1.0),
                mean: Vector::zeros(p),
                scale: Vector::filled(p.max(1), 1.0),
                residual_ss: f64::NAN,
            });
        }
        let mean = Vector::from_iter((0..p).map(|j| x.column(j).mean()));
        let scale = Vector::from_iter((0..p).map(|j| x.column(j).std().max(1e-8)));
        let z = Matrix::from_fn(n, p, |i, j| (x.get(i, j) - mean[j]) / scale[j]);
        let df = (n.saturating_sub(1)).max(1) as f64;
        let corr = Matrix::from_fn(p, p, |i, j| {
            let mut s = 0.0_f64;
            for r in 0..n {
                s += z.get(r, i) * z.get(r, j);
            }
            s / df
        });
        let mut lam =
            Vector::from_iter((0..p).map(|j| (corr.get(j, j).abs().sqrt() * 0.5).clamp(0.1, 0.9)));
        for _ in 0..self.max_iter.max(1) {
            let mut nxt = Vector::zeros(p);
            for j in 0..p {
                let mut num = 0.0_f64;
                let mut den = 0.0_f64;
                for k in 0..p {
                    if k == j {
                        continue;
                    }
                    num += corr.get(j, k) * lam[k];
                    den += lam[k] * lam[k];
                }
                nxt[j] = if den > 1e-12 {
                    (num / den).clamp(-0.99, 0.99)
                } else {
                    lam[j]
                };
            }
            lam = nxt;
        }
        let uniqueness =
            Vector::from_iter((0..p).map(|j| (1.0 - lam[j] * lam[j]).clamp(1e-4, 1.0)));
        let mut rss = 0.0_f64;
        for i in 0..p {
            for j in (i + 1)..p {
                let e = corr.get(i, j) - lam[i] * lam[j];
                rss += e * e;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("ConfirmatoryFactor is one-factor MINRES, not a published LISREL CFA")
                .compromise(NumericalCompromise::new(
                    "statsmodels Factor / LISREL CFA",
                    "iterated MINRES on the correlation matrix with a single common factor",
                    "a user loading pattern, ML standard errors, and multi-factor identification are omitted",
                    "read loadings as a 1-factor communality sketch",
                ))
                .build(),
        );
        ctx.finish(FittedConfirmatoryFactor {
            loadings: lam,
            uniqueness,
            mean,
            scale,
            residual_ss: rss,
        })
    }
}

/// LISREL-lite: one-factor measurement plus a structural slope on \(y\).
///
/// Factor count is not identification `p`. Distinct from [`ConfirmatoryFactor`]
/// (no structural equation) and ordinary OLS (no latent).
#[derive(Clone, Debug, Default)]
pub(crate) struct Lisrel;

impl Lisrel {
    /// Default one-factor LISREL.
    pub(crate) fn new() -> Self {
        Self
    }

    /// Fit CFA scores of \(X\), then OLS of \(y\) on the factor.
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLisrel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let mut cfa = ConfirmatoryFactor::new();
        let q = match cfa.fit(x, &session.child("cfa")) {
            Ok(q) => q,
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                        | IssueCode::MeaninglessFit
                        | IssueCode::CholeskyFailed
                ) {
                    ctx.push(e.primary);
                }
                return ctx.finish(FittedLisrel {
                    loadings: Vector::zeros(x.ncols()),
                    uniqueness: Vector::filled(x.ncols().max(1), 1.0),
                    mean: Vector::zeros(x.ncols()),
                    scale: Vector::filled(x.ncols().max(1), 1.0),
                    intercept: y.mean(),
                    slope: 0.0,
                    scores: Vector::zeros(x.nrows()),
                });
            }
        };
        for issue in q.report.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::MeaninglessFit
                    | IssueCode::CholeskyFailed
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let scores = match q.value.transform(x, &session.child("scores")) {
            Ok(z) => Vector::from_iter((0..z.value.nrows()).map(|i| z.value.get(i, 0))),
            Err(_) => Vector::zeros(x.nrows()),
        };
        let n = scores.len().min(y.len());
        let design = Matrix::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { scores[i] });
        let mut scratch = Report::new("lisrel", "ols");
        let coef = least_squares(&mut scratch, &design, y, &ctx.policy)
            .unwrap_or_else(|| Vector::from_slice(&[y.mean(), 0.0]));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::CholeskyFailed
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Lisrel is 1-factor scores plus OLS, not a published LISREL SEM")
                .compromise(NumericalCompromise::new(
                    "LISREL structural equation",
                    "MINRES factor scores of X, then OLS of y on [1, f]",
                    "a user path diagram, ML, and measurement-error correction are omitted",
                    "read the slope as a factor-score regression sketch",
                ))
                .build(),
        );
        ctx.finish(FittedLisrel {
            loadings: q.value.loadings.clone(),
            uniqueness: q.value.uniqueness.clone(),
            mean: q.value.mean.clone(),
            scale: q.value.scale.clone(),
            intercept: coef.as_slice().first().copied().unwrap_or(0.0),
            slope: coef.as_slice().get(1).copied().unwrap_or(0.0),
            scores,
        })
    }
}

/// Fitted LISREL-lite model.
#[derive(Clone, Debug)]
pub(crate) struct FittedLisrel {
    /// Measurement loadings.
    pub loadings: Vector,
    /// Uniqueness.
    pub uniqueness: Vector,
    /// Indicator means.
    pub mean: Vector,
    /// Indicator scales.
    pub scale: Vector,
    /// Structural intercept.
    pub intercept: f64,
    /// Structural slope on the factor.
    pub slope: f64,
    /// Training factor scores.
    pub scores: Vector,
}

impl FittedLisrel {
    /// Predict \(y\) from new indicators via the stored measurement model.
    pub(crate) fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let cfa = FittedConfirmatoryFactor {
            loadings: self.loadings.clone(),
            uniqueness: self.uniqueness.clone(),
            mean: self.mean.clone(),
            scale: self.scale.clone(),
            residual_ss: 0.0,
        };
        let z = match cfa.transform(x, &session.child("f")) {
            Ok(q) => q.value,
            Err(_) => Matrix::zeros(x.nrows(), 1),
        };
        ctx.finish(Vector::from_iter(
            (0..x.nrows()).map(|i| self.intercept + self.slope * z.get(i, 0)),
        ))
    }
}

fn latent_growth_row(x: &Matrix, i: usize) -> (f64, f64) {
    let p = x.ncols();
    if p < 2 {
        return (x.get(i, 0), 0.0);
    }
    let t_mean = 0.5 * (p as f64 - 1.0);
    let mut x_mean = 0.0_f64;
    for j in 0..p {
        x_mean += x.get(i, j);
    }
    x_mean /= p as f64;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for j in 0..p {
        let dt = j as f64 - t_mean;
        num += dt * (x.get(i, j) - x_mean);
        den += dt * dt;
    }
    let slope = if den > 1e-15 { num / den } else { 0.0 };
    (x_mean - slope * t_mean, slope)
}

/// Latent growth-curve intercepts and slopes (statsmodels SEM / LGM-lite).
///
/// Occasion count is not identification `p`. Distinct from
/// [`ConfirmatoryFactor`] (no time coding) and [`Lisrel`] (no per-row slope).
#[derive(Clone, Debug, Default)]
pub(crate) struct LatentGrowth;

impl LatentGrowth {
    /// Empty growth-curve fitter.
    pub(crate) fn new() -> Self {
        Self
    }

    /// Fit alias.
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedLatentGrowth>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted latent growth curves.
#[derive(Clone, Debug)]
pub(crate) struct FittedLatentGrowth {
    /// Per-row intercepts.
    pub intercepts: Vector,
    /// Per-row slopes on occasion \(t=0,\ldots,p-1\).
    pub slopes: Vector,
    /// Mean intercept.
    pub mean_intercept: f64,
    /// Mean slope.
    pub mean_slope: f64,
}

impl Transform for FittedLatentGrowth {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(Matrix::from_fn(x.nrows(), 2, |i, j| {
            let (a, b) = latent_growth_row(x, i);
            if j == 0 {
                a
            } else {
                b
            }
        }))
    }
}

impl FitUnsupervised for LatentGrowth {
    type Fitted = FittedLatentGrowth;
    fn fit_unsupervised(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedLatentGrowth>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        if p < 2 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message("LatentGrowth needs at least two occasion columns")
                    .build(),
            );
            return ctx.finish(FittedLatentGrowth {
                intercepts: Vector::zeros(n),
                slopes: Vector::zeros(n),
                mean_intercept: 0.0,
                mean_slope: 0.0,
            });
        }
        let intercepts = Vector::from_iter((0..n).map(|i| latent_growth_row(x, i).0));
        let slopes = Vector::from_iter((0..n).map(|i| latent_growth_row(x, i).1));
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("LatentGrowth is per-row OLS on [1, t], not a published LGM")
                .compromise(NumericalCompromise::new(
                    "statsmodels latent growth model",
                    "row-wise OLS of occasion columns on [1, t]",
                    "a random-effects covariance, ML, and time-varying loadings are omitted",
                    "read mean intercept/slope as a growth-curve sketch",
                ))
                .build(),
        );
        ctx.finish(FittedLatentGrowth {
            mean_intercept: intercepts.mean(),
            mean_slope: slopes.mean(),
            intercepts,
            slopes,
        })
    }
}

/// MIMIC: observed causes plus a latent scored from the remaining indicators.
///
/// Cause count is not identification `p`. Distinct from [`Lisrel`] (no
/// cause/indicator split) and [`ConfirmatoryFactor`] (no structural \(y\)).
#[derive(Clone, Debug)]
pub(crate) struct Mimic {
    /// Number of leading cause columns. Not identification `p`.
    pub n_causes: usize,
}

impl Default for Mimic {
    fn default() -> Self {
        Self { n_causes: 1 }
    }
}

impl Mimic {
    /// MIMIC with `n_causes` leading columns of \(X\).
    pub(crate) fn new(n_causes: usize) -> Self {
        Self {
            n_causes: n_causes.max(1),
        }
    }

    /// Score remaining columns of \(X\), then OLS of \(y\) on causes and the factor.
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMimic>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let nc = self.n_causes.max(1).min(x.ncols().saturating_sub(1).max(1));
        inspect_identification(&mut ctx.report, x.nrows(), nc, &ctx.policy);
        let n = x.nrows().min(y.len());
        let n_ind = x.ncols().saturating_sub(nc);
        if n_ind == 0 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message("Mimic needs at least one indicator column after the causes")
                    .build(),
            );
            return ctx.finish(FittedMimic {
                loadings: Vector::zeros(0),
                uniqueness: Vector::zeros(0),
                mean: Vector::zeros(0),
                scale: Vector::zeros(0),
                intercept: y.mean(),
                coef_causes: Vector::zeros(nc),
                slope_factor: 0.0,
                scores: Vector::zeros(n),
                n_causes: nc,
            });
        }
        let ind = Matrix::from_fn(n, n_ind, |i, j| x.get(i, nc + j));
        let mut cfa = ConfirmatoryFactor::new();
        let q = match cfa.fit(&ind, &session.child("mimic-cfa")) {
            Ok(q) => q,
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                        | IssueCode::MeaninglessFit
                        | IssueCode::CholeskyFailed
                ) {
                    ctx.push(e.primary);
                }
                return ctx.finish(FittedMimic {
                    loadings: Vector::zeros(n_ind),
                    uniqueness: Vector::filled(n_ind.max(1), 1.0),
                    mean: Vector::zeros(n_ind),
                    scale: Vector::filled(n_ind.max(1), 1.0),
                    intercept: y.mean(),
                    coef_causes: Vector::zeros(nc),
                    slope_factor: 0.0,
                    scores: Vector::zeros(n),
                    n_causes: nc,
                });
            }
        };
        for issue in q.report.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::MeaninglessFit
                    | IssueCode::CholeskyFailed
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let scores = match q.value.transform(&ind, &session.child("mimic-f")) {
            Ok(z) => Vector::from_iter((0..z.value.nrows()).map(|i| z.value.get(i, 0))),
            Err(_) => Vector::zeros(n),
        };
        let qols = 1 + nc + 1;
        let design = Matrix::from_fn(n, qols, |i, j| {
            if j == 0 {
                1.0
            } else if j <= nc {
                x.get(i, j - 1)
            } else {
                scores[i]
            }
        });
        let mut scratch = Report::new("mimic", "ols");
        let coef = least_squares(&mut scratch, &design, y, &ctx.policy)
            .unwrap_or_else(|| Vector::from_slice(&[y.mean(), 0.0]));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::CholeskyFailed
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Mimic is CFA scores plus OLS on causes, not a published MIMIC SEM")
                .compromise(NumericalCompromise::new(
                    "MIMIC structural equation",
                    "MINRES scores of the indicator block, then OLS of y on [1, causes, f]",
                    "a user path diagram, ML, and measurement-error correction are omitted",
                    "read coefficients as a MIMIC-score regression sketch",
                ))
                .build(),
        );
        ctx.finish(FittedMimic {
            loadings: q.value.loadings.clone(),
            uniqueness: q.value.uniqueness.clone(),
            mean: q.value.mean.clone(),
            scale: q.value.scale.clone(),
            intercept: coef.as_slice().first().copied().unwrap_or(0.0),
            coef_causes: Vector::from_iter(
                (0..nc).map(|j| coef.as_slice().get(1 + j).copied().unwrap_or(0.0)),
            ),
            slope_factor: coef.as_slice().get(1 + nc).copied().unwrap_or(0.0),
            scores,
            n_causes: nc,
        })
    }
}

/// Fitted MIMIC sketch.
#[derive(Clone, Debug)]
pub(crate) struct FittedMimic {
    /// Indicator loadings.
    pub loadings: Vector,
    /// Uniqueness.
    pub uniqueness: Vector,
    /// Indicator means.
    pub mean: Vector,
    /// Indicator scales.
    pub scale: Vector,
    /// Structural intercept.
    pub intercept: f64,
    /// Slopes on the cause columns.
    pub coef_causes: Vector,
    /// Slope on the latent score.
    pub slope_factor: f64,
    /// Training factor scores.
    pub scores: Vector,
    n_causes: usize,
}

impl FittedMimic {
    /// Predict \(y\) from new causes and indicators.
    pub(crate) fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let nc = self.n_causes;
        let n_ind = self.mean.len();
        let ind = Matrix::from_fn(x.nrows(), n_ind, |i, j| x.get(i, nc + j));
        let cfa = FittedConfirmatoryFactor {
            loadings: self.loadings.clone(),
            uniqueness: self.uniqueness.clone(),
            mean: self.mean.clone(),
            scale: self.scale.clone(),
            residual_ss: 0.0,
        };
        let z = match cfa.transform(&ind, &session.child("f")) {
            Ok(q) => q.value,
            Err(_) => Matrix::zeros(x.nrows(), 1),
        };
        ctx.finish(Vector::from_iter((0..x.nrows()).map(|i| {
            let mut s = self.intercept + self.slope_factor * z.get(i, 0);
            for j in 0..nc.min(self.coef_causes.len()) {
                s += x.get(i, j) * self.coef_causes[j];
            }
            s
        })))
    }
}

/// Observed-variable path analysis (statsmodels SEM path-lite).
///
/// Path count is not identification `p`. Distinct from [`Lisrel`] (latent
/// measurement) and [`Mimic`] (cause/indicator split).
#[derive(Clone, Debug, Default)]
pub(crate) struct PathAnalysis;

impl PathAnalysis {
    /// Empty path analysis.
    pub(crate) fn new() -> Self {
        Self
    }

    /// OLS of \(y\) on \(X\), plus each column of \(X\) on the others.
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPathAnalysis>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let n = x.nrows().min(y.len());
        let p = x.ncols();
        let design = Matrix::from_fn(n, 1 + p, |i, j| if j == 0 { 1.0 } else { x.get(i, j - 1) });
        let mut scratch = Report::new("path", "y");
        let coef = least_squares(&mut scratch, &design, y, &ctx.policy)
            .unwrap_or_else(|| Vector::from_slice(&[y.mean()]));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::CholeskyFailed
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let paths = Matrix::from_fn(p, p, |j, k| {
            if j == k {
                0.0
            } else {
                let z = Matrix::from_fn(n, p, |i, c| {
                    if c == 0 {
                        1.0
                    } else if c - 1 == j {
                        0.0
                    } else {
                        x.get(i, c - 1)
                    }
                });
                let yj = Vector::from_iter((0..n).map(|i| x.get(i, j)));
                let mut sc = Report::new("path", "x");
                least_squares(&mut sc, &z, &yj, &ctx.policy)
                    .and_then(|b| b.as_slice().get(if k < j { k + 1 } else { k }).copied())
                    .unwrap_or(0.0)
            }
        });
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("PathAnalysis is OLS paths among observed variables, not a published SEM")
                .compromise(NumericalCompromise::new(
                    "statsmodels path analysis / SEM",
                    "OLS of y on [1, X] and of each X_j on the other columns",
                    "ML, a user path diagram, and measurement error are omitted",
                    "read coefficients as an observed-path sketch",
                ))
                .build(),
        );
        ctx.finish(FittedPathAnalysis {
            intercept: coef.as_slice().first().copied().unwrap_or(0.0),
            coef_y: Vector::from_iter(
                (0..p).map(|j| coef.as_slice().get(1 + j).copied().unwrap_or(0.0)),
            ),
            paths,
        })
    }
}

/// Fitted observed-path model.
#[derive(Clone, Debug)]
pub(crate) struct FittedPathAnalysis {
    /// Structural intercept.
    pub intercept: f64,
    /// Structural slopes on \(X\).
    pub coef_y: Vector,
    /// Path matrix among the columns of \(X\) (zero diagonal).
    pub paths: Matrix,
}

impl FittedPathAnalysis {
    /// Structural prediction \(\hat y=a+X\beta\).
    pub(crate) fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(Vector::from_iter((0..x.nrows()).map(|i| {
            let mut s = self.intercept;
            for j in 0..x.ncols().min(self.coef_y.len()) {
                s += x.get(i, j) * self.coef_y[j];
            }
            s
        })))
    }
}

/// Two-equation feasible GLS SUR (statsmodels `SUR`).
///
/// Equation 1 is \(y\sim[1,X_{\setminus 0}]\); equation 2 is
/// \(X_0\sim[1,X_{\setminus 0,1}]\) so the regressor sets differ (identical
/// regressors would collapse to OLS). Equation count is not identification
/// `p`. Distinct from ordinary OLS and [`PathAnalysis`] (no GLS).
#[derive(Clone, Debug, Default)]
pub(crate) struct SeeminglyUnrelated;

impl SeeminglyUnrelated {
    /// Empty SUR.
    pub(crate) fn new() -> Self {
        Self
    }

    /// Fit two-equation FGLS SUR.
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSeeminglyUnrelated>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let n = x.nrows().min(y.len());
        let p = x.ncols();
        let q1 = 1 + p.saturating_sub(1);
        let z1 = Matrix::from_fn(n, q1, |i, j| if j == 0 { 1.0 } else { x.get(i, j) });
        let q2 = 1 + p.saturating_sub(2);
        let z2 = Matrix::from_fn(n, q2, |i, j| if j == 0 { 1.0 } else { x.get(i, j + 1) });
        let x0 = Vector::from_iter((0..n).map(|i| x.get(i, 0)));
        let mut sc1 = Report::new("sur", "eq1");
        let b1 = least_squares(&mut sc1, &z1, y, &ctx.policy).unwrap_or_else(|| Vector::zeros(q1));
        let mut sc2 = Report::new("sur", "eq2");
        let b2 =
            least_squares(&mut sc2, &z2, &x0, &ctx.policy).unwrap_or_else(|| Vector::zeros(q2));
        for issue in sc1.issues().iter().chain(sc2.issues()) {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::CholeskyFailed
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let e1 = y.sub(&z1.matvec(&b1));
        let e2 = x0.sub(&z2.matvec(&b2));
        let df = n.max(1) as f64;
        let mut s11 = 0.0_f64;
        let mut s22 = 0.0_f64;
        let mut s12 = 0.0_f64;
        for i in 0..n {
            s11 += e1[i] * e1[i];
            s22 += e2[i] * e2[i];
            s12 += e1[i] * e2[i];
        }
        s11 = (s11 / df).max(1e-10);
        s22 = (s22 / df).max(1e-10);
        s12 /= df;
        let det = (s11 * s22 - s12 * s12).max(1e-12);
        let w11 = s22 / det;
        let w22 = s11 / det;
        let w12 = -s12 / det;
        let q = q1 + q2;
        let mut gram = Mat::<f64>::zeros(q, q);
        for a in 0..q1 {
            for b in 0..q1 {
                let mut g = 0.0_f64;
                for i in 0..n {
                    g += z1.get(i, a) * z1.get(i, b);
                }
                gram[(a, b)] = w11 * g;
            }
            for b in 0..q2 {
                let mut g = 0.0_f64;
                for i in 0..n {
                    g += z1.get(i, a) * z2.get(i, b);
                }
                gram[(a, q1 + b)] = w12 * g;
                gram[(q1 + b, a)] = w12 * g;
            }
        }
        for a in 0..q2 {
            for b in 0..q2 {
                let mut g = 0.0_f64;
                for i in 0..n {
                    g += z2.get(i, a) * z2.get(i, b);
                }
                gram[(q1 + a, q1 + b)] = w22 * g;
            }
        }
        for i in 0..q {
            gram[(i, i)] += 1e-10;
        }
        let mut rhs = Vector::zeros(q);
        for a in 0..q1 {
            let mut s = 0.0_f64;
            for i in 0..n {
                s += z1.get(i, a) * (w11 * y[i] + w12 * x0[i]);
            }
            rhs[a] = s;
        }
        for a in 0..q2 {
            let mut s = 0.0_f64;
            for i in 0..n {
                s += z2.get(i, a) * (w12 * y[i] + w22 * x0[i]);
            }
            rhs[q1 + a] = s;
        }
        let mut scg = Report::new("sur", "gls");
        let beta = chol_solve(&mut scg, &gram, &rhs, &ctx.policy).unwrap_or_else(|| {
            let mut v = Vector::zeros(q);
            for j in 0..q1.min(b1.len()) {
                v[j] = b1[j];
            }
            for j in 0..q2.min(b2.len()) {
                v[q1 + j] = b2[j];
            }
            v
        });
        for issue in scg.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::CholeskyFailed
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("SeeminglyUnrelated is two-equation FGLS, not a published SUR system")
                .compromise(NumericalCompromise::new(
                    "statsmodels SUR",
                    "OLS residuals form Σ, then one FGLS step on two stacked equations",
                    "iterated GLS, equation-specific instruments, and a full system ML are omitted",
                    "read the first-equation slopes as a two-equation SUR sketch",
                ))
                .build(),
        );
        ctx.finish(FittedSeeminglyUnrelated {
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            coef_y: Vector::from_iter(
                (1..q1).map(|j| beta.as_slice().get(j).copied().unwrap_or(0.0)),
            ),
            intercept_eq2: beta.as_slice().get(q1).copied().unwrap_or(0.0),
            coef_eq2: Vector::from_iter(
                (1..q2).map(|j| beta.as_slice().get(q1 + j).copied().unwrap_or(0.0)),
            ),
            sigma11: s11,
            sigma12: s12,
            sigma22: s22,
        })
    }
}

/// Fitted two-equation SUR.
#[derive(Clone, Debug)]
pub(crate) struct FittedSeeminglyUnrelated {
    /// Equation-1 intercept.
    pub intercept: f64,
    /// Equation-1 slopes on \(X_{\setminus 0}\).
    pub coef_y: Vector,
    /// Equation-2 intercept.
    pub intercept_eq2: f64,
    /// Equation-2 slopes on \(X_{\setminus 0,1}\).
    pub coef_eq2: Vector,
    /// Residual variance of equation 1.
    pub sigma11: f64,
    /// Residual covariance.
    pub sigma12: f64,
    /// Residual variance of equation 2.
    pub sigma22: f64,
}

impl FittedSeeminglyUnrelated {
    /// Predict equation 1.
    pub(crate) fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(Vector::from_iter((0..x.nrows()).map(|i| {
            let mut s = self.intercept;
            for j in 0..self.coef_y.len() {
                if 1 + j < x.ncols() {
                    s += x.get(i, 1 + j) * self.coef_y[j];
                }
            }
            s
        })))
    }
}

/// Random-effects growth curve (statsmodels LGM second moments).
///
/// Occasion count is not identification `p`. Distinct from [`LatentGrowth`]
/// (no intercept/slope covariance) and [`ConfirmatoryFactor`] (no time coding).
#[derive(Clone, Debug, Default)]
pub(crate) struct GrowthCurve;

impl GrowthCurve {
    /// Empty random-effects growth curve.
    pub(crate) fn new() -> Self {
        Self
    }

    /// Fit alias.
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGrowthCurve>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted random-effects growth curve.
#[derive(Clone, Debug)]
pub(crate) struct FittedGrowthCurve {
    /// Per-row intercepts.
    pub intercepts: Vector,
    /// Per-row slopes.
    pub slopes: Vector,
    /// Mean intercept.
    pub mean_intercept: f64,
    /// Mean slope.
    pub mean_slope: f64,
    /// Intercept variance.
    pub var_intercept: f64,
    /// Slope variance.
    pub var_slope: f64,
    /// Intercept–slope covariance.
    pub cov_is: f64,
}

impl FitUnsupervised for GrowthCurve {
    type Fitted = FittedGrowthCurve;
    fn fit_unsupervised(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGrowthCurve>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        if p < 2 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message("GrowthCurve needs at least two occasion columns")
                    .build(),
            );
            return ctx.finish(FittedGrowthCurve {
                intercepts: Vector::zeros(n),
                slopes: Vector::zeros(n),
                mean_intercept: 0.0,
                mean_slope: 0.0,
                var_intercept: 0.0,
                var_slope: 0.0,
                cov_is: 0.0,
            });
        }
        let intercepts = Vector::from_iter((0..n).map(|i| latent_growth_row(x, i).0));
        let slopes = Vector::from_iter((0..n).map(|i| latent_growth_row(x, i).1));
        let mi = intercepts.mean();
        let ms = slopes.mean();
        let df = (n.saturating_sub(1)).max(1) as f64;
        let mut vi = 0.0_f64;
        let mut vs = 0.0_f64;
        let mut cv = 0.0_f64;
        for i in 0..n {
            let di = intercepts[i] - mi;
            let ds = slopes[i] - ms;
            vi += di * di;
            vs += ds * ds;
            cv += di * ds;
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("GrowthCurve is sample second moments of per-row OLS, not a published LGM")
                .compromise(NumericalCompromise::new(
                    "statsmodels latent growth model",
                    "row-wise [1, t] OLS, then the intercept/slope covariance",
                    "ML random-effects, time-varying loadings, and a structured Ψ are omitted",
                    "read the covariance as a growth-curve moment sketch",
                ))
                .build(),
        );
        ctx.finish(FittedGrowthCurve {
            intercepts,
            slopes,
            mean_intercept: mi,
            mean_slope: ms,
            var_intercept: vi / df,
            var_slope: vs / df,
            cov_is: cv / df,
        })
    }
}

/// Observed-plus-latent structural equation (LISREL-lite with \(X\) slopes).
///
/// Factor count is not identification `p`. Distinct from [`Lisrel`] (no
/// observed \(X\) in the structural equation) and [`Mimic`] (cause/indicator
/// split).
#[derive(Clone, Debug, Default)]
pub(crate) struct StructuralEquation;

impl StructuralEquation {
    /// Empty SEM.
    pub(crate) fn new() -> Self {
        Self
    }

    /// CFA scores of \(X\), then OLS of \(y\) on \([1,X,f]\).
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedStructuralEquation>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let n = x.nrows().min(y.len());
        let p = x.ncols();
        let mut cfa = ConfirmatoryFactor::new();
        let q = match cfa.fit(x, &session.child("sem-cfa")) {
            Ok(q) => q,
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                        | IssueCode::MeaninglessFit
                        | IssueCode::CholeskyFailed
                ) {
                    ctx.push(e.primary);
                }
                return ctx.finish(FittedStructuralEquation {
                    intercept: y.mean(),
                    coef_x: Vector::zeros(p),
                    slope_factor: 0.0,
                    scores: Vector::zeros(n),
                });
            }
        };
        for issue in q.report.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::MeaninglessFit
                    | IssueCode::CholeskyFailed
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let scores = match q.value.transform(x, &session.child("sem-f")) {
            Ok(z) => Vector::from_iter((0..z.value.nrows()).map(|i| z.value.get(i, 0))),
            Err(_) => Vector::zeros(n),
        };
        let qols = 2 + p;
        let design = Matrix::from_fn(n, qols, |i, j| {
            if j == 0 {
                1.0
            } else if j <= p {
                x.get(i, j - 1)
            } else {
                scores[i]
            }
        });
        let mut scratch = Report::new("sem", "ols");
        let coef = least_squares(&mut scratch, &design, y, &ctx.policy)
            .unwrap_or_else(|| Vector::from_slice(&[y.mean()]));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::CholeskyFailed
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message(
                    "StructuralEquation is CFA scores plus OLS on [1, X, f], not published LISREL",
                )
                .compromise(NumericalCompromise::new(
                    "LISREL / SEM",
                    "MINRES scores of X, then OLS of y on [1, X, f]",
                    "a user path diagram, ML, and measurement-error correction are omitted",
                    "read coefficients as an observed-plus-latent SEM sketch",
                ))
                .build(),
        );
        ctx.finish(FittedStructuralEquation {
            intercept: coef.as_slice().first().copied().unwrap_or(0.0),
            coef_x: Vector::from_iter(
                (0..p).map(|j| coef.as_slice().get(1 + j).copied().unwrap_or(0.0)),
            ),
            slope_factor: coef.as_slice().get(1 + p).copied().unwrap_or(0.0),
            scores,
        })
    }
}

/// Fitted observed-plus-latent SEM.
#[derive(Clone, Debug)]
pub(crate) struct FittedStructuralEquation {
    /// Structural intercept.
    pub intercept: f64,
    /// Slopes on the observed columns of \(X\).
    pub coef_x: Vector,
    /// Slope on the latent score.
    pub slope_factor: f64,
    /// Training factor scores.
    pub scores: Vector,
}

impl FittedStructuralEquation {
    /// Predict \(y\) from new \(X\) using stored CFA scores of the new rows.
    pub(crate) fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(Vector::from_iter((0..x.nrows()).map(|i| {
            let mut s = self.intercept;
            for j in 0..x.ncols().min(self.coef_x.len()) {
                s += x.get(i, j) * self.coef_x[j];
            }
            let f = if i < self.scores.len() {
                self.scores[i]
            } else {
                0.0
            };
            s + self.slope_factor * f
        })))
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
        let q = Pca::new(2).fit(&x, &session).expect("pca");
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
        let q = Pca::new(6)
            .fit(&x, &session)
            .expect("pca should warn, not abort");
        assert!(
            q.report.contains(IssueCode::ComponentsExceedRank),
            "expected ComponentsExceedRank, issues={:?}",
            q.report.issues().iter().map(|i| i.code).collect::<Vec<_>>()
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

    #[test]
    fn minibatch_nmf_reconstructs_nonneg() {
        let x = rank1();
        let session = Session::new("mbnmf", "fit");
        let q = MiniBatchNmf {
            n_components: 1,
            batch_size: 10,
            seed: 2,
            ..MiniBatchNmf::default()
        }
        .fit_unsupervised(&x, &session)
        .expect("mbnmf");
        assert!(q.value.reconstruction_err.is_finite());
        assert!(q.value.h.nrows() >= 1);
        let mut mb = MiniBatchNmf::new(1);
        let mut mbsp = MiniBatchSparsePca::new(1);
        let qe = mbsp
            .partial_fit(&x, None, &Session::new("mbsp", "pf"))
            .expect("mbsp");
        assert!(!qe.value.narrative.is_empty());
        mb.partial_fit(&x, None, &Session::new("mbnmf", "pf"))
            .expect("pf");
        assert!(mb.h().is_some());
        let mut mbdl = MiniBatchDictionaryLearning::new(2);
        let qdl = mbdl
            .partial_fit(&x, None, &Session::new("mbdl", "pf"))
            .expect("mbdl");
        assert!(!qdl.value.narrative.is_empty());
        let dict = mbdl.dictionary().cloned().unwrap();
        let sc = SparseCoder::new(dict)
            .transform(&x, &Session::new("sc", "t"))
            .expect("sc")
            .value;
        assert_eq!(sc.nrows(), x.nrows());
        assert!(sc.get(0, 0).is_finite());
        let ycca = Vector::from_iter((0..x.nrows()).map(|i| x.get(i, 0) + 0.1 * (i as f64)));
        let kcca = KernelCca::new(0.05)
            .fit(&x, &ycca, &Session::new("kcca", "fit"))
            .expect("kcca");
        assert!(kcca.value.correlation.is_finite());
        let zk = kcca
            .value
            .transform(&x, &Session::new("kcca", "t"))
            .expect("kccat")
            .value;
        assert_eq!(zk.ncols(), 1);
        assert!(zk.get(0, 0).is_finite());
        let cfa = ConfirmatoryFactor::new()
            .fit(&x, &Session::new("cfa", "fit"))
            .expect("cfa");
        assert_eq!(cfa.value.loadings.len(), x.ncols());
        assert!(cfa.value.residual_ss.is_finite());
        let zf = cfa
            .value
            .transform(&x, &Session::new("cfa", "t"))
            .expect("cfat")
            .value;
        assert_eq!(zf.ncols(), 1);
        assert!(zf.get(0, 0).is_finite());
        let lis = Lisrel::new()
            .fit(&x, &ycca, &Session::new("lis", "fit"))
            .expect("lis");
        assert!(lis.value.slope.is_finite());
        let lisp = lis
            .value
            .predict(&x, &Session::new("lis", "p"))
            .expect("lisp")
            .value;
        assert_eq!(lisp.len(), x.nrows());
        assert!(lisp.as_slice().iter().all(|v| v.is_finite()));
        let lgw = LatentGrowth::new()
            .fit(&x, &Session::new("lgw", "fit"))
            .expect("lgw");
        assert!(lgw.value.mean_slope.is_finite());
        assert_eq!(lgw.value.intercepts.len(), x.nrows());
        let lgwt = lgw
            .value
            .transform(&x, &Session::new("lgw", "t"))
            .expect("lgwt")
            .value;
        assert_eq!(lgwt.shape(), (x.nrows(), 2));
        let mmc = Mimic::new(1)
            .fit(&x, &ycca, &Session::new("mmc", "fit"))
            .expect("mmc");
        assert!(mmc.value.slope_factor.is_finite());
        let mmcp = mmc
            .value
            .predict(&x, &Session::new("mmc", "p"))
            .expect("mmcp")
            .value;
        assert_eq!(mmcp.len(), x.nrows());
        assert!(mmcp.as_slice().iter().all(|v| v.is_finite()));
        let pah = PathAnalysis::new()
            .fit(&x, &ycca, &Session::new("pah", "fit"))
            .expect("pah");
        assert_eq!(pah.value.coef_y.len(), x.ncols());
        let pahp = pah
            .value
            .predict(&x, &Session::new("pah", "p"))
            .expect("pahp")
            .value;
        assert_eq!(pahp.len(), x.nrows());
        let sur = SeeminglyUnrelated::new()
            .fit(&x, &ycca, &Session::new("sur", "fit"))
            .expect("sur");
        assert!(sur.value.sigma11.is_finite() && sur.value.sigma11 > 0.0);
        let surp = sur
            .value
            .predict(&x, &Session::new("sur", "p"))
            .expect("surp")
            .value;
        assert_eq!(surp.len(), x.nrows());
        assert!(surp.as_slice().iter().all(|v| v.is_finite()));
        let gcw = GrowthCurve::new()
            .fit(&x, &Session::new("gcw", "fit"))
            .expect("gcw");
        assert!(gcw.value.var_intercept.is_finite());
        assert_eq!(gcw.value.intercepts.len(), x.nrows());
        let seq = StructuralEquation::new()
            .fit(&x, &ycca, &Session::new("seq", "fit"))
            .expect("seq");
        let seqp = seq
            .value
            .predict(&x, &Session::new("seq", "p"))
            .expect("seqp")
            .value;
        assert_eq!(seqp.len(), x.nrows());
        assert!(seqp.as_slice().iter().all(|v| v.is_finite()));
    }
}
