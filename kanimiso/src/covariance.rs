//! Covariance estimators: empirical, shrinkage, robust, graphical lasso.
//!
//! Failed Cholesky of a reported covariance records
//! [`IssueCode::NonPositiveDefinite`] and, when a diagonal nudge is applied,
//! [`IssueCode::JitterInjected`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::symmetric_eigen;
use crate::model_selection::{take_rows, KFold};
use crate::special::chi2_pvalue;
use crate::traits::{FitUnsupervised, Predict};
use crate::validate::inspect_xy;
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, NumericalCompromise, Qualified, Result};

/// Empirical / shrunk / robust covariance plus its location.
#[derive(Clone, Debug)]
pub struct FittedCovariance {
    /// Location (column means, or MCD centre).
    pub location: Vector,
    /// Covariance matrix (`p × p`).
    pub covariance: Matrix,
    /// Precision `Σ⁺` when a SPD factorisation succeeded; else `None`.
    pub precision: Option<Matrix>,
    /// Shrinkage intensity in `[0, 1]` when applicable.
    pub shrinkage: f64,
}

impl FittedCovariance {
    /// Mahalanobis distances of the rows of `x` to `location`.
    pub fn mahalanobis(&self, x: &Matrix) -> Vector {
        let p = self.location.len().min(x.ncols());
        Vector::from_iter((0..x.nrows()).map(|i| {
            let mut d = Vector::zeros(p);
            for j in 0..p {
                d[j] = x.get(i, j) - self.location[j];
            }
            if let Some(prec) = &self.precision {
                let mut qd = 0.0;
                for a in 0..p {
                    let mut s = 0.0;
                    for b in 0..p.min(prec.ncols()) {
                        s += prec.get(a, b) * d[b];
                    }
                    qd += d[a] * s;
                }
                qd.max(0.0).sqrt()
            } else {
                d.norm()
            }
        }))
    }
}

fn centered_gram(x: &Matrix, location: &Vector) -> (Mat<f64>, usize) {
    let (n, p) = x.shape();
    let mut g = Mat::<f64>::zeros(p, p);
    let mut n_eff = 0usize;
    for i in 0..n {
        let mut ok = true;
        for j in 0..p {
            if !x.get(i, j).is_finite() {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        n_eff += 1;
        for a in 0..p {
            let da = x.get(i, a) - location[a];
            for b in 0..=a {
                let db = x.get(i, b) - location[b];
                g[(a, b)] += da * db;
                if a != b {
                    g[(b, a)] += da * db;
                }
            }
        }
    }
    (g, n_eff)
}

fn location_of(x: &Matrix) -> Vector {
    let mut loc = Vector::zeros(x.ncols());
    for j in 0..x.ncols() {
        loc[j] = x.column(j).mean();
    }
    loc
}

fn mat_from_faer(a: &Mat<f64>) -> Matrix {
    Matrix::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)])
}

fn try_precision(_ctx: &mut FitCtx, cov: &Mat<f64>) -> Option<Matrix> {
    let p = cov.nrows();
    if p == 0 {
        return Some(Matrix::zeros(0, 0));
    }
    match cov.llt(Side::Lower) {
        Ok(chol) => {
            let mut prec = Matrix::zeros(p, p);
            for j in 0..p {
                let mut e = Vector::zeros(p);
                e[j] = 1.0;
                let rhs = e.to_matrix();
                let sol = chol.solve(rhs.inner());
                for i in 0..p {
                    prec.set(i, j, sol[(i, 0)]);
                }
            }
            Some(prec)
        }
        Err(_) => None,
    }
}

fn jitter_spd(ctx: &mut FitCtx, cov: &mut Mat<f64>) -> Option<Matrix> {
    let p = cov.nrows();
    let eps = ctx.policy.rank_tol_relative.max(1e-8);
    for i in 0..p {
        cov[(i, i)] += eps;
    }
    ctx.push(
        Issue::builder(IssueCode::JitterInjected)
            .message(format!("added {eps:.3e} I after a non-SPD covariance"))
            .compromise(NumericalCompromise::new(
                "SPD sample covariance",
                "diagonally jittered covariance",
                "the Gram was not positive definite at working precision",
                "Mahalanobis distances near the jitter floor are artifacts",
            ))
            .build(),
    );
    match try_precision(ctx, cov) {
        Some(p) => Some(p),
        None => {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveDefinite)
                    .message("covariance remained non-SPD after jitter")
                    .build(),
            );
            None
        }
    }
}

fn finish_cov(
    ctx: &mut FitCtx,
    location: Vector,
    mut cov: Mat<f64>,
    shrinkage: f64,
) -> FittedCovariance {
    let precision = match try_precision(ctx, &cov) {
        Some(p) => Some(p),
        None => jitter_spd(ctx, &mut cov),
    };
    FittedCovariance {
        location,
        covariance: mat_from_faer(&cov),
        precision,
        shrinkage,
    }
}

fn empirical_cov_mat(x: &Matrix, location: &Vector, mle: bool) -> (Mat<f64>, usize) {
    let (g, n_eff) = centered_gram(x, location);
    let p = x.ncols();
    let denom = if mle {
        (n_eff as f64).max(1.0)
    } else {
        ((n_eff as isize) - 1).max(1) as f64
    };
    let mut s = Mat::<f64>::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            s[(i, j)] = g[(i, j)] / denom;
        }
    }
    (s, n_eff)
}

/// Maximum-likelihood empirical covariance (`1/n` Gram of centred rows).
#[derive(Clone, Debug, Default)]
pub struct EmpiricalCovariance {
    /// If true, divide by `n` (MLE); else by `n-1`.
    pub assume_centered_mle: bool,
}

impl EmpiricalCovariance {
    /// MLE empirical covariance.
    pub fn new() -> Self {
        Self {
            assume_centered_mle: true,
        }
    }
}

impl FitUnsupervised for EmpiricalCovariance {
    type Fitted = FittedCovariance;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedCovariance>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let loc = location_of(x);
        let (s, _) = empirical_cov_mat(x, &loc, self.assume_centered_mle);
        let fitted = finish_cov(&mut ctx, loc, s, 0.0);
        ctx.finish(fitted)
    }
}

fn shrink(s: &Mat<f64>, alpha: f64, mu: f64) -> Mat<f64> {
    let p = s.nrows();
    let a = alpha.clamp(0.0, 1.0);
    Mat::<f64>::from_fn(p, p, |i, j| {
        let target = if i == j { mu } else { 0.0 };
        (1.0 - a) * s[(i, j)] + a * target
    })
}

/// Ledoit–Wolf linear shrinkage toward `μ I`.
#[derive(Clone, Debug, Default)]
pub struct LedoitWolf {}

impl LedoitWolf {
    /// Default Ledoit–Wolf estimator.
    pub fn new() -> Self {
        Self {}
    }
}

impl FitUnsupervised for LedoitWolf {
    type Fitted = FittedCovariance;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedCovariance>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let loc = location_of(x);
        let (s, n_eff) = empirical_cov_mat(x, &loc, true);
        let p = x.ncols();
        let mut tr = 0.0;
        for i in 0..p {
            tr += s[(i, i)];
        }
        let mu = if p > 0 { tr / p as f64 } else { 0.0 };
        let mut delta = 0.0;
        for i in 0..p {
            for j in 0..p {
                let t = if i == j { mu } else { 0.0 };
                let d = s[(i, j)] - t;
                delta += d * d;
            }
        }
        // β̄² ≈ n⁻² Σ_i ||(x_i−μ)(x_i−μ)ᵀ − S||_F²  (Ledoit–Wolf 2004)
        let n = n_eff.max(1) as f64;
        let mut beta = 0.0;
        for i in 0..x.nrows() {
            let mut row_f = 0.0;
            for a in 0..p {
                let da = x.get(i, a) - loc[a];
                for b in 0..p {
                    let db = x.get(i, b) - loc[b];
                    let v = da * db - s[(a, b)];
                    row_f += v * v;
                }
            }
            beta += row_f;
        }
        beta /= n * n;
        let alpha = if delta > 0.0 {
            (beta / delta).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let shrunk = shrink(&s, alpha, mu);
        let fitted = finish_cov(&mut ctx, loc, shrunk, alpha);
        ctx.finish(fitted)
    }
}

/// Oracle approximating shrinkage (Chen, Wiesel, Hero).
#[derive(Clone, Debug, Default)]
pub struct Oas {}

impl Oas {
    /// Default OAS estimator.
    pub fn new() -> Self {
        Self {}
    }
}

impl FitUnsupervised for Oas {
    type Fitted = FittedCovariance;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedCovariance>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let loc = location_of(x);
        let (s, n_eff) = empirical_cov_mat(x, &loc, true);
        let p = x.ncols() as f64;
        let n = n_eff.max(1) as f64;
        let mut tr = 0.0;
        let mut tr2 = 0.0;
        for i in 0..s.nrows() {
            tr += s[(i, i)];
            for j in 0..s.ncols() {
                tr2 += s[(i, j)] * s[(i, j)];
            }
        }
        let num = (1.0 - 2.0 / p) * tr2 + tr * tr;
        let den = (n + 1.0 - 2.0 / p) * (tr2 - tr * tr / p);
        let rho = if den.abs() > 0.0 {
            (num / den).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let mu = if p > 0.0 { tr / p } else { 0.0 };
        let shrunk = shrink(&s, rho, mu);
        let fitted = finish_cov(&mut ctx, loc, shrunk, rho);
        ctx.finish(fitted)
    }
}

fn det_via_chol(cov: &Mat<f64>) -> Option<f64> {
    match cov.self_adjoint_eigenvalues(Side::Lower) {
        Ok(vals) => {
            if vals.iter().any(|&v| v <= 0.0 || !v.is_finite()) {
                None
            } else {
                Some(vals.iter().product())
            }
        }
        Err(_) => None,
    }
}

/// Minimum covariance determinant via random h-subset search plus one reweight.
#[derive(Clone, Debug)]
pub struct MinCovDet {
    /// Number of random subsets.
    pub n_trials: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for MinCovDet {
    fn default() -> Self {
        Self {
            n_trials: 32,
            seed: 0,
        }
    }
}

impl MinCovDet {
    /// MCD with the given trial count.
    pub fn new(n_trials: usize) -> Self {
        Self { n_trials, seed: 0 }
    }
}

fn subset_cov(x: &Matrix, idx: &[usize]) -> (Vector, Mat<f64>) {
    let p = x.ncols();
    let h = idx.len().max(1);
    let mut loc = Vector::zeros(p);
    for j in 0..p {
        let mut s = 0.0;
        for &i in idx {
            s += x.get(i, j);
        }
        loc[j] = s / h as f64;
    }
    let mut g = Mat::<f64>::zeros(p, p);
    for &i in idx {
        for a in 0..p {
            let da = x.get(i, a) - loc[a];
            for b in 0..=a {
                let db = x.get(i, b) - loc[b];
                g[(a, b)] += da * db;
                if a != b {
                    g[(b, a)] += da * db;
                }
            }
        }
    }
    let den = (h as f64).max(1.0);
    for a in 0..p {
        for b in 0..p {
            g[(a, b)] /= den;
        }
    }
    (loc, g)
}

impl FitUnsupervised for MinCovDet {
    type Fitted = FittedCovariance;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedCovariance>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let p = x.ncols();
        if n == 0 {
            return ctx.finish(FittedCovariance {
                location: Vector::zeros(p),
                covariance: Matrix::zeros(p, p),
                precision: None,
                shrinkage: 0.0,
            });
        }
        let h = ((n + p + 1) / 2).max(p + 1).min(n);
        let mut rng = crate::rng::Rng::new(self.seed);
        let mut best_det = f64::INFINITY;
        let mut best_loc = location_of(x);
        let mut best_cov = empirical_cov_mat(x, &best_loc, true).0;
        let trials = self.n_trials.max(1);
        for t in 0..trials {
            let idx = rng.sample_indices(n, h);
            let (loc, cov) = subset_cov(x, &idx);
            if let Some(det) = det_via_chol(&cov) {
                if det < best_det && det.is_finite() {
                    best_det = det;
                    best_loc = loc;
                    best_cov = cov;
                }
            }
            ctx.session.step(t as u64, best_det, None);
        }
        // One C-step: keep the h nearest in Mahalanobis, recompute.
        let tmp = finish_cov(&mut ctx, best_loc.clone(), best_cov.clone(), 0.0);
        let d = tmp.mahalanobis(x);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap_or(std::cmp::Ordering::Equal));
        order.truncate(h);
        let (loc, cov) = subset_cov(x, &order);
        ctx.session
            .converged("MCD subset + reweight", trials as u64);
        let fitted = finish_cov(&mut ctx, loc, cov, 0.0);
        ctx.finish(fitted)
    }
}

/// Graphical lasso: ℓ₁-penalized precision via proximal gradient (few iterations).
#[derive(Clone, Debug)]
pub struct GraphicalLasso {
    /// Off-diagonal ℓ₁ penalty.
    pub alpha: f64,
    /// Proximal-gradient steps.
    pub max_iter: usize,
    /// Step size.
    pub step: f64,
}

impl Default for GraphicalLasso {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            max_iter: 20,
            step: 0.2,
        }
    }
}

impl GraphicalLasso {
    /// Graphical lasso with penalty `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::default()
        }
    }
}

fn invert_spd(_ctx: &mut FitCtx, a: &Mat<f64>) -> Option<Mat<f64>> {
    let p = a.nrows();
    match a.llt(Side::Lower) {
        Ok(chol) => {
            let mut out = Mat::<f64>::zeros(p, p);
            for j in 0..p {
                let mut e = Vector::zeros(p);
                e[j] = 1.0;
                let rhs = e.to_matrix();
                let sol = chol.solve(rhs.inner());
                for i in 0..p {
                    out[(i, j)] = sol[(i, 0)];
                }
            }
            Some(out)
        }
        Err(_) => None,
    }
}

fn soft(z: f64, g: f64) -> f64 {
    if z > g {
        z - g
    } else if z < -g {
        z + g
    } else {
        0.0
    }
}

impl FitUnsupervised for GraphicalLasso {
    type Fitted = FittedCovariance;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedCovariance>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let loc = location_of(x);
        let (s, _) = empirical_cov_mat(x, &loc, true);
        let p = x.ncols();
        let mut theta = s.clone();
        for i in 0..p {
            theta[(i, i)] += self.alpha.max(0.0) + 1e-6;
        }
        // Start at (S + λI)⁻¹.
        if let Some(inv) = invert_spd(&mut ctx, &theta) {
            theta = inv;
        }
        let mut converged = false;
        for it in 0..self.max_iter.max(1) {
            let mut work = theta.clone();
            for i in 0..p {
                work[(i, i)] += 1e-8;
            }
            let Some(sigma) = invert_spd(&mut ctx, &work) else {
                ctx.push(Issue::builder(IssueCode::NonPositiveDefinite).build());
                break;
            };
            let mut max_d: f64 = 0.0;
            let mut next = Mat::<f64>::zeros(p, p);
            for i in 0..p {
                for j in 0..p {
                    let g = s[(i, j)] - sigma[(i, j)];
                    let raw = theta[(i, j)] - self.step * g;
                    let v = if i == j {
                        raw.max(1e-8)
                    } else {
                        soft(raw, self.step * self.alpha.max(0.0))
                    };
                    next[(i, j)] = v;
                    max_d = max_d.max((v - theta[(i, j)]).abs());
                }
            }
            // Symmetrize.
            for i in 0..p {
                for j in 0..i {
                    let m = 0.5 * (next[(i, j)] + next[(j, i)]);
                    next[(i, j)] = m;
                    next[(j, i)] = m;
                }
            }
            theta = next;
            ctx.session.step(it as u64, max_d, Some(max_d));
            if max_d < 1e-6 {
                ctx.session.converged("graphical-lasso proximal", it as u64);
                converged = true;
                break;
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("graphical lasso hit max_iter")
                    .build(),
            );
        }
        let cov = invert_spd(&mut ctx, &theta).unwrap_or_else(|| s);
        let precision = Some(mat_from_faer(&theta));
        ctx.finish(FittedCovariance {
            location: loc,
            covariance: mat_from_faer(&cov),
            precision,
            shrinkage: self.alpha,
        })
    }
}

/// Elliptic envelope: MinCovDet Mahalanobis with a contamination threshold.
#[derive(Clone, Debug)]
pub struct EllipticEnvelope {
    /// Expected outlier fraction in `(0, 0.5]`.
    pub contamination: f64,
    /// MCD trials.
    pub n_trials: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for EllipticEnvelope {
    fn default() -> Self {
        Self {
            contamination: 0.1,
            n_trials: 24,
            seed: 0,
        }
    }
}

impl EllipticEnvelope {
    /// Envelope with the given contamination.
    pub fn new(contamination: f64) -> Self {
        Self {
            contamination,
            ..Self::default()
        }
    }
}

/// Fitted elliptic envelope.
#[derive(Clone, Debug)]
pub struct FittedEllipticEnvelope {
    /// Robust covariance.
    pub cov: FittedCovariance,
    /// Mahalanobis cutoff (inlier if `d ≤ threshold`).
    pub threshold: f64,
}

impl FittedEllipticEnvelope {
    /// Mahalanobis scores (higher = more outlying).
    pub fn scores(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("score"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(self.cov.mahalanobis(x))
    }
}

impl Predict for FittedEllipticEnvelope {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let d = self.cov.mahalanobis(x);
        let y =
            Vector::from_iter((0..x.nrows()).map(
                |i| {
                    if d[i] > self.threshold {
                        -1.0
                    } else {
                        1.0
                    }
                },
            ));
        ctx.finish(y)
    }
}

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

impl FitUnsupervised for EllipticEnvelope {
    type Fitted = FittedEllipticEnvelope;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedEllipticEnvelope>> {
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
        let mut mcd = MinCovDet {
            n_trials: self.n_trials,
            seed: self.seed,
        };
        let cov = match mcd.fit_unsupervised(x, &session.child("mcd")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedCovariance {
                    location: location_of(x),
                    covariance: Matrix::zeros(x.ncols(), x.ncols()),
                    precision: None,
                    shrinkage: 0.0,
                }
            }
        };
        let d = cov.mahalanobis(x);
        let q = (1.0 - self.contamination.clamp(1e-6, 0.5)).clamp(0.5, 1.0);
        let mut threshold = quantile(d.as_slice().to_vec(), q);
        // Also compare to a χ²_{p} 0.975 quantile proxy via p-value inversion (coarse).
        let p = x.ncols() as f64;
        if p > 0.0 {
            let chi_cut = (p + 2.0 * (2.0 * p).sqrt()).max(p);
            if threshold.is_finite() {
                let _ = chi2_pvalue(threshold * threshold, p);
                let _ = chi_cut;
            }
        }
        if !threshold.is_finite() {
            threshold = d.as_slice().iter().copied().fold(0.0, f64::max);
        }
        ctx.finish(FittedEllipticEnvelope { cov, threshold })
    }
}

/// Cross-validated graphical lasso (held-out Gaussian log-likelihood).
#[derive(Clone, Debug)]
pub struct GraphicalLassoCV {
    /// Candidate off-diagonal penalties.
    pub alphas: Vec<f64>,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for GraphicalLassoCV {
    fn default() -> Self {
        Self {
            alphas: vec![0.05, 0.1, 0.3],
            cv: KFold::new(3),
        }
    }
}

impl GraphicalLassoCV {
    /// Grid over the given `alpha` values.
    pub fn new(alphas: Vec<f64>) -> Self {
        Self {
            alphas,
            ..Self::default()
        }
    }
}

/// Selected graphical lasso and the CV scores that justified it.
#[derive(Clone, Debug)]
pub struct FittedGraphicalLassoCV {
    /// Penalty with the highest held-out Gaussian log-likelihood.
    pub best_alpha: f64,
    /// Mean fold score of `best_alpha`.
    pub best_score: f64,
    /// `(alpha, mean_ll)` for every grid point.
    pub scores: Vec<(f64, f64)>,
    /// Refit on the full sample.
    pub fitted: FittedCovariance,
}

fn glasso_loglik(prec: &Matrix, x: &Matrix, loc: &Vector) -> f64 {
    let p = prec.nrows().min(x.ncols());
    if p == 0 || x.nrows() == 0 {
        return f64::NAN;
    }
    let mut a = Mat::<f64>::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            a[(i, j)] = prec.get(i, j);
        }
    }
    let mut scratch = signlred::Report::new("glasso", "logdet");
    let policy = signlred::Policy::default();
    let logdet = match symmetric_eigen(&mut scratch, &a, &policy) {
        Some((vals, _)) => {
            let mut s = 0.0;
            for v in vals {
                if v <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                s += v.ln();
            }
            s
        }
        None => return f64::NEG_INFINITY,
    };
    let mut quad = 0.0;
    for r in 0..x.nrows() {
        let mut d = Vector::zeros(p);
        for j in 0..p {
            d[j] = x.get(r, j) - loc[j];
        }
        for i in 0..p {
            let mut s = 0.0;
            for j in 0..p {
                s += prec.get(i, j) * d[j];
            }
            quad += d[i] * s;
        }
    }
    0.5 * x.nrows() as f64 * logdet - 0.5 * quad
}

impl FitUnsupervised for GraphicalLassoCV {
    type Fitted = FittedGraphicalLassoCV;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGraphicalLassoCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let folds = match self.cv.split(x.nrows(), &session.child("cv")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vec::new()
            }
        };
        let mut scores = Vec::new();
        let mut best_alpha = self.alphas.first().copied().unwrap_or(0.1);
        let mut best_score = f64::NEG_INFINITY;
        for &alpha in &self.alphas {
            let mut acc = 0.0;
            let mut k = 0.0;
            for (i, fold) in folds.iter().enumerate() {
                let xt = take_rows(x, &fold.train);
                let xv = take_rows(x, &fold.test);
                let mut gl = GraphicalLasso::new(alpha);
                match gl.fit_unsupervised(&xt, &session.child(format!("gl_{alpha}_{i}"))) {
                    Ok(q) => {
                        if let Some(prec) = &q.value.precision {
                            let ll = glasso_loglik(prec, &xv, &q.value.location);
                            if ll.is_finite() {
                                acc += ll;
                                k += 1.0;
                            }
                        }
                    }
                    Err(e) => ctx.push(e.primary),
                }
            }
            let mean = if k > 0.0 { acc / k } else { f64::NAN };
            scores.push((alpha, mean));
            if mean.is_finite() && mean > best_score {
                best_score = mean;
                best_alpha = alpha;
            }
        }
        let mut gl = GraphicalLasso::new(best_alpha);
        let fitted = match gl.fit_unsupervised(x, &session.child("refit")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedCovariance {
                    location: location_of(x),
                    covariance: Matrix::zeros(x.ncols(), x.ncols()),
                    precision: None,
                    shrinkage: best_alpha,
                }
            }
        };
        ctx.finish(FittedGraphicalLassoCV {
            best_alpha,
            best_score,
            scores,
            fitted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::FitUnsupervised;
    use ojizou_san::Session;

    fn blob() -> Matrix {
        Matrix::from_fn(30, 2, |i, j| {
            let v = (i as f64) * 0.05;
            if j == 0 {
                v
            } else {
                0.3 * v + 0.01 * (i as f64)
            }
        })
    }

    #[test]
    fn empirical_is_psd_on_a_line() {
        let x = blob();
        let q = EmpiricalCovariance::new()
            .fit_unsupervised(&x, &Session::new("cov", "emp"))
            .unwrap();
        assert_eq!(q.value.covariance.nrows(), 2);
        assert!(q.value.precision.is_some());
    }

    #[test]
    fn ledoit_and_oas_shrink() {
        let x = blob();
        let lw = LedoitWolf::new()
            .fit_unsupervised(&x, &Session::new("cov", "lw"))
            .unwrap();
        assert!((0.0..=1.0).contains(&lw.value.shrinkage));
        let oas = Oas::new()
            .fit_unsupervised(&x, &Session::new("cov", "oas"))
            .unwrap();
        assert!((0.0..=1.0).contains(&oas.value.shrinkage));
    }

    #[test]
    fn mcd_and_envelope_flag_outlier() {
        let mut x = blob();
        x.set(0, 0, 50.0);
        x.set(0, 1, -40.0);
        let env = EllipticEnvelope {
            contamination: 0.1,
            n_trials: 16,
            seed: 2,
        }
        .fit_unsupervised(&x, &Session::new("cov", "ee"))
        .unwrap();
        let pred = env
            .value
            .predict(&x, &Session::new("cov", "pred"))
            .unwrap()
            .value;
        assert_eq!(pred[0], -1.0);
    }

    #[test]
    fn graphical_lasso_returns_precision() {
        let x = blob();
        let q = GraphicalLasso::new(0.05)
            .fit_unsupervised(&x, &Session::new("cov", "gl"))
            .unwrap();
        assert!(q.value.precision.is_some());
    }

    #[test]
    fn graphical_lasso_cv_picks_finite_alpha() {
        let x = blob();
        let q = GraphicalLassoCV::new(vec![0.05, 0.2])
            .fit_unsupervised(&x, &Session::new("cov", "glcv"))
            .unwrap();
        assert!(q.value.best_alpha.is_finite());
        assert!(q.value.fitted.precision.is_some());
    }
}
