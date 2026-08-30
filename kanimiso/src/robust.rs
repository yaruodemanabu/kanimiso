//! Robust and extra linear estimators: RANSAC, Theil–Sen, GLS, Gamma GLM, OMP.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{chol_solve, least_squares, ridge_solve};
use crate::linear_model::{FittedLinear, LinearRegression};
use crate::rng::Rng;
use crate::traits::{Fit, Predict};
use crate::validate::{inspect_identification, inspect_xy};
use ojizou_san::Session;
use signlred::{
    Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result, Severity,
};

/// RANSAC wrapper around OLS.
#[derive(Clone, Debug)]
pub struct RansacRegressor {
    /// Residual threshold for inliers.
    pub residual_threshold: f64,
    /// Number of random elemental subsets.
    pub max_trials: usize,
    /// Minimum inlier fraction to accept a model.
    pub min_inlier_frac: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for RansacRegressor {
    fn default() -> Self {
        Self {
            residual_threshold: 1.0,
            max_trials: 64,
            min_inlier_frac: 0.5,
            seed: 1,
        }
    }
}

impl RansacRegressor {
    /// Default RANSAC.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted RANSAC: OLS on the consensus set.
#[derive(Clone, Debug)]
pub struct FittedRansac {
    /// Consensus OLS.
    pub model: FittedLinear,
    /// Inlier mask (1 = inlier).
    pub inlier_mask: Vector,
    /// Inlier count.
    pub n_inliers: usize,
}

impl Fit for RansacRegressor {
    type Fitted = FittedRansac;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRansac>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if ctx.report.contains(IssueCode::ConstantTarget) {
            return ctx.finish(FittedRansac {
                model: LinearRegression::new()
                    .fit(x, y, &session.child("inner"))
                    .map(|q| q.value)
                    .unwrap_or_else(|_| dummy_linear(x, y)),
                inlier_mask: Vector::zeros(y.len()),
                n_inliers: 0,
            });
        }
        let n = x.nrows();
        let p = x.ncols() + 1;
        if n < p + 1 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message(format!("RANSAC needs n > p+1; n={n} p_aug={p}"))
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        // Need residual df on each trial so LinearRegression does not abort as interpolation.
        let subset = (p + 3).min(n).max(4);
        let mut best_inliers: Vec<usize> = Vec::new();
        for trial in 0..self.max_trials {
            let idx = rng.sample_indices(n, subset);
            let xs = take_rows(x, &idx).with_intercept();
            let ys = take_y(y, &idx);
            let mut trial_report = signlred::Report::new("ransac", "trial");
            let Some(beta) = least_squares(&mut trial_report, &xs, &ys, &ctx.policy) else {
                continue;
            };
            let intercept = beta[0];
            let pred = Vector::from_iter((1..beta.len()).map(|j| beta[j]));
            let mut inliers = Vec::new();
            for i in 0..n {
                let mut yh = intercept;
                for j in 0..x.ncols() {
                    yh += pred[j] * x.get(i, j);
                }
                if (yh - y[i]).abs() <= self.residual_threshold {
                    inliers.push(i);
                }
            }
            ctx.session.step(trial as u64, inliers.len() as f64, None);
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
            }
        }
        let frac = if n == 0 {
            0.0
        } else {
            best_inliers.len() as f64 / n as f64
        };
        if frac < self.min_inlier_frac {
            ctx.push(
                Issue::builder(IssueCode::OutlierDominated)
                    .message(format!(
                        "RANSAC best inlier fraction {frac:.3} < {}",
                        self.min_inlier_frac
                    ))
                    .metric("inlier_fraction", frac)
                    .build(),
            );
        }
        if best_inliers.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .meaninglessness(Meaninglessness::vacuous(
                        "RANSAC consensus",
                        "no inliers under the residual threshold",
                        "raise residual_threshold or inspect scale",
                    ))
                    .build(),
            );
            return ctx.finish(FittedRansac {
                model: dummy_linear(x, y),
                inlier_mask: Vector::zeros(n),
                n_inliers: 0,
            });
        }
        let xs = take_rows(x, &best_inliers);
        let ys = take_y(y, &best_inliers);
        let model = LinearRegression::new()
            .fit(&xs, &ys, &session.child("ransac_refit"))
            .map(|q| {
                ctx.report.merge(q.report.clone());
                q.value
            })
            .unwrap_or_else(|_| dummy_linear(x, y));
        let mut mask = Vector::zeros(n);
        for &i in &best_inliers {
            mask[i] = 1.0;
        }
        ctx.finish(FittedRansac {
            n_inliers: best_inliers.len(),
            model,
            inlier_mask: mask,
        })
    }
}

impl Predict for FittedRansac {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.model.predict(x, session)
    }
}

fn dummy_linear(x: &Matrix, y: &Vector) -> FittedLinear {
    LinearRegression::new()
        .fit(x, y, &Session::new("ols", "dummy"))
        .map(|q| q.value)
        .unwrap_or_else(|_| FittedLinear {
            coef: Vector::zeros(x.ncols()),
            intercept: y.mean(),
            beta: Vector::zeros(x.ncols() + 1),
            n: y.len(),
            p: x.ncols() + 1,
            df_resid: 0.0,
            r2: f64::NAN,
            adj_r2: f64::NAN,
            sigma2: f64::NAN,
            se: Vector::zeros(x.ncols() + 1),
            t_values: Vector::zeros(x.ncols() + 1),
            p_values: Vector::zeros(x.ncols() + 1),
            aic: f64::NAN,
            bic: f64::NAN,
            f_stat: f64::NAN,
            f_pvalue: f64::NAN,
            durbin_watson: f64::NAN,
            loglik: f64::NAN,
            fitted: Vector::zeros(y.len()),
            resid: Vector::zeros(y.len()),
            leverage: Vector::zeros(y.len()),
            cooks: Vector::zeros(y.len()),
            used_intercept: true,
        })
}

fn take_rows(x: &Matrix, idx: &[usize]) -> Matrix {
    Matrix::from_fn(idx.len(), x.ncols(), |i, j| x.get(idx[i], j))
}

fn take_y(y: &Vector, idx: &[usize]) -> Vector {
    Vector::from_iter(idx.iter().map(|&i| y[i]))
}

/// Theil–Sen pairwise-median slope (multivariate: median of elemental OLS).
#[derive(Clone, Debug)]
pub struct TheilSenRegressor {
    /// Max elemental subsets (all pairs when p=1 and n is small).
    pub max_subsets: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for TheilSenRegressor {
    fn default() -> Self {
        Self {
            max_subsets: 256,
            seed: 2,
        }
    }
}

impl TheilSenRegressor {
    /// Default Theil–Sen.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Theil–Sen.
#[derive(Clone, Debug)]
pub struct FittedTheilSen {
    /// Median slopes.
    pub coef: Vector,
    /// Median intercept.
    pub intercept: f64,
}

impl Fit for TheilSenRegressor {
    type Fitted = FittedTheilSen;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTheilSen>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let n = x.nrows();
        let p = x.ncols();
        if n < 2 {
            ctx.push(Issue::builder(IssueCode::InsufficientSample).build());
            return ctx.finish(FittedTheilSen {
                coef: Vector::zeros(p),
                intercept: y.mean(),
            });
        }
        let mut rng = Rng::new(self.seed);
        if p == 1 {
            let mut slopes = Vec::new();
            let mut intercepts = Vec::new();
            let max_pairs = self.max_subsets;
            let mut seen = 0usize;
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = x.get(j, 0) - x.get(i, 0);
                    if dx.abs() <= ctx.policy.near_zero_variance {
                        continue;
                    }
                    let s = (y[j] - y[i]) / dx;
                    slopes.push(s);
                    intercepts.push(y[i] - s * x.get(i, 0));
                    seen += 1;
                    if seen >= max_pairs {
                        break;
                    }
                }
                if seen >= max_pairs {
                    break;
                }
            }
            if slopes.is_empty() {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("Theil–Sen: every pair had Δx≈0")
                        .build(),
                );
                return ctx.finish(FittedTheilSen {
                    coef: Vector::zeros(1),
                    intercept: y.mean(),
                });
            }
            ctx.finish(FittedTheilSen {
                coef: Vector::from_slice(&[median_of(&mut slopes)]),
                intercept: median_of(&mut intercepts),
            })
        } else {
            let k = (p + 1).min(n);
            let mut coefs: Vec<Vector> = Vec::new();
            let mut intercepts = Vec::new();
            for t in 0..self.max_subsets {
                let idx = rng.sample_indices(n, k);
                let xs = take_rows(x, &idx);
                let ys = take_y(y, &idx);
                if let Ok(q) = LinearRegression::new().fit(&xs, &ys, &session.child("theilsen")) {
                    coefs.push(q.value.coef);
                    intercepts.push(q.value.intercept);
                }
                ctx.session.step(t as u64, coefs.len() as f64, None);
            }
            if coefs.is_empty() {
                ctx.push(Issue::builder(IssueCode::UnidentifiedModel).build());
                return ctx.finish(FittedTheilSen {
                    coef: Vector::zeros(p),
                    intercept: y.mean(),
                });
            }
            let mut med = Vector::zeros(p);
            for j in 0..p {
                let mut col: Vec<f64> = coefs.iter().map(|c| c[j]).collect();
                med[j] = median_of(&mut col);
            }
            ctx.finish(FittedTheilSen {
                coef: med,
                intercept: median_of(&mut intercepts),
            })
        }
    }
}

impl Predict for FittedTheilSen {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            y[i] += self.intercept;
        }
        ctx.finish(y)
    }
}

fn median_of(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

/// Feasible GLS: Ω supplied as a square n×n covariance, or estimated from OLS residuals
/// as a diagonal of squared residuals (heteroscedastic WLS fallback).
#[derive(Clone, Debug)]
pub struct Gls {
    /// If true, prepend intercept.
    pub fit_intercept: bool,
}

impl Default for Gls {
    fn default() -> Self {
        Self {
            fit_intercept: true,
        }
    }
}

impl Gls {
    /// Default GLS.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit with an explicit residual covariance `omega` (n×n, SPD).
    pub fn fit_with_omega(
        &mut self,
        x: &Matrix,
        y: &Vector,
        omega: &faer::Mat<f64>,
        session: &Session,
    ) -> Result<Qualified<FittedLinear>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if omega.nrows() != y.len() || omega.ncols() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("Ω must be n×n")
                    .build(),
            );
            return ctx.finish(dummy_linear(x, y));
        }
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
        // Transform by L^{-1} where Ω = L Lᵀ.
        let mut e0 = Vector::zeros(y.len());
        e0[0] = 1.0;
        if chol_solve(&mut ctx.report, omega, &e0, &ctx.policy).is_none() {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveDefinite)
                    .message("GLS Ω is not SPD; falling back to OLS and recording the compromise")
                    .compromise(NumericalCompromise::new(
                        "GLS with supplied Ω",
                        "OLS",
                        "Cholesky(Ω) failed",
                        "coefficients are not the GLS estimand",
                    ))
                    .build(),
            );
            return LinearRegression {
                fit_intercept: self.fit_intercept,
            }
            .fit(x, y, session);
        }
        // Build L^{-1} X and L^{-1} y via n unit solves (dense, n is expected modest).
        let n = y.len();
        let p = design.ncols();
        let mut xt = Matrix::zeros(n, p);
        let mut yt = Vector::zeros(n);
        for i in 0..n {
            let mut e = Vector::zeros(n);
            e[i] = 1.0;
            let Some(col) = chol_solve(&mut ctx.report, omega, &e, &ctx.policy) else {
                break;
            };
            // This gives Ω^{-1} e_i = i-th column of Ω^{-1}, not L^{-1}.
            // Use Ω^{-1/2} ≈ via solving Ω z = v for each column after Cholesky of Ω
            // on the whitened system: solve Ω β-system as X' Ω^{-1} X.
            let _ = col;
        }
        // Direct GLS normal equations: (X' Ω^{-1} X) β = X' Ω^{-1} y
        let mut xty = Vector::zeros(p);
        let mut xtx = faer::Mat::<f64>::zeros(p, p);
        for j in 0..p {
            let xj = design.column(j);
            let Some(wj) = chol_solve(&mut ctx.report, omega, &xj, &ctx.policy) else {
                ctx.push(Issue::builder(IssueCode::CholeskyFailed).build());
                return ctx.finish(dummy_linear(x, y));
            };
            xty[j] = wj.dot(y);
            for i in 0..=j {
                let xi = design.column(i);
                let g = wj.dot(&xi);
                xtx[(i, j)] = g;
                xtx[(j, i)] = g;
            }
        }
        let Some(beta) = chol_solve(&mut ctx.report, &xtx, &xty, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::InformationMatrixSingular)
                    .message("X'Ω⁻¹X is not SPD")
                    .build(),
            );
            return ctx.finish(dummy_linear(x, y));
        };
        // Reuse OLS inference on the original design as an approximation; flag it.
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("SEs attached via OLS formulas on the original design are not GLS SEs; they ignore Ω")
                .compromise(NumericalCompromise::new(
                    "GLS covariance (X'Ω⁻¹X)⁻¹",
                    "OLS-style SEs on the unwhitened design",
                    "shared FittedLinear inference path",
                    "use the point estimate; do not publish these p-values as GLS",
                ))
                .build(),
        );
        let mut lr = LinearRegression {
            fit_intercept: self.fit_intercept,
        };
        // Return OLS-shaped object but overwrite beta via a predict-consistent fit:
        let mut fitted = lr.fit(x, y, &session.child("gls_shape"))?.value;
        if self.fit_intercept {
            fitted.intercept = beta[0];
            fitted.coef = Vector::from_iter((1..beta.len()).map(|i| beta[i]));
        } else {
            fitted.coef = beta.clone();
            fitted.intercept = 0.0;
        }
        fitted.beta = beta;
        let _ = (xt, yt);
        ctx.finish(fitted)
    }
}

impl Fit for Gls {
    type Fitted = FittedLinear;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinear>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        // Feasible diagonal Ω = diag(e²) from a first OLS, floored.
        let ols = LinearRegression {
            fit_intercept: self.fit_intercept,
        }
        .fit(x, y, &session.child("fgls_ols"))?;
        ctx.report.merge(ols.report.clone());
        let n = y.len();
        let mut omega = faer::Mat::<f64>::zeros(n, n);
        for i in 0..n {
            let e = ols.value.resid[i];
            omega[(i, i)] = (e * e).max(1e-8);
        }
        ctx.push(
            Issue::builder(IssueCode::Heteroscedasticity)
                .message("feasible GLS using diagonal Ω=diag(e²) from a first OLS; this is WLS, not a full GLS")
                .compromise(NumericalCompromise::new(
                    "GLS with a known full Ω",
                    "two-step diagonal FGLS",
                    "no Ω was supplied",
                    "off-diagonal residual dependence is ignored",
                ))
                .build(),
        );
        drop(ctx);
        self.fit_with_omega(x, y, &omega, session)
    }
}

/// Gamma GLM with log link (IRLS).
#[derive(Clone, Debug)]
pub struct GammaRegressor {
    /// ℓ₂ penalty.
    pub alpha: f64,
    /// Max IRLS iterations.
    pub max_iter: usize,
    /// Intercept.
    pub fit_intercept: bool,
}

impl Default for GammaRegressor {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            max_iter: 40,
            fit_intercept: true,
        }
    }
}

impl GammaRegressor {
    /// Default Gamma GLM.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for GammaRegressor {
    type Fitted = FittedLinear;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinear>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        for (i, &yi) in y.as_slice().iter().enumerate() {
            if yi <= 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NonPositiveSeries)
                        .message(format!("Gamma y[{i}]={yi} is not positive"))
                        .build(),
                );
                break;
            }
        }
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        let mut beta = Vector::zeros(design.ncols());
        beta[0] = y.mean().max(1e-6).ln();
        for it in 0..self.max_iter {
            let mut xs = Matrix::zeros(design.nrows(), design.ncols());
            let mut z = Vector::zeros(y.len());
            for i in 0..y.len() {
                let mut eta = 0.0;
                for j in 0..design.ncols() {
                    eta += design.get(i, j) * beta[j];
                }
                let mu = eta.exp().max(1e-12);
                // Gamma log-link: var ∝ μ², weight 1/μ² * μ² = 1 for IRLS working weights μ^{-2}*(dμ/dη)²
                let w: f64 = 1.0;
                let sw = w.sqrt();
                z[i] = (eta + (y[i] - mu) / mu) * sw;
                for j in 0..design.ncols() {
                    xs.set(i, j, design.get(i, j) * sw);
                }
            }
            let next_opt = if self.alpha > 0.0 {
                ridge_solve(&mut ctx.report, &xs, &z, self.alpha, &ctx.policy)
            } else {
                least_squares(&mut ctx.report, &xs, &z, &ctx.policy)
            };
            let Some(next) = next_opt else {
                break;
            };
            let d = next.sub(&beta).norm();
            beta = next;
            ctx.session.step(it as u64, d, None);
            if d < 1e-8 {
                ctx.session.converged("Gamma IRLS", it as u64);
                break;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("Gamma IRLS SEs use the last weighted LS, not the GLM sandwich")
                .build(),
        );
        drop(ctx);
        LinearRegression {
            fit_intercept: self.fit_intercept,
        }
        .fit(x, y, session)
        .map(|mut q| {
            if self.fit_intercept {
                q.value.intercept = beta[0];
                q.value.coef = Vector::from_iter((1..beta.len()).map(|i| beta[i]));
            } else {
                q.value.coef = beta.clone();
            }
            q.value.beta = beta;
            q
        })
    }
}

/// Inverse-Gaussian GLM with a log link (sklearn `TweedieRegressor(power=3)` / statsmodels `GLM`).
///
/// Variance \(\mu^3\). The IRLS working weight is \(1/\mu\).
#[derive(Clone, Debug)]
pub struct InverseGaussianRegressor {
    /// ℓ₂ penalty.
    pub alpha: f64,
    /// Max IRLS iterations.
    pub max_iter: usize,
    /// Intercept.
    pub fit_intercept: bool,
}

impl Default for InverseGaussianRegressor {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            max_iter: 40,
            fit_intercept: true,
        }
    }
}

impl InverseGaussianRegressor {
    /// Default inverse-Gaussian GLM.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for InverseGaussianRegressor {
    type Fitted = FittedLinear;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinear>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        for (i, &yi) in y.as_slice().iter().enumerate() {
            if yi <= 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NonPositiveSeries)
                        .message(format!("InverseGaussian y[{i}]={yi} is not positive"))
                        .build(),
                );
                break;
            }
        }
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        let mut beta = Vector::zeros(design.ncols());
        beta[0] = y.mean().max(1e-6).ln();
        for it in 0..self.max_iter {
            let mut xs = Matrix::zeros(design.nrows(), design.ncols());
            let mut z = Vector::zeros(y.len());
            for i in 0..y.len() {
                let mut eta = 0.0;
                for j in 0..design.ncols() {
                    eta += design.get(i, j) * beta[j];
                }
                let mu = eta.exp().max(1e-12);
                let w = (1.0 / mu).sqrt();
                z[i] = (eta + (y[i] - mu) / mu) * w;
                for j in 0..design.ncols() {
                    xs.set(i, j, design.get(i, j) * w);
                }
            }
            let next_opt = if self.alpha > 0.0 {
                ridge_solve(&mut ctx.report, &xs, &z, self.alpha, &ctx.policy)
            } else {
                least_squares(&mut ctx.report, &xs, &z, &ctx.policy)
            };
            let Some(next) = next_opt else {
                break;
            };
            let d = next.sub(&beta).norm();
            beta = next;
            ctx.session.step(it as u64, d, None);
            if d < 1e-8 {
                ctx.session.converged("InverseGaussian IRLS", it as u64);
                break;
            }
        }
        drop(ctx);
        LinearRegression {
            fit_intercept: self.fit_intercept,
        }
        .fit(x, y, session)
        .map(|mut q| {
            if self.fit_intercept {
                q.value.intercept = beta[0];
                q.value.coef = Vector::from_iter((1..beta.len()).map(|i| beta[i]));
            } else {
                q.value.coef = beta.clone();
            }
            q.value.beta = beta;
            q
        })
    }
}

/// Orthogonal matching pursuit.
#[derive(Clone, Debug)]
pub struct OrthogonalMatchingPursuit {
    /// Number of non-zero coefficients.
    pub n_nonzero: usize,
}

impl Default for OrthogonalMatchingPursuit {
    fn default() -> Self {
        Self { n_nonzero: 1 }
    }
}

impl OrthogonalMatchingPursuit {
    /// OMP with `k` active columns.
    pub fn new(n_nonzero: usize) -> Self {
        Self { n_nonzero }
    }
}

/// Fitted OMP.
#[derive(Clone, Debug)]
pub struct FittedOmp {
    /// Sparse slopes (unselected = 0).
    pub coef: Vector,
    /// Intercept (training y mean).
    pub intercept: f64,
    /// Active feature indices.
    pub support: Vec<usize>,
}

impl Fit for OrthogonalMatchingPursuit {
    type Fitted = FittedOmp;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedOmp>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let (xc, xmean) = x.centered();
        let ymean = y.mean();
        let yc = Vector::from_iter(y.as_slice().iter().map(|v| v - ymean));
        let (n, p) = xc.shape();
        let k = self.n_nonzero.min(p).min(n);
        if k < self.n_nonzero {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!(
                        "OMP requested {} nonzeros, using {k}",
                        self.n_nonzero
                    ))
                    .build(),
            );
        }
        let mut residual = yc.clone();
        let mut support = Vec::new();
        let mut used = vec![false; p];
        for _ in 0..k {
            let mut best = 0usize;
            let mut best_c = 0.0;
            for j in 0..p {
                if used[j] {
                    continue;
                }
                let col = xc.column(j);
                let c = col.dot(&residual).abs();
                if c > best_c {
                    best_c = c;
                    best = j;
                }
            }
            if best_c <= ctx.policy.near_zero_variance {
                ctx.push(
                    Issue::builder(IssueCode::UpdateWithZeroInformation)
                        .message("OMP correlation vanished; remaining coefficients stay 0")
                        .build(),
                );
                break;
            }
            used[best] = true;
            support.push(best);
            let xs = Matrix::from_fn(n, support.len(), |i, j| xc.get(i, support[j]));
            let Some(beta_s) = least_squares(&mut ctx.report, &xs, &yc, &ctx.policy) else {
                break;
            };
            residual = yc.sub(&xs.matvec(&beta_s));
        }
        let xs = Matrix::from_fn(n, support.len(), |i, j| xc.get(i, support[j]));
        let beta_s = least_squares(&mut ctx.report, &xs, &yc, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(support.len()));
        let mut coef = Vector::zeros(p);
        for (t, &j) in support.iter().enumerate() {
            coef[j] = beta_s[t];
        }
        let intercept = ymean - xmean.dot(&coef);
        ctx.finish(FittedOmp {
            coef,
            intercept,
            support,
        })
    }
}

impl Predict for FittedOmp {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            y[i] += self.intercept;
        }
        ctx.finish(y)
    }
}

/// Huber M-estimator (statsmodels `RLM`) via IRLS on a scratch report.
///
/// Inner weighted OLS issues that would abort a valid M-step
/// (`NearSingular`, `ResidualTooLarge`) are not promoted.
#[derive(Clone, Debug)]
pub struct Rlm {
    /// Huber cutoff in residual MAD units.
    pub k: f64,
    /// IRLS iteration cap.
    pub max_iter: usize,
}

impl Default for Rlm {
    fn default() -> Self {
        Self {
            k: 1.345,
            max_iter: 40,
        }
    }
}

impl Rlm {
    /// Default Huber RLM.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted robust slopes.
#[derive(Clone, Debug)]
pub struct FittedRlm {
    /// Slopes.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
}

impl Fit for Rlm {
    type Fitted = FittedRlm;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedRlm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if ctx.report.contains(IssueCode::ConstantTarget) {
            return ctx.finish(FittedRlm {
                coef: Vector::zeros(x.ncols()),
                intercept: y.mean(),
            });
        }
        let design = x.with_intercept();
        let mut scratch = signlred::Report::new("rlm", "ols");
        let Some(mut beta) = least_squares(&mut scratch, &design, y, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("RLM seed OLS failed")
                    .build(),
            );
            return ctx.finish(FittedRlm {
                coef: Vector::zeros(x.ncols()),
                intercept: y.mean(),
            });
        };
        let k = if self.k.is_finite() && self.k > 0.0 {
            self.k
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "RLM k={} is not a positive finite Huber cutoff",
                        self.k
                    ))
                    .build(),
            );
            1.345
        };
        for it in 0..self.max_iter.max(1) {
            let pred = design.matvec(&beta);
            let resid = y.sub(&pred);
            let mut abs: Vec<f64> = resid.as_slice().iter().map(|v| v.abs()).collect();
            abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = if abs.is_empty() {
                1.0
            } else {
                abs[abs.len() / 2].max(1e-12)
            };
            let scale = (mad / 0.6745).max(1e-12);
            let mut xs = Matrix::zeros(design.nrows(), design.ncols());
            let mut ys = Vector::zeros(y.len());
            for i in 0..y.len() {
                let u = resid[i] / (k * scale);
                let w = if u.abs() <= 1.0 { 1.0 } else { 1.0 / u.abs() };
                let sw = w.sqrt();
                ys[i] = y[i] * sw;
                for j in 0..design.ncols() {
                    xs.set(i, j, design.get(i, j) * sw);
                }
            }
            let mut step_rep = signlred::Report::new("rlm", "irls");
            let Some(next) = least_squares(&mut step_rep, &xs, &ys, &ctx.policy) else {
                break;
            };
            let d = next.sub(&beta).norm();
            beta = next;
            ctx.session.step(it as u64, d, None);
            if d < 1e-8 {
                break;
            }
        }
        let intercept = beta.as_slice().first().copied().unwrap_or(0.0);
        let coef = Vector::from_iter((1..beta.len()).map(|j| beta[j]));
        ctx.finish(FittedRlm { coef, intercept })
    }
}

impl Predict for FittedRlm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut y = if x.ncols() == self.coef.len() {
            x.matvec(&self.coef)
        } else {
            Vector::zeros(x.nrows())
        };
        for i in 0..y.len() {
            y[i] += self.intercept;
        }
        ctx.finish(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ransac_recovers_line_with_outliers() {
        let x = Matrix::from_fn(12, 1, |i, _| i as f64);
        let mut y = Vector::from_iter((0..12).map(|i| 2.0 * i as f64));
        y[11] = 1000.0;
        let q = RansacRegressor {
            residual_threshold: 1.5,
            max_trials: 40,
            min_inlier_frac: 0.4,
            seed: 3,
        }
        .fit(&x, &y, &Session::new("ransac", "fit"))
        .expect("ransac");
        assert!(q.value.n_inliers >= 8, "inliers={}", q.value.n_inliers);
        assert!((q.value.model.coef[0] - 2.0).abs() < 0.2);
    }

    #[test]
    fn theil_sen_line() {
        let x = Matrix::from_fn(9, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..9).map(|i| 1.0 + 3.0 * i as f64));
        let q = TheilSenRegressor::new()
            .fit(&x, &y, &Session::new("ts", "fit"))
            .expect("theil");
        assert!((q.value.coef[0] - 3.0).abs() < 1e-8);
        assert!((q.value.intercept - 1.0).abs() < 1e-8);
    }

    #[test]
    fn omp_picks_the_signal_column() {
        let x = Matrix::from_fn(20, 3, |i, j| {
            if j == 1 {
                i as f64
            } else {
                ((i + 3) * (j + 1)) as f64 * 0.01
            }
        });
        let y = Vector::from_iter((0..20).map(|i| 4.0 * i as f64));
        let q = OrthogonalMatchingPursuit::new(1)
            .fit(&x, &y, &Session::new("omp", "fit"))
            .expect("omp");
        assert_eq!(q.value.support, vec![1]);
    }

    #[test]
    fn rlm_recovers_a_line_with_an_outlier() {
        let x = Matrix::from_fn(16, 1, |i, _| i as f64);
        let mut y = Vector::from_iter((0..16).map(|i| 2.0 * i as f64));
        y[15] = 400.0;
        let q = Rlm::new()
            .fit(&x, &y, &Session::new("rlm", "fit"))
            .expect("rlm");
        assert!((q.value.coef[0] - 2.0).abs() < 0.3, "b={}", q.value.coef[0]);
        let yp = Vector::from_iter((0..16).map(|i| (1.0 + 0.15 * i as f64).exp()));
        let ig = InverseGaussianRegressor::new()
            .fit(&x, &yp, &Session::new("ig", "fit"))
            .expect("ig");
        assert!(ig.value.coef.as_slice().iter().all(|v| v.is_finite()));
    }
}
