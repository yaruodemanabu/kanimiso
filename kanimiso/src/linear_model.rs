//! Linear models: OLS / WLS / ridge / lasso / elastic-net / GLM / SGD /
//! robust / isotonic / kernel ridge / PLS — with inference and quality gates.
//!
//! Covers the sklearn `linear_model` surface and the statsmodels OLS/WLS/GLM
//! inference surface.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{least_squares, ridge_solve, thin_svd};
use crate::special::{f_pvalue, student_t_pvalue};
use crate::traits::{Fit, PartialFit, Predict, Transform};
use crate::validate::{inspect_collinearity, inspect_identification, inspect_xy};
use ojizou_san::{IncrementalExplain, Session};
use signlred::{
    IncrementalQuality, Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result,
    Severity,
};

/// Ordinary least squares with full inference (statsmodels-style).
#[derive(Clone, Debug)]
pub struct LinearRegression {
    /// Prepend a column of ones.
    pub fit_intercept: bool,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            fit_intercept: true,
        }
    }
}

impl LinearRegression {
    /// Intercept-on OLS.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted linear model plus diagnostics that decide whether the numbers mean anything.
#[derive(Clone, Debug)]
pub struct FittedLinear {
    /// Slope coefficients (without intercept).
    pub coef: Vector,
    /// Intercept (0 if `fit_intercept` was false and no constant column).
    pub intercept: f64,
    /// Full β including intercept when it was requested.
    pub beta: Vector,
    /// Sample size.
    pub n: usize,
    /// Number of columns actually solved (including intercept).
    pub p: usize,
    /// Residual degrees of freedom.
    pub df_resid: f64,
    /// R².
    pub r2: f64,
    /// Adjusted R².
    pub adj_r2: f64,
    /// Residual variance.
    pub sigma2: f64,
    /// Coefficient standard errors.
    pub se: Vector,
    /// t statistics.
    pub t_values: Vector,
    /// Two-sided p-values.
    pub p_values: Vector,
    /// AIC.
    pub aic: f64,
    /// BIC.
    pub bic: f64,
    /// Overall F.
    pub f_stat: f64,
    /// F p-value.
    pub f_pvalue: f64,
    /// Durbin–Watson.
    pub durbin_watson: f64,
    /// Gaussian log-likelihood.
    pub loglik: f64,
    /// Fitted values.
    pub fitted: Vector,
    /// Residuals.
    pub resid: Vector,
    /// Diagonal of the hat matrix.
    pub leverage: Vector,
    /// Cook's distances.
    pub cooks: Vector,
    /// Whether an intercept column was used.
    pub used_intercept: bool,
}

impl FittedLinear {
    fn predict_vec(&self, x: &Matrix) -> Vector {
        let mut out = x.matvec(&self.coef);
        if self.used_intercept {
            for i in 0..out.len() {
                out[i] += self.intercept;
            }
        }
        out
    }
}

impl Predict for FittedLinear {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!(
                        "predict X is n×{} but model has {} slopes",
                        x.ncols(),
                        self.coef.len()
                    ))
                    .build(),
            );
        }
        crate::validate::inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(self.predict_vec(x))
    }
}

impl Fit for LinearRegression {
    type Fitted = FittedLinear;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinear>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if ctx.report.contains(IssueCode::ConstantTarget)
            || ctx.report.contains(IssueCode::EmptyMatrix)
            || ctx.report.contains(IssueCode::NonFiniteInput)
            || ctx.report.contains(IssueCode::DimensionMismatch)
        {
            return ctx.finish(empty_fitted(x, y, self.fit_intercept));
        }
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
        inspect_collinearity(&mut ctx.report, &design, &ctx.policy);
        let Some(beta) = least_squares(&mut ctx.report, &design, y, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("OLS factorization produced no coefficient vector")
                    .build(),
            );
            return ctx.finish(empty_fitted(x, y, self.fit_intercept));
        };
        let fitted = infer_ols(&mut ctx, &design, y, beta, self.fit_intercept);
        ctx.finish(fitted)
    }
}

fn empty_fitted(x: &Matrix, y: &Vector, used_intercept: bool) -> FittedLinear {
    let p = x.ncols() + if used_intercept { 1 } else { 0 };
    FittedLinear {
        coef: Vector::zeros(x.ncols()),
        intercept: 0.0,
        beta: Vector::zeros(p),
        n: y.len(),
        p,
        df_resid: 0.0,
        r2: f64::NAN,
        adj_r2: f64::NAN,
        sigma2: f64::NAN,
        se: Vector::zeros(p),
        t_values: Vector::zeros(p),
        p_values: Vector::zeros(p),
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
        used_intercept,
    }
}

fn infer_ols(
    ctx: &mut FitCtx,
    design: &Matrix,
    y: &Vector,
    beta: Vector,
    used_intercept: bool,
) -> FittedLinear {
    let n = design.nrows();
    let p = design.ncols();
    let fittedv = design.matvec(&beta);
    let resid = y.sub(&fittedv);
    let sse = resid.dot(&resid);
    let y_mean = y.mean();
    let sst: f64 = y
        .as_slice()
        .iter()
        .map(|yi| {
            let d = yi - y_mean;
            d * d
        })
        .sum();
    let df = n as f64 - p as f64;
    if df <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegreesOfFreedomNonPositive)
                .message(format!("n={n} p={p} ⇒ df_resid={df}"))
                .metric("df_resid", df)
                .meaninglessness(Meaninglessness::vacuous(
                    "σ², SEs, t, p, AIC",
                    "residual degrees of freedom are not positive; the model interpolated or is unidentified",
                    "reduce p or collect data; do not publish p-values",
                ))
                .build(),
        );
    }
    let sigma2 = if df > 0.0 { sse / df } else { f64::NAN };
    let r2 = if sst <= ctx.policy.r2_zero_tol {
        ctx.push(
            Issue::builder(IssueCode::ConstantTarget)
                .message("SST≈0; R² is undefined")
                .build(),
        );
        f64::NAN
    } else {
        1.0 - sse / sst
    };
    if r2.is_finite() && (1.0 - r2).abs() <= ctx.policy.r2_one_tol {
        let mut b = Issue::builder(IssueCode::R2IsOne)
            .message("R² is 1 at working precision")
            .metric("r2", r2);
        if df <= 0.0 {
            b = b.meaninglessness(Meaninglessness::vacuous(
                "OLS R² and coefficient t-tests",
                "the model interpolated (df_resid ≤ 0); in-sample R²=1 is tautological",
                "reduce p or collect data; do not publish in-sample skill",
            ));
        } else {
            b = b.message(
                "R² is 1: the data lie on the fitted hyperplane. Coefficients may be the true generating line, but in-sample skill is not confirmatory.",
            );
        }
        ctx.push(b.build());
    }
    if r2.is_finite() && r2.abs() <= ctx.policy.r2_zero_tol {
        ctx.push(
            Issue::builder(IssueCode::R2IsZero)
                .message("R² is 0; the model is the null (mean) model")
                .metric("r2", r2)
                .build(),
        );
    }
    if r2.is_finite() && r2 < -ctx.policy.r2_zero_tol {
        ctx.push(
            Issue::builder(IssueCode::R2Negative)
                .message(format!("R²={r2:.4e} < 0"))
                .metric("r2", r2)
                .build(),
        );
    }
    let adj_r2 = if df > 0.0 && sst > 0.0 {
        1.0 - (1.0 - r2) * (n as f64 - 1.0) / df
    } else {
        f64::NAN
    };
    let (se, t_values, p_values) = ols_se(ctx, design, &beta, sigma2, df);
    let ssr = (sst - sse).max(0.0);
    let df_model = (p as f64 - if used_intercept { 1.0 } else { 0.0 }).max(0.0);
    let f_stat = if df > 0.0 && df_model > 0.0 && sigma2 > 0.0 {
        (ssr / df_model) / sigma2
    } else {
        f64::NAN
    };
    let f_p = if f_stat.is_finite() {
        f_pvalue(f_stat, df_model, df)
    } else {
        f64::NAN
    };
    let loglik = if sigma2.is_finite() && sigma2 > 0.0 {
        -0.5 * n as f64 * ((2.0 * std::f64::consts::PI * sigma2).ln() + 1.0)
    } else {
        f64::NAN
    };
    let aic = if loglik.is_finite() {
        -2.0 * loglik + 2.0 * p as f64
    } else {
        f64::NAN
    };
    let bic = if loglik.is_finite() {
        -2.0 * loglik + (p as f64) * (n as f64).ln()
    } else {
        f64::NAN
    };
    let mut dw_num = 0.0;
    for i in 1..resid.len() {
        let d = resid[i] - resid[i - 1];
        dw_num += d * d;
    }
    let dw = if sse > 0.0 { dw_num / sse } else { f64::NAN };
    if dw.is_finite() && (dw < 1.0 || dw > 3.0) && n >= 8 {
        ctx.push(
            Issue::builder(IssueCode::AutocorrelatedResiduals)
                .message(format!(
                    "Durbin–Watson={dw:.3}; i.i.d. SEs are not credible"
                ))
                .metric("durbin_watson", dw)
                .build(),
        );
    }
    let (leverage, cooks) = hat_and_cook(design, &resid, sigma2, p);
    let max_h = leverage
        .as_slice()
        .iter()
        .copied()
        .fold(0.0f64, |a, b| a.max(b));
    let h_cut = 2.0 * p as f64 / n as f64;
    if max_h > h_cut && n > p {
        ctx.push(
            Issue::builder(IssueCode::LeveragePoint)
                .message(format!("max leverage {max_h:.4} > 2p/n={h_cut:.4}"))
                .metric("max_leverage", max_h)
                .build(),
        );
    }
    let max_c = cooks
        .as_slice()
        .iter()
        .copied()
        .fold(0.0f64, |a, b| a.max(b));
    if max_c > 1.0 {
        ctx.push(
            Issue::builder(IssueCode::InfluentialPoint)
                .message(format!("max Cook's D={max_c:.4} > 1"))
                .metric("max_cooks", max_c)
                .build(),
        );
    }
    let pred_std = fittedv.std();
    if pred_std <= ctx.policy.near_zero_variance && y.std() > ctx.policy.near_zero_variance {
        ctx.push(
            Issue::builder(IssueCode::PredictionsAreConstant)
                .message("fitted values are constant while y is not")
                .meaninglessness(Meaninglessness::vacuous(
                    "OLS predictor",
                    "the model collapsed to a constant",
                    "inspect collinearity / regularization / target mapping",
                ))
                .build(),
        );
    }

    let (intercept, coef) = split_beta(&beta, used_intercept);
    FittedLinear {
        coef,
        intercept,
        beta,
        n,
        p,
        df_resid: df,
        r2,
        adj_r2,
        sigma2,
        se,
        t_values,
        p_values,
        aic,
        bic,
        f_stat,
        f_pvalue: f_p,
        durbin_watson: dw,
        loglik,
        fitted: fittedv,
        resid,
        leverage,
        cooks,
        used_intercept,
    }
}

fn split_beta(beta: &Vector, used_intercept: bool) -> (f64, Vector) {
    if used_intercept {
        let intercept = beta[0];
        let coef = Vector::from_iter((1..beta.len()).map(|i| beta[i]));
        (intercept, coef)
    } else {
        (0.0, beta.clone())
    }
}

fn ols_se(
    ctx: &mut FitCtx,
    design: &Matrix,
    beta: &Vector,
    sigma2: f64,
    df: f64,
) -> (Vector, Vector, Vector) {
    let p = design.ncols();
    let mut se = Vector::zeros(p);
    let mut t = Vector::zeros(p);
    let mut pv = Vector::zeros(p);
    if !sigma2.is_finite() || sigma2 <= 0.0 || df <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("SEs/p-values withheld: σ² or df is not a valid variance")
                .build(),
        );
        return (se, t, pv);
    }
    let gram = design.gram();
    // Invert Gram via p unit solves.
    let mut diag = vec![f64::NAN; p];
    let mut failed = false;
    for j in 0..p {
        let mut e = Vector::zeros(p);
        e[j] = 1.0;
        match crate::linalg::chol_solve(&mut ctx.report, &gram, &e, &ctx.policy) {
            Some(col) => diag[j] = col[j],
            None => {
                failed = true;
                break;
            }
        }
    }
    if failed {
        ctx.push(
            Issue::builder(IssueCode::InformationMatrixSingular)
                .message("XᵀX is not SPD; Wald SEs are not formed from a Cholesky inverse")
                .compromise(NumericalCompromise::new(
                    "diag((XᵀX)⁻¹)",
                    "SEs left as NaN",
                    "Gram Cholesky failed",
                    "do not publish stars; the information matrix is singular",
                ))
                .build(),
        );
        return (se, t, pv);
    }
    for j in 0..p {
        let v = (sigma2 * diag[j]).max(0.0).sqrt();
        se[j] = v;
        if v > 0.0 && v.is_finite() {
            t[j] = beta[j] / v;
            pv[j] = student_t_pvalue(t[j], df);
        } else {
            t[j] = f64::NAN;
            pv[j] = f64::NAN;
            ctx.push(
                Issue::builder(IssueCode::ConfidenceIntervalDegenerate)
                    .message(format!("SE[{j}] is {v}"))
                    .build(),
            );
        }
    }
    (se, t, pv)
}

fn hat_and_cook(design: &Matrix, resid: &Vector, sigma2: f64, p: usize) -> (Vector, Vector) {
    let n = design.nrows();
    let mut lev = Vector::zeros(n);
    let mut cooks = Vector::zeros(n);
    if n == 0 || p == 0 || !sigma2.is_finite() || sigma2 <= 0.0 {
        return (lev, cooks);
    }
    // h_ii = x_i (X'X)⁺ x_iᵀ  via faer QR of X: Q Qᵀ diagonal.
    let qr = design.inner().qr();
    let q = qr.compute_thin_Q();
    let rnk = q.ncols();
    for i in 0..n {
        let mut h = 0.0;
        for k in 0..rnk {
            let v = q[(i, k)];
            h += v * v;
        }
        lev[i] = h;
        let den = sigma2 * p as f64 * (1.0 - h).max(1e-15);
        cooks[i] = (resid[i] * resid[i] * h) / den;
    }
    (lev, cooks)
}

/// Weighted least squares.
#[derive(Clone, Debug)]
pub struct Wls {
    /// Prepend intercept.
    pub fit_intercept: bool,
}

impl Default for Wls {
    fn default() -> Self {
        Self {
            fit_intercept: true,
        }
    }
}

impl Wls {
    /// Default WLS.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit with observation weights (sqrt-transform).
    pub fn fit_weighted(
        &mut self,
        x: &Matrix,
        y: &Vector,
        weights: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinear>> {
        let mut ctx = FitCtx::with_session(session.child("fit"));
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if weights.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("weights length ≠ n")
                    .build(),
            );
            return ctx.finish(empty_fitted(x, y, self.fit_intercept));
        }
        for (i, &w) in weights.as_slice().iter().enumerate() {
            if !w.is_finite() || w < 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::InvalidWeight)
                        .message(format!("weight[{i}]={w}"))
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
        let mut xs = Matrix::zeros(design.nrows(), design.ncols());
        let mut ys = Vector::zeros(y.len());
        for i in 0..y.len() {
            let s = weights[i].sqrt();
            ys[i] = y[i] * s;
            for j in 0..design.ncols() {
                xs.set(i, j, design.get(i, j) * s);
            }
        }
        ctx.push(
            Issue::builder(IssueCode::IllConditioned)
                .severity(signlred::Severity::Info)
                .message("WLS via √w row scaling; SEs are conditional on the supplied weights")
                .build(),
        );
        // Info severity is advisory-like; IssueCode::IllConditioned default is Warning.
        // Use Audit via a dedicated advisory code instead:
        let _ = IssueCode::MultipleTestingUncorrected;
        let Some(beta) = least_squares(&mut ctx.report, &xs, &ys, &ctx.policy) else {
            return ctx.finish(empty_fitted(x, y, self.fit_intercept));
        };
        let fitted = infer_ols(&mut ctx, &design, y, beta, self.fit_intercept);
        ctx.finish(fitted)
    }
}

/// Ridge regression (ℓ₂).
#[derive(Clone, Debug)]
pub struct Ridge {
    /// Penalty λ ≥ 0 (applied to slopes; intercept is not penalized).
    pub alpha: f64,
    /// Prepend intercept.
    pub fit_intercept: bool,
}

impl Default for Ridge {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
        }
    }
}

impl Ridge {
    /// Ridge with the given λ.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            fit_intercept: true,
        }
    }
}

/// Fitted ridge / lasso / elastic-net (prediction only; inference is not the OLS estimand).
#[derive(Clone, Debug)]
pub struct FittedPenalized {
    /// Slopes.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
    /// Penalty used.
    pub alpha: f64,
    /// ℓ1 ratio (0 = ridge, 1 = lasso).
    pub l1_ratio: f64,
}

impl Predict for FittedPenalized {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("penalized predict shape mismatch")
                    .build(),
            );
        }
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            y[i] += self.intercept;
        }
        ctx.finish(y)
    }
}

impl Fit for Ridge {
    type Fitted = FittedPenalized;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPenalized>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let (xc, xmean) = x.centered();
        let ymean = y.mean();
        let yc = Vector::from_iter(y.as_slice().iter().map(|v| v - ymean));
        if self.alpha == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::RidgeFallbackUsed)
                    .message("ridge α=0: this is OLS, not a regularized estimand")
                    .compromise(NumericalCompromise::new(
                        "ridge with α>0",
                        "OLS on centered data",
                        "caller set α=0",
                        "SEs if computed would be OLS SEs; do not call this ridge",
                    ))
                    .build(),
            );
        }
        let Some(coef) = ridge_solve(&mut ctx.report, &xc, &yc, self.alpha, &ctx.policy) else {
            ctx.push(Issue::builder(IssueCode::UnidentifiedModel).build());
            return ctx.finish(FittedPenalized {
                coef: Vector::zeros(x.ncols()),
                intercept: ymean,
                alpha: self.alpha,
                l1_ratio: 0.0,
            });
        };
        let intercept = if self.fit_intercept {
            ymean - xmean.dot(&coef)
        } else {
            0.0
        };
        ctx.finish(FittedPenalized {
            coef,
            intercept,
            alpha: self.alpha,
            l1_ratio: 0.0,
        })
    }
}

/// Lasso via cyclic coordinate descent.
#[derive(Clone, Debug)]
pub struct Lasso {
    /// λ ≥ 0 on the ℓ1 penalty (on slopes).
    pub alpha: f64,
    /// Max coordinate cycles.
    pub max_iter: usize,
    /// Coordinate change tolerance.
    pub tol: f64,
    /// Prepend intercept (unpenalized).
    pub fit_intercept: bool,
}

impl Default for Lasso {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-6,
            fit_intercept: true,
        }
    }
}

impl Lasso {
    /// Lasso with the given λ.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::default()
        }
    }
}

fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

fn elastic_net_cd(
    ctx: &mut FitCtx,
    x: &Matrix,
    y: &Vector,
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
) -> Vector {
    let (n, p) = x.shape();
    let mut w = Vector::zeros(p);
    let mut col_norm2 = vec![0.0; p];
    for j in 0..p {
        let mut s = 0.0;
        for i in 0..n {
            let v = x.get(i, j);
            s += v * v;
        }
        col_norm2[j] = s;
        if s <= ctx.policy.near_zero_variance {
            ctx.push(signlred::constant_feature_issue(
                j,
                signlred::slice_stats(&(0..n).map(|i| x.get(i, j)).collect::<Vec<_>>()),
            ));
        }
    }
    let l1 = alpha * l1_ratio;
    let l2 = alpha * (1.0 - l1_ratio);
    let mut resid = y.clone();
    let mut converged = false;
    for it in 0..max_iter {
        let mut max_delta = 0.0;
        for j in 0..p {
            if col_norm2[j] <= 0.0 {
                continue;
            }
            let wj = w[j];
            for i in 0..n {
                resid[i] += x.get(i, j) * wj;
            }
            let mut rho = 0.0;
            for i in 0..n {
                rho += x.get(i, j) * resid[i];
            }
            let denom = col_norm2[j] + n as f64 * l2;
            let w_new = soft_threshold(rho / denom, n as f64 * l1 / denom);
            let delta = (w_new - wj).abs();
            if delta > max_delta {
                max_delta = delta;
            }
            w[j] = w_new;
            for i in 0..n {
                resid[i] -= x.get(i, j) * w_new;
            }
        }
        ctx.session
            .step(it as u64, resid.dot(&resid), Some(max_delta));
        if max_delta < tol {
            ctx.session
                .converged(format!("coordinate change < {tol}"), it as u64);
            converged = true;
            break;
        }
    }
    if !converged {
        ctx.push(
            Issue::builder(IssueCode::MaxIterReached)
                .message(format!("elastic-net CD hit {max_iter} cycles"))
                .build(),
        );
    }
    if w.as_slice().iter().all(|c| c.abs() == 0.0) && y.std() > ctx.policy.near_zero_variance {
        ctx.push(
            Issue::builder(IssueCode::InterceptOnlyCollapse)
                .message("all slopes shrank to 0; the model is a mean")
                .meaninglessness(Meaninglessness::vacuous(
                    "lasso / elastic-net slopes",
                    "the penalty ate every direction",
                    "decrease α or disable the penalty",
                ))
                .build(),
        );
    }
    w
}

impl Fit for Lasso {
    type Fitted = FittedPenalized;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPenalized>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let (xc, xmean) = x.centered();
        let ymean = y.mean();
        let yc = Vector::from_iter(y.as_slice().iter().map(|v| v - ymean));
        let coef = elastic_net_cd(&mut ctx, &xc, &yc, self.alpha, 1.0, self.max_iter, self.tol);
        let intercept = if self.fit_intercept {
            ymean - xmean.dot(&coef)
        } else {
            0.0
        };
        ctx.finish(FittedPenalized {
            coef,
            intercept,
            alpha: self.alpha,
            l1_ratio: 1.0,
        })
    }
}

/// Elastic net.
#[derive(Clone, Debug)]
pub struct ElasticNet {
    /// Combined penalty strength.
    pub alpha: f64,
    /// Mixing: 1 = lasso, 0 = ridge.
    pub l1_ratio: f64,
    /// Max coordinate cycles.
    pub max_iter: usize,
    /// Tolerance.
    pub tol: f64,
    /// Intercept.
    pub fit_intercept: bool,
}

impl Default for ElasticNet {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            l1_ratio: 0.5,
            max_iter: 1000,
            tol: 1e-6,
            fit_intercept: true,
        }
    }
}

impl ElasticNet {
    /// Elastic net with α and ℓ1 ratio.
    pub fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self {
            alpha,
            l1_ratio,
            ..Self::default()
        }
    }
}

impl Fit for ElasticNet {
    type Fitted = FittedPenalized;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPenalized>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("l1_ratio={} not in [0,1]", self.l1_ratio))
                    .build(),
            );
        }
        let (xc, xmean) = x.centered();
        let ymean = y.mean();
        let yc = Vector::from_iter(y.as_slice().iter().map(|v| v - ymean));
        let coef = elastic_net_cd(
            &mut ctx,
            &xc,
            &yc,
            self.alpha,
            self.l1_ratio,
            self.max_iter,
            self.tol,
        );
        let intercept = if self.fit_intercept {
            ymean - xmean.dot(&coef)
        } else {
            0.0
        };
        ctx.finish(FittedPenalized {
            coef,
            intercept,
            alpha: self.alpha,
            l1_ratio: self.l1_ratio,
        })
    }
}

/// Binary / multinomial logistic regression via IRLS (Newton–Raphson).
#[derive(Clone, Debug)]
pub struct LogisticRegression {
    /// ℓ₂ penalty (0 = unregularized MLE).
    pub c_inv: f64,
    /// Max IRLS iterations.
    pub max_iter: usize,
    /// Gradient-norm tolerance.
    pub tol: f64,
    /// Intercept.
    pub fit_intercept: bool,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self {
            c_inv: 0.0,
            max_iter: 50,
            tol: 1e-8,
            fit_intercept: true,
        }
    }
}

impl LogisticRegression {
    /// Unregularized logistic.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted logistic model.
#[derive(Clone, Debug)]
pub struct FittedLogistic {
    /// Slopes of the last-vs-rest (binary) or first non-reference (softmax) block.
    pub coef: Vector,
    /// Intercept of that same block.
    pub intercept: f64,
    /// Classes (sorted).
    pub classes: Vec<i64>,
    /// Full β including intercept (binary IRLS only).
    pub beta: Vector,
    /// Joint softmax MLE when \(K>2\). Binary fits leave this `None`.
    pub softmax: Option<crate::multinomial::FittedMultinomial>,
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

impl Fit for LogisticRegression {
    type Fitted = FittedLogistic;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLogistic>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = crate::validate::inspect_classes(&mut ctx.report, y, &ctx.policy);
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        if classes.len() < 2 {
            return ctx.finish(FittedLogistic {
                coef: Vector::zeros(x.ncols()),
                intercept: 0.0,
                classes,
                beta: Vector::zeros(design.ncols()),
                softmax: None,
            });
        }
        if classes.len() > 2 {
            let q = crate::multinomial::MultinomialLogistic {
                c_inv: if self.c_inv > 0.0 { self.c_inv } else { 1e-6 },
                max_iter: self.max_iter,
                tol: self.tol,
                fit_intercept: self.fit_intercept,
            }
            .fit(x, y, session)?;
            return Ok(q.map(|sm| {
                let (intercept, coef, beta) = first_softmax_block(&sm);
                FittedLogistic {
                    coef,
                    intercept,
                    classes: sm.classes.clone(),
                    beta,
                    softmax: Some(sm),
                }
            }));
        }
        let pos = *classes.last().unwrap();
        let y01 = Vector::from_iter(y.as_slice().iter().map(|v| {
            if (*v).round() as i64 == pos {
                1.0
            } else {
                0.0
            }
        }));
        let (n, p) = design.shape();
        let mut beta = Vector::zeros(p);
        let mut converged = false;
        for it in 0..self.max_iter {
            let mut wsqrt = Vector::zeros(n);
            let mut z = Vector::zeros(n);
            let mut sep = 0usize;
            let mut gnorm = 0.0;
            for i in 0..n {
                let mut eta = 0.0;
                for j in 0..p {
                    eta += design.get(i, j) * beta[j];
                }
                let mu = sigmoid(eta);
                if (mu - y01[i]).abs() < 1e-12 && (eta.abs() > 20.0) {
                    sep += 1;
                }
                let var = (mu * (1.0 - mu)).max(1e-12);
                wsqrt[i] = var.sqrt();
                z[i] = eta + (y01[i] - mu) / var;
                let gi = (mu - y01[i]);
                gnorm += gi * gi;
            }
            gnorm = gnorm.sqrt();
            ctx.session.step(it as u64, gnorm, Some(gnorm));
            if sep == n {
                ctx.push(
                    Issue::builder(IssueCode::PerfectSeparation)
                        .message("every observation is perfectly separated; MLE diverges")
                        .meaninglessness(Meaninglessness {
                            what_was_computed: "logistic MLE".into(),
                            why_meaningless: "finite coefficients are an artifact of the iteration cap / ridge".into(),
                            interpretive_value: signlred::InterpretiveValue::False,
                            suggested_action: "use Firth / exact logistic / a penalty; do not publish these odds ratios".into(),
                        })
                        .build(),
                );
                break;
            }
            let mut xs = Matrix::zeros(n, p);
            let mut zs = Vector::zeros(n);
            for i in 0..n {
                zs[i] = z[i] * wsqrt[i];
                for j in 0..p {
                    xs.set(i, j, design.get(i, j) * wsqrt[i]);
                }
            }
            let step_opt = if self.c_inv > 0.0 {
                ridge_solve(&mut ctx.report, &xs, &zs, self.c_inv, &ctx.policy)
            } else {
                least_squares(&mut ctx.report, &xs, &zs, &ctx.policy)
            };
            let Some(step) = step_opt else {
                break;
            };
            if !step.as_slice().iter().all(|v| v.is_finite()) {
                ctx.push(Issue::builder(IssueCode::NonFiniteOutput).build());
                break;
            }
            let delta = step.sub(&beta).norm();
            beta = step;
            if gnorm < self.tol || delta < self.tol {
                ctx.session.converged("IRLS gradient/step small", it as u64);
                converged = true;
                break;
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("logistic IRLS did not meet the tolerance")
                    .build(),
            );
        }
        let (intercept, coef) = split_beta(&beta, self.fit_intercept);
        ctx.finish(FittedLogistic {
            coef,
            intercept,
            classes,
            beta,
            softmax: None,
        })
    }
}

fn first_softmax_block(sm: &crate::multinomial::FittedMultinomial) -> (f64, Vector, Vector) {
    if sm.coef.nrows() == 0 {
        return (0.0, Vector::zeros(0), Vector::zeros(0));
    }
    let p = sm.coef.ncols();
    let mut row = Vector::zeros(p);
    for j in 0..p {
        row[j] = sm.coef.get(0, j);
    }
    let (intercept, coef) = split_beta(&row, sm.used_intercept);
    (intercept, coef, row)
}

impl Predict for FittedLogistic {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        if let Some(sm) = &self.softmax {
            return sm.predict(x, session);
        }
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut scores = x.matvec(&self.coef);
        for i in 0..scores.len() {
            let p = sigmoid(scores[i] + self.intercept);
            scores[i] = if p >= 0.5 {
                *self.classes.last().unwrap_or(&1) as f64
            } else {
                *self.classes.first().unwrap_or(&0) as f64
            };
        }
        ctx.finish(scores)
    }
}

/// Mini-batch / streaming SGD for linear regression (also the river-style online linear model).
#[derive(Clone, Debug)]
pub struct SgdRegressor {
    /// Learning rate.
    pub learning_rate: f64,
    /// ℓ2 penalty.
    pub l2: f64,
    /// Intercept.
    pub fit_intercept: bool,
    coef: Vector,
    intercept: f64,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for SgdRegressor {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            l2: 0.0,
            fit_intercept: true,
            coef: Vector::zeros(0),
            intercept: 0.0,
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl SgdRegressor {
    /// Default SGD regressor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current slopes.
    pub fn coef(&self) -> &Vector {
        &self.coef
    }

    /// Current intercept.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

impl PartialFit for SgdRegressor {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(
                Issue::builder(IssueCode::MissingTarget)
                    .message("SgdRegressor.partial_fit requires y")
                    .build(),
            );
            return ctx.finish(dummy_explain(self.updates, 0, self.n_seen));
        };
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if !self.initialized {
            self.coef = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if self.coef.len() != x.ncols() {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message(format!(
                        "partial_fit X has {} columns; model has {}",
                        x.ncols(),
                        self.coef.len()
                    ))
                    .build(),
            );
            return ctx.finish(dummy_explain(self.updates, x.nrows(), self.n_seen));
        }
        let before = self.coef.clone();
        let loss_before = mse_of(x, y, &self.coef, self.intercept);
        let mut info = 0.0;
        for i in 0..x.nrows() {
            let mut pred = self.intercept;
            for j in 0..x.ncols() {
                pred += self.coef[j] * x.get(i, j);
            }
            let err = pred - y[i];
            info += err * err;
            for j in 0..x.ncols() {
                self.coef[j] -= self.learning_rate * (err * x.get(i, j) + self.l2 * self.coef[j]);
            }
            if self.fit_intercept {
                self.intercept -= self.learning_rate * err;
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.coef.sub(&before);
        let loss_after = mse_of(x, y, &self.coef, self.intercept);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.norm());
        q.parameter_delta_max = Some(delta.max_abs());
        q.loss_before = Some(loss_before);
        q.loss_after = Some(loss_after);
        q.information_gain = Some(info);
        q.still_identified = self.n_seen as usize > x.ncols();
        q.warmup = self.n_seen < 5;
        q.explanation = format!(
            "SGD/OLS update: {} rows, η={}, Δθ_ℓ2={:.4e}, mse {:.6e} → {:.6e}",
            x.nrows(),
            self.learning_rate,
            delta.norm(),
            loss_before,
            loss_after
        );
        if q.is_uninformative(ctx.policy.uninformative_info_eps) {
            ctx.push(
                Issue::builder(IssueCode::UpdateWithZeroInformation)
                    .incremental(q.clone())
                    .message("this SGD batch did not move the parameters or the residual")
                    .build(),
            );
        }
        if !q.still_identified {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("n_seen ≤ p; the online linear model is not yet identified")
                    .build(),
            );
        }
        if self.learning_rate > 1.0 {
            ctx.push(
                Issue::builder(IssueCode::LearningRateTooLarge)
                    .message(format!(
                        "η={} is likely unstable for unscaled features",
                        self.learning_rate
                    ))
                    .build(),
            );
        }
        let expl = IncrementalExplain::from_quality(
            q,
            format!("coef[{}] and intercept", self.coef.len()),
            "stochastic gradient of squared error plus ℓ2",
            format!("mse={loss_before:.6e}"),
            format!("mse={loss_after:.6e}"),
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

fn mse_of(x: &Matrix, y: &Vector, coef: &Vector, intercept: f64) -> f64 {
    if x.nrows() == 0 || coef.len() != x.ncols() {
        return f64::NAN;
    }
    let mut s = 0.0;
    for i in 0..x.nrows() {
        let mut pred = intercept;
        for j in 0..x.ncols() {
            pred += coef[j] * x.get(i, j);
        }
        let e = pred - y[i];
        s += e * e;
    }
    s / x.nrows() as f64
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

impl Predict for SgdRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            y[i] += self.intercept;
        }
        ctx.finish(y)
    }
}

/// Huber IRLS (robust M-estimator).
#[derive(Clone, Debug)]
pub struct HuberRegressor {
    /// Huber threshold.
    pub epsilon: f64,
    /// Max IRLS iterations.
    pub max_iter: usize,
    /// Intercept.
    pub fit_intercept: bool,
}

impl Default for HuberRegressor {
    fn default() -> Self {
        Self {
            epsilon: 1.35,
            max_iter: 50,
            fit_intercept: true,
        }
    }
}

impl HuberRegressor {
    /// Default Huber.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for HuberRegressor {
    type Fitted = FittedLinear;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinear>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        let Some(mut beta) = least_squares(&mut ctx.report, &design, y, &ctx.policy) else {
            return ctx.finish(empty_fitted(x, y, self.fit_intercept));
        };
        for it in 0..self.max_iter {
            let pred = design.matvec(&beta);
            let resid = y.sub(&pred);
            let mad = median_abs(resid.as_slice());
            let scale = (mad / 0.6745).max(1e-12);
            let mut xs = Matrix::zeros(design.nrows(), design.ncols());
            let mut ys = Vector::zeros(y.len());
            let mut n_clipped = 0usize;
            for i in 0..y.len() {
                let u = resid[i] / (self.epsilon * scale);
                let w = if u.abs() <= 1.0 {
                    1.0
                } else {
                    n_clipped += 1;
                    1.0 / u.abs()
                };
                let sw = w.sqrt();
                ys[i] = y[i] * sw;
                for j in 0..design.ncols() {
                    xs.set(i, j, design.get(i, j) * sw);
                }
            }
            ctx.session.step(it as u64, resid.dot(&resid), None);
            let Some(next) = least_squares(&mut ctx.report, &xs, &ys, &ctx.policy) else {
                break;
            };
            let d = next.sub(&beta).norm();
            beta = next;
            if d < 1e-8 {
                ctx.session.converged("Huber IRLS", it as u64);
                break;
            }
            if it == self.max_iter - 1 {
                ctx.push(Issue::builder(IssueCode::MaxIterReached).build());
            }
            if n_clipped == 0 && it > 0 {
                ctx.push(
                    Issue::builder(IssueCode::OutlierDominated)
                        .severity(signlred::Severity::Advisory)
                        .message("no residual exceeded the Huber threshold; the fit is OLS")
                        .build(),
                );
            }
        }
        ctx.push(
            Issue::builder(IssueCode::RidgeFallbackUsed)
                .severity(signlred::Severity::Advisory)
                .message("Huber IRLS SEs below are OLS-style on the last weighted design; they are not the M-estimator sandwich")
                .compromise(NumericalCompromise::new(
                    "Huber sandwich covariance",
                    "OLS inference on the last IRLS weighted system",
                    "a full sandwich is not formed in this path",
                    "treat p-values as approximate",
                ))
                .build(),
        );
        let fitted = infer_ols(&mut ctx, &design, y, beta, self.fit_intercept);
        ctx.finish(fitted)
    }
}

fn median_abs(xs: &[f64]) -> f64 {
    let mut v: Vec<f64> = xs.iter().map(|x| x.abs()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Isotonic regression (PAVA).
#[derive(Clone, Debug, Default)]
pub struct IsotonicRegression {
    /// If true, fit an increasing step function.
    pub increasing: bool,
}

impl IsotonicRegression {
    /// Increasing isotonic.
    pub fn new() -> Self {
        Self { increasing: true }
    }

    /// Fit against a 1-d feature stored as `x` column 0, or against the index if p=0.
    pub fn fit_1d(
        &mut self,
        x: &Vector,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedIsotonic>> {
        let mut ctx = FitCtx::with_session(session.clone());
        if x.len() != y.len() {
            ctx.push(Issue::builder(IssueCode::DimensionMismatch).build());
            return ctx.finish(FittedIsotonic {
                xs: Vector::zeros(0),
                ys: Vector::zeros(0),
                increasing: self.increasing,
            });
        }
        if let Some(issue) = signlred::scan_finite(x.as_slice()).to_issue("x") {
            ctx.push(issue);
        }
        if let Some(issue) = signlred::scan_finite(y.as_slice()).to_issue("y") {
            ctx.push(issue);
        }
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
        let mut xs = Vec::with_capacity(idx.len());
        let mut ys = Vec::with_capacity(idx.len());
        for i in idx {
            xs.push(x[i]);
            ys.push(if self.increasing { y[i] } else { -y[i] });
        }
        // PAVA
        let mut blocks: Vec<(f64, f64, f64)> = Vec::new(); // sum, weight, mean
        for &yi in &ys {
            blocks.push((yi, 1.0, yi));
            while blocks.len() >= 2 {
                let n = blocks.len();
                if blocks[n - 2].2 <= blocks[n - 1].2 {
                    break;
                }
                let (s2, w2, _) = blocks.pop().unwrap();
                let (s1, w1, _) = blocks.pop().unwrap();
                let s = s1 + s2;
                let w = w1 + w2;
                blocks.push((s, w, s / w));
            }
        }
        let mut out_y = Vec::new();
        for (_, w, m) in &blocks {
            for _ in 0..(*w as usize) {
                out_y.push(if self.increasing { *m } else { -m });
            }
        }
        // reconstruct xs unique for predict
        ctx.finish(FittedIsotonic {
            xs: Vector::from_iter(xs),
            ys: Vector::from_iter(out_y),
            increasing: self.increasing,
        })
    }
}

/// Fitted isotonic step function.
#[derive(Clone, Debug)]
pub struct FittedIsotonic {
    /// Sorted unique-ish x.
    pub xs: Vector,
    /// Isotonic y.
    pub ys: Vector,
    /// Direction.
    pub increasing: bool,
}

impl FittedIsotonic {
    /// Predict by left-constant interpolation on the fitted steps.
    pub fn predict_1d(&self, x: &Vector) -> Vector {
        Vector::from_iter(x.as_slice().iter().map(|&xi| {
            if self.xs.is_empty() {
                return f64::NAN;
            }
            let mut y = self.ys[0];
            for i in 0..self.xs.len() {
                if self.xs[i] <= xi {
                    y = self.ys[i];
                } else {
                    break;
                }
            }
            y
        }))
    }
}

/// Kernel ridge (RBF) via faer Cholesky on the regularized Gram.
#[derive(Clone, Debug)]
pub struct KernelRidge {
    /// Ridge on the kernel Gram.
    pub alpha: f64,
    /// RBF length scale.
    pub gamma: f64,
}

impl Default for KernelRidge {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            gamma: 1.0,
        }
    }
}

impl KernelRidge {
    /// Default RBF kernel ridge.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted kernel ridge.
#[derive(Clone, Debug)]
pub struct FittedKernelRidge {
    /// Training features.
    pub x_train: Matrix,
    /// Dual coefficients.
    pub dual: Vector,
    /// RBF γ.
    pub gamma: f64,
}

impl Fit for KernelRidge {
    type Fitted = FittedKernelRidge;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedKernelRidge>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let n = x.nrows();
        let mut k = faer::Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let mut d2 = 0.0;
                for c in 0..x.ncols() {
                    let d = x.get(i, c) - x.get(j, c);
                    d2 += d * d;
                }
                let kij = (-self.gamma * d2).exp();
                k[(i, j)] = kij;
                k[(j, i)] = kij;
            }
            k[(i, i)] += self.alpha;
        }
        // PSD check: try Cholesky
        let Some(dual) = crate::linalg::chol_solve(&mut ctx.report, &k, y, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::KernelNotPd)
                    .message("RBF Gram + αI failed Cholesky")
                    .build(),
            );
            return ctx.finish(FittedKernelRidge {
                x_train: x.clone(),
                dual: Vector::zeros(n),
                gamma: self.gamma,
            });
        };
        if n > 50 {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message("kernel ridge interpolates when n is large vs effective dimension; this is a smoother, not an identified parametric model")
                    .build(),
            );
        }
        ctx.finish(FittedKernelRidge {
            x_train: x.clone(),
            dual,
            gamma: self.gamma,
        })
    }
}

impl Predict for FittedKernelRidge {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut out = Vector::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut s = 0.0;
            for t in 0..self.x_train.nrows() {
                let mut d2 = 0.0;
                for c in 0..x.ncols().min(self.x_train.ncols()) {
                    let d = x.get(i, c) - self.x_train.get(t, c);
                    d2 += d * d;
                }
                s += self.dual[t] * (-self.gamma * d2).exp();
            }
            out[i] = s;
        }
        ctx.finish(out)
    }
}

/// NIPALS PLS regression (single or multiple latent directions).
#[derive(Clone, Debug)]
pub struct PlsRegression {
    /// Latent directions.
    pub n_components: usize,
}

impl Default for PlsRegression {
    fn default() -> Self {
        Self { n_components: 2 }
    }
}

impl PlsRegression {
    /// PLS with `k` components.
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }
}

/// Fitted PLS.
#[derive(Clone, Debug)]
pub struct FittedPls {
    /// X mean.
    pub x_mean: Vector,
    /// y mean.
    pub y_mean: f64,
    /// Regression on the original features (deflated NIPALS collapsed).
    pub coef: Vector,
}

impl Fit for PlsRegression {
    type Fitted = FittedPls;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedPls>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let (mut xs, xmean) = x.centered();
        let ymean = y.mean();
        let mut ys = Vector::from_iter(y.as_slice().iter().map(|v| v - ymean));
        let k = self
            .n_components
            .min(x.ncols())
            .min(x.nrows().saturating_sub(1));
        if k < self.n_components {
            ctx.push(
                Issue::builder(IssueCode::ComponentsExceedRank)
                    .message(format!(
                        "requested {} PLS components, using {k}",
                        self.n_components
                    ))
                    .compromise(NumericalCompromise::new(
                        format!("{} latent directions", self.n_components),
                        format!("{k} NIPALS directions"),
                        "n or p cannot support more",
                        "later components are not estimated",
                    ))
                    .build(),
            );
        }
        let mut coef = Vector::zeros(x.ncols());
        for _ in 0..k {
            let w = xs.matvec_t(&ys);
            let wn = w.norm();
            if wn <= ctx.policy.near_zero_variance {
                ctx.push(
                    Issue::builder(IssueCode::UpdateWithZeroInformation)
                        .message("NIPALS weight vanished; remaining components are unidentified")
                        .build(),
                );
                break;
            }
            let w = w.scale(1.0 / wn);
            let t = xs.matvec(&w);
            let tt = t.dot(&t);
            if tt <= ctx.policy.near_zero_variance {
                break;
            }
            let q = t.dot(&ys) / tt;
            let pvec = xs.matvec_t(&t).scale(1.0 / tt);
            // deflate
            for i in 0..xs.nrows() {
                for j in 0..xs.ncols() {
                    xs.set(i, j, xs.get(i, j) - t[i] * pvec[j]);
                }
                ys[i] -= t[i] * q;
            }
            for j in 0..coef.len() {
                coef[j] += w[j] * q;
            }
        }
        ctx.finish(FittedPls {
            x_mean: xmean,
            y_mean: ymean,
            coef,
        })
    }
}

impl Predict for FittedPls {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut out = Vector::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut s = self.y_mean;
            for j in 0..x.ncols().min(self.coef.len()) {
                s += (x.get(i, j) - self.x_mean[j]) * self.coef[j];
            }
            out[i] = s;
        }
        ctx.finish(out)
    }
}

/// Two-block PLS (sklearn `PLSCanonical`): NIPALS on `X` and a matrix `Y`.
#[derive(Clone, Debug)]
pub struct PlsCanonical {
    /// Latent directions.
    pub n_components: usize,
}

impl Default for PlsCanonical {
    fn default() -> Self {
        Self { n_components: 1 }
    }
}

impl PlsCanonical {
    /// `k` canonical PLS directions.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: n_components.max(1),
        }
    }

    /// Fit on two views. Do not pass `n_components` as identification `p`.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedPlsCanonical>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        if x.nrows() != y.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("PLSCanonical X rows ≠ Y rows")
                    .build(),
            );
            return ctx.finish(FittedPlsCanonical {
                x_weights: Matrix::zeros(x.ncols(), self.n_components),
                y_weights: Matrix::zeros(y.ncols(), self.n_components),
                x_mean: Vector::zeros(x.ncols()),
                y_mean: Vector::zeros(y.ncols()),
            });
        }
        let (mut xs, x_mean) = x.centered();
        let (mut ys, y_mean) = y.centered();
        let k = self
            .n_components
            .min(x.ncols())
            .min(y.ncols())
            .min(x.nrows().saturating_sub(1))
            .max(1);
        let mut xw = Matrix::zeros(x.ncols(), k);
        let mut yw = Matrix::zeros(y.ncols(), k);
        for c in 0..k {
            let mut u = Vector::from_iter((0..ys.ncols()).map(|j| ys.get(0, j)));
            if u.norm() <= ctx.policy.near_zero_variance {
                u = Vector::filled(ys.ncols(), 1.0);
            }
            let un = u.norm().max(1e-12);
            u = u.scale(1.0 / un);
            let mut w = Vector::zeros(xs.ncols());
            for _ in 0..8 {
                w = xs.matvec_t(&ys.matvec(&u));
                let wn = w.norm();
                if wn <= ctx.policy.near_zero_variance {
                    ctx.push(
                        Issue::builder(IssueCode::UpdateWithZeroInformation)
                            .message("PLSCanonical NIPALS weight vanished")
                            .build(),
                    );
                    break;
                }
                w = w.scale(1.0 / wn);
                let t = xs.matvec(&w);
                u = ys.matvec_t(&t);
                let un = u.norm();
                if un <= ctx.policy.near_zero_variance {
                    break;
                }
                u = u.scale(1.0 / un);
            }
            let t = xs.matvec(&w);
            let tt = t.dot(&t).max(1e-12);
            let pvec = xs.matvec_t(&t).scale(1.0 / tt);
            let qvec = ys.matvec_t(&t).scale(1.0 / tt);
            for j in 0..w.len() {
                xw.set(j, c, w[j]);
            }
            for j in 0..u.len() {
                yw.set(j, c, u[j]);
            }
            for i in 0..xs.nrows() {
                for j in 0..xs.ncols() {
                    xs.set(i, j, xs.get(i, j) - t[i] * pvec[j]);
                }
                for j in 0..ys.ncols() {
                    ys.set(i, j, ys.get(i, j) - t[i] * qvec[j]);
                }
            }
        }
        ctx.finish(FittedPlsCanonical {
            x_weights: xw,
            y_weights: yw,
            x_mean,
            y_mean,
        })
    }
}

/// Fitted two-block PLS.
#[derive(Clone, Debug)]
pub struct FittedPlsCanonical {
    /// `X` weights (`p_x` × `k`).
    pub x_weights: Matrix,
    /// `Y` weights (`p_y` × `k`).
    pub y_weights: Matrix,
    /// `X` column means.
    pub x_mean: Vector,
    /// `Y` column means.
    pub y_mean: Vector,
}

impl Transform for FittedPlsCanonical {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        let k = self.x_weights.ncols();
        let z = Matrix::from_fn(x.nrows(), k, |i, c| {
            let mut s = 0.0;
            for j in 0..x.ncols().min(self.x_weights.nrows()).min(self.x_mean.len()) {
                s += (x.get(i, j) - self.x_mean[j]) * self.x_weights.get(j, c);
            }
            s
        });
        ctx.finish(z)
    }
}

/// SVD of the cross-covariance `XᵀY` (sklearn `PLSSVD`).
///
/// Component count is not identification `p`.
#[derive(Clone, Debug)]
pub struct PlsSvd {
    /// Latent directions.
    pub n_components: usize,
}

impl Default for PlsSvd {
    fn default() -> Self {
        Self { n_components: 1 }
    }
}

impl PlsSvd {
    /// `k` SVD directions of `XᵀY`.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: n_components.max(1),
        }
    }

    /// Fit on two views.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedPlsCanonical>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        if x.nrows() != y.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("PLSSVD X rows ≠ Y rows")
                    .build(),
            );
            return ctx.finish(FittedPlsCanonical {
                x_weights: Matrix::zeros(x.ncols(), self.n_components),
                y_weights: Matrix::zeros(y.ncols(), self.n_components),
                x_mean: Vector::zeros(x.ncols()),
                y_mean: Vector::zeros(y.ncols()),
            });
        }
        let (xc, x_mean) = x.centered();
        let (yc, y_mean) = y.centered();
        let p = xc.ncols();
        let t = yc.ncols();
        let n = xc.nrows().min(yc.nrows());
        let cxy = Matrix::from_fn(p, t, |j, k| {
            let mut s = 0.0;
            for i in 0..n {
                s += xc.get(i, j) * yc.get(i, k);
            }
            s
        });
        let mut scratch = signlred::Report::new("plssvd", "svd");
        let Some(svd) = thin_svd(&mut scratch, &cxy, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::SvdDidNotConverge)
                    .message("PLSSVD of cross-covariance failed")
                    .build(),
            );
            return ctx.finish(FittedPlsCanonical {
                x_weights: Matrix::zeros(p, 1),
                y_weights: Matrix::zeros(t, 1),
                x_mean,
                y_mean,
            });
        };
        let k = self
            .n_components
            .max(1)
            .min(svd.singular_values.len())
            .min(p)
            .min(t);
        ctx.finish(FittedPlsCanonical {
            x_weights: Matrix::from_fn(p, k, |j, c| svd.u[(j, c)]),
            y_weights: Matrix::from_fn(t, k, |j, c| svd.v[(j, c)]),
            x_mean,
            y_mean,
        })
    }
}

/// Quantile regression via IRLS (asymmetric Laplace / weighted LS).
#[derive(Clone, Debug)]
pub struct QuantileRegressor {
    /// Quantile in (0, 1).
    pub q: f64,
    /// Max IRLS iterations.
    pub max_iter: usize,
    /// Intercept.
    pub fit_intercept: bool,
}

impl Default for QuantileRegressor {
    fn default() -> Self {
        Self {
            q: 0.5,
            max_iter: 40,
            fit_intercept: true,
        }
    }
}

impl QuantileRegressor {
    /// Median regression by default.
    pub fn new(q: f64) -> Self {
        Self {
            q,
            ..Self::default()
        }
    }
}

impl Fit for QuantileRegressor {
    type Fitted = FittedLinear;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinear>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if !(0.0..=1.0).contains(&self.q) || self.q == 0.0 || self.q == 1.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("quantile q={} is not in (0,1)", self.q))
                    .build(),
            );
        }
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        let Some(mut beta) = least_squares(&mut ctx.report, &design, y, &ctx.policy) else {
            return ctx.finish(empty_fitted(x, y, self.fit_intercept));
        };
        for it in 0..self.max_iter {
            let pred = design.matvec(&beta);
            let mut xs = Matrix::zeros(design.nrows(), design.ncols());
            let mut ys = Vector::zeros(y.len());
            for i in 0..y.len() {
                let r = y[i] - pred[i];
                // asymmetric weight; floor to avoid 1/0
                let w = if r >= 0.0 { self.q } else { 1.0 - self.q };
                let sw = (w / r.abs().max(1e-6)).sqrt();
                ys[i] = y[i] * sw;
                for j in 0..design.ncols() {
                    xs.set(i, j, design.get(i, j) * sw);
                }
            }
            let Some(next) = least_squares(&mut ctx.report, &xs, &ys, &ctx.policy) else {
                break;
            };
            let d = next.sub(&beta).norm();
            beta = next;
            ctx.session.step(it as u64, d, None);
            if d < 1e-8 {
                ctx.session.converged("quantile IRLS", it as u64);
                break;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("quantile IRLS SEs below are not the Koenker sandwich; treat them as a weighted-LS approximation")
                .build(),
        );
        let fitted = infer_ols(&mut ctx, &design, y, beta, self.fit_intercept);
        ctx.finish(fitted)
    }
}

/// Expectile regression via IRLS (asymmetric squared loss).
///
/// Weight is `τ` on positive residuals and `1−τ` on negative residuals.
/// Inner least-squares issues of the residual kind are not promoted.
#[derive(Clone, Debug)]
pub struct ExpectileRegressor {
    /// Expectile in (0, 1).
    pub tau: f64,
    /// Max IRLS iterations.
    pub max_iter: usize,
    /// Intercept.
    pub fit_intercept: bool,
}

impl Default for ExpectileRegressor {
    fn default() -> Self {
        Self {
            tau: 0.5,
            max_iter: 40,
            fit_intercept: true,
        }
    }
}

impl ExpectileRegressor {
    /// Expectile `tau`.
    pub fn new(tau: f64) -> Self {
        Self {
            tau,
            ..Self::default()
        }
    }
}

impl Fit for ExpectileRegressor {
    type Fitted = FittedLinear;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinear>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let tau = if self.tau.is_finite() && self.tau > 0.0 && self.tau < 1.0 {
            self.tau
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "expectile τ={} is not in (0,1); using 0.5",
                        self.tau
                    ))
                    .build(),
            );
            0.5
        };
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        let mut scratch = signlred::Report::new("expectile", "ols");
        let Some(mut beta) = least_squares(&mut scratch, &design, y, &ctx.policy) else {
            return ctx.finish(empty_fitted(x, y, self.fit_intercept));
        };
        for it in 0..self.max_iter {
            let pred = design.matvec(&beta);
            let mut xs = Matrix::zeros(design.nrows(), design.ncols());
            let mut ys = Vector::zeros(y.len());
            for i in 0..y.len() {
                let r = y[i] - pred[i];
                let w = if r >= 0.0 { tau } else { 1.0 - tau };
                let sw = w.max(1e-12).sqrt();
                ys[i] = y[i] * sw;
                for j in 0..design.ncols() {
                    xs.set(i, j, design.get(i, j) * sw);
                }
            }
            let mut inner = signlred::Report::new("expectile", "irls");
            let Some(next) = least_squares(&mut inner, &xs, &ys, &ctx.policy) else {
                break;
            };
            let d = next.sub(&beta).norm();
            beta = next;
            ctx.session.step(it as u64, d, None);
            if d < 1e-8 {
                ctx.session.converged("expectile IRLS", it as u64);
                break;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("expectile IRLS SEs are a weighted-LS approximation, not the expectile sandwich")
                .build(),
        );
        let fitted = infer_ols(&mut ctx, &design, y, beta, self.fit_intercept);
        ctx.finish(fitted)
    }
}

/// Poisson GLM (log link) via IRLS.
#[derive(Clone, Debug)]
pub struct PoissonRegressor {
    /// ℓ₂ penalty.
    pub alpha: f64,
    /// Max IRLS iterations.
    pub max_iter: usize,
    /// Intercept.
    pub fit_intercept: bool,
}

impl Default for PoissonRegressor {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            max_iter: 40,
            fit_intercept: true,
        }
    }
}

impl PoissonRegressor {
    /// Unregularized Poisson GLM.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for PoissonRegressor {
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
            if yi < 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NonPositiveSeries)
                        .message(format!("Poisson y[{i}]={yi} < 0"))
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
                let sw = mu.sqrt();
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
                ctx.session.converged("Poisson IRLS", it as u64);
                break;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("Poisson SEs below use the last weighted LS; they ignore the GLM variance function sandwich")
                .build(),
        );
        let fitted = infer_ols(&mut ctx, &design, y, beta, self.fit_intercept);
        ctx.finish(fitted)
    }
}

/// Dummy mean / majority baseline (sklearn Dummy*).
#[derive(Clone, Debug)]
pub struct DummyRegressor {
    /// Strategy: `"mean"` or `"median"`.
    pub strategy: DummyStrategy,
}

/// Dummy strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DummyStrategy {
    /// Predict the training mean.
    Mean,
    /// Predict the training median.
    Median,
}

impl Default for DummyRegressor {
    fn default() -> Self {
        Self {
            strategy: DummyStrategy::Mean,
        }
    }
}

/// Fitted dummy.
#[derive(Clone, Debug)]
pub struct FittedDummy {
    /// Constant prediction.
    pub value: f64,
}

impl Fit for DummyRegressor {
    type Fitted = FittedDummy;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedDummy>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::InterceptOnlyCollapse)
                .severity(signlred::Severity::Advisory)
                .message("DummyRegressor ignores X by construction")
                .build(),
        );
        let value = match self.strategy {
            DummyStrategy::Mean => y.mean(),
            DummyStrategy::Median => {
                let mut v = y.as_slice().to_vec();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if v.is_empty() {
                    f64::NAN
                } else {
                    v[v.len() / 2]
                }
            }
        };
        ctx.finish(FittedDummy { value })
    }
}

impl Predict for FittedDummy {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        ctx.finish(Vector::filled(x.nrows(), self.value))
    }
}

/// Least-angle regression (Efron, Hastie, Johnstone, Tibshirani).
///
/// The path moves in the equiangular direction of the active set until another
/// variable’s correlation catches up. Asking for more non-zeros than
/// \(\min(n-1, p)\) is overparameterized.
#[derive(Clone, Debug)]
pub struct Lars {
    /// Stop after this many variables (`None` ⇒ \(\min(n-1, p)\)).
    pub n_nonzero: Option<usize>,
    /// Center \(X\) and \(y\).
    pub fit_intercept: bool,
}

impl Default for Lars {
    fn default() -> Self {
        Self {
            n_nonzero: None,
            fit_intercept: true,
        }
    }
}

impl Lars {
    /// Full LARS path (capped at \(\min(n-1, p)\)).
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted LARS model (last path step).
#[derive(Clone, Debug)]
pub struct FittedLars {
    /// Slopes on the original (uncentered) scale.
    pub coef: Vector,
    /// Training mean of \(y\) (0 if `fit_intercept` is false).
    pub intercept: f64,
    /// Indices that entered the active set, in order.
    pub active: Vec<usize>,
}

impl Fit for Lars {
    type Fitted = FittedLars;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedLars>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if ctx.report.contains(IssueCode::ConstantTarget)
            || ctx.report.contains(IssueCode::EmptyMatrix)
        {
            return ctx.finish(FittedLars {
                coef: Vector::zeros(x.ncols()),
                intercept: y.mean(),
                active: Vec::new(),
            });
        }
        let (n, p) = x.shape();
        let (xc, xmean) = if self.fit_intercept {
            x.centered()
        } else {
            (x.clone(), Vector::zeros(p))
        };
        let ymean = if self.fit_intercept { y.mean() } else { 0.0 };
        let yc = Vector::from_iter(y.as_slice().iter().map(|&v| v - ymean));
        let mut beta = Vector::zeros(p);
        let mut mu = Vector::zeros(n);
        let mut active: Vec<usize> = Vec::new();
        let mut signs: Vec<f64> = Vec::new();
        let max_steps = self
            .n_nonzero
            .unwrap_or(p.min(n.saturating_sub(1)))
            .min(p)
            .min(n.saturating_sub(1));
        if self.n_nonzero.unwrap_or(0) > p.min(n.saturating_sub(1)) {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message("LARS n_nonzero exceeds min(n-1, p); the path is truncated")
                    .build(),
            );
        }
        for step in 0..max_steps {
            let resid = yc.sub(&mu);
            let corr = xc.matvec_t(&resid);
            let mut cmax = 0.0;
            let mut jnew = None;
            for j in 0..p {
                if active.contains(&j) {
                    continue;
                }
                if corr[j].abs() > cmax {
                    cmax = corr[j].abs();
                    jnew = Some(j);
                }
            }
            let Some(j) = jnew else {
                break;
            };
            if cmax <= 1e-14 {
                ctx.push(
                    Issue::builder(IssueCode::UpdateWithZeroInformation)
                        .severity(signlred::Severity::Advisory)
                        .message("LARS remaining correlations are ~0")
                        .build(),
                );
                break;
            }
            active.push(j);
            signs.push(if corr[j] >= 0.0 { 1.0 } else { -1.0 });
            let a_n = active.len();
            let xa = Matrix::from_fn(n, a_n, |i, c| xc.get(i, active[c]) * signs[c]);
            let ones = Vector::filled(a_n, 1.0);
            let mut gram = xa.gram();
            for i in 0..a_n {
                gram[(i, i)] += 1e-12;
            }
            let mut scratch = signlred::Report::new("lars", "gram");
            let Some(ginv1) = crate::linalg::chol_solve(&mut scratch, &gram, &ones, &ctx.policy)
            else {
                ctx.push(
                    Issue::builder(IssueCode::SingularMatrix)
                        .severity(signlred::Severity::Warning)
                        .message("LARS active Gram is singular; path stopped")
                        .compromise(NumericalCompromise::new(
                            "(X_Aᵀ X_A)^{-1} 1",
                            "path truncated at the last identified active set",
                            "the equiangular Gram is not SPD at working precision",
                            "later LARS steps are unidentified",
                        ))
                        .build(),
                );
                break;
            };
            let aa = ones.dot(&ginv1).max(1e-18).sqrt();
            let a_dir = 1.0 / aa;
            let w = ginv1.scale(a_dir);
            let u = xa.matvec(&w);
            let a_full = xc.matvec_t(&u);
            let mut gamma = cmax / a_dir.max(1e-18);
            for k in 0..p {
                if active.contains(&k) {
                    continue;
                }
                let den1 = a_dir - a_full[k];
                let den2 = a_dir + a_full[k];
                if den1.abs() > 1e-14 {
                    let g = (cmax - corr[k]) / den1;
                    if g > 1e-15 && g < gamma {
                        gamma = g;
                    }
                }
                if den2.abs() > 1e-14 {
                    let g = (cmax + corr[k]) / den2;
                    if g > 1e-15 && g < gamma {
                        gamma = g;
                    }
                }
            }
            for i in 0..n {
                mu[i] += gamma * u[i];
            }
            for (c, &jidx) in active.iter().enumerate() {
                beta[jidx] += gamma * signs[c] * w[c];
            }
            ctx.session.step(step as u64, resid.norm(), Some(cmax));
        }
        let mut intercept = ymean;
        if self.fit_intercept {
            for j in 0..p {
                intercept -= xmean[j] * beta[j];
            }
        }
        ctx.finish(FittedLars {
            coef: beta,
            intercept,
            active,
        })
    }
}

impl Predict for FittedLars {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("LARS predict column count ≠ coef")
                    .build(),
            );
        }
        let mut out = x.matvec(&self.coef);
        for i in 0..out.len() {
            out[i] += self.intercept;
        }
        ctx.finish(out)
    }
}

/// LARS-Lasso: the LARS path with the Efron sign-drop modification, stopped
/// when the equiangular correlation falls to `alpha`.
#[derive(Clone, Debug)]
pub struct LassoLars {
    /// Soft threshold on the LARS correlation (`0` ⇒ full path).
    pub alpha: f64,
    /// Center \(X\) and \(y\).
    pub fit_intercept: bool,
}

impl Default for LassoLars {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            fit_intercept: true,
        }
    }
}

impl LassoLars {
    /// LassoLars with correlation floor `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::default()
        }
    }
}

impl Fit for LassoLars {
    type Fitted = FittedLars;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedLars>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "LassoLars α={} is not a finite ≥0 value",
                        self.alpha
                    ))
                    .build(),
            );
        }
        if ctx.report.contains(IssueCode::ConstantTarget)
            || ctx.report.contains(IssueCode::EmptyMatrix)
        {
            return ctx.finish(FittedLars {
                coef: Vector::zeros(x.ncols()),
                intercept: y.mean(),
                active: Vec::new(),
            });
        }
        let (n, p) = x.shape();
        let (xc, xmean) = if self.fit_intercept {
            x.centered()
        } else {
            (x.clone(), Vector::zeros(p))
        };
        let ymean = if self.fit_intercept { y.mean() } else { 0.0 };
        let yc = Vector::from_iter(y.as_slice().iter().map(|&v| v - ymean));
        let mut beta = Vector::zeros(p);
        let mut mu = Vector::zeros(n);
        let mut active: Vec<usize> = Vec::new();
        let mut signs: Vec<f64> = Vec::new();
        let alpha = self.alpha.max(0.0);
        let max_steps = p.min(n.saturating_sub(1)).max(1);
        for step in 0..max_steps {
            let resid = yc.sub(&mu);
            let corr = xc.matvec_t(&resid);
            let mut cmax = 0.0;
            let mut jnew = None;
            for j in 0..p {
                if active.contains(&j) {
                    continue;
                }
                if corr[j].abs() > cmax {
                    cmax = corr[j].abs();
                    jnew = Some(j);
                }
            }
            if cmax <= alpha + 1e-14 {
                ctx.session
                    .converged("LassoLars correlation ≤ α", step as u64);
                break;
            }
            let Some(j) = jnew else {
                break;
            };
            active.push(j);
            signs.push(if corr[j] >= 0.0 { 1.0 } else { -1.0 });
            let a_n = active.len();
            let xa = Matrix::from_fn(n, a_n, |i, c| xc.get(i, active[c]) * signs[c]);
            let ones = Vector::filled(a_n, 1.0);
            let mut gram = xa.gram();
            for i in 0..a_n {
                gram[(i, i)] += 1e-12;
            }
            let mut scratch = signlred::Report::new("lassolars", "gram");
            let Some(ginv1) = crate::linalg::chol_solve(&mut scratch, &gram, &ones, &ctx.policy)
            else {
                ctx.push(
                    Issue::builder(IssueCode::SingularMatrix)
                        .severity(Severity::Warning)
                        .message("LassoLars active Gram is singular; path stopped")
                        .build(),
                );
                break;
            };
            let aa = ones.dot(&ginv1).max(1e-18).sqrt();
            let a_dir = 1.0 / aa;
            let w = ginv1.scale(a_dir);
            let u = xa.matvec(&w);
            let a_full = xc.matvec_t(&u);
            let mut gamma = cmax / a_dir.max(1e-18);
            for k in 0..p {
                if active.contains(&k) {
                    continue;
                }
                let den1 = a_dir - a_full[k];
                let den2 = a_dir + a_full[k];
                if den1.abs() > 1e-14 {
                    let g = (cmax - corr[k]) / den1;
                    if g > 1e-15 && g < gamma {
                        gamma = g;
                    }
                }
                if den2.abs() > 1e-14 {
                    let g = (cmax + corr[k]) / den2;
                    if g > 1e-15 && g < gamma {
                        gamma = g;
                    }
                }
            }
            // Lasso modification: drop a variable that would change sign.
            let mut drop = None;
            for (c, &jidx) in active.iter().enumerate() {
                let dir = signs[c] * w[c];
                if dir.abs() > 1e-15 {
                    let g = -beta[jidx] / dir;
                    if g > 1e-15 && g < gamma {
                        gamma = g;
                        drop = Some(c);
                    }
                }
            }
            for i in 0..n {
                mu[i] += gamma * u[i];
            }
            for (c, &jidx) in active.iter().enumerate() {
                beta[jidx] += gamma * signs[c] * w[c];
            }
            if let Some(c) = drop {
                beta[active[c]] = 0.0;
                active.remove(c);
                signs.remove(c);
            }
            ctx.session.step(step as u64, resid.norm(), Some(cmax));
        }
        let mut intercept = ymean;
        if self.fit_intercept {
            for j in 0..p {
                intercept -= xmean[j] * beta[j];
            }
        }
        ctx.finish(FittedLars {
            coef: beta,
            intercept,
            active,
        })
    }
}

/// Information criterion used by [`LassoLarsIc`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcKind {
    /// AIC: \(n\log(\mathrm{SSE}/n)+2k\).
    Aic,
    /// BIC: \(n\log(\mathrm{SSE}/n)+k\log n\).
    Bic,
}

/// LassoLars path scored by AIC/BIC (sklearn `LassoLarsIC`).
///
/// The grid is a documented compromise versus the exact LARS knots.
#[derive(Clone, Debug)]
pub struct LassoLarsIc {
    /// AIC or BIC.
    pub criterion: IcKind,
}

impl Default for LassoLarsIc {
    fn default() -> Self {
        Self {
            criterion: IcKind::Aic,
        }
    }
}

impl LassoLarsIc {
    /// AIC-scored LassoLars.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for LassoLarsIc {
    type Fitted = FittedLars;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedLars>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if ctx.report.contains(IssueCode::ConstantTarget) {
            return ctx.finish(FittedLars {
                coef: Vector::zeros(x.ncols()),
                intercept: y.mean(),
                active: Vec::new(),
            });
        }
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .severity(Severity::Advisory)
                .message("LassoLarsIC scores a coarse α grid, not the exact LARS knots")
                .compromise(NumericalCompromise::new(
                    "IC along the full LARS path",
                    "AIC/BIC on α ∈ {0, 0.01, 0.1, 1} × σ_y",
                    "the grid may skip the IC-optimal knot",
                    "treat the selected α as a discrete approximation",
                ))
                .build(),
        );
        let scale = y.std().max(1e-6);
        let alphas = [0.0, 0.01 * scale, 0.1 * scale, scale];
        let n = y.len().max(1) as f64;
        let mut best: Option<(f64, FittedLars)> = None;
        for (k, &a) in alphas.iter().enumerate() {
            match LassoLars::new(a).fit(x, y, &session.child(format!("llarsic_{k}"))) {
                Ok(q) => {
                    let pred = x.matvec(&q.value.coef);
                    let mut sse = 0.0;
                    for i in 0..y.len() {
                        let e = y[i] - (pred[i] + q.value.intercept);
                        sse += e * e;
                    }
                    let nnz = q
                        .value
                        .coef
                        .as_slice()
                        .iter()
                        .filter(|v| v.abs() > 1e-10)
                        .count() as f64
                        + 1.0;
                    let ic = match self.criterion {
                        IcKind::Aic => n * (sse / n).max(1e-15).ln() + 2.0 * nnz,
                        IcKind::Bic => n * (sse / n).max(1e-15).ln() + nnz * n.ln(),
                    };
                    if best.as_ref().map(|(b, _)| ic < *b).unwrap_or(true) {
                        best = Some((ic, q.value));
                    }
                }
                Err(_) => continue,
            }
        }
        let fitted = best.map(|(_, m)| m).unwrap_or(FittedLars {
            coef: Vector::zeros(x.ncols()),
            intercept: y.mean(),
            active: Vec::new(),
        });
        ctx.finish(fitted)
    }
}

/// Tweedie GLM with log link and variance \(\mu^p\) (sklearn `TweedieRegressor`).
///
/// \(p=0\) is Gaussian, \(p=1\) Poisson, \(p=2\) Gamma, \(1<p<2\) compound
/// Poisson–Gamma. A non-positive response with \(p \ge 1\) is not in the
/// support and aborts.
#[derive(Clone, Debug)]
pub struct TweedieRegressor {
    /// Variance power \(p\).
    pub power: f64,
    /// ℓ₂ penalty.
    pub alpha: f64,
    /// Max IRLS iterations.
    pub max_iter: usize,
    /// Intercept.
    pub fit_intercept: bool,
}

impl Default for TweedieRegressor {
    fn default() -> Self {
        Self {
            power: 1.5,
            alpha: 0.0,
            max_iter: 40,
            fit_intercept: true,
        }
    }
}

impl TweedieRegressor {
    /// Compound Poisson–Gamma Tweedie (\(p=1.5\)).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for TweedieRegressor {
    type Fitted = FittedLinear;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinear>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if self.power >= 1.0 {
            for (i, &yi) in y.as_slice().iter().enumerate() {
                if yi < 0.0 {
                    ctx.push(
                        Issue::builder(IssueCode::NonPositiveSeries)
                            .message(format!("Tweedie p={} forbids y[{i}]={yi} < 0", self.power))
                            .build(),
                    );
                    break;
                }
                if self.power >= 2.0 && yi <= 0.0 {
                    ctx.push(
                        Issue::builder(IssueCode::NonPositiveSeries)
                            .message(format!("Tweedie p={} forbids y[{i}]={yi} ≤ 0", self.power))
                            .build(),
                    );
                    break;
                }
            }
        }
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
        let mut beta = Vector::zeros(design.ncols());
        let ybar = y.mean().max(1e-6);
        if self.fit_intercept {
            beta[0] = ybar.ln();
        }
        let mut converged = false;
        for it in 0..self.max_iter {
            let mut xs = Matrix::zeros(design.nrows(), design.ncols());
            let mut z = Vector::zeros(y.len());
            for i in 0..y.len() {
                let mut eta = 0.0;
                for j in 0..design.ncols() {
                    eta += design.get(i, j) * beta[j];
                }
                let mu = eta.exp().max(1e-12);
                let var = mu.powf(self.power).max(1e-12);
                // log link: dμ/dη = μ, weight = μ² / var = μ^{2-p}
                let w = (mu * mu / var).max(1e-12);
                let sw = w.sqrt();
                z[i] = (eta + (y[i] - mu) / mu) * sw;
                for j in 0..design.ncols() {
                    xs.set(i, j, design.get(i, j) * sw);
                }
            }
            let mut scratch = signlred::Report::new("tweedie", "irls");
            let step_opt = if self.alpha > 0.0 {
                ridge_solve(&mut scratch, &xs, &z, self.alpha, &ctx.policy)
            } else {
                least_squares(&mut scratch, &xs, &z, &ctx.policy)
            };
            for issue in scratch.issues() {
                if issue.code == IssueCode::ResidualTooLarge {
                    continue;
                }
                ctx.push(issue.clone());
            }
            let Some(step) = step_opt else {
                break;
            };
            let delta = step.sub(&beta).norm();
            beta = step;
            ctx.session.step(it as u64, delta, None);
            if delta < 1e-8 {
                ctx.session.converged("Tweedie IRLS", it as u64);
                converged = true;
                break;
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("Tweedie IRLS did not meet the tolerance")
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message(
                    "Tweedie SEs below are the IRLS Gaussian approximation, not the GLM sandwich",
                )
                .build(),
        );
        let fitted = infer_ols(&mut ctx, &design, y, beta, self.fit_intercept);
        ctx.finish(fitted)
    }
}

/// Multi-task lasso: \(\|Y-XW\|_F^2 + \alpha\|W\|_{2,1}\) by block coordinate descent.
#[derive(Clone, Debug)]
pub struct MultiTaskLasso {
    /// Group-ℓ1 penalty on each feature's coefficient vector across tasks.
    pub alpha: f64,
    /// Coordinate cycles.
    pub max_iter: usize,
    /// Coordinate change tolerance.
    pub tol: f64,
}

impl Default for MultiTaskLasso {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            max_iter: 200,
            tol: 1e-6,
        }
    }
}

impl MultiTaskLasso {
    /// Multi-task lasso with the given group penalty.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::default()
        }
    }

    /// Fit `Y ~ X` for a multi-column response.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedMultiTask>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        if x.nrows() != y.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("MultiTaskLasso: Y rows ≠ X rows")
                    .build(),
            );
        }
        let n = x.nrows().min(y.nrows());
        let p = x.ncols();
        let t = y.ncols();
        if n == 0 || p == 0 || t == 0 {
            return ctx.finish(FittedMultiTask {
                coef: Matrix::zeros(p, t),
                intercept: Vector::zeros(t),
                alpha: self.alpha,
            });
        }
        let (xc, xmean) = x.centered();
        let mut ymean = Vector::zeros(t);
        let mut yc = Matrix::zeros(n, t);
        for j in 0..t {
            let col = y.column(j);
            ymean[j] = col.mean();
            for i in 0..n {
                yc.set(i, j, y.get(i, j) - ymean[j]);
            }
        }
        let mut col_norm2 = vec![0.0; p];
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                let v = xc.get(i, j);
                s += v * v;
            }
            col_norm2[j] = s;
        }
        let mut w = Matrix::zeros(p, t);
        let mut resid = yc.clone();
        let mut converged = false;
        let lam = (n as f64) * self.alpha.max(0.0);
        for it in 0..self.max_iter.max(1) {
            let mut max_d: f64 = 0.0;
            for j in 0..p {
                if col_norm2[j] <= ctx.policy.near_zero_variance {
                    continue;
                }
                for task in 0..t {
                    let wj = w.get(j, task);
                    for i in 0..n {
                        resid.set(i, task, resid.get(i, task) + xc.get(i, j) * wj);
                    }
                }
                let mut s_vec = Vector::zeros(t);
                for task in 0..t {
                    let mut rho = 0.0;
                    for i in 0..n {
                        rho += xc.get(i, j) * resid.get(i, task);
                    }
                    s_vec[task] = rho;
                }
                let nrm = s_vec.norm();
                let old: Vec<f64> = (0..t).map(|task| w.get(j, task)).collect();
                if nrm <= lam {
                    for task in 0..t {
                        w.set(j, task, 0.0);
                    }
                } else {
                    let scale = (1.0 - lam / nrm) / col_norm2[j];
                    for task in 0..t {
                        w.set(j, task, scale * s_vec[task]);
                    }
                }
                for task in 0..t {
                    let wj = w.get(j, task);
                    max_d = max_d.max((wj - old[task]).abs());
                    for i in 0..n {
                        resid.set(i, task, resid.get(i, task) - xc.get(i, j) * wj);
                    }
                }
            }
            ctx.session.step(it as u64, max_d, None);
            if max_d < self.tol {
                ctx.session.converged("multi-task CD", it as u64);
                converged = true;
                break;
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::MaxIterReached)
                    .message("MultiTaskLasso hit max_iter")
                    .build(),
            );
        }
        let all_zero = (0..p).all(|j| (0..t).all(|k| w.get(j, k).abs() == 0.0));
        let y_var = (0..t).any(|k| y.column(k).std() > ctx.policy.near_zero_variance);
        if all_zero && y_var {
            ctx.push(
                Issue::builder(IssueCode::InterceptOnlyCollapse)
                    .message("multi-task lasso shrank every feature to 0")
                    .meaninglessness(Meaninglessness::vacuous(
                        "multi-task slopes",
                        "the group penalty ate every direction",
                        "decrease α",
                    ))
                    .build(),
            );
        }
        let intercept = Vector::from_iter((0..t).map(|k| {
            let mut s = ymean[k];
            for j in 0..p {
                s -= xmean[j] * w.get(j, k);
            }
            s
        }));
        ctx.finish(FittedMultiTask {
            coef: w,
            intercept,
            alpha: self.alpha,
        })
    }
}

/// Fitted multi-task lasso.
#[derive(Clone, Debug)]
pub struct FittedMultiTask {
    /// Coefficient matrix (`p` × `n_tasks`).
    pub coef: Matrix,
    /// Intercept per task.
    pub intercept: Vector,
    /// Penalty used.
    pub alpha: f64,
}

impl FittedMultiTask {
    /// Predict a multi-column response.
    pub fn predict_matrix(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.ncols() != self.coef.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("MultiTaskLasso predict column count ≠ p")
                    .build(),
            );
        }
        let t = self.coef.ncols();
        let out = Matrix::from_fn(x.nrows(), t, |i, k| {
            let mut s = if k < self.intercept.len() {
                self.intercept[k]
            } else {
                0.0
            };
            for j in 0..x.ncols().min(self.coef.nrows()) {
                s += x.get(i, j) * self.coef.get(j, k);
            }
            s
        });
        ctx.finish(out)
    }
}

/// Multi-task elastic net: group-ℓ1 plus Frobenius ℓ2 on \(W\).
#[derive(Clone, Debug)]
pub struct MultiTaskElasticNet {
    /// Combined penalty.
    pub alpha: f64,
    /// Mixing: 1 = multi-task lasso, 0 = multi-task ridge.
    pub l1_ratio: f64,
    /// Coordinate cycles.
    pub max_iter: usize,
    /// Coordinate change tolerance.
    pub tol: f64,
}

impl Default for MultiTaskElasticNet {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            l1_ratio: 0.5,
            max_iter: 200,
            tol: 1e-6,
        }
    }
}

impl MultiTaskElasticNet {
    /// Multi-task elastic net with `alpha` and `l1_ratio`.
    pub fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self {
            alpha,
            l1_ratio,
            ..Self::default()
        }
    }

    /// Fit `Y ~ X` for a multi-column response.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedMultiTask>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "MultiTaskElasticNet l1_ratio={} not in [0, 1]; clamping",
                        self.l1_ratio
                    ))
                    .build(),
            );
        }
        if x.nrows() != y.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("MultiTaskElasticNet: Y rows ≠ X rows")
                    .build(),
            );
        }
        let n = x.nrows().min(y.nrows());
        let p = x.ncols();
        let t = y.ncols();
        if n == 0 || p == 0 || t == 0 {
            return ctx.finish(FittedMultiTask {
                coef: Matrix::zeros(p, t),
                intercept: Vector::zeros(t),
                alpha: self.alpha,
            });
        }
        let (xc, xmean) = x.centered();
        let mut ymean = Vector::zeros(t);
        let mut yc = Matrix::zeros(n, t);
        for j in 0..t {
            let col = y.column(j);
            ymean[j] = col.mean();
            for i in 0..n {
                yc.set(i, j, y.get(i, j) - ymean[j]);
            }
        }
        let mut col_norm2 = vec![0.0; p];
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                let v = xc.get(i, j);
                s += v * v;
            }
            col_norm2[j] = s;
        }
        let mut w = Matrix::zeros(p, t);
        let mut resid = yc.clone();
        let mut converged = false;
        let l1 = self.l1_ratio.clamp(0.0, 1.0);
        let lam1 = (n as f64) * self.alpha.max(0.0) * l1;
        let lam2 = (n as f64) * self.alpha.max(0.0) * (1.0 - l1);
        for it in 0..self.max_iter.max(1) {
            let mut max_d: f64 = 0.0;
            for j in 0..p {
                if col_norm2[j] <= ctx.policy.near_zero_variance {
                    continue;
                }
                for task in 0..t {
                    let wj = w.get(j, task);
                    for i in 0..n {
                        resid.set(i, task, resid.get(i, task) + xc.get(i, j) * wj);
                    }
                }
                let mut s_vec = Vector::zeros(t);
                for task in 0..t {
                    let mut rho = 0.0;
                    for i in 0..n {
                        rho += xc.get(i, j) * resid.get(i, task);
                    }
                    s_vec[task] = rho;
                }
                let nrm = s_vec.norm();
                let old: Vec<f64> = (0..t).map(|task| w.get(j, task)).collect();
                if nrm <= lam1 {
                    for task in 0..t {
                        w.set(j, task, 0.0);
                    }
                } else {
                    let scale = (1.0 - lam1 / nrm) / (col_norm2[j] + lam2);
                    for task in 0..t {
                        w.set(j, task, scale * s_vec[task]);
                    }
                }
                for task in 0..t {
                    let wj = w.get(j, task);
                    max_d = max_d.max((wj - old[task]).abs());
                    for i in 0..n {
                        resid.set(i, task, resid.get(i, task) - xc.get(i, j) * wj);
                    }
                }
            }
            ctx.session.step(it as u64, max_d, None);
            if max_d < self.tol {
                ctx.session
                    .converged("multi-task elastic-net CD", it as u64);
                converged = true;
                break;
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::MaxIterReached)
                    .message("MultiTaskElasticNet hit max_iter")
                    .build(),
            );
        }
        let intercept = Vector::from_iter((0..t).map(|k| {
            let mut s = ymean[k];
            for j in 0..p {
                s -= xmean[j] * w.get(j, k);
            }
            s
        }));
        ctx.finish(FittedMultiTask {
            coef: w,
            intercept,
            alpha: self.alpha,
        })
    }
}

/// Target map applied before a linear fit (sklearn `TransformedTargetRegressor`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetTransform {
    /// Identity (plain OLS).
    Identity,
    /// `log(y)` / `exp` inverse. Requires `y > 0`.
    Log,
}

/// OLS on a transformed target, with the inverse applied at predict time.
#[derive(Clone, Debug)]
pub struct TransformedTargetRegressor {
    /// Target map.
    pub transform: TargetTransform,
}

impl Default for TransformedTargetRegressor {
    fn default() -> Self {
        Self {
            transform: TargetTransform::Log,
        }
    }
}

impl TransformedTargetRegressor {
    /// Log-target OLS.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted transformed-target regressor.
#[derive(Clone, Debug)]
pub struct FittedTtr {
    /// Inner OLS on the transformed scale.
    pub inner: FittedLinear,
    /// Map that was applied.
    pub transform: TargetTransform,
}

impl Predict for FittedTtr {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let q = self.inner.predict(x, session)?;
        Ok(q.map(|mut y| {
            if self.transform == TargetTransform::Log {
                for i in 0..y.len() {
                    y[i] = y[i].exp();
                }
            }
            y
        }))
    }
}

impl Fit for TransformedTargetRegressor {
    type Fitted = FittedTtr;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedTtr>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let yt = match self.transform {
            TargetTransform::Identity => y.clone(),
            TargetTransform::Log => {
                for (i, &yi) in y.as_slice().iter().enumerate() {
                    if yi <= 0.0 {
                        ctx.push(
                            Issue::builder(IssueCode::NonPositiveSeries)
                                .message(format!("log-target y[{i}]={yi} is not strictly positive"))
                                .build(),
                        );
                        break;
                    }
                }
                Vector::from_iter(y.as_slice().iter().map(|v| v.max(1e-12).ln()))
            }
        };
        let mut ols = LinearRegression::new();
        let inner = match ols.fit(x, &yt, &session.child("inner")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(issue.code, IssueCode::ResidualTooLarge | IssueCode::R2IsOne) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                q.value
            }
            Err(e) => {
                ctx.push(e.primary);
                empty_fitted(x, y, true)
            }
        };
        ctx.finish(FittedTtr {
            inner,
            transform: self.transform,
        })
    }
}

/// Rolling-window OLS (statsmodels `RollingOLS`).
#[derive(Clone, Debug)]
pub struct RollingOls {
    /// Window length.
    pub window: usize,
    /// Prepend an intercept.
    pub fit_intercept: bool,
}

impl Default for RollingOls {
    fn default() -> Self {
        Self {
            window: 12,
            fit_intercept: true,
        }
    }
}

impl RollingOls {
    /// Rolling OLS with the given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
            fit_intercept: true,
        }
    }

    /// Fit a coefficient path, one row per complete window.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRollingOls>> {
        rolling_path(&mut *self, x, y, false, session)
    }
}

/// Expanding-window OLS (statsmodels `ExpandingOLS`).
#[derive(Clone, Debug)]
pub struct ExpandingOls {
    /// Minimum observations before the first estimate.
    pub min_n: usize,
    /// Prepend an intercept.
    pub fit_intercept: bool,
}

impl Default for ExpandingOls {
    fn default() -> Self {
        Self {
            min_n: 8,
            fit_intercept: true,
        }
    }
}

impl ExpandingOls {
    /// Expanding OLS with the given burn-in.
    pub fn new(min_n: usize) -> Self {
        Self {
            min_n: min_n.max(2),
            fit_intercept: true,
        }
    }

    /// Fit a coefficient path from `min_n` through `n`.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRollingOls>> {
        let mut roll = RollingOls {
            window: self.min_n,
            fit_intercept: self.fit_intercept,
        };
        rolling_path(&mut roll, x, y, true, session)
    }
}

/// Brown–Durbin–Evans recursive residuals (statsmodels `RecursiveLS`).
///
/// Each prefix OLS uses a scratch report. Prefix length is not passed as
/// identification `p` on short windows.
#[derive(Clone, Debug)]
pub struct RecursiveLs {
    /// Minimum observations before the first residual.
    pub min_n: usize,
}

impl Default for RecursiveLs {
    fn default() -> Self {
        Self { min_n: 8 }
    }
}

impl RecursiveLs {
    /// Recursive residuals with the given burn-in.
    pub fn new(min_n: usize) -> Self {
        Self {
            min_n: min_n.max(3),
        }
    }

    /// Fit the recursive residual path.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRecursiveLs>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let n = x.nrows().min(y.len());
        let p = x.ncols() + 1;
        let start = self.min_n.max(p + 1).min(n.max(1));
        if n < start {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("RecursiveLS burn-in {start} > n={n}"))
                    .build(),
            );
        }
        if n > 5 * p {
            inspect_identification(&mut ctx.report, n, p, &ctx.policy);
        }
        let mut resid = Vector::zeros(n.saturating_sub(start.saturating_sub(1)));
        let mut last = Vector::zeros(x.ncols());
        let mut last_int = y.mean();
        let mut k = 0usize;
        for end in start..=n {
            let xt = Matrix::from_fn(end, x.ncols(), |i, j| x.get(i, j));
            let yt = Vector::from_iter((0..end).map(|i| y[i]));
            let design = xt.with_intercept();
            let mut scratch = signlred::Report::new("rls", "ols");
            let Some(beta) = least_squares(&mut scratch, &design, &yt, &ctx.policy) else {
                continue;
            };
            last_int = beta.as_slice().first().copied().unwrap_or(0.0);
            last = Vector::from_iter((1..beta.len()).map(|j| beta[j]));
            if end < n {
                let mut yhat = last_int;
                for j in 0..x.ncols().min(last.len()) {
                    yhat += last[j] * x.get(end, j);
                }
                if k < resid.len() {
                    resid[k] = y[end] - yhat;
                    k += 1;
                }
            }
        }
        if k < resid.len() {
            resid = Vector::from_iter(resid.as_slice().iter().copied().take(k));
        }
        ctx.finish(FittedRecursiveLs {
            coef: last,
            intercept: last_int,
            resid,
        })
    }
}

/// Fitted recursive least squares.
#[derive(Clone, Debug)]
pub struct FittedRecursiveLs {
    /// Last expanding-window slopes.
    pub coef: Vector,
    /// Last intercept.
    pub intercept: f64,
    /// One-step recursive residuals after burn-in.
    pub resid: Vector,
}

/// Coefficient path from a rolling or expanding OLS.
#[derive(Clone, Debug)]
pub struct FittedRollingOls {
    /// Slopes (`n_windows` × `p`).
    pub coef: Matrix,
    /// Intercept per window.
    pub intercept: Vector,
    /// In-window R².
    pub r2: Vector,
    /// Window (or burn-in) length.
    pub window: usize,
}

fn rolling_path(
    spec: &mut RollingOls,
    x: &Matrix,
    y: &Vector,
    expanding: bool,
    session: &Session,
) -> Result<Qualified<FittedRollingOls>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let n = x.nrows().min(y.len());
    let w = spec.window.max(2);
    if n < w {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("rolling/expanding window {w} > n={n}"))
                .build(),
        );
    }
    let p_design = x.ncols() + if spec.fit_intercept { 1 } else { 0 };
    let ends: Vec<usize> = (w..=n).collect();
    let n_win = ends.len();
    let mut coef = Matrix::zeros(n_win, x.ncols());
    let mut intercept = Vector::zeros(n_win);
    let mut r2v = Vector::zeros(n_win);
    for (k, &end) in ends.iter().enumerate() {
        let start = if expanding { 0 } else { end.saturating_sub(w) };
        let nn = end.saturating_sub(start);
        if nn == 0 {
            continue;
        }
        let xs = Matrix::from_fn(nn, x.ncols(), |i, j| x.get(start + i, j));
        let ys = Vector::from_iter((0..nn).map(|i| y[start + i]));
        let design = if spec.fit_intercept {
            xs.with_intercept()
        } else {
            xs.clone()
        };
        if (nn as f64) < ctx.policy.min_samples_per_parameter * p_design as f64 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(signlred::Severity::Warning)
                    .message(format!("window [{start},{end}) has n={nn} p={p_design}"))
                    .build(),
            );
        }
        let mut scratch = signlred::Report::new("rolling", "ols");
        let Some(beta) = least_squares(&mut scratch, &design, &ys, &ctx.policy) else {
            continue;
        };
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::PerfectCollinearity
                    | IssueCode::R2IsOne
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let (b0, slopes) = if spec.fit_intercept {
            (
                beta.as_slice().first().copied().unwrap_or(0.0),
                Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            )
        } else {
            (0.0, beta.clone())
        };
        intercept[k] = b0;
        for j in 0..slopes.len().min(x.ncols()) {
            coef.set(k, j, slopes[j]);
        }
        let fit = design.matvec(&beta);
        let resid = ys.sub(&fit);
        let sse = resid.dot(&resid);
        let ym = ys.mean();
        let sst: f64 = ys.as_slice().iter().map(|v| (v - ym) * (v - ym)).sum();
        r2v[k] = if sst > ctx.policy.r2_zero_tol {
            1.0 - sse / sst
        } else {
            f64::NAN
        };
    }
    if n_win == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("no complete rolling window")
                .build(),
        );
    }
    ctx.finish(FittedRollingOls {
        coef,
        intercept,
        r2: r2v,
        window: w,
    })
}

/// Cochrane–Orcutt GLS with an AR(1) residual (statsmodels `GLSAR`).
#[derive(Clone, Debug)]
pub struct Glsar {
    /// Prepend an intercept.
    pub fit_intercept: bool,
    /// ρ / β iterations.
    pub max_iter: usize,
}

impl Default for Glsar {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            max_iter: 8,
        }
    }
}

impl Glsar {
    /// Default Cochrane–Orcutt GLSAR.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted GLSAR.
#[derive(Clone, Debug)]
pub struct FittedGlsar {
    /// Slopes on the original scale.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
    /// Estimated AR(1) residual coefficient.
    pub rho: f64,
    /// Innovation variance after the AR(1) filter.
    pub sigma2: f64,
}

impl Predict for FittedGlsar {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("GLSAR predict column count ≠ coef")
                    .build(),
            );
        }
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            y[i] += self.intercept;
        }
        ctx.finish(y)
    }
}

impl Fit for Glsar {
    type Fitted = FittedGlsar;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedGlsar>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
        let n = y.len().min(design.nrows());
        if n < 3 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("GLSAR needs n≥3")
                    .build(),
            );
            return ctx.finish(FittedGlsar {
                coef: Vector::zeros(x.ncols()),
                intercept: y.mean(),
                rho: 0.0,
                sigma2: f64::NAN,
            });
        }
        let mut scratch = signlred::Report::new("glsar", "ols0");
        let mut beta = least_squares(&mut scratch, &design, y, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for issue in scratch.issues() {
            if issue.code == IssueCode::ResidualTooLarge {
                continue;
            }
            ctx.push(issue.clone());
        }
        let mut rho = 0.0;
        for it in 0..self.max_iter.max(1) {
            let fitted = design.matvec(&beta);
            let e = y.sub(&fitted);
            let mut num = 0.0;
            let mut den = 0.0;
            for t in 1..n {
                num += e[t] * e[t - 1];
                den += e[t - 1] * e[t - 1];
            }
            rho = if den > ctx.policy.near_zero_variance {
                (num / den).clamp(-0.99, 0.99)
            } else {
                0.0
            };
            let n2 = n.saturating_sub(1);
            let xs = Matrix::from_fn(n2, design.ncols(), |i, j| {
                design.get(i + 1, j) - rho * design.get(i, j)
            });
            let ys = Vector::from_iter((1..n).map(|i| y[i] - rho * y[i - 1]));
            let mut step = signlred::Report::new("glsar", "co");
            let Some(next) = least_squares(&mut step, &xs, &ys, &ctx.policy) else {
                break;
            };
            for issue in step.issues() {
                if matches!(
                    issue.code,
                    IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
                ) {
                    continue;
                }
                ctx.push(issue.clone());
            }
            let d = next.sub(&beta).norm();
            beta = next;
            ctx.session.step(it as u64, d, Some(rho.abs()));
            if d < 1e-8 {
                ctx.session.converged("Cochrane–Orcutt", it as u64);
                break;
            }
        }
        if rho.abs() > 0.2 {
            ctx.push(
                Issue::builder(IssueCode::AutocorrelatedResiduals)
                    .message(format!("GLSAR ρ={rho:.4}"))
                    .metric("rho", rho)
                    .build(),
            );
        }
        let fitted = design.matvec(&beta);
        let e = y.sub(&fitted);
        let mut sse = 0.0;
        for t in 1..n {
            let u = e[t] - rho * e[t - 1];
            sse += u * u;
        }
        let df = (n as f64 - 1.0) - design.ncols() as f64;
        let sigma2 = if df > 0.0 { sse / df } else { f64::NAN };
        let (intercept, coef) = if self.fit_intercept {
            (
                beta.as_slice().first().copied().unwrap_or(0.0),
                Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            )
        } else {
            (0.0, beta)
        };
        ctx.finish(FittedGlsar {
            coef,
            intercept,
            rho,
            sigma2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    #[test]
    fn ols_line_and_inference() {
        let x = Matrix::from_fn(10, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..10).map(|i| 1.0 + 2.0 * i as f64));
        let session = Session::new("ols", "fit");
        let q = LinearRegression::new().fit(&x, &y, &session).expect("ols");
        assert!((q.value.intercept - 1.0).abs() < 1e-8);
        assert!((q.value.coef[0] - 2.0).abs() < 1e-8);
        assert!(q.value.r2 > 0.999);
        assert!(session.ledger().len() > 0);
    }

    #[test]
    fn constant_target_aborts() {
        let x = Matrix::from_fn(8, 2, |i, j| (i + j) as f64);
        let y = Vector::filled(8, 3.0);
        let session = Session::new("ols", "fit");
        let err = LinearRegression::new().fit(&x, &y, &session).unwrap_err();
        assert_eq!(err.primary().code, IssueCode::ConstantTarget);
        assert!(err.primary().meaninglessness.is_some());
    }

    #[test]
    fn sgd_explains_updates() {
        let x = Matrix::from_fn(6, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..6).map(|i| 2.0 * i as f64));
        let session = Session::new("sgd", "partial_fit");
        let mut m = SgdRegressor {
            learning_rate: 0.05,
            ..SgdRegressor::default()
        };
        let q = m.partial_fit(&x, Some(&y), &session).expect("pf");
        assert!(!q.value.narrative.is_empty());
        assert!(session
            .ledger()
            .events()
            .iter()
            .any(|e| e.kind == ojizou_san::EventKind::IncrementalExplanation));
    }

    #[test]
    fn softmax_logistic_three_classes() {
        let x = Matrix::from_fn(30, 2, |i, j| {
            let g = (i / 10) as f64;
            if j == 0 {
                3.0 * g + 0.25 * ((i % 10) as f64 - 4.5)
            } else {
                2.0 * g + 0.15 * ((i % 10) as f64 - 4.5)
            }
        });
        let y = Vector::from_iter((0..30).map(|i| (i / 10) as f64));
        let q = LogisticRegression::new()
            .fit(&x, &y, &Session::new("logit", "fit"))
            .expect("mn");
        assert!(q.value.softmax.is_some());
        assert_eq!(q.value.classes.len(), 3);
        let pred = q
            .value
            .predict(&x, &Session::new("logit", "p"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..30 {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 22, "ok={ok}");
    }

    #[test]
    fn lars_recovers_sparse_line() {
        let x = Matrix::from_fn(20, 3, |i, j| {
            if j == 0 {
                i as f64
            } else {
                0.01 * (i + j) as f64
            }
        });
        let y = Vector::from_iter((0..20).map(|i| 2.0 * i as f64));
        let q = Lars::new()
            .fit(&x, &y, &Session::new("lars", "fit"))
            .expect("lars");
        assert!(!q.value.active.is_empty());
        let pred = q
            .value
            .predict(&x, &Session::new("lars", "p"))
            .unwrap()
            .value;
        let mut sse = 0.0;
        for i in 0..y.len() {
            let e = pred[i] - y[i];
            sse += e * e;
        }
        assert!(
            sse / (y.len() as f64) < 2.0,
            "mse={}",
            sse / (y.len() as f64)
        );
    }

    #[test]
    fn rolling_glsar_ttr_and_multitask() {
        let x = Matrix::from_fn(24, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..24).map(|i| 1.0 + 2.0 * i as f64));
        let roll = RollingOls::new(12)
            .fit(&x, &y, &Session::new("roll", "fit"))
            .expect("roll");
        assert!(roll.value.coef.nrows() >= 10);
        assert!((roll.value.coef.get(roll.value.coef.nrows() - 1, 0) - 2.0).abs() < 0.05);
        let exp = ExpandingOls::new(12)
            .fit(&x, &y, &Session::new("exp", "fit"))
            .expect("exp");
        assert!(exp.value.coef.nrows() >= 10);
        let mut e = y.clone();
        for i in 1..e.len() {
            e[i] += 0.4 * (e[i] - 1.0 - 2.0 * (i as f64 - 1.0));
        }
        let g = Glsar::new()
            .fit(&x, &e, &Session::new("glsar", "fit"))
            .expect("glsar");
        assert!((g.value.coef[0] - 2.0).abs() < 0.5, "b={}", g.value.coef[0]);
        let ylog = Vector::from_iter((0..24).map(|i| (1.0 + 0.1 * i as f64).exp()));
        let ttr = TransformedTargetRegressor::new()
            .fit(&x, &ylog, &Session::new("ttr", "fit"))
            .expect("ttr");
        let pred = ttr
            .value
            .predict(&x, &Session::new("ttr", "p"))
            .unwrap()
            .value;
        assert!((pred[0] - ylog[0]).abs() / ylog[0] < 0.2);
        let ym = Matrix::from_fn(24, 2, |i, k| {
            if k == 0 {
                1.0 + 2.0 * i as f64
            } else {
                0.5 * i as f64
            }
        });
        let mt = MultiTaskLasso::new(0.01)
            .fit(&x, &ym, &Session::new("mt", "fit"))
            .expect("mt");
        assert_eq!(mt.value.coef.shape(), (1, 2));
        assert!(mt.value.coef.get(0, 0).is_finite());
        let ll = LassoLars::new(0.0)
            .fit(&x, &y, &Session::new("llars", "fit"))
            .expect("llars");
        assert!(ll.value.coef.as_slice().iter().any(|v| v.abs() > 1e-6));
        let mte = MultiTaskElasticNet::new(0.01, 0.5)
            .fit(&x, &ym, &Session::new("mte", "fit"))
            .expect("mte");
        assert_eq!(mte.value.coef.shape(), (1, 2));
        let ic = LassoLarsIc::new()
            .fit(&x, &y, &Session::new("llic", "fit"))
            .expect("llic");
        assert!(ic.value.coef.as_slice().iter().any(|v| v.abs() > 1e-6));
        let ym2 = Matrix::from_fn(24, 2, |i, k| {
            if k == 0 {
                1.0 + 2.0 * i as f64
            } else {
                0.5 * i as f64
            }
        });
        let pls = PlsCanonical::new(1)
            .fit(&x, &ym2, &Session::new("plsc", "fit"))
            .expect("plsc");
        let z = pls
            .value
            .transform(&x, &Session::new("plsc", "t"))
            .expect("plst")
            .value;
        assert_eq!(z.ncols(), 1);
        assert!(z.get(0, 0).is_finite());
        let mut psvd = PlsSvd::new(1);
        let sv = psvd
            .fit(&x, &ym2, &Session::new("plssvd", "fit"))
            .expect("plssvd");
        assert_eq!(sv.value.x_weights.ncols(), 1);
        let rls = RecursiveLs::new(10)
            .fit(&x, &y, &Session::new("rls", "fit"))
            .expect("rls");
        assert!(rls.value.coef[0].is_finite());
        assert!(!rls.value.resid.is_empty());
    }

    #[test]
    fn tweedie_positive_response() {
        let x = Matrix::from_fn(16, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..16).map(|i| (1.0 + 0.2 * i as f64).exp()));
        let q = TweedieRegressor {
            power: 1.5,
            max_iter: 30,
            ..TweedieRegressor::default()
        }
        .fit(&x, &y, &Session::new("tw", "fit"))
        .expect("tweedie");
        assert!(q.value.coef[0].is_finite());
        let xe = Matrix::from_fn(24, 1, |i, _| i as f64);
        let ye = Vector::from_iter((0..24).map(|i| 2.0 * i as f64 + 0.1 * ((i % 3) as f64)));
        let ex = ExpectileRegressor::new(0.7)
            .fit(&xe, &ye, &Session::new("expc", "fit"))
            .expect("expectile");
        assert!(ex.value.coef[0].is_finite());
    }
}
