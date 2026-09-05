//! Oracle-backed ordinary least squares and its inference kernel.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{least_squares_with_diagnostics, ThinSvd};
use crate::special::{f_pvalue, student_t_pvalue};
use crate::traits::{Fit, Predict};
use crate::validate::{
    inspect_collinearity, inspect_identification, inspect_xy, inspect_xy_allow_constant_target,
};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result};

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
    /// Raw prediction after callers have checked the feature shape.
    #[doc(hidden)]
    pub fn predict_vec(&self, x: &Matrix) -> Vector {
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
        if ctx
            .report
            .issues()
            .iter()
            .any(|issue| ctx.policy.must_abort(issue))
        {
            return Err(ctx.finish_failure());
        }
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
        self.fit_with_policy(x, y, &signlred::Policy::default(), session)
    }
}

impl LinearRegression {
    /// Fit OLS using an explicit quality policy without mutating the estimator.
    pub fn fit_with_policy(
        &self,
        x: &Matrix,
        y: &Vector,
        policy: &signlred::Policy,
        session: &Session,
    ) -> Result<Qualified<FittedLinear>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.policy = policy.clone();
        if self.fit_intercept {
            inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        } else {
            inspect_xy_allow_constant_target(&mut ctx.report, x, y, &ctx.policy);
        }
        if (self.fit_intercept && ctx.report.contains(IssueCode::ConstantTarget))
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
        let Some(solution) =
            least_squares_with_diagnostics(&mut ctx.report, &design, y, &ctx.policy)
        else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("OLS factorization produced no coefficient vector")
                    .build(),
            );
            return ctx.finish(empty_fitted(x, y, self.fit_intercept));
        };
        let fitted = infer_ols(
            &mut ctx,
            &design,
            y,
            solution.coefficients,
            &solution.decomposition,
            solution.rank,
            self.fit_intercept,
        );
        ctx.finish(fitted)
    }
}

/// Shared compatibility helper for legacy regression adapters.
#[doc(hidden)]
pub fn empty_fitted(x: &Matrix, y: &Vector, used_intercept: bool) -> FittedLinear {
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
        se: Vector::filled(p, f64::NAN),
        t_values: Vector::filled(p, f64::NAN),
        p_values: Vector::filled(p, f64::NAN),
        aic: f64::NAN,
        bic: f64::NAN,
        f_stat: f64::NAN,
        f_pvalue: f64::NAN,
        durbin_watson: f64::NAN,
        loglik: f64::NAN,
        fitted: Vector::zeros(y.len()),
        resid: Vector::zeros(y.len()),
        leverage: Vector::filled(y.len(), f64::NAN),
        cooks: Vector::filled(y.len(), f64::NAN),
        used_intercept,
    }
}

/// Compatibility hook for existing diagnostic replay; prefer `LinearModel::fit`.
#[doc(hidden)]
pub fn infer_ols(
    ctx: &mut FitCtx,
    design: &Matrix,
    y: &Vector,
    beta: Vector,
    decomposition: &ThinSvd,
    rank: usize,
    used_intercept: bool,
) -> FittedLinear {
    infer_linear(
        ctx,
        design,
        y,
        beta,
        Some((decomposition, rank)),
        used_intercept,
    )
}

/// Shared compatibility helper for legacy regression adapters.
#[doc(hidden)]
pub fn infer_point_fit(
    ctx: &mut FitCtx,
    design: &Matrix,
    y: &Vector,
    beta: Vector,
    used_intercept: bool,
) -> FittedLinear {
    infer_linear(ctx, design, y, beta, None, used_intercept)
}

fn infer_linear(
    ctx: &mut FitCtx,
    design: &Matrix,
    y: &Vector,
    beta: Vector,
    ols: Option<(&ThinSvd, usize)>,
    used_intercept: bool,
) -> FittedLinear {
    let n = design.nrows();
    let p = design.ncols();
    let rank = ols.map_or(p, |(_, rank)| rank);
    if ols.is_some() {
        ctx.report.set_n_parameters(rank);
    }
    let fittedv = design.matvec(&beta);
    let resid = y.sub(&fittedv);
    let sse = resid.dot(&resid);
    let sst = if used_intercept {
        let y_mean = y.mean();
        y.as_slice()
            .iter()
            .map(|yi| {
                let difference = yi - y_mean;
                difference * difference
            })
            .sum::<f64>()
    } else {
        y.dot(y)
    };
    let df = if ols.is_some() {
        n as f64 - rank as f64
    } else {
        f64::NAN
    };
    if ols.is_some() && df <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegreesOfFreedomNonPositive)
                .message(format!("n={n} numerical_rank={rank} ⇒ df_resid={df}"))
                .metric("df_resid", df)
                .meaninglessness(Meaninglessness::vacuous(
                    "σ², SEs, t, p, AIC",
                    "residual degrees of freedom are not positive; the model interpolated or is unidentified",
                    "reduce p or collect data; do not publish p-values",
                ))
                .build(),
        );
    }
    let sigma2 = if ols.is_some() && df > 0.0 {
        sse / df
    } else {
        f64::NAN
    };
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
        if ols.is_some() && df <= 0.0 {
            b = b.meaninglessness(Meaninglessness::vacuous(
                "OLS R² and coefficient t-tests",
                "the model interpolated (df_resid ≤ 0); in-sample R²=1 is tautological",
                "reduce p or collect data; do not publish in-sample skill",
            ));
        } else {
            b = b.message("R² is 1: fitted values interpolate the response; in-sample skill is not confirmatory.");
        }
        ctx.push(b.build());
    }
    if r2.is_finite() && r2.abs() <= ctx.policy.r2_zero_tol {
        let reference = if used_intercept {
            "the mean-only model"
        } else {
            "the zero-response model"
        };
        ctx.push(
            Issue::builder(IssueCode::R2IsZero)
                .message(format!("R² is 0; the fitted model matches {reference}"))
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
    let adj_r2 = if ols.is_some() && df > 0.0 && sst > 0.0 {
        let total_degrees = n as f64 - if used_intercept { 1.0 } else { 0.0 };
        1.0 - (1.0 - r2) * total_degrees / df
    } else {
        f64::NAN
    };
    let classical_inference_available =
        ols.is_some() && rank == p && df > 0.0 && sigma2.is_finite() && sigma2 > 0.0;
    if !classical_inference_available && !ctx.report.contains(IssueCode::PValueUnreliable) {
        let reason = if ols.is_none() {
            "this estimator is not Gaussian OLS"
        } else if rank < p {
            "the design is not full column rank"
        } else if df <= 0.0 {
            "residual degrees of freedom are not positive"
        } else {
            "the residual mean square is not positive and finite"
        };
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message(format!(
                    "classical OLS SEs, tests, and Cook's distances withheld: {reason}"
                ))
                .build(),
        );
    }
    let (se, t_values, p_values) = match ols {
        Some((decomposition, _)) if classical_inference_available => {
            ols_se(ctx, decomposition, &beta, sigma2, df)
        }
        _ => (
            Vector::filled(p, f64::NAN),
            Vector::filled(p, f64::NAN),
            Vector::filled(p, f64::NAN),
        ),
    };
    let explained_sum_squares = sst - sse;
    let df_model = rank as f64 - if used_intercept { 1.0 } else { 0.0 };
    let f_stat = if classical_inference_available && df_model > 0.0 && explained_sum_squares >= 0.0
    {
        (explained_sum_squares / df_model) / sigma2
    } else {
        f64::NAN
    };
    let f_p = if f_stat.is_finite() {
        f_pvalue(f_stat, df_model, df)
    } else {
        f64::NAN
    };
    let maximum_likelihood_variance = if ols.is_some() && n > 0 {
        sse / n as f64
    } else {
        f64::NAN
    };
    let loglik = if maximum_likelihood_variance.is_finite() && maximum_likelihood_variance > 0.0 {
        -0.5 * n as f64 * ((2.0 * std::f64::consts::PI * maximum_likelihood_variance).ln() + 1.0)
    } else if maximum_likelihood_variance == 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };
    let aic = if !loglik.is_nan() {
        -2.0 * loglik + 2.0 * rank as f64
    } else {
        f64::NAN
    };
    let bic = if !loglik.is_nan() {
        -2.0 * loglik + (rank as f64) * (n as f64).ln()
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
                    "Durbin–Watson={dw:.3}; independence-based inference is not credible"
                ))
                .metric("durbin_watson", dw)
                .build(),
        );
    }
    let (leverage, cooks) = match ols {
        Some((decomposition, _)) => hat_and_cook(
            decomposition,
            rank,
            &resid,
            sigma2,
            classical_inference_available,
        ),
        None => (Vector::filled(n, f64::NAN), Vector::filled(n, f64::NAN)),
    };
    let max_h = leverage
        .as_slice()
        .iter()
        .copied()
        .fold(0.0f64, |a, b| a.max(b));
    let h_cut = 2.0 * rank as f64 / n as f64;
    if ols.is_some() && max_h > h_cut && n > rank {
        ctx.push(
            Issue::builder(IssueCode::LeveragePoint)
                .message(format!("max leverage {max_h:.4} > 2·rank/n={h_cut:.4}"))
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
                    "linear predictor",
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

/// Shared compatibility helper for legacy regression adapters.
#[doc(hidden)]
pub fn split_beta(beta: &Vector, used_intercept: bool) -> (f64, Vector) {
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
    decomposition: &ThinSvd,
    beta: &Vector,
    sigma2: f64,
    df: f64,
) -> (Vector, Vector, Vector) {
    let p = beta.len();
    let mut se = Vector::filled(p, f64::NAN);
    let mut t = Vector::filled(p, f64::NAN);
    let mut pv = Vector::filled(p, f64::NAN);
    let mut diag = vec![0.0; p];
    for (j, diagonal) in diag.iter_mut().enumerate() {
        for k in 0..p {
            let scaled_loading = decomposition.v[(j, k)] / decomposition.singular_values[k];
            *diagonal += scaled_loading * scaled_loading;
        }
    }
    if diag.iter().any(|value| !value.is_finite() || *value < 0.0) {
        ctx.push(
            Issue::builder(IssueCode::InformationMatrixSingular)
                .message("SVD covariance diagonal is non-finite")
                .compromise(NumericalCompromise::new(
                    "diag((XᵀX)⁻¹)",
                    "SEs left as NaN",
                    "inverse singular-value scaling overflowed",
                    "do not publish coefficient tests",
                ))
                .build(),
        );
        return (se, t, pv);
    }
    for j in 0..p {
        let coefficient_variance = sigma2 * diag[j];
        let v = if coefficient_variance >= 0.0 {
            coefficient_variance.sqrt()
        } else {
            f64::NAN
        };
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

fn hat_and_cook(
    decomposition: &ThinSvd,
    rank: usize,
    resid: &Vector,
    sigma2: f64,
    cook_available: bool,
) -> (Vector, Vector) {
    let n = resid.len();
    let mut lev = Vector::filled(n, f64::NAN);
    let mut cooks = Vector::filled(n, f64::NAN);
    if n == 0 || rank == 0 {
        return (lev, cooks);
    }
    // h_ii = sum_j U_ij² over the retained numerical column space.
    for i in 0..n {
        let mut h = 0.0;
        for k in 0..rank {
            let v = decomposition.u[(i, k)];
            h += v * v;
        }
        lev[i] = h;
        if cook_available {
            let numerator = resid[i] * resid[i] * h;
            let one_minus_leverage = 1.0 - h;
            let denominator = sigma2 * rank as f64 * one_minus_leverage * one_minus_leverage;
            cooks[i] = if denominator > 0.0 {
                numerator / denominator
            } else if numerator == 0.0 {
                f64::NAN
            } else {
                f64::INFINITY
            };
        }
    }
    (lev, cooks)
}
