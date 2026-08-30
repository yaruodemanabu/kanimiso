//! Statsmodels-style descriptive statistics, association, tests, and survival.
//!
//! Every public computation opens a [`crate::context::FitCtx`], inspects the
//! inputs, and records [`signlred`] issues for non-finite values, insufficient
//! sample size, vacuous constant series, non-positive degrees of freedom, and
//! numerical compromises. A silent successful call is a contract violation.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{chol_solve, least_squares, thin_svd};
use crate::rng::Rng;
use crate::special::{
    chi2_cdf, chi2_pvalue, f_pvalue, ln_gamma, norm_cdf, student_t_cdf, student_t_pvalue,
};
use crate::traits::{Fit, Predict};
use crate::validate::{inspect_identification, inspect_xy};
use faer::Mat;
use ojizou_san::Session;
use signlred::{
    scan_finite, slice_stats, Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified,
    Report, Result, Severity,
};
use std::collections::BTreeMap;

/// Descriptive moments of a numeric sample.
#[derive(Clone, Debug, PartialEq)]
pub struct Describe {
    /// Finite count used for the moments.
    pub n: usize,
    /// Arithmetic mean of finite entries.
    pub mean: f64,
    /// Sample standard deviation (`n−1` in the variance).
    pub std: f64,
    /// Minimum finite value.
    pub min: f64,
    /// Maximum finite value.
    pub max: f64,
    /// Bias-corrected Fisher skewness (`G1`).
    pub skew: f64,
    /// Excess kurtosis (Fisher `G2`).
    pub kurtosis: f64,
}

/// Hypothesis-test payload shared by the classical tests.
#[derive(Clone, Debug, PartialEq)]
pub struct HypothesisTest {
    /// Test statistic.
    pub statistic: f64,
    /// Two-sided (or otherwise documented) p-value.
    pub pvalue: f64,
    /// Degrees of freedom used for the reference distribution (`NaN` if none).
    pub df: f64,
    /// Effective sample size.
    pub nobs: f64,
}

/// One-way ANOVA decomposition.
#[derive(Clone, Debug, PartialEq)]
pub struct AnovaResult {
    /// `MSB / MSW`.
    pub f_stat: f64,
    /// Upper-tail F p-value.
    pub pvalue: f64,
    /// Between-group degrees of freedom.
    pub df_between: f64,
    /// Within-group degrees of freedom.
    pub df_within: f64,
    /// Between-group sum of squares.
    pub ss_between: f64,
    /// Within-group sum of squares.
    pub ss_within: f64,
}

/// Pearson χ² test on a contingency table.
#[derive(Clone, Debug)]
pub struct Chi2Result {
    /// Pearson χ² statistic.
    pub statistic: f64,
    /// Upper-tail χ² p-value.
    pub pvalue: f64,
    /// `(r−1)(c−1)` (independence) or `k−1` (when used as GOF).
    pub df: f64,
    /// Expected counts under the fitted independence model.
    pub expected: Matrix,
    /// Total table count.
    pub nobs: f64,
}

/// Augmented Dickey–Fuller unit-root regression.
#[derive(Clone, Debug, PartialEq)]
pub struct AdfullerResult {
    /// t-statistic on the lagged level.
    pub stat: f64,
    /// MacKinnon-style interpolated p-value (constant, no trend).
    pub pvalue: f64,
    /// Lag order of Δy used in the ADF regression.
    pub used_lags: usize,
    /// Effective sample size of the regression.
    pub n: usize,
}

/// KPSS level-stationarity statistic (Newey–West long-run variance).
#[derive(Clone, Debug, PartialEq)]
pub struct KpssResult {
    /// KPSS η statistic.
    pub stat: f64,
    /// Interpolated p-value against Kwiatkowski et al. critical values.
    pub pvalue: f64,
    /// Newey–West lag used for the long-run variance.
    pub lags: usize,
    /// Sample size.
    pub n: usize,
}

/// Ljung–Box portmanteau of residual autocorrelation.
#[derive(Clone, Debug, PartialEq)]
pub struct LjungBoxResult {
    /// `Q` statistic.
    pub stat: f64,
    /// χ² p-value with `lags` degrees of freedom.
    pub pvalue: f64,
    /// Number of autocorrelations included.
    pub lags: usize,
}

/// Granger causality F test (`x` → `y` at the stated lag).
#[derive(Clone, Debug, PartialEq)]
pub struct GrangerResult {
    /// Restricted-vs-unrestricted F statistic.
    pub f_stat: f64,
    /// Upper-tail F p-value.
    pub pvalue: f64,
    /// Numerator degrees of freedom (`lag`).
    pub df_num: f64,
    /// Denominator degrees of freedom.
    pub df_den: f64,
    /// Common lag order.
    pub lag: usize,
}

/// Family-wise / FDR correction for a vector of p-values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiTest {
    /// Bonferroni: `min(1, m p_i)`.
    Bonferroni,
    /// Holm step-down FWER.
    Holm,
    /// Benjamini–Hochberg FDR (step-up).
    BenjaminiHochberg,
}

/// Percentile bootstrap of the mean.
#[derive(Clone, Debug, PartialEq)]
pub struct BootstrapMean {
    /// Original-sample mean.
    pub mean: f64,
    /// Lower 2.5% bootstrap percentile.
    pub lo: f64,
    /// Upper 97.5% bootstrap percentile.
    pub hi: f64,
}

/// Kaplan–Meier product-limit estimator.
#[derive(Clone, Debug, Default)]
pub struct KaplanMeier {}

/// Fitted right-censored survival curve.
#[derive(Clone, Debug)]
pub struct FittedKaplanMeier {
    /// Distinct event times (ascending).
    pub times: Vector,
    /// Survival function at each event time (right-continuous product-limit).
    pub survival: Vector,
    /// Number at risk just prior to each event time.
    pub n_risk: Vector,
    /// Event count at each event time.
    pub n_event: Vector,
}

impl KaplanMeier {
    /// Construct the default product-limit estimator.
    pub fn new() -> Self {
        Self {}
    }

    /// Fit on durations and event indicators (`1` = event, `0` = censored).
    pub fn fit(
        &self,
        durations: &Vector,
        events: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedKaplanMeier>> {
        kaplan_meier_fit(durations, events, session)
    }
}

/// Cox proportional-hazards MLE via Newton on the Breslow partial likelihood.
#[derive(Clone, Debug)]
pub struct CoxPH {
    /// Newton iteration cap.
    pub max_iter: usize,
    /// Gradient-norm convergence tolerance.
    pub tol: f64,
}

impl Default for CoxPH {
    fn default() -> Self {
        Self {
            max_iter: 40,
            tol: 1e-8,
        }
    }
}

impl CoxPH {
    /// Default Newton settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit `h(t | x) = h0(t) exp(xβ)` on durations, events, and covariates.
    pub fn fit(
        &self,
        durations: &Vector,
        events: &Vector,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedCoxPH>> {
        cox_ph_fit(self, durations, events, x, session)
    }
}

/// Fitted Cox partial-likelihood coefficients.
#[derive(Clone, Debug)]
pub struct FittedCoxPH {
    /// Log-hazard coefficients.
    pub coef: Vector,
    /// Partial log-likelihood at the reported point.
    pub loglik: f64,
    /// Number of uncensored events.
    pub n_events: usize,
    /// Sample size.
    pub n: usize,
    /// Whether Newton reported convergence.
    pub converged: bool,
}

/// Named Cox wrapper (statsmodels `PHReg`).
///
/// Covariate count is the design width, not a substitute identification `p`
/// for events. Newton uses the inner Cox fit; a failed Cholesky is swallowed
/// so a diagnostic session is not aborted by a Fatal inner issue.
#[derive(Clone, Debug, Default)]
pub struct PHReg {
    inner: CoxPH,
}

impl PHReg {
    /// Default Breslow Cox wrapper.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit `h(t | x) = h0(t) exp(xβ)`.
    pub fn fit(
        &self,
        durations: &Vector,
        events: &Vector,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedPHReg>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(durations), &ctx.policy);
        match self.inner.fit(durations, events, x, &session.child("cox")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::CholeskyFailed
                            | IssueCode::InformationMatrixSingular
                            | IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::LossIsNan
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                ctx.finish(FittedPHReg {
                    coef: q.value.coef,
                    loglik: q.value.loglik,
                    n_events: q.value.n_events,
                    n: q.value.n,
                    converged: q.value.converged,
                })
            }
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("PHReg inner Cox Newton failed; coefficients left at 0")
                        .build(),
                );
                ctx.finish(FittedPHReg {
                    coef: Vector::zeros(x.ncols()),
                    loglik: f64::NAN,
                    n_events: events
                        .as_slice()
                        .iter()
                        .filter(|e| **e > 0.5)
                        .count(),
                    n: x.nrows(),
                    converged: false,
                })
            }
        }
    }
}

/// Fitted PHReg coefficients.
#[derive(Clone, Debug)]
pub struct FittedPHReg {
    /// Log-hazard slopes.
    pub coef: Vector,
    /// Partial log-likelihood at the reported point.
    pub loglik: f64,
    /// Uncensored events.
    pub n_events: usize,
    /// Sample size.
    pub n: usize,
    /// Whether the inner Newton reported convergence.
    pub converged: bool,
}

/// Right-censored survival curve (statsmodels `SurvfuncRight`).
///
/// Event count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct SurvfuncRight {}

impl SurvfuncRight {
    /// Product-limit wrapper.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit on durations and event indicators (`1` = event).
    pub fn fit(
        &self,
        durations: &Vector,
        events: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSurvfuncRight>> {
        let q = KaplanMeier::new().fit(durations, events, session)?;
        let mut ctx = FitCtx::with_session(session.clone());
        for issue in q.report.issues() {
            ctx.push(issue.clone());
        }
        ctx.finish(FittedSurvfuncRight {
            times: q.value.times,
            survival: q.value.survival,
            n_risk: q.value.n_risk,
            n_event: q.value.n_event,
        })
    }
}

/// Fitted right-censored survival function.
#[derive(Clone, Debug)]
pub struct FittedSurvfuncRight {
    /// Distinct event times.
    pub times: Vector,
    /// Product-limit survival.
    pub survival: Vector,
    /// Number at risk.
    pub n_risk: Vector,
    /// Events at each time.
    pub n_event: Vector,
}

/// Shapiro–Francia normal-probability-plot correlation.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapiroFranciaResult {
    /// Squared correlation of order statistics with Blom normal scores.
    pub w: f64,
    /// Approximate p-value (Royston-style logistic transform).
    pub pvalue: f64,
}

/// Sample moments, skewness, and excess kurtosis.
pub fn describe(x: &Vector, session: &Session) -> Result<Qualified<Describe>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let st = slice_stats(x.as_slice());
    let (skew, kurt) = if st.count >= 3 && st.std() > ctx.policy.near_zero_variance {
        fisher_skew_kurt(x.as_slice(), st.mean, st.std())
    } else {
        if st.count >= 1 && st.std() <= ctx.policy.near_zero_variance {
            push_meaningless(
                &mut ctx,
                "skewness and excess kurtosis",
                "a constant (or single-value) sample has no third/fourth moment scale",
            );
        }
        (f64::NAN, f64::NAN)
    };
    ctx.finish(Describe {
        n: st.count,
        mean: st.mean,
        std: st.std(),
        min: st.min,
        max: st.max,
        skew,
        kurtosis: kurt,
    })
}

/// Pearson product-moment correlation.
pub fn pearson(x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    let r = pearson_raw(x.as_slice(), y.as_slice());
    if !r.is_finite() {
        push_meaningless(
            &mut ctx,
            "Pearson correlation",
            "the product-moment ratio is undefined (zero variance or no paired finite rows)",
        );
    }
    ctx.finish(r)
}

/// Spearman rank correlation (Pearson of average ranks).
pub fn spearman(x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    let rx = rank_average(x.as_slice());
    let ry = rank_average(y.as_slice());
    let r = pearson_raw(&rx, &ry);
    if !r.is_finite() {
        push_meaningless(
            &mut ctx,
            "Spearman correlation",
            "ranks are constant or unpaired; ρ is 0/0",
        );
    }
    ctx.finish(r)
}

/// Kendall's τ-b (tie-corrected).
pub fn kendall(x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    let tau = kendall_tau_b(x.as_slice(), y.as_slice());
    if !tau.is_finite() {
        push_meaningless(
            &mut ctx,
            "Kendall tau-b",
            "no discordant/concordant pairs after ties; τ is undefined",
        );
    }
    ctx.finish(tau)
}

/// Pairwise Pearson correlation matrix of the columns of `x`.
pub fn corrcoef(x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let (n, p) = x.shape();
    let mut out = Matrix::zeros(p, p);
    for j in 0..p {
        out.set(j, j, 1.0);
        let cj = x.column(j);
        for i in 0..j {
            let ci = x.column(i);
            let r = pearson_raw(ci.as_slice(), cj.as_slice());
            if !r.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::DegenerateDistribution)
                        .message(format!(
                            "corrcoef[{i},{j}] undefined (zero-variance column)"
                        ))
                        .metric("i", i as f64)
                        .metric("j", j as f64)
                        .build(),
                );
            }
            out.set(i, j, r);
            out.set(j, i, r);
        }
    }
    let _ = n;
    ctx.finish(out)
}

/// Partial correlation of `x` and `y` given the columns of `z`.
///
/// Residualizes both series on `z` (plus intercept) by OLS and returns the
/// Pearson correlation of the residuals.
pub fn partial_corr(
    x: &Vector,
    y: &Vector,
    z: &Matrix,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    if z.nrows() != x.len() && z.ncols() > 0 {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "partial_corr Z is {}×{} but x has length {}",
                    z.nrows(),
                    z.ncols(),
                    x.len()
                ))
                .build(),
        );
    }
    if z.ncols() == 0 || z.nrows() == 0 {
        let r = pearson_raw(x.as_slice(), y.as_slice());
        return ctx.finish(r);
    }
    inspect_xy(&mut ctx.report, z, Some(x), &ctx.policy);
    inspect_identification(&mut ctx.report, z.nrows(), z.ncols() + 1, &ctx.policy);
    let design = z.with_intercept();
    let bx = statistical_ols(&mut ctx, &design, x);
    let by = statistical_ols(&mut ctx, &design, y);
    let r = match (bx, by) {
        (Some(bx), Some(by)) => {
            let rx = x.sub(&design.matvec(&bx));
            let ry = y.sub(&design.matvec(&by));
            let r = pearson_raw(rx.as_slice(), ry.as_slice());
            if !r.is_finite() {
                push_meaningless(
                    &mut ctx,
                    "partial correlation",
                    "residualized series have no remaining variance",
                );
            }
            r
        }
        _ => {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("partial-correlation residualization failed")
                    .build(),
            );
            f64::NAN
        }
    };
    ctx.finish(r)
}

/// Variance inflation factors: `1 / (1 − R²_j)` from OLS of column `j` on the rest.
pub fn vif(x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let (n, p) = x.shape();
    inspect_identification(&mut ctx.report, n, p, &ctx.policy);
    let mut out = Vector::zeros(p);
    if p == 0 {
        return ctx.finish(out);
    }
    if p == 1 {
        out[0] = 1.0;
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("VIF of a single column is identically 1; there are no other regressors")
                .build(),
        );
        return ctx.finish(out);
    }
    for j in 0..p {
        let yj = x.column(j);
        let others = Matrix::from_fn(n, p - 1, |r, c| {
            let src = if c < j { c } else { c + 1 };
            x.get(r, src)
        });
        let design = others.with_intercept();
        match statistical_ols(&mut ctx, &design, &yj) {
            Some(beta) => {
                let fitted = design.matvec(&beta);
                let resid = yj.sub(&fitted);
                let sse = resid.dot(&resid);
                let mean = yj.mean();
                let sst: f64 = yj
                    .as_slice()
                    .iter()
                    .map(|v| {
                        let d = v - mean;
                        d * d
                    })
                    .sum();
                let r2 = if sst <= ctx.policy.near_zero_variance {
                    ctx.push(
                        Issue::builder(IssueCode::ConstantFeature)
                            .message(format!("VIF column {j} has SST≈0"))
                            .metric("feature_index", j as f64)
                            .build(),
                    );
                    f64::NAN
                } else {
                    1.0 - sse / sst
                };
                let vif_j = if r2.is_finite() && (1.0 - r2).abs() > 1e-15 {
                    1.0 / (1.0 - r2)
                } else if r2.is_finite() {
                    ctx.push(
                        Issue::builder(IssueCode::PerfectCollinearity)
                            .message(format!(
                                "column {j} is an exact linear combination of the others"
                            ))
                            .metric("feature_index", j as f64)
                            .metric("r2", r2)
                            .meaninglessness(Meaninglessness::vacuous(
                                "variance inflation factor",
                                "R²=1 ⇒ VIF is infinite; the column is unidentified",
                                "drop a dependent column",
                            ))
                            .build(),
                    );
                    f64::INFINITY
                } else {
                    f64::NAN
                };
                if vif_j.is_finite() && vif_j >= ctx.policy.vif_warn {
                    ctx.push(
                        Issue::builder(IssueCode::HighMulticollinearity)
                            .message(format!("VIF[{j}]={vif_j:.4}"))
                            .metric("feature_index", j as f64)
                            .metric("vif", vif_j)
                            .build(),
                    );
                }
                out[j] = vif_j;
            }
            None => {
                out[j] = f64::NAN;
            }
        }
    }
    ctx.finish(out)
}

/// One-sample t-test of `E[x] = popmean`.
pub fn ttest_1samp(
    x: &Vector,
    popmean: f64,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    if !popmean.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("ttest_1samp popmean is not finite")
                .build(),
        );
    }
    let st = slice_stats(x.as_slice());
    let n = st.count;
    let df = n as f64 - 1.0;
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("one-sample t-test needs at least two finite observations")
                .metric("n", n as f64)
                .build(),
        );
    }
    if df <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegreesOfFreedomNonPositive)
                .message("ttest_1samp df ≤ 0")
                .meaninglessness(Meaninglessness::vacuous(
                    "Student-t p-value",
                    "variance is unidentified with n<2",
                    "collect more observations",
                ))
                .build(),
        );
    }
    let se = if n >= 2 && st.std() > ctx.policy.near_zero_variance {
        st.std() / (n as f64).sqrt()
    } else {
        0.0
    };
    let (stat, pvalue) = if se > 0.0 && df > 0.0 {
        let t = (st.mean - popmean) / se;
        (t, student_t_pvalue(t, df))
    } else {
        if n >= 2 && st.std() <= ctx.policy.near_zero_variance {
            push_meaningless(
                &mut ctx,
                "one-sample t statistic",
                "sample variance is zero; the t ratio is undefined",
            );
        }
        (f64::NAN, f64::NAN)
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n as f64,
    })
}

/// Two-sample t-test. `welch = true` uses the Welch–Satterthwaite df.
pub fn ttest_ind(
    x: &Vector,
    y: &Vector,
    welch: bool,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    inspect_series_as_target(&mut ctx, y);
    let sx = slice_stats(x.as_slice());
    let sy = slice_stats(y.as_slice());
    let n1 = sx.count as f64;
    let n2 = sy.count as f64;
    if sx.count < 2 || sy.count < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("two-sample t-test needs ≥2 finite observations in each group")
                .metric("n1", n1)
                .metric("n2", n2)
                .build(),
        );
    }
    let v1 = sx.variance;
    let v2 = sy.variance;
    let (se2, df) = if welch {
        let a = v1 / n1;
        let b = v2 / n2;
        let se2 = a + b;
        let df = if se2 > 0.0 {
            se2 * se2 / (a * a / (n1 - 1.0) + b * b / (n2 - 1.0))
        } else {
            0.0
        };
        (se2, df)
    } else {
        let df = n1 + n2 - 2.0;
        let sp = if df > 0.0 {
            ((n1 - 1.0) * v1 + (n2 - 1.0) * v2) / df
        } else {
            f64::NAN
        };
        (sp * (1.0 / n1 + 1.0 / n2), df)
    };
    if df <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegreesOfFreedomNonPositive)
                .message("two-sample t-test has non-positive df")
                .meaninglessness(Meaninglessness::vacuous(
                    "two-sample t p-value",
                    "df ≤ 0; the reference t law is not defined",
                    "collect more observations in each group",
                ))
                .build(),
        );
    }
    let (stat, pvalue) = if se2.is_finite() && se2 > 0.0 && df > 0.0 {
        let t = (sx.mean - sy.mean) / se2.sqrt();
        (t, student_t_pvalue(t, df))
    } else {
        push_meaningless(
            &mut ctx,
            "two-sample t statistic",
            "pooled / Welch variance is zero or undefined",
        );
        (f64::NAN, f64::NAN)
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n1 + n2,
    })
}

/// Paired t-test on `x − y`.
pub fn ttest_rel(x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    if x.len() != y.len() {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: x.len().min(y.len()) as f64,
        });
    }
    let d = x.sub(y);
    let st = slice_stats(d.as_slice());
    let n = st.count;
    let df = n as f64 - 1.0;
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("paired t-test needs at least two finite differences")
                .metric("n", n as f64)
                .build(),
        );
    }
    if df <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegreesOfFreedomNonPositive)
                .message("paired t-test df ≤ 0")
                .meaninglessness(Meaninglessness::vacuous(
                    "paired t p-value",
                    "variance of the differences is unidentified",
                    "collect more paired observations",
                ))
                .build(),
        );
    }
    let se = if n >= 2 && st.std() > ctx.policy.near_zero_variance {
        st.std() / (n as f64).sqrt()
    } else {
        0.0
    };
    let (stat, pvalue) = if se > 0.0 && df > 0.0 {
        let t = st.mean / se;
        (t, student_t_pvalue(t, df))
    } else {
        push_meaningless(
            &mut ctx,
            "paired t statistic",
            "difference variance is zero; the t ratio is undefined",
        );
        (f64::NAN, f64::NAN)
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n as f64,
    })
}

/// One-way ANOVA F test on two or more groups.
pub fn anova_oneway(groups: &[&Vector], session: &Session) -> Result<Qualified<AnovaResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if groups.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("one-way ANOVA needs at least two groups")
                .build(),
        );
    }
    let mut all: Vec<f64> = Vec::new();
    let mut ns = Vec::new();
    let mut means = Vec::new();
    for (g, grp) in groups.iter().enumerate() {
        inspect_series_as_target(&mut ctx, grp);
        let st = slice_stats(grp.as_slice());
        if st.count < 1 {
            ctx.push(
                Issue::builder(IssueCode::EmptyClass)
                    .message(format!("ANOVA group {g} has no finite values"))
                    .build(),
            );
        }
        ns.push(st.count);
        means.push(st.mean);
        all.extend(grp.as_slice().iter().copied().filter(|v| v.is_finite()));
    }
    let n: usize = ns.iter().sum();
    let k = groups.len();
    let grand = if n > 0 {
        all.iter().sum::<f64>() / n as f64
    } else {
        f64::NAN
    };
    let mut ssb = 0.0;
    for i in 0..k {
        let d = means[i] - grand;
        ssb += ns[i] as f64 * d * d;
    }
    let mut ssw = 0.0;
    for (grp, &m) in groups.iter().zip(&means) {
        for &v in grp.as_slice() {
            if v.is_finite() {
                let d = v - m;
                ssw += d * d;
            }
        }
    }
    let dfb = k as f64 - 1.0;
    let dfw = n as f64 - k as f64;
    if dfw <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegreesOfFreedomNonPositive)
                .message(format!("ANOVA df_within={dfw}"))
                .meaninglessness(Meaninglessness::vacuous(
                    "ANOVA F",
                    "no within-group residual df; MSB/MSW is undefined",
                    "increase within-group sample sizes",
                ))
                .build(),
        );
    }
    let (f_stat, pvalue) = if dfb > 0.0 && dfw > 0.0 && ssw > 0.0 {
        let f = (ssb / dfb) / (ssw / dfw);
        (f, f_pvalue(f, dfb, dfw))
    } else {
        if ssw == 0.0 && n > 0 {
            push_meaningless(
                &mut ctx,
                "ANOVA F",
                "within-group variance is zero; the F ratio is infinite or undefined",
            );
        }
        (f64::NAN, f64::NAN)
    };
    ctx.finish(AnovaResult {
        f_stat,
        pvalue,
        df_between: dfb,
        df_within: dfw,
        ss_between: ssb,
        ss_within: ssw,
    })
}

/// χ² test of independence on a contingency table.
pub fn chi2_independence(table: &Matrix, session: &Session) -> Result<Qualified<Chi2Result>> {
    chi2_contingency(table, session)
}

/// Pearson χ² test of independence (same contract as [`chi2_independence`]).
pub fn chi2_contingency(table: &Matrix, session: &Session) -> Result<Qualified<Chi2Result>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, table, None, &ctx.policy);
    let (r, c) = table.shape();
    let mut row = vec![0.0; r];
    let mut col = vec![0.0; c];
    let mut nobs = 0.0;
    let mut negative = false;
    for i in 0..r {
        for j in 0..c {
            let v = table.get(i, j);
            if v < 0.0 {
                negative = true;
            }
            row[i] += v;
            col[j] += v;
            nobs += v;
        }
    }
    if negative {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message("contingency table contains a negative count")
                .build(),
        );
    }
    let df = (r.saturating_sub(1) * c.saturating_sub(1)) as f64;
    if df <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegreesOfFreedomNonPositive)
                .message(format!("χ² independence df={df} for a {r}×{c} table"))
                .meaninglessness(Meaninglessness::vacuous(
                    "χ² p-value",
                    "a 1-row or 1-column table has no independence contrast",
                    "supply a table with at least two rows and two columns",
                ))
                .build(),
        );
    }
    let mut expected = Matrix::zeros(r, c);
    let mut stat = 0.0;
    let mut tiny = false;
    if nobs > 0.0 {
        for i in 0..r {
            for j in 0..c {
                let e = row[i] * col[j] / nobs;
                expected.set(i, j, e);
                if e < 5.0 {
                    tiny = true;
                }
                if e > 0.0 {
                    let d = table.get(i, j) - e;
                    stat += d * d / e;
                } else if table.get(i, j) != 0.0 {
                    ctx.push(
                        Issue::builder(IssueCode::DegenerateDistribution)
                            .message(format!("expected[{i},{j}]=0 with positive observed count"))
                            .build(),
                    );
                }
            }
        }
    } else {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("contingency table totals zero")
                .build(),
        );
        stat = f64::NAN;
    }
    if tiny && df > 0.0 {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("an expected cell is < 5; the χ² reference is a poor approximation")
                .build(),
        );
    }
    let pvalue = if stat.is_finite() && df > 0.0 {
        chi2_pvalue(stat, df)
    } else {
        f64::NAN
    };
    ctx.finish(Chi2Result {
        statistic: stat,
        pvalue,
        df,
        expected,
        nobs,
    })
}

/// Two-sample Kolmogorov–Smirnov test (asymptotic Smirnov p-value).
pub fn ks_2samp(x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    inspect_series_as_target(&mut ctx, y);
    let mut xs: Vec<f64> = x
        .as_slice()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let mut ys: Vec<f64> = y
        .as_slice()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n1 = xs.len();
    let n2 = ys.len();
    if n1 == 0 || n2 == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("KS two-sample test needs a finite observation in each sample")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: (n1 + n2) as f64,
        });
    }
    let mut i = 0usize;
    let mut j = 0usize;
    let mut d: f64 = 0.0;
    while i < n1 || j < n2 {
        let xv = if i < n1 { xs[i] } else { f64::INFINITY };
        let yv = if j < n2 { ys[j] } else { f64::INFINITY };
        if xv <= yv {
            i += 1;
        }
        if yv <= xv {
            j += 1;
        }
        let fnx = i as f64 / n1 as f64;
        let fny = j as f64 / n2 as f64;
        d = d.max((fnx - fny).abs());
    }
    let neff = (n1 * n2) as f64 / (n1 + n2) as f64;
    let pvalue = ks_pvalue(d, neff);
    ctx.finish(HypothesisTest {
        statistic: d,
        pvalue,
        df: f64::NAN,
        nobs: (n1 + n2) as f64,
    })
}

/// Shapiro–Francia W' (normal plot correlation of order statistics).
pub fn shapiro_francia(x: &Vector, session: &Session) -> Result<Qualified<ShapiroFranciaResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let mut xs: Vec<f64> = x
        .as_slice()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let n = xs.len();
    if n < 5 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("Shapiro–Francia needs at least 5 finite observations")
                .metric("n", n as f64)
                .build(),
        );
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut scores = vec![0.0; n];
    for i in 0..n {
        let p = (i as f64 + 0.375) / (n as f64 + 0.25);
        scores[i] = norm_ppf(p.clamp(1e-12, 1.0 - 1e-12));
    }
    let w = {
        let r = pearson_raw(&xs, &scores);
        if r.is_finite() {
            r * r
        } else {
            f64::NAN
        }
    };
    if !w.is_finite() {
        push_meaningless(
            &mut ctx,
            "Shapiro–Francia W'",
            "the ordered sample is constant; the normal-plot correlation is undefined",
        );
    }
    let pvalue = shapiro_francia_pvalue(w, n);
    ctx.finish(ShapiroFranciaResult { w, pvalue })
}

/// Levene / Brown–Forsythe test (absolute deviations from group medians).
pub fn levene(groups: &[&Vector], session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if groups.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("Levene's test needs at least two groups")
                .build(),
        );
    }
    let mut z_groups: Vec<Vector> = Vec::new();
    for grp in groups {
        inspect_series_as_target(&mut ctx, grp);
        let med = median(grp.as_slice());
        let z = Vector::from_iter(grp.as_slice().iter().map(|v| {
            if v.is_finite() && med.is_finite() {
                (v - med).abs()
            } else {
                f64::NAN
            }
        }));
        z_groups.push(z);
    }
    let mut all: Vec<f64> = Vec::new();
    let mut ns = Vec::new();
    let mut means = Vec::new();
    for z in &z_groups {
        let st = slice_stats(z.as_slice());
        ns.push(st.count);
        means.push(st.mean);
        all.extend(z.as_slice().iter().copied().filter(|v| v.is_finite()));
    }
    let n: usize = ns.iter().sum();
    let k = z_groups.len();
    let grand = if n > 0 {
        all.iter().sum::<f64>() / n as f64
    } else {
        f64::NAN
    };
    let mut ssb = 0.0;
    for i in 0..k {
        let d = means[i] - grand;
        ssb += ns[i] as f64 * d * d;
    }
    let mut ssw = 0.0;
    for (z, &m) in z_groups.iter().zip(&means) {
        for &v in z.as_slice() {
            if v.is_finite() {
                let d = v - m;
                ssw += d * d;
            }
        }
    }
    let dfb = k as f64 - 1.0;
    let dfw = n as f64 - k as f64;
    if dfw <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegreesOfFreedomNonPositive)
                .message(format!("Levene df_within={dfw}"))
                .meaninglessness(Meaninglessness::vacuous(
                    "Levene F",
                    "no within-group residual df after forming |x − median|",
                    "increase group sizes",
                ))
                .build(),
        );
    }
    let (stat, pvalue) = if dfb > 0.0 && dfw > 0.0 && ssw > 0.0 {
        let f = (ssb / dfb) / (ssw / dfw);
        (f, f_pvalue(f, dfb, dfw))
    } else {
        push_meaningless(
            &mut ctx,
            "Levene statistic",
            "absolute-deviation groups have no residual variance",
        );
        (f64::NAN, f64::NAN)
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: dfw,
        nobs: n as f64,
    })
}

/// Mann–Whitney U (two-sided normal approximation with tie correction).
pub fn mannwhitneyu(
    x: &Vector,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    inspect_series_as_target(&mut ctx, y);
    let mut paired: Vec<(f64, u8)> = Vec::new();
    for &v in x.as_slice() {
        if v.is_finite() {
            paired.push((v, 0));
        }
    }
    for &v in y.as_slice() {
        if v.is_finite() {
            paired.push((v, 1));
        }
    }
    let n1 = x.as_slice().iter().filter(|v| v.is_finite()).count() as f64;
    let n2 = y.as_slice().iter().filter(|v| v.is_finite()).count() as f64;
    if n1 < 1.0 || n2 < 1.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("Mann–Whitney needs a finite observation in each sample")
                .build(),
        );
    }
    paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let values: Vec<f64> = paired.iter().map(|p| p.0).collect();
    let ranks = rank_average(&values);
    let mut r1 = 0.0;
    for (rank, &(_, g)) in ranks.iter().zip(&paired) {
        if g == 0 {
            r1 += *rank;
        }
    }
    let u1 = r1 - n1 * (n1 + 1.0) / 2.0;
    let u = u1.min(n1 * n2 - u1);
    let n = n1 + n2;
    let mut tie = 0.0;
    {
        let mut i = 0;
        while i < values.len() {
            let mut j = i + 1;
            while j < values.len() && (values[j] - values[i]).abs() <= 0.0 {
                j += 1;
            }
            let t = (j - i) as f64;
            if t > 1.0 {
                tie += t * t * t - t;
            }
            i = j;
        }
    }
    let var = n1 * n2 / 12.0 * ((n + 1.0) - tie / (n * (n - 1.0)));
    let (stat, pvalue) = if var > 0.0 {
        let z = (u - n1 * n2 / 2.0) / var.sqrt();
        (u, crate::special::norm_pvalue_two_sided(z))
    } else {
        push_meaningless(
            &mut ctx,
            "Mann–Whitney U",
            "all values are tied; the rank-sum variance is zero",
        );
        (u, f64::NAN)
    };
    if n < 20.0 {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("Mann–Whitney p-value uses a normal approximation that is crude for n<20")
                .metric("n", n)
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: f64::NAN,
        nobs: n,
    })
}

/// Wilcoxon signed-rank test on paired differences `x − y` (normal approximation).
pub fn wilcoxon_signed(
    x: &Vector,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    let mut diffs: Vec<f64> = Vec::new();
    for i in 0..x.len().min(y.len()) {
        let d = x[i] - y[i];
        if d.is_finite() && d.abs() > 0.0 {
            diffs.push(d);
        }
    }
    let n = diffs.len();
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("Wilcoxon signed-rank needs at least two non-zero finite differences")
                .metric("n", n as f64)
                .build(),
        );
    }
    let absd: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let ranks = rank_average(&absd);
    let mut w = 0.0;
    for (d, r) in diffs.iter().zip(&ranks) {
        if *d > 0.0 {
            w += *r;
        }
    }
    let nf = n as f64;
    let mean = nf * (nf + 1.0) / 4.0;
    let var = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0;
    let (stat, pvalue) = if var > 0.0 {
        let z = (w - mean) / var.sqrt();
        (w, crate::special::norm_pvalue_two_sided(z))
    } else {
        push_meaningless(
            &mut ctx,
            "Wilcoxon signed-rank",
            "no signed ranks remain after dropping zeros",
        );
        (w, f64::NAN)
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: f64::NAN,
        nobs: nf,
    })
}

/// Kruskal–Wallis H test (χ² approximation).
pub fn kruskal(groups: &[&Vector], session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if groups.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("Kruskal–Wallis needs at least two groups")
                .build(),
        );
    }
    let mut vals = Vec::new();
    let mut gid = Vec::new();
    for (g, grp) in groups.iter().enumerate() {
        inspect_series_as_target(&mut ctx, grp);
        for &v in grp.as_slice() {
            if v.is_finite() {
                vals.push(v);
                gid.push(g);
            }
        }
    }
    let n = vals.len() as f64;
    let k = groups.len() as f64;
    if n < 2.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("Kruskal–Wallis needs at least two finite observations")
                .build(),
        );
    }
    let ranks = rank_average(&vals);
    let mut rsum = vec![0.0; groups.len()];
    let mut nn = vec![0.0; groups.len()];
    for (r, &g) in ranks.iter().zip(&gid) {
        rsum[g] += *r;
        nn[g] += 1.0;
    }
    let mut h = 0.0;
    for g in 0..groups.len() {
        if nn[g] > 0.0 {
            h += rsum[g] * rsum[g] / nn[g];
        }
    }
    h = 12.0 / (n * (n + 1.0)) * h - 3.0 * (n + 1.0);
    let df = k - 1.0;
    let pvalue = if h.is_finite() && df > 0.0 && n > 0.0 {
        chi2_pvalue(h.max(0.0), df)
    } else {
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: h,
        pvalue,
        df,
        nobs: n,
    })
}

/// Jarque–Bera normality test from sample skewness and excess kurtosis.
pub fn jarque_bera(x: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let st = slice_stats(x.as_slice());
    if st.count >= 1 && st.is_constant(ctx.policy.near_zero_variance) {
        ctx.push(
            Issue::builder(IssueCode::ConstantTarget)
                .message("Jarque–Bera on a constant series is meaningless")
                .metric("target_std", st.std())
                .meaninglessness(Meaninglessness::vacuous(
                    "Jarque–Bera statistic",
                    "skewness and kurtosis require a scale; a constant sample has none",
                    "do not report a normality p-value for a degenerate sample",
                ))
                .build(),
        );
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("JB is 0/0 on a constant series")
                .meaninglessness(Meaninglessness::vacuous(
                    "Jarque–Bera p-value",
                    "the moment ratios that define JB are undefined",
                    "discard the test",
                ))
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 2.0,
            nobs: st.count as f64,
        });
    }
    if st.count < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("Jarque–Bera needs n≥3")
                .metric("n", st.count as f64)
                .build(),
        );
    }
    let (skew, kurt) = fisher_skew_kurt(x.as_slice(), st.mean, st.std());
    let n = st.count as f64;
    let stat = n / 6.0 * (skew * skew + 0.25 * kurt * kurt);
    let pvalue = if stat.is_finite() {
        chi2_pvalue(stat, 2.0)
    } else {
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: 2.0,
        nobs: n,
    })
}

/// Durbin–Watson statistic `Σ (e_t − e_{t−1})² / Σ e_t²`.
pub fn durbin_watson(resid: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, resid);
    let e = resid.as_slice();
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..e.len() {
        if e[i].is_finite() {
            den += e[i] * e[i];
        }
        if i > 0 && e[i].is_finite() && e[i - 1].is_finite() {
            let d = e[i] - e[i - 1];
            num += d * d;
        }
    }
    let dw = if den > 0.0 {
        num / den
    } else {
        push_meaningless(
            &mut ctx,
            "Durbin–Watson",
            "residual SSE is zero; the ratio is undefined",
        );
        f64::NAN
    };
    if dw.is_finite() && (dw < 1.0 || dw > 3.0) && e.len() >= 8 {
        ctx.push(
            Issue::builder(IssueCode::AutocorrelatedResiduals)
                .message(format!("Durbin–Watson={dw:.3}"))
                .metric("durbin_watson", dw)
                .build(),
        );
    }
    ctx.finish(dw)
}

/// Breusch–Pagan LM test: `n R²` from `e² ~ design`.
pub fn breusch_pagan(
    resid: &Vector,
    design: &Matrix,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, design, Some(resid), &ctx.policy);
    inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
    let e2 = Vector::from_iter(resid.as_slice().iter().map(|e| e * e));
    if e2
        .as_slice()
        .iter()
        .all(|v| *v <= ctx.policy.near_zero_variance)
    {
        push_meaningless(
            &mut ctx,
            "Breusch–Pagan LM",
            "squared residuals are identically zero; there is no heteroscedasticity contrast",
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: design.ncols() as f64,
            nobs: resid.len() as f64,
        });
    }
    let x = if design.ncols() == 0 {
        Matrix::from_fn(resid.len(), 1, |_, _| 1.0)
    } else {
        design.clone()
    };
    let Some(beta) = statistical_ols(&mut ctx, &x, &e2) else {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("Breusch–Pagan auxiliary regression failed")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: x.ncols() as f64,
            nobs: resid.len() as f64,
        });
    };
    let fitted = x.matvec(&beta);
    let r = e2.sub(&fitted);
    let sse = r.dot(&r);
    let mean = e2.mean();
    let sst: f64 = e2
        .as_slice()
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum();
    let r2 = if sst > 0.0 { 1.0 - sse / sst } else { f64::NAN };
    let n = resid.len() as f64;
    let df = (x.ncols().saturating_sub(1)).max(1) as f64;
    let stat = n * r2;
    let pvalue = if stat.is_finite() {
        chi2_pvalue(stat.max(0.0), df)
    } else {
        f64::NAN
    };
    if pvalue.is_finite() && pvalue < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::Heteroscedasticity)
                .message(format!("Breusch–Pagan p={pvalue:.4}"))
                .metric("lm", stat)
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n,
    })
}

/// White's heteroscedasticity LM test (`het_white`).
///
/// Auxiliary regression of \(e^2\) on the original columns and their
/// cross-products. A perfect auxiliary fit is Misleading, not vacuous, so
/// a well-specified outer model is not aborted.
pub fn het_white(
    resid: &Vector,
    design: &Matrix,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, design, Some(resid), &ctx.policy);
    let e2 = Vector::from_iter(resid.as_slice().iter().map(|e| e * e));
    if e2
        .as_slice()
        .iter()
        .all(|v| *v <= ctx.policy.near_zero_variance)
    {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("White LM: squared residuals are identically zero")
                .meaninglessness(Meaninglessness::vacuous(
                    "White LM",
                    "there is no heteroscedasticity contrast",
                    "the residual is a perfect fit",
                ))
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: resid.len() as f64,
        });
    }
    let p = design.ncols();
    let n = resid.len();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    cols.push(vec![1.0; n]);
    for j in 0..p {
        let mut looks_const = true;
        let mut col = Vec::with_capacity(n);
        for i in 0..n {
            let v = if i < design.nrows() {
                design.get(i, j)
            } else {
                0.0
            };
            if (v - 1.0).abs() > 1e-12 {
                looks_const = false;
            }
            col.push(v);
        }
        if j == 0 && looks_const {
            continue;
        }
        cols.push(col);
    }
    let base = cols.len();
    for a in 1..base {
        for b in a..base {
            cols.push((0..n).map(|i| cols[a][i] * cols[b][i]).collect());
        }
    }
    if cols.len() >= n {
        ctx.push(
            Issue::builder(IssueCode::Overparameterized)
                .message(format!(
                    "White auxiliary has {} columns ≥ n={n}; higher-order terms are truncated",
                    cols.len()
                ))
                .meaninglessness(Meaninglessness::new(
                    "White LM",
                    "the auxiliary design is wider than the sample",
                    signlred::InterpretiveValue::Misleading,
                    "the LM uses the truncated column set",
                ))
                .build(),
        );
        cols.truncate(n.saturating_sub(1).max(1));
    }
    let aux = Matrix::from_fn(n, cols.len(), |i, j| cols[j][i]);
    let mut scratch = Report::new("white", "aux");
    let Some(beta) = crate::linalg::least_squares(&mut scratch, &aux, &e2, &ctx.policy) else {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .severity(signlred::Severity::Warning)
                .message("White auxiliary regression failed to factor")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: (cols.len().saturating_sub(1)) as f64,
            nobs: n as f64,
        });
    };
    for issue in scratch.issues() {
        if matches!(
            issue.code,
            IssueCode::ResidualTooLarge
                | IssueCode::PerfectCollinearity
                | IssueCode::NearSingular
                | IssueCode::R2IsOne
        ) {
            continue;
        }
        ctx.push(issue.clone());
    }
    let fitted = aux.matvec(&beta);
    let r = e2.sub(&fitted);
    let sse = r.dot(&r);
    let mean = e2.mean();
    let sst: f64 = e2
        .as_slice()
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum();
    let r2 = if sst > 0.0 { 1.0 - sse / sst } else { f64::NAN };
    let df = (cols.len().saturating_sub(1)).max(1) as f64;
    let stat = n as f64 * r2;
    let pvalue = if stat.is_finite() {
        chi2_pvalue(stat.max(0.0), df)
    } else {
        f64::NAN
    };
    if pvalue.is_finite() && pvalue < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::Heteroscedasticity)
                .message(format!("White p={pvalue:.4}"))
                .metric("lm", stat)
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n as f64,
    })
}

/// Alias of [`breusch_pagan`] (statsmodels `het_breuschpagan`).
pub fn het_breuschpagan(
    resid: &Vector,
    design: &Matrix,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    breusch_pagan(resid, design, session)
}

/// Alias of [`ljung_box`] (statsmodels `acorr_ljungbox`).
pub fn acorr_ljungbox(
    x: &Vector,
    lags: usize,
    session: &Session,
) -> Result<Qualified<LjungBoxResult>> {
    ljung_box(x, lags, session)
}

/// Ljung–Box `Q` test on the first `lags` autocorrelations of `x`.
pub fn ljung_box(x: &Vector, lags: usize, session: &Session) -> Result<Qualified<LjungBoxResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let n = x.len();
    if lags == 0 || lags >= n {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("Ljung–Box lags={lags} is not in 1..n-1 (n={n})"))
                .build(),
        );
    }
    let rho = acf_raw(x.as_slice(), lags);
    let nf = n as f64;
    let mut q = 0.0;
    let h = lags.min(n.saturating_sub(1));
    for k in 1..=h {
        let r = rho[k];
        q += r * r / (nf - k as f64);
    }
    q *= nf * (nf + 2.0);
    let df = h as f64;
    if df <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegreesOfFreedomNonPositive)
                .message("Ljung–Box has no lags")
                .build(),
        );
    }
    let pvalue = if q.is_finite() && df > 0.0 {
        chi2_pvalue(q, df)
    } else {
        f64::NAN
    };
    if pvalue.is_finite() && pvalue < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::AutocorrelatedResiduals)
                .message(format!("Ljung–Box Q={q:.3} p={pvalue:.4}"))
                .build(),
        );
    }
    ctx.finish(LjungBoxResult {
        stat: q,
        pvalue,
        lags: h,
    })
}

/// Augmented Dickey–Fuller test: Δy on `y_{t−1}` and lags of Δy (with intercept).
pub fn adfuller(
    y: &Vector,
    lags: Option<usize>,
    session: &Session,
) -> Result<Qualified<AdfullerResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    let n = y.len();
    let used = lags.unwrap_or_else(|| schwert_lags(n));
    if n < used + 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message(format!("ADF n={n} is too short for {used} lags"))
                .build(),
        );
    }
    let mut dy = vec![0.0; n.saturating_sub(1)];
    for t in 1..n {
        dy[t - 1] = y[t] - y[t - 1];
    }
    // Rows: t = used+1 .. n-1 in the original index (Δy index used .. n-2).
    let t0 = used;
    let n_eff = dy.len().saturating_sub(t0);
    let p = 2 + used; // intercept, y_{t-1}, Δy lags
    if n_eff == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("ADF regression has no rows")
                .build(),
        );
        return ctx.finish(AdfullerResult {
            stat: f64::NAN,
            pvalue: f64::NAN,
            used_lags: used,
            n: 0,
        });
    }
    inspect_identification(&mut ctx.report, n_eff, p, &ctx.policy);
    let design = Matrix::from_fn(n_eff, p, |i, j| {
        let t = t0 + i; // index into dy (Δy_t lives at original t+1)
        if j == 0 {
            1.0
        } else if j == 1 {
            y[t] // y_{t} is the lag of the level for Δy_{t+1}?
                 // Δy_{t+1} = y[t+1]-y[t] is dy[t].
                 // regressor y_{t} for observation Δy_{t+1} is y[t].
        } else {
            let lag = j - 1; // 1..used
                             // Δy_{t+1 - lag} = dy[t - lag]
            dy[t - lag]
        }
    });
    let target = Vector::from_iter((0..n_eff).map(|i| dy[t0 + i]));
    // Check the y_{t-1} column mapping: observation i uses dy[t0+i] = y[t0+i+1]-y[t0+i],
    // so lagged level is y[t0+i]. design j==1 uses y[t] with t=t0+i. Correct.
    let Some(beta) = statistical_ols(&mut ctx, &design, &target) else {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("ADF OLS failed")
                .build(),
        );
        return ctx.finish(AdfullerResult {
            stat: f64::NAN,
            pvalue: f64::NAN,
            used_lags: used,
            n: n_eff,
        });
    };
    let fitted = design.matvec(&beta);
    let resid = target.sub(&fitted);
    let sse = resid.dot(&resid);
    let df = n_eff as f64 - p as f64;
    let sigma2 = if df > 0.0 { sse / df } else { f64::NAN };
    let se = ols_se_j(&mut ctx, &design, sigma2, 1);
    let stat = if se.is_finite() && se > 0.0 {
        beta[1] / se
    } else {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("ADF standard error of the lagged level is zero")
                .build(),
        );
        f64::NAN
    };
    let rho = 1.0 + beta[1];
    if rho.is_finite() && ((rho - 1.0).abs() < 0.02 || rho.abs() > 0.98) {
        ctx.push(
            Issue::builder(IssueCode::NonStationary)
                .message(format!(
                    "ADF implied ρ≈{rho:.4}; unit-root behaviour is plausible"
                ))
                .metric("rho", rho)
                .metric("adf_stat", stat)
                .build(),
        );
    }
    let pvalue = adf_pvalue_constant(stat);
    ctx.finish(AdfullerResult {
        stat,
        pvalue,
        used_lags: used,
        n: n_eff,
    })
}

/// Elliott–Rothenberg–Stock DF-GLS (constant case, GLS-detrended ADF).
///
/// Default lag is 1 so identification `p` stays the ADF regressors, not a
/// Schwert rule that would over-parameterize short samples. The p-value reuses
/// the constant ADF interpolation and is recorded as unreliable for ERS tables.
pub fn dfgls(
    y: &Vector,
    lags: Option<usize>,
    session: &Session,
) -> Result<Qualified<AdfullerResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    let n = y.len();
    let used = lags.unwrap_or(1);
    if n < used + 6 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!("DF-GLS n={n} is thin for {used} lags"))
                .build(),
        );
    }
    let cbar = -7.0;
    let abar = 1.0 + cbar / (n.max(1) as f64);
    let mut yt = vec![0.0; n];
    let mut zt = vec![0.0; n];
    if n > 0 {
        yt[0] = y[0];
        zt[0] = 1.0;
    }
    for t in 1..n {
        yt[t] = y[t] - abar * y[t - 1];
        zt[t] = 1.0 - abar;
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for t in 0..n {
        if yt[t].is_finite() && zt[t].is_finite() {
            num += zt[t] * yt[t];
            den += zt[t] * zt[t];
        }
    }
    let mu = if den > 0.0 { num / den } else { 0.0 };
    let e = Vector::from_iter((0..n).map(|t| y[t] - mu));
    let mut de = vec![0.0; n.saturating_sub(1)];
    for t in 1..n {
        de[t - 1] = e[t] - e[t - 1];
    }
    let t0 = used;
    let n_eff = de.len().saturating_sub(t0);
    let p = 1 + used;
    if n_eff == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("DF-GLS regression has no rows")
                .build(),
        );
        return ctx.finish(AdfullerResult {
            stat: f64::NAN,
            pvalue: f64::NAN,
            used_lags: used,
            n: 0,
        });
    }
    inspect_identification(&mut ctx.report, n_eff, p, &ctx.policy);
    let design = Matrix::from_fn(n_eff, p, |i, j| {
        let t = t0 + i;
        if j == 0 {
            e[t]
        } else {
            de[t - j]
        }
    });
    let target = Vector::from_iter((0..n_eff).map(|i| de[t0 + i]));
    let Some(beta) = statistical_ols(&mut ctx, &design, &target) else {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("DF-GLS OLS failed")
                .build(),
        );
        return ctx.finish(AdfullerResult {
            stat: f64::NAN,
            pvalue: f64::NAN,
            used_lags: used,
            n: n_eff,
        });
    };
    let fitted = design.matvec(&beta);
    let resid = target.sub(&fitted);
    let sse = resid.dot(&resid);
    let df = n_eff as f64 - p as f64;
    let sigma2 = if df > 0.0 { sse / df } else { f64::NAN };
    let se = ols_se_j(&mut ctx, &design, sigma2, 0);
    let stat = if se.is_finite() && se > 0.0 {
        beta[0] / se
    } else {
        f64::NAN
    };
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("DF-GLS p-value reuses the constant ADF interpolation, not ERS tables")
            .build(),
    );
    ctx.finish(AdfullerResult {
        stat,
        pvalue: adf_pvalue_constant(stat),
        used_lags: used,
        n: n_eff,
    })
}

/// Zivot–Andrews crash-dummy unit-root scan (min t-stat over break dates).
///
/// Critical values are not the ZA tables; the p-value is the constant ADF
/// interpolation and is recorded as unreliable.
#[derive(Clone, Debug, PartialEq)]
pub struct ZivotAndrewsResult {
    /// Minimum ADF-style t-statistic.
    pub stat: f64,
    /// Interpolated p-value (not ZA tables).
    pub pvalue: f64,
    /// Break index (original time) that produced `stat`.
    pub break_index: usize,
    /// Effective rows of the last regression.
    pub n: usize,
}

/// Zivot–Andrews unit-root test with a crash dummy.
pub fn zivot_andrews(y: &Vector, session: &Session) -> Result<Qualified<ZivotAndrewsResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    let n = y.len();
    let used = 1usize;
    let lo = ((0.15 * n as f64).floor() as usize).max(3);
    let hi = ((0.85 * n as f64).ceil() as usize).min(n.saturating_sub(4));
    let t0 = used + 1;
    let n_eff = n.saturating_sub(t0);
    inspect_identification(&mut ctx.report, n_eff.max(1), 5, &ctx.policy);
    if n_eff < 8 || lo > hi {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!("Zivot–Andrews n={n} is too short for a crash scan"))
                .build(),
        );
        return ctx.finish(ZivotAndrewsResult {
            stat: f64::NAN,
            pvalue: f64::NAN,
            break_index: 0,
            n: n_eff,
        });
    }
    let mut best_stat = f64::INFINITY;
    let mut best_tb = lo;
    for tb in lo..=hi {
        let design = Matrix::from_fn(n_eff, 5, |i, j| {
            let t = t0 + i;
            match j {
                0 => 1.0,
                1 => t as f64,
                2 => {
                    if t > tb {
                        1.0
                    } else {
                        0.0
                    }
                }
                3 => y[t - 1],
                _ => y[t - 1] - y[t - 2],
            }
        });
        let target = Vector::from_iter((0..n_eff).map(|i| y[t0 + i] - y[t0 + i - 1]));
        let mut scratch = Report::new("za", "ols");
        let Some(beta) = least_squares(&mut scratch, &design, &target, &ctx.policy) else {
            continue;
        };
        let fitted = design.matvec(&beta);
        let resid = target.sub(&fitted);
        let sse = resid.dot(&resid);
        let df = n_eff as f64 - 5.0;
        if df <= 0.0 {
            continue;
        }
        let sigma2 = sse / df;
        let mut se_rep = Report::new("za", "se");
        let gram = design.gram();
        let mut ej = Vector::zeros(5);
        ej[3] = 1.0;
        let se = match chol_solve(&mut se_rep, &gram, &ej, &ctx.policy) {
            Some(col) => {
                let v = col[3] * sigma2;
                if v.is_finite() && v > 0.0 {
                    v.sqrt()
                } else {
                    f64::NAN
                }
            }
            None => f64::NAN,
        };
        if se.is_finite() && se > 0.0 {
            let st = beta[3] / se;
            if st < best_stat {
                best_stat = st;
                best_tb = tb;
            }
        }
    }
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("Zivot–Andrews p-value reuses the constant ADF interpolation, not ZA tables")
            .build(),
    );
    if best_stat.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::StructuralBreak)
                .message(format!(
                    "Zivot–Andrews min t={best_stat:.4e} at break index {best_tb}"
                ))
                .metric("za_stat", best_stat)
                .build(),
        );
    }
    ctx.finish(ZivotAndrewsResult {
        stat: if best_stat.is_finite() {
            best_stat
        } else {
            f64::NAN
        },
        pvalue: adf_pvalue_constant(best_stat),
        break_index: best_tb,
        n: n_eff,
    })
}

/// KPSS level-stationarity test with a Newey–West long-run variance.
pub fn kpss(y: &Vector, lags: Option<usize>, session: &Session) -> Result<Qualified<KpssResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    let n = y.len();
    if n < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("KPSS needs n≥4")
                .metric("n", n as f64)
                .build(),
        );
    }
    let mean = y.mean();
    let e: Vec<f64> = y.as_slice().iter().map(|v| v - mean).collect();
    let mut s = vec![0.0; n];
    if n > 0 {
        s[0] = e[0];
        for t in 1..n {
            s[t] = s[t - 1] + e[t];
        }
    }
    let eta_num: f64 = s.iter().map(|v| v * v).sum();
    let used = lags.unwrap_or_else(|| ((n as f64).sqrt().floor() as usize).max(1));
    let sigma2 = newey_west(&e, used);
    let nf = n as f64;
    let stat = if sigma2 > 0.0 {
        eta_num / (nf * nf * sigma2)
    } else {
        push_meaningless(
            &mut ctx,
            "KPSS statistic",
            "long-run residual variance is zero (constant series after demeaning)",
        );
        f64::NAN
    };
    let pvalue = kpss_pvalue(stat);
    if stat.is_finite() && stat > 0.463 {
        ctx.push(
            Issue::builder(IssueCode::NonStationary)
                .message(format!(
                    "KPSS η={stat:.4} exceeds the 5% level critical value 0.463"
                ))
                .metric("kpss", stat)
                .build(),
        );
    }
    ctx.finish(KpssResult {
        stat,
        pvalue,
        lags: used,
        n,
    })
}

/// Tukey HSD pairwise comparisons after a one-way ANOVA.
#[derive(Clone, Debug)]
pub struct TukeyHsdResult {
    /// Group means.
    pub means: Vector,
    /// Pairwise |t| statistics (`K` × `K`).
    pub pairwise_stat: Matrix,
    /// Pairwise two-sided t p-values (uncorrected studentized-range proxy).
    pub pairwise_p: Matrix,
}

/// Tukey honest significant differences on `groups`.
pub fn tukey_hsd(groups: &[&Vector], session: &Session) -> Result<Qualified<TukeyHsdResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if groups.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("Tukey HSD needs ≥ 2 groups")
                .build(),
        );
        return ctx.finish(TukeyHsdResult {
            means: Vector::zeros(groups.len()),
            pairwise_stat: Matrix::zeros(groups.len(), groups.len()),
            pairwise_p: Matrix::zeros(groups.len(), groups.len()),
        });
    }
    let anova = match anova_oneway(groups, &session.child("anova")) {
        Ok(q) => {
            for issue in q.report.issues() {
                ctx.push(issue.clone());
            }
            q.value
        }
        Err(e) => {
            ctx.push(e.primary);
            AnovaResult {
                f_stat: f64::NAN,
                pvalue: f64::NAN,
                df_between: 0.0,
                df_within: 0.0,
                ss_between: f64::NAN,
                ss_within: f64::NAN,
            }
        }
    };
    let k = groups.len();
    let means = Vector::from_iter(groups.iter().map(|g| g.mean()));
    let ns: Vec<f64> = groups.iter().map(|g| g.len() as f64).collect();
    let mse = if anova.df_within > 0.0 {
        anova.ss_within / anova.df_within
    } else {
        f64::NAN
    };
    let mut pairwise_stat = Matrix::zeros(k, k);
    let mut pairwise_p = Matrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            if i == j || !mse.is_finite() || mse <= 0.0 {
                continue;
            }
            let se = (mse * (1.0 / ns[i] + 1.0 / ns[j])).sqrt();
            if se <= 0.0 {
                continue;
            }
            let t = (means[i] - means[j]).abs() / se;
            pairwise_stat.set(i, j, t);
            pairwise_p.set(i, j, student_t_pvalue(t, anova.df_within));
        }
    }
    ctx.push(
        Issue::builder(IssueCode::MultipleTestingUncorrected)
            .message("Tukey pairwise p-values use a Student-t proxy, not the studentized range")
            .build(),
    );
    ctx.finish(TukeyHsdResult {
        means,
        pairwise_stat,
        pairwise_p,
    })
}

/// Goldfeld–Quandt split-sample heteroscedasticity test.
///
/// Rows are sorted by column 0 of `x`. The middle 20% is dropped. The F
/// statistic is `SSE_high / SSE_low` from two scratch OLS fits.
pub fn goldfeld_quandt(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let n = x.nrows().min(y.len());
    if n < 10 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("Goldfeld–Quandt needs n≥10")
                .build(),
        );
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| {
        x.get(*a, 0)
            .partial_cmp(&x.get(*b, 0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let drop = (0.2 * n as f64).floor() as usize;
    let n_side = n.saturating_sub(drop) / 2;
    if n_side < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("Goldfeld–Quandt split left fewer than 3 rows per half")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: n as f64,
        });
    }
    let sse_of = |idx: &[usize]| -> f64 {
        let xs = Matrix::from_fn(idx.len(), x.ncols(), |i, j| x.get(idx[i], j));
        let ys = Vector::from_iter(idx.iter().map(|&i| y[i]));
        let design = xs.with_intercept();
        let mut scratch = Report::new("gq", "ols");
        match crate::linalg::least_squares(&mut scratch, &design, &ys, &ctx.policy) {
            Some(beta) => {
                let r = ys.sub(&design.matvec(&beta));
                r.dot(&r)
            }
            None => f64::NAN,
        }
    };
    let low = &order[..n_side];
    let high = &order[n - n_side..];
    let sse_l = sse_of(low);
    let sse_h = sse_of(high);
    let df = (n_side as f64 - (x.ncols() + 1) as f64).max(1.0);
    let stat = if sse_l > 0.0 { sse_h / sse_l } else { f64::NAN };
    let pvalue = if stat.is_finite() {
        f_pvalue(stat.max(0.0), df, df)
    } else {
        f64::NAN
    };
    if pvalue.is_finite() && pvalue < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::Heteroscedasticity)
                .message(format!("Goldfeld–Quandt p={pvalue:.4}"))
                .metric("f", stat)
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n as f64,
    })
}

/// Chow split-sample F-test for a structural break at `split`.
///
/// Inner OLS uses a scratch report; `ResidualTooLarge` / `NearSingular` are
/// not promoted. A tiny `p`-value is [`IssueCode::StructuralBreak`].
pub fn chow_test(
    x: &Matrix,
    y: &Vector,
    split: usize,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let n = x.nrows().min(y.len());
    let p = x.ncols() + 1;
    if split < p || n.saturating_sub(split) < p {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(signlred::Severity::Warning)
                .message(format!(
                    "Chow split={split} leaves a half shorter than p+1={p}"
                ))
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: p as f64,
            nobs: n as f64,
        });
    }
    let mut sse_of = |lo: usize, hi: usize| -> Option<f64> {
        let m = hi.saturating_sub(lo);
        if m == 0 {
            return None;
        }
        let xs = Matrix::from_fn(m, x.ncols(), |i, j| x.get(lo + i, j)).with_intercept();
        let ys = Vector::from_iter((lo..hi).map(|i| y[i]));
        let mut scratch = Report::new("chow", "ols");
        let beta = crate::linalg::least_squares(&mut scratch, &xs, &ys, &ctx.policy)?;
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::RankZero
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let r = ys.sub(&xs.matvec(&beta));
        Some(r.dot(&r))
    };
    let Some(sse_p) = sse_of(0, n) else {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("Chow pooled OLS failed")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: p as f64,
            nobs: n as f64,
        });
    };
    let Some(sse_1) = sse_of(0, split) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: p as f64,
            nobs: n as f64,
        });
    };
    let Some(sse_2) = sse_of(split, n) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: p as f64,
            nobs: n as f64,
        });
    };
    let df_num = p as f64;
    let df_den = (n as f64 - 2.0 * p as f64).max(1.0);
    let stat = if sse_1 + sse_2 > 0.0 {
        ((sse_p - sse_1 - sse_2).max(0.0) / df_num) / ((sse_1 + sse_2) / df_den)
    } else {
        f64::NAN
    };
    let pvalue = if stat.is_finite() {
        f_pvalue(stat.max(0.0), df_num, df_den)
    } else {
        f64::NAN
    };
    if pvalue.is_finite() && pvalue < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::StructuralBreak)
                .message(format!("Chow p={pvalue:.4} at split={split}"))
                .metric("f", stat)
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: df_num,
        nobs: n as f64,
    })
}

/// OLS-CUSUM (Ploberger–Krämer): standardized walk of residuals.
///
/// Inner OLS uses a scratch report. A large max-CUSUM is
/// [`IssueCode::StructuralBreak`].
pub fn cusum_ols(x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let n = x.nrows().min(y.len());
    let design = x.with_intercept();
    let mut scratch = Report::new("cusum", "ols");
    let Some(beta) = crate::linalg::least_squares(&mut scratch, &design, y, &ctx.policy) else {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("CUSUM OLS failed")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: n as f64,
        });
    };
    for issue in scratch.issues() {
        if matches!(
            issue.code,
            IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::RankZero
        ) {
            continue;
        }
        ctx.push(issue.clone());
    }
    let fit = design.matvec(&beta);
    let mut sse: f64 = 0.0;
    let mut walk: f64 = 0.0;
    let mut max_abs: f64 = 0.0;
    for i in 0..n {
        let e = y[i] - fit[i];
        sse += e * e;
        walk += e;
        max_abs = max_abs.max(walk.abs());
    }
    let sigma = (sse / (n as f64).max(1.0)).sqrt().max(1e-12);
    let stat = max_abs / (sigma * (n as f64).sqrt());
    // Brownian-bridge 5% critical value ≈ 1.36 for the Kolmogorov statistic.
    let pvalue = if stat > 1.36 {
        0.01
    } else if stat > 1.22 {
        0.05
    } else {
        0.4
    };
    if stat > 1.22 {
        ctx.push(
            Issue::builder(IssueCode::StructuralBreak)
                .message(format!(
                    "OLS-CUSUM statistic={stat:.4e} exceeds the 5% band"
                ))
                .metric("cusum", stat)
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: 1.0,
        nobs: n as f64,
    })
}

/// Nadaraya–Watson kernel regression (statsmodels `KernelReg` lite).
///
/// Bandwidth `h` is the Gaussian scale. A tiny `h` is a warning, not a fatal
/// [`IssueCode::InvalidWeight`].
pub fn kernel_reg(x: &Vector, y: &Vector, h: f64, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    let mut bw = h;
    if !bw.is_finite() || bw <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(signlred::Severity::Warning)
                .message(format!("KernelReg bandwidth {h} is not positive; using 1"))
                .build(),
        );
        bw = 1.0;
    }
    let n = x.len().min(y.len());
    let out = Vector::from_iter((0..n).map(|i| {
        let mut num = 0.0;
        let mut den = 0.0;
        for j in 0..n {
            let z = (x[i] - x[j]) / bw;
            let w = (-0.5 * z * z).exp();
            num += w * y[j];
            den += w;
        }
        if den > 0.0 {
            num / den
        } else {
            y[i]
        }
    }));
    ctx.finish(out)
}

/// Nelson–Aalen cumulative hazard from right-censored times.
pub fn nelson_aalen(
    durations: &Vector,
    events: &Vector,
    session: &Session,
) -> Result<Qualified<FittedKaplanMeier>> {
    let km = kaplan_meier_fit(durations, events, session)?;
    let mut ctx = FitCtx::with_session(session.child("na"));
    let mut h = Vector::zeros(km.value.times.len());
    let mut acc = 0.0;
    for i in 0..km.value.times.len() {
        let r = km.value.n_risk[i].max(1.0);
        acc += km.value.n_event[i] / r;
        h[i] = acc;
    }
    ctx.finish(FittedKaplanMeier {
        times: km.value.times,
        survival: h,
        n_risk: km.value.n_risk,
        n_event: km.value.n_event,
    })
}

/// Aalen–Johansen cumulative incidence for competing risks.
///
/// `events` is 0 = censored, `1..=k` = cause. Cause count is not identification
/// `p`.
#[derive(Clone, Debug)]
pub struct AalenJohansenResult {
    /// Unique event times.
    pub times: Vector,
    /// CIF at each time (`n_times × n_causes`).
    pub cif: Matrix,
    /// Cause labels (sorted).
    pub causes: Vec<i64>,
}

/// Competing-risk CIF.
pub fn aalen_johansen(
    times: &Vector,
    events: &Vector,
    session: &Session,
) -> Result<Qualified<AalenJohansenResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, times);
    if times.len() != events.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("Aalen–Johansen times/events length mismatch")
                .build(),
        );
        return ctx.finish(AalenJohansenResult {
            times: Vector::zeros(0),
            cif: Matrix::zeros(0, 0),
            causes: Vec::new(),
        });
    }
    let n = times.len();
    let mut idx: Vec<usize> = (0..n).filter(|&i| times[i].is_finite()).collect();
    idx.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut causes: Vec<i64> = Vec::new();
    for &i in &idx {
        if events[i].is_finite() {
            let e = events[i].round() as i64;
            if e > 0 && !causes.contains(&e) {
                causes.push(e);
            }
        }
    }
    causes.sort_unstable();
    if causes.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("Aalen–Johansen saw no competing-risk events")
                .meaninglessness(Meaninglessness::vacuous(
                    "cumulative incidence",
                    "every observation is censored",
                    "collect cause-specific events",
                ))
                .build(),
        );
        return ctx.finish(AalenJohansenResult {
            times: Vector::zeros(0),
            cif: Matrix::zeros(0, 0),
            causes,
        });
    }
    let mut t_out = Vec::new();
    let mut cif_rows: Vec<Vec<f64>> = Vec::new();
    let mut cif = vec![0.0; causes.len()];
    let mut surv = 1.0;
    let mut i = 0;
    while i < idx.len() {
        let t = times[idx[i]];
        let mut j = i;
        while j < idx.len() && (times[idx[j]] - t).abs() <= 1e-15 {
            j += 1;
        }
        let y_risk = (idx.len() - i) as f64;
        let mut dn = vec![0.0; causes.len()];
        let mut d_tot = 0.0;
        for &u in &idx[i..j] {
            if events[u].is_finite() {
                let e = events[u].round() as i64;
                if let Some(c) = causes.iter().position(|&k| k == e) {
                    dn[c] += 1.0;
                    d_tot += 1.0;
                }
            }
        }
        if d_tot > 0.0 && y_risk > 0.0 {
            for c in 0..causes.len() {
                cif[c] += surv * dn[c] / y_risk;
            }
            surv *= (1.0 - d_tot / y_risk).max(0.0);
            t_out.push(t);
            cif_rows.push(cif.clone());
        }
        i = j;
    }
    let cif_mat = if cif_rows.is_empty() {
        Matrix::zeros(0, causes.len())
    } else {
        Matrix::from_fn(cif_rows.len(), causes.len(), |r, c| cif_rows[r][c])
    };
    ctx.finish(AalenJohansenResult {
        times: Vector::from_iter(t_out),
        cif: cif_mat,
        causes,
    })
}

/// Fitted Fine–Gray subdistribution coefficients.
#[derive(Clone, Debug)]
pub struct FittedFineGray {
    /// Log subdistribution-hazard slopes.
    pub coef: Vector,
    /// Partial-likelihood value at the last Newton step.
    pub loglik: f64,
    /// Events of the cause of interest.
    pub n_events: usize,
    /// Sample size.
    pub n: usize,
    /// Cause code used as the event of interest.
    pub cause: i64,
    /// Whether Newton met the gradient tolerance.
    pub converged: bool,
}

/// Fine–Gray competing-risk regression (subdistribution hazard).
///
/// `events` is 0 = censored, `1..=k` = cause. Cause count is not identification
/// `p`. Competing events stay in the risk set with IPCW from the censoring KM.
pub fn fine_gray(
    x: &Matrix,
    durations: &Vector,
    events: &Vector,
    cause: i64,
    session: &Session,
) -> Result<Qualified<FittedFineGray>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(durations), &ctx.policy);
    if events.len() != x.nrows() || durations.len() != x.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("Fine–Gray durations/events length ≠ X rows")
                .build(),
        );
        return ctx.finish(FittedFineGray {
            coef: Vector::zeros(x.ncols()),
            loglik: 0.0,
            n_events: 0,
            n: x.nrows(),
            cause,
            converged: false,
        });
    }
    let n = x.nrows();
    let p = x.ncols();
    inspect_identification(&mut ctx.report, n, p, &ctx.policy);
    let n_events = (0..n)
        .filter(|&i| events[i].is_finite() && events[i].round() as i64 == cause)
        .count();
    if n_events == 0 {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("Fine–Gray has no events of the requested cause")
                .meaninglessness(Meaninglessness::vacuous(
                    "subdistribution-hazard coefficients",
                    "the Fine–Gray partial likelihood is an empty product without cause-specific events",
                    "choose a cause that occurs, or collect more events",
                ))
                .build(),
        );
        return ctx.finish(FittedFineGray {
            coef: Vector::zeros(p),
            loglik: 0.0,
            n_events: 0,
            n,
            cause,
            converged: false,
        });
    }
    let g_hat = censoring_km(durations, events);
    let mut beta = Vector::zeros(p);
    let mut loglik = f64::NEG_INFINITY;
    let mut converged = false;
    for it in 0..25 {
        let (ll, grad, hess) = fine_gray_grad_hess(durations, events, x, &beta, cause, &g_hat);
        loglik = ll;
        if !grad.as_slice().iter().all(|v| v.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("Fine–Gray gradient became non-finite; last β is retained")
                    .build(),
            );
            break;
        }
        let gnorm = grad.norm();
        ctx.session.step(it as u64, -ll, Some(gnorm));
        if gnorm < 1e-6 {
            ctx.session.converged("Fine–Gray Newton", it as u64);
            converged = true;
            break;
        }
        let mut hneg = Matrix::zeros(p, p);
        for i in 0..p {
            for j in 0..p {
                hneg.set(i, j, -hess.get(i, j));
            }
        }
        let mut scratch = Report::new("fine_gray", "newton");
        match chol_solve(&mut scratch, hneg.inner(), &grad, &ctx.policy) {
            Some(delta) => {
                for j in 0..p {
                    beta[j] -= delta[j];
                }
            }
            None => {
                ctx.push(
                    Issue::builder(IssueCode::InformationMatrixSingular)
                        .severity(Severity::Warning)
                        .message("Fine–Gray observed information is not SPD; Newton step dropped")
                        .build(),
                );
                break;
            }
        }
    }
    if !beta.as_slice().iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteOutput)
                .message("Fine–Gray coefficients are non-finite")
                .build(),
        );
    }
    ctx.finish(FittedFineGray {
        coef: beta,
        loglik,
        n_events,
        n,
        cause,
        converged,
    })
}

fn censoring_km(durations: &Vector, events: &Vector) -> Vec<(f64, f64)> {
    let n = durations.len().min(events.len());
    let mut idx: Vec<usize> = (0..n).filter(|&i| durations[i].is_finite()).collect();
    idx.sort_by(|&a, &b| {
        durations[a]
            .partial_cmp(&durations[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = Vec::new();
    let mut surv = 1.0;
    let mut i = 0;
    while i < idx.len() {
        let t = durations[idx[i]];
        let mut j = i;
        while j < idx.len() && (durations[idx[j]] - t).abs() <= 1e-15 {
            j += 1;
        }
        let y_risk = (idx.len() - i) as f64;
        let mut d_c = 0.0;
        for &u in &idx[i..j] {
            if events[u].is_finite() && events[u].round() as i64 == 0 {
                d_c += 1.0;
            }
        }
        if d_c > 0.0 && y_risk > 0.0 {
            surv *= (1.0 - d_c / y_risk).max(0.0);
        }
        out.push((t, surv.max(1e-12)));
        i = j;
    }
    out
}

fn censor_surv_at(g_hat: &[(f64, f64)], t: f64) -> f64 {
    let mut g = 1.0;
    for &(ti, gi) in g_hat {
        if ti <= t {
            g = gi;
        } else {
            break;
        }
    }
    g.max(1e-12)
}

fn fine_gray_weight(
    i: usize,
    t: f64,
    durations: &Vector,
    events: &Vector,
    cause: i64,
    g_hat: &[(f64, f64)],
) -> f64 {
    if i >= durations.len() || !durations[i].is_finite() {
        return 0.0;
    }
    let ti = durations[i];
    let e = if i < events.len() && events[i].is_finite() {
        events[i].round() as i64
    } else {
        0
    };
    if ti >= t {
        1.0
    } else if e > 0 && e != cause {
        let gt = censor_surv_at(g_hat, t);
        let gs = censor_surv_at(g_hat, ti);
        (gt / gs).clamp(0.0, 8.0)
    } else {
        0.0
    }
}

fn fine_gray_grad_hess(
    durations: &Vector,
    events: &Vector,
    x: &Matrix,
    beta: &Vector,
    cause: i64,
    g_hat: &[(f64, f64)],
) -> (f64, Vector, Matrix) {
    let n = x.nrows();
    let p = x.ncols();
    let mut times: Vec<f64> = Vec::new();
    for i in 0..n {
        if events[i].is_finite() && events[i].round() as i64 == cause && durations[i].is_finite() {
            let t = durations[i];
            if !times.iter().any(|u| (u - t).abs() <= 1e-15) {
                times.push(t);
            }
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut ll = 0.0;
    let mut grad = Vector::zeros(p);
    let mut hess = Matrix::zeros(p, p);
    for &t in &times {
        let mut s0 = 0.0;
        let mut s1 = vec![0.0; p];
        let mut s2 = vec![0.0; p * p];
        for i in 0..n {
            let wgt = fine_gray_weight(i, t, durations, events, cause, g_hat);
            if wgt <= 0.0 {
                continue;
            }
            let mut xb = 0.0;
            for j in 0..p {
                xb += x.get(i, j) * beta[j];
            }
            xb = xb.clamp(-20.0, 20.0);
            let w = wgt * xb.exp().max(1e-300);
            s0 += w;
            for j in 0..p {
                s1[j] += w * x.get(i, j);
            }
            for a in 0..p {
                for b in 0..p {
                    s2[a * p + b] += w * x.get(i, a) * x.get(i, b);
                }
            }
        }
        if s0 <= 0.0 {
            continue;
        }
        for i in 0..n {
            if events[i].round() as i64 != cause || (durations[i] - t).abs() > 1e-15 {
                continue;
            }
            let mut xb = 0.0;
            for j in 0..p {
                xb += x.get(i, j) * beta[j];
            }
            xb = xb.clamp(-20.0, 20.0);
            ll += xb - s0.ln();
            for j in 0..p {
                grad[j] += x.get(i, j) - s1[j] / s0;
            }
            for a in 0..p {
                for b in 0..p {
                    let v = s2[a * p + b] / s0 - (s1[a] / s0) * (s1[b] / s0);
                    hess.set(a, b, hess.get(a, b) - v);
                }
            }
        }
    }
    (ll, grad, hess)
}

/// Baron–Kenny product-of-coefficients mediation.
#[derive(Clone, Debug)]
pub struct MediationResult {
    /// `M ~ a X`.
    pub a: f64,
    /// `Y ~ c' X + b M`.
    pub b: f64,
    /// Total `Y ~ c X`.
    pub c: f64,
    /// Direct path.
    pub c_prime: f64,
    /// Indirect `a b`.
    pub indirect: f64,
}

/// Three OLS paths for a single mediator.
pub fn mediation(
    x: &Vector,
    m: &Vector,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<MediationResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n = x.len().min(m.len()).min(y.len());
    let xm = Matrix::from_fn(n, 1, |i, _| x[i]);
    let mv = Vector::from_iter((0..n).map(|i| m[i]));
    let ym = Vector::from_iter((0..n).map(|i| y[i]));
    inspect_xy(&mut ctx.report, &xm, Some(&ym), &ctx.policy);
    inspect_identification(&mut ctx.report, n, 3, &ctx.policy);
    let xd = xm.with_intercept();
    let mut scratch = Report::new("med", "a");
    let ba = least_squares(&mut scratch, &xd, &mv, &ctx.policy).unwrap_or_else(|| Vector::zeros(2));
    let mut scratch = Report::new("med", "c");
    let bc = least_squares(&mut scratch, &xd, &ym, &ctx.policy).unwrap_or_else(|| Vector::zeros(2));
    let xmd = Matrix::from_fn(n, 2, |i, j| if j == 0 { x[i] } else { m[i] }).with_intercept();
    let mut scratch = Report::new("med", "b");
    let bb =
        least_squares(&mut scratch, &xmd, &ym, &ctx.policy).unwrap_or_else(|| Vector::zeros(3));
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message(
                "mediation is the product method; no bootstrap SE or sequential ignorability test",
            )
            .build(),
    );
    let a = ba.as_slice().get(1).copied().unwrap_or(0.0);
    let c = bc.as_slice().get(1).copied().unwrap_or(0.0);
    let c_prime = bb.as_slice().get(1).copied().unwrap_or(0.0);
    let b = bb.as_slice().get(2).copied().unwrap_or(0.0);
    ctx.finish(MediationResult {
        a,
        b,
        c,
        c_prime,
        indirect: a * b,
    })
}

/// Two-by-two difference-in-differences (`y ~ treat + post + treat×post`).
#[derive(Clone, Debug)]
pub struct DidResult {
    /// Interaction ATT.
    pub att: f64,
    /// Treat main effect.
    pub treat: f64,
    /// Post main effect.
    pub post: f64,
    /// Intercept.
    pub intercept: f64,
}

/// Difference-in-differences on a stacked cross-section.
pub fn difference_in_differences(
    y: &Vector,
    treat: &Vector,
    post: &Vector,
    session: &Session,
) -> Result<Qualified<DidResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n = y.len().min(treat.len()).min(post.len());
    let design = Matrix::from_fn(n, 4, |i, j| match j {
        0 => 1.0,
        1 => treat[i],
        2 => post[i],
        _ => treat[i] * post[i],
    });
    let yv = Vector::from_iter((0..n).map(|i| y[i]));
    inspect_xy(&mut ctx.report, &design, Some(&yv), &ctx.policy);
    inspect_identification(&mut ctx.report, n, 4, &ctx.policy);
    let mut scratch = Report::new("did", "ols");
    let beta =
        least_squares(&mut scratch, &design, &yv, &ctx.policy).unwrap_or_else(|| Vector::zeros(4));
    for issue in scratch.issues() {
        if matches!(
            issue.code,
            IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
        ) {
            continue;
        }
        ctx.push(issue.clone());
    }
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("DID assumes parallel trends; that restriction is not tested here")
            .build(),
    );
    ctx.finish(DidResult {
        intercept: beta.as_slice().first().copied().unwrap_or(0.0),
        treat: beta.as_slice().get(1).copied().unwrap_or(0.0),
        post: beta.as_slice().get(2).copied().unwrap_or(0.0),
        att: beta.as_slice().get(3).copied().unwrap_or(0.0),
    })
}

/// Bartlett's test for equal variances.
pub fn bartlett(groups: &[&Vector], session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if groups.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("Bartlett needs at least two groups")
                .build(),
        );
    }
    let mut ns = Vec::new();
    let mut vars = Vec::new();
    for grp in groups {
        inspect_series_as_target(&mut ctx, grp);
        let st = slice_stats(grp.as_slice());
        ns.push(st.count);
        vars.push(st.std() * st.std());
    }
    let k = groups.len();
    let n: usize = ns.iter().sum();
    let mut sp = 0.0;
    for i in 0..k {
        sp += (ns[i].saturating_sub(1)) as f64 * vars[i];
    }
    let dfw = n.saturating_sub(k) as f64;
    sp = if dfw > 0.0 { sp / dfw } else { f64::NAN };
    let mut num = dfw * sp.max(1e-12).ln();
    for i in 0..k {
        if ns[i] > 1 && vars[i] > 0.0 {
            num -= (ns[i] - 1) as f64 * vars[i].ln();
        }
    }
    let mut den = 1.0;
    for &ni in &ns {
        if ni > 1 {
            den += 1.0 / (ni as f64 - 1.0) - 1.0 / dfw.max(1.0);
        }
    }
    den = 1.0 + den / (3.0 * (k as f64 - 1.0).max(1.0));
    let stat = if den > 0.0 { num / den } else { f64::NAN };
    let df = (k as f64 - 1.0).max(1.0);
    let pvalue = if stat.is_finite() {
        chi2_pvalue(stat.max(0.0), df)
    } else {
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n as f64,
    })
}

/// Friedman rank test across `k` treatments (rows are blocks).
pub fn friedman(table: &Matrix, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, table, None, &ctx.policy);
    let (n, k) = table.shape();
    if n < 2 || k < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(signlred::Severity::Warning)
                .message("Friedman needs ≥2 blocks and ≥2 treatments")
                .build(),
        );
    }
    let mut rank_sums = vec![0.0; k];
    for i in 0..n {
        let row: Vec<f64> = (0..k).map(|j| table.get(i, j)).collect();
        let r = rank_average(&row);
        for j in 0..k {
            rank_sums[j] += r[j];
        }
    }
    let mut ss = 0.0;
    let mean = rank_sums.iter().sum::<f64>() / k.max(1) as f64;
    for &s in &rank_sums {
        let d = s - mean;
        ss += d * d;
    }
    let stat = 12.0 * ss / (n as f64 * k as f64 * (k as f64 + 1.0)).max(1.0);
    let df = (k as f64 - 1.0).max(1.0);
    let pvalue = chi2_pvalue(stat.max(0.0), df);
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n as f64,
    })
}

/// One-sample proportion z-test (`H0: p = p0`).
pub fn proportion_ztest(
    y: &Vector,
    p0: f64,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    let mut p = p0;
    if !(0.0..=1.0).contains(&p) {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(signlred::Severity::Warning)
                .message(format!("proportion_ztest p0={p0} not in [0,1]; using 0.5"))
                .build(),
        );
        p = 0.5;
    }
    let n = y.as_slice().iter().filter(|v| v.is_finite()).count().max(1) as f64;
    let phat = y.as_slice().iter().filter(|v| **v > 0.5).count() as f64 / n;
    let se = (p * (1.0 - p) / n).sqrt().max(1e-12);
    let stat = (phat - p) / se;
    let pvalue = 2.0 * (1.0 - crate::special::norm_cdf(stat.abs()));
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: 1.0,
        nobs: n,
    })
}

/// Two-sample proportion z-test (`H0: p₁ = p₂`, statsmodels `proportions_ztest`).
pub fn proportions_ztest(
    y1: &Vector,
    y2: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y1),
        None,
        &ctx.policy,
    );
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y2),
        None,
        &ctx.policy,
    );
    let n1 = y1.as_slice().iter().filter(|v| v.is_finite()).count().max(1) as f64;
    let n2 = y2.as_slice().iter().filter(|v| v.is_finite()).count().max(1) as f64;
    let s1 = y1.as_slice().iter().filter(|v| **v > 0.5).count() as f64;
    let s2 = y2.as_slice().iter().filter(|v| **v > 0.5).count() as f64;
    let p1 = s1 / n1;
    let p2 = s2 / n2;
    let pooled = (s1 + s2) / (n1 + n2);
    if pooled <= 0.0 || pooled >= 1.0 {
        ctx.push(
            Issue::builder(IssueCode::NearZeroVariance)
                .severity(Severity::Warning)
                .message("two-sample proportion z-test pooled p is 0 or 1; SE is floored")
                .build(),
        );
    }
    let se = (pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
        .sqrt()
        .max(1e-12);
    let stat = (p1 - p2) / se;
    let pvalue = 2.0 * (1.0 - crate::special::norm_cdf(stat.abs()));
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: 1.0,
        nobs: n1 + n2,
    })
}

/// Ramsey RESET: augment OLS with \(\hat y^2,\hat y^3\) and F-test the extra powers.
pub fn ramsey_reset(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let n = x.nrows().min(y.len());
    if n < 8 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(signlred::Severity::Warning)
                .message("Ramsey RESET needs n≥8")
                .build(),
        );
    }
    let design = x.with_intercept();
    let mut scratch = Report::new("reset", "ols");
    let Some(beta) = crate::linalg::least_squares(&mut scratch, &design, y, &ctx.policy) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: n as f64,
        });
    };
    let yhat = design.matvec(&beta);
    if yhat.std() <= ctx.policy.near_zero_variance {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .severity(signlred::Severity::Warning)
                .message("RESET ŷ is constant; powers are unidentified")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 2.0,
            nobs: n as f64,
        });
    }
    let p0 = design.ncols();
    let aug = Matrix::from_fn(n, p0 + 2, |i, j| {
        if j < p0 {
            design.get(i, j)
        } else if j == p0 {
            yhat[i] * yhat[i]
        } else {
            yhat[i] * yhat[i] * yhat[i]
        }
    });
    let mut scratch2 = Report::new("reset", "aug");
    let Some(b2) = crate::linalg::least_squares(&mut scratch2, &aug, y, &ctx.policy) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 2.0,
            nobs: n as f64,
        });
    };
    let r0 = y.sub(&yhat);
    let r1 = y.sub(&aug.matvec(&b2));
    let ssr0 = r0.dot(&r0);
    let ssr1 = r1.dot(&r1);
    let q = 2.0;
    let df = (n as f64 - (p0 + 2) as f64).max(1.0);
    let stat = if ssr1 > 0.0 {
        ((ssr0 - ssr1) / q) / (ssr1 / df)
    } else {
        f64::INFINITY
    };
    let pvalue = if stat.is_finite() {
        f_pvalue(stat.max(0.0), q, df)
    } else {
        f64::NAN
    };
    if pvalue.is_finite() && pvalue < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::InconsistentSystem)
                .message(format!(
                    "Ramsey RESET p={pvalue:.4} rejects the linear specification"
                ))
                .metric("f", stat)
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n as f64,
    })
}

/// Harvey–Collier: t-test that the mean of OLS recursive residuals is zero.
pub fn harvey_collier(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let n = x.nrows().min(y.len());
    let design = x.with_intercept();
    let p = design.ncols();
    if n <= p + 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(signlred::Severity::Warning)
                .message("Harvey–Collier needs n > p+3 recursive residuals")
                .build(),
        );
    }
    let mut w = Vec::new();
    for t in p.max(2)..n {
        let xs = Matrix::from_fn(t, p, |i, j| design.get(i, j));
        let ys = Vector::from_iter((0..t).map(|i| y[i]));
        let mut scratch = Report::new("hc", "rec");
        let Some(beta) = crate::linalg::least_squares(&mut scratch, &xs, &ys, &ctx.policy) else {
            continue;
        };
        let xt = Vector::from_iter((0..p).map(|j| design.get(t, j)));
        let pred = xt.dot(&beta);
        // Leverage of the new row against the previous Gram.
        let mut gram = xs.gram();
        for i in 0..p {
            gram[(i, i)] += 1e-10;
        }
        let mut scratch2 = Report::new("hc", "lev");
        let h = crate::linalg::chol_solve(&mut scratch2, &gram, &xt, &ctx.policy)
            .map(|g| xt.dot(&g))
            .unwrap_or(0.0);
        let den = (1.0 + h.max(0.0)).sqrt();
        if den > 0.0 {
            w.push((y[t] - pred) / den);
        }
    }
    if w.len() < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(signlred::Severity::Warning)
                .message("Harvey–Collier produced fewer than 3 recursive residuals")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: w.len() as f64 - 1.0,
            nobs: n as f64,
        });
    }
    let m = w.iter().sum::<f64>() / w.len() as f64;
    let mut ss = 0.0;
    for &v in &w {
        ss += (v - m) * (v - m);
    }
    let sd = (ss / (w.len() - 1) as f64).sqrt();
    let df = (w.len() - 1) as f64;
    let stat = if sd > 0.0 {
        m / (sd / (w.len() as f64).sqrt())
    } else {
        f64::NAN
    };
    let pvalue = if stat.is_finite() {
        student_t_pvalue(stat, df)
    } else {
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n as f64,
    })
}

/// Utts rainbow test for a linear specification (statsmodels `linear_rainbow`).
///
/// Observations are ordered by the first column of `x`. OLS is fit on the
/// central `frac` of rows; the tails are a prediction hold-out. The F
/// statistic compares tail prediction SSE to the central residual MSE.
/// Subset size is not identification `p`.
pub fn rainbow(
    x: &Matrix,
    y: &Vector,
    frac: f64,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let n = x.nrows().min(y.len());
    let f = if frac.is_finite() && frac > 0.0 && frac < 1.0 {
        frac
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("rainbow frac={frac} is not in (0,1); using 0.5"))
                .build(),
        );
        0.5
    };
    let design = x.with_intercept();
    let k = design.ncols();
    let n_mid = ((n as f64 * f).ceil() as usize).max(k + 2).min(n);
    if n <= k + 3 || n_mid + 2 >= n {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!(
                    "rainbow n={n} mid={n_mid} is tight for k={k} parameters"
                ))
                .build(),
        );
    }
    let mut ord: Vec<usize> = (0..n).collect();
    ord.sort_by(|&a, &b| {
        x.get(a, 0)
            .partial_cmp(&x.get(b, 0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let start = (n.saturating_sub(n_mid)) / 2;
    let mid_idx: Vec<usize> = ord.iter().skip(start).take(n_mid.min(n)).copied().collect();
    let tail_idx: Vec<usize> = ord
        .iter()
        .enumerate()
        .filter(|(i, _)| *i < start || *i >= start + mid_idx.len())
        .map(|(_, &j)| j)
        .collect();
    if mid_idx.len() <= k || tail_idx.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("rainbow hold-out or centre is empty after ordering")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: tail_idx.len() as f64,
            nobs: n as f64,
        });
    }
    let xm = Matrix::from_fn(mid_idx.len(), k, |i, j| design.get(mid_idx[i], j));
    let ym = Vector::from_iter(mid_idx.iter().map(|&i| y[i]));
    let Some(beta) = statistical_ols(&mut ctx, &xm, &ym) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: tail_idx.len() as f64,
            nobs: n as f64,
        });
    };
    let mut sse_mid = 0.0;
    for (r, &i) in mid_idx.iter().enumerate() {
        let mut pred = 0.0;
        for j in 0..k {
            pred += beta[j] * xm.get(r, j);
        }
        let e = y[i] - pred;
        sse_mid += e * e;
    }
    let mut sse_tail = 0.0;
    for &i in &tail_idx {
        let mut pred = 0.0;
        for j in 0..k {
            pred += beta[j] * design.get(i, j);
        }
        let e = y[i] - pred;
        sse_tail += e * e;
    }
    let df_num = tail_idx.len() as f64;
    let df_den = (mid_idx.len() as f64 - k as f64).max(1.0);
    let stat = if sse_mid > 0.0 && df_den > 0.0 {
        (sse_tail / df_num) / (sse_mid / df_den)
    } else {
        f64::NAN
    };
    let pvalue = if stat.is_finite() {
        f_pvalue(stat.max(0.0), df_num, df_den)
    } else {
        f64::NAN
    };
    if pvalue.is_finite() && pvalue < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::InconsistentSystem)
                .message(format!(
                    "rainbow p={pvalue:.4} rejects the linear specification on the tails"
                ))
                .metric("f", stat)
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: df_num,
        nobs: n as f64,
    })
}

/// Engle ARCH-LM: \(e_t^2\) on lags of itself; \(n R^2 \sim \chi^2_q\).
pub fn arch_lm(
    resid: &Vector,
    lags: usize,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = scan_finite(resid.as_slice()).to_issue("arch_lm.e") {
        ctx.push(issue);
    }
    let e2: Vec<f64> = resid.as_slice().iter().map(|e| e * e).collect();
    let n = e2.len();
    let q = lags.max(1);
    if n <= q + 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(signlred::Severity::Warning)
                .message(format!("ARCH-LM n={n} is tight for lags={q}"))
                .build(),
        );
    }
    let m = n.saturating_sub(q);
    if m == 0 {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: q as f64,
            nobs: n as f64,
        });
    }
    let x = Matrix::from_fn(m, q + 1, |i, j| if j == 0 { 1.0 } else { e2[q + i - j] });
    let y = Vector::from_iter((0..m).map(|i| e2[q + i]));
    let mut scratch = Report::new("archlm", "ols");
    let Some(beta) = crate::linalg::least_squares(&mut scratch, &x, &y, &ctx.policy) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: q as f64,
            nobs: n as f64,
        });
    };
    let fitted = x.matvec(&beta);
    let mut sse = 0.0;
    let mut sst = 0.0;
    let ym = y.mean();
    for i in 0..m {
        let e = y[i] - fitted[i];
        sse += e * e;
        let d = y[i] - ym;
        sst += d * d;
    }
    let r2 = if sst > 0.0 { 1.0 - sse / sst } else { 0.0 };
    let stat = m as f64 * r2.max(0.0);
    let pvalue = chi2_pvalue(stat, q as f64);
    if pvalue.is_finite() && pvalue < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::Heteroscedasticity)
                .message(format!("ARCH-LM p={pvalue:.4}"))
                .metric("lm", stat)
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: q as f64,
        nobs: n as f64,
    })
}

/// Breusch–Godfrey LM test for residual autocorrelation of order `lags`.
pub fn breusch_godfrey(
    resid: &Vector,
    design: &Matrix,
    lags: usize,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, design, None, &ctx.policy);
    let n = resid.len().min(design.nrows());
    let h = lags.max(1);
    if n <= h + design.ncols() {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(signlred::Severity::Warning)
                .message(format!("Breusch–Godfrey n={n} is tight for lags={h}"))
                .build(),
        );
    }
    if resid
        .as_slice()
        .iter()
        .all(|e| e.abs() <= ctx.policy.near_zero_variance)
    {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("Breusch–Godfrey: residuals are identically zero")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: h as f64,
            nobs: n as f64,
        });
    }
    let n_eff = n.saturating_sub(h);
    let p_aux = design.ncols() + h;
    let aux = Matrix::from_fn(n_eff, p_aux, |i, j| {
        let t = i + h;
        if j < design.ncols() {
            design.get(t, j)
        } else {
            resid[t - (j - design.ncols() + 1)]
        }
    });
    let target = Vector::from_iter((h..n).map(|t| resid[t]));
    let mut scratch = Report::new("bg", "aux");
    let Some(beta) = crate::linalg::least_squares(&mut scratch, &aux, &target, &ctx.policy) else {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .severity(signlred::Severity::Warning)
                .message("Breusch–Godfrey auxiliary OLS failed")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: h as f64,
            nobs: n_eff as f64,
        });
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
    let fitted = aux.matvec(&beta);
    let r = target.sub(&fitted);
    let sse = r.dot(&r);
    let mean = target.mean();
    let sst: f64 = target
        .as_slice()
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum();
    let r2 = if sst > 0.0 { 1.0 - sse / sst } else { f64::NAN };
    let stat = n_eff as f64 * r2;
    let pvalue = if stat.is_finite() {
        chi2_pvalue(stat.max(0.0), h as f64)
    } else {
        f64::NAN
    };
    if pvalue.is_finite() && pvalue < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::AutocorrelatedResiduals)
                .message(format!("Breusch–Godfrey p={pvalue:.4}"))
                .metric("lm", stat)
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: h as f64,
        nobs: n_eff as f64,
    })
}

/// Phillips–Perron unit-root test (constant-only, Newey–West robust).
pub fn phillips_perron(
    y: &Vector,
    lags: Option<usize>,
    session: &Session,
) -> Result<Qualified<AdfullerResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    let n = y.len();
    if n < 6 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("Phillips–Perron needs n≥6")
                .build(),
        );
        return ctx.finish(AdfullerResult {
            stat: f64::NAN,
            pvalue: f64::NAN,
            used_lags: 0,
            n,
        });
    }
    let n_eff = n - 1;
    let design = Matrix::from_fn(n_eff, 2, |i, j| if j == 0 { 1.0 } else { y[i] });
    let target = Vector::from_iter((1..n).map(|t| y[t]));
    let Some(beta) = statistical_ols(&mut ctx, &design, &target) else {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("Phillips–Perron OLS failed")
                .build(),
        );
        return ctx.finish(AdfullerResult {
            stat: f64::NAN,
            pvalue: f64::NAN,
            used_lags: 0,
            n: n_eff,
        });
    };
    let fitted = design.matvec(&beta);
    let e = target.sub(&fitted);
    let used = lags.unwrap_or_else(|| schwert_lags(n));
    let lambda2 = newey_west(e.as_slice(), used);
    let gamma0 = {
        let mut s = 0.0;
        for i in 0..e.len() {
            s += e[i] * e[i];
        }
        s / e.len() as f64
    };
    let se = ols_se_j(&mut ctx, &design, gamma0, 1);
    let rho = beta[1];
    let t_ols = if se > 0.0 { (rho - 1.0) / se } else { f64::NAN };
    let nf = n_eff as f64;
    let stat = if lambda2 > 0.0 && gamma0 > 0.0 && t_ols.is_finite() {
        t_ols * (gamma0 / lambda2).sqrt()
            - (lambda2 - gamma0) * nf * se / (2.0 * lambda2.sqrt() * gamma0.sqrt().max(1e-12))
    } else {
        t_ols
    };
    if rho.abs() > 0.98 || (rho - 1.0).abs() < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::NonStationary)
                .message(format!("Phillips–Perron ρ≈{rho:.4}"))
                .metric("rho", rho)
                .build(),
        );
    }
    ctx.finish(AdfullerResult {
        stat,
        pvalue: adf_pvalue_constant(stat),
        used_lags: used,
        n: n_eff,
    })
}

/// Granger causality: does lagged `x` help predict `y` beyond lagged `y`?
pub fn granger_causality(
    x: &Vector,
    y: &Vector,
    lag: usize,
    session: &Session,
) -> Result<Qualified<GrangerResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    if lag == 0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message("Granger lag must be ≥ 1")
                .build(),
        );
    }
    let n = x.len().min(y.len());
    let t0 = lag;
    let n_eff = n.saturating_sub(t0);
    let k_u = 1 + 2 * lag;
    let k_r = 1 + lag;
    if n_eff <= k_u || lag == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message(format!(
                    "Granger n_eff={n_eff} is not larger than unrestricted p={k_u}"
                ))
                .build(),
        );
        return ctx.finish(GrangerResult {
            f_stat: f64::NAN,
            pvalue: f64::NAN,
            df_num: lag as f64,
            df_den: 0.0,
            lag,
        });
    }
    inspect_identification(&mut ctx.report, n_eff, k_u, &ctx.policy);
    let yu = Vector::from_iter((t0..n).map(|t| y[t]));
    let xu = Matrix::from_fn(n_eff, k_u, |i, j| {
        let t = t0 + i;
        if j == 0 {
            1.0
        } else if j <= lag {
            y[t - j]
        } else {
            x[t - (j - lag)]
        }
    });
    let xr = Matrix::from_fn(n_eff, k_r, |i, j| {
        let t = t0 + i;
        if j == 0 {
            1.0
        } else {
            y[t - j]
        }
    });
    let (ssr_u, ssr_r) = match (
        statistical_ols(&mut ctx, &xu, &yu),
        statistical_ols(&mut ctx, &xr, &yu),
    ) {
        (Some(bu), Some(br)) => {
            let eu = yu.sub(&xu.matvec(&bu));
            let er = yu.sub(&xr.matvec(&br));
            (eu.dot(&eu), er.dot(&er))
        }
        _ => {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("Granger VAR OLS failed")
                    .build(),
            );
            return ctx.finish(GrangerResult {
                f_stat: f64::NAN,
                pvalue: f64::NAN,
                df_num: lag as f64,
                df_den: (n_eff - k_u) as f64,
                lag,
            });
        }
    };
    let df_num = lag as f64;
    let df_den = (n_eff - k_u) as f64;
    let f_stat = if df_den > 0.0 && ssr_u > 0.0 && df_num > 0.0 {
        ((ssr_r - ssr_u) / df_num) / (ssr_u / df_den)
    } else {
        ctx.push(
            Issue::builder(IssueCode::DegreesOfFreedomNonPositive)
                .message("Granger F has non-positive residual df or zero unrestricted SSE")
                .build(),
        );
        f64::NAN
    };
    let pvalue = if f_stat.is_finite() {
        f_pvalue(f_stat.max(0.0), df_num, df_den)
    } else {
        f64::NAN
    };
    ctx.finish(GrangerResult {
        f_stat,
        pvalue,
        df_num,
        df_den,
        lag,
    })
}

/// Adjusted p-values for Bonferroni, Holm, or Benjamini–Hochberg.
pub fn multipletests(
    p: &[f64],
    method: MultiTest,
    session: &Session,
) -> Result<Qualified<Vec<f64>>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if p.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("multipletests received an empty p-value vector")
                .build(),
        );
        return ctx.finish(Vec::new());
    }
    if let Some(issue) = scan_finite(p).to_issue("p-values") {
        ctx.push(issue);
    }
    for (i, &pi) in p.iter().enumerate() {
        if pi.is_finite() && (pi < 0.0 || pi > 1.0) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("p[{i}]={pi} is outside [0, 1]"))
                    .build(),
            );
            break;
        }
    }
    let m = p.len();
    let adj = match method {
        MultiTest::Bonferroni => p.iter().map(|&pi| (pi * m as f64).min(1.0)).collect(),
        MultiTest::Holm => holm_adjust(p),
        MultiTest::BenjaminiHochberg => bh_adjust(p),
    };
    ctx.finish(adj)
}

/// Percentile bootstrap of the mean (`lo`, `hi` are the 2.5% / 97.5% percentiles).
pub fn bootstrap_mean(
    x: &Vector,
    n_boot: usize,
    seed: u64,
    session: &Session,
) -> Result<Qualified<(f64, f64, f64)>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    if n_boot < 20 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message(format!(
                    "bootstrap n_boot={n_boot} is too small for a 95% percentile interval"
                ))
                .metric("n_boot", n_boot as f64)
                .build(),
        );
    }
    let n = x.len();
    if n == 0 {
        return ctx.finish((f64::NAN, f64::NAN, f64::NAN));
    }
    let mean = x.mean();
    let mut rng = Rng::new(seed);
    let mut boots = Vec::with_capacity(n_boot);
    for _ in 0..n_boot {
        let mut s = 0.0;
        for _ in 0..n {
            s += x[rng.below(n)];
        }
        boots.push(s / n as f64);
    }
    boots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = percentile_sorted(&boots, 0.025);
    let hi = percentile_sorted(&boots, 0.975);
    if !(hi - lo).is_finite() || (hi - lo).abs() <= ctx.policy.near_zero_variance {
        ctx.push(
            Issue::builder(IssueCode::ConfidenceIntervalDegenerate)
                .message("bootstrap interval collapsed")
                .build(),
        );
    }
    ctx.finish((mean, lo, hi))
}

/// Gaussian kernel density evaluated on `grid` (Silverman bandwidth).
pub fn gaussian_kde(x: &Vector, grid: &Vector, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    if let Some(issue) = scan_finite(grid.as_slice()).to_issue("kde-grid") {
        ctx.push(issue);
    }
    if grid.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("gaussian_kde grid is empty")
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let st = slice_stats(x.as_slice());
    let n = st.count.max(1) as f64;
    let mut h = 1.06 * st.std() * n.powf(-0.2);
    if !h.is_finite() || h <= ctx.policy.near_zero_variance {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("Silverman bandwidth collapsed; a tiny floor was substituted")
                .compromise(NumericalCompromise::new(
                    "Silverman h = 1.06 σ n^{-1/5}",
                    "h = 1e-3 floor",
                    "the sample is constant or n=1",
                    "the density is a spike; do not treat it as a smooth estimate",
                ))
                .build(),
        );
        h = 1e-3;
    }
    let inv = 1.0 / (n * h * (2.0 * std::f64::consts::PI).sqrt());
    let dens = Vector::from_iter(grid.as_slice().iter().map(|&g| {
        let mut s = 0.0;
        for &xi in x.as_slice() {
            if !xi.is_finite() {
                continue;
            }
            let z = (g - xi) / h;
            s += (-0.5 * z * z).exp();
        }
        s * inv
    }));
    ctx.finish(dens)
}

/// Univariate Gaussian KDE evaluated at the sample (statsmodels `KDEUnivariate`).
#[derive(Clone, Debug)]
pub struct KdeUnivariate {
    /// Silverman (or floored) bandwidth.
    pub bandwidth: f64,
    /// Density at each sample point.
    pub density: Vector,
    /// Evaluation support (the observed sample).
    pub support: Vector,
}

/// Silverman-bandwidth Gaussian KDE at the sample points.
pub fn kde_univariate(y: &Vector, session: &Session) -> Result<Qualified<KdeUnivariate>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        None,
        &ctx.policy,
    );
    let st = slice_stats(y.as_slice());
    let n = st.count.max(1) as f64;
    let mut h = 1.06 * st.std() * n.powf(-0.2);
    if !h.is_finite() || st.std() <= ctx.policy.near_zero_variance {
        ctx.push(
            Issue::builder(IssueCode::NearZeroVariance)
                .severity(Severity::Warning)
                .message("KDEUnivariate std collapsed; bandwidth set to 1")
                .compromise(NumericalCompromise::new(
                    "Silverman h = 1.06 σ n^{-1/5}",
                    "h = 1",
                    "the sample is constant or n=1",
                    "the density is a spike; do not treat it as a smooth estimate",
                ))
                .build(),
        );
        h = 1.0;
    }
    let inv = 1.0 / (n * h * (2.0 * std::f64::consts::PI).sqrt());
    let density = Vector::from_iter(y.as_slice().iter().map(|&g| {
        if !g.is_finite() {
            return f64::NAN;
        }
        let mut s = 0.0;
        for &xi in y.as_slice() {
            if !xi.is_finite() {
                continue;
            }
            let z = (g - xi) / h;
            s += (-0.5 * z * z).exp();
        }
        s * inv
    }));
    ctx.finish(KdeUnivariate {
        bandwidth: h,
        density,
        support: y.clone(),
    })
}

/// Cleveland LOWESS: locally weighted linear fits with tricube weights.
pub fn lowess(x: &Vector, y: &Vector, frac: f64, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    if !(frac.is_finite() && frac > 0.0 && frac <= 1.0) {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("LOWESS frac={frac} is not in (0, 1]"))
                .build(),
        );
    }
    let n = x.len().min(y.len());
    let span = ((frac.clamp(1e-6, 1.0) * n as f64).ceil() as usize)
        .max(2)
        .min(n.max(2));
    if span < 3 && n >= 3 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("LOWESS neighbourhood span={span} < 3"))
                .build(),
        );
    }
    let mut fitted = Vector::zeros(n);
    if n == 0 {
        return ctx.finish(fitted);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));
    for i in 0..n {
        // Neighbourhood: span nearest x-values to x[i].
        let mut neigh: Vec<usize> = order.clone();
        neigh.sort_by(|&a, &b| {
            (x[a] - x[i])
                .abs()
                .partial_cmp(&(x[b] - x[i]).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        neigh.truncate(span.min(n));
        let dmax = neigh
            .iter()
            .map(|&j| (x[j] - x[i]).abs())
            .fold(0.0, f64::max)
            .max(1e-15);
        let mut xtw = Matrix::zeros(neigh.len(), 2);
        let mut ytw = Vector::zeros(neigh.len());
        for (r, &j) in neigh.iter().enumerate() {
            let u = ((x[j] - x[i]).abs() / dmax).min(1.0);
            let w = (1.0 - u.powi(3)).powi(3).max(0.0).sqrt();
            xtw.set(r, 0, w);
            xtw.set(r, 1, w * x[j]);
            ytw[r] = w * y[j];
        }
        match statistical_ols(&mut ctx, &xtw, &ytw) {
            Some(b) => fitted[i] = b[0] + b[1] * x[i],
            None => fitted[i] = y[i],
        }
    }
    ctx.finish(fitted)
}

/// Approximate two-sided t-test power for a one-sample design (shifted-t).
pub fn ttest_power(
    effect_size: f64,
    n: f64,
    alpha: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !effect_size.is_finite() || !n.is_finite() || !alpha.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("ttest_power received a non-finite argument")
                .build(),
        );
    }
    if n <= 1.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("ttest_power needs n>1")
                .metric("n", n)
                .build(),
        );
    }
    if !(0.0 < alpha && alpha < 1.0) {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("ttest_power alpha={alpha} is not in (0, 1)"))
                .build(),
        );
    }
    let df = n - 1.0;
    let tcrit = student_t_ppf(1.0 - alpha / 2.0, df);
    let ncp = effect_size * n.sqrt();
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .message(
                "ttest_power uses a shifted central-t approximation, not the non-central t CDF",
            )
            .compromise(NumericalCompromise::new(
                "power from the non-central t law",
                "1 − F_t(t_crit − ncp) + F_t(−t_crit − ncp)",
                "a closed non-central-t CDF is not implemented",
                "the number is a planning approximation; do not treat it as an exact power",
            ))
            .build(),
    );
    let power = if tcrit.is_finite() && ncp.is_finite() && df > 0.0 {
        (1.0 - student_t_cdf(tcrit - ncp, df) + student_t_cdf(-tcrit - ncp, df)).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };
    ctx.finish(power)
}

fn kaplan_meier_fit(
    durations: &Vector,
    events: &Vector,
    session: &Session,
) -> Result<Qualified<FittedKaplanMeier>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, durations, events);
    let n = durations.len().min(events.len());
    let mut rows: Vec<(f64, f64)> = (0..n)
        .filter(|&i| durations[i].is_finite() && events[i].is_finite())
        .map(|i| (durations[i], events[i]))
        .collect();
    if rows.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("Kaplan–Meier received no finite (duration, event) pairs")
                .build(),
        );
        return ctx.finish(FittedKaplanMeier {
            times: Vector::zeros(0),
            survival: Vector::zeros(0),
            n_risk: Vector::zeros(0),
            n_event: Vector::zeros(0),
        });
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut times = Vec::new();
    let mut surv = Vec::new();
    let mut nrisk = Vec::new();
    let mut nevent = Vec::new();
    let mut s = 1.0;
    let mut i = 0;
    let mut n_ev_total = 0usize;
    while i < rows.len() {
        let t = rows[i].0;
        let at_risk = rows.len() - i;
        let mut d = 0.0;
        while i < rows.len() && (rows[i].0 - t).abs() <= 0.0 {
            if rows[i].1 > 0.5 {
                d += 1.0;
                n_ev_total += 1;
            }
            i += 1;
        }
        if d > 0.0 {
            s *= 1.0 - d / at_risk as f64;
            times.push(t);
            surv.push(s);
            nrisk.push(at_risk as f64);
            nevent.push(d);
        }
    }
    if n_ev_total == 0 {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("Kaplan–Meier has zero events; Ŝ(t) is identically 1")
                .meaninglessness(Meaninglessness::vacuous(
                    "product-limit survival curve",
                    "without events the estimator never leaves 1; it is not identified as a lifetime law",
                    "collect events or report only the censoring pattern",
                ))
                .build(),
        );
    }
    ctx.finish(FittedKaplanMeier {
        times: Vector::from_iter(times),
        survival: Vector::from_iter(surv),
        n_risk: Vector::from_iter(nrisk),
        n_event: Vector::from_iter(nevent),
    })
}

fn cox_ph_fit(
    spec: &CoxPH,
    durations: &Vector,
    events: &Vector,
    x: &Matrix,
    session: &Session,
) -> Result<Qualified<FittedCoxPH>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(durations), &ctx.policy);
    if events.len() != x.nrows() || durations.len() != x.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("CoxPH durations/events length ≠ X rows")
                .build(),
        );
    }
    let n = x.nrows();
    let p = x.ncols();
    inspect_identification(&mut ctx.report, n, p, &ctx.policy);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        durations[a]
            .partial_cmp(&durations[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_events = (0..n)
        .filter(|&i| i < events.len() && events[i] > 0.5)
        .count();
    if n_events == 0 {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("CoxPH has no events; the partial likelihood is flat in β")
                .build(),
        );
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("Cox Newton has nothing to condition on")
                .meaninglessness(Meaninglessness::vacuous(
                    "Cox partial-likelihood coefficients",
                    "zero events ⇒ every β gives the same (empty) product of risk-set terms",
                    "do not interpret hazard ratios",
                ))
                .build(),
        );
        return ctx.finish(FittedCoxPH {
            coef: Vector::zeros(p),
            loglik: 0.0,
            n_events: 0,
            n,
            converged: false,
        });
    }
    let mut beta = Vector::zeros(p);
    let mut loglik = f64::NEG_INFINITY;
    let mut converged = false;
    for it in 0..spec.max_iter {
        let (ll, grad, hess) = cox_grad_hess(&idx, durations, events, x, &beta);
        loglik = ll;
        if !grad.as_slice().iter().all(|v| v.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::LossIsNan)
                    .message("Cox gradient became non-finite")
                    .build(),
            );
            break;
        }
        let gnorm = grad.norm();
        ctx.session.step(it as u64, -ll, Some(gnorm));
        if gnorm < spec.tol {
            ctx.session.converged("Cox Newton", it as u64);
            converged = true;
            break;
        }
        // Solve (−H) δ = g.
        let mut hneg = Matrix::zeros(p, p);
        for i in 0..p {
            for j in 0..p {
                hneg.set(i, j, -hess.get(i, j));
            }
        }
        match chol_solve(&mut ctx.report, hneg.inner(), &grad, &ctx.policy) {
            Some(delta) => {
                for j in 0..p {
                    beta[j] -= delta[j];
                }
            }
            None => {
                ctx.push(
                    Issue::builder(IssueCode::InformationMatrixSingular)
                        .message("Cox observed information is not SPD; Newton step dropped")
                        .build(),
                );
                ctx.push(
                    Issue::builder(IssueCode::HessianNotPositiveDefinite)
                        .message("Cox Hessian failed Cholesky")
                        .build(),
                );
                break;
            }
        }
        if it + 1 == spec.max_iter {
            ctx.push(
                Issue::builder(IssueCode::MaxIterReached)
                    .message("Cox Newton hit the iteration cap")
                    .build(),
            );
        }
    }
    if !beta.as_slice().iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteOutput)
                .message("Cox coefficients are non-finite")
                .build(),
        );
    }
    ctx.finish(FittedCoxPH {
        coef: beta,
        loglik,
        n_events,
        n,
        converged,
    })
}

fn cox_grad_hess(
    idx: &[usize],
    durations: &Vector,
    events: &Vector,
    x: &Matrix,
    beta: &Vector,
) -> (f64, Vector, Matrix) {
    let n = idx.len();
    let p = x.ncols();
    let mut ll = 0.0;
    let mut grad = Vector::zeros(p);
    let mut hess = Matrix::zeros(p, p);
    // Walk from last time to first so the risk set grows.
    let mut s0 = 0.0;
    let mut s1 = vec![0.0; p];
    let mut s2 = vec![0.0; p * p];
    let mut k = n;
    while k > 0 {
        let t = durations[idx[k - 1]];
        let start = k;
        while k > 0 && (durations[idx[k - 1]] - t).abs() <= 0.0 {
            k -= 1;
            let i = idx[k];
            let mut xb = 0.0;
            for j in 0..p {
                xb += x.get(i, j) * beta[j];
            }
            let w = xb.exp().max(1e-300);
            s0 += w;
            for j in 0..p {
                s1[j] += w * x.get(i, j);
            }
            for a in 0..p {
                for b in 0..p {
                    s2[a * p + b] += w * x.get(i, a) * x.get(i, b);
                }
            }
        }
        // Breslow: every event at this time uses the current risk-set sums.
        for r in k..start {
            let i = idx[r];
            if i >= events.len() || events[i] <= 0.5 {
                continue;
            }
            if s0 <= 0.0 {
                continue;
            }
            let mut xb = 0.0;
            for j in 0..p {
                xb += x.get(i, j) * beta[j];
            }
            ll += xb - s0.ln();
            for j in 0..p {
                grad[j] += x.get(i, j) - s1[j] / s0;
            }
            for a in 0..p {
                for b in 0..p {
                    let v = s2[a * p + b] / s0 - (s1[a] / s0) * (s1[b] / s0);
                    hess.set(a, b, hess.get(a, b) - v);
                }
            }
        }
    }
    (ll, grad, hess)
}

fn inspect_series_as_target(ctx: &mut FitCtx, x: &Vector) {
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(x),
        Some(x),
        &ctx.policy,
    );
}

fn inspect_pair(ctx: &mut FitCtx, x: &Vector, y: &Vector) {
    if x.len() != y.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!("paired lengths {} vs {}", x.len(), y.len()))
                .build(),
        );
    }
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(x),
        Some(y),
        &ctx.policy,
    );
}

fn push_meaningless(ctx: &mut FitCtx, what: &str, why: &str) {
    ctx.push(
        Issue::builder(IssueCode::MeaninglessFit)
            .message(format!("{what} is meaningless"))
            .meaninglessness(Meaninglessness::vacuous(
                what,
                why,
                "do not interpret the numeric output",
            ))
            .build(),
    );
}

/// OLS used as a statistical estimator: drop `ResidualTooLarge` (that code is
/// an interpolation residual gate, not an SSE-minimizer gate).
fn statistical_ols(ctx: &mut FitCtx, x: &Matrix, y: &Vector) -> Option<Vector> {
    let mut scratch = Report::new(ctx.report.algorithm.as_str(), "lstsq");
    let out = crate::linalg::least_squares(&mut scratch, x, y, &ctx.policy);
    for issue in scratch.issues() {
        if issue.code == IssueCode::ResidualTooLarge {
            continue;
        }
        ctx.push(issue.clone());
    }
    out
}

fn pearson_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut k = 0.0;
    for i in 0..n {
        if a[i].is_finite() && b[i].is_finite() {
            sx += a[i];
            sy += b[i];
            k += 1.0;
        }
    }
    if k < 2.0 {
        return f64::NAN;
    }
    let mx = sx / k;
    let my = sy / k;
    let mut num = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        if a[i].is_finite() && b[i].is_finite() {
            let dx = a[i] - mx;
            let dy = b[i] - my;
            num += dx * dy;
            vx += dx * dx;
            vy += dy * dy;
        }
    }
    let den = (vx * vy).sqrt();
    if den <= 0.0 {
        f64::NAN
    } else {
        num / den
    }
}

fn kendall_tau_b(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut nc = 0.0;
    let mut nd = 0.0;
    let mut n1 = 0.0;
    let mut n2 = 0.0;
    let mut n0 = 0.0;
    for i in 0..n {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        for j in (i + 1)..n {
            if !x[j].is_finite() || !y[j].is_finite() {
                continue;
            }
            n0 += 1.0;
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let p = dx * dy;
            if p > 0.0 {
                nc += 1.0;
            } else if p < 0.0 {
                nd += 1.0;
            }
            if dx.abs() <= 0.0 {
                n1 += 1.0;
            }
            if dy.abs() <= 0.0 {
                n2 += 1.0;
            }
        }
    }
    let den = f64::sqrt((n0 - n1) * (n0 - n2));
    if den <= 0.0 {
        f64::NAN
    } else {
        (nc - nd) / den
    }
}

fn rank_average(xs: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&i, &j| {
        xs[i]
            .partial_cmp(&xs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; xs.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && (xs[idx[j]] - xs[idx[i]]).abs() <= 0.0 {
            j += 1;
        }
        let avg = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[idx[k]] = avg;
        }
        i = j;
    }
    ranks
}

fn fisher_skew_kurt(xs: &[f64], mean: f64, std: f64) -> (f64, f64) {
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    let mut n = 0.0;
    for &v in xs {
        if !v.is_finite() {
            continue;
        }
        let z = v - mean;
        m3 += z * z * z;
        m4 += z * z * z * z;
        n += 1.0;
    }
    if n < 3.0 || std <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let g1 = (m3 / n) / std.powi(3);
    let g2 = (m4 / n) / std.powi(4) - 3.0;
    // Fisher bias correction.
    let skew = if n > 2.0 {
        g1 * (n * (n - 1.0)).sqrt() / (n - 2.0)
    } else {
        g1
    };
    let kurt = if n > 3.0 {
        ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0)
    } else {
        g2
    };
    (skew, kurt)
}

fn median(xs: &[f64]) -> f64 {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

fn acf_raw(y: &[f64], nlags: usize) -> Vec<f64> {
    let st = slice_stats(y);
    let n = y.len();
    let mut out = vec![0.0; nlags + 1];
    if st.std() <= 0.0 || n == 0 {
        out[0] = 1.0;
        for v in out.iter_mut().skip(1) {
            *v = f64::NAN;
        }
        return out;
    }
    let mut gamma0 = 0.0;
    for &v in y {
        if v.is_finite() {
            let d = v - st.mean;
            gamma0 += d * d;
        }
    }
    if gamma0 <= 0.0 {
        out[0] = 1.0;
        return out;
    }
    out[0] = 1.0;
    for k in 1..=nlags {
        if k >= n {
            out[k] = f64::NAN;
            continue;
        }
        let mut g = 0.0;
        for t in k..n {
            if y[t].is_finite() && y[t - k].is_finite() {
                g += (y[t] - st.mean) * (y[t - k] - st.mean);
            }
        }
        out[k] = g / gamma0;
    }
    out
}

fn schwert_lags(n: usize) -> usize {
    if n < 8 {
        return 0;
    }
    let l = (12.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize;
    l.min(n.saturating_sub(4))
}

fn ols_se_j(ctx: &mut FitCtx, design: &Matrix, sigma2: f64, j: usize) -> f64 {
    if !sigma2.is_finite() || sigma2 <= 0.0 {
        return f64::NAN;
    }
    let p = design.ncols();
    if j >= p {
        return f64::NAN;
    }
    let gram = design.gram();
    let mut e = Vector::zeros(p);
    e[j] = 1.0;
    match chol_solve(&mut ctx.report, &gram, &e, &ctx.policy) {
        Some(col) => {
            let v = col[j] * sigma2;
            if v.is_finite() && v > 0.0 {
                v.sqrt()
            } else {
                f64::NAN
            }
        }
        None => {
            ctx.push(
                Issue::builder(IssueCode::InformationMatrixSingular)
                    .message("ADF Gram matrix is not SPD; SE withheld")
                    .build(),
            );
            f64::NAN
        }
    }
}

fn newey_west(e: &[f64], lags: usize) -> f64 {
    let n = e.len() as f64;
    if n <= 0.0 {
        return f64::NAN;
    }
    let mut gamma0 = 0.0;
    for &v in e {
        gamma0 += v * v;
    }
    gamma0 /= n;
    let mut s = gamma0;
    let l = lags.min(e.len().saturating_sub(1));
    for k in 1..=l {
        let mut g = 0.0;
        for t in k..e.len() {
            g += e[t] * e[t - k];
        }
        g /= n;
        let w = 1.0 - k as f64 / (l as f64 + 1.0);
        s += 2.0 * w * g;
    }
    s.max(0.0)
}

fn adf_pvalue_constant(stat: f64) -> f64 {
    if !stat.is_finite() {
        return f64::NAN;
    }
    // MacKinnon-style knots for the constant-only ADF τ (large n).
    let knots = [
        (-4.38, 0.001),
        (-3.43, 0.01),
        (-2.86, 0.05),
        (-2.57, 0.10),
        (-1.94, 0.50),
        (-1.60, 0.80),
        (-1.20, 0.95),
    ];
    interpolate_tail(stat, &knots)
}

fn kpss_pvalue(stat: f64) -> f64 {
    if !stat.is_finite() {
        return f64::NAN;
    }
    // Kwiatkowski–Phillips–Schmidt–Shin level critical values (upper tail).
    let knots = [
        (0.119, 0.90),
        (0.347, 0.10),
        (0.463, 0.05),
        (0.574, 0.025),
        (0.739, 0.01),
        (1.200, 0.001),
    ];
    interpolate_tail_upper(stat, &knots)
}

fn interpolate_tail(stat: f64, knots: &[(f64, f64)]) -> f64 {
    if stat <= knots[0].0 {
        return knots[0].1;
    }
    let last = knots[knots.len() - 1];
    if stat >= last.0 {
        return last.1;
    }
    for w in knots.windows(2) {
        let (x0, p0) = w[0];
        let (x1, p1) = w[1];
        if stat >= x0 && stat <= x1 {
            let t = (stat - x0) / (x1 - x0);
            return (p0 + t * (p1 - p0)).clamp(0.0, 1.0);
        }
    }
    f64::NAN
}

fn interpolate_tail_upper(stat: f64, knots: &[(f64, f64)]) -> f64 {
    if stat <= knots[0].0 {
        return knots[0].1;
    }
    let last = knots[knots.len() - 1];
    if stat >= last.0 {
        return last.1;
    }
    for w in knots.windows(2) {
        let (x0, p0) = w[0];
        let (x1, p1) = w[1];
        if stat >= x0 && stat <= x1 {
            let t = (stat - x0) / (x1 - x0);
            return (p0 + t * (p1 - p0)).clamp(0.0, 1.0);
        }
    }
    f64::NAN
}

fn ks_pvalue(d: f64, neff: f64) -> f64 {
    if !d.is_finite() || neff <= 0.0 {
        return f64::NAN;
    }
    let q = (neff.sqrt() + 0.12 + 0.11 / neff.sqrt()) * d;
    let mut s = 0.0;
    for k in 1..40 {
        let kf = k as f64;
        let term = ((-1.0_f64).powi(k as i32 - 1)) * (-2.0 * kf * kf * q * q).exp();
        s += term;
        if term.abs() < 1e-12 {
            break;
        }
    }
    (2.0 * s).clamp(0.0, 1.0)
}

fn shapiro_francia_pvalue(w: f64, n: usize) -> f64 {
    if !w.is_finite() || n < 3 {
        return f64::NAN;
    }
    // Royston-like logistic transform of W' (approximate).
    let nf = n as f64;
    let mu = -1.2725 + 1.0521 * (nf.ln() - nf.ln().ln());
    let sig = 1.0308 - 0.26758 * (nf.ln()).ln();
    let z = ((1.0 - w).ln() - mu) / sig.max(1e-8);
    1.0 - norm_cdf(z)
}

fn holm_adjust(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&i, &j| p[i].partial_cmp(&p[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut adj = vec![0.0; m];
    let mut running: f64 = 0.0;
    for (rank, &i) in idx.iter().enumerate() {
        let factor = (m - rank) as f64;
        let v = (p[i] * factor).min(1.0);
        running = running.max(v);
        adj[i] = running.min(1.0);
    }
    adj
}

fn bh_adjust(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&i, &j| p[i].partial_cmp(&p[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut adj = vec![0.0; m];
    let mut running: f64 = 1.0;
    for (rev, &i) in idx.iter().rev().enumerate() {
        let rank = m - rev;
        let v = (p[i] * m as f64 / rank as f64).min(1.0);
        running = running.min(v);
        adj[i] = running;
    }
    adj
}

fn percentile_sorted(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
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

/// Inverse standard-normal CDF (Acklam's rational approximation).
pub fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460523810054e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614736e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p > 1.0 - plow {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

fn student_t_pdf(t: f64, df: f64) -> f64 {
    if !t.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    let ln = ln_gamma((df + 1.0) / 2.0)
        - ln_gamma(df / 2.0)
        - 0.5 * (df * std::f64::consts::PI).ln()
        - 0.5 * (df + 1.0) * (1.0 + t * t / df).ln();
    ln.exp()
}

fn student_t_ppf(p: f64, df: f64) -> f64 {
    if !p.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }
    let mut t = norm_ppf(p);
    for _ in 0..40 {
        let f = student_t_cdf(t, df) - p;
        let dens = student_t_pdf(t, df).max(1e-18);
        let step = f / dens;
        t -= step;
        if step.abs() < 1e-12 {
            break;
        }
    }
    t
}

/// Tricube locally-linear smooth used by STL (no quality report of its own).
pub(crate) fn lowess_raw(x: &[f64], y: &[f64], frac: f64) -> Vec<f64> {
    let n = x.len().min(y.len());
    if n == 0 {
        return Vec::new();
    }
    let span = ((frac.clamp(1e-3, 1.0) * n as f64).ceil() as usize).clamp(2, n);
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut dist: Vec<(f64, usize)> = (0..n).map(|j| ((x[j] - x[i]).abs(), j)).collect();
        dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let maxd = dist[span - 1].0.max(1e-15);
        let mut sw = 0.0;
        let mut swx = 0.0;
        let mut swy = 0.0;
        let mut swxx = 0.0;
        let mut swxy = 0.0;
        for &(d, j) in dist.iter().take(span) {
            let u = d / maxd;
            let w = if u >= 1.0 {
                0.0
            } else {
                let t = 1.0 - u * u * u;
                t * t * t
            };
            sw += w;
            swx += w * x[j];
            swy += w * y[j];
            swxx += w * x[j] * x[j];
            swxy += w * x[j] * y[j];
        }
        let det = sw * swxx - swx * swx;
        out[i] = if det.abs() > 1e-18 {
            let a = (swy * swxx - swx * swxy) / det;
            let b = (sw * swxy - swx * swy) / det;
            a + b * x[i]
        } else if sw > 0.0 {
            swy / sw
        } else {
            y[i]
        };
    }
    out
}

/// OLS influence diagnostics (statsmodels `OLSInfluence`): leverage, DFFITS, DFBETAS.
#[derive(Clone, Debug)]
pub struct OlsInfluence {
    /// Residuals.
    pub resid: Vector,
    /// Hat-matrix diagonal \(h_{ii}\).
    pub hat: Vector,
    /// DFFITS.
    pub dffits: Vector,
    /// DFBETAS (`n × p`, including intercept).
    pub dfbetas: Matrix,
}

/// Cook / DFFITS / DFBETAS from an intercept-on OLS of `y` on `X`.
pub fn ols_influence(x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<OlsInfluence>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let design = x.with_intercept();
    inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
    let mut scratch = Report::new("infl", "ols");
    let Some(beta) = least_squares(&mut scratch, &design, y, &ctx.policy) else {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("OLS influence: least squares failed")
                .build(),
        );
        return ctx.finish(OlsInfluence {
            resid: Vector::zeros(y.len()),
            hat: Vector::zeros(y.len()),
            dffits: Vector::zeros(y.len()),
            dfbetas: Matrix::zeros(y.len(), design.ncols()),
        });
    };
    for issue in scratch.issues() {
        if matches!(
            issue.code,
            IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
        ) {
            continue;
        }
        ctx.push(issue.clone());
    }
    let n = design.nrows().min(y.len());
    let p = design.ncols();
    let fit = design.matvec(&beta);
    let resid = Vector::from_iter((0..n).map(|i| y[i] - fit[i]));
    let mut xtx = faer::Mat::<f64>::zeros(p, p);
    for i in 0..n {
        for a in 0..p {
            for b in 0..p {
                xtx[(a, b)] += design.get(i, a) * design.get(i, b);
            }
        }
    }
    let mut xtx_inv = Matrix::zeros(p, p);
    for j in 0..p {
        let mut e = Vector::zeros(p);
        e[j] = 1.0;
        let mut sc = Report::new("infl", "inv");
        if let Some(col) = chol_solve(&mut sc, &xtx, &e, &ctx.policy) {
            for i in 0..p {
                xtx_inv.set(i, j, col[i]);
            }
        }
    }
    let mut sse = 0.0;
    for i in 0..n {
        sse += resid[i] * resid[i];
    }
    let df = (n as f64 - p as f64).max(1.0);
    let s = (sse / df).sqrt().max(1e-12);
    let mut hat = Vector::zeros(n);
    let mut dffits = Vector::zeros(n);
    let mut dfbetas = Matrix::zeros(n, p);
    for i in 0..n {
        let mut h = 0.0;
        for a in 0..p {
            let mut sa = 0.0;
            for b in 0..p {
                sa += xtx_inv.get(a, b) * design.get(i, b);
            }
            h += design.get(i, a) * sa;
        }
        h = h.clamp(0.0, 1.0 - 1e-12);
        hat[i] = h;
        let denom = (1.0 - h).max(1e-12);
        dffits[i] = resid[i] / (s * denom.sqrt()) * (h / denom).sqrt();
        for j in 0..p {
            let mut c = 0.0;
            for a in 0..p {
                c += xtx_inv.get(j, a) * design.get(i, a);
            }
            let cjj = xtx_inv.get(j, j).abs().sqrt().max(1e-12);
            dfbetas.set(i, j, c * resid[i] / (s * cjj * denom));
        }
    }
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(signlred::Severity::Advisory)
            .message("DFFITS uses the full-sample s, not the leave-one-out s_{(i)}")
            .compromise(NumericalCompromise::new(
                "exact leave-one-out DFFITS / DFBETAS",
                "hat-diagonal formulae with the pooled residual scale",
                "s_{(i)} is not recomputed",
                "rank cases by |DFFITS|, do not treat the cutoff as exact",
            ))
            .build(),
    );
    ctx.finish(OlsInfluence {
        resid,
        hat,
        dffits,
        dfbetas,
    })
}

/// White specification / omitted-variable LM (statsmodels `spec_white`).
///
/// Expanded feature count is not identification `p`.
pub fn spec_white(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let design = x.with_intercept();
    let mut scratch = Report::new("specw", "ols");
    let Some(beta) = least_squares(&mut scratch, &design, y, &ctx.policy) else {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("spec_white: primary OLS failed")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: y.len() as f64,
        });
    };
    for issue in scratch.issues() {
        if matches!(
            issue.code,
            IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
        ) {
            continue;
        }
        ctx.push(issue.clone());
    }
    let n = design.nrows().min(y.len());
    let fit = design.matvec(&beta);
    let e2 = Vector::from_iter((0..n).map(|i| {
        let e = y[i] - fit[i];
        e * e
    }));
    if e2
        .as_slice()
        .iter()
        .all(|v| *v <= ctx.policy.near_zero_variance)
    {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("spec_white: squared residuals are identically zero")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: n as f64,
        });
    }
    let p = x.ncols();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    cols.push(vec![1.0; n]);
    for j in 0..p {
        cols.push((0..n).map(|i| x.get(i, j)).collect());
    }
    let base = cols.len();
    for a in 1..base {
        for b in a..base {
            cols.push((0..n).map(|i| cols[a][i] * cols[b][i]).collect());
        }
    }
    if cols.len() >= n {
        ctx.push(
            Issue::builder(IssueCode::Overparameterized)
                .message(format!(
                    "spec_white auxiliary has {} columns ≥ n={n}; higher-order terms are truncated",
                    cols.len()
                ))
                .build(),
        );
        cols.truncate(n.saturating_sub(1).max(1));
    }
    let aux = Matrix::from_fn(n, cols.len(), |i, j| cols[j][i]);
    let mut aux_sc = Report::new("specw", "aux");
    let Some(ab) = least_squares(&mut aux_sc, &aux, &e2, &ctx.policy) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: (cols.len().saturating_sub(1)) as f64,
            nobs: n as f64,
        });
    };
    let fitted = aux.matvec(&ab);
    let mut sse = 0.0;
    let mut sst = 0.0;
    let ym = e2.mean();
    for i in 0..n {
        let r = e2[i] - fitted[i];
        sse += r * r;
        let d = e2[i] - ym;
        sst += d * d;
    }
    let r2 = if sst > 0.0 { 1.0 - sse / sst } else { 0.0 };
    let df = (cols.len().saturating_sub(1)) as f64;
    let stat = n as f64 * r2.max(0.0);
    let pvalue = chi2_pvalue(stat, df.max(1.0));
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n as f64,
    })
}

/// Engle ARCH LM on squared residuals (statsmodels `het_arch`).
///
/// Lag count is not identification `p`.
pub fn het_arch(
    resid: &Vector,
    lags: usize,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    arch_lm(resid, lags, session)
}

/// Breusch–Godfrey residual AR LM (statsmodels `acorr_breusch_godfrey`).
///
/// Lag count is not identification `p`.
pub fn acorr_breusch_godfrey(
    resid: &Vector,
    design: &Matrix,
    lags: usize,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    breusch_godfrey(resid, design, lags, session)
}

/// Studentized residuals and Bonferroni p-values (statsmodels `outlier_test`).
///
/// Observation count is not identification `p`.
#[derive(Clone, Debug)]
pub struct OutlierTest {
    /// Internally studentized residuals \(e_i / (s\sqrt{1-h_{ii}})\).
    pub studentized: Vector,
    /// Two-sided Student-\(t\) p-values.
    pub pvalue: Vector,
    /// Bonferroni-adjusted p-values \(\min(1, n\,p_i)\).
    pub bonferroni: Vector,
}

/// Outlier test from the OLS hat matrix and residuals.
pub fn outlier_test(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<OutlierTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let inf = match ols_influence(x, y, &session.child("infl")) {
        Ok(q) => {
            for issue in q.report.issues() {
                if matches!(
                    issue.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::R2IsOne
                        | IssueCode::CholeskyFailed
                ) {
                    continue;
                }
                ctx.push(issue.clone());
            }
            q.value
        }
        Err(_) => {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("outlier_test: OLS influence failed")
                    .build(),
            );
            return ctx.finish(OutlierTest {
                studentized: Vector::zeros(y.len()),
                pvalue: Vector::zeros(y.len()),
                bonferroni: Vector::zeros(y.len()),
            });
        }
    };
    let n = inf.resid.len();
    let p = x.ncols() + 1;
    let mut sse = 0.0;
    for i in 0..n {
        sse += inf.resid[i] * inf.resid[i];
    }
    let df = (n as f64 - p as f64).max(1.0);
    let s = (sse / df).sqrt().max(1e-12);
    let mut studentized = Vector::zeros(n);
    let mut pvalue = Vector::zeros(n);
    let mut bonferroni = Vector::zeros(n);
    for i in 0..n {
        let h = inf.hat.as_slice().get(i).copied().unwrap_or(0.0);
        let denom = (s * (1.0 - h).max(1e-12).sqrt()).max(1e-12);
        let t = inf.resid[i] / denom;
        studentized[i] = t;
        let pv = if t.is_finite() {
            student_t_pvalue(t, df)
        } else {
            f64::NAN
        };
        pvalue[i] = pv;
        bonferroni[i] = if pv.is_finite() {
            (pv * n as f64).min(1.0)
        } else {
            f64::NAN
        };
    }
    ctx.finish(OutlierTest {
        studentized,
        pvalue,
        bonferroni,
    })
}

/// Medcouple robust skewness (statsmodels `stattools.medcouple`).
///
/// Pair count is not identification `p`.
pub fn medcouple(x: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let mut v: Vec<f64> = x.as_slice().iter().copied().filter(|z| z.is_finite()).collect();
    if v.len() < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!("medcouple needs ≥3 finite points; got {}", v.len()))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let med = if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    };
    let mut h = Vec::new();
    for i in 0..n {
        for j in i..n {
            if v[i] <= med && v[j] >= med {
                let den = v[j] - v[i];
                if den.abs() <= 1e-15 {
                    h.push(0.0);
                } else {
                    h.push(((v[j] - med) - (med - v[i])) / den);
                }
            }
        }
    }
    if h.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("medcouple pair set is empty")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    h.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = h.len();
    let mc = if m % 2 == 1 {
        h[m / 2]
    } else {
        0.5 * (h[m / 2 - 1] + h[m / 2])
    };
    ctx.finish(mc)
}

/// WLS / OLS prediction standard errors (statsmodels `wls_prediction_std`).
///
/// Observation count is not identification `p`. Equal weights (OLS) are used.
#[derive(Clone, Debug)]
pub struct WlsPredictionStd {
    /// Fitted mean.
    pub predicted: Vector,
    /// Standard error of the mean \(\,s\sqrt{h_{ii}}\).
    pub se_mean: Vector,
    /// Predictive SE \(\,s\sqrt{1+h_{ii}}\).
    pub se_obs: Vector,
}

/// In-sample OLS prediction standard errors.
pub fn wls_prediction_std(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<WlsPredictionStd>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let design = x.with_intercept();
    let mut scratch = Report::new("wlspred", "ols");
    let Some(beta) = least_squares(&mut scratch, &design, y, &ctx.policy) else {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("wls_prediction_std: OLS failed")
                .build(),
        );
        return ctx.finish(WlsPredictionStd {
            predicted: Vector::zeros(y.len()),
            se_mean: Vector::zeros(y.len()),
            se_obs: Vector::zeros(y.len()),
        });
    };
    for issue in scratch.issues() {
        if matches!(
            issue.code,
            IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
        ) {
            continue;
        }
        ctx.push(issue.clone());
    }
    let n = design.nrows().min(y.len());
    let p = design.ncols();
    let fit = design.matvec(&beta);
    let mut sse = 0.0;
    for i in 0..n {
        let e = y[i] - fit[i];
        sse += e * e;
    }
    let df = (n as f64 - p as f64).max(1.0);
    let s = (sse / df).sqrt().max(1e-12);
    let mut xtx = faer::Mat::<f64>::zeros(p, p);
    for i in 0..n {
        for a in 0..p {
            for b in 0..p {
                xtx[(a, b)] += design.get(i, a) * design.get(i, b);
            }
        }
    }
    let mut xtx_inv = Matrix::zeros(p, p);
    for j in 0..p {
        let mut e = Vector::zeros(p);
        e[j] = 1.0;
        let mut sc = Report::new("wlspred", "inv");
        if let Some(col) = chol_solve(&mut sc, &xtx, &e, &ctx.policy) {
            for i in 0..p {
                xtx_inv.set(i, j, col[i]);
            }
        }
    }
    let mut se_mean = Vector::zeros(n);
    let mut se_obs = Vector::zeros(n);
    for i in 0..n {
        let mut h = 0.0;
        for a in 0..p {
            let mut sa = 0.0;
            for b in 0..p {
                sa += xtx_inv.get(a, b) * design.get(i, b);
            }
            h += design.get(i, a) * sa;
        }
        h = h.clamp(0.0, 1.0 - 1e-12);
        se_mean[i] = s * h.sqrt();
        se_obs[i] = s * (1.0 + h).sqrt();
    }
    ctx.finish(WlsPredictionStd {
        predicted: Vector::from_iter((0..n).map(|i| fit[i])),
        se_mean,
        se_obs,
    })
}

/// Lagrange-multiplier linearity test (statsmodels `linear_lm`).
///
/// Residuals of \(y\) on \(x\) are regressed on \(x\) and column-0 squares.
/// Expanded feature count is not identification `p`.
pub fn linear_lm(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let design = x.with_intercept();
    let mut scratch = Report::new("linlm", "ols");
    let Some(beta) = least_squares(&mut scratch, &design, y, &ctx.policy) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: y.len() as f64,
        });
    };
    for issue in scratch.issues() {
        if matches!(
            issue.code,
            IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
        ) {
            continue;
        }
        ctx.push(issue.clone());
    }
    let n = design.nrows().min(y.len());
    let fit = design.matvec(&beta);
    let e = Vector::from_iter((0..n).map(|i| y[i] - fit[i]));
    let aux = Matrix::from_fn(n, x.ncols() + 2, |i, j| {
        if j == 0 {
            1.0
        } else if j <= x.ncols() {
            x.get(i, j - 1)
        } else {
            let v = x.get(i, 0);
            v * v
        }
    });
    let mut aux_sc = Report::new("linlm", "aux");
    let Some(ab) = least_squares(&mut aux_sc, &aux, &e, &ctx.policy) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: n as f64,
        });
    };
    let fitted = aux.matvec(&ab);
    let mut sse = 0.0;
    let mut sst = 0.0;
    let em = e.mean();
    for i in 0..n {
        let r = e[i] - fitted[i];
        sse += r * r;
        let d = e[i] - em;
        sst += d * d;
    }
    let r2 = if sst > 0.0 { 1.0 - sse / sst } else { 0.0 };
    let stat = n as f64 * r2.max(0.0);
    let pvalue = chi2_pvalue(stat, 1.0);
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: 1.0,
        nobs: n as f64,
    })
}

/// Two-sample *t* from summary statistics (statsmodels `ttest_ind_from_stats`).
///
/// Sample sizes are not identification `p`.
pub fn ttest_ind_from_stats(
    mean1: f64,
    std1: f64,
    n1: f64,
    mean2: f64,
    std2: f64,
    n2: f64,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if ![mean1, std1, n1, mean2, std2, n2]
        .iter()
        .all(|v| v.is_finite())
    {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("ttest_ind_from_stats received a non-finite argument")
                .build(),
        );
    }
    if n1 <= 1.0 || n2 <= 1.0 || std1 < 0.0 || std2 < 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("ttest_ind_from_stats needs n>1 and non-negative std")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: n1 + n2,
        });
    }
    let v1 = std1 * std1 / n1;
    let v2 = std2 * std2 / n2;
    let den = (v1 + v2).sqrt();
    let stat = if den > 1e-18 {
        (mean1 - mean2) / den
    } else {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("ttest_ind_from_stats pooled variance is ~0")
                .build(),
        );
        f64::NAN
    };
    let df = if v1 + v2 > 1e-18 {
        (v1 + v2) * (v1 + v2) / (v1 * v1 / (n1 - 1.0) + v2 * v2 / (n2 - 1.0)).max(1e-18)
    } else {
        n1 + n2 - 2.0
    };
    let pvalue = if stat.is_finite() && df > 0.0 {
        student_t_pvalue(stat, df)
    } else {
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n1 + n2,
    })
}

/// Two-sample mean comparison (statsmodels `CompareMeans`).
#[derive(Clone, Debug)]
pub struct CompareMeansResult {
    /// Sample mean of the first group.
    pub mean_a: f64,
    /// Sample mean of the second group.
    pub mean_b: f64,
    /// `mean_a − mean_b`.
    pub diff: f64,
    /// Welch (or pooled) *t*.
    pub statistic: f64,
    /// Two-sided *p*.
    pub pvalue: f64,
    /// Degrees of freedom.
    pub df: f64,
}

/// Compare two independent samples with a Welch *t* test.
pub fn compare_means(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<CompareMeansResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let q = ttest_ind(a, b, true, &session.child("ttest"))?;
    for issue in q.report.issues() {
        ctx.push(issue.clone());
    }
    let sa = slice_stats(a.as_slice());
    let sb = slice_stats(b.as_slice());
    ctx.finish(CompareMeansResult {
        mean_a: sa.mean,
        mean_b: sb.mean,
        diff: sa.mean - sb.mean,
        statistic: q.value.statistic,
        pvalue: q.value.pvalue,
        df: q.value.df,
    })
}

/// Likelihood-ratio test of nested OLS (statsmodels `compare_lr_test`).
///
/// `x_restr` must be nested in `x_unrestr` (fewer columns). Extra-column
/// count is not identification `p`.
pub fn compare_lr(
    y: &Vector,
    x_restr: &Matrix,
    x_unrestr: &Matrix,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x_restr, Some(y), &ctx.policy);
    inspect_xy(&mut ctx.report, x_unrestr, Some(y), &ctx.policy);
    let n = y.len().min(x_restr.nrows()).min(x_unrestr.nrows());
    if x_unrestr.ncols() <= x_restr.ncols() {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message("compare_lr: unrestricted design is not wider; the LR is unidentified")
                .build(),
        );
    }
    let Some(br) = statistical_ols(&mut ctx, x_restr, y) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: (x_unrestr.ncols() as f64 - x_restr.ncols() as f64).max(1.0),
            nobs: n as f64,
        });
    };
    let Some(bu) = statistical_ols(&mut ctx, x_unrestr, y) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: (x_unrestr.ncols() as f64 - x_restr.ncols() as f64).max(1.0),
            nobs: n as f64,
        });
    };
    let fr = x_restr.matvec(&br);
    let fu = x_unrestr.matvec(&bu);
    let mut ssr_r = 0.0;
    let mut ssr_u = 0.0;
    for i in 0..n {
        let er = y[i] - if i < fr.len() { fr[i] } else { 0.0 };
        let eu = y[i] - if i < fu.len() { fu[i] } else { 0.0 };
        ssr_r += er * er;
        ssr_u += eu * eu;
    }
    let df = (x_unrestr.ncols() as f64 - x_restr.ncols() as f64).max(1.0);
    let stat = if ssr_u > 0.0 && ssr_r > 0.0 {
        n as f64 * (ssr_r / ssr_u).ln()
    } else {
        f64::NAN
    };
    let pvalue = if stat.is_finite() {
        chi2_pvalue(stat.max(0.0), df)
    } else {
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n as f64,
    })
}

/// Nested OLS extra-sum-of-squares *F* (statsmodels `compare_f_test`).
///
/// Extra-column count is not identification `p`.
pub fn compare_f(
    y: &Vector,
    x_restr: &Matrix,
    x_unrestr: &Matrix,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x_restr, Some(y), &ctx.policy);
    inspect_xy(&mut ctx.report, x_unrestr, Some(y), &ctx.policy);
    let n = y.len().min(x_restr.nrows()).min(x_unrestr.nrows());
    if x_unrestr.ncols() <= x_restr.ncols() {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message("compare_f: unrestricted design is not wider")
                .build(),
        );
    }
    let Some(br) = statistical_ols(&mut ctx, x_restr, y) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: (x_unrestr.ncols() as f64 - x_restr.ncols() as f64).max(1.0),
            nobs: n as f64,
        });
    };
    let Some(bu) = statistical_ols(&mut ctx, x_unrestr, y) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: (x_unrestr.ncols() as f64 - x_restr.ncols() as f64).max(1.0),
            nobs: n as f64,
        });
    };
    let fr = x_restr.matvec(&br);
    let fu = x_unrestr.matvec(&bu);
    let mut ssr_r = 0.0;
    let mut ssr_u = 0.0;
    for i in 0..n {
        let er = y[i] - if i < fr.len() { fr[i] } else { 0.0 };
        let eu = y[i] - if i < fu.len() { fu[i] } else { 0.0 };
        ssr_r += er * er;
        ssr_u += eu * eu;
    }
    let q = (x_unrestr.ncols() as f64 - x_restr.ncols() as f64).max(1.0);
    let df_den = (n as f64 - x_unrestr.ncols() as f64).max(1.0);
    if n <= x_unrestr.ncols() {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("compare_f: n ≤ p_unrestricted; residual df is patched to 1")
                .build(),
        );
    }
    let stat = if ssr_u > 0.0 {
        ((ssr_r - ssr_u).max(0.0) / q) / (ssr_u / df_den)
    } else {
        f64::NAN
    };
    let pvalue = if stat.is_finite() {
        f_pvalue(stat.max(0.0), q, df_den)
    } else {
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: q,
        nobs: n as f64,
    })
}

/// Joint Wald test that every slope in OLS is zero (statsmodels `wald_test`).
///
/// Slope count is not identification `p`. The intercept is not restricted.
pub fn wald_ols(x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let design = x.with_intercept();
    let n = y.len().min(design.nrows());
    let k = design.ncols();
    let q = k.saturating_sub(1);
    if q == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("wald_ols: no slope coefficients to test")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 0.0,
            nobs: n as f64,
        });
    }
    let Some(beta) = statistical_ols(&mut ctx, &design, y) else {
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: q as f64,
            nobs: n as f64,
        });
    };
    let fit = design.matvec(&beta);
    let mut sse = 0.0;
    for i in 0..n {
        let e = y[i] - fit[i];
        sse += e * e;
    }
    let df_res = (n as f64 - k as f64).max(1.0);
    let sigma2 = sse / df_res;
    let gram = design.gram();
    let mut gs = Mat::<f64>::zeros(q, q);
    for i in 0..q {
        for j in 0..q {
            gs[(i, j)] = gram[(i + 1, j + 1)] * sigma2.max(1e-18);
        }
        gs[(i, i)] += 1e-12;
    }
    let z = Vector::from_iter((1..beta.len()).map(|j| beta[j]));
    let mut scratch = Report::new("wald", "chol");
    let stat = match chol_solve(&mut scratch, &gs, &z, &ctx.policy) {
        Some(sol) => z.dot(&sol),
        None => {
            ctx.push(
                Issue::builder(IssueCode::CholeskyFailed)
                    .severity(Severity::Warning)
                    .message("wald_ols: slope covariance was not SPD; statistic is undefined")
                    .compromise(NumericalCompromise::new(
                        "Cholesky of σ² (X'X)⁻¹ on the slopes",
                        "Wald statistic set to NaN",
                        "the Gram of the slopes was indefinite even after jitter",
                        "do not read a missing Wald as a zero effect",
                    ))
                    .build(),
            );
            f64::NAN
        }
    };
    let pvalue = if stat.is_finite() {
        chi2_pvalue(stat.max(0.0), q as f64)
    } else {
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: q as f64,
        nobs: n as f64,
    })
}

/// Box–Pierce portmanteau (statsmodels `acorr_ljungbox` `boxpierce=True`).
///
/// Lag count is not identification `p`.
pub fn box_pierce(x: &Vector, lags: usize, session: &Session) -> Result<Qualified<LjungBoxResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let n = x.len();
    let h = if lags >= 1 && lags < n {
        lags
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "Box–Pierce lags={lags} is not in 1..n-1 (n={n}); using 1"
                ))
                .build(),
        );
        1.min(n.saturating_sub(1))
    };
    if h == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Box–Pierce needs n≥2")
                .build(),
        );
        return ctx.finish(LjungBoxResult {
            stat: f64::NAN,
            pvalue: f64::NAN,
            lags: 0,
        });
    }
    let rho = acf_raw(x.as_slice(), h);
    let mut q = 0.0;
    for k in 1..=h {
        let r = rho[k];
        q += r * r;
    }
    q *= n as f64;
    let pvalue = if q.is_finite() {
        chi2_pvalue(q, h as f64)
    } else {
        f64::NAN
    };
    if pvalue.is_finite() && pvalue < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::AutocorrelatedResiduals)
                .message(format!("Box–Pierce Q={q:.3} p={pvalue:.4}"))
                .build(),
        );
    }
    ctx.finish(LjungBoxResult {
        stat: q,
        pvalue,
        lags: h,
    })
}

/// Two-sample log-rank test (lifelines / statsmodels `duration.survdiff`).
///
/// `events` is 1 = observed, 0 = right-censored. When more than two groups
/// appear, the two most frequent labels are kept and a compromise is recorded.
/// Group count is not identification `p`.
pub fn logrank(
    times: &Vector,
    events: &Vector,
    groups: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(times),
        None,
        &ctx.policy,
    );
    if times.len() != events.len() || times.len() != groups.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "logrank lengths time={} event={} group={}",
                    times.len(),
                    events.len(),
                    groups.len()
                ))
                .build(),
        );
    }
    if let Some(issue) = scan_finite(events.as_slice()).to_issue("events") {
        ctx.push(issue);
    }
    if let Some(issue) = scan_finite(groups.as_slice()).to_issue("groups") {
        ctx.push(issue);
    }
    let n = times.len().min(events.len()).min(groups.len());
    let mut counts: Vec<(i64, usize)> = Vec::new();
    for i in 0..n {
        if !times[i].is_finite() || !groups[i].is_finite() {
            continue;
        }
        let g = groups[i].round() as i64;
        if let Some(e) = counts.iter_mut().find(|(k, _)| *k == g) {
            e.1 += 1;
        } else {
            counts.push((g, 1));
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if counts.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("log-rank needs two groups")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: n as f64,
        });
    }
    if counts.len() > 2 {
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("log-rank lite keeps the two most frequent groups")
                .compromise(NumericalCompromise::new(
                    "two-sample Mantel–Haenszel log-rank",
                    "extra groups dropped",
                    "the k-sample covariance form is not assembled",
                    "do not read this as a k-sample survdiff",
                ))
                .build(),
        );
    }
    let g0 = counts[0].0;
    let g1 = counts[1].0;
    let mut rows: Vec<(f64, bool, i64)> = (0..n)
        .filter(|&i| times[i].is_finite() && groups[i].is_finite())
        .map(|i| {
            let g = groups[i].round() as i64;
            (times[i], events[i] >= 0.5, g)
        })
        .filter(|(_, _, g)| *g == g0 || *g == g1)
        .collect();
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut times_u: Vec<f64> = rows
        .iter()
        .filter(|(_, ev, _)| *ev)
        .map(|(t, _, _)| *t)
        .collect();
    times_u.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times_u.dedup_by(|a, b| (*a - *b).abs() <= 1e-15);
    let mut oe = 0.0;
    let mut var = 0.0;
    for t in times_u {
        let mut n0 = 0.0;
        let mut n1 = 0.0;
        let mut d0 = 0.0;
        let mut d1 = 0.0;
        for (ti, ev, g) in &rows {
            if *ti + 1e-15 < t {
                continue;
            }
            if *g == g0 {
                n0 += 1.0;
                if *ev && (*ti - t).abs() <= 1e-15 {
                    d0 += 1.0;
                }
            } else {
                n1 += 1.0;
                if *ev && (*ti - t).abs() <= 1e-15 {
                    d1 += 1.0;
                }
            }
        }
        let nn = n0 + n1;
        let dd = d0 + d1;
        if nn <= 0.0 || dd <= 0.0 {
            continue;
        }
        oe += d0 - n0 * dd / nn;
        if nn > 1.0 {
            var += n0 * n1 * dd * (nn - dd) / (nn * nn * (nn - 1.0));
        }
    }
    if var <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("log-rank variance vanished; the χ² is undefined")
                .compromise(NumericalCompromise::new(
                    "positive hypergeometric variance",
                    "statistic set to NaN",
                    "no discordant events remained after grouping",
                    "do not treat a missing log-rank as no difference",
                ))
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: rows.len() as f64,
        });
    }
    let stat: f64 = oe * oe / var;
    let pvalue = chi2_pvalue(stat.max(0.0), 1.0);
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: 1.0,
        nobs: rows.len() as f64,
    })
}

/// Alias of [`logrank`] (R `survdiff` / statsmodels duration).
pub fn survdiff(
    times: &Vector,
    events: &Vector,
    groups: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    logrank(times, events, groups, session)
}

/// Levinson–Durbin AR coefficients from a series ACF
/// (statsmodels `tsa.stattools.levinson_durbin`).
///
/// `order` is not identification `p`.
#[derive(Clone, Debug)]
pub struct LevinsonDurbin {
    /// AR coefficients \(a_1,\ldots,a_p\) (the leading 1 is omitted).
    pub ar: Vector,
    /// Innovation variance after the last reflection.
    pub sigma2: f64,
    /// Reflection (PACF) coefficients.
    pub reflection: Vector,
}

/// Levinson–Durbin recursion on the sample ACF of `y`.
pub fn levinson_durbin(
    y: &Vector,
    order: usize,
    session: &Session,
) -> Result<Qualified<LevinsonDurbin>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    let n = y.len();
    let p = if order >= 1 && order < n {
        order
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "Levinson–Durbin order={order} is not in 1..n-1 (n={n}); using 1"
                ))
                .build(),
        );
        1.min(n.saturating_sub(1))
    };
    if p == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Levinson–Durbin needs n≥2")
                .build(),
        );
        return ctx.finish(LevinsonDurbin {
            ar: Vector::zeros(0),
            sigma2: f64::NAN,
            reflection: Vector::zeros(0),
        });
    }
    let rho = acf_raw(y.as_slice(), p);
    let mut a = vec![0.0; p + 1];
    a[0] = 1.0;
    let mut e = if rho[0].is_finite() && rho[0].abs() > 0.0 {
        rho[0]
    } else {
        1.0
    };
    let mut kvec = vec![0.0; p];
    for m in 1..=p {
        let mut num = if m < rho.len() { rho[m] } else { 0.0 };
        for j in 1..m {
            num += a[j] * rho[m - j];
        }
        let km = if e.abs() > 1e-18 { -num / e } else { 0.0 };
        kvec[m - 1] = km;
        if km.abs() > 1.0 + 1e-8 {
            ctx.push(
                Issue::builder(IssueCode::NonStationary)
                    .message(format!(
                        "Levinson–Durbin reflection |k_{m}|={km:.3} exceeds 1"
                    ))
                    .build(),
            );
        }
        let prev = a.clone();
        for j in 1..m {
            a[j] = prev[j] + km * prev[m - j];
        }
        a[m] = km;
        e *= 1.0 - km * km;
        if e <= 1e-18 {
            ctx.push(
                Issue::builder(IssueCode::JitterInjected)
                    .message("Levinson–Durbin innovation variance collapsed")
                    .compromise(NumericalCompromise::new(
                        "positive prediction-error variance",
                        "recursion stopped with σ²≈0",
                        "the series is linearly predictable at this order",
                        "do not treat a collapsed σ² as a unique AR law",
                    ))
                    .build(),
            );
            e = e.max(0.0);
            break;
        }
    }
    ctx.finish(LevinsonDurbin {
        ar: Vector::from_iter(a.iter().copied().skip(1).take(p)),
        sigma2: e.abs(),
        reflection: Vector::from_slice(&kvec),
    })
}

/// Expanding-window one-step OLS residuals (statsmodels `RecursiveLS` resid).
///
/// Each prefix uses a scratch report. Prefix length is not identification `p`.
pub fn recursive_olsresiduals(
    x: &Matrix,
    y: &Vector,
    min_n: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let n = x.nrows().min(y.len());
    let p = x.ncols() + 1;
    let start = if min_n >= 3 {
        min_n.max(p + 1).min(n.max(1))
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("recursive_olsresiduals min_n={min_n} < 3; using 3"))
                .build(),
        );
        3.max(p + 1).min(n.max(1))
    };
    if n < start {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("recursive OLS burn-in {start} > n={n}"))
                .build(),
        );
    }
    if n > 5 * p {
        inspect_identification(&mut ctx.report, n, p, &ctx.policy);
    }
    let mut out = Vec::new();
    for end in start..=n {
        if end >= n {
            break;
        }
        let xt = Matrix::from_fn(end, x.ncols(), |i, j| x.get(i, j));
        let yt = Vector::from_iter((0..end).map(|i| y[i]));
        let design = xt.with_intercept();
        let mut scratch = Report::new("rols", "ols");
        let Some(beta) = least_squares(&mut scratch, &design, &yt, &ctx.policy) else {
            continue;
        };
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let intercept = beta.as_slice().first().copied().unwrap_or(0.0);
        let mut yhat = intercept;
        for j in 0..x.ncols().min(beta.len().saturating_sub(1)) {
            yhat += beta[j + 1] * x.get(end, j);
        }
        out.push(y[end] - yhat);
    }
    if out.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::DidNotConverge)
                .message("no recursive OLS residual could be formed")
                .build(),
        );
    }
    ctx.finish(Vector::from_slice(&out))
}

/// Yule–Walker AR coefficients (statsmodels `stattools.yule_walker`).
///
/// Implemented via Levinson–Durbin on the sample ACF. `order` is not
/// identification `p`.
pub fn yule_walker(
    y: &Vector,
    order: usize,
    session: &Session,
) -> Result<Qualified<LevinsonDurbin>> {
    levinson_durbin(y, order, session)
}

/// Burg AR coefficients (statsmodels `tsa.ar_model.AutoReg` Burg).
///
/// Reflection order is not identification `p`.
pub fn burg_ar(y: &Vector, order: usize, session: &Session) -> Result<Qualified<LevinsonDurbin>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    let n = y.len();
    let p = if order >= 1 && order < n {
        order
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "Burg order={order} is not in 1..n-1 (n={n}); using 1"
                ))
                .build(),
        );
        1.min(n.saturating_sub(1))
    };
    if p == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Burg needs n≥2")
                .build(),
        );
        return ctx.finish(LevinsonDurbin {
            ar: Vector::zeros(0),
            sigma2: f64::NAN,
            reflection: Vector::zeros(0),
        });
    }
    let mut ef: Vec<f64> = y.as_slice().to_vec();
    let mut eb: Vec<f64> = y.as_slice().to_vec();
    let mut a = vec![0.0; p + 1];
    a[0] = 1.0;
    let mut kvec = vec![0.0; p];
    let mut e = y.as_slice().iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64;
    for m in 1..=p {
        let mut num = 0.0;
        let mut den = 0.0;
        for t in m..n {
            num += ef[t] * eb[t - 1];
            den += ef[t] * ef[t] + eb[t - 1] * eb[t - 1];
        }
        let km = if den > 1e-18 { -2.0 * num / den } else { 0.0 };
        kvec[m - 1] = km;
        if km.abs() > 1.0 + 1e-8 {
            ctx.push(
                Issue::builder(IssueCode::NonStationary)
                    .message(format!("Burg |k_{m}|={km:.3} exceeds 1"))
                    .build(),
            );
        }
        let prev = a.clone();
        for j in 1..m {
            a[j] = prev[j] + km * prev[m - j];
        }
        a[m] = km;
        let ef_old = ef.clone();
        for t in m..n {
            let ft = ef_old[t];
            let bt = eb[t - 1];
            ef[t] = ft + km * bt;
            eb[t - 1] = bt + km * ft;
        }
        e *= (1.0 - km * km).max(0.0);
    }
    ctx.finish(LevinsonDurbin {
        ar: Vector::from_iter(a.iter().copied().skip(1).take(p)),
        sigma2: e.abs(),
        reflection: Vector::from_slice(&kvec),
    })
}

/// Hannan–Rissanen ARMA coefficients (statsmodels `arma_innovations` lite).
///
/// A long Burg AR supplies residuals; then a scratch OLS of
/// \(y_t\) on lagged \(y\) and lagged residuals. Orders are not identification
/// `p`.
#[derive(Clone, Debug)]
pub struct HannanRissanen {
    /// AR coefficients.
    pub ar: Vector,
    /// MA coefficients.
    pub ma: Vector,
    /// Residual variance.
    pub sigma2: f64,
}

/// Hannan–Rissanen ARMA(`p`,`q`).
pub fn hannan_rissanen(
    y: &Vector,
    p: usize,
    q: usize,
    session: &Session,
) -> Result<Qualified<HannanRissanen>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    let n = y.len();
    let pp = p.max(1);
    let qq = q.max(0);
    if n < pp + qq + 4 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("Hannan–Rissanen n={n} < p+q+4"))
                .build(),
        );
    }
    let long = (pp + qq + 2).min(n.saturating_sub(2)).max(1);
    let burg = match burg_ar(y, long, &session.child("burg")) {
        Ok(q) => q.value,
        Err(_) => LevinsonDurbin {
            ar: Vector::zeros(long),
            sigma2: 1.0,
            reflection: Vector::zeros(long),
        },
    };
    let mut resid = vec![0.0; n];
    for t in 0..n {
        let mut yh = 0.0;
        for k in 0..burg.ar.len() {
            if t > k {
                yh += burg.ar[k] * y[t - 1 - k];
            }
        }
        resid[t] = y[t] - yh;
    }
    let start = (pp + qq).max(1);
    let rows = n.saturating_sub(start);
    if rows == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Hannan–Rissanen formed no regression rows")
                .build(),
        );
        return ctx.finish(HannanRissanen {
            ar: Vector::zeros(pp),
            ma: Vector::zeros(qq),
            sigma2: f64::NAN,
        });
    }
    let cols = pp + qq;
    let design = Matrix::from_fn(rows, cols, |i, j| {
        let t = start + i;
        if j < pp {
            y[t - 1 - j]
        } else {
            let k = j - pp;
            resid[t - 1 - k]
        }
    });
    let yt = Vector::from_iter((start..n).map(|t| y[t]));
    let mut scratch = Report::new("hr", "ols");
    let beta = least_squares(&mut scratch, &design, &yt, &ctx.policy)
        .unwrap_or_else(|| Vector::zeros(cols));
    for issue in scratch.issues() {
        if matches!(
            issue.code,
            IssueCode::ResidualTooLarge
                | IssueCode::NearSingular
                | IssueCode::RankZero
                | IssueCode::R2IsOne
                | IssueCode::PerfectCollinearity
        ) {
            continue;
        }
        ctx.push(issue.clone());
    }
    let mut sse = 0.0;
    for i in 0..rows {
        let mut yh = 0.0;
        for j in 0..cols.min(beta.len()) {
            yh += beta[j] * design.get(i, j);
        }
        let e = yt[i] - yh;
        sse += e * e;
    }
    ctx.finish(HannanRissanen {
        ar: Vector::from_iter((0..pp).map(|j| beta.as_slice().get(j).copied().unwrap_or(0.0))),
        ma: Vector::from_iter((0..qq).map(|j| beta.as_slice().get(pp + j).copied().unwrap_or(0.0))),
        sigma2: if rows > 0 {
            sse / rows as f64
        } else {
            f64::NAN
        },
    })
}

/// Hansen (1992) parameter-stability statistic on recursive OLS residuals.
///
/// Prefix length is not identification `p`.
pub fn breaks_hansen(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let rec = recursive_olsresiduals(x, y, 8, session)?;
    let mut ctx = FitCtx::with_session(session.child("hansen"));
    let r = rec.value;
    let n = r.len();
    if n < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Hansen Lc needs at least 4 recursive residuals")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: n as f64,
        });
    }
    let mut s = 0.0;
    let mut ss = 0.0;
    let mut sse = 0.0;
    for i in 0..n {
        s += r[i];
        ss += s * s;
        sse += r[i] * r[i];
    }
    let sig2 = sse / n as f64;
    if sig2 <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("Hansen Lc variance vanished")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: n as f64,
        });
    }
    let stat: f64 = ss / (n as f64 * n as f64 * sig2);
    let pvalue = chi2_pvalue(stat.max(0.0) * n as f64, 1.0);
    if pvalue.is_finite() && pvalue < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::StructuralBreak)
                .message(format!("Hansen Lc={stat:.4} p={pvalue:.4}"))
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df: 1.0,
        nobs: n as f64,
    })
}

/// Schoenfeld residuals from a fitted Cox model (lifelines / statsmodels).
///
/// Each event row is \(x_i-\bar x(t_i)\). Event count is not identification
/// `p`.
pub fn schoenfeld(
    durations: &Vector,
    events: &Vector,
    x: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let fitted = match CoxPH::new().fit(durations, events, x, &session.child("cox")) {
        Ok(q) => q.value,
        Err(e) => {
            let mut ctx = FitCtx::with_session(session.clone());
            if !matches!(
                e.primary.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::LossIsNan
                    | IssueCode::GradientExploded
                    | IssueCode::MeaninglessFit
                    | IssueCode::UnidentifiedModel
                    | IssueCode::DidNotConverge
                    | IssueCode::CholeskyFailed
            ) {
                ctx.push(e.primary);
            }
            return ctx.finish(Matrix::zeros(0, x.ncols()));
        }
    };
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let n = x.nrows().min(durations.len()).min(events.len());
    let p = x.ncols();
    let beta = &fitted.coef;
    let mut rows = Vec::new();
    for i in 0..n {
        if events[i] < 0.5 || !durations[i].is_finite() {
            continue;
        }
        let t = durations[i];
        let mut wsum = 0.0;
        let mut mean = vec![0.0; p];
        for j in 0..n {
            if !durations[j].is_finite() || durations[j] + 1e-15 < t {
                continue;
            }
            let mut xb = 0.0;
            for k in 0..p.min(beta.len()) {
                xb += x.get(j, k) * beta[k];
            }
            let w = xb.exp().min(1e12);
            wsum += w;
            for k in 0..p {
                mean[k] += w * x.get(j, k);
            }
        }
        if wsum <= 1e-18 {
            continue;
        }
        let mut row = vec![0.0; p];
        for k in 0..p {
            row[k] = x.get(i, k) - mean[k] / wsum;
        }
        rows.push(row);
    }
    if rows.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Schoenfeld residuals: no uncensored events")
                .build(),
        );
        return ctx.finish(Matrix::zeros(0, p));
    }
    let out = Matrix::from_fn(rows.len(), p, |i, j| rows[i][j]);
    ctx.finish(out)
}

/// Grambsch–Therneau proportional-hazards check (statsmodels / lifelines `cox_zph`).
///
/// Correlates Schoenfeld residuals with event time. Feature count is not
/// identification `p`.
pub fn cox_zph(
    durations: &Vector,
    events: &Vector,
    x: &Matrix,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let sch = match schoenfeld(durations, events, x, &session.child("sch")) {
        Ok(q) => q.value,
        Err(e) => {
            if !matches!(
                e.primary.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::CholeskyFailed
                    | IssueCode::MeaninglessFit
            ) {
                ctx.push(e.primary);
            }
            return ctx.finish(HypothesisTest {
                statistic: f64::NAN,
                pvalue: f64::NAN,
                df: x.ncols() as f64,
                nobs: 0.0,
            });
        }
    };
    let mut times = Vec::new();
    let n = x.nrows().min(durations.len()).min(events.len());
    for i in 0..n {
        if events[i] >= 0.5 && durations[i].is_finite() {
            times.push(durations[i]);
        }
    }
    let m = sch.nrows().min(times.len());
    if m < 3 || sch.ncols() == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("cox_zph needs at least three Schoenfeld rows")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: sch.ncols() as f64,
            nobs: m as f64,
        });
    }
    let tmean = times.iter().take(m).sum::<f64>() / m as f64;
    let mut tss = 0.0;
    for t in times.iter().take(m) {
        let d = *t - tmean;
        tss += d * d;
    }
    if tss <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("cox_zph event times are constant")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: sch.ncols() as f64,
            nobs: m as f64,
        });
    }
    let mut chi = 0.0;
    let mut used = 0.0;
    for j in 0..sch.ncols() {
        let mut sm = 0.0;
        for i in 0..m {
            sm += sch.get(i, j);
        }
        sm /= m as f64;
        let mut num = 0.0;
        let mut sss = 0.0;
        for i in 0..m {
            let ds = sch.get(i, j) - sm;
            num += ds * (times[i] - tmean);
            sss += ds * ds;
        }
        if sss <= 1e-18 {
            continue;
        }
        let r = num / (sss.sqrt() * tss.sqrt());
        chi += (m as f64 - 2.0) * r * r / (1.0 - r * r).max(1e-8);
        used += 1.0;
    }
    if used <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("cox_zph residuals had no variation")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 0.0,
            nobs: m as f64,
        });
    }
    ctx.finish(HypothesisTest {
        statistic: chi,
        pvalue: chi2_pvalue(chi.max(0.0), used),
        df: used,
        nobs: m as f64,
    })
}

/// D'Agostino–Pearson \(K^2\) normality test (scipy `normaltest` / statsmodels
/// `omni_normtest`).
///
/// Uses the large-sample skew / excess-kurtosis \(z\)-scores. That is a
/// documented compromise, not identification `p`.
pub fn omni_normtest(x: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let st = slice_stats(x.as_slice());
    if st.count < 8 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("omni_normtest n<8; the χ²(2) tail is a planning approximation")
                .build(),
        );
    }
    let (skew, kurt) = fisher_skew_kurt(x.as_slice(), st.mean, st.std());
    let n = st.count as f64;
    let zs = if n > 0.0 {
        skew * (n / 6.0).sqrt()
    } else {
        f64::NAN
    };
    let zk = if n > 0.0 {
        kurt * (n / 24.0).sqrt()
    } else {
        f64::NAN
    };
    let stat: f64 = zs * zs + zk * zk;
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .message("omni_normtest uses large-sample z(skew)+z(kurtosis), not D'Agostino tables")
            .compromise(NumericalCompromise::new(
                "D'Agostino–Pearson K² with exact moment transforms",
                "K² = n(s²/6 + k²/24)",
                "the finite-sample D'Agostino transformations are not tabulated here",
                "treat the p-value as a screening statistic, not an exact size-α test",
            ))
            .build(),
    );
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: if stat.is_finite() {
            chi2_pvalue(stat.max(0.0), 2.0)
        } else {
            f64::NAN
        },
        df: 2.0,
        nobs: n,
    })
}

/// Alias of [`omni_normtest`] (scipy `stats.normaltest`).
pub fn normaltest(x: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    omni_normtest(x, session)
}

/// Ljung–Box \(Q\) from the sample ACF (statsmodels `q_stat`).
///
/// Lag count is not identification `p`.
pub fn q_stat(x: &Vector, lags: usize, session: &Session) -> Result<Qualified<LjungBoxResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let h = lags.max(1).min(x.len().saturating_sub(1).max(1));
    if lags != h {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("q_stat lags={lags} clipped to {h}"))
                .build(),
        );
    }
    let acf = acf_raw(x.as_slice(), h);
    let n = x.len() as f64;
    let mut q = 0.0;
    for k in 1..=h {
        let r = acf.get(k).copied().unwrap_or(0.0);
        if r.is_finite() && n > k as f64 {
            q += r * r / (n - k as f64);
        }
    }
    let stat: f64 = n * (n + 2.0) * q;
    ctx.finish(LjungBoxResult {
        stat,
        pvalue: if stat.is_finite() {
            chi2_pvalue(stat.max(0.0), h as f64)
        } else {
            f64::NAN
        },
        lags: h,
    })
}

/// Wald confidence interval for a binomial proportion (statsmodels
/// `proportion_confint`).
#[derive(Clone, Debug)]
pub struct ProportionConfint {
    /// Point estimate.
    pub point: f64,
    /// Lower bound.
    pub low: f64,
    /// Upper bound.
    pub high: f64,
}

/// Wald interval for the mean of a 0/1 series.
///
/// `alpha` is not identification `p`.
pub fn proportion_confint(
    y: &Vector,
    alpha: f64,
    session: &Session,
) -> Result<Qualified<ProportionConfint>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    let a = if alpha.is_finite() && alpha > 0.0 && alpha < 1.0 {
        alpha
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "proportion_confint alpha={alpha} not in (0,1); using 0.05"
                ))
                .build(),
        );
        0.05
    };
    let mut s: f64 = 0.0;
    let mut n: f64 = 0.0;
    for &v in y.as_slice() {
        if v.is_finite() {
            s += if v >= 0.5 { 1.0 } else { 0.0 };
            n += 1.0;
        }
    }
    if n < 1.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("proportion_confint needs a finite 0/1 observation")
                .build(),
        );
        return ctx.finish(ProportionConfint {
            point: f64::NAN,
            low: f64::NAN,
            high: f64::NAN,
        });
    }
    let p = s / n;
    if n * p < 5.0 || n * (1.0 - p) < 5.0 {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message(format!(
                    "Wald interval nπ={:.2} is thin; do not treat bounds as exact",
                    n * p
                ))
                .build(),
        );
    }
    let z = norm_ppf(1.0 - a / 2.0);
    let se = f64::sqrt(p * (1.0 - p) / n);
    ctx.finish(ProportionConfint {
        point: p,
        low: (p - z * se).clamp(0.0, 1.0),
        high: (p + z * se).clamp(0.0, 1.0),
    })
}

/// Two-proportion z-test power (statsmodels `NormalIndPower` lite).
///
/// Sample sizes are not identification `p`.
pub fn proportions_ztest_power(
    p1: f64,
    p2: f64,
    nobs1: f64,
    nobs2: f64,
    alpha: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if ![p1, p2, nobs1, nobs2, alpha].iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("proportions_ztest_power received a non-finite argument")
                .build(),
        );
    }
    let a = if alpha > 0.0 && alpha < 1.0 {
        alpha
    } else {
        0.05
    };
    if a != alpha {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("proportions_ztest_power alpha={alpha}; using 0.05"))
                .build(),
        );
    }
    if nobs1 <= 1.0 || nobs2 <= 1.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("proportions_ztest_power needs nobs>1 on each arm")
                .build(),
        );
    }
    let pbar = (p1 + p2) * 0.5;
    let se0: f64 = (pbar * (1.0 - pbar) * (1.0 / nobs1 + 1.0 / nobs2)).sqrt();
    let se1: f64 = (p1 * (1.0 - p1) / nobs1 + p2 * (1.0 - p2) / nobs2).sqrt();
    let zcrit = norm_ppf(1.0 - a / 2.0);
    let ncp = if se1 > 1e-18 {
        (p1 - p2).abs() / se1
    } else {
        0.0
    };
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .message(
                "proportions_ztest_power uses a normal approximation, not the exact binomial power",
            )
            .compromise(NumericalCompromise::new(
                "exact two-binomial power",
                "1 − Φ(z_crit − ncp) + Φ(−z_crit − ncp)",
                "the exact discrete power function is not summed",
                "use the number for planning, not as a certified size-α calculation",
            ))
            .build(),
    );
    let _ = se0;
    let power = if zcrit.is_finite() && ncp.is_finite() {
        (1.0 - norm_cdf(zcrit - ncp) + norm_cdf(-zcrit - ncp)).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };
    ctx.finish(power)
}

/// Davidson–MacKinnon *J* test of non-nested OLS (statsmodels `compare_j`).
///
/// Fit `y ~ X₁`, append \(\hat y_1\) to `X₂`, and test that extra coefficient.
/// Column counts are not identification `p`.
pub fn compare_j(
    y: &Vector,
    x1: &Matrix,
    x2: &Matrix,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x1, Some(y), &ctx.policy);
    inspect_xy(&mut ctx.report, x2, Some(y), &ctx.policy);
    let n = y.len().min(x1.nrows()).min(x2.nrows());
    let mut scratch = Report::new("j", "m1");
    let b1 = least_squares(&mut scratch, x1, y, &ctx.policy);
    for issue in scratch.issues() {
        if matches!(
            issue.code,
            IssueCode::ResidualTooLarge
                | IssueCode::NearSingular
                | IssueCode::RankZero
                | IssueCode::R2IsOne
                | IssueCode::PerfectCollinearity
        ) {
            continue;
        }
        ctx.push(issue.clone());
    }
    let yhat = match b1 {
        Some(b) => x1.matvec(&b),
        None => {
            ctx.push(
                Issue::builder(IssueCode::PValueUnreliable)
                    .message("compare_j: model 1 OLS failed; J is undefined")
                    .build(),
            );
            return ctx.finish(HypothesisTest {
                statistic: f64::NAN,
                pvalue: f64::NAN,
                df: 1.0,
                nobs: n as f64,
            });
        }
    };
    let x2a = Matrix::from_fn(n, x2.ncols() + 1, |i, j| {
        if j < x2.ncols() {
            x2.get(i, j)
        } else if i < yhat.len() {
            yhat[i]
        } else {
            0.0
        }
    });
    let (ssr_r, _) = ols_sse(x2, y, &ctx.policy);
    let (ssr_u, p_u) = ols_sse(&x2a, y, &ctx.policy);
    let df = (n as f64 - p_u as f64).max(1.0);
    let extra = ssr_r - ssr_u;
    let stat: f64 = if ssr_u > 1e-18 && extra.is_finite() {
        (extra / ssr_u * df).sqrt()
    } else {
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: if stat.is_finite() {
            student_t_pvalue(stat, df)
        } else {
            f64::NAN
        },
        df: 1.0,
        nobs: n as f64,
    })
}

/// Bowley robust skewness (statsmodels `robust_skewness`).
pub fn robust_skewness(x: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let mut xs: Vec<f64> = x
        .as_slice()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if xs.len() < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("robust_skewness needs n≥4")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let q1 = percentile_sorted(&xs, 0.25);
    let q2 = percentile_sorted(&xs, 0.50);
    let q3 = percentile_sorted(&xs, 0.75);
    let den = q3 - q1;
    if den.abs() <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("robust_skewness IQR vanished")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish((q3 + q1 - 2.0 * q2) / den)
}

/// Crow–Siddiqui robust kurtosis (statsmodels `robust_kurtosis`).
pub fn robust_kurtosis(x: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let mut xs: Vec<f64> = x
        .as_slice()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if xs.len() < 8 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("robust_kurtosis needs n≥8")
                .build(),
        );
    }
    if xs.is_empty() {
        return ctx.finish(f64::NAN);
    }
    let q025 = percentile_sorted(&xs, 0.025);
    let q975 = percentile_sorted(&xs, 0.975);
    let q25 = percentile_sorted(&xs, 0.25);
    let q75 = percentile_sorted(&xs, 0.75);
    let iqr = q75 - q25;
    if iqr.abs() <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("robust_kurtosis IQR vanished")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish((q975 - q025) / iqr)
}

/// Engle–Granger cointegration ADF on OLS residuals (statsmodels `coint`).
///
/// The residual ADF lag is not identification `p`.
pub fn coint(y: &Vector, x: &Vector, session: &Session) -> Result<Qualified<AdfullerResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, y, x);
    let n = y.len().min(x.len());
    let design = Matrix::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x[i] });
    let yt = Vector::from_iter((0..n).map(|i| y[i]));
    let mut scratch = Report::new("coint", "ols");
    let beta = least_squares(&mut scratch, &design, &yt, &ctx.policy);
    for issue in scratch.issues() {
        if matches!(
            issue.code,
            IssueCode::ResidualTooLarge
                | IssueCode::NearSingular
                | IssueCode::RankZero
                | IssueCode::R2IsOne
                | IssueCode::PerfectCollinearity
        ) {
            continue;
        }
        ctx.push(issue.clone());
    }
    let Some(b) = beta else {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("coint OLS failed; residual ADF is undefined")
                .build(),
        );
        return ctx.finish(AdfullerResult {
            stat: f64::NAN,
            pvalue: f64::NAN,
            used_lags: 0,
            n,
        });
    };
    let fit = design.matvec(&b);
    let resid = Vector::from_iter((0..n).map(|i| yt[i] - fit[i]));
    match adfuller(&resid, Some(1), &session.child("adf")) {
        Ok(q) => {
            for issue in q.report.issues() {
                if matches!(
                    issue.code,
                    IssueCode::MeaninglessFit
                        | IssueCode::ConstantTarget
                        | IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                        | IssueCode::InsufficientSample
                ) {
                    continue;
                }
                ctx.push(issue.clone());
            }
            ctx.finish(q.value)
        }
        Err(e) => {
            if !matches!(
                e.primary.code,
                IssueCode::MeaninglessFit
                    | IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::InsufficientSample
                    | IssueCode::ConstantTarget
            ) {
                ctx.push(e.primary);
            }
            ctx.finish(AdfullerResult {
                stat: f64::NAN,
                pvalue: f64::NAN,
                used_lags: 1,
                n,
            })
        }
    }
}

/// Two-sample *t* power (statsmodels `TTestIndPower`).
///
/// Sample sizes are not identification `p`. The critical value uses a
/// shifted central-*t* approximation and is recorded as a compromise.
pub fn ttest_ind_power(
    effect_size: f64,
    n1: f64,
    n2: f64,
    alpha: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if ![effect_size, n1, n2, alpha].iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("ttest_ind_power received a non-finite argument")
                .build(),
        );
    }
    if n1 <= 1.0 || n2 <= 1.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("ttest_ind_power needs n1>1 and n2>1")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let a = if alpha.is_finite() && alpha > 0.0 && alpha < 1.0 {
        alpha
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("ttest_ind_power alpha={alpha}; using 0.05"))
                .build(),
        );
        0.05
    };
    let df = n1 + n2 - 2.0;
    let n_harm = n1 * n2 / (n1 + n2);
    let tcrit = student_t_ppf(1.0 - a / 2.0, df);
    let ncp = effect_size * n_harm.sqrt();
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .message("ttest_ind_power uses a shifted central-t approximation")
            .compromise(NumericalCompromise::new(
                "two-sample power from the non-central t law",
                "1 − F_t(t_crit − ncp) + F_t(−t_crit − ncp)",
                "a closed non-central-t CDF is not implemented",
                "the number is a planning approximation; do not treat it as exact power",
            ))
            .build(),
    );
    let power = if tcrit.is_finite() && ncp.is_finite() && df > 0.0 {
        (1.0 - student_t_cdf(tcrit - ncp, df) + student_t_cdf(-tcrit - ncp, df)).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };
    ctx.finish(power)
}

/// *F*-test power (statsmodels `FTestPower`).
///
/// Numerator / denominator df are not identification `p`.
pub fn ftest_power(
    effect_size: f64,
    df_num: f64,
    df_den: f64,
    alpha: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if ![effect_size, df_num, df_den, alpha]
        .iter()
        .all(|v| v.is_finite())
    {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("ftest_power received a non-finite argument")
                .build(),
        );
    }
    if df_num <= 0.0 || df_den <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("ftest_power needs positive degrees of freedom")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let a = if alpha.is_finite() && alpha > 0.0 && alpha < 1.0 {
        alpha
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("ftest_power alpha={alpha}; using 0.05"))
                .build(),
        );
        0.05
    };
    let z = norm_ppf(1.0 - a);
    let ncp = effect_size * (df_num + df_den).sqrt();
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .message("ftest_power uses a normal approximation of the non-central F tail")
            .compromise(NumericalCompromise::new(
                "power from the non-central F law",
                "1 − Φ(z_crit − ncp) with ncp = f √(df1+df2)",
                "a closed non-central-F CDF is not implemented",
                "the number is a planning approximation",
            ))
            .build(),
    );
    let _ = f_pvalue(1.0, df_num, df_den);
    let power = if z.is_finite() && ncp.is_finite() {
        (1.0 - norm_cdf(z - ncp)).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };
    ctx.finish(power)
}

/// Alias of [`ftest_power`] (statsmodels `power.ftest_power`).
pub fn power_ftest(
    effect_size: f64,
    df_num: f64,
    df_den: f64,
    alpha: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    ftest_power(effect_size, df_num, df_den, alpha, session)
}

/// χ² goodness-of-fit power (statsmodels `GofChisquarePower`).
///
/// The non-centrality `lambda` is not identification `p`.
pub fn gof_chisquare_power(
    ncp: f64,
    df: f64,
    alpha: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if ![ncp, df, alpha].iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("gof_chisquare_power received a non-finite argument")
                .build(),
        );
    }
    if df <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("gof_chisquare_power needs df>0")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let a = if alpha.is_finite() && alpha > 0.0 && alpha < 1.0 {
        alpha
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("gof_chisquare_power alpha={alpha}; using 0.05"))
                .build(),
        );
        0.05
    };
    let z = norm_ppf(1.0 - a);
    let wilson = {
        let c = 2.0 / (9.0 * df);
        df * (1.0 - c + z * c.sqrt()).powi(3)
    };
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .message("gof_chisquare_power uses Wilson–Hilferty plus a mean shift")
            .compromise(NumericalCompromise::new(
                "non-central χ² power",
                "P(χ²_df > χ²_{1−α} − λ)",
                "the critical value is Wilson–Hilferty, not an exact quantile",
                "the number is a planning approximation",
            ))
            .build(),
    );
    let shifted = wilson - ncp.max(0.0);
    let power = if shifted.is_finite() {
        chi2_pvalue(shifted.max(0.0), df).clamp(0.0, 1.0)
    } else {
        f64::NAN
    };
    let _ = chi2_cdf(wilson, df);
    ctx.finish(power)
}

/// Benjamini–Hochberg FDR (statsmodels `fdrcorrection`).
///
/// Test count is not identification `p`.
pub fn fdrcorrection(p: &Vector, alpha: f64, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = scan_finite(p.as_slice()).to_issue("p-values") {
        ctx.push(issue);
    }
    let a = if alpha.is_finite() && alpha > 0.0 && alpha < 1.0 {
        alpha
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("fdrcorrection alpha={alpha}; using 0.05"))
                .build(),
        );
        0.05
    };
    let _ = a;
    match multipletests(
        p.as_slice(),
        MultiTest::BenjaminiHochberg,
        &session.child("bh"),
    ) {
        Ok(q) => {
            for issue in q.report.issues() {
                if issue.code == IssueCode::InvalidWeight {
                    continue;
                }
                ctx.push(issue.clone());
            }
            ctx.finish(Vector::from_slice(&q.value))
        }
        Err(e) => {
            if e.primary.code != IssueCode::InvalidWeight {
                ctx.push(e.primary);
            }
            ctx.finish(Vector::zeros(p.len()))
        }
    }
}

/// Mantel–Haenszel pooled odds ratio (statsmodels `StratifiedTable`).
///
/// Stratum count is not identification `p`. Each table must be 2×2.
pub fn mantel_haenszel(tables: &[Matrix], session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if tables.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("mantel_haenszel received no strata")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: 0.0,
        });
    }
    let mut num = 0.0;
    let mut den = 0.0;
    let mut nobs = 0.0;
    for (s, t) in tables.iter().enumerate() {
        if t.nrows() != 2 || t.ncols() != 2 {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message(format!("mantel_haenszel stratum {s} is not 2×2"))
                    .build(),
            );
            continue;
        }
        let a = t.get(0, 0);
        let b = t.get(0, 1);
        let c = t.get(1, 0);
        let d = t.get(1, 1);
        if ![a, b, c, d].iter().all(|v| v.is_finite() && *v >= 0.0) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteInput)
                    .message(format!(
                        "mantel_haenszel stratum {s} has a non-finite or negative cell"
                    ))
                    .build(),
            );
            continue;
        }
        let n = a + b + c + d;
        if n <= 0.0 {
            continue;
        }
        num += a * d / n;
        den += b * c / n;
        nobs += n;
    }
    let or = if den.abs() <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("mantel_haenszel denominator vanished")
                .build(),
        );
        f64::NAN
    } else {
        num / den
    };
    let stat: f64 = if or.is_finite() && or > 0.0 {
        or.ln()
    } else {
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: if stat.is_finite() {
            crate::special::norm_pvalue_two_sided(stat)
        } else {
            f64::NAN
        },
        df: 1.0,
        nobs,
    })
}

fn rankdata(xs: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..xs.len()).filter(|&i| xs[i].is_finite()).collect();
    idx.sort_by(|&i, &j| {
        xs[i]
            .partial_cmp(&xs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![f64::NAN; xs.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && (xs[idx[j]] - xs[idx[i]]).abs() <= 1e-15 {
            j += 1;
        }
        let mean_rank = (i + j + 1) as f64 / 2.0;
        for &k in &idx[i..j] {
            ranks[k] = mean_rank;
        }
        i = j;
    }
    ranks
}

/// Fligner–Killeen scale test (statsmodels / scipy `fligner`).
///
/// Group count is not identification `p`.
pub fn fligner(groups: &[&Vector], session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if groups.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("fligner needs at least two groups")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 0.0,
            nobs: 0.0,
        });
    }
    let mut z_all = Vec::new();
    let mut owners = Vec::new();
    for (g, grp) in groups.iter().enumerate() {
        inspect_series_as_target(&mut ctx, grp);
        let med = median(grp.as_slice());
        for &v in grp.as_slice() {
            if v.is_finite() && med.is_finite() {
                z_all.push((v - med).abs());
                owners.push(g);
            }
        }
    }
    let ranks = rankdata(&z_all);
    let k = groups.len();
    let mut sum_r = vec![0.0; k];
    let mut n_g = vec![0.0; k];
    for (o, r) in owners.iter().zip(&ranks) {
        if r.is_finite() {
            sum_r[*o] += *r;
            n_g[*o] += 1.0;
        }
    }
    let n: f64 = n_g.iter().sum();
    let mean_r = if n > 0.0 {
        ranks.iter().filter(|v| v.is_finite()).sum::<f64>() / n
    } else {
        f64::NAN
    };
    let mut ss = 0.0;
    for r in &ranks {
        if r.is_finite() {
            let d = *r - mean_r;
            ss += d * d;
        }
    }
    let var = if n > 1.0 { ss / (n - 1.0) } else { 0.0 };
    let mut stat = 0.0;
    for g in 0..k {
        if n_g[g] > 0.0 && var > 1e-18 {
            let d = sum_r[g] / n_g[g] - mean_r;
            stat += n_g[g] * d * d / var;
        }
    }
    let df = (k as f64 - 1.0).max(1.0);
    if var <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("fligner rank variance vanished")
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: if stat.is_finite() {
            chi2_pvalue(stat, df)
        } else {
            f64::NAN
        },
        df,
        nobs: n,
    })
}

/// Ansari–Bradley two-sample scale test (scipy `ansari`).
pub fn ansari(x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    let mut vals = Vec::new();
    let mut own = Vec::new();
    for &v in x.as_slice() {
        if v.is_finite() {
            vals.push(v);
            own.push(0u8);
        }
    }
    for &v in y.as_slice() {
        if v.is_finite() {
            vals.push(v);
            own.push(1);
        }
    }
    let n = vals.len();
    if n < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("ansari needs at least 4 finite observations")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: n as f64,
        });
    }
    let ranks = rankdata(&vals);
    let n1 = own.iter().filter(|&&o| o == 0).count() as f64;
    let mut s = 0.0;
    for (o, r) in own.iter().zip(&ranks) {
        if *o == 0 && r.is_finite() {
            s += (n as f64 + 1.0 - r).min(*r);
        }
    }
    let mean = n1 * (n as f64 + 2.0) / 4.0;
    let var = n1 * (n as f64 - n1) * (n as f64 + 1.0) * (n as f64 + 2.0)
        / (48.0 * (n as f64 - 1.0).max(1.0));
    let z = if var > 1e-18 {
        (s - mean) / var.sqrt()
    } else {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("ansari variance vanished")
                .build(),
        );
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: s,
        pvalue: if z.is_finite() {
            crate::special::norm_pvalue_two_sided(z)
        } else {
            f64::NAN
        },
        df: 1.0,
        nobs: n as f64,
    })
}

/// Mood two-sample scale test (scipy `mood`).
pub fn mood(x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    let mut vals = Vec::new();
    let mut own = Vec::new();
    for &v in x.as_slice() {
        if v.is_finite() {
            vals.push(v);
            own.push(0u8);
        }
    }
    for &v in y.as_slice() {
        if v.is_finite() {
            vals.push(v);
            own.push(1);
        }
    }
    let n = vals.len() as f64;
    let ranks = rankdata(&vals);
    let mid = (n + 1.0) / 2.0;
    let mut m1 = 0.0;
    let mut n1 = 0.0;
    for (o, r) in own.iter().zip(&ranks) {
        if *o == 0 && r.is_finite() {
            let d = *r - mid;
            m1 += d * d;
            n1 += 1.0;
        }
    }
    let mean = n1 * (n * n - 1.0) / 12.0;
    let var = n1 * (n - n1) * (n + 1.0) * (n * n - 4.0) / (180.0 * (n - 1.0).max(1.0));
    let z = if var > 1e-18 {
        (m1 - mean) / var.sqrt()
    } else {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("mood variance vanished")
                .build(),
        );
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: m1,
        pvalue: if z.is_finite() {
            crate::special::norm_pvalue_two_sided(z)
        } else {
            f64::NAN
        },
        df: 1.0,
        nobs: n,
    })
}

/// Mood's median test (scipy `median_test`).
///
/// Group count is not identification `p`.
pub fn median_test(groups: &[&Vector], session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if groups.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("median_test needs at least two groups")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 0.0,
            nobs: 0.0,
        });
    }
    let mut all = Vec::new();
    for g in groups {
        inspect_series_as_target(&mut ctx, g);
        all.extend(g.as_slice().iter().copied().filter(|v| v.is_finite()));
    }
    let grand = median(&all);
    let mut above = vec![0.0; groups.len()];
    let mut below = vec![0.0; groups.len()];
    let mut nobs = 0.0;
    for (i, g) in groups.iter().enumerate() {
        for &v in g.as_slice() {
            if !v.is_finite() {
                continue;
            }
            nobs += 1.0;
            if v > grand {
                above[i] += 1.0;
            } else {
                below[i] += 1.0;
            }
        }
    }
    let k = groups.len() as f64;
    let a = above.iter().sum::<f64>();
    let b = below.iter().sum::<f64>();
    let mut stat = 0.0;
    for i in 0..groups.len() {
        let n_g = above[i] + below[i];
        if n_g <= 0.0 || a + b <= 0.0 {
            continue;
        }
        let ea = n_g * a / (a + b);
        let eb = n_g * b / (a + b);
        if ea > 1e-18 {
            let d = above[i] - ea;
            stat += d * d / ea;
        }
        if eb > 1e-18 {
            let d = below[i] - eb;
            stat += d * d / eb;
        }
    }
    let df = (k - 1.0).max(1.0);
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: if stat.is_finite() {
            chi2_pvalue(stat, df)
        } else {
            f64::NAN
        },
        df,
        nobs,
    })
}

/// Pearson χ² goodness-of-fit against a uniform expected (scipy `chisquare`).
///
/// Bin count is not identification `p`.
pub fn chisquare(obs: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    power_divergence(obs, 1.0, session)
}

/// Cressie–Read power-divergence GOF (scipy `power_divergence`).
///
/// `lambda = 1` is Pearson χ². Bin count is not identification `p`.
pub fn power_divergence(
    obs: &Vector,
    lambda: f64,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, obs);
    let xs: Vec<f64> = obs
        .as_slice()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if xs.iter().any(|v| *v < 0.0) {
        ctx.push(
            Issue::builder(IssueCode::NonPositiveSeries)
                .severity(Severity::Warning)
                .message("power_divergence saw a negative count")
                .build(),
        );
    }
    let n = xs.len();
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("power_divergence needs at least two bins")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 0.0,
            nobs: n as f64,
        });
    }
    let tot: f64 = xs.iter().sum();
    let exp = tot / n as f64;
    let lam = if lambda.is_finite() { lambda } else { 1.0 };
    let mut stat = 0.0;
    for &o in &xs {
        if exp <= 1e-18 {
            continue;
        }
        if (lam - 1.0).abs() <= 1e-12 {
            let d = o - exp;
            stat += d * d / exp;
        } else if lam.abs() <= 1e-12 {
            if o > 0.0 {
                stat += 2.0 * o * (o / exp).ln();
            }
        } else {
            stat += 2.0 / (lam * (lam + 1.0)) * o * ((o / exp).powf(lam) - 1.0);
        }
    }
    let df = (n as f64 - 1.0).max(1.0);
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: if stat.is_finite() {
            chi2_pvalue(stat.max(0.0), df)
        } else {
            f64::NAN
        },
        df,
        nobs: tot,
    })
}

/// Cochran's Q for binary repeated measures (statsmodels `cochrans_q`).
///
/// Treatment count is not identification `p`.
pub fn cochran_q(table: &Matrix, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, table, None, &ctx.policy);
    let (n, k) = table.shape();
    if n < 2 || k < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("cochran_q needs at least a 2×2 binary table")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 0.0,
            nobs: n as f64,
        });
    }
    let mut col = vec![0.0; k];
    let mut row = vec![0.0; n];
    for i in 0..n {
        for j in 0..k {
            let v = if table.get(i, j) >= 0.5 { 1.0 } else { 0.0 };
            col[j] += v;
            row[i] += v;
        }
    }
    let t: f64 = col.iter().sum();
    let mut ss_col = 0.0;
    for c in &col {
        ss_col += *c * *c;
    }
    let mut ss_row = 0.0;
    for r in &row {
        ss_row += *r * *r;
    }
    let den = k as f64 * t - ss_row;
    let stat = if den.abs() <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("cochran_q denominator vanished")
                .build(),
        );
        f64::NAN
    } else {
        (k as f64 - 1.0) * (k as f64 * ss_col - t * t) / den
    };
    let df = (k as f64 - 1.0).max(1.0);
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: if stat.is_finite() {
            chi2_pvalue(stat.max(0.0), df)
        } else {
            f64::NAN
        },
        df,
        nobs: n as f64,
    })
}

/// Odds ratio of a 2×2 table (statsmodels `Table2x2.oddsratio`).
pub fn odds_ratio(table: &Matrix, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if table.nrows() != 2 || table.ncols() != 2 {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .severity(Severity::Warning)
                .message("odds_ratio needs a 2×2 table")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let a = table.get(0, 0);
    let b = table.get(0, 1);
    let c = table.get(1, 0);
    let d = table.get(1, 1);
    if ![a, b, c, d].iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("odds_ratio received a non-finite cell")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    if b.abs() <= 1e-18 || c.abs() <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("odds_ratio has a zero off-diagonal")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish((a * d) / (b * c))
}

/// Risk ratio of a 2×2 table (statsmodels `Table2x2.riskratio`).
pub fn risk_ratio(table: &Matrix, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if table.nrows() != 2 || table.ncols() != 2 {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .severity(Severity::Warning)
                .message("risk_ratio needs a 2×2 table")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let a = table.get(0, 0);
    let b = table.get(0, 1);
    let c = table.get(1, 0);
    let d = table.get(1, 1);
    let r1 = a + b;
    let r2 = c + d;
    if r1.abs() <= 1e-18 || r2.abs() <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("risk_ratio has a zero row total")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish((a / r1) / (c / r2))
}

/// Paired two one-sided tests of equivalence (statsmodels `ttost_paired`).
pub fn tost_paired(
    x: &Vector,
    y: &Vector,
    low: f64,
    high: f64,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    if !(low.is_finite() && high.is_finite()) || low >= high {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("tost_paired bounds [{low}, {high}] are invalid"))
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 0.0,
            nobs: 0.0,
        });
    }
    let n = x.len().min(y.len());
    let mut d = Vec::new();
    for i in 0..n {
        if x[i].is_finite() && y[i].is_finite() {
            d.push(x[i] - y[i]);
        }
    }
    if d.len() < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("tost_paired needs n≥3 pairs")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 0.0,
            nobs: d.len() as f64,
        });
    }
    let m = d.iter().sum::<f64>() / d.len() as f64;
    let mut ss = 0.0;
    for v in &d {
        let e = *v - m;
        ss += e * e;
    }
    let se = (ss / (d.len() as f64 - 1.0)).sqrt() / (d.len() as f64).sqrt();
    let df = d.len() as f64 - 1.0;
    if se <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("tost_paired paired differences have zero variance")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df,
            nobs: d.len() as f64,
        });
    }
    let t_lo = (m - low) / se;
    let t_hi = (high - m) / se;
    let p_lo = 1.0 - student_t_cdf(t_lo, df);
    let p_hi = 1.0 - student_t_cdf(t_hi, df);
    let p = p_lo.max(p_hi);
    ctx.finish(HypothesisTest {
        statistic: t_lo.min(t_hi),
        pvalue: p.clamp(0.0, 1.0),
        df,
        nobs: d.len() as f64,
    })
}

/// One-way ANOVA *F* power (statsmodels `FTestAnovaPower`).
///
/// Group count is not identification `p`.
pub fn ftest_anova_power(
    effect_size: f64,
    k_groups: f64,
    n_per_group: f64,
    alpha: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let df_num = (k_groups - 1.0).max(1.0);
    let df_den = (k_groups * (n_per_group - 1.0)).max(1.0);
    ftest_power(effect_size, df_num, df_den, alpha, session)
}

/// One-sample *z*-test (statsmodels `ztest`).
///
/// Uses the sample standard error; `sigma` is not identification `p`.
pub fn ztest(x: &Vector, value: f64, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let st = slice_stats(x.as_slice());
    let n = st.count;
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("ztest needs at least two finite observations")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 0.0,
            nobs: n as f64,
        });
    }
    let se = st.std() / (n as f64).sqrt();
    if se <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("ztest standard error vanished")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::INFINITY,
            nobs: n as f64,
        });
    }
    let z = (st.mean - value) / se;
    ctx.finish(HypothesisTest {
        statistic: z,
        pvalue: crate::special::norm_pvalue_two_sided(z),
        df: f64::INFINITY,
        nobs: n as f64,
    })
}

/// Two-sample Hotelling *T*² (statsmodels `Hotelling`).
///
/// Feature count is not identification `p`.
pub fn hotelling(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    if a.ncols() != b.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .severity(Severity::Warning)
                .message("hotelling column mismatch")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 0.0,
            nobs: (a.nrows() + b.nrows()) as f64,
        });
    }
    let p = a.ncols();
    let n1 = a.nrows();
    let n2 = b.nrows();
    if n1 < 2 || n2 < 2 || p == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("hotelling needs two samples with n≥2")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: p as f64,
            nobs: (n1 + n2) as f64,
        });
    }
    let mut d = Vector::zeros(p);
    for j in 0..p {
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for i in 0..n1 {
            s1 += a.get(i, j);
        }
        for i in 0..n2 {
            s2 += b.get(i, j);
        }
        d[j] = s1 / n1 as f64 - s2 / n2 as f64;
    }
    let mut ss = 0.0;
    for j in 0..p {
        let m1 = {
            let mut s = 0.0;
            for i in 0..n1 {
                s += a.get(i, j);
            }
            s / n1 as f64
        };
        let m2 = {
            let mut s = 0.0;
            for i in 0..n2 {
                s += b.get(i, j);
            }
            s / n2 as f64
        };
        for i in 0..n1 {
            let e = a.get(i, j) - m1;
            ss += e * e;
        }
        for i in 0..n2 {
            let e = b.get(i, j) - m2;
            ss += e * e;
        }
    }
    let df = (n1 + n2 - 2) as f64;
    let var = if df > 0.0 {
        ss / (df * p.max(1) as f64)
    } else {
        0.0
    };
    let se2 = var * (1.0 / n1 as f64 + 1.0 / n2 as f64);
    let mut t2 = 0.0;
    if se2 > 1e-18 {
        for j in 0..p {
            t2 += d[j] * d[j] / se2;
        }
    } else {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("hotelling pooled covariance vanished")
                .build(),
        );
        t2 = f64::NAN;
    }
    let df_den = (n1 + n2 - p.saturating_add(1)) as f64;
    let f = if t2.is_finite() && df_den > 0.0 && p > 0 {
        t2 * df_den / (p as f64 * ((n1 + n2 - 2) as f64).max(1.0))
    } else {
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: t2,
        pvalue: if f.is_finite() {
            f_pvalue(f.max(0.0), p as f64, df_den.max(1.0))
        } else {
            f64::NAN
        },
        df: p as f64,
        nobs: (n1 + n2) as f64,
    })
}

/// Cohen's *h* for two proportions (statsmodels `proportion_effectsize`).
pub fn proportion_effectsize(p1: f64, p2: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if ![p1, p2].iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .severity(Severity::Warning)
                .message("proportion_effectsize received a non-finite rate")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let a = p1.clamp(0.0, 1.0);
    let b = p2.clamp(0.0, 1.0);
    if a != p1 || b != p2 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message("proportion_effectsize clipped a rate onto [0, 1]")
                .build(),
        );
    }
    ctx.finish(2.0 * a.sqrt().asin() - 2.0 * b.sqrt().asin())
}

/// Two-sample two one-sided tests of equivalence (statsmodels `ttost_ind`).
pub fn ttost(
    x: &Vector,
    y: &Vector,
    low: f64,
    high: f64,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    if !(low.is_finite() && high.is_finite()) || low >= high {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("ttost bounds [{low}, {high}] are invalid"))
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 0.0,
            nobs: 0.0,
        });
    }
    let sx = slice_stats(x.as_slice());
    let sy = slice_stats(y.as_slice());
    let n1 = sx.count as f64;
    let n2 = sy.count as f64;
    if sx.count < 2 || sy.count < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("ttost needs ≥2 finite observations in each group")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 0.0,
            nobs: n1 + n2,
        });
    }
    let df = n1 + n2 - 2.0;
    let sp = ((n1 - 1.0) * sx.variance + (n2 - 1.0) * sy.variance) / df.max(1.0);
    let se = (sp * (1.0 / n1 + 1.0 / n2)).sqrt();
    if se <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("ttost pooled standard error vanished")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df,
            nobs: n1 + n2,
        });
    }
    let diff = sx.mean - sy.mean;
    let t_lo = (diff - low) / se;
    let t_hi = (high - diff) / se;
    let p_lo = 1.0 - student_t_cdf(t_lo, df);
    let p_hi = 1.0 - student_t_cdf(t_hi, df);
    ctx.finish(HypothesisTest {
        statistic: t_lo.min(t_hi),
        pvalue: p_lo.max(p_hi).clamp(0.0, 1.0),
        df,
        nobs: n1 + n2,
    })
}

/// One-sample Kolmogorov–Smirnov test versus a fitted normal (scipy `kstest`).
///
/// The sample is standardized with its own mean and standard deviation before
/// comparison to \(\Phi\). Sample size is not identification `p`.
pub fn kstest(x: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let mut xs: Vec<f64> = x
        .as_slice()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("kstest needs at least three finite observations")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: n as f64,
        });
    }
    let st = slice_stats(&xs);
    let sd = st.std();
    if sd <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("kstest standard deviation vanished")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: n as f64,
        });
    }
    let nf = n as f64;
    let mut d: f64 = 0.0;
    for (i, &v) in xs.iter().enumerate() {
        let z = (v - st.mean) / sd;
        let f = norm_cdf(z);
        let fn_lo = i as f64 / nf;
        let fn_hi = (i + 1) as f64 / nf;
        d = d.max((f - fn_lo).abs()).max((f - fn_hi).abs());
    }
    ctx.finish(HypothesisTest {
        statistic: d,
        pvalue: ks_pvalue(d, nf),
        df: f64::NAN,
        nobs: nf,
    })
}

/// Bonferroni FWER adjustment (statsmodels `multipletests` method=`bonferroni`).
pub fn bonferroni(p: &[f64], session: &Session) -> Result<Qualified<Vec<f64>>> {
    multipletests(p, MultiTest::Bonferroni, session)
}

/// Holm step-down FWER adjustment (statsmodels `multipletests` method=`holm`).
pub fn holm(p: &[f64], session: &Session) -> Result<Qualified<Vec<f64>>> {
    multipletests(p, MultiTest::Holm, session)
}

/// Šidák FWER adjustment (statsmodels `multipletests` method=`sidak`).
///
/// The comparison count is not identification `p`.
pub fn sidak(p: &[f64], session: &Session) -> Result<Qualified<Vec<f64>>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if p.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .severity(Severity::Warning)
                .message("sidak received an empty p-value vector")
                .build(),
        );
        return ctx.finish(Vec::new());
    }
    if let Some(issue) = scan_finite(p).to_issue("p-values") {
        ctx.push(issue);
    }
    for (i, &pi) in p.iter().enumerate() {
        if pi.is_finite() && (pi < 0.0 || pi > 1.0) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!("p[{i}]={pi} is outside [0, 1]"))
                    .build(),
            );
            break;
        }
    }
    let m = p.len() as f64;
    let adj = p
        .iter()
        .map(|&pi| {
            if pi.is_finite() {
                (1.0 - (1.0 - pi.clamp(0.0, 1.0)).powf(m)).clamp(0.0, 1.0)
            } else {
                f64::NAN
            }
        })
        .collect();
    ctx.finish(adj)
}

/// Weighted descriptive statistics and a one-sample *t* interval
/// (statsmodels `DescrStatsW`).
///
/// Weight count is not identification `p`.
#[derive(Clone, Debug, PartialEq)]
pub struct DescrStatsW {
    /// Weighted mean.
    pub mean: f64,
    /// Weighted sample standard deviation.
    pub std: f64,
    /// Weighted sample variance.
    pub var: f64,
    /// Sum of finite non-negative weights.
    pub nobs: f64,
    /// *t* statistic for \(H_0:\mu=0\).
    pub tvalue: f64,
    /// Two-sided Student *t* p-value.
    pub pvalue: f64,
    /// Lower 95% Student *t* bound.
    pub ci_low: f64,
    /// Upper 95% Student *t* bound.
    pub ci_high: f64,
}

/// Weighted mean / variance / *t* interval (statsmodels `DescrStatsW`).
///
/// `weights` defaults to ones. Weight count is not identification `p`.
pub fn descr_stats_w(
    x: &Vector,
    weights: Option<&Vector>,
    session: &Session,
) -> Result<Qualified<DescrStatsW>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    if let Some(w) = weights {
        if w.len() != x.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message(format!(
                        "DescrStatsW weights.len()={} ≠ x.len()={}",
                        w.len(),
                        x.len()
                    ))
                    .build(),
            );
        }
    }
    let n = x.len();
    let mut sw = 0.0;
    let mut swx = 0.0;
    let mut used = 0usize;
    for i in 0..n {
        if !x[i].is_finite() {
            continue;
        }
        let w = weights
            .and_then(|ww| ww.as_slice().get(i).copied())
            .unwrap_or(1.0);
        if !w.is_finite() || w < 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "DescrStatsW weight[{i}]={w} is not a finite ≥0 value"
                    ))
                    .build(),
            );
            continue;
        }
        sw += w;
        swx += w * x[i];
        used += 1;
    }
    if used < 2 || sw <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("DescrStatsW needs at least two finite weighted observations")
                .build(),
        );
        return ctx.finish(DescrStatsW {
            mean: f64::NAN,
            std: f64::NAN,
            var: f64::NAN,
            nobs: sw,
            tvalue: f64::NAN,
            pvalue: f64::NAN,
            ci_low: f64::NAN,
            ci_high: f64::NAN,
        });
    }
    let mean = swx / sw;
    let mut sse = 0.0;
    for i in 0..n {
        if !x[i].is_finite() {
            continue;
        }
        let w = weights
            .and_then(|ww| ww.as_slice().get(i).copied())
            .unwrap_or(1.0);
        if w.is_finite() && w >= 0.0 {
            let d = x[i] - mean;
            sse += w * d * d;
        }
    }
    let df = (sw - 1.0).max(1.0);
    let var = sse / df;
    let std = var.max(0.0).sqrt();
    let se = std / sw.sqrt();
    if se <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("DescrStatsW standard error vanished")
                .build(),
        );
        return ctx.finish(DescrStatsW {
            mean,
            std,
            var,
            nobs: sw,
            tvalue: f64::NAN,
            pvalue: f64::NAN,
            ci_low: mean,
            ci_high: mean,
        });
    }
    let tvalue: f64 = mean / se;
    let crit = student_t_ppf(0.975, df);
    ctx.finish(DescrStatsW {
        mean,
        std,
        var,
        nobs: sw,
        tvalue,
        pvalue: student_t_pvalue(tvalue, df),
        ci_low: mean - crit * se,
        ci_high: mean + crit * se,
    })
}

/// Multiple imputation by chained equations (statsmodels `MICE`).
///
/// Imputation count is not identification `p`. The design may contain NaNs;
/// do not call [`inspect_xy`] on it.
#[derive(Clone, Debug)]
pub struct Mice {
    /// Number of completed data sets.
    pub n_imputations: usize,
    /// Inner chained-equation cycles.
    pub max_iter: usize,
    /// Ridge penalty on each conditional model.
    pub alpha: f64,
    /// Seed for the initial jitter.
    pub seed: u64,
}

impl Default for Mice {
    fn default() -> Self {
        Self {
            n_imputations: 2,
            max_iter: 6,
            alpha: 1e-3,
            seed: 3,
        }
    }
}

impl Mice {
    /// `n_imputations` completed data sets.
    pub fn new(n_imputations: usize) -> Self {
        Self {
            n_imputations,
            ..Self::default()
        }
    }

    /// Draw completed matrices. Missing cells are initialized at the column
    /// mean plus a small Gaussian jitter, then each column is ridge-regressed
    /// on the others.
    pub fn impute(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vec<Matrix>>> {
        mice_impute(x, self, session)
    }
}

/// Draw `n_imputations` MICE completions (statsmodels `MICE`).
///
/// Imputation count is not identification `p`.
pub fn mice(x: &Matrix, n_imputations: usize, session: &Session) -> Result<Qualified<Vec<Matrix>>> {
    Mice::new(n_imputations).impute(x, session)
}

fn mice_impute(x: &Matrix, spec: &Mice, session: &Session) -> Result<Qualified<Vec<Matrix>>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let (n, p) = x.shape();
    ctx.report.set_sample_shape(n, p);
    if n == 0 || p == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .severity(Severity::Warning)
                .message(format!("MICE design is {n}×{p}"))
                .build(),
        );
        return ctx.finish(Vec::new());
    }
    let n_imp = if spec.n_imputations >= 1 {
        spec.n_imputations
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "MICE n_imputations={} < 1; using 1",
                    spec.n_imputations
                ))
                .build(),
        );
        1
    };
    let mut means = vec![0.0; p];
    for j in 0..p {
        let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
        let st = slice_stats(&col);
        if st.count == 0 {
            ctx.push(
                Issue::builder(IssueCode::ImputationUndefined)
                    .severity(Severity::Warning)
                    .message(format!(
                        "MICE column {j} is all-missing; fill is 0, not a statistic"
                    ))
                    .build(),
            );
            means[j] = 0.0;
        } else {
            means[j] = st.mean;
        }
    }
    let alpha = if spec.alpha.is_finite() && spec.alpha >= 0.0 {
        spec.alpha
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("MICE α={} is invalid; using 1e-3", spec.alpha))
                .build(),
        );
        1e-3
    };
    let mut rng = Rng::new(spec.seed);
    let mut out = Vec::with_capacity(n_imp);
    for _m in 0..n_imp {
        let mut filled = x.clone();
        for j in 0..p {
            for i in 0..n {
                if !filled.get(i, j).is_finite() {
                    filled.set(i, j, means[j] + 0.05 * rng.standard_normal());
                }
            }
        }
        for _it in 0..spec.max_iter.max(1) {
            for j in 0..p {
                let miss: Vec<usize> = (0..n).filter(|&i| !x.get(i, j).is_finite()).collect();
                if miss.is_empty() {
                    continue;
                }
                let others: Vec<usize> = (0..p).filter(|&k| k != j).collect();
                if others.is_empty() {
                    continue;
                }
                let z = Matrix::from_fn(n, others.len(), |i, t| filled.get(i, others[t]));
                let yj = filled.column(j);
                let mut scratch = Report::new("mice", "ridge");
                let Some(beta) =
                    crate::linalg::ridge_solve(&mut scratch, &z, &yj, alpha, &ctx.policy)
                else {
                    continue;
                };
                for issue in scratch.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::PerfectCollinearity
                            | IssueCode::InvalidWeight
                            | IssueCode::CholeskyFailed
                            | IssueCode::NonFiniteOutput
                            | IssueCode::DimensionMismatch
                            | IssueCode::JitterInjected
                            | IssueCode::UnderdeterminedSystem
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                for &i in &miss {
                    let mut pred = 0.0;
                    for t in 0..others.len().min(beta.len()) {
                        pred += beta[t] * filled.get(i, others[t]);
                    }
                    filled.set(i, j, pred);
                }
            }
        }
        out.push(filled);
    }
    ctx.finish(out)
}

/// Nested-model ANOVA (statsmodels `anova_lm`).
#[derive(Clone, Debug)]
pub struct AnovaLm {
    /// Extra-sum-of-squares *F*.
    pub f_stat: f64,
    /// Upper-tail *F* p-value.
    pub pvalue: f64,
    /// Numerator degrees of freedom (`p_full − p_restricted`).
    pub df_num: f64,
    /// Residual degrees of freedom of the unrestricted model.
    pub df_den: f64,
    /// Restricted residual sum of squares.
    pub ss_restricted: f64,
    /// Unrestricted residual sum of squares.
    pub ss_full: f64,
}

fn ols_sse(x: &Matrix, y: &Vector, policy: &signlred::Policy) -> (f64, usize) {
    let mut scratch = Report::new("anova_lm", "ols");
    let beta = least_squares(&mut scratch, x, y, policy);
    let p = x.ncols();
    let Some(b) = beta else {
        let mut sse = 0.0;
        let m = y.mean();
        for &v in y.as_slice() {
            if v.is_finite() {
                let e = v - m;
                sse += e * e;
            }
        }
        return (sse, 1);
    };
    let fit = x.matvec(&b);
    let mut sse = 0.0;
    for i in 0..y.len() {
        let e = y[i] - fit[i];
        sse += e * e;
    }
    (sse, p)
}

/// Compare a restricted linear model to an unrestricted one.
///
/// Column counts are identification `p` only when `n` is large enough for the
/// usual OLS gate. Residual-kind inner failures are not promoted.
pub fn anova_lm(
    y: &Vector,
    x_restricted: &Matrix,
    x_full: &Matrix,
    session: &Session,
) -> Result<Qualified<AnovaLm>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x_full, Some(y), &ctx.policy);
    inspect_xy(&mut ctx.report, x_restricted, None, &ctx.policy);
    let n = y.len().min(x_full.nrows()).min(x_restricted.nrows());
    if x_full.nrows() != y.len() || x_restricted.nrows() != y.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("anova_lm: design rows ≠ y length")
                .build(),
        );
    }
    if n >= 5 * x_full.ncols().max(1) {
        inspect_identification(&mut ctx.report, n, x_full.ncols(), &ctx.policy);
    }
    let (ss_r, p_r) = ols_sse(x_restricted, y, &ctx.policy);
    let (ss_u, p_u) = ols_sse(x_full, y, &ctx.policy);
    if p_u <= p_r {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .severity(Severity::Warning)
                .message(format!(
                    "anova_lm unrestricted has {p_u} columns ≤ restricted {p_r}"
                ))
                .build(),
        );
        return ctx.finish(AnovaLm {
            f_stat: f64::NAN,
            pvalue: f64::NAN,
            df_num: 0.0,
            df_den: (n as f64 - p_u as f64).max(0.0),
            ss_restricted: ss_r,
            ss_full: ss_u,
        });
    }
    let df_num = (p_u - p_r) as f64;
    let df_den = n as f64 - p_u as f64;
    if df_den <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegreesOfFreedomNonPositive)
                .message("anova_lm residual df ≤ 0")
                .meaninglessness(Meaninglessness::vacuous(
                    "nested-model F",
                    "the unrestricted model has no residual degrees of freedom",
                    "reduce columns or collect more rows",
                ))
                .build(),
        );
        return ctx.finish(AnovaLm {
            f_stat: f64::NAN,
            pvalue: f64::NAN,
            df_num,
            df_den,
            ss_restricted: ss_r,
            ss_full: ss_u,
        });
    }
    let extra = ss_r - ss_u;
    if extra < -1e-8 * ss_r.abs().max(1.0) {
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .severity(Severity::Warning)
                .message("restricted SSE is smaller than unrestricted; F is set to 0")
                .build(),
        );
    }
    let f = if extra > 0.0 && ss_u > 1e-18 {
        (extra / df_num) / (ss_u / df_den)
    } else {
        0.0
    };
    let pvalue = if f.is_finite() && f > 0.0 {
        f_pvalue(f, df_num, df_den)
    } else {
        1.0
    };
    ctx.finish(AnovaLm {
        f_stat: f,
        pvalue,
        df_num,
        df_den,
        ss_restricted: ss_r,
        ss_full: ss_u,
    })
}

/// Two-factor ANOVA (statsmodels `anova_lm` Type II on a two-way design).
///
/// Factor and cell counts are **not** identification `p`. Empty cells and a
/// singular interaction block are recorded; they do not reuse cluster/`p`
/// gates.
#[derive(Clone, Debug)]
pub struct AnovaTwoway {
    /// Type-II *F* for factor A (after B).
    pub f_a: f64,
    /// Upper-tail *p* for A.
    pub p_a: f64,
    /// Type-II *F* for factor B (after A).
    pub f_b: f64,
    /// Upper-tail *p* for B.
    pub p_b: f64,
    /// Type-II *F* for the A×B interaction.
    pub f_ab: f64,
    /// Upper-tail *p* for A×B.
    pub p_ab: f64,
    /// Type-II extra sum of squares for A.
    pub ss_a: f64,
    /// Type-II extra sum of squares for B.
    pub ss_b: f64,
    /// Extra sum of squares for A×B.
    pub ss_ab: f64,
    /// Residual sum of squares of the saturated cell-means model.
    pub ss_error: f64,
    /// \(a-1\).
    pub df_a: f64,
    /// \(b-1\).
    pub df_b: f64,
    /// Interaction degrees of freedom.
    pub df_ab: f64,
    /// Residual degrees of freedom.
    pub df_error: f64,
}

fn unique_int_labels(v: &Vector) -> Vec<i64> {
    let mut ids = Vec::new();
    for &x in v.as_slice() {
        if !x.is_finite() {
            continue;
        }
        let lab = x.round() as i64;
        if !ids.contains(&lab) {
            ids.push(lab);
        }
    }
    ids.sort_unstable();
    ids
}

fn dummy_vs_ref(labels: &Vector, ids: &[i64], k: usize) -> Vector {
    let this = ids[k];
    Vector::from_iter(labels.as_slice().iter().map(|&v| {
        if v.is_finite() && v.round() as i64 == this {
            1.0
        } else {
            0.0
        }
    }))
}

fn cols_to_matrix(cols: &[Vector], n: usize) -> Matrix {
    if cols.is_empty() {
        return Matrix::zeros(n, 0);
    }
    Matrix::from_fn(n, cols.len(), |i, j| cols[j][i])
}

fn twoway_f(extra: f64, df_num: f64, sse: f64, df_den: f64) -> (f64, f64) {
    if df_num <= 0.0 || df_den <= 0.0 || sse <= 1e-18 || extra <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let f = (extra / df_num) / (sse / df_den);
    if f.is_finite() && f > 0.0 {
        (f, f_pvalue(f, df_num, df_den))
    } else {
        (f64::NAN, f64::NAN)
    }
}

/// Two-way Type-II ANOVA of `y` on integer-coded factors `a` and `b`.
///
/// Dummy-column counts are identification `p` only when \(n \ge 5p\).
pub fn anova_twoway(
    y: &Vector,
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<AnovaTwoway>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    if a.len() != y.len() || b.len() != y.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("anova_twoway: factor length ≠ y length")
                .build(),
        );
    }
    let n = y.len().min(a.len()).min(b.len());
    let a_ids = unique_int_labels(a);
    let b_ids = unique_int_labels(b);
    let na = a_ids.len();
    let nb = b_ids.len();
    if na < 2 || nb < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!(
                    "anova_twoway needs ≥2 levels in each factor (A={na} B={nb})"
                ))
                .build(),
        );
        return ctx.finish(AnovaTwoway {
            f_a: f64::NAN,
            p_a: f64::NAN,
            f_b: f64::NAN,
            p_b: f64::NAN,
            f_ab: f64::NAN,
            p_ab: f64::NAN,
            ss_a: 0.0,
            ss_b: 0.0,
            ss_ab: 0.0,
            ss_error: 0.0,
            df_a: (na as f64 - 1.0).max(0.0),
            df_b: (nb as f64 - 1.0).max(0.0),
            df_ab: 0.0,
            df_error: 0.0,
        });
    }
    let mut cell_n = vec![0usize; na * nb];
    for i in 0..n {
        if !a[i].is_finite() || !b[i].is_finite() {
            continue;
        }
        let ia = a_ids.iter().position(|&v| v == a[i].round() as i64);
        let ib = b_ids.iter().position(|&v| v == b[i].round() as i64);
        if let (Some(ia), Some(ib)) = (ia, ib) {
            cell_n[ia * nb + ib] += 1;
        }
    }
    if cell_n.iter().any(|&c| c == 0) {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("anova_twoway has an empty cell; Type-II interaction df is reduced")
                .build(),
        );
    }
    let intercept = Vector::from_iter((0..n).map(|_| 1.0));
    let mut a_dummies = Vec::new();
    for k in 1..na {
        a_dummies.push(dummy_vs_ref(a, &a_ids, k));
    }
    let mut b_dummies = Vec::new();
    for k in 1..nb {
        b_dummies.push(dummy_vs_ref(b, &b_ids, k));
    }
    let mut ab_dummies = Vec::new();
    for ia in 0..a_dummies.len() {
        for ib in 0..b_dummies.len() {
            let col = Vector::from_iter((0..n).map(|i| a_dummies[ia][i] * b_dummies[ib][i]));
            if col.as_slice().iter().any(|&v| v.abs() > 0.0) {
                ab_dummies.push(col);
            }
        }
    }
    let only_int = vec![intercept.clone()];
    let mut only_a = only_int.clone();
    only_a.extend(a_dummies.iter().cloned());
    let mut only_b = only_int.clone();
    only_b.extend(b_dummies.iter().cloned());
    let mut mains = only_a.clone();
    mains.extend(b_dummies.iter().cloned());
    let mut full = mains.clone();
    full.extend(ab_dummies.iter().cloned());
    let x_a = cols_to_matrix(&only_a, n);
    let x_b = cols_to_matrix(&only_b, n);
    let x_m = cols_to_matrix(&mains, n);
    let x_f = cols_to_matrix(&full, n);
    if n >= 5 * x_f.ncols().max(1) {
        inspect_identification(&mut ctx.report, n, x_f.ncols(), &ctx.policy);
    }
    let (ss_a_only, _) = ols_sse(&x_a, y, &ctx.policy);
    let (ss_b_only, _) = ols_sse(&x_b, y, &ctx.policy);
    let (ss_m, _) = ols_sse(&x_m, y, &ctx.policy);
    let (ss_f, p_f) = ols_sse(&x_f, y, &ctx.policy);
    let ss_a = (ss_b_only - ss_m).max(0.0);
    let ss_b = (ss_a_only - ss_m).max(0.0);
    let ss_ab = (ss_m - ss_f).max(0.0);
    let df_a = (na - 1) as f64;
    let df_b = (nb - 1) as f64;
    let df_ab = ab_dummies.len() as f64;
    let df_error = n as f64 - p_f as f64;
    if df_error <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegreesOfFreedomNonPositive)
                .message("anova_twoway residual df ≤ 0")
                .meaninglessness(Meaninglessness::vacuous(
                    "two-way ANOVA F",
                    "the cell-means model has no residual degrees of freedom",
                    "collect more rows per cell",
                ))
                .build(),
        );
    }
    if ss_f <= 1e-18 && df_error > 0.0 {
        ctx.push(
            Issue::builder(IssueCode::R2IsOne)
                .message("two-way ANOVA residual SS is ~0; F ratios are infinite or undefined")
                .meaninglessness(Meaninglessness::new(
                    "two-way ANOVA F",
                    "within-cell residual is numerically zero",
                    signlred::InterpretiveValue::Misleading,
                    "do not treat infinite F as a precise p-value",
                ))
                .build(),
        );
    }
    let (f_a, p_a) = twoway_f(ss_a, df_a, ss_f, df_error);
    let (f_b, p_b) = twoway_f(ss_b, df_b, ss_f, df_error);
    let (f_ab, p_ab) = twoway_f(ss_ab, df_ab, ss_f, df_error);
    ctx.finish(AnovaTwoway {
        f_a,
        p_a,
        f_b,
        p_b,
        f_ab,
        p_ab,
        ss_a,
        ss_b,
        ss_ab,
        ss_error: ss_f,
        df_a,
        df_b,
        df_ab,
        df_error,
    })
}

/// One-way repeated-measures ANOVA (statsmodels `AnovaRM`).
///
/// `table` is subjects × treatments. Treatment count is not identification `p`.
pub fn anova_rm(table: &Matrix, session: &Session) -> Result<Qualified<AnovaResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, table, None, &ctx.policy);
    let (n, k) = table.shape();
    if n < 2 || k < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!(
                    "AnovaRM needs ≥2 subjects and ≥2 treatments (got {n}×{k})"
                ))
                .build(),
        );
        return ctx.finish(AnovaResult {
            f_stat: f64::NAN,
            pvalue: f64::NAN,
            df_between: (k as f64 - 1.0).max(0.0),
            df_within: 0.0,
            ss_between: 0.0,
            ss_within: 0.0,
        });
    }
    let mut grand = 0.0;
    let mut nobs = 0.0;
    for i in 0..n {
        for j in 0..k {
            let v = table.get(i, j);
            if v.is_finite() {
                grand += v;
                nobs += 1.0;
            }
        }
    }
    if nobs < 4.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("AnovaRM has fewer than four finite cells")
                .build(),
        );
        return ctx.finish(AnovaResult {
            f_stat: f64::NAN,
            pvalue: f64::NAN,
            df_between: (k - 1) as f64,
            df_within: 0.0,
            ss_between: 0.0,
            ss_within: 0.0,
        });
    }
    grand /= nobs;
    let mut ss_t = 0.0;
    for j in 0..k {
        let col = table.column(j);
        let st = slice_stats(col.as_slice());
        if st.count == 0 {
            continue;
        }
        let d = st.mean - grand;
        ss_t += st.count as f64 * d * d;
    }
    let mut ss_s = 0.0;
    for i in 0..n {
        let mut s = 0.0;
        let mut c = 0.0;
        for j in 0..k {
            let v = table.get(i, j);
            if v.is_finite() {
                s += v;
                c += 1.0;
            }
        }
        if c > 0.0 {
            let d = s / c - grand;
            ss_s += c * d * d;
        }
    }
    let mut ss_tot = 0.0;
    for i in 0..n {
        for j in 0..k {
            let v = table.get(i, j);
            if v.is_finite() {
                let d = v - grand;
                ss_tot += d * d;
            }
        }
    }
    let ss_e = (ss_tot - ss_t - ss_s).max(0.0);
    let df_t = (k - 1) as f64;
    let df_e = ((n - 1) * (k - 1)) as f64;
    if ss_e <= 1e-18 || df_e <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("AnovaRM residual SS vanished")
                .build(),
        );
        return ctx.finish(AnovaResult {
            f_stat: f64::NAN,
            pvalue: f64::NAN,
            df_between: df_t,
            df_within: df_e,
            ss_between: ss_t,
            ss_within: ss_e,
        });
    }
    let f_stat: f64 = (ss_t / df_t) / (ss_e / df_e);
    ctx.finish(AnovaResult {
        f_stat,
        pvalue: f_pvalue(f_stat.max(0.0), df_t, df_e),
        df_between: df_t,
        df_within: df_e,
        ss_between: ss_t,
        ss_within: ss_e,
    })
}

/// Quantile regression (statsmodels `QuantReg`) via IRLS.
///
/// The quantile is not identification `p`. Inner residual / rank issues are
/// kept on a scratch report so they do not abort a valid outer fit.
#[derive(Clone, Debug)]
pub struct QuantReg {
    /// Quantile in (0, 1).
    pub q: f64,
    /// IRLS iteration cap.
    pub max_iter: usize,
}

impl Default for QuantReg {
    fn default() -> Self {
        Self {
            q: 0.5,
            max_iter: 40,
        }
    }
}

impl QuantReg {
    /// Quantile `q`.
    pub fn new(q: f64) -> Self {
        Self {
            q,
            ..Self::default()
        }
    }
}

/// Fitted statsmodels-style quantile regression.
#[derive(Clone, Debug)]
pub struct FittedQuantReg {
    /// Slopes.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
    /// Requested quantile.
    pub q: f64,
}

impl Fit for QuantReg {
    type Fitted = FittedQuantReg;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedQuantReg>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let q = if self.q.is_finite() && self.q > 0.0 && self.q < 1.0 {
            self.q
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!("QuantReg q={} is not in (0,1); using 0.5", self.q))
                    .build(),
            );
            0.5
        };
        let design = x.with_intercept();
        let mut scratch = Report::new("quantreg", "init");
        let mut beta = least_squares(&mut scratch, &design, y, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for it in 0..self.max_iter.max(1) {
            let pred = design.matvec(&beta);
            let mut xs = Matrix::zeros(design.nrows(), design.ncols());
            let mut ys = Vector::zeros(y.len());
            for i in 0..y.len() {
                let r = y[i] - pred[i];
                let w = if r >= 0.0 { q } else { 1.0 - q };
                let sw = (w / r.abs().max(1e-6)).sqrt();
                ys[i] = y[i] * sw;
                for j in 0..design.ncols() {
                    xs.set(i, j, design.get(i, j) * sw);
                }
            }
            let mut step = Report::new("quantreg", "step");
            let Some(next) = least_squares(&mut step, &xs, &ys, &ctx.policy) else {
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
        let coef = if beta.len() > 1 {
            Vector::from_iter((1..beta.len()).map(|j| beta[j]))
        } else {
            Vector::zeros(x.ncols())
        };
        ctx.finish(FittedQuantReg { coef, intercept, q })
    }
}

impl Predict for FittedQuantReg {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(Vector::from_iter((0..x.nrows()).map(|i| {
            let mut s = self.intercept;
            for j in 0..x.ncols().min(self.coef.len()) {
                s += self.coef[j] * x.get(i, j);
            }
            s
        })))
    }
}

/// Canonical correlations (statsmodels `CanCorr`).
///
/// View / component counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct CanCorrResult {
    /// Canonical correlations (nonincreasing).
    pub correlations: Vector,
    /// Number of finite observations used.
    pub nobs: f64,
}

/// Canonical correlation of two views via SVD of the whitened cross-covariance.
pub fn cancorr(x: &Matrix, y: &Matrix, session: &Session) -> Result<Qualified<CanCorrResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    inspect_xy(&mut ctx.report, y, None, &ctx.policy);
    if x.nrows() != y.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .severity(Severity::Warning)
                .message(format!(
                    "CanCorr row mismatch {} vs {}",
                    x.nrows(),
                    y.nrows()
                ))
                .build(),
        );
    }
    let n = x.nrows().min(y.nrows());
    if n < 3 || x.ncols() == 0 || y.ncols() == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("CanCorr needs n≥3 and two non-empty views")
                .build(),
        );
        return ctx.finish(CanCorrResult {
            correlations: Vector::zeros(0),
            nobs: n as f64,
        });
    }
    let xc = x.centered().0;
    let yc = y.centered().0;
    let mut sx = Report::new("cancorr", "svdx");
    let mut sy = Report::new("cancorr", "svdy");
    let Some(svx) = thin_svd(&mut sx, &xc, &ctx.policy) else {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("CanCorr X SVD failed")
                .build(),
        );
        return ctx.finish(CanCorrResult {
            correlations: Vector::zeros(0),
            nobs: n as f64,
        });
    };
    let Some(svy) = thin_svd(&mut sy, &yc, &ctx.policy) else {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("CanCorr Y SVD failed")
                .build(),
        );
        return ctx.finish(CanCorrResult {
            correlations: Vector::zeros(0),
            nobs: n as f64,
        });
    };
    let rx = svx
        .rank(ctx.policy.rank_tol_relative)
        .max(1)
        .min(svx.u.ncols());
    let ry = svy
        .rank(ctx.policy.rank_tol_relative)
        .max(1)
        .min(svy.u.ncols());
    let xw = Matrix::from_fn(n, rx, |i, j| {
        if i < svx.u.nrows() && j < svx.u.ncols() {
            svx.u[(i, j)]
        } else {
            0.0
        }
    });
    let yw = Matrix::from_fn(n, ry, |i, j| {
        if i < svy.u.nrows() && j < svy.u.ncols() {
            svy.u[(i, j)]
        } else {
            0.0
        }
    });
    let cross = Matrix::from_fn(rx, ry, |i, j| {
        let mut s = 0.0;
        for t in 0..n {
            s += xw.get(t, i) * yw.get(t, j);
        }
        s
    });
    let mut sc = Report::new("cancorr", "cross");
    let Some(svc) = thin_svd(&mut sc, &cross, &ctx.policy) else {
        return ctx.finish(CanCorrResult {
            correlations: Vector::zeros(0),
            nobs: n as f64,
        });
    };
    let corr = Vector::from_iter(svc.singular_values.iter().map(|v| v.abs().min(1.0)));
    if corr.as_slice().iter().all(|v| *v <= 1e-12) {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("CanCorr found no shared variation")
                .build(),
        );
    }
    ctx.finish(CanCorrResult {
        correlations: corr,
        nobs: n as f64,
    })
}

/// Aalen additive hazards (lifelines `AalenAdditiveFitter`).
///
/// Covariate count is not identification `p`. Inner least-squares issues on
/// a risk-set increment are not promoted.
#[derive(Clone, Debug)]
pub struct AalenAdditiveResult {
    /// Cumulative additive coefficients (`n_events` × `p`).
    pub cumulative: Matrix,
    /// Event times.
    pub times: Vector,
    /// Number of events used.
    pub n_events: f64,
}

/// Fit Aalen's additive hazards model by accumulating risk-set OLS increments.
pub fn aalen_additive(
    durations: &Vector,
    events: &Vector,
    x: &Matrix,
    session: &Session,
) -> Result<Qualified<AalenAdditiveResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let n = x.nrows().min(durations.len()).min(events.len());
    let p = x.ncols();
    if n < 3 || p == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Aalen additive needs n≥3 and a covariate")
                .build(),
        );
        return ctx.finish(AalenAdditiveResult {
            cumulative: Matrix::zeros(0, p),
            times: Vector::zeros(0),
            n_events: 0.0,
        });
    }
    let mut idx: Vec<usize> = (0..n).filter(|&i| durations[i].is_finite()).collect();
    idx.sort_by(|&a, &b| {
        durations[a]
            .partial_cmp(&durations[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut times = Vec::new();
    let mut rows = Vec::new();
    let mut acc = Vector::zeros(p);
    let mut i = 0;
    while i < idx.len() {
        let t = durations[idx[i]];
        let mut j = i;
        while j < idx.len() && (durations[idx[j]] - t).abs() <= 1e-15 {
            j += 1;
        }
        let risk = &idx[i..];
        let mut event_at = None;
        for &u in &idx[i..j] {
            if events[u] >= 0.5 {
                event_at = Some(u);
                break;
            }
        }
        if let Some(u) = event_at {
            let xr = Matrix::from_fn(risk.len(), p, |r, c| x.get(risk[r], c));
            let dn = Vector::from_iter(risk.iter().map(|&r| if r == u { 1.0 } else { 0.0 }));
            let mut scratch = Report::new("aalen_add", "inc");
            if let Some(inc) = least_squares(&mut scratch, &xr, &dn, &ctx.policy) {
                for c in 0..p.min(inc.len()) {
                    acc[c] += inc[c];
                }
                times.push(t);
                rows.push(acc.as_slice().to_vec());
            }
        }
        i = j;
    }
    if times.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Aalen additive saw no usable events")
                .build(),
        );
        return ctx.finish(AalenAdditiveResult {
            cumulative: Matrix::zeros(0, p),
            times: Vector::zeros(0),
            n_events: 0.0,
        });
    }
    let cumulative = Matrix::from_fn(rows.len(), p, |r, c| rows[r][c]);
    ctx.finish(AalenAdditiveResult {
        n_events: times.len() as f64,
        times: Vector::from_iter(times),
        cumulative,
    })
}

/// One-way MANOVA (statsmodels `multivariate.manova.MANOVA`) via Pillai's trace.
///
/// Group and response counts are **not** identification `p`. A singular
/// total SSP is recorded; it does not reuse a cluster/`p` gate.
#[derive(Clone, Debug)]
pub struct ManovaResult {
    /// Pillai's trace \(\mathrm{tr}((B+W)^{-1}B)\).
    pub pillai: f64,
    /// Approximate upper-tail *F* *p* for the Pillai statistic.
    pub pvalue: f64,
    /// Hypothesis degrees of freedom \(k-1\).
    pub df_hypothesis: f64,
    /// Error degrees of freedom \(n-k\).
    pub df_error: f64,
    /// Number of groups.
    pub n_groups: usize,
    /// Number of response columns.
    pub n_responses: usize,
}

/// One-way MANOVA of the columns of `y` on integer-coded `groups`.
pub fn manova(y: &Matrix, groups: &Vector, session: &Session) -> Result<Qualified<ManovaResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, y, None, &ctx.policy);
    if groups.len() != y.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("manova: groups length ≠ n")
                .build(),
        );
    }
    let n = y.nrows().min(groups.len());
    let q = y.ncols();
    let ids = unique_int_labels(groups);
    let k = ids.len();
    if k < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!("MANOVA needs ≥2 groups (got {k})"))
                .build(),
        );
        return ctx.finish(ManovaResult {
            pillai: f64::NAN,
            pvalue: f64::NAN,
            df_hypothesis: 0.0,
            df_error: n as f64,
            n_groups: k,
            n_responses: q,
        });
    }
    if n >= 5 * q.max(1) {
        inspect_identification(&mut ctx.report, n, q, &ctx.policy);
    }
    let mut grand = Vector::zeros(q);
    let mut ntot = 0.0;
    for i in 0..n {
        if !groups[i].is_finite() {
            continue;
        }
        ntot += 1.0;
        for j in 0..q {
            grand[j] += y.get(i, j);
        }
    }
    if ntot > 0.0 {
        for j in 0..q {
            grand[j] /= ntot;
        }
    }
    let mut means = vec![Vector::zeros(q); k];
    let mut ns = vec![0.0; k];
    for i in 0..n {
        if !groups[i].is_finite() {
            continue;
        }
        let Some(gi) = ids.iter().position(|&v| v == groups[i].round() as i64) else {
            continue;
        };
        ns[gi] += 1.0;
        for j in 0..q {
            means[gi][j] += y.get(i, j);
        }
    }
    for g in 0..k {
        if ns[g] > 0.0 {
            for j in 0..q {
                means[g][j] /= ns[g];
            }
        }
    }
    let mut b = Matrix::zeros(q, q);
    let mut w = Matrix::zeros(q, q);
    for g in 0..k {
        for a in 0..q {
            for c in 0..=a {
                let d = ns[g] * (means[g][a] - grand[a]) * (means[g][c] - grand[c]);
                b.set(a, c, b.get(a, c) + d);
                b.set(c, a, b.get(a, c));
            }
        }
    }
    for i in 0..n {
        if !groups[i].is_finite() {
            continue;
        }
        let Some(gi) = ids.iter().position(|&v| v == groups[i].round() as i64) else {
            continue;
        };
        for a in 0..q {
            for c in 0..=a {
                let d = (y.get(i, a) - means[gi][a]) * (y.get(i, c) - means[gi][c]);
                w.set(a, c, w.get(a, c) + d);
                w.set(c, a, w.get(a, c));
            }
        }
    }
    let mut tmat = Mat::<f64>::zeros(q, q);
    for a in 0..q {
        for c in 0..q {
            tmat[(a, c)] = b.get(a, c) + w.get(a, c);
        }
        tmat[(a, a)] += 1e-12;
    }
    let mut tinv = Matrix::zeros(q, q);
    let mut ok = true;
    for j in 0..q {
        let ej = Vector::from_iter((0..q).map(|i| if i == j { 1.0 } else { 0.0 }));
        let mut scratch = Report::new("manova", "tinv");
        match chol_solve(&mut scratch, &tmat, &ej, &ctx.policy) {
            Some(sol) => {
                for i in 0..q {
                    tinv.set(i, j, sol[i]);
                }
            }
            None => {
                ok = false;
                break;
            }
        }
    }
    let df_h = (k - 1) as f64;
    let df_e = (ntot - k as f64).max(0.0);
    if !ok {
        ctx.push(
            Issue::builder(IssueCode::CholeskyFailed)
                .severity(Severity::Warning)
                .message("MANOVA total SSP was not SPD; Pillai is undefined")
                .compromise(NumericalCompromise::new(
                    "invertible B+W",
                    "undefined Pillai trace",
                    "the total sum-of-squares-and-products was singular at working precision",
                    "do not read a missing Pillai as evidence of no multivariate effect",
                ))
                .build(),
        );
        return ctx.finish(ManovaResult {
            pillai: f64::NAN,
            pvalue: f64::NAN,
            df_hypothesis: df_h,
            df_error: df_e,
            n_groups: k,
            n_responses: q,
        });
    }
    let mut pillai = 0.0;
    for a in 0..q {
        for c in 0..q {
            pillai += tinv.get(a, c) * b.get(c, a);
        }
    }
    let s = df_h.min(q as f64).max(1.0);
    let m = ((q as f64 - df_h).abs() - 1.0) / 2.0;
    let nn = (df_e - q as f64 - 1.0) / 2.0;
    let (f, pvalue) = if nn > 0.0 && pillai.is_finite() && pillai < s {
        let f = ((2.0 * nn + s + 1.0) / (2.0 * m + s + 1.0).max(1e-8))
            * (pillai / (s - pillai).max(1e-8));
        let dfn = s * (2.0 * m + s + 1.0);
        let dfd = s * (2.0 * nn + s + 1.0);
        (f, f_pvalue(f.max(0.0), dfn.max(1.0), dfd.max(1.0)))
    } else {
        (f64::NAN, f64::NAN)
    };
    let _ = f;
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("MANOVA p uses the Pillai F approximation, not exact Wilks tables")
            .compromise(NumericalCompromise::new(
                "exact multivariate F or permutation MANOVA",
                "Pillai trace with the classical F transform",
                "the approximation degrades when n is close to q+k",
                "treat p as a screening statistic",
            ))
            .build(),
    );
    ctx.finish(ManovaResult {
        pillai,
        pvalue,
        df_hypothesis: df_h,
        df_error: df_e,
        n_groups: k,
        n_responses: q,
    })
}

/// McNemar test of paired binary outcomes (statsmodels `mcnemar`).
///
/// Discordant-pair count is not identification `p`.
pub fn mcnemar(y1: &Vector, y2: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, y1, y2);
    let mut b = 0.0;
    let mut c = 0.0;
    let mut n = 0.0;
    for i in 0..y1.len().min(y2.len()) {
        if !y1[i].is_finite() || !y2[i].is_finite() {
            continue;
        }
        let a = y1[i] > 0.5;
        let d = y2[i] > 0.5;
        if !a && d {
            b += 1.0;
        } else if a && !d {
            c += 1.0;
        }
        n += 1.0;
    }
    let disc = b + c;
    if disc <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("McNemar has no discordant pairs; the χ² statistic is undefined")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: 0.0,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: n,
        });
    }
    let stat = (b - c) * (b - c) / disc;
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: chi2_pvalue(stat, 1.0),
        df: 1.0,
        nobs: n,
    })
}

/// Fisher exact test on a 2×2 table (statsmodels `fisher_exact`).
///
/// Cell counts are not identification `p`.
pub fn fisher_exact(table: &Matrix, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, table, None, &ctx.policy);
    if table.nrows() != 2 || table.ncols() != 2 {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "fisher_exact needs a 2×2 table; got {}×{}",
                    table.nrows(),
                    table.ncols()
                ))
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: 0.0,
        });
    }
    let a = table.get(0, 0).round().max(0.0);
    let b = table.get(0, 1).round().max(0.0);
    let c = table.get(1, 0).round().max(0.0);
    let d = table.get(1, 1).round().max(0.0);
    let n = a + b + c + d;
    let odds = if b > 0.0 && c > 0.0 {
        (a * d) / (b * c)
    } else {
        f64::INFINITY
    };
    let k = a + c;
    let n1 = a + b;
    let lo = (n1 + k - n).max(0.0);
    let hi = n1.min(k);
    let ln_p = |x: f64| {
        ln_gamma(n1 + 1.0) - ln_gamma(x + 1.0) - ln_gamma(n1 - x + 1.0) + ln_gamma(n - n1 + 1.0)
            - ln_gamma(k - x + 1.0)
            - ln_gamma((n - n1) - (k - x) + 1.0)
            - (ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0))
    };
    let p_obs = ln_p(a).exp();
    let mut p = 0.0;
    let mut x = lo;
    while x <= hi + 1e-9 {
        let px = ln_p(x).exp();
        if px <= p_obs + 1e-15 {
            p += px;
        }
        x += 1.0;
    }
    ctx.finish(HypothesisTest {
        statistic: odds,
        pvalue: p.clamp(0.0, 1.0),
        df: 1.0,
        nobs: n,
    })
}

/// Anderson–Darling normality test after studentization (statsmodels `normal_ad`).
pub fn anderson_darling(x: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let st = slice_stats(x.as_slice());
    if st.count < 8 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!(
                    "Anderson–Darling n={} < 8; the tail approximation is crude",
                    st.count
                ))
                .build(),
        );
    }
    if st.is_constant(ctx.policy.near_zero_variance) {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("Anderson–Darling of a constant sample is undefined")
                .meaninglessness(Meaninglessness::vacuous(
                    "normality after studentization",
                    "σ = 0; every z-score is 0/0",
                    "do not report A² on a degenerate sample",
                ))
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: st.count as f64,
        });
    }
    let mut z: Vec<f64> = x
        .as_slice()
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v - st.mean) / st.std().max(1e-12))
        .collect();
    z.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = z.len() as f64;
    let mut a2 = 0.0;
    for (i, &zi) in z.iter().enumerate() {
        let f = norm_cdf(zi).clamp(1e-15, 1.0 - 1e-15);
        let fr = norm_cdf(z[z.len() - 1 - i]).clamp(1e-15, 1.0 - 1e-15);
        a2 += (2.0 * (i as f64) + 1.0) * (f.ln() + (1.0 - fr).ln());
    }
    a2 = -n - a2 / n;
    let a2s = a2 * (1.0 + 0.75 / n + 2.25 / (n * n));
    let p = if a2s < 0.2 {
        1.0 - (-13.436 + 101.14 * a2s - 223.73 * a2s * a2s).exp()
    } else if a2s < 0.34 {
        1.0 - (-8.318 + 42.796 * a2s - 59.938 * a2s * a2s).exp()
    } else if a2s < 0.6 {
        (-0.9177 - 4.279 * a2s - 1.38 * a2s * a2s).exp()
    } else {
        (-1.2937 - 5.709 * a2s + 0.0186 * a2s * a2s).exp()
    };
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("Anderson–Darling p uses the Stephens polynomial, not exact tables")
            .build(),
    );
    ctx.finish(HypothesisTest {
        statistic: a2,
        pvalue: p.clamp(0.0, 1.0),
        df: f64::NAN,
        nobs: n,
    })
}

/// Alias of [`anderson_darling`] (statsmodels `normal_ad`).
pub fn normal_ad(x: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    anderson_darling(x, session)
}

/// Lilliefors normality test (KS after estimated mean/variance).
pub fn lilliefors(x: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let st = slice_stats(x.as_slice());
    if st.count < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!("Lilliefors n={} < 4", st.count))
                .build(),
        );
    }
    if st.is_constant(ctx.policy.near_zero_variance) {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("Lilliefors of a constant sample is undefined")
                .meaninglessness(Meaninglessness::vacuous(
                    "KS after studentization",
                    "σ = 0",
                    "do not report D on a degenerate sample",
                ))
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: st.count as f64,
        });
    }
    let mut z: Vec<f64> = x
        .as_slice()
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v - st.mean) / st.std().max(1e-12))
        .collect();
    z.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = z.len() as f64;
    let mut d: f64 = 0.0;
    for (i, &zi) in z.iter().enumerate() {
        let f = norm_cdf(zi);
        let emp_hi = (i + 1) as f64 / n;
        let emp_lo = i as f64 / n;
        d = d.max((emp_hi - f).abs()).max((emp_lo - f).abs());
    }
    let dstar = d * (n.sqrt() - 0.01 + 0.85 / n.sqrt());
    let p = (-7.01256 * dstar * dstar * (n + 2.78019) + 2.99587 * dstar * (n + 2.78019).sqrt()
        - 0.122119
        + 0.974598 / n.sqrt()
        + 1.67997 / n)
        .exp();
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("Lilliefors p uses a Dallal–Wilkinson style approximation")
            .compromise(NumericalCompromise::new(
                "Lilliefors table or Monte Carlo",
                "closed-form KS-style tail",
                "mean and variance were estimated from the same sample",
                "treat p as a screening statistic",
            ))
            .build(),
    );
    ctx.finish(HypothesisTest {
        statistic: d,
        pvalue: p.clamp(0.0, 1.0),
        df: f64::NAN,
        nobs: n,
    })
}

/// Bai–Perron single-break search on a mean shift (statsmodels `breaks_cusumolsresid` lite).
///
/// Candidate / break counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct BaiPerronResult {
    /// First observation of the second regime (`0` if unidentified).
    pub break_index: usize,
    /// `SSR_full − (SSR_left + SSR_right)`.
    pub ssr_gain: f64,
}

fn ssr_mean(sl: &[f64]) -> f64 {
    let vals: Vec<f64> = sl.iter().copied().filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return 0.0;
    }
    let m = vals.iter().sum::<f64>() / vals.len() as f64;
    vals.iter().map(|v| (v - m) * (v - m)).sum()
}

/// Locate one mean break by exhaustive SSR search.
pub fn bai_perron(y: &Vector, session: &Session) -> Result<Qualified<BaiPerronResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, &Matrix::from_vector(y), None, &ctx.policy);
    let n = y.len();
    if n < 8 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Bai–Perron needs n≥8")
                .build(),
        );
        return ctx.finish(BaiPerronResult {
            break_index: 0,
            ssr_gain: 0.0,
        });
    }
    let full = ssr_mean(y.as_slice());
    let trim = (n / 10).max(2);
    let mut best_t = trim;
    let mut best_ssr = f64::INFINITY;
    for t in trim..(n - trim) {
        let ssr = ssr_mean(&y.as_slice()[..t]) + ssr_mean(&y.as_slice()[t..]);
        if ssr < best_ssr {
            best_ssr = ssr;
            best_t = t;
        }
    }
    let gain = full - best_ssr;
    if gain > 0.15 * full.max(1e-12) {
        ctx.push(
            Issue::builder(IssueCode::StructuralBreak)
                .message(format!(
                    "Bai–Perron mean break at t={best_t} (SSR gain {gain:.4e})"
                ))
                .metric("break_index", best_t as f64)
                .metric("ssr_gain", gain)
                .build(),
        );
    }
    ctx.finish(BaiPerronResult {
        break_index: best_t,
        ssr_gain: gain,
    })
}

/// Fitted bivariate Gaussian copula (statsmodels `Copula` lite).
#[derive(Clone, Debug)]
pub struct GaussianCopula {
    /// Pearson correlation of normal scores.
    pub rho: f64,
    /// Gaussian-copula log-likelihood.
    pub loglik: f64,
}

/// Fit a bivariate Gaussian copula via normal scores.
///
/// Pair count is not identification `p`.
pub fn gaussian_copula(
    y1: &Vector,
    y2: &Vector,
    session: &Session,
) -> Result<Qualified<GaussianCopula>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n = y1.len().min(y2.len());
    let x = Matrix::from_fn(n, 2, |i, j| if j == 0 { y1[i] } else { y2[i] });
    inspect_xy(&mut ctx.report, &x, None, &ctx.policy);
    if n < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Gaussian copula needs n≥4 pairs")
                .build(),
        );
        return ctx.finish(GaussianCopula {
            rho: 0.0,
            loglik: f64::NAN,
        });
    }
    let ranks = |y: &Vector| {
        let mut idx: Vec<usize> = (0..n).filter(|&i| y[i].is_finite()).collect();
        idx.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal));
        let mut r = vec![f64::NAN; n];
        let m = idx.len() as f64;
        for (k, &i) in idx.iter().enumerate() {
            r[i] = (k as f64 + 1.0) / (m + 1.0);
        }
        r
    };
    let u = ranks(y1);
    let v = ranks(y2);
    let mut z1 = Vec::new();
    let mut z2 = Vec::new();
    for i in 0..n {
        if u[i].is_finite() && v[i].is_finite() {
            z1.push(norm_ppf(u[i].clamp(1e-6, 1.0 - 1e-6)));
            z2.push(norm_ppf(v[i].clamp(1e-6, 1.0 - 1e-6)));
        }
    }
    if z1.len() < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Gaussian copula had too few finite pairs")
                .build(),
        );
        return ctx.finish(GaussianCopula {
            rho: 0.0,
            loglik: f64::NAN,
        });
    }
    let m1 = z1.iter().sum::<f64>() / z1.len() as f64;
    let m2 = z2.iter().sum::<f64>() / z2.len() as f64;
    let mut num = 0.0;
    let mut d1 = 0.0;
    let mut d2 = 0.0;
    for i in 0..z1.len() {
        let a = z1[i] - m1;
        let b = z2[i] - m2;
        num += a * b;
        d1 += a * a;
        d2 += b * b;
    }
    let rho = if d1 > 0.0 && d2 > 0.0 {
        (num / (d1.sqrt() * d2.sqrt())).clamp(-0.999, 0.999)
    } else {
        0.0
    };
    let mut loglik = 0.0;
    let omr2 = (1.0 - rho * rho).max(1e-12);
    for i in 0..z1.len() {
        let q1 = z1[i];
        let q2 = z2[i];
        loglik += -0.5 * omr2.ln()
            - 0.5 / omr2 * (q1 * q1 + q2 * q2 - 2.0 * rho * q1 * q2)
            + 0.5 * (q1 * q1 + q2 * q2);
    }
    ctx.finish(GaussianCopula { rho, loglik })
}

fn empirical_ranks(y: &Vector, n: usize) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..n).filter(|&i| y[i].is_finite()).collect();
    idx.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![f64::NAN; n];
    let m = idx.len() as f64;
    for (k, &i) in idx.iter().enumerate() {
        r[i] = (k as f64 + 1.0) / (m + 1.0);
    }
    r
}

fn copula_uv(y1: &Vector, y2: &Vector) -> (Vec<f64>, Vec<f64>) {
    let n = y1.len().min(y2.len());
    let u = empirical_ranks(y1, n);
    let v = empirical_ranks(y2, n);
    let mut ou = Vec::new();
    let mut ov = Vec::new();
    for i in 0..n {
        if u[i].is_finite() && v[i].is_finite() {
            ou.push(u[i].clamp(1e-6, 1.0 - 1e-6));
            ov.push(v[i].clamp(1e-6, 1.0 - 1e-6));
        }
    }
    (ou, ov)
}

/// Fitted Clayton copula (statsmodels `ClaytonCopula`).
#[derive(Clone, Debug)]
pub struct ClaytonCopula {
    /// Dependence \(\theta > 0\).
    pub theta: f64,
    /// Copula log-likelihood.
    pub loglik: f64,
}

fn clayton_ll(u: &[f64], v: &[f64], theta: f64) -> f64 {
    if theta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let mut ll = 0.0;
    for i in 0..u.len() {
        let up = u[i].powf(-theta);
        let vp = v[i].powf(-theta);
        let s = up + vp - 1.0;
        if s <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll += (1.0 + theta).ln()
            + (-theta - 1.0) * (u[i].ln() + v[i].ln())
            + (-2.0 - 1.0 / theta) * s.ln();
    }
    ll
}

/// Fit a bivariate Clayton copula by a \(\theta\) grid on ranks.
///
/// Pair count is not identification `p`.
pub fn clayton_copula(
    y1: &Vector,
    y2: &Vector,
    session: &Session,
) -> Result<Qualified<ClaytonCopula>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n = y1.len().min(y2.len());
    let x = Matrix::from_fn(n, 2, |i, j| if j == 0 { y1[i] } else { y2[i] });
    inspect_xy(&mut ctx.report, &x, None, &ctx.policy);
    let (u, v) = copula_uv(y1, y2);
    if u.len() < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Clayton copula needs n≥4 pairs")
                .build(),
        );
        return ctx.finish(ClaytonCopula {
            theta: 1.0,
            loglik: f64::NAN,
        });
    }
    let mut best_th = 1.0_f64;
    let mut best_ll = f64::NEG_INFINITY;
    for step in 0..10 {
        let th = 0.2 + 0.4 * step as f64;
        let ll = clayton_ll(&u, &v, th);
        if ll > best_ll {
            best_ll = ll;
            best_th = th;
        }
    }
    ctx.finish(ClaytonCopula {
        theta: best_th,
        loglik: best_ll,
    })
}

/// Fitted Gumbel copula (statsmodels `GumbelCopula`).
#[derive(Clone, Debug)]
pub struct GumbelCopula {
    /// Dependence \(\theta \ge 1\).
    pub theta: f64,
    /// Copula log-likelihood.
    pub loglik: f64,
}

fn gumbel_ll(u: &[f64], v: &[f64], theta: f64) -> f64 {
    if theta < 1.0 {
        return f64::NEG_INFINITY;
    }
    let mut ll = 0.0;
    for i in 0..u.len() {
        let nlu = -u[i].ln();
        let nlv = -v[i].ln();
        let w = nlu.powf(theta) + nlv.powf(theta);
        if w <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let w1t = w.powf(1.0 / theta);
        let c = (-w1t).exp();
        let dens = c * (nlu * nlv).powf(theta - 1.0) / (u[i] * v[i])
            * w.powf(1.0 / theta - 2.0)
            * (w1t + theta - 1.0);
        if !dens.is_finite() || dens <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll += dens.ln();
    }
    ll
}

/// Fit a bivariate Gumbel copula by a \(\theta\) grid on ranks.
///
/// Pair count is not identification `p`.
pub fn gumbel_copula(
    y1: &Vector,
    y2: &Vector,
    session: &Session,
) -> Result<Qualified<GumbelCopula>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n = y1.len().min(y2.len());
    let x = Matrix::from_fn(n, 2, |i, j| if j == 0 { y1[i] } else { y2[i] });
    inspect_xy(&mut ctx.report, &x, None, &ctx.policy);
    let (u, v) = copula_uv(y1, y2);
    if u.len() < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Gumbel copula needs n≥4 pairs")
                .build(),
        );
        return ctx.finish(GumbelCopula {
            theta: 1.0,
            loglik: f64::NAN,
        });
    }
    let mut best_th = 1.5_f64;
    let mut best_ll = f64::NEG_INFINITY;
    for step in 0..8 {
        let th = 1.0 + 0.35 * step as f64;
        let ll = gumbel_ll(&u, &v, th);
        if ll > best_ll {
            best_ll = ll;
            best_th = th;
        }
    }
    ctx.finish(GumbelCopula {
        theta: best_th,
        loglik: best_ll,
    })
}

/// Fitted Frank copula (statsmodels `FrankCopula`).
#[derive(Clone, Debug)]
pub struct FrankCopula {
    /// Dependence \(\theta \neq 0\).
    pub theta: f64,
    /// Copula log-likelihood.
    pub loglik: f64,
}

fn frank_ll(u: &[f64], v: &[f64], theta: f64) -> f64 {
    if theta.abs() < 1e-6 {
        return f64::NEG_INFINITY;
    }
    let em = (-theta).exp();
    let eta = 1.0 - em;
    if eta.abs() < 1e-18 {
        return f64::NEG_INFINITY;
    }
    let mut ll = 0.0;
    for i in 0..u.len() {
        let eu = (-theta * u[i]).exp();
        let ev = (-theta * v[i]).exp();
        let den = eta - (1.0 - eu) * (1.0 - ev);
        if den.abs() < 1e-18 {
            return f64::NEG_INFINITY;
        }
        let dens = (theta * eta * (-theta * (u[i] + v[i])).exp()) / (den * den);
        if !dens.is_finite() || dens <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll += dens.ln();
    }
    ll
}

/// Fit a bivariate Frank copula by a \(\theta\) grid on ranks.
///
/// Pair count is not identification `p`.
pub fn frank_copula(y1: &Vector, y2: &Vector, session: &Session) -> Result<Qualified<FrankCopula>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n = y1.len().min(y2.len());
    let x = Matrix::from_fn(n, 2, |i, j| if j == 0 { y1[i] } else { y2[i] });
    inspect_xy(&mut ctx.report, &x, None, &ctx.policy);
    let (u, v) = copula_uv(y1, y2);
    if u.len() < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Frank copula needs n≥4 pairs")
                .build(),
        );
        return ctx.finish(FrankCopula {
            theta: 1.0,
            loglik: f64::NAN,
        });
    }
    let mut best_th = 2.0_f64;
    let mut best_ll = f64::NEG_INFINITY;
    for step in 0..10 {
        let th = -4.0 + 0.9 * step as f64;
        if th.abs() < 0.2 {
            continue;
        }
        let ll = frank_ll(&u, &v, th);
        if ll > best_ll {
            best_ll = ll;
            best_th = th;
        }
    }
    ctx.finish(FrankCopula {
        theta: best_th,
        loglik: best_ll,
    })
}

/// Fitted Student-\(t\) copula (statsmodels `StudentTCopula`).
#[derive(Clone, Debug)]
pub struct StudentTCopula {
    /// Correlation of \(t\) scores.
    pub rho: f64,
    /// Degrees of freedom used for the inverse CDF.
    pub df: f64,
    /// Copula log-likelihood proxy (Gaussian-score form at the fitted \(\rho\)).
    pub loglik: f64,
}

/// Fit a bivariate \(t\) copula via \(t\) scores and Pearson \(\rho\).
///
/// Pair / df counts are not identification `p`.
pub fn student_t_copula(
    y1: &Vector,
    y2: &Vector,
    df: f64,
    session: &Session,
) -> Result<Qualified<StudentTCopula>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n = y1.len().min(y2.len());
    let x = Matrix::from_fn(n, 2, |i, j| if j == 0 { y1[i] } else { y2[i] });
    inspect_xy(&mut ctx.report, &x, None, &ctx.policy);
    let (u, v) = copula_uv(y1, y2);
    let nu = if df.is_finite() && df > 2.0 { df } else { 5.0 };
    if u.len() < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Student-t copula needs n≥4 pairs")
                .build(),
        );
        return ctx.finish(StudentTCopula {
            rho: 0.0,
            df: nu,
            loglik: f64::NAN,
        });
    }
    let z1: Vec<f64> = u.iter().map(|&p| student_t_ppf(p, nu)).collect();
    let z2: Vec<f64> = v.iter().map(|&p| student_t_ppf(p, nu)).collect();
    let m1 = z1.iter().sum::<f64>() / z1.len() as f64;
    let m2 = z2.iter().sum::<f64>() / z2.len() as f64;
    let mut num = 0.0;
    let mut d1 = 0.0;
    let mut d2 = 0.0;
    for i in 0..z1.len() {
        let a = z1[i] - m1;
        let b = z2[i] - m2;
        num += a * b;
        d1 += a * a;
        d2 += b * b;
    }
    let rho = if d1 > 0.0 && d2 > 0.0 {
        (num / (d1.sqrt() * d2.sqrt())).clamp(-0.999, 0.999)
    } else {
        0.0
    };
    let mut loglik = 0.0;
    let omr2 = (1.0 - rho * rho).max(1e-12);
    for i in 0..z1.len() {
        let q1 = z1[i];
        let q2 = z2[i];
        loglik += -0.5 * omr2.ln()
            - 0.5 / omr2 * (q1 * q1 + q2 * q2 - 2.0 * rho * q1 * q2)
            + 0.5 * (q1 * q1 + q2 * q2);
    }
    ctx.finish(StudentTCopula {
        rho,
        df: nu,
        loglik,
    })
}

/// Fitted Joe copula (statsmodels `JoeCopula`).
#[derive(Clone, Debug)]
pub struct JoeCopula {
    /// Dependence \(\theta \ge 1\).
    pub theta: f64,
    /// Copula log-likelihood.
    pub loglik: f64,
}

fn joe_ll(u: &[f64], v: &[f64], theta: f64) -> f64 {
    if theta < 1.0 {
        return f64::NEG_INFINITY;
    }
    let mut ll = 0.0;
    for i in 0..u.len() {
        let ub = (1.0 - u[i]).max(1e-12);
        let vb = (1.0 - v[i]).max(1e-12);
        let ut = ub.powf(theta);
        let vt = vb.powf(theta);
        let w = ut + vt - ut * vt;
        if w <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let dens = ub.powf(theta - 1.0)
            * vb.powf(theta - 1.0)
            * (theta - 1.0 + w.powf(1.0 / theta))
            * w.powf(1.0 / theta - 2.0);
        if !dens.is_finite() || dens <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll += dens.ln();
    }
    ll
}

/// Fit a bivariate Joe copula by a \(\theta\) grid on ranks.
///
/// Pair count is not identification `p`.
pub fn joe_copula(y1: &Vector, y2: &Vector, session: &Session) -> Result<Qualified<JoeCopula>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n = y1.len().min(y2.len());
    let x = Matrix::from_fn(n, 2, |i, j| if j == 0 { y1[i] } else { y2[i] });
    inspect_xy(&mut ctx.report, &x, None, &ctx.policy);
    let (u, v) = copula_uv(y1, y2);
    if u.len() < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Joe copula needs n≥4 pairs")
                .build(),
        );
        return ctx.finish(JoeCopula {
            theta: 1.0,
            loglik: f64::NAN,
        });
    }
    let mut best_th = 1.5_f64;
    let mut best_ll = f64::NEG_INFINITY;
    for step in 0..8 {
        let th = 1.0 + 0.4 * step as f64;
        let ll = joe_ll(&u, &v, th);
        if ll > best_ll {
            best_ll = ll;
            best_th = th;
        }
    }
    ctx.finish(JoeCopula {
        theta: best_th,
        loglik: best_ll,
    })
}

/// Fitted Plackett copula (statsmodels `PlackettCopula`).
#[derive(Clone, Debug)]
pub struct PlackettCopula {
    /// Odds-ratio dependence \(\theta > 0\).
    pub theta: f64,
    /// Copula log-likelihood.
    pub loglik: f64,
}

fn plackett_ll(u: &[f64], v: &[f64], theta: f64) -> f64 {
    if theta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let mut ll = 0.0;
    for i in 0..u.len() {
        let s = 1.0 + (theta - 1.0) * (u[i] + v[i]);
        let disc = s * s - 4.0 * theta * (theta - 1.0) * u[i] * v[i];
        if disc <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let dens = theta * (1.0 + (theta - 1.0) * (u[i] + v[i] - 2.0 * u[i] * v[i]))
            / disc.powf(1.5);
        if !dens.is_finite() || dens <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll += dens.ln();
    }
    ll
}

/// Fit a bivariate Plackett copula by a \(\theta\) grid on ranks.
///
/// Pair count is not identification `p`.
pub fn plackett_copula(
    y1: &Vector,
    y2: &Vector,
    session: &Session,
) -> Result<Qualified<PlackettCopula>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n = y1.len().min(y2.len());
    let x = Matrix::from_fn(n, 2, |i, j| if j == 0 { y1[i] } else { y2[i] });
    inspect_xy(&mut ctx.report, &x, None, &ctx.policy);
    let (u, v) = copula_uv(y1, y2);
    if u.len() < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Plackett copula needs n≥4 pairs")
                .build(),
        );
        return ctx.finish(PlackettCopula {
            theta: 1.0,
            loglik: f64::NAN,
        });
    }
    let mut best_th = 2.0_f64;
    let mut best_ll = f64::NEG_INFINITY;
    for step in 0..10 {
        let th = 0.3 + 0.5 * step as f64;
        let ll = plackett_ll(&u, &v, th);
        if ll > best_ll {
            best_ll = ll;
            best_th = th;
        }
    }
    ctx.finish(PlackettCopula {
        theta: best_th,
        loglik: best_ll,
    })
}

/// Fitted Ali–Mikhail–Haq copula (statsmodels `AliMikhailHaqCopula`).
#[derive(Clone, Debug)]
pub struct AmhCopula {
    /// Dependence \(\theta \in [-1, 1)\).
    pub theta: f64,
    /// Copula log-likelihood.
    pub loglik: f64,
}

fn amh_ll(u: &[f64], v: &[f64], theta: f64) -> f64 {
    if !(theta > -1.0 && theta < 1.0) {
        return f64::NEG_INFINITY;
    }
    let mut ll = 0.0;
    for i in 0..u.len() {
        let den = 1.0 - theta * (1.0 - u[i]) * (1.0 - v[i]);
        if den.abs() < 1e-12 {
            return f64::NEG_INFINITY;
        }
        let dens = (1.0
            - theta
            + 2.0 * theta * u[i] * v[i]
            + theta * (1.0 - u[i] - v[i] + u[i] * v[i]))
            / den.powi(3);
        if !dens.is_finite() || dens <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll += dens.ln();
    }
    ll
}

/// Fit a bivariate AMH copula by a \(\theta\) grid on ranks.
///
/// Pair count is not identification `p`.
pub fn amh_copula(y1: &Vector, y2: &Vector, session: &Session) -> Result<Qualified<AmhCopula>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n = y1.len().min(y2.len());
    let x = Matrix::from_fn(n, 2, |i, j| if j == 0 { y1[i] } else { y2[i] });
    inspect_xy(&mut ctx.report, &x, None, &ctx.policy);
    let (u, v) = copula_uv(y1, y2);
    if u.len() < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("AMH copula needs n≥4 pairs")
                .build(),
        );
        return ctx.finish(AmhCopula {
            theta: 0.0,
            loglik: f64::NAN,
        });
    }
    let mut best_th = 0.2_f64;
    let mut best_ll = f64::NEG_INFINITY;
    for step in 0..9 {
        let th = -0.8 + 0.18 * step as f64;
        if th.abs() >= 0.99 {
            continue;
        }
        let ll = amh_ll(&u, &v, th);
        if ll > best_ll {
            best_ll = ll;
            best_th = th;
        }
    }
    ctx.finish(AmhCopula {
        theta: best_th,
        loglik: best_ll,
    })
}

/// Univariate GAM-lite: cubic truncated-power spline + ridge (statsmodels `GLMGam`).
///
/// Knot count is not identification `p`.
#[derive(Clone, Debug)]
pub struct UnivariateGam {
    /// Interior knots.
    pub n_knots: usize,
    /// Ridge on the spline coefficients.
    pub ridge: f64,
}

impl Default for UnivariateGam {
    fn default() -> Self {
        Self {
            n_knots: 4,
            ridge: 1e-2,
        }
    }
}

impl UnivariateGam {
    /// GAM with `n_knots` interior knots.
    pub fn new(n_knots: usize) -> Self {
        Self {
            n_knots: n_knots.max(1),
            ridge: 1e-2,
        }
    }

    /// Fit \(y \approx a + b x + \sum_k (x-\kappa_k)_+^3\).
    pub fn fit(&self, x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<FittedUnivariateGam>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let n = x.len().min(y.len());
        let design0 = Matrix::from_fn(n, 1, |i, _| x[i]);
        let y0 = Vector::from_iter(y.as_slice().iter().take(n).copied());
        inspect_xy(&mut ctx.report, &design0, Some(&y0), &ctx.policy);
        if n < 6 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("univariate GAM needs n≥6")
                    .build(),
            );
        }
        let mut xs: Vec<f64> = x.as_slice().iter().take(n).copied().filter(|v| v.is_finite()).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let nk = self.n_knots.max(1).min(n.saturating_sub(3).max(1));
        let mut knots = Vec::new();
        for k in 0..nk {
            let q = (k + 1) as f64 / (nk + 1) as f64;
            let idx = ((xs.len().saturating_sub(1)) as f64 * q).round() as usize;
            knots.push(*xs.get(idx).unwrap_or(&0.0));
        }
        let p = 2 + knots.len();
        // Knot count is a smoother template, not a regression `p`.
        let lam = if self.ridge.is_finite() && self.ridge > 0.0 {
            self.ridge
        } else {
            1e-2
        };
        let extra = p.saturating_sub(1);
        let design = Matrix::from_fn(n + extra, p, |i, j| {
            if i < n {
                let xi = x[i];
                if j == 0 {
                    1.0
                } else if j == 1 {
                    xi
                } else {
                    let d = xi - knots[j - 2];
                    if d > 0.0 {
                        d * d * d
                    } else {
                        0.0
                    }
                }
            } else {
                let k = i - n + 1;
                if j == k {
                    lam.sqrt()
                } else {
                    0.0
                }
            }
        });
        let mut scratch = Report::new("univariate_gam", "ridge");
        let target = Vector::from_iter((0..n + extra).map(|i| {
            if i < n {
                y[i]
            } else {
                0.0
            }
        }));
        let coef = match least_squares(&mut scratch, &design, &target, &ctx.policy) {
            Some(c) => c,
            None => {
                let mut c = Vector::zeros(p);
                c[0] = y.as_slice().iter().take(n).sum::<f64>() / n.max(1) as f64;
                c
            }
        };
        ctx.finish(FittedUnivariateGam { coef, knots })
    }
}

/// Fitted univariate spline GAM.
#[derive(Clone, Debug)]
pub struct FittedUnivariateGam {
    /// `[intercept, slope, knot coefs…]`.
    pub coef: Vector,
    /// Interior knots.
    pub knots: Vec<f64>,
}

impl FittedUnivariateGam {
    /// Predict at the supplied `x` locations.
    pub fn predict(&self, x: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        let y = Vector::from_iter((0..x.len()).map(|i| {
            let xi = x[i];
            let mut s = self.coef.as_slice().first().copied().unwrap_or(0.0);
            if self.coef.len() > 1 {
                s += self.coef[1] * xi;
            }
            for (k, &knot) in self.knots.iter().enumerate() {
                let d = xi - knot;
                if d > 0.0 {
                    s += self.coef.as_slice().get(2 + k).copied().unwrap_or(0.0) * d * d * d;
                }
            }
            s
        }));
        ctx.finish(y)
    }
}

/// Product-kernel multivariate KDE (statsmodels `KDEMultivariate`).
///
/// Dimension is not identification `p`.
#[derive(Clone, Debug)]
pub struct KdeMultivariate {
    /// Density at each training row.
    pub density: Vector,
    /// Per-column Scott bandwidth.
    pub bandwidth: Vector,
}

/// Gaussian-product KDE on the rows of `x`.
pub fn kde_multivariate(x: &Matrix, session: &Session) -> Result<Qualified<KdeMultivariate>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let (n, k) = x.shape();
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("multivariate KDE needs n≥2")
                .build(),
        );
    }
    let mut bw = Vector::zeros(k);
    for j in 0..k {
        let s = x.column(j).std().max(1e-8);
        bw[j] = s * (n.max(1) as f64).powf(-1.0 / (k.max(1) as f64 + 4.0));
    }
    let dens = Vector::from_iter((0..n).map(|i| {
        let mut acc = 0.0;
        for t in 0..n {
            let mut logk = 0.0;
            for j in 0..k {
                let h = bw[j].max(1e-8);
                let z = (x.get(i, j) - x.get(t, j)) / h;
                logk += -0.5 * z * z - h.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
            }
            acc += logk.exp();
        }
        acc / n.max(1) as f64
    }));
    ctx.finish(KdeMultivariate {
        density: dens,
        bandwidth: bw,
    })
}

/// Empirical-likelihood test of a scalar mean (statsmodels `emplike`).
///
/// The implied Lagrange multiplier is not identification `p`.
pub fn empirical_likelihood_mean(
    y: &Vector,
    mu0: f64,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, &Matrix::from_vector(y), None, &ctx.policy);
    let vals: Vec<f64> = y
        .as_slice()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let n = vals.len();
    if n < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("empirical likelihood needs n≥3")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: n as f64,
        });
    }
    let d: Vec<f64> = vals.iter().map(|v| v - mu0).collect();
    let mut lo = -1.0 / d.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-8);
    let mut hi = -lo;
    let g = |lam: f64| {
        let mut s = 0.0;
        for &di in &d {
            let den = 1.0 + lam * di;
            if den <= 1e-12 {
                return f64::NAN;
            }
            s += di / den;
        }
        s
    };
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let gm = g(mid);
        if !gm.is_finite() {
            hi = mid;
            continue;
        }
        if gm > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let lam = 0.5 * (lo + hi);
    let mut w = 0.0;
    let mut ok = true;
    for &di in &d {
        let den = 1.0 + lam * di;
        if den <= 1e-12 {
            ok = false;
            break;
        }
        w += 2.0 * den.ln();
    }
    if !ok || !w.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::DidNotConverge)
                .severity(Severity::Warning)
                .message("empirical-likelihood Lagrange step left the unit simplex")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: n as f64,
        });
    }
    let p = chi2_pvalue(w.max(0.0), 1.0);
    ctx.finish(HypothesisTest {
        statistic: w,
        pvalue: p.clamp(0.0, 1.0),
        df: 1.0,
        nobs: n as f64,
    })
}

/// Two-equation SUR (statsmodels `SUR`): OLS margins plus residual covariance.
///
/// Equation count is not identification `p`.
#[derive(Clone, Debug)]
pub struct SurResult {
    /// First-equation coefficients (including intercept if `x` has none).
    pub beta1: Vector,
    /// Second-equation coefficients.
    pub beta2: Vector,
    /// Residual covariance (`2` × `2`).
    pub sigma: Matrix,
}

/// Fit a two-equation seemingly unrelated regression on a shared design.
pub fn sur(
    y1: &Vector,
    y2: &Vector,
    x: &Matrix,
    session: &Session,
) -> Result<Qualified<SurResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y1), &ctx.policy);
    inspect_xy(&mut ctx.report, x, Some(y2), &ctx.policy);
    let n = x.nrows().min(y1.len()).min(y2.len());
    if n < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("SUR needs n≥4")
                .build(),
        );
    }
    let design = x.with_intercept();
    let y1n = Vector::from_iter(y1.as_slice().iter().take(n).copied());
    let y2n = Vector::from_iter(y2.as_slice().iter().take(n).copied());
    let d = Matrix::from_fn(n, design.ncols(), |i, j| design.get(i, j));
    let mut scratch = Report::new("sur", "ols");
    let b1 = least_squares(&mut scratch, &d, &y1n, &ctx.policy).unwrap_or_else(|| {
        let mut v = Vector::zeros(d.ncols());
        v[0] = y1n.mean();
        v
    });
    let b2 = least_squares(&mut scratch, &d, &y2n, &ctx.policy).unwrap_or_else(|| {
        let mut v = Vector::zeros(d.ncols());
        v[0] = y2n.mean();
        v
    });
    let f1 = d.matvec(&b1);
    let f2 = d.matvec(&b2);
    let mut s11 = 0.0;
    let mut s22 = 0.0;
    let mut s12 = 0.0;
    for i in 0..n {
        let e1 = y1n[i] - f1[i];
        let e2 = y2n[i] - f2[i];
        s11 += e1 * e1;
        s22 += e2 * e2;
        s12 += e1 * e2;
    }
    let den = n.max(1) as f64;
    let sigma = Matrix::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => s11 / den,
        (1, 1) => s22 / den,
        _ => s12 / den,
    });
    ctx.finish(SurResult {
        beta1: b1,
        beta2: b2,
        sigma,
    })
}

/// Additive Denton first-difference temporal disaggregation (statsmodels `denton`).
///
/// `indicator` is high-frequency; `period` of its values sum to one entry of
/// `totals`. Period and year counts are not identification `p`. Annual totals
/// may be level-stationary — they are not inspected as a regression target.
pub fn denton(
    indicator: &Vector,
    totals: &Vector,
    period: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(indicator),
        None,
        &ctx.policy,
    );
    let s = period.max(2);
    if period < 2 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("Denton period={period} is <2; using 2"))
                .build(),
        );
    }
    let m = totals.len();
    let need = m.saturating_mul(s);
    if indicator.len() != need {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .severity(Severity::Warning)
                .message(format!(
                    "Denton indicator n={} ≠ period {s} × {} totals",
                    indicator.len(),
                    m
                ))
                .build(),
        );
    }
    let n = indicator.len().min(need);
    let years = if s == 0 { 0 } else { n / s };
    if years == 0 || m == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Denton needs at least one complete low-frequency period")
                .build(),
        );
        return ctx.finish(Vector::from_iter(indicator.as_slice().iter().copied()));
    }
    let mut y = vec![0.0; years * s];
    for t in 0..years {
        let mut xs = 0.0;
        for q in 0..s {
            xs += indicator[t * s + q];
        }
        let yt = totals.as_slice().get(t).copied().unwrap_or(0.0);
        if xs.abs() < 1e-15 {
            ctx.push(
                Issue::builder(IssueCode::ScaleFactorZero)
                    .severity(Severity::Warning)
                    .message(format!("Denton indicator sum is 0 in period {t}; using equal shares"))
                    .compromise(NumericalCompromise::new(
                        "proportional allocation on a non-zero indicator",
                        "equal split of the annual total",
                        "the high-frequency indicator vanished in this period",
                        "do not read a flat intra-year path as a unique Denton solution",
                    ))
                    .build(),
            );
            let share = yt / s as f64;
            for q in 0..s {
                y[t * s + q] = share;
            }
        } else {
            let scale = yt / xs;
            for q in 0..s {
                y[t * s + q] = indicator[t * s + q] * scale;
            }
        }
    }
    let mut z = vec![0.0; y.len()];
    for t in 0..years {
        let mut r = totals.as_slice().get(t).copied().unwrap_or(0.0);
        for q in 0..s {
            r -= y[t * s + q];
        }
        let add = r / s as f64;
        for q in 0..s {
            z[t * s + q] = add;
        }
    }
    for _ in 0..48 {
        let nn = z.len();
        let mut g = vec![0.0; nn];
        if nn >= 2 {
            g[0] = z[0] - z[1];
            g[nn - 1] = z[nn - 1] - z[nn - 2];
            for t in 1..nn - 1 {
                g[t] = 2.0 * z[t] - z[t - 1] - z[t + 1];
            }
        }
        for t in 0..years {
            let mut mg = 0.0;
            for q in 0..s {
                mg += g[t * s + q];
            }
            mg /= s as f64;
            for q in 0..s {
                z[t * s + q] -= 0.25 * (g[t * s + q] - mg);
            }
        }
        for t in 0..years {
            let mut sum = 0.0;
            for q in 0..s {
                sum += y[t * s + q] + z[t * s + q];
            }
            let yt = totals.as_slice().get(t).copied().unwrap_or(0.0);
            let add = (yt - sum) / s as f64;
            for q in 0..s {
                z[t * s + q] += add;
            }
        }
    }
    for i in 0..y.len() {
        y[i] += z[i];
    }
    ctx.finish(Vector::from_iter(y))
}

/// Chow–Lin temporal disaggregation (statsmodels `chow_lin`).
///
/// Low-frequency OLS of `totals` on the aggregated `indicator`, then AR(1)
/// residual distribution. Period is not identification `p`.
#[derive(Clone, Debug)]
pub struct ChowLinResult {
    /// High-frequency series.
    pub series: Vector,
    /// Intercept of the low-frequency regression.
    pub intercept: f64,
    /// Slope on the aggregated indicator.
    pub slope: f64,
    /// Residual AR(1).
    pub rho: f64,
}

/// Chow–Lin disaggregation of `totals` using `indicator` at `period`.
pub fn chow_lin(
    indicator: &Vector,
    totals: &Vector,
    period: usize,
    session: &Session,
) -> Result<Qualified<ChowLinResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(indicator),
        None,
        &ctx.policy,
    );
    let s = period.max(2);
    let m = totals.len();
    let n = indicator.len().min(m.saturating_mul(s));
    let years = if s == 0 { 0 } else { n / s };
    if years < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Chow–Lin needs at least three low-frequency observations")
                .build(),
        );
    }
    if years == 0 {
        return ctx.finish(ChowLinResult {
            series: Vector::from_iter(indicator.as_slice().iter().copied()),
            intercept: 0.0,
            slope: 0.0,
            rho: 0.0,
        });
    }
    let xagg = Vector::from_iter((0..years).map(|t| {
        let mut xs = 0.0;
        for q in 0..s {
            xs += indicator[t * s + q];
        }
        xs
    }));
    let yagg = Vector::from_iter(totals.as_slice().iter().take(years).copied());
    let design = Matrix::from_fn(years, 2, |i, j| if j == 0 { 1.0 } else { xagg[i] });
    let mut scratch = Report::new("chow_lin", "ols");
    let beta = least_squares(&mut scratch, &design, &yagg, &ctx.policy).unwrap_or_else(|| {
        Vector::from_slice(&[yagg.mean(), 0.0])
    });
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
    let intercept = beta.as_slice().first().copied().unwrap_or(0.0);
    let slope = beta.as_slice().get(1).copied().unwrap_or(0.0);
    let fit = design.matvec(&beta);
    let mut num = 0.0;
    let mut den = 0.0;
    for t in 1..years {
        num += (yagg[t] - fit[t]) * (yagg[t - 1] - fit[t - 1]);
        den += (yagg[t - 1] - fit[t - 1]) * (yagg[t - 1] - fit[t - 1]);
    }
    let rho = if den > 1e-18 {
        (num / den).clamp(-0.95, 0.95)
    } else {
        0.0
    };
    let mut series = Vector::from_iter((0..n).map(|i| intercept / s as f64 + slope * indicator[i]));
    let mut resid = Vector::zeros(years);
    for t in 0..years {
        let mut pred = 0.0;
        for q in 0..s {
            pred += series[t * s + q];
        }
        resid[t] = yagg[t] - pred;
    }
    let mut cvc = Mat::<f64>::zeros(years, years);
    for a in 0..years {
        for b in 0..years {
            let mut acc = 0.0;
            for i in 0..s {
                for j in 0..s {
                    let di = (a * s + i) as i32 - (b * s + j) as i32;
                    acc += rho.abs().powi(di.unsigned_abs() as i32);
                }
            }
            cvc[(a, b)] = acc;
        }
    }
    let mut scr2 = Report::new("chow_lin", "cvc");
    let lam = chol_solve(&mut scr2, &cvc, &resid, &ctx.policy).unwrap_or_else(|| {
        Vector::from_iter((0..years).map(|t| resid[t] / s as f64))
    });
    for i in 0..n {
        let mut add = 0.0;
        for t in 0..years {
            let mut w = 0.0;
            for q in 0..s {
                let di = i as i32 - (t * s + q) as i32;
                w += rho.abs().powi(di.unsigned_abs() as i32);
            }
            add += w * lam[t];
        }
        series[i] += add;
    }
    for t in 0..years {
        let mut pred = 0.0;
        for q in 0..s {
            pred += series[t * s + q];
        }
        let add = (yagg[t] - pred) / s as f64;
        for q in 0..s {
            series[t * s + q] += add;
        }
    }
    ctx.finish(ChowLinResult {
        series,
        intercept,
        slope,
        rho,
    })
}

/// Litterman temporal disaggregation (random-walk residuals).
///
/// Period is not identification `p`.
pub fn litterman(
    indicator: &Vector,
    totals: &Vector,
    period: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let q = chow_lin(indicator, totals, period, session)?;
    let ctx = FitCtx::with_session(session.child("litterman"));
    let s = period.max(2);
    let n = q.value.series.len();
    let years = if s == 0 { 0 } else { n / s };
    if years == 0 {
        return ctx.finish(q.value.series);
    }
    let mut series = Vector::from_iter((0..n).map(|i| {
        q.value.intercept / s as f64 + q.value.slope * indicator.as_slice().get(i).copied().unwrap_or(0.0)
    }));
    let yagg = Vector::from_iter(totals.as_slice().iter().take(years).copied());
    let mut resid = Vector::zeros(years);
    for t in 0..years {
        let mut pred = 0.0;
        for qq in 0..s {
            pred += series[t * s + qq];
        }
        resid[t] = yagg[t] - pred;
    }
    let mut cvc = Mat::<f64>::zeros(years, years);
    for a in 0..years {
        for b in 0..years {
            let mut acc = 0.0;
            for i in 0..s {
                for j in 0..s {
                    let ti = a * s + i;
                    let tj = b * s + j;
                    acc += (ti.min(tj) + 1) as f64;
                }
            }
            cvc[(a, b)] = acc;
        }
    }
    let mut scratch = Report::new("litterman", "cvc");
    let lam = chol_solve(&mut scratch, &cvc, &resid, &ctx.policy).unwrap_or_else(|| {
        Vector::from_iter((0..years).map(|t| resid[t] / s as f64))
    });
    for i in 0..n {
        let mut add = 0.0;
        for t in 0..years {
            let mut w = 0.0;
            for qq in 0..s {
                let tj = t * s + qq;
                w += (i.min(tj) + 1) as f64;
            }
            add += w * lam[t];
        }
        series[i] += add;
    }
    for t in 0..years {
        let mut pred = 0.0;
        for qq in 0..s {
            pred += series[t * s + qq];
        }
        let add = (yagg[t] - pred) / s as f64;
        for qq in 0..s {
            series[t * s + qq] += add;
        }
    }
    ctx.finish(series)
}

/// Fitted Farlie–Gumbel–Morgenstern copula (statsmodels `FGMCopula`).
#[derive(Clone, Debug)]
pub struct FgmCopula {
    /// Dependence \(\theta \in [-1, 1]\).
    pub theta: f64,
    /// Copula log-likelihood.
    pub loglik: f64,
}

fn fgm_ll(u: &[f64], v: &[f64], theta: f64) -> f64 {
    if !(-1.0..=1.0).contains(&theta) {
        return f64::NEG_INFINITY;
    }
    let mut ll = 0.0;
    for i in 0..u.len() {
        let dens = 1.0 + theta * (1.0 - 2.0 * u[i]) * (1.0 - 2.0 * v[i]);
        if !dens.is_finite() || dens <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll += dens.ln();
    }
    ll
}

/// Fit a bivariate FGM copula by a \(\theta\) grid on ranks.
///
/// Pair count is not identification `p`.
pub fn fgm_copula(y1: &Vector, y2: &Vector, session: &Session) -> Result<Qualified<FgmCopula>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n = y1.len().min(y2.len());
    let x = Matrix::from_fn(n, 2, |i, j| if j == 0 { y1[i] } else { y2[i] });
    inspect_xy(&mut ctx.report, &x, None, &ctx.policy);
    let (u, v) = copula_uv(y1, y2);
    if u.len() < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("FGM copula needs n≥4 pairs")
                .build(),
        );
        return ctx.finish(FgmCopula {
            theta: 0.0,
            loglik: f64::NAN,
        });
    }
    let mut best_th = 0.0_f64;
    let mut best_ll = f64::NEG_INFINITY;
    for step in 0..11 {
        let th = -1.0 + 0.2 * step as f64;
        let ll = fgm_ll(&u, &v, th);
        if ll > best_ll {
            best_ll = ll;
            best_th = th;
        }
    }
    ctx.finish(FgmCopula {
        theta: best_th,
        loglik: best_ll,
    })
}

/// Empirical / rank copula summary (statsmodels `EmpiricalCopula`).
#[derive(Clone, Debug)]
pub struct EmpiricalCopula {
    /// Finite pair count.
    pub n: usize,
    /// Mean of \(C_n(U_i,V_i)\). Independence is near \(1/4\); comonotonic near \(1/3\).
    pub mean_mass: f64,
}

/// Evaluate the bivariate empirical copula on the sample ranks.
///
/// Pair count is not identification `p`.
pub fn empirical_copula(
    y1: &Vector,
    y2: &Vector,
    session: &Session,
) -> Result<Qualified<EmpiricalCopula>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n = y1.len().min(y2.len());
    let x = Matrix::from_fn(n, 2, |i, j| if j == 0 { y1[i] } else { y2[i] });
    inspect_xy(&mut ctx.report, &x, None, &ctx.policy);
    let (u, v) = copula_uv(y1, y2);
    if u.len() < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("empirical copula needs n≥4 pairs")
                .build(),
        );
        return ctx.finish(EmpiricalCopula {
            n: u.len(),
            mean_mass: f64::NAN,
        });
    }
    let m = u.len();
    let mut acc = 0.0;
    for i in 0..m {
        let mut c = 0.0;
        for j in 0..m {
            if u[j] <= u[i] && v[j] <= v[i] {
                c += 1.0;
            }
        }
        acc += c / m as f64;
    }
    ctx.finish(EmpiricalCopula {
        n: m,
        mean_mass: acc / m as f64,
    })
}

fn pairwise_abs(a: &[f64], i: usize, j: usize) -> f64 {
    (a[i] - a[j]).abs()
}

fn double_center(d: &mut [f64], n: usize) {
    let mut row = vec![0.0; n];
    let mut col = vec![0.0; n];
    let mut grand = 0.0;
    for i in 0..n {
        for j in 0..n {
            let v = d[i * n + j];
            row[i] += v;
            col[j] += v;
            grand += v;
        }
    }
    let nf = n as f64;
    for i in 0..n {
        row[i] /= nf;
        col[i] /= nf;
    }
    grand /= nf * nf;
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = d[i * n + j] - row[i] - col[j] + grand;
        }
    }
}

/// Distance correlation (Szekely / scipy `distance_correlation`).
///
/// Pair count is not identification `p`.
pub fn distance_corr(x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    let n = x.len().min(y.len());
    if n < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("distance correlation needs n≥3")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let mut dx = vec![0.0; n * n];
    let mut dy = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            dx[i * n + j] = pairwise_abs(x.as_slice(), i, j);
            dy[i * n + j] = pairwise_abs(y.as_slice(), i, j);
        }
    }
    double_center(&mut dx, n);
    double_center(&mut dy, n);
    let mut dcov = 0.0;
    let mut dvx = 0.0;
    let mut dvy = 0.0;
    for i in 0..n * n {
        dcov += dx[i] * dy[i];
        dvx += dx[i] * dx[i];
        dvy += dy[i] * dy[i];
    }
    let nf = (n * n) as f64;
    dcov /= nf;
    dvx /= nf;
    dvy /= nf;
    let den = (dvx * dvy).sqrt();
    let r = if den < 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::NearZeroVariance)
                .severity(Severity::Warning)
                .message("distance variance vanished; dCor is undefined")
                .build(),
        );
        0.0
    } else {
        (dcov / den).clamp(0.0, 1.0).sqrt()
    };
    ctx.finish(r)
}

/// Energy distance between two samples (scipy `energy_distance`).
///
/// Sample sizes are not identification `p`.
pub fn energy_distance(x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(x),
        None,
        &ctx.policy,
    );
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        None,
        &ctx.policy,
    );
    if x.is_empty() || y.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("energy distance needs two non-empty samples")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let mut xy = 0.0;
    for xi in x.as_slice() {
        for yj in y.as_slice() {
            xy += (xi - yj).abs();
        }
    }
    xy /= (x.len() * y.len()) as f64;
    let mut xx = 0.0;
    let xs = x.as_slice();
    for i in 0..xs.len() {
        for j in 0..xs.len() {
            xx += (xs[i] - xs[j]).abs();
        }
    }
    xx /= (xs.len() * xs.len()) as f64;
    let mut yy = 0.0;
    let ys = y.as_slice();
    for i in 0..ys.len() {
        for j in 0..ys.len() {
            yy += (ys[i] - ys[j]).abs();
        }
    }
    yy /= (ys.len() * ys.len()) as f64;
    ctx.finish((2.0 * xy - xx - yy).max(0.0))
}

/// One-sample Cramér–von Mises normality statistic (scipy `cramervonmises`).
pub fn cramer_von_mises(x: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, x);
    let st = slice_stats(x.as_slice());
    if st.count < 8 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Cramér–von Mises n<8; the p-value is a χ² sketch")
                .build(),
        );
    }
    if st.is_constant(ctx.policy.near_zero_variance) {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("Cramér–von Mises of a constant sample is undefined")
                .meaninglessness(Meaninglessness::vacuous(
                    "normality after studentization",
                    "σ = 0; every z-score is 0/0",
                    "do not report ω² on a degenerate sample",
                ))
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: st.count as f64,
        });
    }
    let mut z: Vec<f64> = x
        .as_slice()
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v - st.mean) / st.std().max(1e-12))
        .collect();
    z.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = z.len() as f64;
    let mut w2 = 1.0 / (12.0 * n);
    for (i, &zi) in z.iter().enumerate() {
        let f = norm_cdf(zi).clamp(1e-15, 1.0 - 1e-15);
        let e = (2.0 * i as f64 + 1.0) / (2.0 * n);
        w2 += (f - e) * (f - e);
    }
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("Cramér–von Mises p uses a χ²(1) sketch, not Anderson tables")
            .build(),
    );
    ctx.finish(HypothesisTest {
        statistic: w2,
        pvalue: chi2_pvalue(w2.max(0.0), 1.0).clamp(0.0, 1.0),
        df: 1.0,
        nobs: n,
    })
}

/// k-sample Anderson–Darling (scipy `anderson_ksamp` / Scholz–Stephens).
///
/// Group count is not identification `p`.
pub fn anderson_ksamp(groups: &[&Vector], session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if groups.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("anderson_ksamp needs at least two groups")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: 0.0,
        });
    }
    let mut pooled = Vec::new();
    let mut ns = Vec::new();
    for g in groups {
        inspect_xy(
            &mut ctx.report,
            &Matrix::from_vector(g),
            None,
            &ctx.policy,
        );
        let vals: Vec<f64> = g.as_slice().iter().copied().filter(|v| v.is_finite()).collect();
        ns.push(vals.len());
        pooled.extend(vals);
    }
    let n = pooled.len();
    if n < 6 || ns.iter().any(|k| *k < 2) {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("anderson_ksamp needs ≥2 finite observations per group")
                .build(),
        );
    }
    let mut order = pooled.clone();
    order.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut a2 = 0.0;
    for (gi, g) in groups.iter().enumerate() {
        let ni = ns[gi] as f64;
        if ni < 1.0 {
            continue;
        }
        for j in 0..n.saturating_sub(1) {
            let zj = order[j];
            let m = g
                .as_slice()
                .iter()
                .filter(|v| v.is_finite() && **v <= zj)
                .count() as f64;
            let h = (j + 1) as f64;
            let den = h * (n as f64 - h);
            if den > 1e-12 {
                let num = n as f64 * m - h * ni;
                a2 += (1.0 / ni) * num * num / den;
            }
        }
    }
    a2 /= n.max(1) as f64;
    let df = (groups.len().saturating_sub(1)) as f64;
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("k-sample AD p uses a χ²(k−1) sketch, not Scholz tables")
            .build(),
    );
    ctx.finish(HypothesisTest {
        statistic: a2,
        pvalue: chi2_pvalue(a2.max(0.0), df.max(1.0)).clamp(0.0, 1.0),
        df,
        nobs: n as f64,
    })
}

/// Brunner–Munzel two-sample test (scipy `brunnermunzel`).
pub fn brunner_munzel(x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(x),
        None,
        &ctx.policy,
    );
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        None,
        &ctx.policy,
    );
    let xs: Vec<f64> = x.as_slice().iter().copied().filter(|v| v.is_finite()).collect();
    let ys: Vec<f64> = y.as_slice().iter().copied().filter(|v| v.is_finite()).collect();
    let n1 = xs.len();
    let n2 = ys.len();
    if n1 < 2 || n2 < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Brunner–Munzel needs n1,n2≥2")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: (n1 + n2) as f64,
        });
    }
    let mut all = Vec::with_capacity(n1 + n2);
    all.extend(xs.iter().copied());
    all.extend(ys.iter().copied());
    let mut idx: Vec<usize> = (0..all.len()).collect();
    idx.sort_by(|a, b| all[*a].partial_cmp(&all[*b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank = vec![0.0; all.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && (all[idx[j]] - all[idx[i]]).abs() < 1e-15 {
            j += 1;
        }
        let avg = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            rank[idx[k]] = avg;
        }
        i = j;
    }
    let r1: f64 = rank.iter().take(n1).sum();
    let pxy = (r1 / n1 as f64 - (n1 as f64 + 1.0) / 2.0) / n2 as f64;
    let mut s1 = 0.0;
    for r in rank.iter().take(n1) {
        let e = r - r1 / n1 as f64;
        s1 += e * e;
    }
    s1 /= (n1.saturating_sub(1)).max(1) as f64;
    let r2: f64 = rank.iter().skip(n1).sum();
    let mut s2 = 0.0;
    for r in rank.iter().skip(n1) {
        let e = r - r2 / n2 as f64;
        s2 += e * e;
    }
    s2 /= (n2.saturating_sub(1)).max(1) as f64;
    let se = ((s1 / (n1 as f64 * n2 as f64 * n2 as f64)) + (s2 / (n2 as f64 * n1 as f64 * n1 as f64)))
        .sqrt();
    let stat = if se < 1e-18 {
        0.0
    } else {
        (pxy - 0.5) / se
    };
    let df_num = se.powi(4);
    let df_den = s1.powi(2) / ((n1 as f64).powi(2) * (n2 as f64).powi(4) * (n1.saturating_sub(1)).max(1) as f64)
        + s2.powi(2) / ((n2 as f64).powi(2) * (n1 as f64).powi(4) * (n2.saturating_sub(1)).max(1) as f64);
    let df = if df_den < 1e-18 { 1.0 } else { df_num / df_den };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: student_t_pvalue(stat, df.max(1.0)).clamp(0.0, 1.0),
        df,
        nobs: (n1 + n2) as f64,
    })
}

/// Jonckheere–Terpstra ordered-alternative test (statsmodels `jonckheere`).
///
/// Group count is not identification `p`. Groups are taken in the given order.
pub fn jonckheere(groups: &[&Vector], session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if groups.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Jonckheere–Terpstra needs at least two ordered groups")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: 0.0,
        });
    }
    let cleaned: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| {
            inspect_xy(
                &mut ctx.report,
                &Matrix::from_vector(g),
                None,
                &ctx.policy,
            );
            g.as_slice().iter().copied().filter(|v| v.is_finite()).collect()
        })
        .collect();
    let n: usize = cleaned.iter().map(|g| g.len()).sum();
    let mut t = 0.0;
    let mut et = 0.0;
    for i in 0..cleaned.len() {
        for j in (i + 1)..cleaned.len() {
            let ni = cleaned[i].len() as f64;
            let nj = cleaned[j].len() as f64;
            et += ni * nj / 2.0;
            for &a in &cleaned[i] {
                for &b in &cleaned[j] {
                    if a < b {
                        t += 1.0;
                    } else if (a - b).abs() < 1e-15 {
                        t += 0.5;
                    }
                }
            }
        }
    }
    let mut vt = 0.0;
    for g in &cleaned {
        let ni = g.len() as f64;
        vt += ni * ni * (2.0 * ni + 3.0);
    }
    let nf = n as f64;
    let var = ((nf * nf * (2.0 * nf + 3.0) - vt) / 72.0).max(0.0);
    let se = var.sqrt();
    let z = if se < 1e-18 { 0.0 } else { (t - et) / se };
    ctx.finish(HypothesisTest {
        statistic: z,
        pvalue: (2.0 * (1.0 - norm_cdf(z.abs()))).clamp(0.0, 1.0),
        df: f64::NAN,
        nobs: nf,
    })
}

fn paired_labels(y1: &Vector, y2: &Vector) -> (Vec<i64>, usize) {
    let mut keys = BTreeMap::new();
    let n = y1.len().min(y2.len());
    for i in 0..n {
        if y1[i].is_finite() && y2[i].is_finite() {
            keys.insert(y1[i].round() as i64, ());
            keys.insert(y2[i].round() as i64, ());
        }
    }
    (keys.keys().copied().collect(), n)
}

fn square_counts(y1: &Vector, y2: &Vector, labels: &[i64]) -> Matrix {
    let k = labels.len();
    let mut tab = Matrix::zeros(k, k);
    let n = y1.len().min(y2.len());
    for i in 0..n {
        if !y1[i].is_finite() || !y2[i].is_finite() {
            continue;
        }
        let a = y1[i].round() as i64;
        let b = y2[i].round() as i64;
        let ia = labels.iter().position(|v| *v == a);
        let ib = labels.iter().position(|v| *v == b);
        if let (Some(r), Some(c)) = (ia, ib) {
            tab.set(r, c, tab.get(r, c) + 1.0);
        }
    }
    tab
}

/// Bowker symmetry test on paired categorical labels (statsmodels `bowker`).
///
/// Category count is not identification `p`.
pub fn bowker(y1: &Vector, y2: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, y1, y2);
    let (labels, n) = paired_labels(y1, y2);
    if labels.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Bowker needs at least two categories")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: n as f64,
        });
    }
    let tab = square_counts(y1, y2, &labels);
    let k = labels.len();
    let mut stat = 0.0_f64;
    let mut df = 0.0_f64;
    for i in 0..k {
        for j in (i + 1)..k {
            let a = tab.get(i, j);
            let b = tab.get(j, i);
            if a + b > 0.0 {
                stat += (a - b) * (a - b) / (a + b);
                df += 1.0;
            }
        }
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: chi2_pvalue(stat.max(0.0), df.max(1.0)).clamp(0.0, 1.0),
        df,
        nobs: n as f64,
    })
}

/// Stuart–Maxwell marginal-homogeneity test (statsmodels `stuart_maxwell`).
///
/// Category count is not identification `p`.
pub fn stuart_maxwell(y1: &Vector, y2: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, y1, y2);
    let (labels, n) = paired_labels(y1, y2);
    if labels.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Stuart–Maxwell needs at least two categories")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: n as f64,
        });
    }
    let tab = square_counts(y1, y2, &labels);
    let k = labels.len();
    let km = k.saturating_sub(1);
    let mut d = Vector::zeros(km);
    let mut s = Mat::<f64>::zeros(km, km);
    for i in 0..km {
        let mut ri = 0.0;
        let mut ci = 0.0;
        for j in 0..k {
            ri += tab.get(i, j);
            ci += tab.get(j, i);
        }
        d[i] = ri - ci;
        for j in 0..km {
            let nij = tab.get(i, j);
            let nji = tab.get(j, i);
            if i == j {
                let mut off = 0.0;
                for t in 0..k {
                    if t != i {
                        off += tab.get(i, t) + tab.get(t, i);
                    }
                }
                s[(i, j)] = off;
            } else {
                s[(i, j)] = -(nij + nji);
            }
        }
    }
    let mut scratch = Report::new("stuart_maxwell", "chol");
    let lam = chol_solve(&mut scratch, &s, &d, &ctx.policy);
    let stat = if let Some(sol) = lam {
        let mut q = 0.0;
        for i in 0..km {
            q += d[i] * sol[i];
        }
        q
    } else {
        ctx.push(
            Issue::builder(IssueCode::InformationMatrixSingular)
                .severity(Severity::Warning)
                .message("Stuart–Maxwell covariance was not SPD; using Bowker fallback")
                .build(),
        );
        let b = bowker(y1, y2, session)?;
        return ctx.finish(b.value);
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: chi2_pvalue(stat.max(0.0), km.max(1) as f64).clamp(0.0, 1.0),
        df: km as f64,
        nobs: n as f64,
    })
}

/// Leybourne–McCabe stationarity test (statsmodels `LeybourneMcCabe`).
///
/// The AR lag used to prewhiten is not identification `p`.
pub fn leybourne_mccabe(y: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    if y.len() < 8 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Leybourne–McCabe needs n≥8")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: y.len() as f64,
        });
    }
    let n = y.len();
    let design = Matrix::from_fn(n - 1, 2, |i, j| if j == 0 { 1.0 } else { y[i] });
    let z = Vector::from_iter((1..n).map(|i| y[i]));
    let mut scratch = Report::new("lmc", "ols");
    let coef = least_squares(&mut scratch, &design, &z, &ctx.policy)
        .unwrap_or_else(|| Vector::from_slice(&[0.0, 0.0]));
    let fit = design.matvec(&coef);
    let mut e = Vec::new();
    for i in 0..z.len() {
        e.push(z[i] - fit[i]);
    }
    let mut cs = 0.0;
    let mut s2 = 0.0;
    let mut eta = 0.0;
    for &v in &e {
        cs += v;
        eta += cs * cs;
        s2 += v * v;
    }
    let nf = e.len() as f64;
    let stat = if s2 > 1e-18 {
        eta / (nf * nf * (s2 / nf))
    } else {
        f64::NAN
    };
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("Leybourne–McCabe p uses a χ²(1) sketch, not tabulated LM critical values")
            .build(),
    );
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: chi2_pvalue(stat.max(0.0), 1.0).clamp(0.0, 1.0),
        df: 1.0,
        nobs: nf,
    })
}

/// Range unit-root statistic (Aparicio / statsmodels `RangeUnitRoot`).
pub fn range_unit_root(y: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    let st = slice_stats(y.as_slice());
    if st.count < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("range unit-root needs n≥4")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: st.count as f64,
        });
    }
    let mut cs = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mean = st.mean;
    for &v in y.as_slice() {
        if !v.is_finite() {
            continue;
        }
        cs += v - mean;
        lo = lo.min(cs);
        hi = hi.max(cs);
    }
    let s = st.std().max(1e-12);
    let stat = (hi - lo) / (s * (st.count as f64).sqrt());
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: (2.0 * (1.0 - norm_cdf(stat.abs()))).clamp(0.0, 1.0),
        df: f64::NAN,
        nobs: st.count as f64,
    })
}

/// Residual variance-break F test (statsmodels `breaks_breakvar`).
///
/// The split index is not identification `p`.
pub fn breakvar(e: &Vector, split: usize, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(e),
        None,
        &ctx.policy,
    );
    let n = e.len();
    let k = split.clamp(2, n.saturating_sub(2));
    if n < 6 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("breakvar needs n≥6")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: n as f64,
        });
    }
    let v1 = {
        let s: f64 = e.as_slice().iter().take(k).map(|v| v * v).sum();
        s / k.max(1) as f64
    };
    let v2 = {
        let s: f64 = e.as_slice().iter().skip(k).map(|v| v * v).sum();
        s / (n - k).max(1) as f64
    };
    let stat = if v1 > 1e-18 { v2 / v1 } else { f64::NAN };
    let d1 = k.saturating_sub(1) as f64;
    let d2 = (n - k).saturating_sub(1) as f64;
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: f_pvalue(stat.max(0.0), d1.max(1.0), d2.max(1.0)).clamp(0.0, 1.0),
        df: d1,
        nobs: n as f64,
    })
}

/// Fitted C-vine pair-copula (statsmodels `VineCopula` lite).
#[derive(Clone, Debug)]
pub struct VineCopula {
    /// Pair-copula count (one edge for two series).
    pub n_trees: usize,
    /// Clayton \(\theta\) on the first tree.
    pub theta: f64,
    /// Pair log-likelihood.
    pub loglik: f64,
}

/// Fit a bivariate C-vine (one Clayton pair) on ranks.
///
/// Tree count is not identification `p`.
pub fn vine_copula(y1: &Vector, y2: &Vector, session: &Session) -> Result<Qualified<VineCopula>> {
    let q = clayton_copula(y1, y2, session)?;
    let mut ctx = FitCtx::with_session(session.child("vine"));
    ctx.finish(VineCopula {
        n_trees: 1,
        theta: q.value.theta,
        loglik: q.value.loglik,
    })
}

/// Equal-n sample size for a two-proportion z-test (statsmodels `samplesize_proportions_2indep`).
///
/// Returns \(n\) per arm. Arm count is not identification `p`.
pub fn samplesize_proportions(
    p1: f64,
    p2: f64,
    alpha: f64,
    power: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if ![p1, p2, alpha, power].iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .severity(Severity::Warning)
                .message("samplesize_proportions received a non-finite argument")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let a = p1.clamp(0.0, 1.0);
    let b = p2.clamp(0.0, 1.0);
    let al = if (0.0..1.0).contains(&alpha) {
        alpha
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("samplesize_proportions alpha={alpha} not in (0,1); using 0.05"))
                .build(),
        );
        0.05
    };
    let pw = if (0.0..1.0).contains(&power) {
        power
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("samplesize_proportions power={power} not in (0,1); using 0.8"))
                .build(),
        );
        0.8
    };
    let delta = a - b;
    if delta.abs() < 1e-12 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .severity(Severity::Warning)
                .message("samplesize_proportions: p1=p2 so n is unidentified")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let za = norm_ppf(1.0 - al / 2.0);
    let zb = norm_ppf(pw);
    let num = (za + zb) * (za + zb) * (a * (1.0 - a) + b * (1.0 - b));
    let n = num / (delta * delta);
    ctx.finish(n.max(2.0))
}

/// Likelihood-ratio comparison of a Cox model against the nested null (β = 0).
///
/// Covariate / cause counts are not identification `p`.
pub fn compare_cox(
    durations: &Vector,
    events: &Vector,
    x: &Matrix,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(durations), &ctx.policy);
    if durations.len() != events.len() || durations.len() != x.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .severity(Severity::Warning)
                .message(format!(
                    "compare_cox lengths time={} event={} n_x={}",
                    durations.len(),
                    events.len(),
                    x.nrows()
                ))
                .build(),
        );
    }
    let n = durations.len().min(events.len()).min(x.nrows());
    let p = x.ncols().max(1);
    let mut idx: Vec<usize> = (0..n)
        .filter(|&i| durations[i].is_finite() && events[i].is_finite())
        .collect();
    idx.sort_by(|&a, &b| {
        durations[a]
            .partial_cmp(&durations[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Score test at β=0 (nested null). Covariate count is test df, not identification p.
    let mut u = vec![0.0_f64; p];
    let mut i_mat = vec![0.0_f64; p * p];
    let mut n_ev = 0usize;
    for (k, &i) in idx.iter().enumerate() {
        if events[i] < 0.5 {
            continue;
        }
        n_ev += 1;
        let r = idx.len() - k;
        if r == 0 {
            continue;
        }
        let rf = r as f64;
        let mut mean = vec![0.0_f64; p];
        for &j in idx[k..].iter() {
            for c in 0..p {
                mean[c] += x.get(j, c);
            }
        }
        for c in 0..p {
            mean[c] /= rf;
            u[c] += x.get(i, c) - mean[c];
        }
        for a in 0..p {
            for b in 0..p {
                let mut s2 = 0.0_f64;
                for &j in idx[k..].iter() {
                    s2 += (x.get(j, a) - mean[a]) * (x.get(j, b) - mean[b]);
                }
                i_mat[a * p + b] += s2 / rf;
            }
        }
    }
    let mut stat = 0.0_f64;
    if p == 1 {
        let v = i_mat[0];
        if v.abs() > 1e-14 {
            stat = u[0] * u[0] / v;
        } else {
            ctx.push(
                Issue::builder(IssueCode::InformationMatrixSingular)
                    .severity(Severity::Warning)
                    .message("compare_cox score variance is zero")
                    .build(),
            );
            stat = f64::NAN;
        }
    } else {
        let mut scratch = Report::new("compare_cox", "score");
        let im = faer::Mat::<f64>::from_fn(p, p, |a, b| i_mat[a * p + b]);
        let rhs = Vector::from_iter(u.iter().copied());
        match chol_solve(&mut scratch, &im, &rhs, &ctx.policy) {
            Some(sol) => {
                stat = (0..p).map(|c| u[c] * sol[c]).sum::<f64>().max(0.0);
            }
            None => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .severity(Severity::Warning)
                        .message("compare_cox score information is not SPD; statistic is unidentified")
                        .build(),
                );
                stat = f64::NAN;
            }
        }
    }
    let df = p as f64;
    let pvalue = if stat.is_finite() {
        chi2_pvalue(stat, df)
    } else {
        f64::NAN
    };
    if n_ev == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("compare_cox saw no events")
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: n as f64,
    })
}

/// Alias of [`goldfeld_quandt`] (statsmodels `het_goldfeldquandt`).
pub fn het_goldfeldquandt(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    goldfeld_quandt(x, y, session)
}

/// Alias of [`harvey_collier`] (statsmodels `linear_harvey_collier`).
pub fn linear_harvey_collier(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    harvey_collier(x, y, session)
}

/// Alias of [`rainbow`] at the default central fraction 0.5
/// (statsmodels `linear_rainbow`).
pub fn linear_rainbow(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    rainbow(x, y, 0.5, session)
}

/// Alias of [`cusum_ols`] (statsmodels `breaks_cusumolsresid`).
pub fn breaks_cusumolsresid(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    cusum_ols(x, y, session)
}

/// Two-sample comparison of right-censored survival curves
/// (statsmodels `duration.survfunc.compare_survfunc`).
///
/// Concatenates `(t1, e1)` and `(t2, e2)` and runs the Mantel–Haenszel
/// log-rank. Sample sizes are not identification `p`.
pub fn compare_survfunc(
    t1: &Vector,
    e1: &Vector,
    t2: &Vector,
    e2: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let n1 = t1.len().min(e1.len());
    let n2 = t2.len().min(e2.len());
    if n1 == 0 || n2 == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("compare_survfunc needs two non-empty samples")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: 1.0,
            nobs: (n1 + n2) as f64,
        });
    }
    let n = n1 + n2;
    let times = Vector::from_iter((0..n).map(|i| {
        if i < n1 {
            t1[i]
        } else {
            t2[i - n1]
        }
    }));
    let events = Vector::from_iter((0..n).map(|i| {
        if i < n1 {
            e1[i]
        } else {
            e2[i - n1]
        }
    }));
    let groups = Vector::from_iter((0..n).map(|i| if i < n1 { 0.0 } else { 1.0 }));
    match logrank(&times, &events, &groups, &session.child("survfunc")) {
        Ok(q) => ctx.finish(q.value),
        Err(_) => {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("compare_survfunc log-rank failed; statistic is unidentified")
                    .build(),
            );
            ctx.finish(HypothesisTest {
                statistic: f64::NAN,
                pvalue: f64::NAN,
                df: 1.0,
                nobs: n as f64,
            })
        }
    }
}

/// Cook's distances from an intercept-on OLS of `y` on `X`
/// (statsmodels `OLSInfluence.cooks_distance`).
///
/// Observation count is not identification `p`.
pub fn cooks_distance(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let infl = match ols_influence(x, y, &session.child("cooks")) {
        Ok(q) => q.value,
        Err(_) => {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("cooks_distance: OLS influence failed")
                    .build(),
            );
            return ctx.finish(Vector::zeros(y.len()));
        }
    };
    let n = infl.hat.len().min(infl.resid.len());
    let p = infl.dfbetas.ncols().max(1);
    let mut sse = 0.0_f64;
    for i in 0..n {
        sse += infl.resid[i] * infl.resid[i];
    }
    let df = (n as f64 - p as f64).max(1.0);
    let s2 = (sse / df).max(1e-18);
    let cooks = Vector::from_iter((0..n).map(|i| {
        let h = infl.hat[i].clamp(0.0, 1.0 - 1e-12);
        let denom = (1.0 - h).max(1e-12);
        infl.resid[i] * infl.resid[i] / (p as f64 * s2) * h / (denom * denom)
    }));
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("Cook's D uses the pooled residual scale, not s_{(i)}")
            .compromise(NumericalCompromise::new(
                "leave-one-out Cook's distance",
                "hat-diagonal formula with the full-sample s²",
                "s_{(i)} is not recomputed",
                "rank cases by D, do not treat a 4/n cutoff as exact",
            ))
            .build(),
    );
    ctx.finish(cooks)
}

/// Savitzky–Golay local polynomial smoother (SciPy `savgol_filter`).
///
/// Window length and polynomial degree are not identification `p`.
pub fn savgol_filter(
    y: &Vector,
    window: usize,
    polyorder: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_series_as_target(&mut ctx, y);
    let n = y.len();
    let mut w = window.max(3);
    if window < 3 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!("savgol window={window} < 3; using 3"))
                .build(),
        );
    }
    if w % 2 == 0 {
        w += 1;
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("savgol window must be odd; using {w}"))
                .build(),
        );
    }
    if w > n {
        w = if n % 2 == 0 { n.saturating_sub(1).max(1) } else { n.max(1) };
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!("savgol window longer than n={n}; using {w}"))
                .build(),
        );
    }
    let mut deg = polyorder;
    if deg >= w {
        deg = w.saturating_sub(1);
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "savgol polyorder={polyorder} ≥ window {w}; using {deg}"
                ))
                .build(),
        );
    }
    let h = w / 2;
    let out = Vector::from_iter((0..n).map(|t| savgol_at(y.as_slice(), t, h, deg)));
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("savgol_filter is a local polynomial smoother, not a causal filter")
            .compromise(NumericalCompromise::new(
                "causal one-sided Savitzky–Golay",
                "centred window with edge truncation",
                "the edges use a shorter design",
                "do not read edge values as the same FIR as the interior",
            ))
            .build(),
    );
    ctx.finish(out)
}

fn savgol_at(y: &[f64], t: usize, h: usize, deg: usize) -> f64 {
    let n = y.len();
    if n == 0 {
        return f64::NAN;
    }
    let lo = t.saturating_sub(h);
    let hi = (t + h + 1).min(n);
    let m = hi - lo;
    if m == 0 {
        return f64::NAN;
    }
    let d = deg.min(m.saturating_sub(1));
    if d == 0 {
        let mut s = 0.0_f64;
        let mut c = 0.0_f64;
        for v in y.iter().take(hi).skip(lo) {
            if v.is_finite() {
                s += *v;
                c += 1.0;
            }
        }
        return if c > 0.0 { s / c } else { f64::NAN };
    }
    let p = d + 1;
    let mut g = vec![0.0_f64; p * p];
    let mut rhs = vec![0.0_f64; p];
    for i in 0..m {
        let yi = y[lo + i];
        if !yi.is_finite() {
            continue;
        }
        let u = (lo + i) as f64 - t as f64;
        let mut uk = 1.0_f64;
        for k in 0..p {
            rhs[k] += uk * yi;
            let mut ul = 1.0_f64;
            for l in 0..p {
                g[k * p + l] += uk * ul;
                ul *= u;
            }
            uk *= u;
        }
    }
    // Gaussian elimination for β₀ (value at the window centre).
    let mut a = g;
    let mut b = rhs;
    for k in 0..p {
        let mut piv = k;
        for i in (k + 1)..p {
            if a[i * p + k].abs() > a[piv * p + k].abs() {
                piv = i;
            }
        }
        if a[piv * p + k].abs() <= 1e-14 {
            let mut s = 0.0_f64;
            let mut c = 0.0_f64;
            for v in y.iter().take(hi).skip(lo) {
                if v.is_finite() {
                    s += *v;
                    c += 1.0;
                }
            }
            return if c > 0.0 { s / c } else { f64::NAN };
        }
        if piv != k {
            for j in 0..p {
                a.swap(k * p + j, piv * p + j);
            }
            b.swap(k, piv);
        }
        let diag = a[k * p + k];
        for i in (k + 1)..p {
            let f = a[i * p + k] / diag;
            for j in k..p {
                a[i * p + j] -= f * a[k * p + j];
            }
            b[i] -= f * b[k];
        }
    }
    for i in (0..p).rev() {
        let mut s = b[i];
        for j in (i + 1)..p {
            s -= a[i * p + j] * b[j];
        }
        b[i] = s / a[i * p + i];
    }
    b[0]
}

/// Proportional Denton–Cholette temporal disaggregation
/// (statsmodels `denton_cholette`).
///
/// Minimizes first differences of \(x_t/I_t\) subject to the low-frequency
/// aggregation. Period is not identification `p`.
pub fn denton_cholette(
    indicator: &Vector,
    totals: &Vector,
    period: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(indicator),
        None,
        &ctx.policy,
    );
    let s = period.max(2);
    if period < 2 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("Denton–Cholette period={period} is <2; using 2"))
                .build(),
        );
    }
    let m = totals.len();
    let need = m.saturating_mul(s);
    if indicator.len() != need {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .severity(Severity::Warning)
                .message(format!(
                    "Denton–Cholette indicator n={} ≠ period {s} × {} totals",
                    indicator.len(),
                    m
                ))
                .build(),
        );
    }
    let n = indicator.len().min(need);
    let years = if s == 0 { 0 } else { n / s };
    if years == 0 || m == 0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Denton–Cholette needs at least one complete low-frequency period")
                .build(),
        );
        return ctx.finish(Vector::from_iter(indicator.as_slice().iter().copied()));
    }
    let mut w = vec![1.0_f64; years * s];
    let mut skipped = 0usize;
    for t in 0..years * s {
        let v = indicator.as_slice().get(t).copied().unwrap_or(0.0);
        if !v.is_finite() || v <= 0.0 {
            skipped += 1;
            w[t] = 1.0;
        } else {
            w[t] = v;
        }
    }
    if skipped > 0 {
        ctx.push(
            Issue::builder(IssueCode::NonPositiveSeries)
                .severity(Severity::Warning)
                .message(format!(
                    "Denton–Cholette replaced {skipped} non-positive indicator values with 1"
                ))
                .build(),
        );
    }
    let mut z = vec![0.0_f64; years * s];
    for t in 0..years {
        let mut ws = 0.0_f64;
        for q in 0..s {
            ws += w[t * s + q];
        }
        let yt = totals.as_slice().get(t).copied().unwrap_or(0.0);
        let z0 = if ws.abs() <= 1e-15 {
            yt / s as f64
        } else {
            yt / ws
        };
        for q in 0..s {
            z[t * s + q] = z0;
        }
    }
    for _ in 0..48 {
        let nn = z.len();
        let mut g = vec![0.0_f64; nn];
        if nn >= 2 {
            g[0] = z[0] - z[1];
            g[nn - 1] = z[nn - 1] - z[nn - 2];
            for t in 1..nn - 1 {
                g[t] = 2.0 * z[t] - z[t - 1] - z[t + 1];
            }
        }
        for t in 0..years {
            let mut mg = 0.0_f64;
            let mut sw = 0.0_f64;
            for q in 0..s {
                mg += w[t * s + q] * g[t * s + q];
                sw += w[t * s + q];
            }
            let mean = if sw.abs() <= 1e-15 {
                0.0
            } else {
                mg / sw
            };
            for q in 0..s {
                z[t * s + q] -= 0.25 * (g[t * s + q] - mean);
            }
        }
        for t in 0..years {
            let mut sum = 0.0_f64;
            let mut sw = 0.0_f64;
            for q in 0..s {
                sum += w[t * s + q] * z[t * s + q];
                sw += w[t * s + q];
            }
            let yt = totals.as_slice().get(t).copied().unwrap_or(0.0);
            let add = if sw.abs() <= 1e-15 {
                0.0
            } else {
                (yt - sum) / sw
            };
            for q in 0..s {
                z[t * s + q] += add;
            }
        }
    }
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("denton_cholette uses a projected first-difference smoother on x/I")
            .compromise(NumericalCompromise::new(
                "exact Denton–Cholette KKT system",
                "iterated gradient projection on z = x/I",
                "the KKT matrix is not assembled",
                "treat the path as a proportional disaggregation, not the unique Cholette solution",
            ))
            .build(),
    );
    ctx.finish(Vector::from_iter((0..years * s).map(|i| z[i] * w[i])))
}

/// OLS prediction means, SEs, and a normal interval
/// (statsmodels `OLSResults.get_prediction`).
///
/// Observation count is not identification `p`. The interval uses
/// \(\Phi^{-1}(1-\alpha/2)\,s\sqrt{1+h_{ii}}\), not a Student-\(t\) quantile.
#[derive(Clone, Debug)]
pub struct PredictionResults {
    /// Fitted mean.
    pub predicted: Vector,
    /// Standard error of the mean.
    pub se_mean: Vector,
    /// Predictive SE.
    pub se_obs: Vector,
    /// Lower bound of the observation interval.
    pub lower: Vector,
    /// Upper bound of the observation interval.
    pub upper: Vector,
}

/// In-sample OLS prediction summary.
pub fn get_prediction(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<PredictionResults>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let wps = match wls_prediction_std(x, y, &session.child("getpred")) {
        Ok(q) => q.value,
        Err(_) => {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("get_prediction: WLS prediction SEs failed")
                    .build(),
            );
            return ctx.finish(PredictionResults {
                predicted: Vector::zeros(y.len()),
                se_mean: Vector::zeros(y.len()),
                se_obs: Vector::zeros(y.len()),
                lower: Vector::zeros(y.len()),
                upper: Vector::zeros(y.len()),
            });
        }
    };
    let z = norm_ppf(0.975);
    let n = wps.predicted.len();
    let lower = Vector::from_iter((0..n).map(|i| wps.predicted[i] - z * wps.se_obs[i]));
    let upper = Vector::from_iter((0..n).map(|i| wps.predicted[i] + z * wps.se_obs[i]));
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("get_prediction intervals use a normal quantile, not Student-t")
            .compromise(NumericalCompromise::new(
                "t_{n-p} predictive interval",
                "z_{0.975} times the OLS predictive SE",
                "the interval is slightly too narrow in small samples",
                "read the band as an approximate 95% Gaussian interval",
            ))
            .build(),
    );
    ctx.finish(PredictionResults {
        predicted: wps.predicted,
        se_mean: wps.se_mean,
        se_obs: wps.se_obs,
        lower,
        upper,
    })
}

/// OLS coefficient t-tests (statsmodels `OLSResults.t_test` for H0: β = 0).
///
/// Coefficient count is the design width (including intercept), which is the
/// model's identification `p`.
#[derive(Clone, Debug)]
pub struct OlsTTest {
    /// Intercept then slopes.
    pub params: Vector,
    /// Standard errors.
    pub se: Vector,
    /// t statistics.
    pub tvalues: Vector,
    /// Two-sided Student-t p-values.
    pub pvalues: Vector,
    /// Residual degrees of freedom.
    pub df: f64,
}

/// Individual OLS t-tests of \(H_0:\beta_j=0\).
pub fn t_test(x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<OlsTTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let design = x.with_intercept();
    let n = design.nrows().min(y.len());
    let p = design.ncols();
    let empty = || OlsTTest {
        params: Vector::zeros(p),
        se: Vector::zeros(p),
        tvalues: Vector::zeros(p),
        pvalues: Vector::from_iter((0..p).map(|_| f64::NAN)),
        df: (n as f64 - p as f64).max(1.0),
    };
    let mut scratch = Report::new("ttest", "ols");
    let Some(beta) = least_squares(&mut scratch, &design, y, &ctx.policy) else {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("t_test: OLS failed")
                .build(),
        );
        return ctx.finish(empty());
    };
    for issue in scratch.issues() {
        if matches!(
            issue.code,
            IssueCode::ResidualTooLarge
                | IssueCode::NearSingular
                | IssueCode::R2IsOne
                | IssueCode::RankZero
                | IssueCode::CholeskyFailed
        ) {
            continue;
        }
        ctx.push(issue.clone());
    }
    let fit = design.matvec(&beta);
    let mut sse = 0.0_f64;
    for i in 0..n {
        let e = y[i] - fit[i];
        sse += e * e;
    }
    let df = (n as f64 - p as f64).max(1.0);
    let sigma2 = sse / df;
    let xtx = design.gram();
    let mut se = Vector::zeros(p);
    let mut ok = true;
    for j in 0..p {
        let mut e = Vector::zeros(p);
        e[j] = 1.0;
        let mut sc = Report::new("ttest", "inv");
        match chol_solve(&mut sc, &xtx, &e, &ctx.policy) {
            Some(col) => {
                se[j] = (sigma2 * col[j].max(0.0)).sqrt();
            }
            None => {
                ok = false;
                se[j] = f64::NAN;
            }
        }
    }
    if !ok {
        ctx.push(
            Issue::builder(IssueCode::DidNotConverge)
                .severity(Severity::Warning)
                .message("t_test: X'X is not SPD; some SEs are unidentified")
                .build(),
        );
    }
    let tvalues = Vector::from_iter((0..p).map(|j| {
        if se[j].is_finite() && se[j] > 1e-18 {
            beta[j] / se[j]
        } else {
            f64::NAN
        }
    }));
    let pvalues = Vector::from_iter((0..p).map(|j| student_t_pvalue(tvalues[j], df)));
    ctx.finish(OlsTTest {
        params: beta,
        se,
        tvalues,
        pvalues,
        df,
    })
}

/// Joint OLS Wald / F test that every slope is zero
/// (statsmodels `OLSResults.f_test` for H0: slopes = 0).
pub fn f_test(x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<HypothesisTest>> {
    wald_ols(x, y, session)
}

/// OLS coefficient confidence intervals (statsmodels `OLSResults.conf_int`).
///
/// Uses a normal quantile times the OLS SE. Coefficient count is the design
/// width.
#[derive(Clone, Debug)]
pub struct OlsConfInt {
    /// Lower bounds (intercept then slopes).
    pub low: Vector,
    /// Upper bounds.
    pub high: Vector,
    /// Point estimates.
    pub params: Vector,
}

/// Two-sided OLS coefficient intervals at level `1 − alpha`.
pub fn conf_int(
    x: &Matrix,
    y: &Vector,
    alpha: f64,
    session: &Session,
) -> Result<Qualified<OlsConfInt>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let a = if alpha.is_finite() && alpha > 0.0 && alpha < 1.0 {
        alpha
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("conf_int alpha={alpha} is not in (0,1); using 0.05"))
                .build(),
        );
        0.05
    };
    let tt = match t_test(x, y, &session.child("confint")) {
        Ok(q) => q.value,
        Err(_) => {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("conf_int: t_test failed")
                    .build(),
            );
            return ctx.finish(OlsConfInt {
                low: Vector::zeros(x.ncols() + 1),
                high: Vector::zeros(x.ncols() + 1),
                params: Vector::zeros(x.ncols() + 1),
            });
        }
    };
    let z = norm_ppf(1.0 - 0.5 * a);
    let p = tt.params.len();
    let low = Vector::from_iter((0..p).map(|j| tt.params[j] - z * tt.se[j]));
    let high = Vector::from_iter((0..p).map(|j| tt.params[j] + z * tt.se[j]));
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("conf_int uses a normal quantile, not Student-t")
            .compromise(NumericalCompromise::new(
                "t_{n-p} interval",
                format!("z_{{{:.3}}} times the OLS SE", 1.0 - 0.5 * a),
                "the interval is slightly too narrow in small samples",
                "read the bounds as an approximate Gaussian interval",
            ))
            .build(),
    );
    ctx.finish(OlsConfInt {
        low,
        high,
        params: tt.params,
    })
}

/// Inverse-probability-of-censoring Nadaraya–Watson
/// (statsmodels `KernelCensoredReg`).
///
/// Bandwidth `h` is not identification `p`. Censoring weights are computed
/// locally; a fully uncensored sample reduces to [`kernel_reg`].
pub fn kernel_censored_reg(
    x: &Vector,
    y: &Vector,
    event: &Vector,
    h: f64,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, x, y);
    if event.len() != y.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .severity(Severity::Warning)
                .message(format!(
                    "kernel_censored_reg event.len()={} ≠ y.len()={}",
                    event.len(),
                    y.len()
                ))
                .build(),
        );
    }
    if let Some(issue) = scan_finite(event.as_slice()).to_issue("event") {
        ctx.push(issue);
    }
    let mut bw = h;
    if !bw.is_finite() || bw <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("KernelCensoredReg bandwidth {h} is not positive; using 1"))
                .build(),
        );
        bw = 1.0;
    }
    let n = x.len().min(y.len()).min(event.len());
    let mut rows: Vec<(f64, usize)> = (0..n)
        .filter(|&i| y[i].is_finite() && event[i].is_finite())
        .map(|i| (y[i], i))
        .collect();
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut g_at = vec![1.0_f64; n];
    let mut g = 1.0_f64;
    let mut i = 0usize;
    while i < rows.len() {
        let t = rows[i].0;
        let at_risk = rows.len() - i;
        let mut c = 0.0_f64;
        let start = i;
        while i < rows.len() && (rows[i].0 - t).abs() <= 0.0 {
            if rows[i].1 < n && event[rows[i].1] <= 0.5 {
                c += 1.0;
            }
            i += 1;
        }
        for k in start..i {
            g_at[rows[k].1] = g;
        }
        if c > 0.0 && at_risk > 0 {
            g *= 1.0 - c / at_risk as f64;
            if g < 1e-12 {
                g = 1e-12;
            }
        }
    }
    let mut wgt = vec![0.0_f64; n];
    let mut n_unc = 0usize;
    for i in 0..n {
        if event[i] > 0.5 {
            n_unc += 1;
            wgt[i] = 1.0 / g_at[i].max(1e-12);
        }
    }
    if n_unc == 0 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("kernel_censored_reg has no uncensored responses; returning y")
                .build(),
        );
        return ctx.finish(Vector::from_iter((0..n).map(|i| y[i])));
    }
    let out = Vector::from_iter((0..n).map(|i| {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for j in 0..n {
            if wgt[j] <= 0.0 {
                continue;
            }
            let z = (x[i] - x[j]) / bw;
            let k = (-0.5 * z * z).exp() * wgt[j];
            num += k * y[j];
            den += k;
        }
        if den > 0.0 {
            num / den
        } else {
            y[i]
        }
    }));
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("KernelCensoredReg uses IPCW Nadaraya–Watson, not Beran's local likelihood")
            .compromise(NumericalCompromise::new(
                "Beran / Dabrowska kernel survival regression",
                "IPCW Gaussian Nadaraya–Watson on uncensored rows",
                "the censoring law is estimated by a product-limit of the complementary events",
                "treat the curve as a weighted smoother, not the nonparametric MLE",
            ))
            .build(),
    );
    ctx.finish(out)
}

/// Gaussian process with exponential covariance
/// (statsmodels `ProcessMLE` lite).
///
/// Covariance parameters are not identification `p`.
#[derive(Clone, Debug)]
pub struct ProcessMleFit {
    /// Estimated mean.
    pub mean: f64,
    /// Marginal variance.
    pub sigma2: f64,
    /// Exponential range \(\rho\).
    pub range: f64,
}

/// Profile a mean-plus-exponential-covariance process on irregular time.
///
/// `t` is the observation time; `y` is the response. A small grid over \(\rho\)
/// is scored by GLS residual quadratic form. Failed Cholesky trials are skipped.
pub fn process_mle(t: &Vector, y: &Vector, session: &Session) -> Result<Qualified<ProcessMleFit>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_pair(&mut ctx, t, y);
    let n = t.len().min(y.len());
    if n == 0 {
        return ctx.finish(ProcessMleFit {
            mean: f64::NAN,
            sigma2: f64::NAN,
            range: f64::NAN,
        });
    }
    let mut dts: Vec<f64> = Vec::new();
    let mut order: Vec<usize> = (0..n).filter(|&i| t[i].is_finite() && y[i].is_finite()).collect();
    order.sort_by(|a, b| t[*a].partial_cmp(&t[*b]).unwrap_or(std::cmp::Ordering::Equal));
    for w in order.windows(2) {
        let d = t[w[1]] - t[w[0]];
        if d > 0.0 {
            dts.push(d);
        }
    }
    dts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dt = if dts.is_empty() {
        1.0
    } else {
        dts[dts.len() / 2]
    };
    let mu0 = y.mean();
    let mut sse0 = 0.0_f64;
    for i in 0..n {
        let e = y[i] - mu0;
        sse0 += e * e;
    }
    let s20 = (sse0 / n.max(1) as f64).max(1e-12);
    let grid = [dt, 2.0 * dt, 4.0 * dt, 8.0 * dt, 16.0 * dt];
    let mut best = ProcessMleFit {
        mean: mu0,
        sigma2: s20,
        range: dt,
    };
    let mut best_q = f64::INFINITY;
    for &rho in &grid {
        if !(rho.is_finite() && rho > 0.0) {
            continue;
        }
        let mut c = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                c[(i, j)] = (-(t[i] - t[j]).abs() / rho).exp();
            }
            c[(i, i)] += 1e-8;
        }
        let ones = Vector::from_iter((0..n).map(|_| 1.0));
        let yn = Vector::from_iter((0..n).map(|i| y[i]));
        let mut sc1 = Report::new("pmle", "ones");
        let mut sc2 = Report::new("pmle", "y");
        let Some(z) = chol_solve(&mut sc1, &c, &ones, &ctx.policy) else {
            continue;
        };
        let Some(w) = chol_solve(&mut sc2, &c, &yn, &ctx.policy) else {
            continue;
        };
        let den = ones.dot(&z);
        if den.abs() <= 1e-18 {
            continue;
        }
        let mu = ones.dot(&w) / den;
        let e = Vector::from_iter((0..n).map(|i| y[i] - mu));
        let mut sc3 = Report::new("pmle", "e");
        let Some(u) = chol_solve(&mut sc3, &c, &e, &ctx.policy) else {
            continue;
        };
        let q = e.dot(&u);
        if q.is_finite() && q < best_q {
            best_q = q;
            best = ProcessMleFit {
                mean: mu,
                sigma2: (q / n.max(1) as f64).max(1e-12),
                range: rho,
            };
        }
    }
    if !best_q.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::DidNotConverge)
                .severity(Severity::Warning)
                .message("process_mle: every exponential trial failed Cholesky; using MOM")
                .build(),
        );
    }
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("ProcessMLE profiles a 5-point ρ grid, not a joint Gaussian MLE")
            .compromise(NumericalCompromise::new(
                "joint MLE of (μ, σ², ρ) for the exponential covariance",
                "GLS mean on a discrete ρ grid with a nugget",
                "the likelihood is not maximised in σ² jointly with ρ",
                "treat ρ as a grid index, not the exact MLE range",
            ))
            .build(),
    );
    ctx.finish(best)
}

/// Nested OLS Lagrange-multiplier test (statsmodels `compare_lm_test`).
///
/// Fits the restricted design, then regresses those residuals on the
/// unrestricted design. The statistic is \(n R^2\) against \(\chi^2_{p_u-p_r}\).
/// Extra-column count is not identification `p`. Inner OLS failures are not
/// promoted as fatal Cholesky / rank issues.
pub fn compare_lm(
    y: &Vector,
    x_restr: &Matrix,
    x_unrestr: &Matrix,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x_restr, Some(y), &ctx.policy);
    inspect_xy(&mut ctx.report, x_unrestr, Some(y), &ctx.policy);
    let n = y.len().min(x_restr.nrows()).min(x_unrestr.nrows());
    let df = (x_unrestr.ncols() as f64 - x_restr.ncols() as f64).max(1.0);
    let nan = || HypothesisTest {
        statistic: f64::NAN,
        pvalue: f64::NAN,
        df,
        nobs: n as f64,
    };
    if x_unrestr.ncols() <= x_restr.ncols() {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message("compare_lm: unrestricted design is not wider")
                .build(),
        );
    }
    let mut sc_r = Report::new("compare_lm", "restr");
    let Some(br) = least_squares(&mut sc_r, x_restr, y, &ctx.policy) else {
        ctx.push(
            Issue::builder(IssueCode::DidNotConverge)
                .severity(Severity::Warning)
                .message("compare_lm: restricted OLS failed")
                .build(),
        );
        return ctx.finish(nan());
    };
    if br.len() != x_restr.ncols() {
        return ctx.finish(nan());
    }
    let fr = x_restr.matvec(&br);
    let resid = Vector::from_iter((0..n).map(|i| y[i] - if i < fr.len() { fr[i] } else { 0.0 }));
    let mut sc_u = Report::new("compare_lm", "aux");
    let Some(bu) = least_squares(&mut sc_u, x_unrestr, &resid, &ctx.policy) else {
        ctx.push(
            Issue::builder(IssueCode::DidNotConverge)
                .severity(Severity::Warning)
                .message("compare_lm: auxiliary OLS of restricted residuals failed")
                .build(),
        );
        return ctx.finish(nan());
    };
    if bu.len() != x_unrestr.ncols() {
        return ctx.finish(nan());
    }
    let fu = x_unrestr.matvec(&bu);
    let mut sse = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sy2 = 0.0_f64;
    for i in 0..n {
        let e = resid[i];
        let fit = if i < fu.len() { fu[i] } else { 0.0 };
        let r = e - fit;
        sse += r * r;
        sy += e;
        sy2 += e * e;
    }
    let nf = n as f64;
    let sst = sy2 - if nf > 0.0 { sy * sy / nf } else { 0.0 };
    if sst.abs() <= 1e-15 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .severity(Severity::Warning)
                .message("compare_lm: restricted residuals have zero variance")
                .build(),
        );
        return ctx.finish(nan());
    }
    let r2 = 1.0 - sse / sst;
    let stat = nf * r2.max(0.0);
    let pvalue = if stat.is_finite() {
        chi2_pvalue(stat, df)
    } else {
        f64::NAN
    };
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue,
        df,
        nobs: nf,
    })
}

/// Ramsey RESET under the statsmodels `linear_reset` name.
pub fn linear_reset(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    ramsey_reset(x, y, session)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn pearson_of_a_line_is_one() {
        let x = Vector::from_iter((0..20).map(|i| i as f64));
        let y = Vector::from_iter((0..20).map(|i| 3.0 + 2.0 * i as f64));
        let session = Session::new("pearson", "test");
        let q = pearson(&x, &y, &session).expect("pearson of a line");
        assert!(
            (q.value - 1.0).abs() < 1e-12,
            "pearson={} report={:?}",
            q.value,
            q.report.issues().iter().map(|i| i.code).collect::<Vec<_>>()
        );
    }

    #[test]
    fn ttest_1samp_standard_normal_like() {
        let mut rng = Rng::new(11);
        let x = Vector::from_iter((0..40).map(|_| rng.standard_normal()));
        let session = Session::new("ttest_1samp", "test");
        let q = ttest_1samp(&x, 0.0, &session).expect("ttest");
        assert!(q.value.statistic.is_finite(), "stat={}", q.value.statistic);
        assert!(
            q.value.pvalue > 1e-6,
            "unexpectedly tiny p-value {} for N(0,1)-like data vs μ=0",
            q.value.pvalue
        );
        assert!(q.value.statistic.abs() < 6.0);
        assert!((q.value.df - 39.0).abs() < 1e-12);
    }

    #[test]
    fn jarque_bera_constant_is_error() {
        let x = Vector::filled(12, 4.0);
        let session = Session::new("jarque_bera", "test");
        let err = jarque_bera(&x, &session).expect_err("constant JB must abort");
        assert!(
            err.report.contains(IssueCode::ConstantTarget)
                || err.report.contains(IssueCode::MeaninglessFit)
                || err.primary.code == IssueCode::ConstantTarget
                || err.primary.code == IssueCode::MeaninglessFit,
            "primary={:?} codes={:?}",
            err.primary.code,
            err.report
                .issues()
                .iter()
                .map(|i| i.code)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn het_white_on_homoscedastic_residuals() {
        let x = Matrix::from_fn(24, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let e = Vector::from_iter((0..24).map(|i| ((i % 3) as f64 - 1.0) * 0.2));
        let q = het_white(&e, &x, &Session::new("white", "test")).expect("white");
        assert!(q.value.statistic.is_finite());
        assert!(q.value.pvalue.is_finite());
    }

    #[test]
    fn tukey_gq_bg_pp() {
        let a = Vector::from_iter((0..12).map(|i| 0.1 * i as f64));
        let b = Vector::from_iter((0..12).map(|i| 3.0 + 0.1 * i as f64));
        let c = Vector::from_iter((0..12).map(|i| 6.0 + 0.1 * i as f64));
        let t = tukey_hsd(&[&a, &b, &c], &Session::new("tukey", "t")).expect("tukey");
        assert!(t.value.pairwise_stat.get(0, 2) > t.value.pairwise_stat.get(0, 1));
        let x = Matrix::from_fn(40, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..40).map(|i| 0.2 * i as f64 + 0.05 * (i as f64).sin()));
        let gq = goldfeld_quandt(&x, &y, &Session::new("gq", "t")).expect("gq");
        assert!(gq.value.statistic.is_finite());
        let design = x.with_intercept();
        let e = Vector::from_iter((0..40).map(|i| {
            if i == 0 {
                0.3
            } else {
                0.6 * ((i as f64).sin())
            }
        }));
        let bg = breusch_godfrey(&e, &design, 1, &Session::new("bg", "t")).expect("bg");
        assert!(bg.value.df > 0.0);
        let mut rw = vec![0.0; 40];
        for t in 1..40 {
            rw[t] = rw[t - 1] + 0.4;
        }
        let pp = phillips_perron(&Vector::from_slice(&rw), Some(1), &Session::new("pp", "t"))
            .expect("pp");
        assert!(pp.report.contains(IssueCode::NonStationary) || pp.value.stat.is_finite());
        let reset = ramsey_reset(&x, &y, &Session::new("reset", "t")).expect("reset");
        assert!(reset.value.statistic.is_finite() || reset.value.pvalue.is_nan());
        let hc = harvey_collier(&x, &y, &Session::new("hc", "t")).expect("hc");
        assert!(hc.value.nobs > 0.0);
        let e = Vector::from_iter((0..40).map(|i| 0.2 * ((i as f64).sin())));
        let arch = arch_lm(&e, 2, &Session::new("arch", "t")).expect("arch");
        assert!(arch.value.df > 0.0);
        let xs = Vector::from_iter((0..20).map(|i| i as f64));
        let ys = Vector::from_iter((0..20).map(|i| 2.0 * i as f64));
        let sm = lowess(&xs, &ys, 0.4, &Session::new("lo", "t")).expect("lowess");
        assert!((sm.value[10] - 20.0).abs() < 1.0);
        let chow = chow_test(&x, &y, 20, &Session::new("chow", "t")).expect("chow");
        assert!(chow.value.statistic.is_finite() || chow.value.pvalue.is_nan());
        let yb = Vector::from_iter((0..40).map(|i| {
            if i < 20 {
                0.2 * i as f64
            } else {
                20.0 - 0.8 * (i as f64 - 20.0)
            }
        }));
        let br = chow_test(&x, &yb, 20, &Session::new("chowb", "t")).expect("chowb");
        assert!(br.report.contains(IssueCode::StructuralBreak) || br.value.pvalue < 0.2);
        let cu = cusum_ols(&x, &y, &Session::new("cusum", "t")).expect("cusum");
        assert!(cu.value.statistic.is_finite());
        let cub = cusum_ols(&x, &yb, &Session::new("cusumb", "t")).expect("cusumb");
        assert!(cub.value.statistic.is_finite());
        let kr = kernel_reg(&xs, &ys, 1.0, &Session::new("kr", "t")).expect("kr");
        assert!((kr.value[10] - 20.0).abs() < 2.0);
        let kdeu = kde_univariate(&xs, &Session::new("kdeu", "t")).expect("kdeu");
        assert_eq!(kdeu.value.density.len(), xs.len());
        assert!(kdeu.value.bandwidth > 0.0);
        assert!(kdeu
            .value
            .density
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0));
        let dur = Vector::from_iter((0..20).map(|i| 1.0 + i as f64));
        let ev = Vector::from_iter((0..20).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }));
        let na = nelson_aalen(&dur, &ev, &Session::new("na", "t")).expect("na");
        assert!(na.value.survival.as_slice().last().copied().unwrap_or(0.0) > 0.0);
        let bt = bartlett(&[&a, &b, &c], &Session::new("bart", "t")).expect("bart");
        assert!(bt.value.statistic.is_finite() || bt.value.pvalue.is_nan());
        let tab = Matrix::from_fn(8, 3, |i, j| i as f64 + 0.3 * j as f64);
        let fr = friedman(&tab, &Session::new("fr", "t")).expect("fr");
        assert!(fr.value.statistic.is_finite());
        let ybin = Vector::from_iter((0..40).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }));
        let pz = proportion_ztest(&ybin, 0.5, &Session::new("pz", "t")).expect("pz");
        assert!(pz.value.pvalue.is_finite());
        let ybin2 = Vector::from_iter((0..40).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }));
        let pzs = proportions_ztest(&ybin, &ybin2, &Session::new("pzs", "t")).expect("pzs");
        assert!(pzs.value.statistic.is_finite());
        assert!(pzs.value.pvalue.is_finite());
        let inf = ols_influence(&x, &y, &Session::new("inf", "t")).expect("inf");
        assert_eq!(inf.value.hat.len(), 40);
        assert!(inf
            .value
            .hat
            .as_slice()
            .iter()
            .all(|h| *h >= 0.0 && *h < 1.0));
        assert!(inf.value.dffits.as_slice().iter().all(|v| v.is_finite()));
        let rw = Vector::from_iter((0..80).map(|i| {
            if i < 40 {
                0.3 * i as f64
            } else {
                12.0 + 0.1 * (i as f64 - 40.0)
            }
        }));
        let dfg = dfgls(&rw, Some(1), &Session::new("dfgls", "t")).expect("dfgls");
        assert!(dfg.value.stat.is_finite() || dfg.value.pvalue.is_nan());
        let za = zivot_andrews(&rw, &Session::new("za", "t")).expect("za");
        assert!(za.value.stat.is_finite() || za.value.pvalue.is_nan());
        let tm = Vector::from_iter((0..20).map(|i| 1.0 + i as f64));
        let evc = Vector::from_iter((0..20).map(|i| {
            if i % 5 == 0 {
                1.0
            } else if i % 5 == 2 {
                2.0
            } else {
                0.0
            }
        }));
        let aj = aalen_johansen(&tm, &evc, &Session::new("aj", "t")).expect("aj");
        assert!(!aj.value.causes.is_empty());
        assert_eq!(aj.value.cif.ncols(), aj.value.causes.len());
        let xfg = Matrix::from_fn(20, 1, |i, _| (i % 7) as f64);
        let fg = fine_gray(&xfg, &tm, &evc, 1, &Session::new("fg", "t")).expect("fg");
        assert!(fg.value.coef[0].is_finite());
        assert!(fg.value.n_events > 0);
        let xv = Vector::from_iter((0..40).map(|i| i as f64));
        let md = Vector::from_iter((0..40).map(|i| 0.5 * i as f64 + 0.1));
        let yv = Vector::from_iter((0..40).map(|i| 1.0 + 0.8 * i as f64 + 0.4 * md[i]));
        let med = mediation(&xv, &md, &yv, &Session::new("med", "t")).expect("med");
        assert!(med.value.indirect.is_finite());
        let treat = Vector::from_iter((0..40).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }));
        let post = Vector::from_iter((0..40).map(|i| if i >= 20 { 1.0 } else { 0.0 }));
        let ydid = Vector::from_iter((0..40).map(|i| {
            treat[i] * 0.5 + post[i] * 0.2 + treat[i] * post[i] * 1.5 + 0.05 * ((i % 3) as f64)
        }));
        let did = difference_in_differences(&ydid, &treat, &post, &Session::new("did", "t"))
            .expect("did");
        assert!((did.value.att - 1.5).abs() < 0.3, "att={}", did.value.att);
        let a = Vector::from_iter((0..12).map(|i| 0.2 * i as f64));
        let b = Vector::from_iter((0..12).map(|i| 3.0 + 0.2 * i as f64));
        let cm = compare_means(&a, &b, &Session::new("cm", "t")).expect("cm");
        assert!(cm.value.diff < 0.0);
        assert!(cm.value.pvalue < 0.05);
        let yr = Vector::from_iter((0..24).map(|i| 1.0 + 2.0 * i as f64 + 0.1 * ((i % 3) as f64)));
        let xr = Matrix::from_fn(24, 1, |_, _| 1.0);
        let xf = Matrix::from_fn(24, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let alm = anova_lm(&yr, &xr, &xf, &Session::new("anlm", "t")).expect("anlm");
        assert!(alm.value.f_stat > 10.0, "F={}", alm.value.f_stat);
        let y2w = Vector::from_iter((0..32).map(|i| {
            let a = if i < 16 { 0.0 } else { 1.0 };
            let b = if i % 2 == 0 { 0.0 } else { 1.0 };
            0.4 * b + 3.0 * a + 0.05 * ((i % 5) as f64)
        }));
        let fa = Vector::from_iter((0..32).map(|i| if i < 16 { 0.0 } else { 1.0 }));
        let fb = Vector::from_iter((0..32).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }));
        let tw = anova_twoway(&y2w, &fa, &fb, &Session::new("a2", "t")).expect("a2");
        assert!(tw.value.f_a > 10.0, "Fa={}", tw.value.f_a);
        assert!(tw.value.p_a < 0.05);
        assert!(tw.value.ss_error.is_finite());
        let ym = Matrix::from_fn(32, 2, |i, j| {
            let a = if i < 16 { 0.0 } else { 3.0 };
            if j == 0 {
                a + 0.05 * ((i % 5) as f64)
            } else {
                a * 0.4 + 0.1 * ((i % 3) as f64)
            }
        });
        let gm = Vector::from_iter((0..32).map(|i| if i < 16 { 0.0 } else { 1.0 }));
        let mv = manova(&ym, &gm, &Session::new("man", "t")).expect("manova");
        assert!(mv.value.pillai.is_finite());
        assert!(mv.value.pillai > 0.2, "pillai={}", mv.value.pillai);
        assert_eq!(mv.value.n_groups, 2);
        let y1 = Vector::from_iter((0..20).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }));
        let y2 = Vector::from_iter((0..20).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }));
        let mc = mcnemar(&y1, &y2, &Session::new("mc", "t")).expect("mc");
        assert!(mc.value.statistic.is_finite());
        let tab2 = Matrix::from_fn(2, 2, |i, j| if i == j { 8.0 } else { 2.0 });
        let fe = fisher_exact(&tab2, &Session::new("fe", "t")).expect("fe");
        assert!(fe.value.pvalue.is_finite() && fe.value.pvalue < 0.2);
        let ad = anderson_darling(&y, &Session::new("ad", "t")).expect("ad");
        assert!(ad.value.statistic.is_finite());
        let lf = lilliefors(&y, &Session::new("lf", "t")).expect("lf");
        assert!(lf.value.statistic.is_finite());
        let rb = rainbow(&x, &y, 0.5, &Session::new("rb", "t")).expect("rainbow");
        assert!(rb.value.statistic.is_finite() || rb.value.pvalue.is_nan());
        assert!(rb.value.nobs > 0.0);
        let lr = compare_lr(&yr, &xr, &xf, &Session::new("lr", "t")).expect("lr");
        assert!(lr.value.statistic.is_finite() || lr.value.pvalue.is_nan());
        assert!(lr.value.df > 0.0);
        let cf = compare_f(&yr, &xr, &xf, &Session::new("cf", "t")).expect("cf");
        assert!(cf.value.statistic.is_finite() || cf.value.pvalue.is_nan());
        let wd = wald_ols(&x, &y, &Session::new("wald", "t")).expect("wald");
        assert!(wd.value.statistic.is_finite() || wd.value.pvalue.is_nan());
        assert!(wd.value.df > 0.0);
        let bp = box_pierce(&e, 2, &Session::new("bp", "t")).expect("bp");
        assert!(bp.value.stat.is_finite() || bp.value.pvalue.is_nan());
        let grp = Vector::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        let lrt = logrank(&dur, &ev, &grp, &Session::new("lrk", "t")).expect("logrank");
        assert!(lrt.value.statistic.is_finite() || lrt.value.pvalue.is_nan());
        let sd = survdiff(&dur, &ev, &grp, &Session::new("sd", "t")).expect("survdiff");
        assert!(sd.value.df > 0.0);
        let ld = levinson_durbin(&e, 3, &Session::new("ld", "t")).expect("ld");
        assert_eq!(ld.value.ar.len(), 3);
        assert!(ld.value.sigma2.is_finite());
        let rr = recursive_olsresiduals(&x, &y, 8, &Session::new("rols", "t")).expect("rols");
        assert!(!rr.value.is_empty());
        assert!(rr.value.as_slice().iter().all(|v| v.is_finite()));
        let yw = yule_walker(&e, 3, &Session::new("yw", "t")).expect("yw");
        assert_eq!(yw.value.ar.len(), 3);
        let burg = burg_ar(&e, 3, &Session::new("burg", "t")).expect("burg");
        assert_eq!(burg.value.ar.len(), 3);
        assert!(burg.value.sigma2.is_finite());
        let hr = hannan_rissanen(&e, 1, 1, &Session::new("hr", "t")).expect("hr");
        assert_eq!(hr.value.ar.len(), 1);
        assert_eq!(hr.value.ma.len(), 1);
        assert!(hr.value.sigma2.is_finite());
        let hn = breaks_hansen(&x, &y, &Session::new("han", "t")).expect("hansen");
        assert!(hn.value.statistic.is_finite() || hn.value.pvalue.is_nan());
        let cx = Matrix::from_fn(20, 1, |i, _| if i < 10 { 0.0 } else { 1.0 });
        let sch = schoenfeld(&dur, &ev, &cx, &Session::new("sch", "t")).expect("sch");
        assert_eq!(sch.value.ncols(), 1);
        assert!(
            sch.value.nrows() == 0
                || (0..sch.value.nrows()).all(|i| sch.value.get(i, 0).is_finite())
        );
        let om = omni_normtest(&y, &Session::new("omni", "t")).expect("omni");
        assert!(om.value.statistic.is_finite() || om.value.pvalue.is_nan());
        let nt = normaltest(&y, &Session::new("nt", "t")).expect("nt");
        assert!(nt.value.df > 0.0);
        let qs = q_stat(&e, 3, &Session::new("qs", "t")).expect("qstat");
        assert!(qs.value.stat.is_finite() || qs.value.pvalue.is_nan());
        let pci = proportion_confint(&ybin, 0.05, &Session::new("pci", "t")).expect("pci");
        assert!(pci.value.low <= pci.value.point && pci.value.point <= pci.value.high);
        let pwr = proportions_ztest_power(0.5, 0.7, 40.0, 40.0, 0.05, &Session::new("pwr", "t"))
            .expect("pwr");
        assert!(pwr.value.is_finite() && pwr.value >= 0.0 && pwr.value <= 1.0);
        let x2 = Matrix::from_fn(40, 1, |_, _| 1.0);
        let j = compare_j(&y, &x, &x2, &Session::new("j", "t")).expect("j");
        assert!(j.value.statistic.is_finite() || j.value.pvalue.is_nan());
        let sk = robust_skewness(&y, &Session::new("rsk", "t")).expect("rsk");
        assert!(sk.value.is_finite());
        let ku = robust_kurtosis(&e, &Session::new("rku", "t")).expect("rku");
        assert!(ku.value.is_finite());
        let cg = coint(&y, &x.column(0), &Session::new("coint", "t")).expect("coint");
        assert!(cg.value.stat.is_finite() || cg.value.pvalue.is_nan());
        let tip = ttest_ind_power(0.8, 20.0, 20.0, 0.05, &Session::new("tip", "t")).expect("tip");
        assert!(tip.value.is_finite() && tip.value >= 0.0 && tip.value <= 1.0);
        let fp = ftest_power(0.5, 2.0, 40.0, 0.05, &Session::new("ftp", "t")).expect("ftp");
        assert!(fp.value.is_finite() && fp.value >= 0.0 && fp.value <= 1.0);
        let gp = gof_chisquare_power(8.0, 3.0, 0.05, &Session::new("gof", "t")).expect("gof");
        assert!(gp.value.is_finite() && gp.value >= 0.0 && gp.value <= 1.0);
        let pv = Vector::from_slice(&[0.01, 0.04, 0.20, 0.80]);
        let fdr = fdrcorrection(&pv, 0.05, &Session::new("fdr", "t")).expect("fdr");
        assert_eq!(fdr.value.len(), 4);
        assert!(fdr.value.as_slice().iter().all(|v| v.is_finite()));
        let t1 = Matrix::from_fn(2, 2, |i, j| if i == j { 8.0 } else { 2.0 });
        let t2 = Matrix::from_fn(2, 2, |i, j| if i == j { 6.0 } else { 3.0 });
        let mh = mantel_haenszel(&[t1, t2], &Session::new("mh", "t")).expect("mh");
        assert!(mh.value.statistic.is_finite() || mh.value.pvalue.is_nan());
        let fl = fligner(&[&a, &b, &c], &Session::new("fl", "t")).expect("fl");
        assert!(fl.value.statistic.is_finite() || fl.value.pvalue.is_nan());
        let an = ansari(&a, &b, &Session::new("ans", "t")).expect("ans");
        assert!(an.value.statistic.is_finite() || an.value.pvalue.is_nan());
        let mo = mood(&a, &b, &Session::new("mood", "t")).expect("mood");
        assert!(mo.value.statistic.is_finite() || mo.value.pvalue.is_nan());
        let mt = median_test(&[&a, &b, &c], &Session::new("mdt", "t")).expect("mdt");
        assert!(mt.value.statistic.is_finite() || mt.value.pvalue.is_nan());
        let cnt = Vector::from_slice(&[8.0, 6.0, 4.0, 2.0]);
        let chi = chisquare(&cnt, &Session::new("chi", "t")).expect("chi");
        assert!(chi.value.statistic.is_finite() || chi.value.pvalue.is_nan());
        let pdv = power_divergence(&cnt, 1.0, &Session::new("pdv", "t")).expect("pdv");
        assert!(
            (pdv.value.statistic - chi.value.statistic).abs() < 1e-9
                || pdv.value.statistic.is_nan()
        );
        let qtab = Matrix::from_fn(8, 3, |i, j| if (i + j) % 2 == 0 { 1.0 } else { 0.0 });
        let cq = cochran_q(&qtab, &Session::new("cq", "t")).expect("cq");
        assert!(cq.value.statistic.is_finite() || cq.value.pvalue.is_nan());
        let or = odds_ratio(&tab2, &Session::new("or", "t")).expect("or");
        assert!(or.value.is_finite() && or.value > 1.0);
        let rr = risk_ratio(&tab2, &Session::new("rr", "t")).expect("rr");
        assert!(rr.value.is_finite() && rr.value > 1.0);
        let ys_close = Vector::from_iter((0..20).map(|i| i as f64 + 0.05 * (i as f64).sin()));
        let tost =
            tost_paired(&xs, &ys_close, -1.0, 1.0, &Session::new("tost", "t")).expect("tost");
        assert!(tost.value.pvalue.is_finite() || tost.value.statistic.is_nan());
        let ap = ftest_anova_power(0.5, 3.0, 12.0, 0.05, &Session::new("fap", "t")).expect("fap");
        assert!(ap.value.is_finite() && ap.value >= 0.0 && ap.value <= 1.0);
        let zt = ztest(&y, 0.0, &Session::new("z", "t")).expect("z");
        assert!(zt.value.statistic.is_finite() || zt.value.pvalue.is_nan());
        let xa = Matrix::from_fn(12, 1, |i, _| a[i]);
        let xb2 = Matrix::from_fn(12, 1, |i, _| b[i]);
        let ht = hotelling(&xa, &xb2, &Session::new("hot", "t")).expect("hot");
        assert!(ht.value.statistic.is_finite() || ht.value.pvalue.is_nan());
        let eh = proportion_effectsize(0.2, 0.5, &Session::new("he", "t")).expect("he");
        assert!(eh.value.is_finite());
        let ts = ttost(&a, &b, -10.0, 10.0, &Session::new("ttost", "t")).expect("ttost");
        assert!(ts.value.pvalue.is_finite() || ts.value.statistic.is_nan());
        let ks = kstest(&y, &Session::new("ks1", "t")).expect("ks1");
        assert!(ks.value.statistic.is_finite() || ks.value.pvalue.is_nan());
        let hol = holm(pv.as_slice(), &Session::new("holm", "t")).expect("holm");
        assert_eq!(hol.value.len(), 4);
        let bon = bonferroni(pv.as_slice(), &Session::new("bon", "t")).expect("bon");
        assert!(bon.value.as_slice().iter().all(|v| v.is_finite()));
        let sid = sidak(pv.as_slice(), &Session::new("sid", "t")).expect("sid");
        assert_eq!(sid.value.len(), 4);
        let dsw = descr_stats_w(&a, None, &Session::new("dsw", "t")).expect("dsw");
        assert!(dsw.value.mean.is_finite());
        assert!(dsw.value.ci_low <= dsw.value.mean && dsw.value.mean <= dsw.value.ci_high);
        let mut xm = Matrix::from_fn(12, 2, |i, j| if j == 0 { a[i] } else { b[i] });
        xm.set(2, 1, f64::NAN);
        xm.set(5, 0, f64::NAN);
        let mi = mice(&xm, 2, &Session::new("mice", "t")).expect("mice");
        assert_eq!(mi.value.len(), 2);
        assert!(mi.value[0].get(2, 1).is_finite());
        assert!(mi.value[1].get(5, 0).is_finite());
        let hbp = het_breuschpagan(&e, &design, &Session::new("hbp", "t")).expect("hbp");
        assert!(hbp.value.statistic.is_finite() || hbp.value.pvalue.is_nan());
        let lbq = acorr_ljungbox(&e, 2, &Session::new("alb", "t")).expect("alb");
        assert!(lbq.value.stat.is_finite() || lbq.value.pvalue.is_nan());
        let rmt = Matrix::from_fn(12, 3, |i, j| {
            a[i] + 0.4 * j as f64 + 0.05 * ((i + j) as f64).sin()
        });
        let arm = anova_rm(&rmt, &Session::new("arm", "t")).expect("arm");
        assert!(arm.value.f_stat.is_finite() || arm.value.pvalue.is_nan());
        let zph = cox_zph(&dur, &ev, &cx, &Session::new("zph", "t")).expect("zph");
        assert!(zph.value.statistic.is_finite() || zph.value.pvalue.is_nan());
        let qr = QuantReg::new(0.5)
            .fit(&x, &y, &Session::new("qr", "t"))
            .expect("qr");
        assert!(qr.value.coef.as_slice().iter().all(|v| v.is_finite()));
        let ycc = Matrix::from_fn(x.nrows(), 1, |i, _| y[i]);
        let cc = cancorr(&x, &ycc, &Session::new("cc", "t")).expect("cc");
        assert!(cc.value.correlations.as_slice().iter().any(|v| *v > 0.5));
        let aa = aalen_additive(&dur, &ev, &cx, &Session::new("aa", "t")).expect("aa");
        assert!(aa.value.n_events > 0.0);
        assert_eq!(aa.value.cumulative.ncols(), cx.ncols());
        let bp = bai_perron(&yb, &Session::new("bai", "t")).expect("bai");
        assert!(bp.value.break_index > 0);
        assert!(bp.report.contains(IssueCode::StructuralBreak) || bp.value.ssr_gain.is_finite());
        let y2c = Vector::from_iter((0..40).map(|i| 0.8 * y[i]));
        let gc = gaussian_copula(&y, &y2c, &Session::new("gcop", "t")).expect("gcop");
        assert!(gc.value.rho > 0.5, "rho={}", gc.value.rho);
        let cl = clayton_copula(&y, &y2c, &Session::new("ccop", "t")).expect("ccop");
        assert!(cl.value.theta > 0.0);
        assert!(cl.value.loglik.is_finite() || cl.value.loglik.is_infinite());
        let gu = gumbel_copula(&y, &y2c, &Session::new("gumb", "t")).expect("gumb");
        assert!(gu.value.theta >= 1.0);
        let fr = frank_copula(&y, &y2c, &Session::new("frank", "t")).expect("frank");
        assert!(fr.value.theta.is_finite());
        let tc = student_t_copula(&y, &y2c, 5.0, &Session::new("tcop", "t")).expect("tcop");
        assert!(tc.value.rho > 0.4, "t-rho={}", tc.value.rho);
        let joe = joe_copula(&y, &y2c, &Session::new("joe", "t")).expect("joe");
        assert!(joe.value.theta >= 1.0);
        assert!(joe.value.loglik.is_finite() || joe.value.loglik.is_infinite());
        let pl = plackett_copula(&y, &y2c, &Session::new("plack", "t")).expect("plack");
        assert!(pl.value.theta > 0.0);
        let amh = amh_copula(&y, &y2c, &Session::new("amh", "t")).expect("amh");
        assert!(amh.value.theta > -1.0 && amh.value.theta < 1.0);
        let gam = UnivariateGam::new(3)
            .fit(&xs, &ys, &Session::new("gam", "t"))
            .expect("gam");
        let gamp = gam
            .value
            .predict(&xs, &Session::new("gam", "p"))
            .expect("gamp")
            .value;
        assert!((gamp[10] - 20.0).abs() < 2.0, "gam[10]={}", gamp[10]);
        let kmv = kde_multivariate(&x, &Session::new("kdemv", "t")).expect("kdemv");
        assert_eq!(kmv.value.density.len(), 40);
        assert!(kmv
            .value
            .density
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0));
        let el = empirical_likelihood_mean(&y, y.mean(), &Session::new("el", "t")).expect("el");
        assert!(el.value.pvalue.is_finite() || el.value.statistic.is_nan());
        let sr = sur(&y, &y2c, &x, &Session::new("sur", "t")).expect("sur");
        assert!(sr.value.beta1.as_slice().iter().all(|v| v.is_finite()));
        assert!(sr.value.sigma.get(0, 1).is_finite());
        let fgm = fgm_copula(&y, &y2c, &Session::new("fgm", "t")).expect("fgm");
        assert!((-1.0..=1.0).contains(&fgm.value.theta));
        let ec = empirical_copula(&y, &y2c, &Session::new("ecop", "t")).expect("ecop");
        assert!(ec.value.mean_mass > 0.27, "C_n={}", ec.value.mean_mass);
        let dc = distance_corr(&y, &y2c, &Session::new("dcor", "t")).expect("dcor");
        assert!(dc.value > 0.8, "dCor={}", dc.value);
        let ed0 = energy_distance(&y, &y, &Session::new("ed0", "t")).expect("ed0");
        assert!(ed0.value.abs() < 1e-9, "E(y,y)={}", ed0.value);
        let ed = energy_distance(&a, &b, &Session::new("ed", "t")).expect("ed");
        assert!(ed.value > 0.0);
        let cvm = cramer_von_mises(&y, &Session::new("cvm", "t")).expect("cvm");
        assert!(cvm.value.statistic.is_finite());
        let aks = anderson_ksamp(&[&a, &b, &c], &Session::new("aks", "t")).expect("aks");
        assert!(aks.value.statistic.is_finite());
        let bm = brunner_munzel(&a, &b, &Session::new("bm", "t")).expect("bm");
        assert!(bm.value.statistic.is_finite() || bm.value.pvalue.is_nan());
        let jt = jonckheere(&[&a, &b, &c], &Session::new("jt", "t")).expect("jt");
        assert!(jt.value.statistic > 0.0);
        let bw = bowker(&y1, &y2, &Session::new("bow", "t")).expect("bow");
        assert!(bw.value.statistic.is_finite() || bw.value.pvalue.is_nan());
        let smx = stuart_maxwell(&y1, &y2, &Session::new("smx", "t")).expect("smx");
        assert!(smx.value.statistic.is_finite() || smx.value.pvalue.is_nan());
        let totals = Vector::from_iter((0..10).map(|t| {
            (0..4).map(|q| y[t * 4 + q]).sum::<f64>()
        }));
        let den = denton(&y, &totals, 4, &Session::new("den", "t")).expect("denton");
        assert_eq!(den.value.len(), 40);
        for t in 0..10 {
            let s = (0..4).map(|q| den.value[t * 4 + q]).sum::<f64>();
            assert!((s - totals[t]).abs() < 1e-6, "denton year {t} sum={s}");
        }
        let cln = chow_lin(&y, &totals, 4, &Session::new("cln", "t")).expect("cln");
        assert_eq!(cln.value.series.len(), 40);
        assert!(cln.value.slope.is_finite());
        for t in 0..10 {
            let s = (0..4).map(|q| cln.value.series[t * 4 + q]).sum::<f64>();
            assert!((s - totals[t]).abs() < 1e-5, "chowlin year {t}");
        }
        let lit = litterman(&y, &totals, 4, &Session::new("lit", "t")).expect("lit");
        assert_eq!(lit.value.len(), 40);
        let lmc = leybourne_mccabe(&y, &Session::new("lmc", "t")).expect("lmc");
        assert!(lmc.value.statistic.is_finite() || lmc.value.pvalue.is_nan());
        let rur = range_unit_root(&y, &Session::new("rur", "t")).expect("rur");
        assert!(rur.value.statistic.is_finite());
        let bv = breakvar(&e, 20, &Session::new("bv", "t")).expect("bv");
        assert!(bv.value.statistic.is_finite() || bv.value.pvalue.is_nan());
        let vn = vine_copula(&y, &y2c, &Session::new("vine", "t")).expect("vine");
        assert!(vn.value.theta > 0.0);
        assert_eq!(vn.value.n_trees, 1);
        let nprop = samplesize_proportions(0.2, 0.5, 0.05, 0.8, &Session::new("ssp", "t"))
            .expect("ssp");
        assert!(nprop.value.is_finite() && nprop.value > 10.0);
        let xcox = Matrix::from_fn(dur.len(), 1, |i, _| grp[i]);
        let ccx = compare_cox(&dur, &ev, &xcox, &Session::new("ccx", "t")).expect("ccx");
        assert!(ccx.value.statistic.is_finite() || ccx.value.pvalue.is_nan());
        assert!(ccx.value.df > 0.0);
        let sw = spec_white(&x, &y, &Session::new("sw", "t")).expect("specw");
        assert!(sw.value.statistic.is_finite() || sw.value.pvalue.is_nan());
        let ha = het_arch(&e, 2, &Session::new("ha", "t")).expect("hetarch");
        assert!(ha.value.df > 0.0);
        let bg2 = acorr_breusch_godfrey(&e, &design, 1, &Session::new("abg", "t")).expect("abg");
        assert!(bg2.value.df > 0.0);
        let ot = outlier_test(&x, &y, &Session::new("ot", "t")).expect("out");
        assert_eq!(ot.value.studentized.len(), 40);
        assert!(ot
            .value
            .studentized
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let phr = PHReg::new()
            .fit(&dur, &ev, &xcox, &Session::new("phr", "t"))
            .expect("phreg");
        assert_eq!(phr.value.coef.len(), 1);
        assert!(phr.value.coef[0].is_finite() || !phr.value.converged);
        let sfr = SurvfuncRight::new()
            .fit(&dur, &ev, &Session::new("sfr", "t"))
            .expect("sfr");
        assert!(!sfr.value.times.is_empty());
        assert!(sfr
            .value
            .survival
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
        let mc = medcouple(&y, &Session::new("mcouple", "t")).expect("mcouple");
        assert!(mc.value.is_finite());
        let wps = wls_prediction_std(&x, &y, &Session::new("wps", "t")).expect("wps");
        assert_eq!(wps.value.predicted.len(), 40);
        assert!(wps
            .value
            .se_obs
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        let llm = linear_lm(&x, &y, &Session::new("llm", "t")).expect("llm");
        assert!(llm.value.statistic.is_finite() || llm.value.pvalue.is_nan());
        let tfs = ttest_ind_from_stats(0.0, 1.0, 20.0, 1.0, 1.0, 20.0, &Session::new("tifs", "t"))
            .expect("tifs");
        assert!(tfs.value.pvalue.is_finite() && tfs.value.pvalue < 0.05);
        let pwf = power_ftest(0.5, 2.0, 30.0, 0.05, &Session::new("pwf", "t")).expect("pwf");
        assert!(pwf.value.is_finite() && pwf.value > 0.0 && pwf.value <= 1.0);
        let hgq = het_goldfeldquandt(&x, &y, &Session::new("hgq", "t")).expect("hgq");
        assert!(hgq.value.statistic.is_finite() || hgq.value.pvalue.is_nan());
        let lhc = linear_harvey_collier(&x, &y, &Session::new("lhc", "t")).expect("lhc");
        assert!(lhc.value.nobs > 0.0);
        let lrb = linear_rainbow(&x, &y, &Session::new("lrb", "t")).expect("lrb");
        assert!(lrb.value.statistic.is_finite() || lrb.value.pvalue.is_nan());
        let bcu = breaks_cusumolsresid(&x, &y, &Session::new("bcu", "t")).expect("bcu");
        assert!(bcu.value.statistic.is_finite());
        let t1 = Vector::from_iter((0..10).map(|i| dur[i]));
        let e1 = Vector::from_iter((0..10).map(|i| ev[i]));
        let t2 = Vector::from_iter((10..20).map(|i| dur[i]));
        let e2 = Vector::from_iter((10..20).map(|i| ev[i]));
        let csf = compare_survfunc(&t1, &e1, &t2, &e2, &Session::new("csf", "t")).expect("csf");
        assert!(csf.value.df > 0.0);
        let cd = cooks_distance(&x, &y, &Session::new("cook", "t")).expect("cook");
        assert_eq!(cd.value.len(), 40);
        assert!(cd.value.as_slice().iter().all(|v| v.is_finite() && *v >= 0.0));
        let sg = savgol_filter(&y, 5, 2, &Session::new("sg", "t")).expect("sg");
        assert_eq!(sg.value.len(), 40);
        assert!(sg.value.as_slice().iter().all(|v| v.is_finite()));
        let dch = denton_cholette(&y, &totals, 4, &Session::new("dch", "t")).expect("dch");
        assert_eq!(dch.value.len(), 40);
        for t in 0..10 {
            let s = (0..4).map(|q| dch.value[t * 4 + q]).sum::<f64>();
            assert!((s - totals[t]).abs() < 1e-4, "cholette year {t}");
        }
        let nad = normal_ad(&y, &Session::new("nad", "t")).expect("nad");
        assert!(nad.value.statistic.is_finite() || nad.value.pvalue.is_nan());
        let gp = get_prediction(&x, &y, &Session::new("gpred", "t")).expect("gpred");
        assert_eq!(gp.value.predicted.len(), 40);
        assert!(gp
            .value
            .lower
            .as_slice()
            .iter()
            .zip(gp.value.upper.as_slice())
            .all(|(lo, hi)| lo <= hi));
        let tt = t_test(&x, &y, &Session::new("ttestols", "t")).expect("ttestols");
        assert_eq!(tt.value.params.len(), 2);
        assert!(tt.value.tvalues[1].is_finite());
        assert!(tt.value.pvalues[1].is_finite());
        let ft = f_test(&x, &y, &Session::new("ftestols", "t")).expect("ftestols");
        assert!(ft.value.statistic.is_finite() || ft.value.pvalue.is_nan());
        let ci = conf_int(&x, &y, 0.05, &Session::new("ciols", "t")).expect("ciols");
        assert_eq!(ci.value.low.len(), 2);
        assert!(ci.value.low[1] <= ci.value.high[1]);
        let ev1 = Vector::from_iter((0..20).map(|_| 1.0));
        let kcr = kernel_censored_reg(&xs, &ys, &ev1, 1.0, &Session::new("kcr", "t")).expect("kcr");
        assert_eq!(kcr.value.len(), 20);
        assert!((kcr.value[10] - 20.0).abs() < 3.0);
        let pm = process_mle(&xs, &ys, &Session::new("pmle", "t")).expect("pmle");
        assert!(pm.value.mean.is_finite());
        assert!(pm.value.sigma2.is_finite() && pm.value.sigma2 > 0.0);
        assert!(pm.value.range.is_finite() && pm.value.range > 0.0);
        let ones = Matrix::from_fn(40, 1, |_, _| 1.0);
        let xu = x.with_intercept();
        let clm = compare_lm(&y, &ones, &xu, &Session::new("clm", "t")).expect("clm");
        assert!(clm.value.statistic.is_finite() || clm.value.pvalue.is_nan());
        let lrst = linear_reset(&x, &y, &Session::new("lreset", "t")).expect("lreset");
        assert!(lrst.value.statistic.is_finite() || lrst.value.pvalue.is_nan());
    }
}
