//! Statsmodels-style descriptive statistics, association, tests, and survival.
//!
//! Every public computation opens a [`crate::context::FitCtx`], inspects the
//! inputs, and records [`signlred`] issues for non-finite values, insufficient
//! sample size, vacuous constant series, non-positive degrees of freedom, and
//! numerical compromises. A silent successful call is a contract violation.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::chol_solve;
use crate::rng::Rng;
use crate::special::{chi2_pvalue, f_pvalue, ln_gamma, norm_cdf, student_t_cdf, student_t_pvalue};
use crate::validate::{inspect_identification, inspect_xy};
use ojizou_san::Session;
use signlred::{
    scan_finite, slice_stats, Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified,
    Report, Result,
};

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
}
