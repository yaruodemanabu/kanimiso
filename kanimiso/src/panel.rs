//! Panel estimators: within (fixed effects), between, and first-difference OLS.
//!
//! Group identifiers are a [`Vector`] of numeric codes. The within estimand
//! subtracts group means; first differences use consecutive rows that share a
//! group (callers should sort by group then time). A handful of groups is a
//! warning, not [`IssueCode::InsufficientSample`] as an error — the same
//! contract as [`crate::mixed::MixedLM`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::least_squares;
use crate::special::chi2_pvalue;
use crate::stats::HypothesisTest;
use crate::traits::Predict;
use crate::validate::inspect_xy;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result, Severity};
use std::collections::BTreeMap;

fn group_sizes(groups: &Vector) -> BTreeMap<i64, usize> {
    let mut sizes = BTreeMap::new();
    for &g in groups.as_slice() {
        if g.is_finite() {
            *sizes.entry(g.round() as i64).or_insert(0) += 1;
        }
    }
    sizes
}

fn warn_panel_groups(ctx: &mut FitCtx, sizes: &BTreeMap<i64, usize>, what: &str) {
    let n_groups = sizes.len();
    if n_groups <= 1 {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message(format!(
                    "{what}: a single group cannot identify a panel estimand"
                ))
                .meaninglessness(Meaninglessness::vacuous(
                    "panel coefficients",
                    "within / between / first-difference contrasts need at least two groups",
                    "use pooled OLS, or collect more groups",
                ))
                .build(),
        );
    } else if n_groups < 5 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!("{n_groups} groups is a thin panel for {what}"))
                .metric("n_groups", n_groups as f64)
                .build(),
        );
    }
    let n_singletons = sizes.values().filter(|&&c| c <= 1).count();
    if n_groups > 0 && n_singletons == n_groups {
        ctx.push(
            Issue::builder(IssueCode::IncrementalUnidentifiable)
                .message(format!("{what}: every group has size 1"))
                .meaninglessness(Meaninglessness::vacuous(
                    "within / first-difference slopes",
                    "no repeated measures; group demeaning and differencing are empty",
                    "need groups with size ≥ 2",
                ))
                .build(),
        );
    }
}

fn group_means(
    x: &Matrix,
    y: &Vector,
    groups: &Vector,
    sizes: &BTreeMap<i64, usize>,
) -> (BTreeMap<i64, Vector>, BTreeMap<i64, f64>) {
    let mut sum_x: BTreeMap<i64, Vector> = BTreeMap::new();
    let mut sum_y: BTreeMap<i64, f64> = BTreeMap::new();
    for i in 0..y.len().min(x.nrows()).min(groups.len()) {
        if !groups[i].is_finite() {
            continue;
        }
        let g = groups[i].round() as i64;
        let entry = sum_x.entry(g).or_insert_with(|| Vector::zeros(x.ncols()));
        for j in 0..x.ncols() {
            entry[j] += x.get(i, j);
        }
        *sum_y.entry(g).or_insert(0.0) += y[i];
    }
    let mut mx = BTreeMap::new();
    let mut my = BTreeMap::new();
    for (&g, &c) in sizes {
        let cf = c.max(1) as f64;
        if let Some(sx) = sum_x.get(&g) {
            mx.insert(g, sx.scale(1.0 / cf));
        }
        if let Some(&sy) = sum_y.get(&g) {
            my.insert(g, sy / cf);
        }
    }
    (mx, my)
}

fn empty_panel(p: usize) -> FittedPanel {
    FittedPanel {
        coef: Vector::zeros(p),
        intercept: 0.0,
        n_groups: 0,
        n_eff: 0,
    }
}

/// Fitted panel slopes (within, between, or first-difference).
#[derive(Clone, Debug)]
pub struct FittedPanel {
    /// Slopes on the transformed design.
    pub coef: Vector,
    /// Intercept (0 for within / first-difference).
    pub intercept: f64,
    /// Number of groups that contributed.
    pub n_groups: usize,
    /// Rows used in the transformed OLS.
    pub n_eff: usize,
}

impl Predict for FittedPanel {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("panel predict column count ≠ coef")
                    .build(),
            );
        }
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

/// Within (fixed-effects) OLS: \(y_{it}-\bar y_i\) on \(X_{it}-\bar X_i\).
#[derive(Clone, Debug, Default)]
pub struct PanelFe;

impl PanelFe {
    /// Default within estimator.
    pub fn new() -> Self {
        Self
    }

    /// Fit `y | groups` after group demeaning. No intercept is identified.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if groups.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("PanelFe groups length ≠ n")
                    .build(),
            );
            return ctx.finish(empty_panel(x.ncols()));
        }
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "PanelFe");
        if ctx.report.contains(IssueCode::UnidentifiedModel)
            || ctx.report.contains(IssueCode::IncrementalUnidentifiable)
        {
            return ctx.finish(empty_panel(x.ncols()));
        }
        let (mx, my) = group_means(x, y, groups, &sizes);
        let n = y.len().min(x.nrows());
        let mut xw = Matrix::zeros(n, x.ncols());
        let mut yw = Vector::zeros(n);
        for i in 0..n {
            let g = groups[i].round() as i64;
            let gx = mx.get(&g);
            let gy = my.get(&g).copied().unwrap_or(0.0);
            yw[i] = y[i] - gy;
            for j in 0..x.ncols() {
                let m = gx.map(|v| v[j]).unwrap_or(0.0);
                xw.set(i, j, x.get(i, j) - m);
            }
        }
        if yw.std() <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("within y has ~0 variance; the FE estimand is empty")
                    .meaninglessness(Meaninglessness::vacuous(
                        "within slopes",
                        "after demeaning there is no leftover variation in y",
                        "the regressor is time-invariant, or groups have no within movement",
                    ))
                    .build(),
            );
            return ctx.finish(empty_panel(x.ncols()));
        }
        let mut scratch = signlred::Report::new("panel_fe", "ols");
        let Some(coef) = least_squares(&mut scratch, &xw, &yw, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("within OLS failed")
                    .build(),
            );
            return ctx.finish(empty_panel(x.ncols()));
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
        ctx.finish(FittedPanel {
            coef,
            intercept: 0.0,
            n_groups: sizes.len(),
            n_eff: n,
        })
    }
}

/// Between OLS: group means \(\bar y_g\) on \(\bar X_g\).
#[derive(Clone, Debug, Default)]
pub struct BetweenOls {
    /// Include an intercept on the group-mean regression.
    pub fit_intercept: bool,
}

impl BetweenOls {
    /// Intercept-on between estimator.
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
        }
    }

    /// Fit on the group-mean cross-section.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if groups.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("BetweenOLS groups length ≠ n")
                    .build(),
            );
            return ctx.finish(empty_panel(x.ncols()));
        }
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "BetweenOLS");
        if sizes.len() <= 1 {
            return ctx.finish(empty_panel(x.ncols()));
        }
        let (mx, my) = group_means(x, y, groups, &sizes);
        let keys: Vec<i64> = mx.keys().copied().collect();
        let g = keys.len();
        let xb = Matrix::from_fn(g, x.ncols(), |i, j| mx[&keys[i]][j]);
        let yb = Vector::from_iter(keys.iter().map(|k| my.get(k).copied().unwrap_or(0.0)));
        if yb.std() <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::ConstantTarget)
                    .message("between y is constant across groups")
                    .build(),
            );
            return ctx.finish(empty_panel(x.ncols()));
        }
        let design = if self.fit_intercept {
            xb.with_intercept()
        } else {
            xb
        };
        // Do not inspect_identification(n_groups, p): 4 groups would abort as Error.
        let mut scratch = signlred::Report::new("between", "ols");
        let Some(beta) = least_squares(&mut scratch, &design, &yb, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("between OLS failed")
                    .build(),
            );
            return ctx.finish(empty_panel(x.ncols()));
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
        let (intercept, coef) = if self.fit_intercept {
            (
                beta.as_slice().first().copied().unwrap_or(0.0),
                Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            )
        } else {
            (0.0, beta)
        };
        ctx.finish(FittedPanel {
            coef,
            intercept,
            n_groups: g,
            n_eff: g,
        })
    }
}

/// First-difference OLS: \(\Delta y\) on \(\Delta X\) within group (row order = time).
#[derive(Clone, Debug, Default)]
pub struct FirstDifferenceOls;

impl FirstDifferenceOls {
    /// Default first-difference estimator.
    pub fn new() -> Self {
        Self
    }

    /// Fit on consecutive within-group differences.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if groups.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("FirstDifferenceOLS groups length ≠ n")
                    .build(),
            );
            return ctx.finish(empty_panel(x.ncols()));
        }
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "FirstDifferenceOLS");
        let n = y.len().min(x.nrows()).min(groups.len());
        let mut xd = Vec::new();
        let mut yd = Vec::new();
        for i in 1..n {
            if !groups[i].is_finite() || !groups[i - 1].is_finite() {
                continue;
            }
            if groups[i].round() as i64 != groups[i - 1].round() as i64 {
                continue;
            }
            yd.push(y[i] - y[i - 1]);
            let mut row = vec![0.0; x.ncols()];
            for j in 0..x.ncols() {
                row[j] = x.get(i, j) - x.get(i - 1, j);
            }
            xd.push(row);
        }
        if xd.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("no consecutive within-group pairs to difference")
                    .meaninglessness(Meaninglessness::vacuous(
                        "first-difference slopes",
                        "the panel has no adjacent repeats in row order",
                        "sort by group then time, or collect longer panels",
                    ))
                    .build(),
            );
            return ctx.finish(empty_panel(x.ncols()));
        }
        let m = xd.len();
        let p = x.ncols();
        let xmat = Matrix::from_fn(m, p, |i, j| xd[i][j]);
        let yvec = Vector::from_iter(yd);
        let mut x_energy: f64 = 0.0;
        for j in 0..p {
            for i in 0..m {
                x_energy = x_energy.max(xmat.get(i, j).abs());
            }
        }
        // Constant nonzero ΔX (a linear time trend) still identifies β = Δy/ΔX.
        // The empty estimand is the zero map: X is time-invariant, so ΔX ≡ 0.
        if x_energy <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("ΔX is the zero map; first-difference slopes are unidentified")
                    .meaninglessness(Meaninglessness::vacuous(
                        "first-difference slopes",
                        "every within-group change in X is ~0 (time-invariant regressors)",
                        "need time-varying regressors, or a longer panel",
                    ))
                    .build(),
            );
            return ctx.finish(empty_panel(p));
        }
        if yvec.std() <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .severity(Severity::Warning)
                    .message("Δy is constant; first-difference OLS is a ratio of constants")
                    .build(),
            );
        }
        let mut scratch = signlred::Report::new("fdols", "ols");
        let Some(coef) = least_squares(&mut scratch, &xmat, &yvec, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("first-difference OLS failed")
                    .build(),
            );
            return ctx.finish(empty_panel(p));
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
        ctx.finish(FittedPanel {
            coef,
            intercept: 0.0,
            n_groups: sizes.len(),
            n_eff: m,
        })
    }
}

/// Pooled OLS: ignore the group structure except for a thin-panel warning.
#[derive(Clone, Debug, Default)]
pub struct PooledOls {
    /// Include an intercept.
    pub fit_intercept: bool,
}

impl PooledOls {
    /// Intercept-on pooled OLS.
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
        }
    }

    /// Fit pooled OLS of `y` on `X`.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "PooledOLS");
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        let mut scratch = signlred::Report::new("pooled", "ols");
        let Some(beta) = least_squares(&mut scratch, &design, y, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("pooled OLS failed")
                    .build(),
            );
            return ctx.finish(empty_panel(x.ncols()));
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
        let (intercept, coef) = if self.fit_intercept {
            (
                beta.as_slice().first().copied().unwrap_or(0.0),
                Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            )
        } else {
            (0.0, beta)
        };
        ctx.finish(FittedPanel {
            coef,
            intercept,
            n_groups: sizes.len(),
            n_eff: y.len(),
        })
    }
}

/// Swamy–Arora random-effects GLS (linearmodels `RandomEffects` lite).
///
/// `θ` is computed from within and between residual scales. A collapsed
/// `σ_α²` is a warning that the GLS is pooled OLS.
#[derive(Clone, Debug, Default)]
pub struct RandomEffects;

impl RandomEffects {
    /// Default RE estimator.
    pub fn new() -> Self {
        Self
    }

    /// Fit quasi-demeaned GLS.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "RandomEffects");
        if sizes.len() <= 1 {
            return ctx.finish(empty_panel(x.ncols()));
        }
        let fe = match PanelFe::new().fit(x, y, groups, &session.child("re-fe")) {
            Ok(q) => q.value,
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .severity(signlred::Severity::Warning)
                        .message("RE within step failed; falling back to pooled OLS")
                        .build(),
                );
                return PooledOls::new().fit(x, y, groups, session);
            }
        };
        let be = match BetweenOls::new().fit(x, y, groups, &session.child("re-be")) {
            Ok(q) => q.value,
            Err(_) => {
                return PooledOls::new().fit(x, y, groups, session);
            }
        };
        let n = y.len().min(x.nrows()).min(groups.len());
        let p = x.ncols();
        let n_g = sizes.len().max(1);
        let t_bar = n as f64 / n_g as f64;
        let (mx, my) = group_means(x, y, groups, &sizes);
        let mut sse_w = 0.0;
        for i in 0..n {
            let g = groups[i].round() as i64;
            let gy = my.get(&g).copied().unwrap_or(0.0);
            let mut fit = 0.0;
            for j in 0..p {
                let m = mx.get(&g).map(|v| v[j]).unwrap_or(0.0);
                fit += fe.coef[j] * (x.get(i, j) - m);
            }
            let e = (y[i] - gy) - fit;
            sse_w += e * e;
        }
        let df_w = (n as f64 - n_g as f64 - p as f64).max(1.0);
        let sig_e2 = (sse_w / df_w).max(1e-12);
        let mut sse_b = 0.0;
        for (&g, gx) in &mx {
            let gy = my.get(&g).copied().unwrap_or(0.0);
            let mut fit = be.intercept;
            for j in 0..p {
                fit += be.coef[j] * gx[j];
            }
            let e = gy - fit;
            sse_b += e * e;
        }
        let df_b = (n_g as f64 - p as f64 - 1.0).max(1.0);
        let sig_b2 = sse_b / df_b;
        let sig_a2 = (sig_b2 - sig_e2 / t_bar.max(1.0)).max(0.0);
        if sig_a2 <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .message("RE between variance collapsed; θ≈0 (pooled)")
                    .build(),
            );
        }
        let mut xq = Matrix::zeros(n, p);
        let mut yq = Vector::zeros(n);
        let mut theta_mean = 0.0;
        for i in 0..n {
            let g = groups[i].round() as i64;
            let t_i = *sizes.get(&g).unwrap_or(&1) as f64;
            let theta = 1.0 - (sig_e2 / (t_i * sig_a2 + sig_e2)).sqrt();
            theta_mean += theta;
            let gy = my.get(&g).copied().unwrap_or(0.0);
            yq[i] = y[i] - theta * gy;
            for j in 0..p {
                let m = mx.get(&g).map(|v| v[j]).unwrap_or(0.0);
                xq.set(i, j, x.get(i, j) - theta * m);
            }
        }
        let _ = theta_mean / n as f64;
        let design = xq.with_intercept();
        let mut scratch = signlred::Report::new("re", "gls");
        let Some(beta) = least_squares(&mut scratch, &design, &yq, &ctx.policy) else {
            return PooledOls::new().fit(x, y, groups, session);
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
        ctx.finish(FittedPanel {
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            n_groups: n_g,
            n_eff: n,
        })
    }
}

/// Hausman specification test: FE vs RE slopes.
///
/// Covariance of the difference is approximated by a residual-scale
/// diagonal; that is recorded as a compromise. A large statistic is
/// [`IssueCode::CausalClaimUnidentified`] (RE is inconsistent if FE differs).
pub fn hausman(
    x: &Matrix,
    y: &Vector,
    groups: &Vector,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let fe = match PanelFe::new().fit(x, y, groups, &session.child("haus-fe")) {
        Ok(q) => q.value,
        Err(e) => {
            ctx.push(e.primary);
            return ctx.finish(HypothesisTest {
                statistic: f64::NAN,
                pvalue: f64::NAN,
                df: x.ncols() as f64,
                nobs: y.len() as f64,
            });
        }
    };
    let re = match RandomEffects::new().fit(x, y, groups, &session.child("haus-re")) {
        Ok(q) => q.value,
        Err(e) => {
            ctx.push(e.primary);
            return ctx.finish(HypothesisTest {
                statistic: f64::NAN,
                pvalue: f64::NAN,
                df: x.ncols() as f64,
                nobs: y.len() as f64,
            });
        }
    };
    let p = fe.coef.len().min(re.coef.len());
    let mut d2 = 0.0;
    for j in 0..p {
        let d = fe.coef[j] - re.coef[j];
        d2 += d * d;
    }
    let n = fe.n_eff.max(1) as f64;
    let stat = n * d2;
    let df = p.max(1) as f64;
    let pvalue = chi2_pvalue(stat.max(0.0), df);
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("Hausman uses ||β_FE−β_RE||² n as χ²; Var(Δβ) is not the full sandwich")
            .build(),
    );
    if pvalue.is_finite() && pvalue < 0.05 {
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .message(format!(
                    "Hausman p={pvalue:.4}; RE is inconsistent relative to FE"
                ))
                .metric("h", stat)
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

/// Arellano–Bond one-step GMM for a dynamic panel (first-difference IV).
///
/// \(\Delta y_{it}=\rho\Delta y_{i,t-1}+\Delta x_{it}\beta+\Delta\varepsilon_{it}\)
/// with collapsed instrument \(y_{i,t-2}\) (and \(\Delta x\)). Group count is
/// **not** passed as identification `p`.
#[derive(Clone, Debug, Default)]
pub struct ArellanoBondGmm;

impl ArellanoBondGmm {
    /// Default collapsed Arellano–Bond.
    pub fn new() -> Self {
        Self
    }

    /// Fit on rows sorted by group then time.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedArellanoBond>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if groups.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("ArellanoBond groups length ≠ n")
                    .build(),
            );
            return ctx.finish(FittedArellanoBond {
                rho: 0.0,
                coef: Vector::zeros(x.ncols()),
                n_eff: 0,
                n_groups: 0,
            });
        }
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "ArellanoBond");
        let n = y.len().min(x.nrows()).min(groups.len());
        let p = x.ncols();
        let mut yd = Vec::new();
        let mut xd = Vec::new();
        let mut zd = Vec::new();
        for i in 2..n {
            if !groups[i].is_finite() || !groups[i - 1].is_finite() || !groups[i - 2].is_finite() {
                continue;
            }
            let g = groups[i].round() as i64;
            if groups[i - 1].round() as i64 != g || groups[i - 2].round() as i64 != g {
                continue;
            }
            yd.push(y[i] - y[i - 1]);
            let mut row_x = vec![0.0; p + 1];
            row_x[0] = y[i - 1] - y[i - 2];
            let mut row_z = vec![0.0; p + 1];
            row_z[0] = y[i - 2];
            for j in 0..p {
                row_x[j + 1] = x.get(i, j) - x.get(i - 1, j);
                row_z[j + 1] = x.get(i, j) - x.get(i - 1, j);
            }
            xd.push(row_x);
            zd.push(row_z);
        }
        if xd.len() < 3 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("Arellano–Bond has too few first-difference triples")
                    .meaninglessness(Meaninglessness::vacuous(
                        "dynamic-panel ρ",
                        "need groups with at least three consecutive times",
                        "lengthen the panel or sort by group then time",
                    ))
                    .build(),
            );
            return ctx.finish(FittedArellanoBond {
                rho: 0.0,
                coef: Vector::zeros(p),
                n_eff: xd.len(),
                n_groups: sizes.len(),
            });
        }
        let m = xd.len();
        let xmat = Matrix::from_fn(m, p + 1, |i, j| xd[i][j]);
        let zmat = Matrix::from_fn(m, p + 1, |i, j| zd[i][j]);
        let yvec = Vector::from_iter(yd);
        let mut xhat = Matrix::zeros(m, p + 1);
        for j in 0..(p + 1) {
            let xj = xmat.column(j);
            let mut scratch = signlred::Report::new("ab", "s1");
            if let Some(g) = least_squares(&mut scratch, &zmat, &xj, &ctx.policy) {
                let f = zmat.matvec(&g);
                for i in 0..m {
                    xhat.set(i, j, f[i]);
                }
            }
        }
        let mut scratch = signlred::Report::new("ab", "s2");
        let beta = least_squares(&mut scratch, &xhat, &yvec, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(p + 1));
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
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(Severity::Advisory)
                .message("Arellano–Bond is collapsed one-step 2SLS, not two-step Windmeijer GMM")
                .build(),
        );
        ctx.finish(FittedArellanoBond {
            rho: beta.as_slice().first().copied().unwrap_or(0.0),
            coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            n_eff: m,
            n_groups: sizes.len(),
        })
    }
}

/// Fitted Arellano–Bond.
#[derive(Clone, Debug)]
pub struct FittedArellanoBond {
    /// Coefficient on \(\Delta y_{i,t-1}\).
    pub rho: f64,
    /// Slopes on \(\Delta X\).
    pub coef: Vector,
    /// First-difference rows used.
    pub n_eff: usize,
    /// Groups in the panel.
    pub n_groups: usize,
}

/// Blundell–Bond system GMM (collapsed FD + levels).
///
/// Level instruments are `Δy_{i,t-1}`. Group count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct BlundellBond;

impl BlundellBond {
    /// Default collapsed system GMM.
    pub fn new() -> Self {
        Self
    }

    /// Fit on rows sorted by group then time.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedBlundellBond>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if groups.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("BlundellBond groups length ≠ n")
                    .build(),
            );
            return ctx.finish(FittedBlundellBond {
                rho: 0.0,
                coef: Vector::zeros(x.ncols()),
                n_eff: 0,
                n_groups: 0,
            });
        }
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "BlundellBond");
        let n = y.len().min(x.nrows()).min(groups.len());
        let p = x.ncols();
        let mut yd = Vec::new();
        let mut xd = Vec::new();
        let mut zd = Vec::new();
        for i in 2..n {
            if !groups[i].is_finite() || !groups[i - 1].is_finite() || !groups[i - 2].is_finite() {
                continue;
            }
            let g = groups[i].round() as i64;
            if groups[i - 1].round() as i64 != g || groups[i - 2].round() as i64 != g {
                continue;
            }
            yd.push(y[i] - y[i - 1]);
            let mut row_x = vec![0.0; p + 1];
            row_x[0] = y[i - 1] - y[i - 2];
            let mut row_z = vec![0.0; p + 1];
            row_z[0] = y[i - 2];
            for j in 0..p {
                row_x[j + 1] = x.get(i, j) - x.get(i - 1, j);
                row_z[j + 1] = x.get(i, j) - x.get(i - 1, j);
            }
            xd.push(row_x);
            zd.push(row_z);
            yd.push(y[i]);
            let mut row_xl = vec![0.0; p + 1];
            row_xl[0] = y[i - 1];
            let mut row_zl = vec![0.0; p + 1];
            row_zl[0] = y[i - 1] - y[i - 2];
            for j in 0..p {
                row_xl[j + 1] = x.get(i, j);
                row_zl[j + 1] = x.get(i, j);
            }
            xd.push(row_xl);
            zd.push(row_zl);
        }
        if xd.len() < 4 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("Blundell–Bond has too few stacked FD/level rows")
                    .meaninglessness(Meaninglessness::vacuous(
                        "system-GMM ρ",
                        "need groups with at least three consecutive times",
                        "lengthen the panel or sort by group then time",
                    ))
                    .build(),
            );
            return ctx.finish(FittedBlundellBond {
                rho: 0.0,
                coef: Vector::zeros(p),
                n_eff: xd.len(),
                n_groups: sizes.len(),
            });
        }
        let m = xd.len();
        let xmat = Matrix::from_fn(m, p + 1, |i, j| xd[i][j]);
        let zmat = Matrix::from_fn(m, p + 1, |i, j| zd[i][j]);
        let yvec = Vector::from_iter(yd);
        let mut xhat = Matrix::zeros(m, p + 1);
        for j in 0..(p + 1) {
            let xj = xmat.column(j);
            let mut scratch = signlred::Report::new("bb", "s1");
            if let Some(g) = least_squares(&mut scratch, &zmat, &xj, &ctx.policy) {
                let f = zmat.matvec(&g);
                for i in 0..m {
                    xhat.set(i, j, f[i]);
                }
            }
        }
        let mut scratch = signlred::Report::new("bb", "s2");
        let beta = least_squares(&mut scratch, &xhat, &yvec, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(p + 1));
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
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(Severity::Advisory)
                .message("Blundell–Bond is collapsed one-step 2SLS, not two-step Windmeijer GMM")
                .build(),
        );
        ctx.finish(FittedBlundellBond {
            rho: beta.as_slice().first().copied().unwrap_or(0.0),
            coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            n_eff: m,
            n_groups: sizes.len(),
        })
    }
}

/// Fitted Blundell–Bond.
#[derive(Clone, Debug)]
pub struct FittedBlundellBond {
    /// Coefficient on the lagged level / difference of `y`.
    pub rho: f64,
    /// Slopes on `X`.
    pub coef: Vector,
    /// Stacked FD+level rows used.
    pub n_eff: usize,
    /// Groups in the panel.
    pub n_groups: usize,
}

/// Difference GMM (linearmodels `DifferenceGMM` / Arellano–Bond).
///
/// Group count is not identification `p`. Delegates to [`ArellanoBondGmm`].
#[derive(Clone, Debug, Default)]
pub struct DifferenceGmm;

impl DifferenceGmm {
    /// Default collapsed first-difference GMM.
    pub fn new() -> Self {
        Self
    }

    /// Fit on rows sorted by group then time.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedArellanoBond>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("DifferenceGmm is collapsed Arellano–Bond, not two-step Windmeijer")
                .build(),
        );
        match ArellanoBondGmm::new().fit(x, y, groups, &session.child("dgmm-ab")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::CholeskyFailed
                            | IssueCode::MeaninglessFit
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                ctx.finish(q.value)
            }
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("DifferenceGmm inner Arellano–Bond failed")
                        .build(),
                );
                ctx.finish(FittedArellanoBond {
                    rho: 0.0,
                    coef: Vector::zeros(x.ncols()),
                    n_eff: 0,
                    n_groups: 0,
                })
            }
        }
    }
}

/// System GMM (linearmodels `SystemGMM` / Blundell–Bond).
///
/// Group count is not identification `p`. Delegates to [`BlundellBond`].
#[derive(Clone, Debug, Default)]
pub struct SystemGmm;

impl SystemGmm {
    /// Default collapsed system GMM.
    pub fn new() -> Self {
        Self
    }

    /// Fit on rows sorted by group then time.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedBlundellBond>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("SystemGmm is collapsed Blundell–Bond, not two-step Windmeijer")
                .build(),
        );
        match BlundellBond::new().fit(x, y, groups, &session.child("sgmm-bb")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::CholeskyFailed
                            | IssueCode::MeaninglessFit
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                ctx.finish(q.value)
            }
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("SystemGmm inner Blundell–Bond failed")
                        .build(),
                );
                ctx.finish(FittedBlundellBond {
                    rho: 0.0,
                    coef: Vector::zeros(x.ncols()),
                    n_eff: 0,
                    n_groups: 0,
                })
            }
        }
    }
}

/// Fama–MacBeth two-pass slopes (linearmodels `FamaMacBeth`).
///
/// Time count is not identification `p`. Cross-section OLS at each time uses a
/// scratch report; aborting inner codes are not promoted.
#[derive(Clone, Debug, Default)]
pub struct FamaMacBeth;

/// Averaged Fama–MacBeth slopes.
#[derive(Clone, Debug)]
pub struct FittedFamaMacBeth {
    /// Mean cross-sectional slopes.
    pub coef: Vector,
    /// Mean cross-sectional intercept.
    pub intercept: f64,
    /// Time-series standard errors of the slopes.
    pub se: Vector,
    /// Times that produced a finite slope.
    pub n_times: usize,
    /// Average cross-section size.
    pub n_cs_avg: f64,
}

impl FamaMacBeth {
    /// Default two-pass estimator.
    pub fn new() -> Self {
        Self
    }

    /// Fit `y ~ 1 + X` in each time slice, then average the slopes.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        times: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedFamaMacBeth>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let n = y.len().min(x.nrows()).min(times.len());
        if times.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("FamaMacBeth times length ≠ n")
                    .build(),
            );
        }
        let p = x.ncols();
        let mut by_t: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            if times[i].is_finite() {
                by_t.entry(times[i].round() as i64).or_default().push(i);
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("FamaMacBeth is two-pass OLS, not Newey–West / clustered SE")
                .compromise(NumericalCompromise::new(
                    "Fama–MacBeth with HAC standard errors",
                    "plain time-series SD of the cross-section slopes",
                    "serial correlation across times is ignored",
                    "read se as a planning SE, not a published FM t",
                ))
                .build(),
        );
        let mut betas: Vec<Vector> = Vec::new();
        let mut intercepts: Vec<f64> = Vec::new();
        let mut n_cs = 0.0_f64;
        for (_t, rows) in &by_t {
            if rows.len() < 2 {
                continue;
            }
            let xt = Matrix::from_fn(rows.len(), p, |r, j| x.get(rows[r], j));
            let yt = Vector::from_iter(rows.iter().map(|&i| y[i]));
            let design = xt.with_intercept();
            let mut scratch = signlred::Report::new("famamacbeth", "cs-ols");
            let Some(sol) = least_squares(&mut scratch, &design, &yt, &ctx.policy) else {
                continue;
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
            if !sol.as_slice().iter().all(|v| v.is_finite()) {
                continue;
            }
            intercepts.push(sol.as_slice().first().copied().unwrap_or(0.0));
            betas.push(Vector::from_iter((0..p).map(|j| {
                sol.as_slice().get(j + 1).copied().unwrap_or(0.0)
            })));
            n_cs += rows.len() as f64;
        }
        let n_times = betas.len();
        if n_times == 0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("FamaMacBeth has no usable time slice")
                    .build(),
            );
            return ctx.finish(FittedFamaMacBeth {
                coef: Vector::zeros(p),
                intercept: 0.0,
                se: Vector::zeros(p),
                n_times: 0,
                n_cs_avg: 0.0,
            });
        }
        let intercept = intercepts.iter().sum::<f64>() / n_times as f64;
        let coef = Vector::from_iter((0..p).map(|j| {
            betas.iter().map(|b| b[j]).sum::<f64>() / n_times as f64
        }));
        let se = Vector::from_iter((0..p).map(|j| {
            if n_times < 2 {
                return f64::NAN;
            }
            let m = coef[j];
            let ss: f64 = betas.iter().map(|b| {
                let d = b[j] - m;
                d * d
            }).sum();
            (ss / (n_times as f64 - 1.0)).sqrt() / (n_times as f64).sqrt()
        }));
        ctx.finish(FittedFamaMacBeth {
            coef,
            intercept,
            se,
            n_times,
            n_cs_avg: n_cs / n_times as f64,
        })
    }
}

/// Absorbing least squares (linearmodels `AbsorbingLS`): named within FE.
///
/// Group count is not identification `p`. Delegates to [`PanelFe`].
#[derive(Clone, Debug, Default)]
pub struct AbsorbingLs;

impl AbsorbingLs {
    /// Default one-way absorbed FE.
    pub fn new() -> Self {
        Self
    }

    /// Fit after absorbing `groups`.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("AbsorbingLs is one-way within FE, not multi-way HDFE")
                .compromise(NumericalCompromise::new(
                    "multi-way absorbed least squares",
                    "group-mean demeaning then OLS",
                    "only one absorbed factor is implemented",
                    "do not read this as a two-way or iterated HDFE fit",
                ))
                .build(),
        );
        match PanelFe::new().fit(x, y, groups, &session.child("abs-fe")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::CholeskyFailed
                            | IssueCode::MeaninglessFit
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                ctx.finish(q.value)
            }
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("AbsorbingLs inner within OLS failed")
                        .build(),
                );
                ctx.finish(empty_panel(x.ncols()))
            }
        }
    }
}

/// Two-way within FE: \(x_{it}-\bar x_{i\cdot}-\bar x_{\cdot t}+\bar x\)
/// (linearmodels `AbsorbingLS` with entity + time).
///
/// Group and time counts are not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct TwoWayFe;

impl TwoWayFe {
    /// Default two-way within estimator.
    pub fn new() -> Self {
        Self
    }

    /// Fit after entity and time demeaning.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        times: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if groups.len() != y.len() || times.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("TwoWayFe groups/times length ≠ n")
                    .build(),
            );
        }
        let sizes_g = group_sizes(groups);
        let sizes_t = group_sizes(times);
        warn_panel_groups(&mut ctx, &sizes_g, "TwoWayFe");
        if sizes_t.len() <= 1 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("TwoWayFe: a single time cannot identify a two-way estimand")
                    .meaninglessness(Meaninglessness::vacuous(
                        "two-way within slopes",
                        "time demeaning needs at least two times",
                        "use one-way FE, or collect more times",
                    ))
                    .build(),
            );
            return ctx.finish(empty_panel(x.ncols()));
        }
        if ctx.report.contains(IssueCode::UnidentifiedModel)
            || ctx.report.contains(IssueCode::IncrementalUnidentifiable)
        {
            return ctx.finish(empty_panel(x.ncols()));
        }
        let (mx_g, my_g) = group_means(x, y, groups, &sizes_g);
        let (mx_t, my_t) = group_means(x, y, times, &sizes_t);
        let n = y.len().min(x.nrows()).min(groups.len()).min(times.len());
        let p = x.ncols();
        let mut mx_all = vec![0.0_f64; p];
        let mut my_all = 0.0_f64;
        let mut n_fin = 0.0_f64;
        for i in 0..n {
            if !y[i].is_finite() {
                continue;
            }
            my_all += y[i];
            n_fin += 1.0;
            for j in 0..p {
                mx_all[j] += x.get(i, j);
            }
        }
        if n_fin > 0.0 {
            my_all /= n_fin;
            for j in 0..p {
                mx_all[j] /= n_fin;
            }
        }
        let mut xw = Matrix::zeros(n, p);
        let mut yw = Vector::zeros(n);
        for i in 0..n {
            let g = groups[i].round() as i64;
            let t = times[i].round() as i64;
            let gx = mx_g.get(&g);
            let tx = mx_t.get(&t);
            let gy = my_g.get(&g).copied().unwrap_or(0.0);
            let ty = my_t.get(&t).copied().unwrap_or(0.0);
            yw[i] = y[i] - gy - ty + my_all;
            for j in 0..p {
                let mg = gx.map(|v| v[j]).unwrap_or(0.0);
                let mt = tx.map(|v| v[j]).unwrap_or(0.0);
                xw.set(i, j, x.get(i, j) - mg - mt + mx_all[j]);
            }
        }
        if yw.std() <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("two-way within y has ~0 variance")
                    .meaninglessness(Meaninglessness::vacuous(
                        "two-way within slopes",
                        "after entity+time demeaning there is no leftover variation in y",
                        "the regressor is a sum of entity and time effects",
                    ))
                    .build(),
            );
            return ctx.finish(empty_panel(p));
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("TwoWayFe is sequential demeaning, not iterated HDFE")
                .compromise(NumericalCompromise::new(
                    "iterated multi-way absorbed least squares",
                    "entity and time means subtracted once",
                    "unbalanced panels are not re-centred to convergence",
                    "do not read this as linearmodels AbsorbingLS HDFE",
                ))
                .build(),
        );
        let mut scratch = signlred::Report::new("twoway_fe", "ols");
        let Some(coef) = least_squares(&mut scratch, &xw, &yw, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("two-way within OLS failed")
                    .build(),
            );
            return ctx.finish(empty_panel(p));
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
        ctx.finish(FittedPanel {
            coef,
            intercept: 0.0,
            n_groups: sizes_g.len(),
            n_eff: n,
        })
    }
}

/// Mundlak correlated random effects: pooled OLS of \(y\) on \((X, \bar X_g)\).
///
/// Group count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct Mundlak;

impl Mundlak {
    /// Default Mundlak device.
    pub fn new() -> Self {
        Self
    }

    /// Fit \(y \sim 1 + X + \bar X_g\).
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if groups.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("Mundlak groups length ≠ n")
                    .build(),
            );
        }
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "Mundlak");
        if ctx.report.contains(IssueCode::UnidentifiedModel) {
            return ctx.finish(empty_panel(x.ncols()));
        }
        let (mx, _my) = group_means(x, y, groups, &sizes);
        let n = y.len().min(x.nrows()).min(groups.len());
        let p = x.ncols();
        let design = Matrix::from_fn(n, 1 + 2 * p, |i, j| {
            if j == 0 {
                1.0
            } else if j <= p {
                x.get(i, j - 1)
            } else {
                let g = groups[i].round() as i64;
                mx.get(&g).map(|v| v[j - 1 - p]).unwrap_or(0.0)
            }
        });
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Mundlak is pooled OLS on (X, group means), not CRE GLS")
                .compromise(NumericalCompromise::new(
                    "Mundlak CRE / GLS",
                    "pooled OLS with appended group means of X",
                    "random-effect GLS weights are omitted",
                    "read the X slopes as a correlated-RE sketch",
                ))
                .build(),
        );
        let y_use = Vector::from_iter((0..n).map(|i| y[i]));
        let mut scratch = signlred::Report::new("mundlak", "ols");
        let Some(beta) = least_squares(&mut scratch, &design, &y_use, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("Mundlak OLS failed")
                    .build(),
            );
            return ctx.finish(empty_panel(p));
        };
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::R2IsOne
                    | IssueCode::RankZero
                    | IssueCode::CholeskyFailed
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let intercept = beta.as_slice().first().copied().unwrap_or(0.0);
        let coef = Vector::from_iter((0..p).map(|j| {
            beta.as_slice().get(j + 1).copied().unwrap_or(0.0)
        }));
        ctx.finish(FittedPanel {
            coef,
            intercept,
            n_groups: sizes.len(),
            n_eff: n,
        })
    }
}

/// Hausman–Taylor sketch: within slopes plus a between residual intercept
/// (linearmodels `HausmanTaylor`).
///
/// Group count is not identification `p`. Not the full HT IV system.
#[derive(Clone, Debug, Default)]
pub struct HausmanTaylor;

impl HausmanTaylor {
    /// Default Hausman–Taylor sketch.
    pub fn new() -> Self {
        Self
    }

    /// Fit within slopes, then a pooled residual intercept.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("HausmanTaylor is within OLS + residual intercept, not HT IV")
                .compromise(NumericalCompromise::new(
                    "Hausman–Taylor IV for time-invariant endogenous regressors",
                    "PanelFe slopes and mean(y − Xβ)",
                    "no time-invariant block and no instrument partition",
                    "do not read this as linearmodels HausmanTaylor",
                ))
                .build(),
        );
        match PanelFe::new().fit(x, y, groups, &session.child("ht-fe")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::CholeskyFailed
                            | IssueCode::MeaninglessFit
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                let n = y.len().min(x.nrows());
                let p = q.value.coef.len();
                let mut rsum = 0.0_f64;
                let mut n_fin = 0.0_f64;
                for i in 0..n {
                    if !y[i].is_finite() {
                        continue;
                    }
                    let mut xb = 0.0_f64;
                    for j in 0..p.min(x.ncols()) {
                        xb += x.get(i, j) * q.value.coef[j];
                    }
                    rsum += y[i] - xb;
                    n_fin += 1.0;
                }
                let intercept = if n_fin > 0.0 { rsum / n_fin } else { 0.0 };
                ctx.finish(FittedPanel {
                    coef: q.value.coef,
                    intercept,
                    n_groups: q.value.n_groups,
                    n_eff: q.value.n_eff,
                })
            }
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("HausmanTaylor inner within OLS failed")
                        .build(),
                );
                ctx.finish(empty_panel(x.ncols()))
            }
        }
    }
}

/// Fama–MacBeth with lag-1 Newey–West SEs on the time series of slopes
/// (linearmodels clustered / HAC FM).
///
/// Time count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct ClusteredFamaMacBeth;

impl ClusteredFamaMacBeth {
    /// Default clustered two-pass estimator.
    pub fn new() -> Self {
        Self
    }

    /// Fit `y ~ 1 + X` in each time slice; SE is lag-1 HAC of the slopes.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        times: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedFamaMacBeth>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let n = y.len().min(x.nrows()).min(times.len());
        let p = x.ncols();
        let mut by_t: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            if times[i].is_finite() {
                by_t.entry(times[i].round() as i64).or_default().push(i);
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("ClusteredFamaMacBeth is lag-1 HAC on CS slopes, not entity-clustered FM")
                .compromise(NumericalCompromise::new(
                    "entity-clustered Fama–MacBeth",
                    "Newey–West lag 1 on the time series of cross-section slopes",
                    "within-time residual clustering is omitted",
                    "read se as a date-HAC planning SE",
                ))
                .build(),
        );
        let mut betas: Vec<Vector> = Vec::new();
        let mut intercepts: Vec<f64> = Vec::new();
        let mut n_cs = 0.0_f64;
        for (_t, rows) in &by_t {
            if rows.len() < 2 {
                continue;
            }
            let xt = Matrix::from_fn(rows.len(), p, |r, j| x.get(rows[r], j));
            let yt = Vector::from_iter(rows.iter().map(|&i| y[i]));
            let design = xt.with_intercept();
            let mut scratch = signlred::Report::new("cfm", "cs-ols");
            let Some(sol) = least_squares(&mut scratch, &design, &yt, &ctx.policy) else {
                continue;
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
            if !sol.as_slice().iter().all(|v| v.is_finite()) {
                continue;
            }
            intercepts.push(sol.as_slice().first().copied().unwrap_or(0.0));
            betas.push(Vector::from_iter((0..p).map(|j| {
                sol.as_slice().get(j + 1).copied().unwrap_or(0.0)
            })));
            n_cs += rows.len() as f64;
        }
        let n_times = betas.len();
        if n_times == 0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("ClusteredFamaMacBeth has no usable time slice")
                    .build(),
            );
            return ctx.finish(FittedFamaMacBeth {
                coef: Vector::zeros(p),
                intercept: 0.0,
                se: Vector::zeros(p),
                n_times: 0,
                n_cs_avg: 0.0,
            });
        }
        let intercept = intercepts.iter().sum::<f64>() / n_times as f64;
        let coef = Vector::from_iter((0..p).map(|j| {
            betas.iter().map(|b| b[j]).sum::<f64>() / n_times as f64
        }));
        let se = Vector::from_iter((0..p).map(|j| {
            if n_times < 2 {
                return f64::NAN;
            }
            let m = coef[j];
            let mut g0 = 0.0_f64;
            let mut g1 = 0.0_f64;
            for t in 0..n_times {
                let d = betas[t][j] - m;
                g0 += d * d;
                if t + 1 < n_times {
                    g1 += d * (betas[t + 1][j] - m);
                }
            }
            g0 /= n_times as f64;
            g1 /= n_times as f64;
            let nw = (g0 + g1).max(0.0);
            (nw / n_times as f64).sqrt()
        }));
        ctx.finish(FittedFamaMacBeth {
            coef,
            intercept,
            se,
            n_times,
            n_cs_avg: n_cs / n_times as f64,
        })
    }
}

/// Between 2SLS: group-mean IV (linearmodels `BetweenIV` / 2SLS on \(\bar y_g\)).
///
/// Group count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct Between2Sls;

impl Between2Sls {
    /// Default between IV.
    pub fn new() -> Self {
        Self
    }

    /// Fit \(\bar y_g\) on \(\widehat{\bar X}_g(Z)\).
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        z: &Matrix,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if groups.len() != y.len() || z.nrows() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("Between2Sls groups/Z length ≠ n")
                    .build(),
            );
        }
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "Between2Sls");
        if sizes.len() <= 1 {
            return ctx.finish(empty_panel(x.ncols()));
        }
        let (mx, my) = group_means(x, y, groups, &sizes);
        let z_dummy = Vector::zeros(y.len().min(z.nrows()));
        let (mz, _) = group_means(z, &z_dummy, groups, &sizes);
        let keys: Vec<i64> = mx.keys().copied().collect();
        let g = keys.len();
        let p = x.ncols();
        let q = z.ncols();
        let xb = Matrix::from_fn(g, p, |i, j| mx[&keys[i]][j]);
        let zb = Matrix::from_fn(g, q, |i, j| mz.get(&keys[i]).map(|v| v[j]).unwrap_or(0.0));
        let yb = Vector::from_iter(keys.iter().map(|k| my.get(k).copied().unwrap_or(0.0)));
        if yb.std() <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::ConstantTarget)
                    .message("Between2Sls between y is constant")
                    .build(),
            );
            return ctx.finish(empty_panel(p));
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Between2Sls is group-mean 2SLS, not clustered LIML")
                .compromise(NumericalCompromise::new(
                    "panel IV with clustered covariance",
                    "between OLS first and second stages on group means",
                    "within variation is discarded",
                    "read the slope as a between-IV sketch",
                ))
                .build(),
        );
        let z1 = zb.with_intercept();
        let mut xhat = Matrix::zeros(g, p);
        for j in 0..p {
            let xj = Vector::from_iter((0..g).map(|i| xb.get(i, j)));
            let mut scratch = signlred::Report::new("b2sls", "fs");
            if let Some(pi) = least_squares(&mut scratch, &z1, &xj, &ctx.policy) {
                for i in 0..g {
                    let mut s = 0.0_f64;
                    for k in 0..pi.len().min(z1.ncols()) {
                        s += z1.get(i, k) * pi[k];
                    }
                    xhat.set(i, j, s);
                }
            }
            for issue in scratch.issues() {
                if matches!(
                    issue.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::R2IsOne
                        | IssueCode::RankZero
                        | IssueCode::CholeskyFailed
                        | IssueCode::PerfectCollinearity
                ) {
                    continue;
                }
                ctx.push(issue.clone());
            }
        }
        let design = xhat.with_intercept();
        let mut scratch = signlred::Report::new("b2sls", "ss");
        let Some(beta) = least_squares(&mut scratch, &design, &yb, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("Between2Sls second stage failed")
                    .build(),
            );
            return ctx.finish(empty_panel(p));
        };
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::R2IsOne
                    | IssueCode::RankZero
                    | IssueCode::CholeskyFailed
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedPanel {
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            coef: Vector::from_iter((0..p).map(|j| {
                beta.as_slice().get(j + 1).copied().unwrap_or(0.0)
            })),
            n_groups: g,
            n_eff: g,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panel_xy() -> (Matrix, Vector, Vector) {
        // 4 groups × 8 times: y = 2 x + group FE.
        let n = 32;
        let x = Matrix::from_fn(n, 1, |i, _| (i % 8) as f64);
        let y = Vector::from_iter((0..n).map(|i| {
            let g = (i / 8) as f64;
            2.0 * (i % 8) as f64 + 5.0 * g
        }));
        let g = Vector::from_iter((0..n).map(|i| (i / 8) as f64));
        (x, y, g)
    }

    #[test]
    fn fe_between_fd_recover_slope() {
        let (x, y, g) = panel_xy();
        let fe = PanelFe::new()
            .fit(&x, &y, &g, &Session::new("fe", "fit"))
            .expect("fe");
        assert!(
            (fe.value.coef[0] - 2.0).abs() < 1e-8,
            "fe={}",
            fe.value.coef[0]
        );
        let be = BetweenOls::new()
            .fit(&x, &y, &g, &Session::new("be", "fit"))
            .expect("be");
        assert!(be.value.coef[0].is_finite());
        assert_eq!(be.value.n_groups, 4);
        let fd = FirstDifferenceOls::new()
            .fit(&x, &y, &g, &Session::new("fd", "fit"))
            .expect("fd");
        assert!(
            (fd.value.coef[0] - 2.0).abs() < 1e-8,
            "fd={}",
            fd.value.coef[0]
        );
        let po = PooledOls::new()
            .fit(&x, &y, &g, &Session::new("po", "fit"))
            .expect("pooled");
        assert!(po.value.coef[0].is_finite());
        let re = RandomEffects::new()
            .fit(&x, &y, &g, &Session::new("re", "fit"))
            .expect("re");
        assert!(
            (re.value.coef[0] - 2.0).abs() < 0.5,
            "re={}",
            re.value.coef[0]
        );
        let h = hausman(&x, &y, &g, &Session::new("haus", "t")).expect("haus");
        assert!(h.value.statistic.is_finite() || h.value.pvalue.is_nan());
        let mut ydyn = y.clone();
        for i in 1..32 {
            if g[i] == g[i - 1] {
                ydyn[i] = 0.4 * ydyn[i - 1] + 2.0 * x.get(i, 0) + 0.1 * (i / 8) as f64;
            }
        }
        let ab = ArellanoBondGmm::new()
            .fit(&x, &ydyn, &g, &Session::new("ab", "fit"))
            .expect("ab");
        assert!(ab.value.n_eff > 0);
        assert!(ab.value.rho.is_finite());
        let bb = BlundellBond::new()
            .fit(&x, &ydyn, &g, &Session::new("bb", "fit"))
            .expect("bb");
        assert!(bb.value.n_eff > 0);
        assert!(bb.value.rho.is_finite());
        let dg = DifferenceGmm::new()
            .fit(&x, &ydyn, &g, &Session::new("dgmm", "fit"))
            .expect("dgmm");
        assert!(dg.value.n_eff > 0);
        assert!(dg.value.rho.is_finite());
        let sg = SystemGmm::new()
            .fit(&x, &ydyn, &g, &Session::new("sgmm", "fit"))
            .expect("sgmm");
        assert!(sg.value.n_eff > 0);
        assert!(sg.value.rho.is_finite());
        let time = Vector::from_iter((0..32).map(|i| (i % 8) as f64));
        let xfm = Matrix::from_fn(32, 1, |i, _| (i / 8) as f64);
        let fm = FamaMacBeth::new()
            .fit(&xfm, &y, &time, &Session::new("fmb", "fit"))
            .expect("fmb");
        assert_eq!(fm.value.coef.len(), 1);
        assert!(fm.value.coef[0].is_finite());
        assert!(fm.value.n_times > 0);
        let als = AbsorbingLs::new()
            .fit(&x, &y, &g, &Session::new("als", "fit"))
            .expect("als");
        assert!((als.value.coef[0] - 2.0).abs() < 0.25, "als={}", als.value.coef[0]);
        let xtw = Matrix::from_fn(32, 1, |i, _| {
            (i % 8) as f64 * (1.0 + (i / 8) as f64)
        });
        let ytw = Vector::from_iter((0..32).map(|i| {
            2.0 * (i % 8) as f64 * (1.0 + (i / 8) as f64) + 5.0 * (i / 8) as f64
        }));
        let tw = TwoWayFe::new()
            .fit(&xtw, &ytw, &g, &time, &Session::new("twfe", "fit"))
            .expect("twfe");
        assert!(
            (tw.value.coef[0] - 2.0).abs() < 1e-6,
            "twfe={}",
            tw.value.coef[0]
        );
        let mk = Mundlak::new()
            .fit(&x, &y, &g, &Session::new("mund", "fit"))
            .expect("mund");
        assert!(mk.value.coef[0].is_finite());
        let ht = HausmanTaylor::new()
            .fit(&x, &y, &g, &Session::new("ht", "fit"))
            .expect("ht");
        assert!((ht.value.coef[0] - 2.0).abs() < 0.25, "ht={}", ht.value.coef[0]);
        let cfm = ClusteredFamaMacBeth::new()
            .fit(&xfm, &y, &time, &Session::new("cfm", "fit"))
            .expect("cfm");
        assert_eq!(cfm.value.coef.len(), 1);
        assert!(cfm.value.coef[0].is_finite());
        let b2 = Between2Sls::new()
            .fit(&xfm, &y, &xfm, &g, &Session::new("b2s", "fit"))
            .expect("b2s");
        assert!(b2.value.coef[0].is_finite());
    }
}
