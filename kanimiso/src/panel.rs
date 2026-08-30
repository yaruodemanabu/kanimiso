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

/// First-difference 2SLS (linearmodels `IV2SLS` on \(\Delta y,\Delta X,\Delta Z\)).
///
/// Group count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct FirstDifferenceIv;

impl FirstDifferenceIv {
    /// Default first-difference IV.
    pub fn new() -> Self {
        Self
    }

    /// Fit \(\Delta y\) on \(\widehat{\Delta X}(\Delta Z)\) within group.
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
                    .message("FirstDifferenceIv groups/Z length ≠ n")
                    .build(),
            );
        }
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "FirstDifferenceIv");
        let n = y.len().min(x.nrows()).min(groups.len()).min(z.nrows());
        let p = x.ncols();
        let q = z.ncols();
        let mut xd = Vec::new();
        let mut zd = Vec::new();
        let mut yd = Vec::new();
        for i in 1..n {
            if !groups[i].is_finite() || !groups[i - 1].is_finite() {
                continue;
            }
            if groups[i].round() as i64 != groups[i - 1].round() as i64 {
                continue;
            }
            yd.push(y[i] - y[i - 1]);
            xd.push((0..p).map(|j| x.get(i, j) - x.get(i - 1, j)).collect::<Vec<_>>());
            zd.push((0..q).map(|j| z.get(i, j) - z.get(i - 1, j)).collect::<Vec<_>>());
        }
        if xd.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("FirstDifferenceIv has no consecutive within-group pairs")
                    .meaninglessness(Meaninglessness::vacuous(
                        "first-difference IV slopes",
                        "the panel has no adjacent repeats in row order",
                        "sort by group then time",
                    ))
                    .build(),
            );
            return ctx.finish(empty_panel(p));
        }
        let m = xd.len();
        let xm = Matrix::from_fn(m, p, |i, j| xd[i][j]);
        let zm = Matrix::from_fn(m, q, |i, j| zd[i][j]);
        let ym = Vector::from_iter(yd);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("FirstDifferenceIv is collapsed FD 2SLS, not GMM")
                .compromise(NumericalCompromise::new(
                    "Arellano–Bond / FD GMM",
                    "2SLS on first differences with contemporaneous ΔZ",
                    "lagged-level instruments and Windmeijer SE are omitted",
                    "read the slope as a planning FD-IV sketch",
                ))
                .build(),
        );
        let z1 = zm.with_intercept();
        let mut xhat = Matrix::zeros(m, p);
        for j in 0..p {
            let xj = Vector::from_iter((0..m).map(|i| xm.get(i, j)));
            let mut scratch = signlred::Report::new("fdiv", "fs");
            if let Some(pi) = least_squares(&mut scratch, &z1, &xj, &ctx.policy) {
                for i in 0..m {
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
        let mut scratch = signlred::Report::new("fdiv", "ss");
        let Some(coef) = least_squares(&mut scratch, &xhat, &ym, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("FirstDifferenceIv second stage failed")
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
            coef,
            intercept: 0.0,
            n_groups: sizes.len(),
            n_eff: m,
        })
    }
}

/// Pooled 2SLS (linearmodels `IV2SLS` without group demeaning).
///
/// Group count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct Pooled2Sls;

impl Pooled2Sls {
    /// Default pooled IV.
    pub fn new() -> Self {
        Self
    }

    /// Fit \(y\) on \(\widehat X(Z)\) with an intercept.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        z: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let n = y.len().min(x.nrows()).min(z.nrows());
        let p = x.ncols();
        if z.nrows() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("Pooled2Sls Z rows ≠ X rows")
                    .build(),
            );
        }
        let y_use = Vector::from_iter((0..n).map(|i| y[i]));
        let z1 = Matrix::from_fn(n, z.ncols(), |i, j| z.get(i, j)).with_intercept();
        let mut xhat = Matrix::zeros(n, p);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Pooled2Sls is 2SLS, not limited-information ML")
                .compromise(NumericalCompromise::new(
                    "LIML / clustered IV",
                    "pooled first and second stage OLS",
                    "group structure is ignored",
                    "read the slope as a pooled-IV sketch",
                ))
                .build(),
        );
        for j in 0..p {
            let xj = Vector::from_iter((0..n).map(|i| x.get(i, j)));
            let mut scratch = signlred::Report::new("p2sls", "fs");
            if let Some(pi) = least_squares(&mut scratch, &z1, &xj, &ctx.policy) {
                for i in 0..n {
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
        let mut scratch = signlred::Report::new("p2sls", "ss");
        let Some(beta) = least_squares(&mut scratch, &design, &y_use, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("Pooled2Sls second stage failed")
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
            n_groups: 1,
            n_eff: n,
        })
    }
}

fn skip_panel_inner(issue: &signlred::Issue) -> bool {
    matches!(
        issue.code,
        IssueCode::ResidualTooLarge
            | IssueCode::NearSingular
            | IssueCode::R2IsOne
            | IssueCode::RankZero
            | IssueCode::CholeskyFailed
            | IssueCode::PerfectCollinearity
            | IssueCode::MeaninglessFit
    )
}

fn twosls_stages(
    x: &Matrix,
    y: &Vector,
    z: &Matrix,
    intercept: bool,
    policy: &signlred::Policy,
    tag: &str,
    ctx: &mut FitCtx,
) -> Option<Vector> {
    let n = y.len().min(x.nrows()).min(z.nrows());
    let p = x.ncols();
    let z1 = if intercept {
        Matrix::from_fn(n, z.ncols(), |i, j| z.get(i, j)).with_intercept()
    } else {
        Matrix::from_fn(n, z.ncols(), |i, j| z.get(i, j))
    };
    let mut xhat = Matrix::zeros(n, p);
    for j in 0..p {
        let xj = Vector::from_iter((0..n).map(|i| x.get(i, j)));
        let mut scratch = signlred::Report::new(tag, "fs");
        if let Some(pi) = least_squares(&mut scratch, &z1, &xj, policy) {
            for i in 0..n {
                let mut s = 0.0_f64;
                for k in 0..pi.len().min(z1.ncols()) {
                    s += z1.get(i, k) * pi[k];
                }
                xhat.set(i, j, s);
            }
        }
        for issue in scratch.issues() {
            if skip_panel_inner(issue) {
                continue;
            }
            ctx.push(issue.clone());
        }
    }
    let design = if intercept { xhat.with_intercept() } else { xhat };
    let y_use = Vector::from_iter((0..n).map(|i| y[i]));
    let mut scratch = signlred::Report::new(tag, "ss");
    let beta = least_squares(&mut scratch, &design, &y_use, policy);
    for issue in scratch.issues() {
        if skip_panel_inner(issue) {
            continue;
        }
        ctx.push(issue.clone());
    }
    beta
}

/// Named between IV (linearmodels `BetweenIV`).
#[derive(Clone, Debug, Default)]
pub struct BetweenIv {
    inner: Between2Sls,
}

impl BetweenIv {
    /// Default between IV.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit group-mean 2SLS.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        z: &Matrix,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        self.inner.fit(x, y, z, groups, session)
    }
}

/// Absorbing / within 2SLS (linearmodels `AbsorbingLS` + IV).
///
/// Group count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct Absorbing2Sls;

impl Absorbing2Sls {
    /// Default within IV.
    pub fn new() -> Self {
        Self
    }

    /// Fit 2SLS after group demeaning (no intercept).
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
                    .message("Absorbing2Sls groups/Z length ≠ n")
                    .build(),
            );
        }
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "Absorbing2Sls");
        if ctx.report.contains(IssueCode::UnidentifiedModel)
            || ctx.report.contains(IssueCode::IncrementalUnidentifiable)
        {
            return ctx.finish(empty_panel(x.ncols()));
        }
        let n = y.len().min(x.nrows()).min(groups.len()).min(z.nrows());
        let p = x.ncols();
        let q = z.ncols();
        let z_dummy = Vector::zeros(n);
        let x_n = Matrix::from_fn(n, p, |i, j| x.get(i, j));
        let z_n = Matrix::from_fn(n, q, |i, j| z.get(i, j));
        let y_n = Vector::from_iter((0..n).map(|i| y[i]));
        let g_n = Vector::from_iter((0..n).map(|i| groups[i]));
        let (mx, my) = group_means(&x_n, &y_n, &g_n, &sizes);
        let (mz, _) = group_means(&z_n, &z_dummy, &g_n, &sizes);
        let mut xw = Matrix::zeros(n, p);
        let mut zw = Matrix::zeros(n, q);
        let mut yw = Vector::zeros(n);
        for i in 0..n {
            if !g_n[i].is_finite() {
                continue;
            }
            let g = g_n[i].round() as i64;
            yw[i] = y_n[i] - my.get(&g).copied().unwrap_or(0.0);
            for j in 0..p {
                xw.set(i, j, x_n.get(i, j) - mx.get(&g).map(|v| v[j]).unwrap_or(0.0));
            }
            for j in 0..q {
                zw.set(i, j, z_n.get(i, j) - mz.get(&g).map(|v| v[j]).unwrap_or(0.0));
            }
        }
        if yw.std() <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("Absorbing2Sls within y has ~0 variance")
                    .meaninglessness(Meaninglessness::vacuous(
                        "within IV slopes",
                        "after demeaning there is no leftover variation in y",
                        "need within movement",
                    ))
                    .build(),
            );
            return ctx.finish(empty_panel(p));
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Absorbing2Sls is within 2SLS, not high-dimensional absorbed LIML")
                .compromise(NumericalCompromise::new(
                    "absorbing IV / 2SLS with many FE",
                    "group-demean then 2SLS without an intercept",
                    "clustered SE and singleton drops are omitted",
                    "read the slope as a within-IV sketch",
                ))
                .build(),
        );
        let policy = ctx.policy.clone();
        match twosls_stages(&xw, &yw, &zw, false, &policy, "a2s", &mut ctx) {
            Some(coef) => ctx.finish(FittedPanel {
                coef,
                intercept: 0.0,
                n_groups: sizes.len(),
                n_eff: n,
            }),
            None => {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("Absorbing2Sls second stage failed")
                        .build(),
                );
                ctx.finish(empty_panel(p))
            }
        }
    }
}

/// Random-effects IV via a fixed quasi-demeaning θ (Baltagi / Hausman–Taylor lite).
///
/// Group count is not identification `p`. θ is not identification `p`.
#[derive(Clone, Debug)]
pub struct RandomEffectsIv {
    /// Quasi-demeaning weight in \([0,1]\).
    pub theta: f64,
}

impl Default for RandomEffectsIv {
    fn default() -> Self {
        Self { theta: 0.5 }
    }
}

impl RandomEffectsIv {
    /// RE-IV with quasi-demeaning `theta`.
    pub fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Fit 2SLS on \(y-\theta\bar y_g\), \(X-\theta\bar X_g\), \(Z-\theta\bar Z_g\).
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
                    .message("RandomEffectsIv groups/Z length ≠ n")
                    .build(),
            );
        }
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "RandomEffectsIv");
        if ctx.report.contains(IssueCode::UnidentifiedModel) {
            return ctx.finish(empty_panel(x.ncols()));
        }
        let mut theta = self.theta;
        if !theta.is_finite() || !(0.0..=1.0).contains(&theta) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message("RandomEffectsIv theta outside [0,1]; using 0.5")
                    .build(),
            );
            theta = 0.5;
        }
        let n = y.len().min(x.nrows()).min(groups.len()).min(z.nrows());
        let p = x.ncols();
        let q = z.ncols();
        let z_dummy = Vector::zeros(n);
        let x_n = Matrix::from_fn(n, p, |i, j| x.get(i, j));
        let z_n = Matrix::from_fn(n, q, |i, j| z.get(i, j));
        let y_n = Vector::from_iter((0..n).map(|i| y[i]));
        let g_n = Vector::from_iter((0..n).map(|i| groups[i]));
        let (mx, my) = group_means(&x_n, &y_n, &g_n, &sizes);
        let (mz, _) = group_means(&z_n, &z_dummy, &g_n, &sizes);
        let mut xq = Matrix::zeros(n, p);
        let mut zq = Matrix::zeros(n, q);
        let mut yq = Vector::zeros(n);
        for i in 0..n {
            if !g_n[i].is_finite() {
                continue;
            }
            let g = g_n[i].round() as i64;
            yq[i] = y_n[i] - theta * my.get(&g).copied().unwrap_or(0.0);
            for j in 0..p {
                xq.set(
                    i,
                    j,
                    x_n.get(i, j) - theta * mx.get(&g).map(|v| v[j]).unwrap_or(0.0),
                );
            }
            for j in 0..q {
                zq.set(
                    i,
                    j,
                    z_n.get(i, j) - theta * mz.get(&g).map(|v| v[j]).unwrap_or(0.0),
                );
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("RandomEffectsIv uses a fixed θ, not Swamy–Arora GLS")
                .compromise(NumericalCompromise::new(
                    "Baltagi EC2SLS / Hausman–Taylor",
                    "quasi-demeaned 2SLS with a user θ",
                    "variance-component θ and GLS weights are omitted",
                    "read the slope as an RE-IV sketch",
                ))
                .build(),
        );
        let policy = ctx.policy.clone();
        match twosls_stages(&xq, &yq, &zq, true, &policy, "reiv", &mut ctx) {
            Some(beta) => ctx.finish(FittedPanel {
                intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                coef: Vector::from_iter((0..p).map(|j| {
                    beta.as_slice().get(j + 1).copied().unwrap_or(0.0)
                })),
                n_groups: sizes.len(),
                n_eff: n,
            }),
            None => {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("RandomEffectsIv second stage failed")
                        .build(),
                );
                ctx.finish(empty_panel(p))
            }
        }
    }
}

/// Chamberlain / two-way Mundlak CRE: \(y\sim 1+X+\bar X_g+\bar X_t\).
///
/// Group and time counts are not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct Chamberlain;

impl Chamberlain {
    /// Default two-way Mundlak device.
    pub fn new() -> Self {
        Self
    }

    /// Fit pooled OLS on \((1, X, \bar X_g, \bar X_t)\).
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        time: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if groups.len() != y.len() || time.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("Chamberlain groups/time length ≠ n")
                    .build(),
            );
        }
        let gsz = group_sizes(groups);
        let tsz = group_sizes(time);
        warn_panel_groups(&mut ctx, &gsz, "Chamberlain");
        if ctx.report.contains(IssueCode::UnidentifiedModel) {
            return ctx.finish(empty_panel(x.ncols()));
        }
        let n = y.len().min(x.nrows()).min(groups.len()).min(time.len());
        let p = x.ncols();
        let x_n = Matrix::from_fn(n, p, |i, j| x.get(i, j));
        let y_n = Vector::from_iter((0..n).map(|i| y[i]));
        let g_n = Vector::from_iter((0..n).map(|i| groups[i]));
        let t_n = Vector::from_iter((0..n).map(|i| time[i]));
        let (mxg, _) = group_means(&x_n, &y_n, &g_n, &gsz);
        let (mxt, _) = group_means(&x_n, &y_n, &t_n, &tsz);
        let design = Matrix::from_fn(n, 1 + 3 * p, |i, j| {
            if j == 0 {
                1.0
            } else if j <= p {
                x_n.get(i, j - 1)
            } else if j <= 2 * p {
                let g = g_n[i].round() as i64;
                mxg.get(&g).map(|v| v[j - 1 - p]).unwrap_or(0.0)
            } else {
                let t = t_n[i].round() as i64;
                mxt.get(&t).map(|v| v[j - 1 - 2 * p]).unwrap_or(0.0)
            }
        });
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Chamberlain is pooled OLS on (X, group means, time means)")
                .compromise(NumericalCompromise::new(
                    "Chamberlain CRE / two-way Mundlak GLS",
                    "pooled OLS with appended group and time means of X",
                    "random-effect GLS and minimum-distance π are omitted",
                    "read the X slopes as a two-way CRE sketch",
                ))
                .build(),
        );
        let mut scratch = signlred::Report::new("chamb", "ols");
        let Some(beta) = least_squares(&mut scratch, &design, &y_n, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("Chamberlain OLS failed")
                    .build(),
            );
            return ctx.finish(empty_panel(p));
        };
        for issue in scratch.issues() {
            if skip_panel_inner(issue) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedPanel {
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            coef: Vector::from_iter((0..p).map(|j| {
                beta.as_slice().get(j + 1).copied().unwrap_or(0.0)
            })),
            n_groups: gsz.len(),
            n_eff: n,
        })
    }
}

/// Named two-way Mundlak alias of [`Chamberlain`].
#[derive(Clone, Debug, Default)]
pub struct TwoWayMundlak {
    inner: Chamberlain,
}

impl TwoWayMundlak {
    /// Default two-way Mundlak.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit \(y\sim 1+X+\bar X_g+\bar X_t\).
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        time: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPanel>> {
        self.inner.fit(x, y, groups, time, session)
    }
}

/// Pesaran–Smith mean-group estimator (linearmodels / entity-wise OLS).
///
/// Group count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct MeanGroup;

impl MeanGroup {
    /// Default mean-group OLS.
    pub fn new() -> Self {
        Self
    }

    /// Average within-group OLS slopes (with intercept).
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
                    .message("MeanGroup groups length ≠ n")
                    .build(),
            );
        }
        let sizes = group_sizes(groups);
        warn_panel_groups(&mut ctx, &sizes, "MeanGroup");
        let n = y.len().min(x.nrows()).min(groups.len());
        let p = x.ncols();
        let mut acc = Vector::zeros(p);
        let mut icept = 0.0_f64;
        let mut used = 0usize;
        for (&g, &sz) in &sizes {
            if sz < 2 {
                continue;
            }
            let rows: Vec<usize> = (0..n)
                .filter(|&i| groups[i].is_finite() && groups[i].round() as i64 == g)
                .collect();
            if rows.len() < 2 {
                continue;
            }
            let xd = Matrix::from_fn(rows.len(), p, |r, j| x.get(rows[r], j)).with_intercept();
            let yd = Vector::from_iter(rows.iter().map(|&i| y[i]));
            let mut scratch = signlred::Report::new("mg", "ols");
            let Some(beta) = least_squares(&mut scratch, &xd, &yd, &ctx.policy) else {
                continue;
            };
            for issue in scratch.issues() {
                if skip_panel_inner(issue) {
                    continue;
                }
                ctx.push(issue.clone());
            }
            icept += beta.as_slice().first().copied().unwrap_or(0.0);
            for j in 0..p {
                acc[j] += beta.as_slice().get(j + 1).copied().unwrap_or(0.0);
            }
            used += 1;
        }
        if used == 0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("MeanGroup has no group with a usable OLS")
                    .build(),
            );
            return ctx.finish(empty_panel(p));
        }
        let u = used as f64;
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("MeanGroup averages entity OLS, not a hierarchical Bayes slope")
                .compromise(NumericalCompromise::new(
                    "Pesaran–Smith MG / random-coefficient GLS",
                    "unweighted average of per-group OLS slopes",
                    "Swamy weights and cross-section dependence are omitted",
                    "read the slope as a mean-group sketch",
                ))
                .build(),
        );
        ctx.finish(FittedPanel {
            coef: acc.scale(1.0 / u),
            intercept: icept / u,
            n_groups: used,
            n_eff: n,
        })
    }
}

/// Entity-clustered Fama–MacBeth (linearmodels clustered covariance).
///
/// Time and entity counts are not identification `p`. Point estimates match
/// the two-pass average; SE come from entity-summed scores, not date HAC.
#[derive(Clone, Debug, Default)]
pub struct EntityClusteredFamaMacBeth;

impl EntityClusteredFamaMacBeth {
    /// Default entity-clustered two-pass estimator.
    pub fn new() -> Self {
        Self
    }

    /// Fit `y ~ 1 + X` in each time slice; cluster the scores by `groups`.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        times: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedFamaMacBeth>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if times.len() != y.len() || groups.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("EntityClusteredFamaMacBeth times/groups length ≠ n")
                    .build(),
            );
        }
        let n = y.len().min(x.nrows()).min(times.len()).min(groups.len());
        let p = x.ncols();
        let mut by_t: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            if times[i].is_finite() {
                by_t.entry(times[i].round() as i64).or_default().push(i);
            }
        }
        let mut betas: Vec<Vector> = Vec::new();
        let mut intercepts: Vec<f64> = Vec::new();
        let mut alpha_t: BTreeMap<i64, f64> = BTreeMap::new();
        let mut n_cs = 0.0_f64;
        for (&t, rows) in &by_t {
            if rows.len() < 2 {
                continue;
            }
            let xt = Matrix::from_fn(rows.len(), p, |r, j| x.get(rows[r], j));
            let yt = Vector::from_iter(rows.iter().map(|&i| y[i]));
            let design = xt.with_intercept();
            let mut scratch = signlred::Report::new("ecfm", "cs-ols");
            let Some(sol) = least_squares(&mut scratch, &design, &yt, &ctx.policy) else {
                continue;
            };
            for issue in scratch.issues() {
                if skip_panel_inner(issue) {
                    continue;
                }
                ctx.push(issue.clone());
            }
            if !sol.as_slice().iter().all(|v| v.is_finite()) {
                continue;
            }
            let a = sol.as_slice().first().copied().unwrap_or(0.0);
            intercepts.push(a);
            alpha_t.insert(t, a);
            betas.push(Vector::from_iter((0..p).map(|j| {
                sol.as_slice().get(j + 1).copied().unwrap_or(0.0)
            })));
            n_cs += rows.len() as f64;
        }
        let n_times = betas.len();
        if n_times == 0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("EntityClusteredFamaMacBeth has no usable time slice")
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
        let mut scores: BTreeMap<i64, Vector> = BTreeMap::new();
        for i in 0..n {
            if !times[i].is_finite() || !groups[i].is_finite() {
                continue;
            }
            let t = times[i].round() as i64;
            let g = groups[i].round() as i64;
            let a = alpha_t.get(&t).copied().unwrap_or(intercept);
            let mut xb = a;
            for j in 0..p {
                xb += x.get(i, j) * coef[j];
            }
            let e = y[i] - xb;
            let s = scores.entry(g).or_insert_with(|| Vector::zeros(p));
            for j in 0..p {
                s[j] += x.get(i, j) * e;
            }
        }
        let gcount = scores.len().max(1) as f64;
        let se = Vector::from_iter((0..p).map(|j| {
            let mut ss = 0.0_f64;
            for s in scores.values() {
                ss += s[j] * s[j];
            }
            (ss.max(0.0) / (gcount * n_times as f64)).sqrt()
        }));
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("EntityClusteredFamaMacBeth uses entity-summed CS scores, not Cameron–Gelbach–Miller")
                .compromise(NumericalCompromise::new(
                    "entity-clustered Fama–MacBeth sandwich",
                    "two-pass slopes plus Σ_g s_g s_g' / (G T)",
                    "finite-sample CGM and multi-way clustering are omitted",
                    "read se as an entity-clustered planning SE",
                ))
                .build(),
        );
        ctx.finish(FittedFamaMacBeth {
            coef,
            intercept,
            se,
            n_times,
            n_cs_avg: n_cs / n_times as f64,
        })
    }
}

/// Two-by-two difference-in-differences (statsmodels / linearmodels TWFE lite).
///
/// Group / period counts are not identification `p`. The design is
/// \(y\sim 1+D+T+D{\times}T\).
#[derive(Clone, Debug, Default)]
pub struct DiffInDiff;

/// Fitted DiD interaction.
#[derive(Clone, Debug)]
pub struct FittedDiffInDiff {
    /// \(\hat\tau\) on \(D\times T\).
    pub att: f64,
    /// Treat, post, and interaction slopes.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
}

impl DiffInDiff {
    /// Default 2×2 DiD.
    pub fn new() -> Self {
        Self
    }

    /// OLS of `y` on treat, post, and treat×post.
    pub fn fit(
        &self,
        y: &Vector,
        treat: &Vector,
        post: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedDiffInDiff>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let n = y.len().min(treat.len()).min(post.len());
        let x = Matrix::from_fn(n, 3, |i, j| match j {
            0 => {
                if treat[i] >= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            1 => {
                if post[i] >= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            _ => {
                if treat[i] >= 0.5 && post[i] >= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
        });
        inspect_xy(&mut ctx.report, &x, Some(y), &ctx.policy);
        let n1 = (0..n).filter(|&i| treat[i] >= 0.5).count();
        let n0 = n.saturating_sub(n1);
        let np = (0..n).filter(|&i| post[i] >= 0.5).count();
        if n1 == 0 || n0 == 0 || np == 0 || np == n {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("DiffInDiff needs variation in treat and post")
                    .meaninglessness(Meaninglessness::vacuous(
                        "2×2 DiD interaction",
                        "a missing arm or period leaves D×T unidentified",
                        "collect a 2×2 panel",
                    ))
                    .build(),
            );
        }
        let design = x.with_intercept();
        let yn = Vector::from_iter((0..n).map(|i| y[i]));
        let mut scratch = signlred::Report::new("did", "ols");
        let sol = least_squares(&mut scratch, &design, &yn, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(4));
        for issue in scratch.issues() {
            if skip_panel_inner(issue) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("DiffInDiff is OLS on D+T+D×T, not a published TWFE / Callaway–Sant’Anna estimator")
                .compromise(NumericalCompromise::new(
                    "difference-in-differences",
                    "four-cell OLS interaction",
                    "staggered timing, never-treated controls, and clustered SE are omitted",
                    "read ATT as a 2×2 interaction, not a published DiD",
                ))
                .build(),
        );
        ctx.finish(FittedDiffInDiff {
            att: sol.as_slice().get(3).copied().unwrap_or(0.0),
            coef: Vector::from_iter((1..sol.len()).map(|j| sol[j])),
            intercept: sol.as_slice().first().copied().unwrap_or(0.0),
        })
    }
}

/// Synthetic control (Abadie–Diamond–Hainmueller lite).
///
/// Donor / pre-period counts are not identification `p`. Weights are
/// projected onto the simplex after ISTA on the pre-period SSE.
#[derive(Clone, Debug, Default)]
pub struct SyntheticControl;

/// Fitted donor weights and post-period gap.
#[derive(Clone, Debug)]
pub struct FittedSyntheticControl {
    /// Simplex weights on donors (columns of `donors`).
    pub weights: Vector,
    /// Pre-period SSE.
    pub pre_rmse: f64,
    /// Mean post-period treated − synthetic gap.
    pub att: f64,
}

impl SyntheticControl {
    /// Default synthetic control.
    pub fn new() -> Self {
        Self
    }

    /// Match `treated` to `donors` on the first `n_pre` rows.
    ///
    /// `n_pre` is a time-window width, not identification `p`.
    pub fn fit(
        &self,
        treated: &Vector,
        donors: &Matrix,
        n_pre: usize,
        session: &Session,
    ) -> Result<Qualified<FittedSyntheticControl>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, donors, Some(treated), &ctx.policy);
        let t = treated.len().min(donors.nrows());
        let j = donors.ncols();
        if t < 2 || j == 0 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("SyntheticControl needs a treated series and at least one donor")
                    .meaninglessness(Meaninglessness::vacuous(
                        "synthetic-control weights",
                        "an empty donor pool cannot match the treated path",
                        "add donor units",
                    ))
                    .build(),
            );
            return ctx.finish(FittedSyntheticControl {
                weights: Vector::zeros(j),
                pre_rmse: f64::NAN,
                att: f64::NAN,
            });
        }
        let pre = n_pre.max(1).min(t.saturating_sub(1)).max(1);
        let mut w = Vector::filled(j, 1.0 / j as f64);
        let lr = 0.05 / (1.0 + j as f64);
        for _ in 0..80 {
            let mut g = Vector::zeros(j);
            for s in 0..pre {
                let mut syn = 0.0;
                for k in 0..j {
                    syn += donors.get(s, k) * w[k];
                }
                let r = syn - treated[s];
                for k in 0..j {
                    g[k] += r * donors.get(s, k);
                }
            }
            for k in 0..j {
                w[k] = (w[k] - lr * g[k] / pre.max(1) as f64).max(0.0);
            }
            let z: f64 = (0..j).map(|k| w[k]).sum();
            if z > 1e-15 {
                for k in 0..j {
                    w[k] /= z;
                }
            } else {
                w = Vector::filled(j, 1.0 / j as f64);
            }
        }
        let mut sse = 0.0;
        for s in 0..pre {
            let mut syn = 0.0;
            for k in 0..j {
                syn += donors.get(s, k) * w[k];
            }
            let d = treated[s] - syn;
            sse += d * d;
        }
        let mut gap = 0.0_f64;
        let mut m = 0.0_f64;
        for s in pre..t {
            let mut syn = 0.0;
            for k in 0..j {
                syn += donors.get(s, k) * w[k];
            }
            gap += treated[s] - syn;
            m += 1.0;
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("SyntheticControl is simplex ISTA on pre-period SSE, not ADH / SCM")
                .compromise(NumericalCompromise::new(
                    "synthetic control",
                    "projected ISTA weights matching the pre-period path",
                    "V-optimization, nested donor search, and placebo inference are omitted",
                    "read ATT as a post-period gap, not a published SCM estimate",
                ))
                .build(),
        );
        ctx.finish(FittedSyntheticControl {
            weights: w,
            pre_rmse: (sse / pre.max(1) as f64).sqrt(),
            att: gap / m.max(1.0_f64),
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
        let fdiv = FirstDifferenceIv::new()
            .fit(&x, &y, &x, &g, &Session::new("fdiv", "fit"))
            .expect("fdiv");
        assert!(fdiv.value.coef[0].is_finite());
        let p2 = Pooled2Sls::new()
            .fit(&x, &y, &x, &Session::new("p2s", "fit"))
            .expect("p2s");
        assert!(p2.value.coef[0].is_finite());
        let biv = BetweenIv::new()
            .fit(&xfm, &y, &xfm, &g, &Session::new("biv", "fit"))
            .expect("biv");
        assert!(biv.value.coef[0].is_finite());
        let a2 = Absorbing2Sls::new()
            .fit(&x, &y, &x, &g, &Session::new("a2s", "fit"))
            .expect("a2s");
        assert!((a2.value.coef[0] - 2.0).abs() < 0.25, "a2s={}", a2.value.coef[0]);
        let reiv = RandomEffectsIv::new(0.5)
            .fit(&x, &y, &x, &g, &Session::new("reiv", "fit"))
            .expect("reiv");
        assert!(reiv.value.coef[0].is_finite());
        let ch = Chamberlain::new()
            .fit(&xtw, &ytw, &g, &time, &Session::new("ch", "fit"))
            .expect("ch");
        assert!(ch.value.coef[0].is_finite());
        let twm = TwoWayMundlak::new()
            .fit(&xtw, &ytw, &g, &time, &Session::new("twm", "fit"))
            .expect("twm");
        assert!(twm.value.coef[0].is_finite());
        let mg = MeanGroup::new()
            .fit(&x, &y, &g, &Session::new("mg", "fit"))
            .expect("mg");
        assert!((mg.value.coef[0] - 2.0).abs() < 0.25, "mg={}", mg.value.coef[0]);
        let ecfm = EntityClusteredFamaMacBeth::new()
            .fit(&xfm, &y, &time, &g, &Session::new("ecfm", "fit"))
            .expect("ecfm");
        assert_eq!(ecfm.value.coef.len(), 1);
        assert!(ecfm.value.coef[0].is_finite());
        assert!(ecfm.value.se[0].is_finite() || ecfm.value.se[0].is_nan());
        let treat = Vector::from_iter((0..32).map(|i| if i / 8 >= 2 { 1.0 } else { 0.0 }));
        let post = Vector::from_iter((0..32).map(|i| if i % 8 >= 4 { 1.0 } else { 0.0 }));
        let did = DiffInDiff::new()
            .fit(&y, &treat, &post, &Session::new("did", "fit"))
            .expect("did");
        assert!(did.value.att.is_finite());
        let ysc = Vector::from_iter((0..8).map(|t| 2.0 * t as f64 + 15.0));
        let dsc = Matrix::from_fn(8, 3, |t, j| 2.0 * t as f64 + 5.0 * j as f64);
        let sc = SyntheticControl::new()
            .fit(&ysc, &dsc, 4, &Session::new("sc", "fit"))
            .expect("sc");
        assert_eq!(sc.value.weights.len(), 3);
        assert!(sc.value.att.is_finite());
    }
}
