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
use crate::traits::Predict;
use crate::validate::inspect_xy;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, Qualified, Result, Severity};
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
    }
}
