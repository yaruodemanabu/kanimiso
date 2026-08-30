//! Instrumental variables (2SLS), Newey–West HAC, and Engle–Granger cointegration.
//!
//! A first-stage \(F < 10\) is a weak-instrument finding: the 2SLS point is
//! algebraically defined and inferentially misleading. HAC covariances are not
//! OLS SEs and say so.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{chol_solve, least_squares};
use crate::stats::adfuller;
use crate::traits::Predict;
use crate::validate::{inspect_identification, inspect_xy};
use faer::Mat;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result};

/// Two-stage least squares: \(X\) endogenous, \(Z\) instruments.
#[derive(Clone, Debug, Default)]
pub struct TwoSls {
    /// Include an intercept in both stages.
    pub fit_intercept: bool,
}

impl TwoSls {
    /// Default 2SLS.
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
        }
    }

    /// Fit `y` on `x` using instruments `z`.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        z: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedTwoSls>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_xy(&mut ctx.report, z, None, &ctx.policy);
        if z.nrows() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("instrument rows ≠ n")
                    .build(),
            );
            return ctx.finish(empty_2sls(x));
        }
        if z.ncols() < x.ncols() {
            ctx.push(
                Issue::builder(IssueCode::UnderdeterminedSystem)
                    .message(format!(
                        "order condition fails: {} instruments < {} endogenous columns",
                        z.ncols(),
                        x.ncols()
                    ))
                    .meaninglessness(Meaninglessness::vacuous(
                        "2SLS coefficients",
                        "fewer instruments than endogenous regressors leaves β unidentified",
                        "add instruments, or drop endogenous columns",
                    ))
                    .build(),
            );
            return ctx.finish(empty_2sls(x));
        }
        let zdes = if self.fit_intercept {
            z.with_intercept()
        } else {
            z.clone()
        };
        inspect_identification(&mut ctx.report, zdes.nrows(), zdes.ncols(), &ctx.policy);
        let mut xhat = Matrix::zeros(x.nrows(), x.ncols());
        let mut min_f = f64::INFINITY;
        for j in 0..x.ncols() {
            let xj = x.column(j);
            let mut scratch = signlred::Report::new("2sls", "stage1");
            let g_opt = least_squares(&mut scratch, &zdes, &xj, &ctx.policy);
            for issue in scratch.issues() {
                if issue.code == IssueCode::ResidualTooLarge {
                    continue;
                }
                ctx.push(issue.clone());
            }
            let Some(g) = g_opt else {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message(format!("first-stage OLS failed for column {j}"))
                        .build(),
                );
                continue;
            };
            let fit = zdes.matvec(&g);
            for i in 0..x.nrows() {
                xhat.set(i, j, fit[i]);
            }
            let mut sse = 0.0;
            let mut sst = 0.0;
            let m = xj.mean();
            for i in 0..xj.len() {
                let e = xj[i] - fit[i];
                sse += e * e;
                let d = xj[i] - m;
                sst += d * d;
            }
            let k = zdes.ncols().saturating_sub(1).max(1) as f64;
            let df = (x.nrows() as f64 - zdes.ncols() as f64).max(1.0);
            let f = if sse > 0.0 {
                ((sst - sse) / k) / (sse / df)
            } else {
                f64::INFINITY
            };
            min_f = min_f.min(f);
        }
        if min_f < 10.0 {
            ctx.push(
                Issue::builder(IssueCode::CausalClaimUnidentified)
                    .message(format!(
                        "weak instruments: min first-stage F={min_f:.4e} < 10"
                    ))
                    .meaninglessness(Meaninglessness::new(
                        "2SLS coefficient",
                        "a weak first stage makes the IV estimand concentrated-asymptotically biased toward OLS",
                        signlred::InterpretiveValue::Misleading,
                        "report the first-stage F; do not treat 2SLS as identified",
                    ))
                    .metric("min_first_stage_F", min_f)
                    .build(),
            );
        }
        let design = if self.fit_intercept {
            xhat.with_intercept()
        } else {
            xhat.clone()
        };
        let Some(beta) = least_squares(&mut ctx.report, &design, y, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("second-stage OLS failed")
                    .build(),
            );
            return ctx.finish(empty_2sls(x));
        };
        let (intercept, coef) = if self.fit_intercept {
            (beta[0], Vector::from_iter((1..beta.len()).map(|j| beta[j])))
        } else {
            (0.0, beta)
        };
        ctx.finish(FittedTwoSls {
            coef,
            intercept,
            first_stage_f: min_f,
        })
    }
}

/// Fitted 2SLS.
#[derive(Clone, Debug)]
pub struct FittedTwoSls {
    /// Second-stage slopes.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
    /// Smallest first-stage \(F\).
    pub first_stage_f: f64,
}

impl Predict for FittedTwoSls {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(signlred::Severity::Advisory)
                .message("2SLS predict uses the structural X, not the first-stage projection")
                .build(),
        );
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            y[i] += self.intercept;
        }
        ctx.finish(y)
    }
}

fn empty_2sls(x: &Matrix) -> FittedTwoSls {
    FittedTwoSls {
        coef: Vector::zeros(x.ncols()),
        intercept: 0.0,
        first_stage_f: f64::NAN,
    }
}

fn stage1_xhat(ctx: &mut FitCtx, x: &Matrix, zdes: &Matrix, fit_intercept: bool) -> (Matrix, f64) {
    let mut xhat = Matrix::zeros(x.nrows(), x.ncols());
    let mut min_f = f64::INFINITY;
    for j in 0..x.ncols() {
        let xj = x.column(j);
        let mut scratch = signlred::Report::new("3sls", "stage1");
        let g_opt = least_squares(&mut scratch, zdes, &xj, &ctx.policy);
        for issue in scratch.issues() {
            if issue.code == IssueCode::ResidualTooLarge {
                continue;
            }
            ctx.push(issue.clone());
        }
        let Some(g) = g_opt else {
            continue;
        };
        let fit = zdes.matvec(&g);
        for i in 0..x.nrows() {
            xhat.set(i, j, fit[i]);
        }
        let mut sse = 0.0;
        let mut sst = 0.0;
        let m = xj.mean();
        for i in 0..xj.len() {
            let e = xj[i] - fit[i];
            sse += e * e;
            let d = xj[i] - m;
            sst += d * d;
        }
        let k = zdes.ncols().saturating_sub(1).max(1) as f64;
        let df = (x.nrows() as f64 - zdes.ncols() as f64).max(1.0);
        let f = if sse > 0.0 {
            ((sst - sse) / k) / (sse / df)
        } else {
            f64::INFINITY
        };
        min_f = min_f.min(f);
    }
    let design = if fit_intercept {
        xhat.with_intercept()
    } else {
        xhat
    };
    (design, min_f)
}

/// Three-stage least squares for a two-equation system (Zellner–Theil).
///
/// Each equation is 2SLS-projected, the residual covariance \(\Sigma\) is
/// formed, and the stacked system is GLS-solved with \(\Sigma^{-1}\).
#[derive(Clone, Debug, Default)]
pub struct ThreeSls {
    /// Intercept in both stages.
    pub fit_intercept: bool,
}

impl ThreeSls {
    /// Default 3SLS.
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
        }
    }

    /// Fit `(y1, x1)` and `(y2, x2)` with instruments `z1`, `z2`.
    pub fn fit(
        &mut self,
        y1: &Vector,
        x1: &Matrix,
        z1: &Matrix,
        y2: &Vector,
        x2: &Matrix,
        z2: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedThreeSls>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x1, Some(y1), &ctx.policy);
        inspect_xy(&mut ctx.report, x2, Some(y2), &ctx.policy);
        inspect_xy(&mut ctx.report, z1, None, &ctx.policy);
        inspect_xy(&mut ctx.report, z2, None, &ctx.policy);
        let n = y1.len().min(y2.len()).min(x1.nrows()).min(x2.nrows());
        if n == 0 {
            return ctx.finish(FittedThreeSls {
                eq1: empty_2sls(x1),
                eq2: empty_2sls(x2),
                sigma: Matrix::zeros(2, 2),
            });
        }
        let z1d = if self.fit_intercept {
            z1.with_intercept()
        } else {
            z1.clone()
        };
        let z2d = if self.fit_intercept {
            z2.with_intercept()
        } else {
            z2.clone()
        };
        let (xh1, f1) = stage1_xhat(&mut ctx, x1, &z1d, self.fit_intercept);
        let (xh2, f2) = stage1_xhat(&mut ctx, x2, &z2d, self.fit_intercept);
        let min_f = f1.min(f2);
        if min_f < 10.0 {
            ctx.push(
                Issue::builder(IssueCode::CausalClaimUnidentified)
                    .message(format!(
                        "3SLS weak instruments: min first-stage F={min_f:.4e}"
                    ))
                    .meaninglessness(Meaninglessness::new(
                        "3SLS coefficients",
                        "a weak first stage makes the GLS step concentrated-asymptotically biased",
                        signlred::InterpretiveValue::Misleading,
                        "report the first-stage F; do not treat 3SLS as identified",
                    ))
                    .metric("min_first_stage_F", min_f)
                    .build(),
            );
        }
        let mut scratch = signlred::Report::new("3sls", "eq");
        let b1 = least_squares(&mut scratch, &xh1, y1, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(xh1.ncols()));
        let b2 = least_squares(&mut scratch, &xh2, y2, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(xh2.ncols()));
        let e1 = y1.sub(&xh1.matvec(&b1));
        let e2 = y2.sub(&xh2.matvec(&b2));
        let nf = n as f64;
        let s11 = e1.dot(&e1) / nf;
        let s22 = e2.dot(&e2) / nf;
        let s12 = e1.dot(&e2) / nf;
        let det = s11 * s22 - s12 * s12;
        let (w11, w12, w22) = if det.abs() <= 1e-14 {
            ctx.push(
                Issue::builder(IssueCode::JitterInjected)
                    .message("3SLS residual Σ is singular; a 1e-8 jitter was added")
                    .compromise(NumericalCompromise::new(
                        "Σ^{-1} GLS",
                        "Σ + 1e-8 I",
                        "the two equation residuals are linearly dependent",
                        "the GLS step is close to equation-by-equation 2SLS",
                    ))
                    .build(),
            );
            let d = (s11 + 1e-8) * (s22 + 1e-8) - s12 * s12;
            ((s22 + 1e-8) / d, -s12 / d, (s11 + 1e-8) / d)
        } else {
            (s22 / det, -s12 / det, s11 / det)
        };
        let p1 = xh1.ncols();
        let p2 = xh2.ncols();
        let p = p1 + p2;
        let mut gram = Mat::<f64>::zeros(p, p);
        let mut rhs = Vector::zeros(p);
        for a in 0..p1 {
            for b in 0..p1 {
                let mut s = 0.0;
                for i in 0..n {
                    s += xh1.get(i, a) * xh1.get(i, b);
                }
                gram[(a, b)] += w11 * s;
            }
            for b in 0..p2 {
                let mut s = 0.0;
                for i in 0..n {
                    s += xh1.get(i, a) * xh2.get(i, b);
                }
                gram[(a, p1 + b)] += w12 * s;
                gram[(p1 + b, a)] += w12 * s;
            }
            let mut s = 0.0;
            for i in 0..n {
                s += xh1.get(i, a) * (w11 * y1[i] + w12 * y2[i]);
            }
            rhs[a] = s;
        }
        for a in 0..p2 {
            for b in 0..p2 {
                let mut s = 0.0;
                for i in 0..n {
                    s += xh2.get(i, a) * xh2.get(i, b);
                }
                gram[(p1 + a, p1 + b)] += w22 * s;
            }
            let mut s = 0.0;
            for i in 0..n {
                s += xh2.get(i, a) * (w12 * y1[i] + w22 * y2[i]);
            }
            rhs[p1 + a] = s;
        }
        let mut scratch2 = signlred::Report::new("3sls", "gls");
        let beta = chol_solve(&mut scratch2, &gram, &rhs, &ctx.policy).unwrap_or_else(|| {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("3SLS GLS Cholesky failed")
                    .build(),
            );
            Vector::zeros(p)
        });
        for issue in scratch2.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let (i1, c1) = if self.fit_intercept && p1 > 0 {
            (beta[0], Vector::from_iter((1..p1).map(|j| beta[j])))
        } else {
            (0.0, Vector::from_iter((0..p1).map(|j| beta[j])))
        };
        let (i2, c2) = if self.fit_intercept && p2 > 0 {
            (beta[p1], Vector::from_iter((p1 + 1..p).map(|j| beta[j])))
        } else {
            (0.0, Vector::from_iter((p1..p).map(|j| beta[j])))
        };
        let mut sigma = Matrix::zeros(2, 2);
        sigma.set(0, 0, s11);
        sigma.set(0, 1, s12);
        sigma.set(1, 0, s12);
        sigma.set(1, 1, s22);
        ctx.finish(FittedThreeSls {
            eq1: FittedTwoSls {
                coef: c1,
                intercept: i1,
                first_stage_f: f1,
            },
            eq2: FittedTwoSls {
                coef: c2,
                intercept: i2,
                first_stage_f: f2,
            },
            sigma,
        })
    }
}

/// Fitted two-equation 3SLS.
#[derive(Clone, Debug)]
pub struct FittedThreeSls {
    /// First equation.
    pub eq1: FittedTwoSls,
    /// Second equation.
    pub eq2: FittedTwoSls,
    /// Residual covariance \(\Sigma\).
    pub sigma: Matrix,
}

/// Newey–West HAC covariance of OLS scores \(X_i e_i\).
///
/// Returns a \(p \times p\) matrix. The OLS Hessian inverse is **not** applied;
/// callers who want \(\mathrm{Var}(\hat\beta)\) should sandwich this between
/// \((X'X)^{-1}\).
pub fn newey_west(scores: &Matrix, lags: usize, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, scores, None, &ctx.policy);
    let (n, p) = scores.shape();
    if n == 0 || p == 0 {
        return ctx.finish(Matrix::zeros(p, p));
    }
    let l = lags.min(n.saturating_sub(1));
    if l == 0 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(signlred::Severity::Advisory)
                .message("Newey–West with 0 lags is just the Eicker–White meat")
                .compromise(NumericalCompromise::new(
                    "HAC with a data-driven bandwidth",
                    "Γ₀ only",
                    "the caller requested L=0",
                    "serial correlation is not corrected",
                ))
                .build(),
        );
    }
    let mut s = Matrix::zeros(p, p);
    for i in 0..n {
        for a in 0..p {
            for b in 0..p {
                s.set(a, b, s.get(a, b) + scores.get(i, a) * scores.get(i, b));
            }
        }
    }
    for lag in 1..=l {
        let w = 1.0 - lag as f64 / (l as f64 + 1.0);
        let mut g = Matrix::zeros(p, p);
        for i in lag..n {
            for a in 0..p {
                for b in 0..p {
                    g.set(
                        a,
                        b,
                        g.get(a, b) + scores.get(i, a) * scores.get(i - lag, b),
                    );
                }
            }
        }
        for a in 0..p {
            for b in 0..p {
                let v = s.get(a, b) + w * (g.get(a, b) + g.get(b, a));
                s.set(a, b, v);
            }
        }
    }
    let inv_n = 1.0 / n as f64;
    for a in 0..p {
        for b in 0..p {
            s.set(a, b, s.get(a, b) * inv_n);
        }
    }
    ctx.finish(s)
}

/// Engle–Granger residual-based cointegration test.
#[derive(Clone, Debug)]
pub struct CointEngleGranger {
    /// ADF lags (`None` ⇒ Schwert).
    pub lags: Option<usize>,
}

impl Default for CointEngleGranger {
    fn default() -> Self {
        Self { lags: None }
    }
}

impl CointEngleGranger {
    /// Default Engle–Granger test.
    pub fn new() -> Self {
        Self::default()
    }

    /// OLS `y` on `x` (with intercept), then ADF on the residual.
    pub fn fit(&self, y: &Vector, x: &Vector, session: &Session) -> Result<Qualified<CointResult>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let xm = Matrix::from_vector(x).with_intercept();
        inspect_xy(&mut ctx.report, &xm, Some(y), &ctx.policy);
        let Some(beta) = least_squares(&mut ctx.report, &xm, y, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("Engle–Granger first-step OLS failed")
                    .build(),
            );
            return ctx.finish(CointResult {
                coef: 0.0,
                intercept: 0.0,
                adf_stat: f64::NAN,
                adf_pvalue: f64::NAN,
            });
        };
        let resid = y.sub(&xm.matvec(&beta));
        let adf = adfuller(&resid, self.lags, &session.child("adf"))?;
        if adf.value.pvalue > 0.05 {
            ctx.push(
                Issue::builder(IssueCode::NonStationary)
                    .severity(signlred::Severity::Warning)
                    .message(format!(
                        "Engle–Granger residual ADF p={:.4e}; no cointegration at 5%",
                        adf.value.pvalue
                    ))
                    .meaninglessness(Meaninglessness::new(
                        "cointegrating slope",
                        "the residual still looks like a unit root, so β is a spurious-regression coefficient",
                        signlred::InterpretiveValue::Misleading,
                        "do not interpret the OLS slope as a long-run relation",
                    ))
                    .build(),
            );
        }
        ctx.finish(CointResult {
            coef: beta[1],
            intercept: beta[0],
            adf_stat: adf.value.stat,
            adf_pvalue: adf.value.pvalue,
        })
    }
}

/// Engle–Granger result.
#[derive(Clone, Debug)]
pub struct CointResult {
    /// OLS slope of \(y\) on \(x\).
    pub coef: f64,
    /// OLS intercept.
    pub intercept: f64,
    /// ADF statistic on the residual.
    pub adf_stat: f64,
    /// ADF p-value (MacKinnon approximation for a *unit-root* residual, not the EG critical values).
    pub adf_pvalue: f64,
}

/// Eicker–Huber–White sandwich kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandwichKind {
    /// HC0: \(\mathrm{meat} = \sum e_i^2 x_i x_i'\).
    Hc0,
    /// HC3: divide by \((1-h_{ii})^2\).
    Hc3,
}

/// OLS sandwich covariance \((X'X)^{-1} \mathrm{meat} (X'X)^{-1}\).
///
/// `x` must already include an intercept column if one was used in the fit.
pub fn sandwich_hc(
    x: &Matrix,
    resid: &Vector,
    kind: SandwichKind,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    if let Some(issue) = signlred::scan_finite(resid.as_slice()).to_issue("resid") {
        ctx.push(issue);
    }
    if resid.len() != x.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("sandwich residual length ≠ n")
                .build(),
        );
        return ctx.finish(Matrix::zeros(x.ncols(), x.ncols()));
    }
    let p = x.ncols();
    let n = x.nrows();
    if n == 0 || p == 0 {
        return ctx.finish(Matrix::zeros(p, p));
    }
    if resid
        .as_slice()
        .iter()
        .all(|e| e.abs() <= ctx.policy.near_zero_variance)
    {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("sandwich meat is 0 because every residual is ~0")
                .build(),
        );
    }
    let xtx = x.gram();
    let mut inv = Mat::<f64>::zeros(p, p);
    for j in 0..p {
        let mut e = Vector::zeros(p);
        e[j] = 1.0;
        let mut scratch = signlred::Report::new("hc", "xtx");
        match chol_solve(&mut scratch, &xtx, &e, &ctx.policy) {
            Some(col) => {
                for i in 0..p {
                    inv[(i, j)] = col[i];
                }
            }
            None => {
                ctx.push(
                    Issue::builder(IssueCode::CholeskyFailed)
                        .message("X'X refused Cholesky; sandwich is unidentified")
                        .meaninglessness(Meaninglessness::vacuous(
                            "sandwich covariance",
                            "X'X is not SPD so (X'X)⁻¹ does not exist",
                            "drop collinear columns",
                        ))
                        .build(),
                );
                return ctx.finish(Matrix::zeros(p, p));
            }
        }
    }
    let mut meat = Matrix::zeros(p, p);
    for i in 0..n {
        let mut h = 0.0;
        for a in 0..p {
            let mut s = 0.0;
            for b in 0..p {
                s += inv[(a, b)] * x.get(i, b);
            }
            h += x.get(i, a) * s;
        }
        let mut e2 = resid[i] * resid[i];
        if kind == SandwichKind::Hc3 {
            let den = (1.0 - h).max(1e-12);
            if (1.0 - h).abs() < 1e-8 {
                ctx.push(
                    Issue::builder(IssueCode::LeveragePoint)
                        .message(format!("row {i} has leverage h={h:.4}; HC3 is inflated"))
                        .build(),
                );
            }
            e2 /= den * den;
        }
        for a in 0..p {
            for b in 0..p {
                meat.set(a, b, meat.get(a, b) + e2 * x.get(i, a) * x.get(i, b));
            }
        }
    }
    let mut out = Matrix::zeros(p, p);
    for a in 0..p {
        for b in 0..p {
            let mut s = 0.0;
            for k in 0..p {
                let mut t = 0.0;
                for m in 0..p {
                    t += meat.get(k, m) * inv[(m, b)];
                }
                s += inv[(a, k)] * t;
            }
            out.set(a, b, s);
        }
    }
    ctx.push(
        Issue::builder(IssueCode::Heteroscedasticity)
            .severity(signlred::Severity::Advisory)
            .message(match kind {
                SandwichKind::Hc0 => "HC0 sandwich is not the OLS information covariance",
                SandwichKind::Hc3 => "HC3 sandwich inflates levered rows; it is not HC0 or OLS",
            })
            .compromise(NumericalCompromise::new(
                "model-based OLS covariance σ²(X'X)⁻¹",
                format!("{kind:?} sandwich"),
                "the meat uses squared residuals",
                "do not interpret these as Gauss–Markov SEs",
            ))
            .build(),
    );
    ctx.finish(out)
}

/// HC0 sandwich.
pub fn hc0(x: &Matrix, resid: &Vector, session: &Session) -> Result<Qualified<Matrix>> {
    sandwich_hc(x, resid, SandwichKind::Hc0, session)
}

/// HC3 sandwich.
pub fn hc3(x: &Matrix, resid: &Vector, session: &Session) -> Result<Qualified<Matrix>> {
    sandwich_hc(x, resid, SandwichKind::Hc3, session)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twosls_recovers_when_z_is_x() {
        let x = Matrix::from_fn(20, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..20).map(|i| 1.0 + 2.0 * i as f64));
        let q = TwoSls::new()
            .fit(&x, &y, &x, &Session::new("iv", "fit"))
            .expect("2sls");
        assert!((q.value.coef[0] - 2.0).abs() < 1e-8);
        assert!(q.value.first_stage_f.is_infinite() || q.value.first_stage_f > 100.0);
    }

    #[test]
    fn newey_west_is_psd_on_white_scores() {
        let s = Matrix::from_fn(12, 2, |i, j| if j == 0 { 1.0 } else { (i as f64) - 5.5 });
        let q = newey_west(&s, 2, &Session::new("hac", "fit")).expect("nw");
        assert_eq!(q.value.shape(), (2, 2));
        assert!(q.value.get(0, 0) >= 0.0);
    }

    #[test]
    fn hc0_is_psd_on_a_line() {
        let x = Matrix::from_fn(16, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let e = Vector::from_iter((0..16).map(|i| 0.1 * ((i % 3) as f64 - 1.0)));
        let q = hc0(&x, &e, &Session::new("hc", "0")).expect("hc0");
        assert!(q.value.get(0, 0).is_finite());
        assert!(q.value.get(0, 0) >= 0.0);
        let q3 = hc3(&x, &e, &Session::new("hc", "3")).expect("hc3");
        assert_eq!(q3.value.shape(), (2, 2));
    }

    #[test]
    fn threesls_recovers_when_z_is_x() {
        let x = Matrix::from_fn(24, 1, |i, _| i as f64);
        let y1 = Vector::from_iter((0..24).map(|i| 1.0 + 2.0 * i as f64));
        let y2 = Vector::from_iter((0..24).map(|i| 0.5 + 1.5 * i as f64));
        let q = ThreeSls::new()
            .fit(&y1, &x, &x, &y2, &x, &x, &Session::new("3sls", "fit"))
            .expect("3sls");
        assert!(
            (q.value.eq1.coef[0] - 2.0).abs() < 0.05,
            "b1={}",
            q.value.eq1.coef[0]
        );
        assert!(
            (q.value.eq2.coef[0] - 1.5).abs() < 0.05,
            "b2={}",
            q.value.eq2.coef[0]
        );
    }
}
