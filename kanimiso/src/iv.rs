//! Instrumental variables (2SLS), Newey–West HAC, and Engle–Granger cointegration.
//!
//! A first-stage \(F < 10\) is a weak-instrument finding: the 2SLS point is
//! algebraically defined and inferentially misleading. HAC covariances are not
//! OLS SEs and say so.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::least_squares;
use crate::stats::adfuller;
use crate::traits::Predict;
use crate::validate::{inspect_identification, inspect_xy};
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
}
