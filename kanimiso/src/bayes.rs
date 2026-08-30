//! Bayesian ridge (MacKay evidence approximation; sklearn `BayesianRidge`).
//!
//! The noise precision \(\alpha\) and weight precision \(\lambda\) are updated
//! from the SVD of the design. A constant target leaves both precisions
//! unidentified.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::thin_svd;
use crate::traits::{Fit, Predict};
use crate::validate::{inspect_identification, inspect_xy};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, NumericalCompromise, Qualified, Result};

/// Bayesian ridge regression (automatic relevance of the noise / weight precisions).
#[derive(Clone, Debug)]
pub struct BayesianRidge {
    /// Max evidence iterations.
    pub max_iter: usize,
    /// Prepend an intercept.
    pub fit_intercept: bool,
}

impl Default for BayesianRidge {
    fn default() -> Self {
        Self {
            max_iter: 40,
            fit_intercept: true,
        }
    }
}

impl BayesianRidge {
    /// Default Bayesian ridge.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Bayesian ridge.
#[derive(Clone, Debug)]
pub struct FittedBayesianRidge {
    /// Slopes.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
    /// Noise precision \(\alpha\).
    pub alpha: f64,
    /// Weight precision \(\lambda\).
    pub lambda: f64,
}

impl Fit for BayesianRidge {
    type Fitted = FittedBayesianRidge;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedBayesianRidge>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if ctx.report.contains(IssueCode::ConstantTarget)
            || ctx.report.contains(IssueCode::EmptyMatrix)
        {
            return ctx.finish(FittedBayesianRidge {
                coef: Vector::zeros(x.ncols()),
                intercept: y.mean(),
                alpha: f64::NAN,
                lambda: f64::NAN,
            });
        }
        let (xc, xmean) = if self.fit_intercept {
            x.centered()
        } else {
            (x.clone(), Vector::zeros(x.ncols()))
        };
        let ymean = if self.fit_intercept { y.mean() } else { 0.0 };
        let yc = Vector::from_iter(y.as_slice().iter().map(|&v| v - ymean));
        inspect_identification(&mut ctx.report, xc.nrows(), xc.ncols(), &ctx.policy);
        let Some(svd) = thin_svd(&mut ctx.report, &xc, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::SvdDidNotConverge)
                    .message("BayesianRidge SVD failed")
                    .build(),
            );
            return ctx.finish(FittedBayesianRidge {
                coef: Vector::zeros(x.ncols()),
                intercept: ymean,
                alpha: f64::NAN,
                lambda: f64::NAN,
            });
        };
        let n = xc.nrows() as f64;
        let p = xc.ncols();
        let mut alpha = 1.0;
        let mut lambda = 1.0;
        let mut coef = Vector::zeros(p);
        let mut converged = false;
        for it in 0..self.max_iter.max(1) {
            // μ = Σ_j (α σ_j / (α σ_j² + λ)) (u_jᵀ y) v_j
            let mut m = Vector::zeros(p);
            let mut gamma = 0.0;
            for k in 0..svd.singular_values.len() {
                let s = svd.singular_values[k];
                let s2 = s * s;
                let w = alpha * s2 + lambda;
                if w <= 1e-18 {
                    continue;
                }
                gamma += alpha * s2 / w;
                let mut uty = 0.0;
                for i in 0..yc.len().min(svd.u.nrows()) {
                    uty += svd.u[(i, k)] * yc[i];
                }
                let scale = (alpha * s / w) * uty;
                for j in 0..p.min(svd.v.nrows()) {
                    m[j] += svd.v[(j, k)] * scale;
                }
            }
            let pred = xc.matvec(&m);
            let mut sse = 0.0;
            for i in 0..yc.len() {
                let e = yc[i] - pred[i];
                sse += e * e;
            }
            let coef_n2 = m.dot(&m);
            let alpha_n = ((n - gamma) / sse.max(1e-18)).max(1e-12);
            let lambda_n = (gamma / coef_n2.max(1e-18)).max(1e-12);
            let da = (alpha_n - alpha).abs();
            alpha = alpha_n;
            lambda = lambda_n;
            coef = m;
            ctx.session.step(it as u64, sse, Some(da));
            if da < 1e-8 && it > 0 {
                ctx.session.converged("BayesianRidge evidence", it as u64);
                converged = true;
                break;
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(signlred::Severity::Warning)
                    .message("BayesianRidge evidence iteration did not meet the tolerance")
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::RidgeFallbackUsed)
                .severity(signlred::Severity::Advisory)
                .message(format!(
                    "BayesianRidge reports α={alpha:.4e} λ={lambda:.4e}; this is not OLS"
                ))
                .compromise(NumericalCompromise::new(
                    "unregularized OLS",
                    "MacKay type-II maximum likelihood of (α, λ)",
                    "the posterior mean is a ridge estimator with a data-chosen penalty",
                    "coefficients are shrunk; SEs from OLS are the wrong estimand",
                ))
                .build(),
        );
        let mut intercept = ymean;
        if self.fit_intercept {
            for j in 0..p {
                intercept -= xmean[j] * coef[j];
            }
        }
        ctx.finish(FittedBayesianRidge {
            coef,
            intercept,
            alpha,
            lambda,
        })
    }
}

impl Predict for FittedBayesianRidge {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("BayesianRidge predict column count ≠ coef")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayesian_ridge_recovers_a_line() {
        let x = Matrix::from_fn(20, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..20).map(|i| 1.0 + 2.0 * i as f64));
        let q = BayesianRidge::new()
            .fit(&x, &y, &Session::new("br", "fit"))
            .expect("br");
        assert!(
            (q.value.coef[0] - 2.0).abs() < 0.05,
            "{:?}",
            q.value.coef.as_slice()
        );
        assert!(q.value.alpha.is_finite() && q.value.lambda.is_finite());
    }
}
