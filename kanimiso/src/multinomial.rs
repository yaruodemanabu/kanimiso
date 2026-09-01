//! Softmax multinomial logistic regression (K ≥ 2), Fisher scoring.
//!
//! The last class is the reference. Coefficients are a `(K−1) × p` matrix of
//! log-odds versus that reference. Perfect separation makes the MLE diverge
//! and is treated as a false inferential claim.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::{Fit, Predict};
use crate::validate::{inspect_classes, inspect_identification, inspect_xy};
use ojizou_san::Session;
use signlred::{
    Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result, Severity,
};

/// Softmax multinomial logistic.
#[derive(Clone, Debug)]
pub(crate) struct MultinomialLogistic {
    /// ℓ₂ penalty on every non-reference coefficient block.
    pub c_inv: f64,
    /// Max Fisher-scoring iterations.
    pub max_iter: usize,
    /// Gradient-norm tolerance.
    pub tol: f64,
    /// Prepend an intercept column.
    pub fit_intercept: bool,
}

impl Default for MultinomialLogistic {
    fn default() -> Self {
        Self {
            c_inv: 1e-6,
            max_iter: 80,
            tol: 1e-7,
            fit_intercept: true,
        }
    }
}

impl MultinomialLogistic {
    /// Lightly ridge-regularized multinomial logistic.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Fitted softmax model (reference = last class).
#[derive(Clone, Debug)]
pub(crate) struct FittedMultinomial {
    /// Sorted class labels.
    pub classes: Vec<i64>,
    /// `(K−1) × p_design` coefficients (row k vs the reference).
    pub coef: Matrix,
    /// Whether the first design column is an intercept.
    pub used_intercept: bool,
}

impl FittedMultinomial {
    /// Softmax probabilities, rows sum to 1.
    pub(crate) fn predict_proba(&self, x: &Matrix) -> Matrix {
        let design = if self.used_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        let k = self.classes.len();
        let n = design.nrows();
        let p = self.coef.ncols();
        Matrix::from_fn(n, k, |i, c| {
            let mut logits = vec![0.0; k];
            for r in 0..(k.saturating_sub(1)) {
                let mut s = 0.0;
                for j in 0..p.min(design.ncols()) {
                    s += self.coef.get(r, j) * design.get(i, j);
                }
                logits[r] = s;
            }
            let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut den = 0.0;
            let mut exps = vec![0.0; k];
            for t in 0..k {
                exps[t] = (logits[t] - m).exp();
                den += exps[t];
            }
            if den <= 0.0 {
                1.0 / k as f64
            } else {
                exps[c] / den
            }
        })
    }
}

impl Fit for MultinomialLogistic {
    type Fitted = FittedMultinomial;
    fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMultinomial>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let k = classes.len();
        let (n, p) = design.shape();
        if k < 2 {
            return ctx.finish(FittedMultinomial {
                classes,
                coef: Matrix::zeros(0, p),
                used_intercept: self.fit_intercept,
            });
        }
        // One-hot of non-reference classes
        let mut yoh = Matrix::zeros(n, k - 1);
        for i in 0..n {
            let lab = y[i].round() as i64;
            if let Some(c) = classes.iter().position(|&z| z == lab) {
                if c + 1 < k {
                    yoh.set(i, c, 1.0);
                }
            }
        }
        // β rows = K-1, cols = p. Joint gradient descent on the softmax NLL
        // (block IRLS ignores off-diagonal Hessian blocks and often diverges).
        let mut beta = Matrix::zeros(k - 1, p);
        let mut converged = false;
        let n_f = n as f64;
        for it in 0..self.max_iter {
            let mut grad = Matrix::zeros(k - 1, p);
            let mut sep = 0usize;
            let mut nll = 0.0;
            for i in 0..n {
                let mut logits = vec![0.0; k];
                for rr in 0..(k - 1) {
                    let mut s = 0.0;
                    for j in 0..p {
                        s += beta.get(rr, j) * design.get(i, j);
                    }
                    logits[rr] = s;
                }
                let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut den = 0.0;
                let mut exps = vec![0.0; k];
                for t in 0..k {
                    exps[t] = (logits[t] - m).exp();
                    den += exps[t];
                }
                let inv_den = if den > 0.0 { 1.0 / den } else { 1.0 / k as f64 };
                let lab = y[i].round() as i64;
                let yi = classes.iter().position(|&z| z == lab).unwrap_or(k - 1);
                nll -= (exps[yi] * inv_den).max(1e-15).ln();
                for r in 0..(k - 1) {
                    let pi = exps[r] * inv_den;
                    if (pi - yoh.get(i, r)).abs() < 1e-12 && logits[r].abs() > 20.0 {
                        sep += 1;
                    }
                    let gi = pi - yoh.get(i, r);
                    for j in 0..p {
                        grad.set(r, j, grad.get(r, j) + gi * design.get(i, j));
                    }
                }
            }
            let mut gnorm = 0.0;
            for r in 0..(k - 1) {
                for j in 0..p {
                    let g = grad.get(r, j) / n_f + self.c_inv * beta.get(r, j);
                    grad.set(r, j, g);
                    gnorm += g * g;
                }
            }
            gnorm = gnorm.sqrt();
            ctx.session.step(it as u64, nll, Some(gnorm));
            if sep == n * (k - 1) {
                ctx.push(
                    Issue::builder(IssueCode::PerfectSeparation)
                        .message("softmax MLE is separated; finite coefficients are an artifact")
                        .meaninglessness(Meaninglessness {
                            what_was_computed: "multinomial logistic MLE".into(),
                            why_meaningless:
                                "perfect separation makes log-odds infinite; iteration just stops"
                                    .into(),
                            interpretive_value: signlred::InterpretiveValue::False,
                            suggested_action: "penalize, or use a separable-aware estimator".into(),
                        })
                        .build(),
                );
                break;
            }
            let step = 0.8 / (1.0 + 0.05 * it as f64);
            let mut delta = 0.0;
            for r in 0..(k - 1) {
                for j in 0..p {
                    let d = step * grad.get(r, j);
                    beta.set(r, j, beta.get(r, j) - d);
                    delta += d * d;
                }
            }
            if gnorm < self.tol || delta.sqrt() < self.tol {
                ctx.session.converged("softmax gradient descent", it as u64);
                converged = true;
                break;
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(signlred::Severity::Warning)
                    .message("multinomial logistic did not meet the tolerance")
                    .build(),
            );
        }
        if self.c_inv > 0.0 {
            ctx.push(
                Issue::builder(IssueCode::RidgeFallbackUsed)
                    .severity(signlred::Severity::Advisory)
                    .message(format!(
                        "softmax blocks use ℓ₂={}; this is not the unpenalized MLE",
                        self.c_inv
                    ))
                    .compromise(NumericalCompromise::new(
                        "unpenalized multinomial MLE",
                        format!("softmax gradient descent with ridge {}", self.c_inv),
                        "unregularized softmax Hessians are often singular",
                        "odds ratios are shrunk; say so",
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedMultinomial {
            classes,
            coef: beta,
            used_intercept: self.fit_intercept,
        })
    }
}

impl Predict for FittedMultinomial {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        let proba = self.predict_proba(x);
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut best = 0usize;
            let mut bp = f64::NEG_INFINITY;
            for c in 0..self.classes.len() {
                let p = proba.get(i, c);
                if p > bp {
                    bp = p;
                    best = c;
                }
            }
            self.classes[best] as f64
        }));
        ctx.finish(out)
    }
}

/// statsmodels `MNLogit` name around [`MultinomialLogistic`].
///
/// Records that IIA is assumed and not tested.
#[derive(Clone, Debug)]
pub(crate) struct MnLogit {
    /// ℓ₂ penalty on every non-reference coefficient block.
    pub c_inv: f64,
    /// Max Fisher-scoring iterations.
    pub max_iter: usize,
    /// Gradient-norm tolerance.
    pub tol: f64,
    /// Prepend an intercept column.
    pub fit_intercept: bool,
}

impl Default for MnLogit {
    fn default() -> Self {
        Self {
            c_inv: 1e-6,
            max_iter: 80,
            tol: 1e-7,
            fit_intercept: true,
        }
    }
}

impl MnLogit {
    /// Default MNLogit.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl Fit for MnLogit {
    type Fitted = FittedMultinomial;
    fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMultinomial>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message(
                    "MNLogit assumes IIA; independence of irrelevant alternatives is not tested",
                )
                .build(),
        );
        let mut inner = MultinomialLogistic {
            c_inv: self.c_inv,
            max_iter: self.max_iter,
            tol: self.tol,
            fit_intercept: self.fit_intercept,
        };
        let q = inner.fit(x, y, &session.child("softmax"))?;
        for issue in q.report.issues() {
            ctx.push(issue.clone());
        }
        ctx.finish(q.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_class_separable_recovers_labels() {
        let x = Matrix::from_fn(30, 2, |i, j| {
            let g = (i / 10) as f64;
            if j == 0 {
                3.0 * g + 0.25 * ((i % 10) as f64 - 4.5)
            } else {
                2.0 * g + 0.15 * ((i % 10) as f64 - 4.5)
            }
        });
        let y = Vector::from_iter((0..30).map(|i| (i / 10) as f64));
        let q = MultinomialLogistic::new()
            .fit(&x, &y, &Session::new("mn", "fit"))
            .expect("mn");
        assert_eq!(q.value.classes.len(), 3);
        let pred = q.value.predict(&x, &Session::new("mn", "p")).unwrap().value;
        let mut ok = 0;
        for i in 0..30 {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 22, "ok={ok}");
        let mn = MnLogit::new()
            .fit(&x, &y, &Session::new("mnlogit", "fit"))
            .expect("mnlogit");
        assert_eq!(mn.value.classes.len(), 3);
    }
}
