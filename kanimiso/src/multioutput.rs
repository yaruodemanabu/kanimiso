//! Multi-output regression and classifier chains (sklearn `multioutput`).
//!
//! Each output is a separate identified problem. A constant column of `Y`
//! aborts only that output (Misleading), not the remaining columns.

use crate::classification::{FittedRidgeClassifier, RidgeClassifier};
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linear_model::{FittedPenalized, Ridge};
use crate::traits::{Fit, Predict};
use crate::validate::inspect_xy;
use ojizou_san::Session;
use signlred::{
    Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result, Severity,
};

/// Independent ridge regressor per column of `Y`.
#[derive(Clone, Debug)]
pub struct MultiOutputRegressor {
    /// Shared ℓ₂ penalty.
    pub alpha: f64,
}

impl Default for MultiOutputRegressor {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl MultiOutputRegressor {
    /// Multi-output ridge with the given `α`.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Fit one ridge per column of `y`.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedMultiOutput>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        if y.nrows() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("multi-output Y rows ≠ X rows")
                    .build(),
            );
            return ctx.finish(FittedMultiOutput {
                models: Vec::new(),
                n_outputs: y.ncols(),
            });
        }
        if y.ncols() == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("multi-output Y has 0 columns")
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("outputs are fit independently; residual cross-correlation is ignored")
                .compromise(NumericalCompromise::new(
                    "SUR / multivariate GLS",
                    "separate ridge per column",
                    "the joint residual covariance is not estimated",
                    "do not read the collection of fits as a joint likelihood",
                ))
                .build(),
        );
        let mut models = Vec::with_capacity(y.ncols());
        for j in 0..y.ncols() {
            let yj = y.column(j);
            let st = signlred::slice_stats(yj.as_slice());
            if st.count >= 2 && st.is_constant(ctx.policy.near_zero_variance) {
                ctx.push(
                    Issue::builder(IssueCode::ConstantTarget)
                        .severity(Severity::Warning)
                        .message(format!("output {j} is constant; that column is a mean"))
                        .meaninglessness(Meaninglessness::new(
                            format!("output {j} slopes"),
                            "a constant column has no residual variation",
                            signlred::InterpretiveValue::Misleading,
                            "the remaining outputs stay identified",
                        ))
                        .build(),
                );
            }
            match Ridge::new(self.alpha).fit(x, &yj, &session.child(format!("out_{j}"))) {
                Ok(q) => models.push(q.value),
                Err(e) => {
                    ctx.push(e.primary);
                    models.push(FittedPenalized {
                        coef: Vector::zeros(x.ncols()),
                        intercept: yj.mean(),
                        alpha: self.alpha,
                        l1_ratio: 0.0,
                    });
                }
            }
        }
        ctx.finish(FittedMultiOutput {
            models,
            n_outputs: y.ncols(),
        })
    }
}

/// Fitted independent ridges.
#[derive(Clone, Debug)]
pub struct FittedMultiOutput {
    /// One ridge per output column.
    pub models: Vec<FittedPenalized>,
    /// Number of output columns.
    pub n_outputs: usize,
}

impl FittedMultiOutput {
    /// Predict an `n × n_outputs` matrix.
    pub fn predict_matrix(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut out = Matrix::zeros(x.nrows(), self.n_outputs);
        for (j, m) in self.models.iter().enumerate() {
            match m.predict(x, &session.child(format!("p_{j}"))) {
                Ok(q) => {
                    for i in 0..x.nrows().min(q.value.len()) {
                        out.set(i, j, q.value[i]);
                    }
                }
                Err(e) => ctx.push(e.primary),
            }
        }
        ctx.finish(out)
    }
}

/// Binary classifier chain: column `k` sees `X` plus predictions of `0..k-1`.
#[derive(Clone, Debug)]
pub struct ClassifierChain {
    /// Ridge classifier penalty.
    pub alpha: f64,
}

impl Default for ClassifierChain {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl ClassifierChain {
    /// Chain with the given ridge `α`.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Fit on a binary `n × k` label matrix.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedClassifierChain>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        if y.nrows() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("classifier-chain Y rows ≠ X rows")
                    .build(),
            );
            return ctx.finish(FittedClassifierChain {
                models: Vec::new(),
                n_outputs: y.ncols(),
                p: x.ncols(),
            });
        }
        if y.ncols() == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("classifier chain has no labels")
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::TargetLeakageSuspected)
                .severity(Severity::Advisory)
                .message(
                    "later chain members see earlier training labels, not out-of-fold predictions",
                )
                .build(),
        );
        let mut models = Vec::new();
        for k in 0..y.ncols() {
            let xk = Matrix::from_fn(x.nrows(), x.ncols() + k, |i, j| {
                if j < x.ncols() {
                    x.get(i, j)
                } else {
                    y.get(i, j - x.ncols())
                }
            });
            let yk = y.column(k);
            let mut clf = RidgeClassifier::new(self.alpha);
            match clf.fit(&xk, &yk, &session.child(format!("chain_{k}"))) {
                Ok(q) => models.push(q.value),
                Err(e) => {
                    ctx.push(e.primary);
                    models.push(FittedRidgeClassifier::from_penalized(
                        FittedPenalized {
                            coef: Vector::zeros(x.ncols() + k),
                            intercept: 0.0,
                            alpha: self.alpha,
                            l1_ratio: 0.0,
                        },
                        vec![0, 1],
                    ));
                }
            }
        }
        ctx.finish(FittedClassifierChain {
            models,
            n_outputs: y.ncols(),
            p: x.ncols(),
        })
    }
}

/// Fitted classifier chain.
#[derive(Clone, Debug)]
pub struct FittedClassifierChain {
    models: Vec<FittedRidgeClassifier>,
    /// Number of label columns.
    pub n_outputs: usize,
    p: usize,
}

impl FittedClassifierChain {
    /// Sequential binary predictions (`n × k`).
    pub fn predict_matrix(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut prev = Matrix::zeros(x.nrows(), 0);
        let mut out = Matrix::zeros(x.nrows(), self.n_outputs);
        for (k, m) in self.models.iter().enumerate() {
            let xk = Matrix::from_fn(x.nrows(), self.p + k, |i, j| {
                if j < self.p {
                    x.get(i, j)
                } else if j - self.p < prev.ncols() {
                    prev.get(i, j - self.p)
                } else {
                    0.0
                }
            });
            match m.predict(&xk, &session.child(format!("p_{k}"))) {
                Ok(q) => {
                    for i in 0..x.nrows().min(q.value.len()) {
                        out.set(i, k, q.value[i]);
                    }
                }
                Err(e) => ctx.push(e.primary),
            }
            prev = Matrix::from_fn(x.nrows(), k + 1, |i, j| out.get(i, j));
        }
        ctx.finish(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multioutput_recovers_two_lines() {
        let x = Matrix::from_fn(16, 1, |i, _| i as f64);
        let y = Matrix::from_fn(16, 2, |i, j| {
            if j == 0 {
                1.0 + 2.0 * i as f64
            } else {
                3.0 - 0.5 * i as f64
            }
        });
        let q = MultiOutputRegressor::new(0.01)
            .fit(&x, &y, &Session::new("mo", "fit"))
            .expect("mo");
        assert_eq!(q.value.n_outputs, 2);
        assert!((q.value.models[0].coef[0] - 2.0).abs() < 0.1);
        let hat = q
            .value
            .predict_matrix(&x, &Session::new("mo", "p"))
            .unwrap()
            .value;
        assert!((hat.get(5, 0) - y.get(5, 0)).abs() < 0.5);
    }

    #[test]
    fn chain_fits_two_binary_columns() {
        let x = Matrix::from_fn(20, 1, |i, _| if i < 10 { -1.2 } else { 1.2 });
        let y = Matrix::from_fn(20, 2, |i, j| {
            if j == 0 {
                if i < 10 {
                    0.0
                } else {
                    1.0
                }
            } else if i < 8 || i >= 16 {
                0.0
            } else {
                1.0
            }
        });
        let q = ClassifierChain::new(0.5)
            .fit(&x, &y, &Session::new("cc", "fit"))
            .expect("cc");
        let hat = q
            .value
            .predict_matrix(&x, &Session::new("cc", "p"))
            .unwrap()
            .value;
        assert_eq!(hat.shape(), (20, 2));
        assert!(hat.get(0, 0).is_finite());
    }
}
