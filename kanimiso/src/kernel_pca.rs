//! Kernel PCA (Schölkopf, Smola, Müller) on an RBF Gram.
//!
//! The Gram is centred in feature space and factored with
//! the shared internal symmetric eigensolver. Negative eigenvalues are dropped and
//! recorded as [`IssueCode::NegativeEigenvalueDropped`]. A non-PD kernel is
//! [`IssueCode::KernelNotPd`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::symmetric_eigen;
use crate::traits::{FitUnsupervised, Transform};
use crate::validate::{inspect_identification, inspect_xy};
use faer::Mat;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, NumericalCompromise, Qualified, Result};

/// RBF kernel PCA.
#[derive(Clone, Debug)]
pub struct KernelPca {
    /// Number of components.
    pub n_components: usize,
    /// RBF length-scale \(\gamma = 1/(2\ell^2)\).
    pub gamma: f64,
}

impl Default for KernelPca {
    fn default() -> Self {
        Self {
            n_components: 2,
            gamma: 1.0,
        }
    }
}

impl KernelPca {
    /// Keep `k` kernel components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedKernelPca>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted kernel PCA.
#[derive(Clone, Debug)]
pub struct FittedKernelPca {
    x_train: Matrix,
    /// Eigenvectors of the centred Gram (`n` × `k`).
    alphas: Matrix,
    /// Positive eigenvalues used (length `k`).
    pub eigenvalues: Vector,
    /// RBF \(\gamma\).
    pub gamma: f64,
    row_mean: Vector,
    grand_mean: f64,
}

fn rbf(a: &Matrix, i: usize, b: &Matrix, j: usize, gamma: f64) -> f64 {
    let mut d2 = 0.0;
    for c in 0..a.ncols().min(b.ncols()) {
        let d = a.get(i, c) - b.get(j, c);
        d2 += d * d;
    }
    (-gamma.max(1e-12) * d2).exp()
}

fn center_gram(k: &Mat<f64>) -> (Mat<f64>, Vector, f64) {
    let n = k.nrows();
    let mut row_mean = Vector::zeros(n);
    let mut grand = 0.0;
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += k[(i, j)];
        }
        row_mean[i] = s / n.max(1) as f64;
        grand += row_mean[i];
    }
    grand /= n.max(1) as f64;
    let mut kc = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            kc[(i, j)] = k[(i, j)] - row_mean[i] - row_mean[j] + grand;
        }
    }
    (kc, row_mean, grand)
}

impl FitUnsupervised for KernelPca {
    type Fitted = FittedKernelPca;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedKernelPca>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_components.max(1),
            &ctx.policy,
        );
        let n = x.nrows();
        if n == 0 {
            return ctx.finish(FittedKernelPca {
                x_train: x.clone(),
                alphas: Matrix::zeros(0, 0),
                eigenvalues: Vector::zeros(0),
                gamma: self.gamma,
                row_mean: Vector::zeros(0),
                grand_mean: 0.0,
            });
        }
        let mut k = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let v = rbf(x, i, x, j, self.gamma);
                k[(i, j)] = v;
                k[(j, i)] = v;
            }
        }
        let (kc, row_mean, grand) = center_gram(&k);
        let Some((vals, vecs)) = symmetric_eigen(&mut ctx.report, &kc, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::EigenDidNotConverge)
                    .message("KernelPCA eigendecomposition failed")
                    .build(),
            );
            return ctx.finish(FittedKernelPca {
                x_train: x.clone(),
                alphas: Matrix::zeros(n, 0),
                eigenvalues: Vector::zeros(0),
                gamma: self.gamma,
                row_mean,
                grand_mean: grand,
            });
        };
        let mut pairs: Vec<(f64, usize)> = vals
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut dropped = 0usize;
        let mut kept: Vec<(f64, usize)> = Vec::new();
        for (v, i) in pairs {
            if v <= ctx.policy.rank_tol_relative.max(1e-12) {
                if v < -ctx.policy.rank_tol_relative {
                    dropped += 1;
                }
                continue;
            }
            kept.push((v, i));
        }
        if dropped > 0 {
            ctx.push(
                Issue::builder(IssueCode::NegativeEigenvalueDropped)
                    .message(format!(
                        "centred RBF Gram had {dropped} negative eigenvalues; they were dropped"
                    ))
                    .compromise(NumericalCompromise::new(
                        "PSD feature-space covariance",
                        "drop λ<0 from the centred Gram",
                        "finite-precision centring of a theoretically PSD kernel",
                        "the embedding is a truncated PSD projection, not the full KPCA map",
                    ))
                    .build(),
            );
            ctx.push(
                Issue::builder(IssueCode::KernelNotPd)
                    .message("centred Gram was indefinite at working precision")
                    .build(),
            );
        }
        let want = self.n_components.max(1);
        if kept.len() < want {
            ctx.push(
                Issue::builder(IssueCode::ComponentsExceedRank)
                    .message(format!(
                        "requested {} kernel components but numerical rank is {}",
                        want,
                        kept.len()
                    ))
                    .build(),
            );
        }
        let kkeep = want.min(kept.len());
        let mut evals = Vector::zeros(kkeep);
        let mut alphas = Matrix::zeros(n, kkeep);
        for c in 0..kkeep {
            let (lam, idx) = kept[c];
            evals[c] = lam;
            let scale = 1.0 / lam.sqrt();
            for i in 0..n {
                alphas.set(i, c, vecs[(i, idx)] * scale);
            }
        }
        ctx.finish(FittedKernelPca {
            x_train: x.clone(),
            alphas,
            eigenvalues: evals,
            gamma: self.gamma,
            row_mean,
            grand_mean: grand,
        })
    }
}

impl Transform for FittedKernelPca {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.ncols() != self.x_train.ncols() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("KernelPCA transform feature count ≠ training")
                    .build(),
            );
        }
        let n = x.nrows();
        let m = self.x_train.nrows();
        let k = self.alphas.ncols();
        if m == 0 || k == 0 {
            return ctx.finish(Matrix::zeros(n, k));
        }
        let out = Matrix::from_fn(n, k, |i, c| {
            let mut acc = 0.0;
            let mut kmean = 0.0;
            for j in 0..m {
                let kij = rbf(x, i, &self.x_train, j, self.gamma);
                kmean += kij;
                acc += (kij - self.row_mean[j]) * self.alphas.get(j, c);
            }
            kmean /= m as f64;
            acc - (kmean - self.grand_mean) * self.alphas.column(c).as_slice().iter().sum::<f64>()
        });
        ctx.finish(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kpca_embeds_two_blobs() {
        let x = Matrix::from_fn(16, 2, |i, j| {
            if i < 8 {
                if j == 0 {
                    -2.0
                } else {
                    0.1 * (i as f64)
                }
            } else if j == 0 {
                2.0
            } else {
                0.1 * (i as f64)
            }
        });
        let q = KernelPca {
            n_components: 2,
            gamma: 0.5,
        }
        .fit(&x, &Session::new("kpca", "fit"))
        .expect("kpca");
        assert_eq!(q.value.eigenvalues.len(), 2);
        let z = q
            .value
            .transform(&x, &Session::new("kpca", "t"))
            .unwrap()
            .value;
        assert_eq!(z.shape(), (16, 2));
        assert!(z.to_row_major().iter().all(|v| v.is_finite()));
    }
}
