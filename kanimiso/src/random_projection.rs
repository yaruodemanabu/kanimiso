//! Johnson–Lindenstrauss random projections (sklearn `random_projection`).
//!
//! The embedding is a documented numerical compromise: pairwise distances are
//! preserved only up to `(ε, δ)` that the chosen `n_components` can support.
//! A projection dimension that is too small for the JL bound is a warning, not
//! a silent success.

use crate::context::FitCtx;
use crate::data::Matrix;
use crate::rng::Rng;
use crate::traits::{FitUnsupervised, Transform};
use crate::validate::inspect_xy;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, NumericalCompromise, Qualified, Result, Severity};

fn jl_min_components(n: usize, eps: f64) -> usize {
    let e = eps.clamp(0.05, 0.9);
    let ln_n = (n.max(2) as f64).ln();
    ((4.0 * ln_n) / (e * e / 2.0 - e * e * e / 3.0))
        .ceil()
        .max(1.0) as usize
}

fn warn_jl(ctx: &mut FitCtx, n: usize, k: usize, eps: f64, what: &str) {
    let need = jl_min_components(n, eps);
    if k < need {
        ctx.push(
            Issue::builder(IssueCode::UnderdeterminedSystem)
                .severity(Severity::Warning)
                .message(format!(
                    "{what}: n_components={k} < JL minimum {need} for n={n}, ε={eps:.2}"
                ))
                .compromise(NumericalCompromise::new(
                    format!("Johnson–Lindenstrauss embedding with ε={eps}"),
                    format!("a {k}-dimensional random projection"),
                    "the target dimension is below the classical JL sample-size bound",
                    "pairwise distances may distort by more than ε; do not treat the embedding as isometric",
                ))
                .metric("jl_min", need as f64)
                .metric("n_components", k as f64)
                .build(),
        );
    }
}

/// Dense Gaussian random projection.
#[derive(Clone, Debug)]
pub(crate) struct GaussianRandomProjection {
    /// Target dimension.
    pub n_components: usize,
    /// JL distortion `ε` used only for the warning bound.
    pub eps: f64,
    /// PRNG seed.
    pub seed: u64,
    components: Matrix,
    fitted: bool,
}

impl Default for GaussianRandomProjection {
    fn default() -> Self {
        Self {
            n_components: 8,
            eps: 0.5,
            seed: 1,
            components: Matrix::zeros(0, 0),
            fitted: false,
        }
    }
}

impl GaussianRandomProjection {
    /// Project to `n_components`.
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components: n_components.max(1),
            ..Self::default()
        }
    }
}

impl FitUnsupervised for GaussianRandomProjection {
    type Fitted = Self;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut this = self.clone();
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let k = this.n_components.max(1);
        // Do not inspect_identification(n, k): k is a sketch size, not a model order.
        warn_jl(&mut ctx, x.nrows(), k, this.eps, "GaussianRandomProjection");
        let p = x.ncols();
        let mut rng = Rng::new(this.seed | 1);
        let scale = 1.0 / (k as f64).sqrt();
        this.components = Matrix::from_fn(p, k, |_, _| rng.standard_normal() * scale);
        this.fitted = true;
        ctx.finish(this.clone())
    }
}

impl Transform for GaussianRandomProjection {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(Matrix::zeros(x.nrows(), self.n_components.max(1)));
        }
        if x.ncols() != self.components.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("GaussianRandomProjection p ≠ fitted p")
                    .build(),
            );
        }
        let k = self.components.ncols();
        let z = Matrix::from_fn(x.nrows(), k, |i, c| {
            let mut s = 0.0;
            for j in 0..x.ncols().min(self.components.nrows()) {
                s += x.get(i, j) * self.components.get(j, c);
            }
            s
        });
        ctx.finish(z)
    }
}

/// Achlioptas sparse random projection.
#[derive(Clone, Debug)]
pub(crate) struct SparseRandomProjection {
    /// Target dimension.
    pub n_components: usize,
    /// Density `1/s` (`s = √p` when `None`).
    pub density: Option<f64>,
    /// JL distortion `ε`.
    pub eps: f64,
    /// PRNG seed.
    pub seed: u64,
    components: Matrix,
    fitted: bool,
}

impl Default for SparseRandomProjection {
    fn default() -> Self {
        Self {
            n_components: 8,
            density: None,
            eps: 0.5,
            seed: 2,
            components: Matrix::zeros(0, 0),
            fitted: false,
        }
    }
}

impl SparseRandomProjection {
    /// Project to `n_components`.
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components: n_components.max(1),
            ..Self::default()
        }
    }
}

impl FitUnsupervised for SparseRandomProjection {
    type Fitted = Self;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut this = self.clone();
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let k = this.n_components.max(1);
        warn_jl(&mut ctx, x.nrows(), k, this.eps, "SparseRandomProjection");
        let p = x.ncols().max(1);
        let s = if let Some(d) = this.density {
            if !d.is_finite() || d <= 0.0 || d > 1.0 {
                ctx.push(
                    Issue::builder(IssueCode::InvalidWeight)
                        .severity(Severity::Warning)
                        .message(format!("SparseRandomProjection density={d} not in (0, 1]"))
                        .build(),
                );
                (p as f64).sqrt().max(1.0)
            } else {
                (1.0 / d).max(1.0)
            }
        } else {
            (p as f64).sqrt().max(1.0)
        };
        let mut rng = Rng::new(this.seed | 3);
        let scale = (s / k as f64).sqrt();
        let inv_s = 1.0 / s;
        this.components = Matrix::from_fn(x.ncols(), k, |_, _| {
            let u = rng.uniform();
            if u < inv_s / 2.0 {
                scale
            } else if u < inv_s {
                -scale
            } else {
                0.0
            }
        });
        this.fitted = true;
        ctx.finish(this.clone())
    }
}

impl Transform for SparseRandomProjection {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(Matrix::zeros(x.nrows(), self.n_components.max(1)));
        }
        let k = self.components.ncols();
        let z = Matrix::from_fn(x.nrows(), k, |i, c| {
            let mut s = 0.0;
            for j in 0..x.ncols().min(self.components.nrows()) {
                s += x.get(i, j) * self.components.get(j, c);
            }
            s
        });
        ctx.finish(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{FitUnsupervised, Transform};

    #[test]
    fn gaussian_and_sparse_project() {
        let x = Matrix::from_fn(20, 6, |i, j| (i + j) as f64);
        let mut g = GaussianRandomProjection::new(3);
        g.fit_unsupervised(&x, &Session::new("grp", "fit"))
            .expect("gfit");
        let z = g
            .transform(&x, &Session::new("grp", "t"))
            .expect("gt")
            .value;
        assert_eq!(z.shape(), (20, 3));
        assert!(z.get(0, 0).is_finite());
        let mut s = SparseRandomProjection::new(4);
        s.fit_unsupervised(&x, &Session::new("srp", "fit"))
            .expect("sfit");
        let w = s
            .transform(&x, &Session::new("srp", "t"))
            .expect("st")
            .value;
        assert_eq!(w.ncols(), 4);
    }
}
