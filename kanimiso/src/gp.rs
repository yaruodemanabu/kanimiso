//! Gaussian-process regression and Laplace binary classification (sklearn `gaussian_process`).
//!
//! The RBF Gram is factored through the shared internal Cholesky solver. A non-PD
//! kernel receives diagonal jitter and a [`NumericalCompromise`]. \(n\) much
//! larger than a few hundred is overparameterized for a dense GP.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::chol_solve;
use crate::special::norm_cdf;
use crate::traits::{Fit, Predict};
use crate::validate::{inspect_classes, inspect_xy};
use faer::Mat;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result};

/// Squared-exponential GP regressor.
#[derive(Clone, Debug)]
pub struct GaussianProcessRegressor {
    /// Length-scale \(\ell\).
    pub length_scale: f64,
    /// Signal variance \(\sigma_f^2\).
    pub signal_variance: f64,
    /// Observation noise \(\sigma_n^2\).
    pub noise: f64,
}

impl Default for GaussianProcessRegressor {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise: 1e-6,
        }
    }
}

impl GaussianProcessRegressor {
    /// Default RBF GP.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted GP regressor (representer coefficients).
#[derive(Clone, Debug)]
pub struct FittedGpr {
    x_train: Matrix,
    alpha: Vector,
    /// Kernel hyperparameters used at fit.
    pub length_scale: f64,
    /// Signal variance.
    pub signal_variance: f64,
    /// Noise variance (including any jitter that was added).
    pub noise: f64,
}

/// Binary Laplace GP classifier.
#[derive(Clone, Debug)]
pub struct GaussianProcessClassifier {
    /// Length-scale \(\ell\).
    pub length_scale: f64,
    /// Signal variance.
    pub signal_variance: f64,
    /// Newton iterations.
    pub max_iter: usize,
}

impl Default for GaussianProcessClassifier {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
            max_iter: 20,
        }
    }
}

impl GaussianProcessClassifier {
    /// Default Laplace GPC.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Laplace GPC.
#[derive(Clone, Debug)]
pub struct FittedGpc {
    x_train: Matrix,
    latent: Vector,
    /// Sorted classes.
    pub classes: Vec<i64>,
    /// Length-scale.
    pub length_scale: f64,
    /// Signal variance.
    pub signal_variance: f64,
}

fn rbf_gram(a: &Matrix, b: &Matrix, ell: f64, sf2: f64, noise: f64, square: bool) -> Mat<f64> {
    let ell2 = (2.0 * ell * ell).max(1e-18);
    let mut k = Mat::<f64>::zeros(a.nrows(), b.nrows());
    for i in 0..a.nrows() {
        for j in 0..b.nrows() {
            let mut d2 = 0.0;
            for c in 0..a.ncols().min(b.ncols()) {
                let d = a.get(i, c) - b.get(j, c);
                d2 += d * d;
            }
            let mut v = sf2 * (-d2 / ell2).exp();
            if square && i == j {
                v += noise;
            }
            k[(i, j)] = v;
        }
    }
    k
}

fn matvec_k(k: &Mat<f64>, v: &Vector) -> Vector {
    Vector::from_iter((0..k.nrows()).map(|i| {
        let mut s = 0.0;
        for j in 0..k.ncols().min(v.len()) {
            s += k[(i, j)] * v[j];
        }
        s
    }))
}

fn jittered_solve(
    ctx: &mut FitCtx,
    k: &Mat<f64>,
    rhs: &Vector,
    noise: f64,
) -> Option<(Vector, f64)> {
    let mut used = noise;
    if let Some(sol) = chol_solve(&mut ctx.report, k, rhs, &ctx.policy) {
        return Some((sol, used));
    }
    let mut kj = k.clone();
    used = noise + 1e-8 * (1.0 + noise);
    for i in 0..kj.nrows() {
        kj[(i, i)] += 1e-8 * (1.0 + noise);
    }
    ctx.push(
        Issue::builder(IssueCode::JitterInjected)
            .message("RBF Gram was not SPD; diagonal jitter 1e-8(1+σn²) was added")
            .compromise(NumericalCompromise::new(
                "exact GP posterior (K + σn²I)⁻¹ y",
                "Cholesky on a jittered Gram",
                "the kernel matrix is indefinite at working precision",
                "predictive variances are slightly inflated; do not treat them as exact",
            ))
            .build(),
    );
    chol_solve(&mut ctx.report, &kj, rhs, &ctx.policy).map(|sol| (sol, used))
}

impl Fit for GaussianProcessRegressor {
    type Fitted = FittedGpr;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedGpr>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if ctx.report.contains(IssueCode::ConstantTarget)
            || ctx.report.contains(IssueCode::EmptyMatrix)
        {
            return ctx.finish(FittedGpr {
                x_train: x.clone(),
                alpha: Vector::zeros(x.nrows()),
                length_scale: self.length_scale,
                signal_variance: self.signal_variance,
                noise: self.noise,
            });
        }
        if x.nrows() > 400 {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!(
                        "dense GP on n={} is O(n³); this is a numerical, not a statistical, model of the sample",
                        x.nrows()
                    ))
                    .build(),
            );
        }
        if self.length_scale <= 0.0 || self.signal_variance <= 0.0 || self.noise < 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message("GP hyperparameters must satisfy ℓ>0, σf²>0, σn²≥0")
                    .build(),
            );
        }
        let k = rbf_gram(
            x,
            x,
            self.length_scale.max(1e-8),
            self.signal_variance.max(1e-12),
            self.noise.max(0.0),
            true,
        );
        let (alpha, noise) = match jittered_solve(&mut ctx, &k, y, self.noise.max(0.0)) {
            Some(v) => v,
            None => {
                ctx.push(
                    Issue::builder(IssueCode::CholeskyFailed)
                        .message("GP kernel solve failed even after jitter")
                        .build(),
                );
                (Vector::zeros(x.nrows()), self.noise)
            }
        };
        ctx.finish(FittedGpr {
            x_train: x.clone(),
            alpha,
            length_scale: self.length_scale,
            signal_variance: self.signal_variance,
            noise,
        })
    }
}

impl Predict for FittedGpr {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.ncols() != self.x_train.ncols() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("GP predict feature count ≠ training")
                    .build(),
            );
        }
        let kstar = rbf_gram(
            x,
            &self.x_train,
            self.length_scale,
            self.signal_variance,
            0.0,
            false,
        );
        ctx.finish(matvec_k(&kstar, &self.alpha))
    }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

impl Fit for GaussianProcessClassifier {
    type Fitted = FittedGpc;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedGpc>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        if classes.len() != 2 {
            if classes.len() > 2 {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("Laplace GPC is binary; K>2 is not a joint GP")
                        .meaninglessness(Meaninglessness::new(
                            "Gaussian process classifier",
                            "the Laplace posterior is derived for a single latent function",
                            signlred::InterpretiveValue::Misleading,
                            "use MultinomialLogistic or one GP per class",
                        ))
                        .build(),
                );
            }
            return ctx.finish(FittedGpc {
                x_train: x.clone(),
                latent: Vector::zeros(x.nrows()),
                classes,
                length_scale: self.length_scale,
                signal_variance: self.signal_variance,
            });
        }
        let pos = classes[1];
        let ypm = Vector::from_iter(y.as_slice().iter().map(|&v| {
            if v.round() as i64 == pos {
                1.0
            } else {
                0.0
            }
        }));
        let n = x.nrows();
        let k = rbf_gram(
            x,
            x,
            self.length_scale.max(1e-8),
            self.signal_variance.max(1e-12),
            1e-6,
            true,
        );
        let mut f = Vector::zeros(n);
        let mut converged = false;
        for it in 0..self.max_iter.max(1) {
            let mut w = Vector::zeros(n);
            let mut t = Vector::zeros(n);
            for i in 0..n {
                let p = sigmoid(f[i]).clamp(1e-8, 1.0 - 1e-8);
                w[i] = p * (1.0 - p);
                t[i] = ypm[i] - p + w[i] * f[i];
            }
            // (I + K W) f = K t
            let mut a = Matrix::zeros(n, n);
            let kt = matvec_k(&k, &t);
            for i in 0..n {
                for j in 0..n {
                    let mut v = k[(i, j)] * w[j];
                    if i == j {
                        v += 1.0;
                    }
                    a.set(i, j, v);
                }
            }
            let mut scratch = signlred::Report::new("gpc", "newton");
            let Some(fnxt) = crate::linalg::least_squares(&mut scratch, &a, &kt, &ctx.policy)
            else {
                break;
            };
            let delta = fnxt.sub(&f).norm();
            f = fnxt;
            ctx.session.step(it as u64, delta, None);
            if delta < 1e-7 {
                ctx.session.converged("Laplace GP Newton", it as u64);
                converged = true;
                break;
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(signlred::Severity::Warning)
                    .message("Laplace GPC Newton did not meet the tolerance")
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(signlred::Severity::Advisory)
                .message("predictive probabilities use Φ(f_*) without the full Laplace variance correction")
                .compromise(NumericalCompromise::new(
                    "Rasmussen–Williams Laplace GP posterior",
                    "Newton latent mode; predict uses Φ(k_*ᵀ K⁻¹ f)",
                    "the probit variance correction is omitted",
                    "probabilities are a mode-plug-in, not the integrated posterior",
                ))
                .build(),
        );
        ctx.finish(FittedGpc {
            x_train: x.clone(),
            latent: f,
            classes,
            length_scale: self.length_scale,
            signal_variance: self.signal_variance,
        })
    }
}

impl Predict for FittedGpc {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if self.classes.len() < 2 {
            return ctx.finish(Vector::filled(
                x.nrows(),
                self.classes.first().copied().unwrap_or(0) as f64,
            ));
        }
        let k = rbf_gram(
            &self.x_train,
            &self.x_train,
            self.length_scale,
            self.signal_variance,
            1e-6,
            true,
        );
        let mut scratch = signlred::Report::new("gpc", "alpha");
        let alpha = match chol_solve(&mut scratch, &k, &self.latent, &ctx.policy) {
            Some(a) => a,
            None => self.latent.clone(),
        };
        let kstar = rbf_gram(
            x,
            &self.x_train,
            self.length_scale,
            self.signal_variance,
            0.0,
            false,
        );
        let fstar = matvec_k(&kstar, &alpha);
        let pos = *self.classes.last().unwrap_or(&1) as f64;
        let neg = *self.classes.first().unwrap_or(&0) as f64;
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            if norm_cdf(fstar[i]) >= 0.5 {
                pos
            } else {
                neg
            }
        }));
        ctx.finish(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpr_interpolates_a_smooth_line() {
        let x = Matrix::from_fn(12, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..12).map(|i| (i as f64).sin()));
        let q = GaussianProcessRegressor {
            length_scale: 2.0,
            noise: 1e-4,
            ..GaussianProcessRegressor::default()
        }
        .fit(&x, &y, &Session::new("gp", "fit"))
        .expect("gpr");
        let pred = q.value.predict(&x, &Session::new("gp", "p")).unwrap().value;
        let mut sse = 0.0;
        for i in 0..y.len() {
            let e = pred[i] - y[i];
            sse += e * e;
        }
        assert!(
            sse / (y.len() as f64) < 0.05,
            "mse={}",
            sse / y.len() as f64
        );
    }

    #[test]
    fn gpc_separates_two_blobs() {
        let x = Matrix::from_fn(16, 1, |i, _| if i < 8 { -1.5 } else { 1.5 });
        let y = Vector::from_iter((0..16).map(|i| if i < 8 { 0.0 } else { 1.0 }));
        let q = GaussianProcessClassifier::new()
            .fit(&x, &y, &Session::new("gpc", "fit"))
            .expect("gpc");
        let pred = q
            .value
            .predict(&x, &Session::new("gpc", "p"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..16 {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 14, "ok={ok}");
    }
}
