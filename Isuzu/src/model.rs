//! SDE model specification (`setModel` in YUIMA).
//!
//! A model is a (possibly time-inhomogeneous) Itô process
//!
//! ```text
//! dX_t = a(t, X_t) dt + σ(t, X_t) dW_t + γ(t, X_{t-}) dJ_t
//! ```
//!
//! plus optional fractional-Brownian and Lévy drivers.

use crate::error::{Error, Result};
use crate::noise::LevyMeasure;

/// Concrete SDE used for simulation (parameters baked in).
pub trait Sde: Send + Sync {
    /// State dimension `n`.
    fn dim(&self) -> usize;
    /// Driving Wiener dimension `m`.
    fn n_noise(&self) -> usize {
        1
    }

    fn drift(&self, t: f64, x: &[f64], out: &mut [f64]);
    /// Diffusion matrix `σ(t,x)` stored row-major, shape `n × m`.
    fn diffusion(&self, t: f64, x: &[f64], out: &mut [f64]);

    /// Optional state-Jacobian of the diffusion for the Milstein scheme.
    ///
    /// `out[i * n * m + p * m + j] = ∂σ_{ij} / ∂x_p`.
    /// Return `false` if unavailable (Euler will still work; Milstein falls
    /// back to a finite-difference approximation).
    fn diffusion_jacobian(&self, _t: f64, _x: &[f64], _out: &mut [f64]) -> bool {
        false
    }

    /// Jump coefficient `γ(t,x)` multiplying a scalar Lévy increment (additive
    /// in each coordinate). `None` means a continuous diffusion.
    fn jump_coeff(&self, _t: f64, _x: &[f64], _out: &mut [f64]) -> bool {
        false
    }

    fn levy(&self) -> Option<&LevyMeasure> {
        None
    }

    /// Hurst index of an fBM driver. `None` means standard Wiener (`H = 1/2`).
    fn hurst(&self) -> Option<f64> {
        None
    }

    /// If `true`, jumps scale the current state (Merton: `ΔX = X * (e^Z − 1)`
    /// is expressed by putting `γ = X` and adding `X * increment`).
    fn multiplicative_jumps(&self) -> bool {
        false
    }

    /// Closed-form one-step transition when the model has one.
    ///
    /// `dw` is the Wiener increment (`N(0, Δt)` per coordinate), not a
    /// unit Gaussian. Write the new state into `out` and return `true`.
    /// The default returns `false` (`Scheme::Exact` then errors).
    fn exact_step(&self, _t: f64, _x: &[f64], _dt: f64, _dw: &[f64], _out: &mut [f64]) -> bool {
        false
    }

    fn validate(&self) -> Result<()> {
        if self.dim() == 0 {
            return Err(Error::dim("state dimension must be positive"));
        }
        if self.n_noise() == 0 {
            return Err(Error::dim("noise dimension must be positive"));
        }
        if let Some(h) = self.hurst() {
            if !(0.0 < h && h < 1.0) {
                return Err(Error::param("Hurst must lie in (0, 1)"));
            }
            if self.n_noise() != 1 {
                return Err(Error::unsupported("fBM driver is implemented for m = 1"));
            }
        }
        Ok(())
    }
}

/// Parametric family used for inference (`θ ↦` an [`Sde`]).
pub trait ParametricSde: Clone + Send + Sync {
    /// Concrete model type produced by [`ParametricSde::freeze`].
    type Frozen: Sde;

    fn param_names(&self) -> &[&'static str];
    fn params(&self) -> &[f64];
    fn set_params(&mut self, p: &[f64]) -> Result<()>;

    fn n_params(&self) -> usize {
        self.param_names().len()
    }

    fn freeze(&self) -> Result<Self::Frozen>;

    fn with_params(&self, p: &[f64]) -> Result<Self> {
        let mut c = self.clone();
        c.set_params(p)?;
        Ok(c)
    }

    fn check_params(p: &[f64], names: &[&str]) -> Result<()> {
        if p.len() != names.len() {
            return Err(Error::param(format!(
                "expected {} parameters {:?}, got {}",
                names.len(),
                names,
                p.len()
            )));
        }
        if p.iter().any(|x| !x.is_finite()) {
            return Err(Error::param("parameters must be finite"));
        }
        Ok(())
    }
}

/// Closure-based SDE (the Rust analogue of YUIMA's symbolic `setModel`).
pub struct FnSde<A, S> {
    dim: usize,
    n_noise: usize,
    drift_fn: A,
    diffusion_fn: S,
    levy: Option<LevyMeasure>,
    hurst: Option<f64>,
    jump: Option<Box<dyn Fn(f64, &[f64], &mut [f64]) + Send + Sync>>,
    multiplicative_jumps: bool,
}

impl<A, S> FnSde<A, S>
where
    A: Fn(f64, &[f64], &mut [f64]) + Send + Sync,
    S: Fn(f64, &[f64], &mut [f64]) + Send + Sync,
{
    pub fn new(dim: usize, n_noise: usize, drift: A, diffusion: S) -> Self {
        Self {
            dim,
            n_noise,
            drift_fn: drift,
            diffusion_fn: diffusion,
            levy: None,
            hurst: None,
            jump: None,
            multiplicative_jumps: false,
        }
    }

    pub fn with_levy(mut self, levy: LevyMeasure) -> Self {
        self.levy = Some(levy);
        self
    }

    pub fn with_hurst(mut self, hurst: f64) -> Self {
        self.hurst = Some(hurst);
        self
    }

    pub fn with_jump<J>(mut self, jump: J) -> Self
    where
        J: Fn(f64, &[f64], &mut [f64]) + Send + Sync + 'static,
    {
        self.jump = Some(Box::new(jump));
        self
    }

    pub fn multiplicative(mut self, yes: bool) -> Self {
        self.multiplicative_jumps = yes;
        self
    }
}

impl<A, S> Sde for FnSde<A, S>
where
    A: Fn(f64, &[f64], &mut [f64]) + Send + Sync,
    S: Fn(f64, &[f64], &mut [f64]) + Send + Sync,
{
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_noise(&self) -> usize {
        self.n_noise
    }
    fn drift(&self, t: f64, x: &[f64], out: &mut [f64]) {
        (self.drift_fn)(t, x, out);
    }
    fn diffusion(&self, t: f64, x: &[f64], out: &mut [f64]) {
        (self.diffusion_fn)(t, x, out);
    }
    fn jump_coeff(&self, t: f64, x: &[f64], out: &mut [f64]) -> bool {
        if let Some(j) = &self.jump {
            j(t, x, out);
            true
        } else {
            false
        }
    }
    fn levy(&self) -> Option<&LevyMeasure> {
        self.levy.as_ref()
    }
    fn hurst(&self) -> Option<f64> {
        self.hurst
    }
    fn multiplicative_jumps(&self) -> bool {
        self.multiplicative_jumps
    }
}

/// Linear state-space diffusion
/// `dX = (A X + b) dt + σ dW`, optionally with an observation map `Y = H X`.
#[derive(Clone, Debug)]
pub struct LinearStateSpace {
    pub a: faer::Mat<f64>,
    pub b: faer::Col<f64>,
    pub sigma: faer::Mat<f64>,
    pub h: Option<faer::Mat<f64>>,
}

impl LinearStateSpace {
    pub fn new(a: faer::Mat<f64>, b: faer::Col<f64>, sigma: faer::Mat<f64>) -> Result<Self> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(Error::dim("A must be square"));
        }
        if b.nrows() != n || sigma.nrows() != n {
            return Err(Error::dim("A, b, σ dimension mismatch"));
        }
        Ok(Self {
            a,
            b,
            sigma,
            h: None,
        })
    }

    pub fn with_observation(mut self, h: faer::Mat<f64>) -> Result<Self> {
        if h.ncols() != self.a.nrows() {
            return Err(Error::dim("H must have n columns"));
        }
        self.h = Some(h);
        Ok(self)
    }
}

impl Sde for LinearStateSpace {
    fn dim(&self) -> usize {
        self.a.nrows()
    }
    fn n_noise(&self) -> usize {
        self.sigma.ncols()
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let xv = crate::linalg::col_from_slice(x);
        let d = &self.a * &xv + &self.b;
        for i in 0..out.len() {
            out[i] = d[i];
        }
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        let n = self.sigma.nrows();
        let m = self.sigma.ncols();
        for i in 0..n {
            for j in 0..m {
                out[i * m + j] = self.sigma[(i, j)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fn_sde_gbm_shape() {
        let m = FnSde::new(
            1,
            1,
            |_t, x, out| out[0] = 0.1 * x[0],
            |_t, x, out| out[0] = 0.2 * x[0],
        );
        let mut a = [0.0];
        let mut s = [0.0];
        m.drift(0.0, &[2.0], &mut a);
        m.diffusion(0.0, &[2.0], &mut s);
        assert!((a[0] - 0.2).abs() < 1e-14);
        assert!((s[0] - 0.4).abs() < 1e-14);
    }
}
