//! Stochastic numerical analysis: Euler / Itô–Taylor expectations
//! (a Pure-Rust stand-in for YUIMA's Malliavin asymptotic expansion).
//!
//! Full automatic Malliavin calculus is not reproduced; instead we provide
//! the practical objects YUIMA users reach for: Monte Carlo of path
//! functionals, an Euler bias-corrected expectation, and a first-order
//! Itô–Taylor expansion of `E[f(X_T)]` for scalar diffusions.

use amatsuki::Rng;

use crate::error::{Error, Result};
use crate::model::Sde;
use crate::sampling::Sampling;
use crate::simulate::{expectation, simulate_n, SimConfig};

/// Monte Carlo of a terminal functional `f(X_T)`.
pub fn mc_functional<M, R, F>(
    model: &M,
    sampling: &Sampling,
    x0: &[f64],
    nsim: usize,
    rng: &mut R,
    f: F,
) -> Result<f64>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
    F: Fn(&[f64]) -> f64,
{
    expectation(model, sampling, x0, nsim, rng, &SimConfig::default(), f)
}

/// Path functional `φ = F(X_T) + ∫₀ᵀ f(t, X_t) dt` estimated by trapezoid + MC.
pub fn mc_path_functional<M, R, Term, Run>(
    model: &M,
    sampling: &Sampling,
    x0: &[f64],
    nsim: usize,
    rng: &mut R,
    terminal: Term,
    running: Run,
) -> Result<f64>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
    Term: Fn(&[f64]) -> f64,
    Run: Fn(f64, &[f64]) -> f64,
{
    let ens = simulate_n(model, sampling, x0, nsim, rng, &SimConfig::default())?;
    let mut acc = 0.0;
    for p in &ens.paths {
        let mut integ = 0.0;
        for i in 0..p.n_steps() {
            let t0 = p.times()[i];
            let t1 = p.times()[i + 1];
            let f0 = running(t0, p.state(i));
            let f1 = running(t1, p.state(i + 1));
            integ += 0.5 * (f0 + f1) * (t1 - t0);
        }
        acc += terminal(p.terminal()) + integ;
    }
    Ok(acc / nsim as f64)
}

/// First-order Itô–Taylor / Euler expansion of `E[f(X_T)]` for a **scalar**
/// autonomous diffusion `dX = a(x) dt + σ(x) dW`.
///
/// ```text
/// E[f(X_T)] ≈ f(x₀) + T ℒf(x₀) + ½ T² ℒ²f(x₀)
/// ℒ = a ∂ₓ + ½ σ² ∂ₓₓ
/// ```
///
/// Derivatives of `f` are supplied by the caller.
#[derive(Clone, Copy, Debug)]
pub struct ScalarJet {
    pub f: f64,
    pub fx: f64,
    pub fxx: f64,
    pub fxxx: f64,
    pub fxxxx: f64,
}

pub fn ito_taylor_expectation<M: Sde + ?Sized>(
    model: &M,
    x0: f64,
    t: f64,
    jet: ScalarJet,
) -> Result<f64> {
    if model.dim() != 1 || model.n_noise() != 1 {
        return Err(Error::unsupported(
            "Itô–Taylor helper is scalar (n = m = 1)",
        ));
    }
    let x = [x0];
    let mut a = [0.0];
    let mut s = [0.0];
    model.drift(0.0, &x, &mut a);
    model.diffusion(0.0, &x, &mut s);
    let eps = 1e-5;
    let mut ap = [0.0];
    let mut am = [0.0];
    let mut sp = [0.0];
    let mut sm = [0.0];
    model.drift(0.0, &[x0 + eps], &mut ap);
    model.drift(0.0, &[x0 - eps], &mut am);
    model.diffusion(0.0, &[x0 + eps], &mut sp);
    model.diffusion(0.0, &[x0 - eps], &mut sm);
    let ax = (ap[0] - am[0]) / (2.0 * eps);
    let sx = (sp[0] - sm[0]) / (2.0 * eps);
    let axx = (ap[0] - 2.0 * a[0] + am[0]) / (eps * eps);
    let sxx = (sp[0] - 2.0 * s[0] + sm[0]) / (eps * eps);

    let gen = |f1: f64, f2: f64| a[0] * f1 + 0.5 * s[0] * s[0] * f2;
    let lf = gen(jet.fx, jet.fxx);
    // ℒ² f ≈ a ∂(ℒf) + ½ σ² ∂²(ℒf) evaluated with product rules at x0.
    // ∂(ℒf) = aₓ fₓ + a fₓₓ + σ σₓ fₓₓ + ½ σ² fₓₓₓ
    let d_lf = ax * jet.fx + a[0] * jet.fxx + s[0] * sx * jet.fxx + 0.5 * s[0] * s[0] * jet.fxxx;
    let d2_lf = axx * jet.fx
        + 2.0 * ax * jet.fxx
        + a[0] * jet.fxxx
        + (sx * sx + s[0] * sxx) * jet.fxx
        + 2.0 * s[0] * sx * jet.fxxx
        + 0.5 * s[0] * s[0] * jet.fxxxx;
    let l2f = a[0] * d_lf + 0.5 * s[0] * s[0] * d2_lf;
    Ok(jet.f + t * lf + 0.5 * t * t * l2f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::OrnsteinUhlenbeck;

    #[test]
    fn ito_taylor_ou_mean() {
        // For OU, E[X_T] = θ + (x0−θ)e^{-κT}; f(x)=x, all higher deriv 0
        // so the expansion is only first order in the generator and will be
        // the Euler mean x0 + T κ(θ−x0).
        let m = OrnsteinUhlenbeck::new(1.0, 0.0, 0.3).unwrap();
        let jet = ScalarJet {
            f: 1.0,
            fx: 1.0,
            fxx: 0.0,
            fxxx: 0.0,
            fxxxx: 0.0,
        };
        let e = ito_taylor_expectation(&m, 1.0, 0.1, jet).unwrap();
        // ℒx = κ(θ−x) = −1, ℒ²x = κ²(x−θ) = 1, so
        // f + T ℒf + ½ T² ℒ²f = 1 − 0.1 + 0.005
        assert!((e - 0.905).abs() < 1e-8);
    }
}
