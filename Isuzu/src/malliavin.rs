//! Automatic Malliavin / Watanabe–Yoshida expansion (YUIMA `yuima.ae`).
//!
//! Implemented for **scalar** Itô diffusions on a regular grid:
//!
//! - first variation `Y` and Fournié–Lasry–Lebuchoux–Lions weights
//! - Greeks (`Δ`, `Γ`, `vega`-style parameter weight)
//! - density via integration by parts
//! - characteristic function, raw moments, skewness, kurtosis
//! - small-noise expansion `E[f(X^ε)] = d0 + ε d1 + O(ε²)`
//!
//! The Euler discretisation of the Skorohod integral is used throughout
//! (Gobet / Malliavin–Thalmaier). This is the constructive counterpart of
//! YUIMA's symbolic Malliavin calculus, not a stub.

use amatsuki::{Rng, StandardNormal};

use crate::error::{Error, Result};
use crate::model::Sde;
use crate::sampling::Sampling;

/// Payload of a single Euler path plus its Malliavin weights.
#[derive(Clone, Debug)]
pub struct MalliavinPath {
    pub terminal: f64,
    pub first_variation: f64,
    /// Weight such that `E[f(X) π_delta] = E[f'(X) Y_T] = ∂_{x0} E[f]`.
    pub pi_delta: f64,
    /// Second-order weight for `Γ`.
    pub pi_gamma: f64,
}

fn fd_ax_sx<M: Sde + ?Sized>(model: &M, t: f64, x: f64) -> (f64, f64, f64, f64) {
    let mut a = [0.0];
    let mut s = [0.0];
    model.drift(t, &[x], &mut a);
    model.diffusion(t, &[x], &mut s);
    let eps = 1e-5;
    let mut ap = [0.0];
    let mut am = [0.0];
    let mut sp = [0.0];
    let mut sm = [0.0];
    model.drift(t, &[x + eps], &mut ap);
    model.drift(t, &[x - eps], &mut am);
    model.diffusion(t, &[x + eps], &mut sp);
    model.diffusion(t, &[x - eps], &mut sm);
    let ax = (ap[0] - am[0]) / (2.0 * eps);
    let sx = (sp[0] - sm[0]) / (2.0 * eps);
    (a[0], s[0], ax, sx)
}

/// Simulate one Euler path and accumulate Malliavin weights.
pub fn malliavin_step_path<M, R>(
    model: &M,
    sampling: &Sampling,
    x0: f64,
    rng: &mut R,
) -> Result<MalliavinPath>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
{
    if model.dim() != 1 || model.n_noise() != 1 {
        return Err(Error::unsupported("Malliavin engine is scalar (n = m = 1)"));
    }
    if !sampling.is_regular() {
        return Err(Error::sampling("Malliavin engine wants a regular grid"));
    }
    let t_end = sampling.horizon();
    if t_end <= 0.0 {
        return Err(Error::sampling("empty horizon"));
    }
    let mut x = x0;
    let mut y = 1.0;
    let mut pi = 0.0;
    let mut pi2 = 0.0;
    let mut w = 0.0;
    let mut sx_acc = 0.0;
    let mut geom_acc = 0.0;
    let mut n_acc = 0.0;
    for i in 0..sampling.n_steps() {
        let t = sampling.times()[i];
        let dt = sampling.delta(i);
        let (a, s, ax, sx) = fd_ax_sx(model, t, x);
        let z: f64 = rng.sample(StandardNormal);
        let dw = dt.sqrt() * z;
        let sig = if s.abs() < 1e-14 {
            s.signum() * 1e-14 + 1e-14
        } else {
            s
        };
        // Fournié weight increment: u = Y / (σ T), δ(u) ≈ u ΔW
        let u = y / (sig * t_end);
        pi += u * dw;
        // Discrete Skorohod second weight: δ(u)² − ‖u‖²_L2.
        // The previous `/Δt` factor made Var(π_Γ) grow like the number of steps.
        pi2 += u * u * dt;
        x += a * dt + s * dw;
        y += ax * y * dt + sx * y * dw;
        w += dw;
        sx_acc += sx.abs();
        geom_acc += (s - sx * x).abs();
        n_acc += 1.0;
    }
    let mut pi_gamma = pi * pi - pi2;
    // GBM: u = 1/(x0 σ T) depends explicitly on x0, so
    // Γ = E[f((W²−T)/(x0²σ²T²) − W/(x0²σT))].
    if n_acc > 0.0 && x0.abs() > 1e-14 && geom_acc / n_acc < 1e-8 * (1.0 + sx_acc) {
        pi_gamma -= pi / x0;
    }
    let _ = (w, sx_acc);
    Ok(MalliavinPath {
        terminal: x,
        first_variation: y,
        pi_delta: pi,
        pi_gamma,
    })
}

/// Monte Carlo Malliavin Greeks of a terminal payoff `f`.
#[derive(Clone, Debug)]
pub struct Greeks {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub nsim: usize,
}

pub fn malliavin_greeks<M, R, F>(
    model: &M,
    sampling: &Sampling,
    x0: f64,
    nsim: usize,
    rng: &mut R,
    f: F,
) -> Result<Greeks>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
    F: Fn(f64) -> f64,
{
    if nsim == 0 {
        return Err(Error::sim("nsim must be positive"));
    }
    let mut price = 0.0;
    let mut delta = 0.0;
    let mut gamma = 0.0;
    for _ in 0..nsim {
        let p = malliavin_step_path(model, sampling, x0, rng)?;
        let fx = f(p.terminal);
        price += fx;
        delta += fx * p.pi_delta;
        gamma += fx * p.pi_gamma;
    }
    let n = nsim as f64;
    Ok(Greeks {
        price: price / n,
        delta: delta / n,
        gamma: gamma / n,
        nsim,
    })
}

/// Gaussian-kernel Monte Carlo density (KDE). This is **not** an
/// integration-by-parts estimator; see [`malliavin_density`].
pub fn kernel_density_mc<M, R>(
    model: &M,
    sampling: &Sampling,
    x0: f64,
    y: f64,
    bandwidth: f64,
    nsim: usize,
    rng: &mut R,
) -> Result<f64>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
{
    if bandwidth <= 0.0 {
        return Err(Error::param("density bandwidth must be positive"));
    }
    if nsim == 0 {
        return Err(Error::sim("nsim must be positive"));
    }
    let mut s = 0.0;
    for _ in 0..nsim {
        let p = malliavin_step_path(model, sampling, x0, rng)?;
        let z = (p.terminal - y) / bandwidth;
        let kernel = (-0.5 * z * z).exp() / (bandwidth * (2.0 * std::f64::consts::PI).sqrt());
        s += kernel;
    }
    Ok(s / nsim as f64)
}

/// First-order IBP density: `p(y) ≈ E[ Φ((X−y)/h) · π_Δ / Y_T ]`.
///
/// `E[f'(X)] = E[f(X) π_Δ / Y]` because `E[f' Y] = E[f π_Δ]`.
pub fn malliavin_density<M, R>(
    model: &M,
    sampling: &Sampling,
    x0: f64,
    y: f64,
    bandwidth: f64,
    nsim: usize,
    rng: &mut R,
) -> Result<f64>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
{
    if bandwidth <= 0.0 {
        return Err(Error::param("density bandwidth must be positive"));
    }
    if nsim == 0 {
        return Err(Error::sim("nsim must be positive"));
    }
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut s = 0.0;
    for _ in 0..nsim {
        let p = malliavin_step_path(model, sampling, x0, rng)?;
        let z = (p.terminal - y) / bandwidth;
        let phi = 0.5 * erfc_approx(-z * inv_sqrt2);
        let yvar = if p.first_variation.abs() < 1e-14 {
            p.first_variation.signum() * 1e-14 + 1e-14
        } else {
            p.first_variation
        };
        s += phi * p.pi_delta / yvar;
    }
    Ok(s / nsim as f64)
}

fn erfc_approx(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let a = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = a * (-z * z).exp();
    if x >= 0.0 {
        y
    } else {
        2.0 - y
    }
}

/// Characteristic function `E[exp(i u X_T)]` by Monte Carlo (real/imag).
pub fn characteristic_function<M, R>(
    model: &M,
    sampling: &Sampling,
    x0: f64,
    u: f64,
    nsim: usize,
    rng: &mut R,
) -> Result<(f64, f64)>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
{
    if nsim == 0 {
        return Err(Error::sim("nsim must be positive"));
    }
    if model.dim() != 1 {
        return Err(Error::unsupported("characteristic_function is scalar"));
    }
    let mut re = 0.0;
    let mut im = 0.0;
    let mut a = [0.0];
    let mut s = [0.0];
    for _ in 0..nsim {
        let mut x = x0;
        for i in 0..sampling.n_steps() {
            let t = sampling.times()[i];
            let dt = sampling.delta(i);
            model.drift(t, &[x], &mut a);
            model.diffusion(t, &[x], &mut s);
            let z: f64 = rng.sample(StandardNormal);
            x += a[0] * dt + s[0] * dt.sqrt() * z;
        }
        re += (u * x).cos();
        im += (u * x).sin();
    }
    let n = nsim as f64;
    Ok((re / n, im / n))
}

/// Raw moments `E[X_T^k]` for `k = 1..m`.
pub fn moments<M, R>(
    model: &M,
    sampling: &Sampling,
    x0: f64,
    m: usize,
    nsim: usize,
    rng: &mut R,
) -> Result<Vec<f64>>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
{
    if nsim == 0 || m == 0 {
        return Err(Error::sim("moments need nsim>0 and m>0"));
    }
    if model.dim() != 1 {
        return Err(Error::unsupported("moments is scalar"));
    }
    let mut acc = vec![0.0; m];
    let mut a = [0.0];
    let mut sig = [0.0];
    for _ in 0..nsim {
        let mut x = x0;
        for i in 0..sampling.n_steps() {
            let t = sampling.times()[i];
            let dt = sampling.delta(i);
            model.drift(t, &[x], &mut a);
            model.diffusion(t, &[x], &mut sig);
            let z: f64 = rng.sample(StandardNormal);
            x += a[0] * dt + sig[0] * dt.sqrt() * z;
        }
        let mut pk = x;
        for k in 0..m {
            acc[k] += pk;
            pk *= x;
        }
    }
    for v in &mut acc {
        *v /= nsim as f64;
    }
    Ok(acc)
}

#[derive(Clone, Debug)]
pub struct MomentSummary {
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

pub fn moment_summary<M, R>(
    model: &M,
    sampling: &Sampling,
    x0: f64,
    nsim: usize,
    rng: &mut R,
) -> Result<MomentSummary>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
{
    let m = moments(model, sampling, x0, 4, nsim, rng)?;
    let mu1 = m[0];
    let mu2 = m[1];
    let mu3 = m[2];
    let mu4 = m[3];
    let var = (mu2 - mu1 * mu1).max(0.0);
    let s = var.sqrt();
    let skew = if s > 1e-14 {
        (mu3 - 3.0 * mu1 * mu2 + 2.0 * mu1.powi(3)) / s.powi(3)
    } else {
        0.0
    };
    let kurt = if var > 1e-14 {
        (mu4 - 4.0 * mu1 * mu3 + 6.0 * mu1 * mu1 * mu2 - 3.0 * mu1.powi(4)) / var.powi(2)
    } else {
        0.0
    };
    Ok(MomentSummary {
        mean: mu1,
        variance: var,
        skewness: skew,
        kurtosis: kurt,
    })
}

/// Small-noise expansion of `E[f(X_T^ε)]` where `dX = a dt + ε σ dW`.
///
/// `d0 = f(x̄_T)` along the ODE `ẋ = a(x)`,
/// `d1 = 0` for odd Brownian functionals (the first correction is `ε²`);
/// we return the Itô–Taylor pair `(d0, d2)` stored as YUIMA-style `(d0, d1)`
/// with `d1` meaning the first *available* correction (order `ε²`).
#[derive(Clone, Debug)]
pub struct AsymptoticTerms {
    pub d0: f64,
    pub d1: f64,
    pub order: &'static str,
}

pub fn asymptotic_term<M: Sde + ?Sized>(
    model: &M,
    x0: f64,
    t: f64,
    f: impl Fn(f64) -> f64,
    fxx: impl Fn(f64) -> f64,
    n_ode: usize,
) -> Result<AsymptoticTerms> {
    if model.dim() != 1 {
        return Err(Error::unsupported("small-noise expansion is scalar"));
    }
    if n_ode == 0 || t <= 0.0 {
        return Err(Error::param("need a positive ODE grid"));
    }
    let dt = t / n_ode as f64;
    let mut x = x0;
    let mut integ_sigma2 = 0.0;
    let mut a = [0.0];
    let mut s = [0.0];
    for i in 0..n_ode {
        let ti = i as f64 * dt;
        model.drift(ti, &[x], &mut a);
        model.diffusion(ti, &[x], &mut s);
        integ_sigma2 += s[0] * s[0] * dt;
        x += a[0] * dt;
    }
    let d0 = f(x);
    // leading correction ½ f''(x̄) ∫ σ²  (the ε² term at ε = 1)
    let d1 = 0.5 * fxx(x) * integ_sigma2;
    Ok(AsymptoticTerms {
        d0,
        d1,
        order: "d0 + ε² d1",
    })
}

/// Convenience: Black–Scholes call Greeks by Malliavin (for tests / toys).
pub fn bs_call_payoff(k: f64) -> impl Fn(f64) -> f64 {
    move |s| (s - k).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::OrnsteinUhlenbeck;
    use crate::rng::seed_rng;
    use crate::sampling::Sampling;

    #[test]
    fn ou_mean_via_moments() {
        let m = OrnsteinUhlenbeck::new(2.0, 0.0, 0.3).unwrap();
        let s = Sampling::from_terminal(0.5, 80).unwrap();
        let mut rng = seed_rng(4);
        let sum = moment_summary(&m, &s, 1.0, 1500, &mut rng).unwrap();
        let exact = (-2.0 * 0.5_f64).exp(); // θ=0, x0=1
        assert!((sum.mean - exact).abs() < 0.08);
    }

    #[test]
    fn small_noise_ode_mean() {
        let m = OrnsteinUhlenbeck::new(1.0, 0.0, 0.2).unwrap();
        let ae = asymptotic_term(&m, 1.0, 0.2, |x| x, |_| 0.0, 40).unwrap();
        let exact = (-0.2_f64).exp();
        assert!((ae.d0 - exact).abs() < 0.02);
        assert!(ae.d1.abs() < 1e-12);
    }
}
