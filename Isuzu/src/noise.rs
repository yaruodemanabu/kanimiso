//! Driving noise: Wiener, fractional Gaussian, Lévy, compound Poisson.

use amatsuki::{
    sample_gamma, sample_inverse_gaussian, sample_stable_cms, Distribution, Exp1, Normal, Poisson,
    Rng, StandardNormal, StudentT, Uniform,
};
use faer::Mat;
use rustfft::{num_complex::Complex, FftPlanner};

use crate::error::{Error, Result};
use crate::linalg::cholesky;
use crate::sampling::Sampling;

/// Independent standard Brownian increments `ΔW ~ N(0, Δt)` on a grid.
///
/// Output layout: `inc[i * n_noise + j]` is noise `j` on step `i`.
pub fn brownian_increments<R: Rng + ?Sized>(
    sampling: &Sampling,
    n_noise: usize,
    rng: &mut R,
) -> Result<Vec<f64>> {
    if n_noise == 0 {
        return Err(Error::dim("n_noise must be positive"));
    }
    let n = sampling.n_steps();
    let mut out = vec![0.0; n * n_noise];
    let normal = Normal::new(0.0, 1.0).map_err(|e| Error::numeric(e.to_string()))?;
    for i in 0..n {
        let s = sampling.delta(i).sqrt();
        for j in 0..n_noise {
            out[i * n_noise + j] = s * normal.sample(rng);
        }
    }
    Ok(out)
}

/// Correlated Brownian increments with instantaneous correlation `R` (`m × m`).
pub fn correlated_brownian_increments<R: Rng + ?Sized>(
    sampling: &Sampling,
    corr: &Mat<f64>,
    rng: &mut R,
) -> Result<Vec<f64>> {
    let m = corr.nrows();
    if corr.ncols() != m {
        return Err(Error::dim("correlation matrix must be square"));
    }
    let l = cholesky(corr)?;
    let raw = brownian_increments(sampling, m, rng)?;
    let n = sampling.n_steps();
    let mut out = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0;
            for k in 0..m {
                s += l[(j, k)] * raw[i * m + k];
            }
            out[i * m + j] = s;
        }
    }
    Ok(out)
}

/// Fractional Gaussian noise covariance `γ(k) = ½(|k+1|²ᴴ − 2|k|²ᴴ + |k−1|²ᴴ)`.
pub fn fgn_acf(k: i64, hurst: f64) -> f64 {
    let h2 = 2.0 * hurst;
    0.5 * (((k + 1).unsigned_abs() as f64).powf(h2) - 2.0 * (k.unsigned_abs() as f64).powf(h2)
        + ((k - 1).unsigned_abs() as f64).powf(h2))
}

/// Davies–Harte / Wood–Chan simulation of fractional Gaussian noise.
///
/// Returns `n` increments of an fGn with variance 1 on the unit-step grid.
/// Scale by `Δtᴴ` (or `Δt^{H}` for fBM increments on a regular grid of step `Δt`).
pub fn fractional_gaussian_noise<R: Rng + ?Sized>(
    n: usize,
    hurst: f64,
    rng: &mut R,
) -> Result<Vec<f64>> {
    if !(hurst > 0.0 && hurst < 1.0) {
        return Err(Error::param("Hurst must lie in (0, 1)"));
    }
    if n == 0 {
        return Err(Error::sampling("fGn length must be positive"));
    }
    // Standard Davies–Harte embedding: circulant of length m = 2n, including γ(n).
    // rustfft handles arbitrary lengths; padding to a power of two is unnecessary
    // and dropping γ(n) makes the embedding indefinite for H ≳ 0.85.
    let m = 2 * n;
    let mut circ = vec![0.0; m];
    circ[0] = fgn_acf(0, hurst);
    for k in 1..n {
        let g = fgn_acf(k as i64, hurst);
        circ[k] = g;
        circ[m - k] = g;
    }
    circ[n] = fgn_acf(n as i64, hurst);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(m);
    let ifft = planner.plan_fft_inverse(m);
    let mut spec: Vec<Complex<f64>> = circ.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut spec);
    // Eigenvalues of the circulant must be non-negative (up to roundoff).
    let mut lam = Vec::with_capacity(m);
    for z in &spec {
        let v = z.re;
        if v < -1e-10 {
            return Err(Error::numeric(
                "Davies–Harte embedding not positive; try a different n or Hurst",
            ));
        }
        lam.push(v.max(0.0));
    }
    let mut z: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); m];
    z[0] = Complex::new(lam[0].sqrt() * rng.sample(StandardNormal), 0.0);
    if m % 2 == 0 {
        z[m / 2] = Complex::new(lam[m / 2].sqrt() * rng.sample(StandardNormal), 0.0);
    }
    let last = if m % 2 == 0 { m / 2 } else { (m + 1) / 2 };
    for k in 1..last {
        let s = (0.5 * lam[k]).sqrt();
        let re: f64 = rng.sample(StandardNormal);
        let im: f64 = rng.sample(StandardNormal);
        z[k] = Complex::new(s * re, s * im);
        z[m - k] = z[k].conj();
    }
    ifft.process(&mut z);
    let scale = 1.0 / (m as f64).sqrt();
    Ok(z.iter().take(n).map(|c| c.re * scale).collect())
}

/// Fractional Brownian increments on a **regular** grid (`ΔBᴴ_i ≈ Δtᴴ ξᵢ`).
pub fn fractional_brownian_increments<R: Rng + ?Sized>(
    sampling: &Sampling,
    hurst: f64,
    rng: &mut R,
) -> Result<Vec<f64>> {
    if !sampling.is_regular() {
        return Err(Error::sampling("fBM increments require a regular grid"));
    }
    let n = sampling.n_steps();
    let fgn = fractional_gaussian_noise(n, hurst, rng)?;
    let dt = sampling.mean_delta();
    let scale = dt.powf(hurst);
    Ok(fgn.into_iter().map(|z| scale * z).collect())
}

/// Jump-size law for compound Poisson / Lévy jumps.
#[derive(Clone, Debug)]
pub enum JumpLaw {
    /// `N(μ, σ²)`.
    Normal {
        mu: f64,
        sigma: f64,
    },
    /// Kou double-exponential: `p η₊ e^{-η₊ z} 1_{z>0} + (1-p) η₋ e^{η₋ z} 1_{z<0}`.
    DoubleExponential {
        p: f64,
        eta_plus: f64,
        eta_minus: f64,
    },
    Constant {
        size: f64,
    },
    /// Scaled Student-t.
    StudentT {
        df: f64,
        scale: f64,
    },
    /// Symmetric Laplace (difference of exponentials).
    Laplace {
        scale: f64,
    },
}

impl JumpLaw {
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<f64> {
        match *self {
            JumpLaw::Normal { mu, sigma } => {
                let n = Normal::new(mu, sigma).map_err(|e| Error::param(e.to_string()))?;
                Ok(n.sample(rng))
            }
            JumpLaw::DoubleExponential {
                p,
                eta_plus,
                eta_minus,
            } => {
                if !(0.0..=1.0).contains(&p) || eta_plus <= 0.0 || eta_minus <= 0.0 {
                    return Err(Error::param("invalid Kou parameters"));
                }
                let u: f64 = rng.sample(Uniform::new(0.0, 1.0));
                let e: f64 = rng.sample(Exp1);
                if u < p {
                    Ok(e / eta_plus)
                } else {
                    Ok(-e / eta_minus)
                }
            }
            JumpLaw::Constant { size } => Ok(size),
            JumpLaw::StudentT { df, scale } => {
                let t = StudentT::new(df).map_err(|e| Error::param(e.to_string()))?;
                Ok(scale * t.sample(rng))
            }
            JumpLaw::Laplace { scale } => {
                if scale <= 0.0 {
                    return Err(Error::param("Laplace scale must be positive"));
                }
                let e1: f64 = rng.sample(Exp1);
                let e2: f64 = rng.sample(Exp1);
                Ok(scale * (e1 - e2))
            }
        }
    }
}

/// Lévy measure / driving noise specification (YUIMA `measure`).
#[derive(Clone, Debug)]
pub enum LevyMeasure {
    /// Compound Poisson: intensity `λ`, jump sizes from `law`.
    CompoundPoisson { intensity: f64, law: JumpLaw },
    /// Brownian motion with drift (used as a Lévy driver).
    Gaussian { mu: f64, sigma: f64 },
    /// Variance Gamma: `X = θ G + σ W(G)`, `G ~ Gamma(t/ν, ν)`.
    VarianceGamma { sigma: f64, theta: f64, nu: f64 },
    /// Normal Inverse Gaussian via IG subordinator.
    Nig { alpha: f64, beta: f64, delta: f64 },
    /// α-stable via Chambers–Mallows–Stuck.
    Stable {
        alpha: f64,
        beta: f64,
        sigma: f64,
        mu: f64,
    },
    /// Gamma subordinator: increment `Gamma(rate · dt, scale)`
    /// (shape-scale), so `E[Δ] = rate · scale · dt`,
    /// `Var(Δ) = rate · scale² · dt`.
    Gamma { rate: f64, scale: f64 },
}

impl LevyMeasure {
    /// Simulate a Lévy increment over an interval of length `dt`.
    pub fn increment<R: Rng + ?Sized>(&self, dt: f64, rng: &mut R) -> Result<f64> {
        if dt < 0.0 {
            return Err(Error::sampling("dt must be non-negative"));
        }
        if dt == 0.0 {
            return Ok(0.0);
        }
        match self {
            LevyMeasure::CompoundPoisson { intensity, law } => {
                if *intensity < 0.0 {
                    return Err(Error::param("intensity must be non-negative"));
                }
                let lam = intensity * dt;
                let n = if lam == 0.0 {
                    0
                } else {
                    let p = Poisson::new(lam).map_err(|e| Error::numeric(e.to_string()))?;
                    p.sample(rng) as usize
                };
                let mut s = 0.0;
                for _ in 0..n {
                    s += law.sample(rng)?;
                }
                Ok(s)
            }
            LevyMeasure::Gaussian { mu, sigma } => {
                let n = Normal::new(mu * dt, sigma * dt.sqrt())
                    .map_err(|e| Error::param(e.to_string()))?;
                Ok(n.sample(rng))
            }
            LevyMeasure::VarianceGamma { sigma, theta, nu } => {
                if *sigma <= 0.0 || *nu <= 0.0 {
                    return Err(Error::param("VG requires sigma > 0, nu > 0"));
                }
                // G ~ Gamma(dt/ν, ν) so E[G]=dt, Var(G)=ν dt.
                let shape = dt / nu;
                let scale = *nu;
                let g = sample_gamma(shape, scale, rng).map_err(|e| Error::param(e.to_string()))?;
                let z: f64 = rng.sample(StandardNormal);
                Ok(theta * g + sigma * g.sqrt() * z)
            }
            LevyMeasure::Nig { alpha, beta, delta } => {
                if *alpha <= beta.abs() || *delta <= 0.0 {
                    return Err(Error::param("NIG requires α > |β|, δ > 0"));
                }
                let gamma = (alpha * alpha - beta * beta).sqrt();
                // IG(δ dt / γ, (δ dt)²)  — inverse Gaussian subordinator increment.
                let ig_mean = delta * dt / gamma;
                let ig_shape = (delta * dt) * (delta * dt);
                let g = sample_inverse_gaussian(ig_mean, ig_shape, rng)
                    .map_err(|e| Error::param(e.to_string()))?;
                let z: f64 = rng.sample(StandardNormal);
                Ok(beta * g + g.sqrt() * z)
            }
            LevyMeasure::Stable {
                alpha,
                beta,
                sigma,
                mu,
            } => {
                if !(0.0 < *alpha && *alpha <= 2.0) || !(beta.abs() <= 1.0) || *sigma < 0.0 {
                    return Err(Error::param("invalid stable parameters"));
                }
                let x = sample_stable_cms(*alpha, *beta, rng)
                    .map_err(|e| Error::param(e.to_string()))?;
                Ok(mu * dt + sigma * dt.powf(1.0 / alpha) * x)
            }
            LevyMeasure::Gamma { rate, scale } => {
                if !(*rate > 0.0 && *scale > 0.0 && rate.is_finite() && scale.is_finite()) {
                    return Err(Error::param("Gamma subordinator needs rate>0, scale>0"));
                }
                let shape = rate * dt;
                sample_gamma(shape, *scale, rng).map_err(|e| Error::param(e.to_string()))
            }
        }
    }

    /// Lévy increments on a grid (one-dimensional driver).
    pub fn increments<R: Rng + ?Sized>(
        &self,
        sampling: &Sampling,
        rng: &mut R,
    ) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(sampling.n_steps());
        for i in 0..sampling.n_steps() {
            out.push(self.increment(sampling.delta(i), rng)?);
        }
        Ok(out)
    }
}

/// Homogeneous Poisson arrival times on `[t0, t1)`.
pub fn poisson_arrivals<R: Rng + ?Sized>(
    t0: f64,
    t1: f64,
    intensity: f64,
    rng: &mut R,
) -> Result<Vec<f64>> {
    if t1 <= t0 {
        return Err(Error::sampling("poisson interval must be non-empty"));
    }
    if !intensity.is_finite() || intensity < 0.0 {
        return Err(Error::param("intensity must be finite and non-negative"));
    }
    if !t0.is_finite() || !t1.is_finite() {
        return Err(Error::sampling("poisson window must be finite"));
    }
    if intensity == 0.0 {
        return Ok(Vec::new());
    }
    let mut t = t0;
    let mut out = Vec::new();
    const MAX_EVENTS: usize = 10_000_000;
    loop {
        let e: f64 = rng.sample(Exp1);
        let t_new = t + e / intensity;
        if !t_new.is_finite() || t_new <= t {
            return Err(Error::numeric(
                "poisson clock did not advance (non-finite intensity or RNG)",
            ));
        }
        t = t_new;
        if t >= t1 {
            break;
        }
        out.push(t);
        if out.len() > MAX_EVENTS {
            return Err(Error::sim("poisson event cap exceeded"));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed_rng;
    use crate::sampling::Sampling;

    #[test]
    fn brownian_qv_near_t() {
        let s = Sampling::from_terminal(1.0, 4000).unwrap();
        let mut rng = seed_rng(7);
        let dw = brownian_increments(&s, 1, &mut rng).unwrap();
        let qv: f64 = dw.iter().map(|x| x * x).sum();
        assert!((qv - 1.0).abs() < 0.08);
    }

    #[test]
    fn fgn_unit_variance() {
        let mut rng = seed_rng(3);
        let z = fractional_gaussian_noise(2048, 0.7, &mut rng).unwrap();
        let m: f64 = z.iter().sum::<f64>() / z.len() as f64;
        let v: f64 = z.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / z.len() as f64;
        assert!((v - 1.0).abs() < 0.15);
    }

    #[test]
    fn fgn_high_hurst_embeds() {
        let mut rng = seed_rng(4);
        for n in [64usize, 100, 256] {
            let z = fractional_gaussian_noise(n, 0.9, &mut rng)
                .unwrap_or_else(|e| panic!("H=0.9 n={n}: {e}"));
            assert_eq!(z.len(), n);
            assert!(z.iter().all(|x| x.is_finite()));
        }
        let z = fractional_gaussian_noise(64, 0.95, &mut rng).unwrap();
        assert!(z.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn poisson_rejects_infinite_intensity() {
        let mut rng = seed_rng(1);
        assert!(poisson_arrivals(0.0, 1.0, f64::INFINITY, &mut rng).is_err());
        assert!(poisson_arrivals(0.0, 1.0, f64::NAN, &mut rng).is_err());
    }

    #[test]
    fn compound_poisson_mean() {
        let law = LevyMeasure::CompoundPoisson {
            intensity: 4.0,
            law: JumpLaw::Normal {
                mu: 1.0,
                sigma: 0.01,
            },
        };
        let mut rng = seed_rng(11);
        let mut s = 0.0;
        let n = 2000;
        for _ in 0..n {
            s += law.increment(1.0, &mut rng).unwrap();
        }
        let mean = s / n as f64;
        assert!((mean - 4.0).abs() < 0.3);
    }
}
