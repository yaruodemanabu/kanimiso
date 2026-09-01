//! Point processes: Poisson, inhomogeneous Poisson, exponential Hawkes.

use amatsuki::{Exp1, Rng, Uniform};

use crate::error::{Error, Result};
use crate::noise::poisson_arrivals;
use crate::path::TickSeries;

/// Homogeneous Poisson process of intensity `λ`.
#[derive(Clone, Debug)]
pub struct HomogeneousPoisson {
    pub intensity: f64,
}

impl HomogeneousPoisson {
    pub fn new(intensity: f64) -> Result<Self> {
        if !intensity.is_finite() || intensity < 0.0 {
            return Err(Error::param("intensity must be finite and non-negative"));
        }
        Ok(Self { intensity })
    }

    pub fn simulate<R: Rng + ?Sized>(&self, t0: f64, t1: f64, rng: &mut R) -> Result<TickSeries> {
        let times = poisson_arrivals(t0, t1, self.intensity, rng)?;
        let values: Vec<f64> = (1..=times.len()).map(|k| k as f64).collect();
        if times.is_empty() {
            return TickSeries::new(vec![t0, t1], vec![0.0, 0.0]);
        }
        let mut t = Vec::with_capacity(times.len() + 1);
        let mut v = Vec::with_capacity(times.len() + 1);
        t.push(t0);
        v.push(0.0);
        t.extend(times);
        v.extend(values);
        TickSeries::new(t, v)
    }

    /// Log-likelihood of arrival times (excluding the endpoints).
    pub fn loglik(&self, arrivals: &[f64], t0: f64, t1: f64) -> Result<f64> {
        if self.intensity < 0.0 {
            return Err(Error::param("intensity must be non-negative"));
        }
        if self.intensity == 0.0 {
            return if arrivals.is_empty() {
                Ok(0.0)
            } else {
                Ok(f64::NEG_INFINITY)
            };
        }
        let n = arrivals.len() as f64;
        Ok(n * self.intensity.ln() - self.intensity * (t1 - t0))
    }
}

/// Inhomogeneous Poisson with intensity `λ(t)`.
pub struct InhomogeneousPoisson<F> {
    pub intensity: F,
    pub intensity_max: f64,
}

impl<F> InhomogeneousPoisson<F>
where
    F: Fn(f64) -> f64,
{
    pub fn new(intensity: F, intensity_max: f64) -> Result<Self> {
        if intensity_max <= 0.0 {
            return Err(Error::param("intensity_max must be positive"));
        }
        Ok(Self {
            intensity,
            intensity_max,
        })
    }

    /// Ogata thinning.
    pub fn simulate<R: Rng + ?Sized>(&self, t0: f64, t1: f64, rng: &mut R) -> Result<Vec<f64>> {
        let cand = poisson_arrivals(t0, t1, self.intensity_max, rng)?;
        let mut out = Vec::new();
        for t in cand {
            let lam = (self.intensity)(t);
            if lam < 0.0 {
                return Err(Error::param("intensity must be non-negative"));
            }
            let u: f64 = rng.sample(Uniform::new(0.0, 1.0));
            if u * self.intensity_max <= lam {
                out.push(t);
            }
        }
        Ok(out)
    }
}

/// Univariate exponential Hawkes process
/// `λ(t) = μ + Σ_{tᵢ < t} α e^{-β (t − tᵢ)}`.
#[derive(Clone, Debug)]
pub struct ExponentialHawkes {
    pub mu: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl ExponentialHawkes {
    pub fn new(mu: f64, alpha: f64, beta: f64) -> Result<Self> {
        if mu < 0.0 || alpha < 0.0 || beta <= 0.0 {
            return Err(Error::param("Hawkes requires μ ≥ 0, α ≥ 0, β > 0"));
        }
        if alpha >= beta {
            // still simulable but explosive
        }
        Ok(Self { mu, alpha, beta })
    }

    pub fn branching_ratio(&self) -> f64 {
        self.alpha / self.beta
    }

    pub fn stationary_intensity(&self) -> Result<f64> {
        let r = self.branching_ratio();
        if r >= 1.0 {
            return Err(Error::param("Hawkes is critical/explosive (α ≥ β)"));
        }
        Ok(self.mu / (1.0 - r))
    }

    /// Intensity at time `t` given the history `arrivals` (strictly before `t`).
    pub fn intensity(&self, t: f64, arrivals: &[f64]) -> f64 {
        let mut s = self.mu;
        for &ti in arrivals {
            if ti < t {
                s += self.alpha * (-self.beta * (t - ti)).exp();
            }
        }
        s
    }

    /// Fast intensity via the recursive state `R` (`λ = μ + α R`).
    pub fn intensity_from_state(&self, r: f64) -> f64 {
        self.mu + self.alpha * r
    }

    /// Ogata thinning on `[t0, t1)`.
    pub fn simulate<R: Rng + ?Sized>(&self, t0: f64, t1: f64, rng: &mut R) -> Result<Vec<f64>> {
        if t1 <= t0 {
            return Err(Error::sampling("Hawkes interval must be non-empty"));
        }
        let mut t = t0;
        let mut arrivals = Vec::new();
        let mut r = 0.0; // Σ e^{-β(t − tᵢ)}
        loop {
            let lam_bar = self.mu + self.alpha * r;
            if lam_bar <= 0.0 {
                break;
            }
            let e: f64 = rng.sample(Exp1);
            t += e / lam_bar;
            if t >= t1 {
                break;
            }
            r *= (-self.beta * e / lam_bar).exp();
            let lam = self.mu + self.alpha * r;
            let u: f64 = rng.sample(Uniform::new(0.0, 1.0));
            if u * lam_bar <= lam {
                arrivals.push(t);
                r += 1.0;
            }
        }
        Ok(arrivals)
    }

    /// Exact log-likelihood (exponential kernel, closed integral).
    pub fn loglik(&self, arrivals: &[f64], t0: f64, t1: f64) -> Result<f64> {
        if self.beta <= 0.0 {
            return Err(Error::param("beta must be positive"));
        }
        let mut ll = 0.0;
        let mut r = 0.0;
        let mut prev = t0;
        for &ti in arrivals {
            if ti < t0 || ti > t1 {
                return Err(Error::sampling("arrival outside window"));
            }
            let dt = ti - prev;
            r *= (-self.beta * dt).exp();
            let lam = self.mu + self.alpha * r;
            if lam <= 0.0 {
                return Ok(f64::NEG_INFINITY);
            }
            ll += lam.ln();
            r += 1.0;
            prev = ti;
        }
        // ∫ λ = μ (T) + (α/β) Σ (1 − e^{-β(T−tᵢ)})
        let tspan = t1 - t0;
        let mut integral = self.mu * tspan;
        for &ti in arrivals {
            integral += (self.alpha / self.beta) * (1.0 - (-self.beta * (t1 - ti)).exp());
        }
        Ok(ll - integral)
    }

    /// MLE by Nelder–Mead on `(μ, α, β)` from a start point.
    pub fn mle(arrivals: &[f64], t0: f64, t1: f64, start: [f64; 3]) -> Result<(Self, f64)> {
        use crate::optimize::{nelder_mead, OptOptions};

        let f = |p: &[f64]| {
            if p[0] <= 0.0 || p[1] < 0.0 || p[2] <= 0.0 {
                return 1e16;
            }
            match Self::new(p[0], p[1], p[2]).and_then(|h| h.loglik(arrivals, t0, t1)) {
                Ok(ll) if ll.is_finite() => -ll,
                _ => 1e16,
            }
        };
        let opt = nelder_mead(&f, &start, None, None, OptOptions::default())?;
        let h = Self::new(opt.x[0], opt.x[1], opt.x[2])?;
        let ll = h.loglik(arrivals, t0, t1)?;
        Ok((h, ll))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed_rng;

    #[test]
    fn poisson_count_mean() {
        let p = HomogeneousPoisson::new(5.0).unwrap();
        let mut rng = seed_rng(1);
        let mut s = 0.0;
        let n = 400;
        for _ in 0..n {
            let path = p.simulate(0.0, 1.0, &mut rng).unwrap();
            s += path.values.last().copied().unwrap_or(0.0);
        }
        assert!((s / n as f64 - 5.0).abs() < 0.5);
    }

    #[test]
    fn hawkes_stationary() {
        let h = ExponentialHawkes::new(0.5, 0.4, 1.0).unwrap();
        assert!((h.stationary_intensity().unwrap() - 0.5 / 0.6).abs() < 1e-14);
    }
}
