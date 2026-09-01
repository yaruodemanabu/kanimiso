//! Continuous-time GARCH (`setCogarch` in YUIMA).
//!
//! Klüppelberg–Lindner–Maller COGARCH(1,1):
//!
//! ```text
//! dG_t = √V_t dL_t
//! dV_t = (β − η V_t) dt + φ V_{t−} d[L,L]ᵈ_t
//! ```
//!
//! Brockwell–Marquardt–Maller COGARCH(p,q) uses a linear state `Y`:
//!
//! ```text
//! V_t = a₀ + aᵀ Y_{t−}
//! dY_t = B Y_t dt + e V_{t−} (ΔL_t)²
//! dG_t = √V_t dL_t
//! ```
//!
//! State is `[G, Y₁, …, Y_q]` (`q` latent coordinates). For (1,1),
//! `Y` is scalar and `V = a₀ + a₁ Y`.

use crate::error::{Error, Result};
use crate::model::Sde;
use crate::noise::LevyMeasure;

/// COGARCH(p, q) with `p = ma.len()`, `q = ar.len()`.
///
/// `ar` = `(b₁,…,b_q)` (mean-reversion / companion),
/// `ma` = `(a₁,…,a_p)` (volatility loadings), `a0` = location.
#[derive(Clone, Debug)]
pub struct Cogarch {
    pub p: usize,
    pub q: usize,
    pub a0: f64,
    pub ma: Vec<f64>,
    pub ar: Vec<f64>,
    pub levy: LevyMeasure,
}

impl Cogarch {
    pub fn new(a0: f64, ma: Vec<f64>, ar: Vec<f64>, levy: LevyMeasure) -> Result<Self> {
        if a0 <= 0.0 {
            return Err(Error::param("COGARCH a0 must be positive"));
        }
        if ma.is_empty() || ar.is_empty() {
            return Err(Error::param("COGARCH requires p,q ≥ 1"));
        }
        if ma.iter().any(|x| *x < 0.0) || ar.iter().any(|x| *x <= 0.0) {
            return Err(Error::param("COGARCH MA ≥ 0 and AR > 0"));
        }
        Ok(Self {
            p: ma.len(),
            q: ar.len(),
            a0,
            ma,
            ar,
            levy,
        })
    }

    /// Klüppelberg COGARCH(1,1) in `(β, η, φ)` coordinates
    /// (`a0 = β/η`, `a1 = φ`, `b1 = η` up to the usual identification).
    pub fn cogarch11(beta: f64, eta: f64, phi: f64, levy: LevyMeasure) -> Result<Self> {
        if beta <= 0.0 || eta <= 0.0 || phi <= 0.0 {
            return Err(Error::param("COGARCH(1,1) β, η, φ must be positive"));
        }
        Self::new(beta / eta, vec![phi], vec![eta], levy)
    }

    pub fn variance(&self, y: &[f64]) -> f64 {
        let mut v = self.a0;
        let k = self.p.min(y.len());
        for i in 0..k {
            v += self.ma[i] * y[i];
        }
        v.max(1e-16)
    }
}

impl Sde for Cogarch {
    fn dim(&self) -> usize {
        1 + self.q
    }
    fn n_noise(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        // x = [G, Y1, …, Yq]
        let y = &x[1..];
        out[0] = 0.0; // G is a pure Lévy integral
                      // Companion drift B Y
        if self.q == 1 {
            out[1] = -self.ar[0] * y[0];
        } else {
            for i in 0..self.q.saturating_sub(1) {
                out[1 + i] = y[i + 1];
            }
            let mut last = 0.0;
            for j in 0..self.q {
                last -= self.ar[self.q - 1 - j] * y[j];
            }
            out[self.q] = last;
        }
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        // Continuous Gaussian driver handled via jump/Lévy path; if L is
        // Gaussian we put dG = √V dW in diffusion.
        if let LevyMeasure::Gaussian { sigma, .. } = self.levy {
            // filled in simulate via jump_coeff + levy; still allow σ dW
            let _ = sigma;
        }
    }
    fn jump_coeff(&self, _t: f64, x: &[f64], out: &mut [f64]) -> bool {
        let y = &x[1..];
        let v = self.variance(y);
        let sv = v.sqrt();
        out[0] = sv;
        // Discrete quadratic variation feeds the latent equation:
        // ΔY += e * V * (ΔL)²  is *quadratic* in the jump, so the linear
        // jump_coeff cannot express it. `simulate` special-cases Cogarch
        // via `cogarch_jump`. We still mark jumps as present.
        for i in 1..out.len() {
            out[i] = 0.0;
        }
        true
    }
    fn levy(&self) -> Option<&LevyMeasure> {
        Some(&self.levy)
    }
}

/// Extra COGARCH jump: `ΔY += e V (ΔL)²` after the linear `ΔG += √V ΔL`.
pub fn cogarch_state_jump(model: &Cogarch, x: &mut [f64], dl: f64) {
    let y = &x[1..];
    let v = model.variance(y);
    let dq = dl * dl;
    if model.q == 1 {
        x[1] += v * dq;
    } else {
        x[model.q] += v * dq;
    }
    x[1..].iter_mut().for_each(|yi| {
        if *yi < 0.0 {
            *yi = 0.0;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::{JumpLaw, LevyMeasure};

    #[test]
    fn cogarch11_variance() {
        let c = Cogarch::cogarch11(
            0.04,
            0.8,
            0.3,
            LevyMeasure::CompoundPoisson {
                intensity: 2.0,
                law: JumpLaw::Normal {
                    mu: 0.0,
                    sigma: 0.2,
                },
            },
        )
        .unwrap();
        assert!((c.variance(&[0.0]) - 0.05).abs() < 1e-12);
    }
}
