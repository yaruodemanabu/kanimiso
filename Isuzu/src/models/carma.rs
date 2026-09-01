//! Continuous-time ARMA (`setCarma` in YUIMA).
//!
//! Brockwell (2000) state-space form:
//!
//! ```text
//! Y_t = bᵀ X_t + c
//! dX_t = A X_t dt + e dL_t
//! ```
//!
//! with companion matrix `A` built from AR coefficients `a₁,…,aₚ` and MA
//! coefficients `b₀,…,b_q` (`q < p`), `e = (0,…,0,1)ᵀ`.

use faer::{Col, Mat};

use crate::error::{Error, Result};
use crate::linalg::{is_hurwitz, matvec_slice};
use crate::model::Sde;
use crate::noise::LevyMeasure;

/// CARMA(p, q) process.
#[derive(Clone, Debug)]
pub struct Carma {
    pub p: usize,
    pub q: usize,
    /// AR coefficients `a[0]=a₁, …, a[p-1]=aₚ` of the companion last row.
    pub ar: Vec<f64>,
    /// MA coefficients `b[0]=b₀, …, b[q]=b_q`.
    pub ma: Vec<f64>,
    pub loc: f64,
    pub levy: LevyMeasure,
    a: Mat<f64>,
    bvec: Col<f64>,
}

impl Carma {
    pub fn new(ar: Vec<f64>, ma: Vec<f64>, loc: f64, levy: LevyMeasure) -> Result<Self> {
        let p = ar.len();
        if p == 0 {
            return Err(Error::param("CARMA requires p ≥ 1"));
        }
        if ma.len() > p {
            return Err(Error::param("CARMA requires q < p (len(ma) ≤ p)"));
        }
        if ma.is_empty() {
            return Err(Error::param("CARMA requires at least b0"));
        }
        let q = ma.len() - 1;
        let mut a = crate::linalg::mat_zeros(p, p);
        for i in 0..p.saturating_sub(1) {
            a[(i, i + 1)] = 1.0;
        }
        for j in 0..p {
            a[(p - 1, j)] = -ar[p - 1 - j];
        }
        let mut bvec = crate::linalg::col_zeros(p);
        for (i, &bi) in ma.iter().enumerate() {
            bvec[i] = bi;
        }
        Ok(Self {
            p,
            q,
            ar,
            ma,
            loc,
            levy,
            a,
            bvec,
        })
    }

    pub fn companion(&self) -> &Mat<f64> {
        &self.a
    }

    pub fn is_causal(&self) -> bool {
        is_hurwitz(&self.a)
    }

    /// Observation `Y = bᵀ X + c`.
    pub fn observe(&self, x: &[f64]) -> f64 {
        let mut s = self.loc;
        for i in 0..self.p {
            s += self.bvec[i] * x[i];
        }
        s
    }
}

impl Sde for Carma {
    fn dim(&self) -> usize {
        self.p
    }
    fn n_noise(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let d = matvec_slice(&self.a, x);
        for i in 0..self.p {
            out[i] = d[i];
        }
        if let LevyMeasure::Gaussian { mu, .. } = self.levy {
            out[self.p - 1] += mu;
        }
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        // Continuous Gaussian part is zero unless the Lévy measure is Gaussian.
        // The last coordinate is driven by dL; if L is Brownian we put σ = e.
        match self.levy {
            LevyMeasure::Gaussian { sigma, .. } => {
                for v in out.iter_mut() {
                    *v = 0.0;
                }
                out[self.p - 1] = sigma;
            }
            _ => {
                for v in out.iter_mut() {
                    *v = 0.0;
                }
            }
        }
    }
    fn jump_coeff(&self, _t: f64, _x: &[f64], out: &mut [f64]) -> bool {
        match self.levy {
            LevyMeasure::Gaussian { .. } => false,
            _ => {
                for v in out.iter_mut() {
                    *v = 0.0;
                }
                out[self.p - 1] = 1.0;
                true
            }
        }
    }
    fn levy(&self) -> Option<&LevyMeasure> {
        match self.levy {
            LevyMeasure::Gaussian { .. } => None,
            _ => Some(&self.levy),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::JumpLaw;

    #[test]
    fn carma11_companion() {
        let c = Carma::new(
            vec![1.5],
            vec![1.0],
            0.0,
            LevyMeasure::CompoundPoisson {
                intensity: 1.0,
                law: JumpLaw::Normal {
                    mu: 0.0,
                    sigma: 1.0,
                },
            },
        )
        .unwrap();
        assert_eq!(c.p, 1);
        assert!(c.is_causal());
        assert!((c.companion()[(0, 0)] + 1.5).abs() < 1e-14);
    }
}
