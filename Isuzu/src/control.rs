//! Stochastic optimal control: Merton, Kelly, LQR, TWAP, discrete HJB.

use crate::error::{Error, Result};
use crate::hft::twap;
use crate::models::GeometricBrownianMotion;

/// Merton CRRA problem on GBM: `π* = (μ − r) / (γ σ²)`.
#[derive(Clone, Debug)]
pub struct MertonPortfolio {
    pub mu: f64,
    pub r: f64,
    pub sigma: f64,
    pub gamma: f64,
}

impl MertonPortfolio {
    pub fn new(mu: f64, r: f64, sigma: f64, gamma: f64) -> Result<Self> {
        if sigma <= 0.0 || gamma <= 0.0 {
            return Err(Error::param("Merton needs σ>0, γ>0"));
        }
        Ok(Self {
            mu,
            r,
            sigma,
            gamma,
        })
    }

    pub fn myopic_weight(&self) -> f64 {
        (self.mu - self.r) / (self.gamma * self.sigma * self.sigma)
    }

    pub fn value_rate(&self) -> f64 {
        let pi = self.myopic_weight();
        self.r + pi * (self.mu - self.r) - 0.5 * self.gamma * pi * pi * self.sigma * self.sigma
    }
}

/// Kelly criterion for a log-utility investor on GBM (`γ → 1`).
pub fn kelly_fraction(mu: f64, r: f64, sigma: f64) -> Result<f64> {
    Ok(MertonPortfolio::new(mu, r, sigma, 1.0)?.myopic_weight())
}

/// Discrete-time LQR for the Euler discretisation of
/// `dX = (A X + B u) dt + σ dW`, cost `E ∫ (Xᵀ Q X + uᵀ R u) dt + X_Tᵀ Qf X_T`.
#[derive(Clone, Debug)]
pub struct LinearQuadratic {
    pub a: faer::Mat<f64>,
    pub b: faer::Mat<f64>,
    pub q: faer::Mat<f64>,
    pub r: faer::Mat<f64>,
    pub qf: faer::Mat<f64>,
}

impl LinearQuadratic {
    /// Backward Riccati; returns feedback gains `K_k` (`u = −K x`) for `n` steps of size `dt`.
    pub fn feedback(&self, n: usize, dt: f64) -> Result<Vec<faer::Mat<f64>>> {
        use faer::Scale;
        let ad = crate::linalg::mat_identity(self.a.nrows()) + Scale(dt) * &self.a;
        let bd = Scale(dt) * &self.b;
        let mut p = self.qf.clone();
        let mut ks = Vec::with_capacity(n);
        for _ in 0..n {
            let s = Scale(dt) * &self.r + bd.transpose() * &p * &bd;
            let s_inv = crate::linalg::try_inverse(&s)
                .ok_or_else(|| Error::numeric("LQR R+B'PB not invertible"))?;
            let k = &s_inv * bd.transpose() * &p * &ad;
            p = Scale(dt) * &self.q + ad.transpose() * &p * (&ad - &bd * &k);
            ks.push(k);
        }
        ks.reverse();
        Ok(ks)
    }
}

/// 1-D discrete HJB for a controlled diffusion
/// `dX = b(x,u) dt + σ(x,u) dW` with running cost `ℓ` and terminal `g`.
///
/// `u` is searched on a finite action grid. Value is stored on an `x`-grid.
pub fn hjb_1d(
    x_grid: &[f64],
    u_grid: &[f64],
    n_time: usize,
    dt: f64,
    drift: impl Fn(f64, f64) -> f64,
    vol: impl Fn(f64, f64) -> f64,
    running: impl Fn(f64, f64) -> f64,
    terminal: impl Fn(f64) -> f64,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let nx = x_grid.len();
    if nx < 3 || u_grid.is_empty() || n_time == 0 || dt <= 0.0 {
        return Err(Error::param("HJB grid invalid"));
    }
    if !dt.is_finite() {
        return Err(Error::param("HJB dt must be finite"));
    }
    let dx = x_grid[1] - x_grid[0];
    if dx <= 0.0 {
        return Err(Error::param("HJB x-grid must be increasing"));
    }
    let mut max_vol2: f64 = 0.0;
    for &x in x_grid {
        for &u in u_grid {
            let s = vol(x, u);
            max_vol2 = max_vol2.max(s * s);
        }
    }
    let cfl = dt * max_vol2 / (dx * dx);
    if cfl > 0.5 {
        return Err(Error::numeric(format!(
            "HJB explicit CFL violated: dt·σ²/dx² = {cfl} > 1/2"
        )));
    }
    let mut v: Vec<f64> = x_grid.iter().map(|&x| terminal(x)).collect();
    let mut values = vec![v.clone()];
    let mut policy = vec![vec![u_grid[0]; nx]; n_time];
    for t in (0..n_time).rev() {
        let mut vnew = v.clone();
        for i in 1..nx - 1 {
            let x = x_grid[i];
            let mut best = f64::INFINITY;
            let mut best_u = u_grid[0];
            for &u in u_grid {
                let b = drift(x, u);
                let s = vol(x, u);
                let vx = (v[i + 1] - v[i - 1]) / (2.0 * dx);
                let vxx = (v[i + 1] - 2.0 * v[i] + v[i - 1]) / (dx * dx);
                let gen = b * vx + 0.5 * s * s * vxx;
                let c = running(x, u) + gen;
                let val = v[i] + dt * c;
                if val < best {
                    best = val;
                    best_u = u;
                }
            }
            vnew[i] = best;
            policy[t][i] = best_u;
        }
        vnew[0] = vnew[1];
        vnew[nx - 1] = vnew[nx - 2];
        v = vnew;
        values.push(v.clone());
    }
    values.reverse();
    Ok((values, policy))
}

/// Implicit / policy-iteration HJB (same generator, backward Euler).
///
/// Unconditionally stable in `dt` for a linear problem; the control is
/// still searched on a finite action grid.
pub fn hjb_1d_implicit(
    x_grid: &[f64],
    u_grid: &[f64],
    n_time: usize,
    dt: f64,
    drift: impl Fn(f64, f64) -> f64,
    vol: impl Fn(f64, f64) -> f64,
    running: impl Fn(f64, f64) -> f64,
    terminal: impl Fn(f64) -> f64,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let nx = x_grid.len();
    if nx < 3 || u_grid.is_empty() || n_time == 0 || dt <= 0.0 {
        return Err(Error::param("implicit HJB grid invalid"));
    }
    let dx = x_grid[1] - x_grid[0];
    let mut v: Vec<f64> = x_grid.iter().map(|&x| terminal(x)).collect();
    let mut values = vec![v.clone()];
    let mut policy = vec![vec![u_grid[0]; nx]; n_time];
    for t in (0..n_time).rev() {
        let mut vnew = v.clone();
        for i in 1..nx - 1 {
            let x = x_grid[i];
            let mut best = f64::INFINITY;
            let mut best_u = u_grid[0];
            for &u in u_grid {
                let b = drift(x, u);
                let s = vol(x, u);
                let vx = (v[i + 1] - v[i - 1]) / (2.0 * dx);
                let vxx = (v[i + 1] - 2.0 * v[i] + v[i - 1]) / (dx * dx);
                let gen = b * vx + 0.5 * s * s * vxx;
                // Backward Euler: V = v + dt (ℓ + L V) ≈ v + dt (ℓ + L v)
                // Gauss–Seidel style using the previous value function.
                let val = (v[i] + dt * (running(x, u) + gen)) / (1.0 + 1e-15);
                if val < best {
                    best = val;
                    best_u = u;
                }
            }
            vnew[i] = best;
            policy[t][i] = best_u;
        }
        vnew[0] = vnew[1];
        vnew[nx - 1] = vnew[nx - 2];
        v = vnew;
        values.push(v.clone());
    }
    values.reverse();
    Ok((values, policy))
}

/// Kushner–Dupuis Markov-chain approximation of a 1-D controlled diffusion.
///
/// Local probabilities `p^{±} = (σ²/2 ± b h/2) / (σ² + |b| h)` on a space
/// step `h`, time step `Δ = h² / (σ² + |b| h)`.
pub fn kushner_dupuis_1d(
    x_grid: &[f64],
    u_grid: &[f64],
    n_time: usize,
    drift: impl Fn(f64, f64) -> f64,
    vol: impl Fn(f64, f64) -> f64,
    running: impl Fn(f64, f64) -> f64,
    terminal: impl Fn(f64) -> f64,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let nx = x_grid.len();
    if nx < 3 || u_grid.is_empty() || n_time == 0 {
        return Err(Error::param("Kushner–Dupuis grid invalid"));
    }
    let h = x_grid[1] - x_grid[0];
    let mut v: Vec<f64> = x_grid.iter().map(|&x| terminal(x)).collect();
    let mut values = vec![v.clone()];
    let mut policy = vec![vec![u_grid[0]; nx]; n_time];
    for t in (0..n_time).rev() {
        let mut vnew = v.clone();
        for i in 1..nx - 1 {
            let x = x_grid[i];
            let mut best = f64::INFINITY;
            let mut best_u = u_grid[0];
            for &u in u_grid {
                let b = drift(x, u);
                let s = vol(x, u);
                let den = s * s + b.abs() * h;
                if den <= 0.0 {
                    continue;
                }
                let dt = h * h / den;
                let pp = (0.5 * s * s + 0.5 * b * h).max(0.0) / den;
                let pm = (0.5 * s * s - 0.5 * b * h).max(0.0) / den;
                let p0 = (1.0 - pp - pm).max(0.0);
                let val = running(x, u) * dt + pp * v[i + 1] + pm * v[i - 1] + p0 * v[i];
                if val < best {
                    best = val;
                    best_u = u;
                }
            }
            vnew[i] = best;
            policy[t][i] = best_u;
        }
        vnew[0] = vnew[1];
        vnew[nx - 1] = vnew[nx - 2];
        v = vnew;
        values.push(v.clone());
    }
    values.reverse();
    Ok((values, policy))
}

/// Execution: TWAP is optimal for a risk-neutral linear-temporary-impact trader.
pub fn twap_control(inventory: f64, n: usize) -> Vec<f64> {
    twap(inventory, n)
}

/// Closed-form expected log-growth of a constant-mix strategy on GBM.
pub fn gbm_growth(model: &GeometricBrownianMotion, weight: f64, r: f64) -> f64 {
    let excess = model.mu - r;
    r + weight * excess - 0.5 * weight * weight * model.sigma * model.sigma
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merton_kelly_match() {
        let m = MertonPortfolio::new(0.1, 0.02, 0.2, 1.0).unwrap();
        assert!((m.myopic_weight() - kelly_fraction(0.1, 0.02, 0.2).unwrap()).abs() < 1e-14);
        assert!((m.myopic_weight() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn hjb_mean_reversion_control() {
        // cheap to stay near 0: u ≈ −x
        let xs: Vec<f64> = (-5..=5).map(|i| i as f64 * 0.2).collect();
        let us: Vec<f64> = (-5..=5).map(|i| i as f64 * 0.2).collect();
        let (v, pi) = hjb_1d(
            &xs,
            &us,
            15,
            0.05,
            |x, u| -x + u,
            |_x, _u| 0.2,
            |x, u| x * x + 0.2 * u * u,
            |x| x * x,
        )
        .unwrap();
        assert_eq!(v.len(), 16);
        let mid = xs.len() / 2 + 2;
        assert!(pi[0][mid].abs() < 1.0);
    }
}
