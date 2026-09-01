//! Path diagnostics: quadratic variation, first passage, Itô integral.

use crate::error::{Error, Result};
use crate::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossingDirection {
    Up,
    Down,
}

/// Realized quadratic variation of every coordinate: `Σ ‖ΔX‖²`.
pub fn quadratic_variation(path: &Path) -> Result<f64> {
    let mut s = 0.0;
    for j in 0..path.dim() {
        s += path.quadratic_variation(j)?;
    }
    Ok(s)
}

/// Realized covariation of two coordinates.
pub fn covariation(x: &Path, y: &Path) -> Result<f64> {
    if x.n_nodes() != y.n_nodes() {
        return Err(Error::dim("covariation needs a common grid"));
    }
    if x.dim() == 0 || y.dim() == 0 {
        return Err(Error::dim("empty path"));
    }
    let dx = x.increments(0)?;
    let dy = y.increments(0)?;
    Ok(dx.iter().zip(dy.iter()).map(|(a, b)| a * b).sum())
}

/// Running maximum of coordinate `j`.
pub fn running_maximum(path: &Path, j: usize) -> Result<Vec<f64>> {
    let xs = path.component(j)?;
    let mut out = Vec::with_capacity(xs.len());
    let mut m = f64::NEG_INFINITY;
    for x in xs {
        m = m.max(x);
        out.push(m);
    }
    Ok(out)
}

/// Running minimum of coordinate `j`.
pub fn running_minimum(path: &Path, j: usize) -> Result<Vec<f64>> {
    let xs = path.component(j)?;
    let mut out = Vec::with_capacity(xs.len());
    let mut m = f64::INFINITY;
    for x in xs {
        m = m.min(x);
        out.push(m);
    }
    Ok(out)
}

/// First time coordinate `j` crosses `barrier`.
pub fn first_passage_time(
    path: &Path,
    barrier: f64,
    direction: CrossingDirection,
    j: usize,
) -> Result<Option<f64>> {
    let xs = path.component(j)?;
    for (i, &x) in xs.iter().enumerate() {
        let hit = match direction {
            CrossingDirection::Up => x >= barrier,
            CrossingDirection::Down => x <= barrier,
        };
        if hit {
            return Ok(Some(path.times()[i]));
        }
    }
    Ok(None)
}

/// Discrete Itô integral `Σ H_{t_i} (W_{t_{i+1}} − W_{t_i})`.
pub fn ito_integral(integrand: &[f64], driver: &Path, j: usize) -> Result<f64> {
    let dw = driver.increments(j)?;
    if integrand.len() != dw.len() {
        return Err(Error::dim("Itô integral: integrand length != n_steps"));
    }
    Ok(integrand.iter().zip(dw.iter()).map(|(h, w)| h * w).sum())
}

/// Itô isometry check: `E[(∫ H dW)²]` vs `E[∫ H² dt]` on an ensemble.
pub fn ito_isometry_gap(integrals: &[f64], energy: &[f64]) -> Result<f64> {
    if integrals.len() != energy.len() || integrals.is_empty() {
        return Err(Error::dim("isometry samples must align"));
    }
    let n = integrals.len() as f64;
    let lhs = integrals.iter().map(|x| x * x).sum::<f64>() / n;
    let rhs = energy.iter().sum::<f64>() / n;
    Ok(lhs - rhs)
}

/// Brownian-bridge interpolation of a discrete-monitoring barrier hit
/// probability on one step (`P(hit | W_0=a, W_Δt=b)`).
pub fn brownian_bridge_hit_prob(a: f64, b: f64, barrier: f64, vol: f64, dt: f64) -> Result<f64> {
    if !(vol > 0.0 && dt > 0.0) {
        return Err(Error::param("bridge hit needs σ>0, Δt>0"));
    }
    if (a - barrier) * (b - barrier) <= 0.0 {
        return Ok(1.0);
    }
    // Reflection: exp(−2 (h−a)(h−b) / (σ² Δt))
    Ok((-2.0 * (barrier - a) * (barrier - b) / (vol * vol * dt)).exp())
}

/// Discrete-monitoring survival probability with a Brownian-bridge
/// correction: `∏_i (1 − p_hit(S_i, S_{i+1}))`.
pub fn barrier_survival_bb(
    path: &Path,
    asset: usize,
    barrier: f64,
    direction: CrossingDirection,
    vol: f64,
) -> Result<f64> {
    let xs = path.component(asset)?;
    let ts = path.times();
    let mut p = 1.0;
    for i in 0..xs.len().saturating_sub(1) {
        let dt = ts[i + 1] - ts[i];
        if dt <= 0.0 {
            continue;
        }
        let a = xs[i];
        let b = xs[i + 1];
        let crossed = match direction {
            CrossingDirection::Up => a >= barrier || b >= barrier,
            CrossingDirection::Down => a <= barrier || b <= barrier,
        };
        if crossed {
            return Ok(0.0);
        }
        let ph = brownian_bridge_hit_prob(a, b, barrier, vol, dt)?;
        p *= 1.0 - ph;
    }
    Ok(p.clamp(0.0, 1.0))
}

/// Exact Brownian-bridge step from `(t, x)` toward `(T, b)`.
pub fn brownian_bridge_step(
    x: f64,
    t: f64,
    t_end: f64,
    b: f64,
    sigma: f64,
    dt: f64,
    z: f64,
) -> f64 {
    let rem = (t_end - t).max(1e-16);
    let mean = x * (rem - dt).max(0.0) / rem + b * dt / rem;
    let var = sigma * sigma * dt * (rem - dt).max(0.0) / rem;
    mean + var.max(0.0).sqrt() * z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::Path;

    #[test]
    fn qv_and_bridge_hit() {
        let p = Path::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 1.0], 1).unwrap();
        assert!((quadratic_variation(&p).unwrap() - 1.0).abs() < 1e-14);
        assert_eq!(
            first_passage_time(&p, 1.0, CrossingDirection::Up, 0).unwrap(),
            Some(1.0)
        );
        let h = brownian_bridge_hit_prob(1.0, 1.1, 1.5, 1.0, 1.0).unwrap();
        assert!(h > 0.0 && h < 1.0);
        assert!((h - (-0.4_f64).exp()).abs() < 1e-14);
        let ito = ito_integral(&[1.0, 1.0], &p, 0).unwrap();
        assert!((ito - 1.0).abs() < 1e-14);
    }
}
