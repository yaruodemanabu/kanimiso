//! Quasi-Monte Carlo: Sobol', Brownian-bridge construction, antithetic pairing.

use crate::error::{Error, Result};
use crate::finance::special::norm_inv;
use crate::path::Path;
use crate::sampling::Sampling;

/// Sobol' generator in the unit cube (Joe–Kuo direction numbers, dim ≤ 16).
#[derive(Clone, Debug)]
pub struct Sobol {
    dim: usize,
    index: u32,
    direction: Vec<Vec<u32>>,
    x: Vec<u32>,
}

impl Sobol {
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 || dim > DIRECTIONS.len() {
            return Err(Error::param(format!(
                "Sobol dimension must be in 1..={}",
                DIRECTIONS.len()
            )));
        }
        let mut direction = vec![vec![0u32; 32]; dim];
        // Dimension 0: van der Corput in base 2.
        for k in 0..32 {
            direction[0][k] = 1u32 << (31 - k);
        }
        for d in 1..dim {
            let (deg, poly, m) = DIRECTIONS[d];
            for (k, &mk) in m.iter().enumerate() {
                direction[d][k] = mk << (31 - k);
            }
            for k in deg..32 {
                let mut v = direction[d][k - deg] >> deg;
                let mut p = poly;
                for j in 1..deg {
                    if (p & 1) == 1 {
                        v ^= direction[d][k - j];
                    }
                    p >>= 1;
                }
                v ^= direction[d][k - deg];
                direction[d][k] = v;
            }
        }
        Ok(Self {
            dim,
            index: 0,
            direction,
            x: vec![0; dim],
        })
    }

    /// Next point in `[0, 1)^d` (Gray-code counter).
    pub fn next_point(&mut self) -> Vec<f64> {
        if self.index == 0 {
            self.index = 1;
            return vec![0.0; self.dim];
        }
        let c = self.index.trailing_zeros() as usize;
        for d in 0..self.dim {
            self.x[d] ^= self.direction[d][c];
        }
        self.index = self.index.wrapping_add(1);
        self.x
            .iter()
            .map(|&v| (v as f64) * (0.5f64.powi(32)))
            .collect()
    }
}

/// Joe–Kuo: (degree, primitive polynomial, initial m_i). Dimension 0 unused.
const DIRECTIONS: &[(usize, u32, &[u32])] = &[
    (0, 0, &[]),
    (1, 0, &[1]),
    (2, 1, &[1, 3]),
    (3, 1, &[1, 3, 1]),
    (3, 2, &[1, 1, 1]),
    (4, 1, &[1, 1, 3, 3]),
    (4, 4, &[1, 3, 5, 13]),
    (5, 2, &[1, 1, 3, 5, 17]),
    (5, 4, &[1, 1, 5, 5, 5]),
    (5, 7, &[1, 1, 5, 5, 17]),
    (5, 11, &[1, 1, 7, 11, 19]),
    (5, 13, &[1, 1, 5, 1, 1]),
    (5, 14, &[1, 1, 1, 3, 11]),
    (6, 1, &[1, 3, 5, 5, 31, 15]),
    (6, 13, &[1, 3, 3, 9, 7, 49]),
    (6, 16, &[1, 1, 3, 13, 3, 35]),
];

/// Map a Sobol point through `Φ⁻¹` to a standard normal vector.
pub fn sobol_normals(s: &mut Sobol) -> Vec<f64> {
    s.next_point()
        .into_iter()
        .map(|u| norm_inv(u.clamp(1e-12, 1.0 - 1e-12)))
        .collect()
}

/// Brownian-bridge construction of a Wiener path from `n` normals.
///
/// The first normal is `W_T`, then midpoints are filled recursively.
pub fn brownian_bridge_path(normals: &[f64], times: &[f64]) -> Result<Path> {
    if times.len() < 2 || normals.len() + 1 != times.len() {
        return Err(Error::dim("bridge: n_normals + 1 = n_nodes"));
    }
    let n = times.len() - 1;
    let mut w = vec![0.0; n + 1];
    w[n] = normals[0] * (times[n] - times[0]).sqrt();
    fill_bridge(&mut w, times, normals, 1, 0, n);
    Path::new(times.to_vec(), w, 1)
}

fn fill_bridge(w: &mut [f64], times: &[f64], z: &[f64], mut k: usize, left: usize, right: usize) {
    if right - left <= 1 || k >= z.len() {
        return;
    }
    let mid = (left + right) / 2;
    let t0 = times[left];
    let t1 = times[right];
    let tm = times[mid];
    let var = (tm - t0) * (t1 - tm) / (t1 - t0);
    w[mid] = ((t1 - tm) * w[left] + (tm - t0) * w[right]) / (t1 - t0) + var.max(0.0).sqrt() * z[k];
    k += 1;
    fill_bridge(w, times, z, k, left, mid);
    let used = mid - left;
    fill_bridge(w, times, z, k + used.saturating_sub(1), mid, right);
}

/// Pair `z` with `−z` (antithetic normals).
pub fn antithetic(z: &[f64]) -> Vec<f64> {
    z.iter().map(|x| -x).collect()
}

/// Build a sampling grid and a Sobol Brownian path.
pub fn sobol_brownian(sampling: &Sampling, stream: &mut Sobol) -> Result<Path> {
    let n = sampling.n_steps();
    let mut z = Vec::with_capacity(n);
    // Consume enough 1-D points. A 1-D Sobol is a van der Corput sequence.
    let mut s1 = Sobol::new(1)?;
    s1.index = stream.index;
    for _ in 0..n {
        z.extend(sobol_normals(&mut s1));
    }
    stream.index = s1.index;
    brownian_bridge_path(&z, sampling.times())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sobol_first_points_in_unit_cube() {
        let mut s = Sobol::new(2).unwrap();
        for _ in 0..32 {
            let p = s.next_point();
            assert_eq!(p.len(), 2);
            assert!(p.iter().all(|u| (0.0..1.0).contains(u)));
        }
        let samp = Sampling::from_terminal(1.0, 8).unwrap();
        let mut s = Sobol::new(1).unwrap();
        let path = sobol_brownian(&samp, &mut s).unwrap();
        assert_eq!(path.n_steps(), 8);
        assert!((path.state(0)[0]).abs() < 1e-14);
    }
}
