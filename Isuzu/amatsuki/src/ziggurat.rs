//! Marsaglia–Tsang (2000) Ziggurat for `N(0,1)`.
//!
//! [`crate::dist::StandardNormal`] is Box–Muller and spends two uniforms
//! (the sine sample is discarded). This sampler uses one uniform on the
//! typical rectangle and only falls back to a tail draw in the base layer.

use crate::dist::OpenClosed01;
use crate::rng::{Distribution, Rng};
use std::sync::OnceLock;

/// Published 128-box constants (Marsaglia–Tsang / Doornik).
const N: usize = 128;
const R: f64 = 3.442_619_855_899;
const V: f64 = 9.912_563_035_262_17e-3;

struct Tables {
    /// Right edge `x_i` of layer `i`, `x[N] = r`, `x[0] = 0`.
    x: [f64; N + 1],
}

fn pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp()
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut x = [0.0; N + 1];
        x[N] = R;
        for i in (1..N).rev() {
            x[i] = (-2.0 * (V / x[i + 1] + pdf(x[i + 1])).ln()).sqrt();
        }
        x[0] = 0.0;
        Tables { x }
    })
}

fn sample_tail<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    loop {
        let e1 = -rng.sample(OpenClosed01).ln();
        let e2 = -rng.sample(OpenClosed01).ln();
        let z = (R * R + 2.0 * e1).sqrt();
        if e2 > 0.5 * (z - R) * (z - R) {
            return z;
        }
    }
}

/// Standard normal via the 128-box Ziggurat.
#[derive(Clone, Copy, Debug, Default)]
pub struct StandardNormalZiggurat;

impl Distribution for StandardNormalZiggurat {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        sample_normal_ziggurat(rng)
    }
}

/// Draw `N(0,1)` with the Marsaglia–Tsang Ziggurat.
pub fn sample_normal_ziggurat<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let t = tables();
    loop {
        let bits = rng.next_u64();
        let i = (bits as usize) & (N - 1);
        let sign = if (bits >> 63) == 0 { 1.0 } else { -1.0 };
        // Uniform in (0, 1] from the remaining stream.
        let u = rng.sample(OpenClosed01);
        if i == 0 {
            // Base layer: rectangle [0, r] × [0, f(r)] plus the Gaussian tail.
            let u0 = rng.sample(OpenClosed01) * V;
            let tail = R * pdf(R);
            // V = r f(r) + ∫_r^∞ f; the rectangle occupies r f(r).
            if u0 > tail {
                return sign * sample_tail(rng);
            }
            return sign * rng.sample(OpenClosed01) * R;
        }
        let x = u * t.x[i];
        if x < t.x[i + 1] {
            return sign * x;
        }
        let y = rng.sample(OpenClosed01);
        let f_lo = pdf(t.x[i]);
        let f_hi = pdf(t.x[i - 1]);
        if f_hi <= f_lo {
            continue;
        }
        if y * (f_hi - f_lo) <= pdf(x) - f_lo {
            return sign * x;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chacha::seed_rng;

    #[test]
    fn ziggurat_moments() {
        let mut rng = seed_rng(99);
        let n = 80_000usize;
        let mut m = 0.0;
        let mut m2 = 0.0;
        for _ in 0..n {
            let z = sample_normal_ziggurat(&mut rng);
            m += z;
            m2 += z * z;
        }
        let mean = m / n as f64;
        let var = m2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }

    #[test]
    fn layer_edges_decrease() {
        let t = tables();
        assert!((t.x[N] - R).abs() < 1e-12);
        for i in 1..N {
            assert!(
                t.x[i] < t.x[i + 1],
                "x[{i}]={} !< x[{}]={}",
                t.x[i],
                i + 1,
                t.x[i + 1]
            );
        }
    }
}
