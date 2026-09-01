//! Lévy processes as time-indexed paths (gamma, IG, VG, NIG, stable).

use amatsuki::Rng;

use crate::error::Result;
use crate::noise::LevyMeasure;
use crate::path::Path;
use crate::sampling::Sampling;

/// Accumulate independent Lévy increments into a càdlàg path starting at `x0`.
pub fn levy_path<R: Rng + ?Sized>(
    measure: &LevyMeasure,
    sampling: &Sampling,
    x0: f64,
    rng: &mut R,
) -> Result<Path> {
    let n = sampling.n_nodes();
    let mut values = vec![0.0; n];
    values[0] = x0;
    let mut x = x0;
    for i in 0..sampling.n_steps() {
        x += measure.increment(sampling.delta(i), rng)?;
        values[i + 1] = x;
    }
    Path::new(sampling.times().to_vec(), values, 1)
}

/// Gamma process (subordinator) with increment `Gamma(shape_rate · Δt, scale)`.
///
/// Mean `shape_rate · scale · t`, variance `shape_rate · scale² · t`.
pub fn gamma_process<R: Rng + ?Sized>(
    sampling: &Sampling,
    shape_rate: f64,
    scale: f64,
    rng: &mut R,
) -> Result<Path> {
    levy_path(
        &LevyMeasure::Gamma {
            rate: shape_rate,
            scale,
        },
        sampling,
        0.0,
        rng,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::LevyMeasure;
    use crate::rng::seed_rng;
    use crate::sampling::Sampling;

    #[test]
    fn gaussian_levy_is_bm() {
        let s = Sampling::from_terminal(1.0, 500).unwrap();
        let mut rng = seed_rng(1);
        let p = levy_path(
            &LevyMeasure::Gaussian {
                mu: 0.0,
                sigma: 1.0,
            },
            &s,
            0.0,
            &mut rng,
        )
        .unwrap();
        let qv = p.quadratic_variation(0).unwrap();
        assert!((qv - 1.0).abs() < 0.25);
    }

    #[test]
    fn gamma_process_mean() {
        let s = Sampling::from_terminal(1.0, 200).unwrap();
        let p = gamma_process(&s, 2.0, 0.5, &mut seed_rng(2)).unwrap();
        assert!(p
            .as_univariate()
            .unwrap()
            .windows(2)
            .all(|w| w[1] + 1e-15 >= w[0]));
        // E[X_1] = rate * scale * t = 1. One Gamma(2, 1/2) draw has sd ≈ 0.71.
        let mut acc = 0.0;
        for seed in 1u64..21 {
            let q = gamma_process(&s, 2.0, 0.5, &mut seed_rng(seed)).unwrap();
            acc += q.terminal()[0];
        }
        let m = acc / 20.0;
        assert!((m - 1.0).abs() < 0.4, "gamma process mean {m}");
    }
}
