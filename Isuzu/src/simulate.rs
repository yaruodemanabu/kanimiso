//! Path simulation (`simulate` in YUIMA).

use amatsuki::{OpenClosed01, Rng, StandardNormal};

use crate::error::{Error, Result};
use crate::model::Sde;
use crate::models::cogarch::{cogarch_state_jump, Cogarch};
use crate::models::diffusion::{GeometricBrownianMotion, OrnsteinUhlenbeck};
use crate::noise::{brownian_increments, fractional_brownian_increments};
use crate::path::{Ensemble, Path};
use crate::sampling::Sampling;
use crate::scheme::{euler_step, kp15_scalar_step, milstein_step, Scheme};

/// Simulation options.
#[derive(Clone, Debug)]
pub struct SimConfig {
    pub scheme: Scheme,
    /// Reflect negative coordinates through zero (`x ← |x|`).
    ///
    /// Default is `false`: OU and other signed processes must be allowed
    /// to go negative. Enable for CIR / Heston variance Euler steps.
    pub reflect_nonnegative: bool,
    /// Optional precomputed Wiener increments (`n_steps × m`, row-major).
    pub increment_w: Option<Vec<f64>>,
    /// Optional precomputed Lévy increments (`n_steps`).
    pub increment_l: Option<Vec<f64>>,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            scheme: Scheme::EulerMaruyama,
            reflect_nonnegative: false,
            increment_w: None,
            increment_l: None,
        }
    }
}

/// Simulate one path of `model` on `sampling` starting at `x0`.
pub fn simulate<M, R>(
    model: &M,
    sampling: &Sampling,
    x0: &[f64],
    rng: &mut R,
    cfg: &SimConfig,
) -> Result<Path>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
{
    model.validate()?;
    let n = model.dim();
    let m = model.n_noise();
    if x0.len() != n {
        return Err(Error::dim(format!("x0 length {} != dim {n}", x0.len())));
    }
    if matches!(cfg.scheme, Scheme::Exact) {
        return simulate_exact(model, sampling, x0, rng, cfg);
    }
    if matches!(cfg.scheme, Scheme::KloedenPlaten15) && (n != 1 || m != 1) {
        return Err(Error::unsupported("KP1.5 is implemented for scalar SDEs"));
    }
    if matches!(cfg.scheme, Scheme::Milstein) && n > 1 && m > 1 {
        return Err(Error::unsupported(
            "Milstein is the commutative-noise formula; n>1 and m>1 needs a Lévy area",
        ));
    }

    let n_steps = sampling.n_steps();
    let dw = if let Some(w) = &cfg.increment_w {
        if w.len() != n_steps * m {
            return Err(Error::dim("increment_w has the wrong length"));
        }
        w.clone()
    } else if let Some(h) = model.hurst() {
        if (h - 0.5).abs() < 1e-15 {
            brownian_increments(sampling, m, rng)?
        } else {
            fractional_brownian_increments(sampling, h, rng)?
        }
    } else {
        brownian_increments(sampling, m, rng)?
    };

    let dl = if let Some(l) = &cfg.increment_l {
        if l.len() != n_steps {
            return Err(Error::dim("increment_l has the wrong length"));
        }
        Some(l.clone())
    } else if let Some(levy) = model.levy() {
        Some(levy.increments(sampling, rng)?)
    } else {
        None
    };

    let mut values = vec![0.0; sampling.n_nodes() * n];
    values[..n].copy_from_slice(x0);
    let mut x = x0.to_vec();
    let mut a = vec![0.0; n];
    let mut s = vec![0.0; n * m];
    let mut jac = vec![0.0; n * n * m];
    let mut g = vec![0.0; n];

    for i in 0..n_steps {
        let t = sampling.times()[i];
        let dt = sampling.delta(i);
        let dwi = &dw[i * m..(i + 1) * m];
        match cfg.scheme {
            Scheme::EulerMaruyama => euler_step(model, t, &mut x, dt, dwi, &mut a, &mut s),
            Scheme::Milstein => milstein_step(model, t, &mut x, dt, dwi, &mut a, &mut s, &mut jac),
            Scheme::KloedenPlaten15 => {
                let z: f64 = rng.sample(StandardNormal);
                kp15_scalar_step(model, t, &mut x, dt, dwi[0], z)?;
            }
            Scheme::Exact => unreachable!(),
        }
        if let Some(ref jumps) = dl {
            apply_jump(model, t, &mut x, jumps[i], &mut g);
        }
        for xi in &mut x {
            if !xi.is_finite() {
                return Err(Error::sim("non-finite state during simulation"));
            }
            if cfg.reflect_nonnegative && *xi < 0.0 {
                *xi = -*xi;
            }
        }
        values[(i + 1) * n..(i + 2) * n].copy_from_slice(&x);
    }
    Path::new(sampling.times().to_vec(), values, n)
}

fn apply_jump<M: Sde + ?Sized>(model: &M, t: f64, x: &mut [f64], dl: f64, g: &mut [f64]) {
    if !model.jump_coeff(t, x, g) {
        return;
    }
    if model.multiplicative_jumps() {
        // Merton/Kou: ΔX = X * (e^{ΔL} − 1) when γ = X and ΔL is the log-jump.
        for i in 0..x.len() {
            x[i] += g[i] * (dl.exp() - 1.0);
        }
    } else {
        for i in 0..x.len() {
            x[i] += g[i] * dl;
        }
    }
}

fn simulate_exact<M, R>(
    model: &M,
    sampling: &Sampling,
    x0: &[f64],
    rng: &mut R,
    cfg: &SimConfig,
) -> Result<Path>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
{
    let n = model.dim();
    let m = model.n_noise();
    let n_steps = sampling.n_steps();
    let mut probe = vec![0.0; n];
    let dw0 = vec![0.0; m];
    if !model.exact_step(sampling.times()[0], x0, sampling.delta(0), &dw0, &mut probe) {
        return Err(Error::unsupported(
            "Scheme::Exact requires Sde::exact_step (GBM / OU implement it)",
        ));
    }
    let dw = if let Some(w) = &cfg.increment_w {
        if w.len() != n_steps * m {
            return Err(Error::dim("increment_w has the wrong length"));
        }
        w.clone()
    } else if let Some(h) = model.hurst() {
        if (h - 0.5).abs() < 1e-15 {
            brownian_increments(sampling, m, rng)?
        } else {
            return Err(Error::unsupported("Scheme::Exact is for Brownian drivers"));
        }
    } else {
        brownian_increments(sampling, m, rng)?
    };
    let mut values = vec![0.0; sampling.n_nodes() * n];
    values[..n].copy_from_slice(x0);
    let mut x = x0.to_vec();
    let mut nxt = vec![0.0; n];
    for i in 0..n_steps {
        let t = sampling.times()[i];
        let dt = sampling.delta(i);
        let dwi = &dw[i * m..(i + 1) * m];
        if !model.exact_step(t, &x, dt, dwi, &mut nxt) {
            return Err(Error::unsupported("exact_step failed mid-path"));
        }
        x.copy_from_slice(&nxt);
        values[(i + 1) * n..(i + 2) * n].copy_from_slice(&x);
    }
    Path::new(sampling.times().to_vec(), values, n)
}

/// Exact GBM path (log-Euler is exact for GBM).
pub fn simulate_gbm_exact<R: Rng + ?Sized>(
    model: &GeometricBrownianMotion,
    sampling: &Sampling,
    x0: f64,
    rng: &mut R,
) -> Result<Path> {
    if x0 <= 0.0 {
        return Err(Error::param("GBM initial value must be positive"));
    }
    let n = sampling.n_nodes();
    let mut values = vec![0.0; n];
    values[0] = x0;
    let mut x = x0;
    for i in 0..sampling.n_steps() {
        let dt = sampling.delta(i);
        let z: f64 = rng.sample(StandardNormal);
        x *=
            ((model.mu - 0.5 * model.sigma * model.sigma) * dt + model.sigma * dt.sqrt() * z).exp();
        values[i + 1] = x;
    }
    Path::new(sampling.times().to_vec(), values, 1)
}

/// Exact OU / Vasicek path.
pub fn simulate_ou_exact<R: Rng + ?Sized>(
    model: &OrnsteinUhlenbeck,
    sampling: &Sampling,
    x0: f64,
    rng: &mut R,
) -> Result<Path> {
    let n = sampling.n_nodes();
    let mut values = vec![0.0; n];
    values[0] = x0;
    let mut x = x0;
    for i in 0..sampling.n_steps() {
        let z: f64 = rng.sample(StandardNormal);
        x = model.exact_step(x, sampling.delta(i), z);
        values[i + 1] = x;
    }
    Path::new(sampling.times().to_vec(), values, 1)
}

/// COGARCH path with the quadratic-variation state update.
pub fn simulate_cogarch<R: Rng + ?Sized>(
    model: &Cogarch,
    sampling: &Sampling,
    g0: f64,
    y0: &[f64],
    rng: &mut R,
) -> Result<Path> {
    if y0.len() != model.q {
        return Err(Error::dim("COGARCH y0 length must equal q"));
    }
    let n = 1 + model.q;
    let mut x = vec![0.0; n];
    x[0] = g0;
    x[1..].copy_from_slice(y0);
    let mut values = vec![0.0; sampling.n_nodes() * n];
    values[..n].copy_from_slice(&x);
    let mut a = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut g = vec![0.0; n];
    for i in 0..sampling.n_steps() {
        let t = sampling.times()[i];
        let dt = sampling.delta(i);
        let dl = model.levy.increment(dt, rng)?;
        euler_step(model, t, &mut x, dt, &[0.0], &mut a, &mut s);
        // ΔG += √V ΔL
        model.jump_coeff(t, &x, &mut g);
        x[0] += g[0] * dl;
        cogarch_state_jump(model, &mut x, dl);
        values[(i + 1) * n..(i + 2) * n].copy_from_slice(&x);
    }
    Path::new(sampling.times().to_vec(), values, n)
}

/// Simulate `nsim` independent paths.
pub fn simulate_n<M, R>(
    model: &M,
    sampling: &Sampling,
    x0: &[f64],
    nsim: usize,
    rng: &mut R,
    cfg: &SimConfig,
) -> Result<Ensemble>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
{
    if nsim == 0 {
        return Err(Error::sim("nsim must be positive"));
    }
    let mut paths = Vec::with_capacity(nsim);
    for _ in 0..nsim {
        paths.push(simulate(model, sampling, x0, rng, cfg)?);
    }
    Ensemble::new(paths)
}

/// Thin a regular path by independent Poisson observation (YUIMA
/// `poisson.random.sampling`). `rate[j]` is the keep-probability per original
/// node for series `j` (or a relative intensity in `(0, 1]`).
pub fn poisson_random_sampling<R: Rng + ?Sized>(
    path: &Path,
    rate: &[f64],
    rng: &mut R,
) -> Result<crate::path::AsyncData> {
    if rate.len() != path.dim() {
        return Err(Error::dim("rate length must equal path dim"));
    }
    let mut series = Vec::with_capacity(path.dim());
    for j in 0..path.dim() {
        if !(0.0..=1.0).contains(&rate[j]) {
            return Err(Error::param(
                "poisson.random.sampling rate must be in [0, 1]",
            ));
        }
        let mut times = Vec::new();
        let mut values = Vec::new();
        for i in 0..path.n_nodes() {
            let u: f64 = rng.sample(OpenClosed01);
            if u <= rate[j] {
                times.push(path.times()[i]);
                values.push(path.state(i)[j]);
            }
        }
        if times.len() < 2 {
            return Err(Error::sampling(
                "poisson sampling left fewer than two observations",
            ));
        }
        series.push(crate::path::TickSeries::new(times, values)?);
    }
    crate::path::AsyncData::new(series)
}

/// Monte Carlo estimate of `E[f(X_T)]` by simulating `nsim` paths.
pub fn expectation<M, R, F>(
    model: &M,
    sampling: &Sampling,
    x0: &[f64],
    nsim: usize,
    rng: &mut R,
    cfg: &SimConfig,
    f: F,
) -> Result<f64>
where
    M: Sde + ?Sized,
    R: Rng + ?Sized,
    F: Fn(&[f64]) -> f64,
{
    let ens = simulate_n(model, sampling, x0, nsim, rng, cfg)?;
    let mut s = 0.0;
    for p in &ens.paths {
        s += f(p.terminal());
    }
    Ok(s / nsim as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::GeometricBrownianMotion;
    use crate::rng::seed_rng;
    use crate::sampling::Sampling;

    #[test]
    fn gbm_exact_mean() {
        let m = GeometricBrownianMotion::new(0.1, 0.2).unwrap();
        let s = Sampling::from_terminal(1.0, 1).unwrap();
        let mut rng = seed_rng(42);
        let ens = {
            let mut paths = Vec::new();
            for _ in 0..4000 {
                paths.push(simulate_gbm_exact(&m, &s, 1.0, &mut rng).unwrap());
            }
            Ensemble::new(paths).unwrap()
        };
        let mean = ens.terminal_mean(0).unwrap();
        let expected = (0.1_f64).exp();
        assert!((mean - expected).abs() < 0.03);
    }

    #[test]
    fn euler_gbm_runs() {
        let m = GeometricBrownianMotion::new(0.05, 0.3).unwrap();
        let s = Sampling::from_terminal(1.0, 200).unwrap();
        let mut rng = seed_rng(1);
        let p = simulate(&m, &s, &[100.0], &mut rng, &SimConfig::default()).unwrap();
        assert_eq!(p.n_nodes(), 201);
        assert!(p.terminal()[0] > 0.0);
    }
}
