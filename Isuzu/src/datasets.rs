//! Toy datasets: simulate a known process, then recover its parameters.

use crate::error::Result;
use crate::model::Sde;
use crate::models::point::ExponentialHawkes;
use crate::models::point_more::CarmaHawkes;
use crate::models::{Cir, GeometricBrownianMotion, OrnsteinUhlenbeck};
use crate::path::Path;
use crate::rng::seed_rng;
use crate::sampling::Sampling;
use crate::simulate::{simulate, simulate_gbm_exact, simulate_ou_exact, SimConfig};

/// A labelled toy path plus the parameters that generated it.
#[derive(Clone, Debug)]
pub struct Toy<M> {
    pub model: M,
    pub path: Path,
    pub truth: Vec<f64>,
    pub names: Vec<&'static str>,
}

pub fn make_ou(
    kappa: f64,
    theta: f64,
    sigma: f64,
    t: f64,
    n: usize,
    seed: u64,
) -> Result<Toy<OrnsteinUhlenbeck>> {
    let model = OrnsteinUhlenbeck::new(kappa, theta, sigma)?;
    let samp = Sampling::from_terminal(t, n)?;
    let mut rng = seed_rng(seed);
    let path = simulate_ou_exact(&model, &samp, theta, &mut rng)?;
    Ok(Toy {
        truth: vec![kappa, theta, sigma],
        names: vec!["kappa", "theta", "sigma"],
        model,
        path,
    })
}

pub fn make_gbm(
    mu: f64,
    sigma: f64,
    t: f64,
    n: usize,
    seed: u64,
) -> Result<Toy<GeometricBrownianMotion>> {
    let model = GeometricBrownianMotion::new(mu, sigma)?;
    let samp = Sampling::from_terminal(t, n)?;
    let mut rng = seed_rng(seed);
    let path = simulate_gbm_exact(&model, &samp, 1.0, &mut rng)?;
    Ok(Toy {
        truth: vec![mu, sigma],
        names: vec!["mu", "sigma"],
        model,
        path,
    })
}

pub fn make_cir(
    kappa: f64,
    theta: f64,
    sigma: f64,
    t: f64,
    n: usize,
    seed: u64,
) -> Result<Toy<Cir>> {
    let model = Cir::new(kappa, theta, sigma)?;
    let samp = Sampling::from_terminal(t, n)?;
    let mut rng = seed_rng(seed);
    let path = simulate(
        &model,
        &samp,
        &[theta],
        &mut rng,
        &SimConfig {
            reflect_nonnegative: true,
            ..SimConfig::default()
        },
    )?;
    Ok(Toy {
        truth: vec![kappa, theta, sigma],
        names: vec!["kappa", "theta", "sigma"],
        model,
        path,
    })
}

pub fn make_hawkes(
    mu: f64,
    alpha: f64,
    beta: f64,
    t: f64,
    seed: u64,
) -> Result<(ExponentialHawkes, Vec<f64>)> {
    let model = ExponentialHawkes::new(mu, alpha, beta)?;
    let mut rng = seed_rng(seed);
    let arr = model.simulate(0.0, t, &mut rng)?;
    Ok((model, arr))
}

pub fn make_carma_hawkes(
    mu: f64,
    ar: Vec<f64>,
    ma: Vec<f64>,
    t: f64,
    seed: u64,
) -> Result<(CarmaHawkes, Vec<f64>)> {
    let model = CarmaHawkes::new(mu, ar, ma)?;
    let mut rng = seed_rng(seed);
    let arr = model.simulate(0.0, t, &mut rng)?;
    Ok((model, arr))
}

/// Three well-separated univariate Gaussians for DP-mixture recovery.
pub fn make_dp_gaussians(
    n_per: usize,
    means: &[f64],
    sigma: f64,
    seed: u64,
) -> Result<(Vec<f64>, Vec<usize>)> {
    use amatsuki::{Rng, StandardNormal};
    if means.is_empty() || n_per == 0 || !(sigma > 0.0) {
        return Err(crate::error::Error::param(
            "make_dp_gaussians needs nonempty means, n_per ≥ 1, σ > 0",
        ));
    }
    let mut rng = seed_rng(seed);
    let mut x = Vec::with_capacity(n_per * means.len());
    let mut z = Vec::with_capacity(n_per * means.len());
    for (k, &m) in means.iter().enumerate() {
        for _ in 0..n_per {
            x.push(m + sigma * rng.sample(StandardNormal));
            z.push(k);
        }
    }
    Ok((x, z))
}

/// Linear-Gaussian IBP toy: $X = ZA + E$ with a known binary $Z$.
pub fn make_ibp_linear_gaussian(
    n: usize,
    alpha: f64,
    sigma_x: f64,
    sigma_a: f64,
    d: usize,
    seed: u64,
) -> Result<(
    faer::Mat<f64>,
    crate::npbayes::FeatureMatrix,
    faer::Mat<f64>,
)> {
    use crate::npbayes::{sample_ibp_sequential, IbpParams};
    use amatsuki::{Rng, StandardNormal};
    if n == 0 || d == 0 || !(sigma_x > 0.0 && sigma_a > 0.0) {
        return Err(crate::error::Error::param(
            "make_ibp_linear_gaussian: bad dims",
        ));
    }
    let mut rng = seed_rng(seed);
    let z = sample_ibp_sequential(n, IbpParams::new(alpha)?, &mut rng)?;
    let k = z.k;
    let mut a = faer::Mat::<f64>::zeros(k, d);
    for r in 0..k {
        for c in 0..d {
            a[(r, c)] = sigma_a * rng.sample(StandardNormal);
        }
    }
    let mut x = faer::Mat::<f64>::zeros(n, d);
    for i in 0..n {
        for c in 0..d {
            let mut s = 0.0;
            for r in 0..k {
                s += z.get(i, r) as f64 * a[(r, c)];
            }
            x[(i, c)] = s + sigma_x * rng.sample(StandardNormal);
        }
    }
    Ok((x, z, a))
}

/// Generic Euler toy for any [`Sde`].
pub fn make_euler<M: Sde>(model: M, x0: &[f64], t: f64, n: usize, seed: u64) -> Result<Path> {
    let samp = Sampling::from_terminal(t, n)?;
    let mut rng = seed_rng(seed);
    simulate(&model, &samp, x0, &mut rng, &SimConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ou_toy_length() {
        let toy = make_ou(1.0, 0.0, 0.3, 2.0, 200, 1).unwrap();
        assert_eq!(toy.path.n_steps(), 200);
    }
}
