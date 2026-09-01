//! Non-Gaussian particle models: `ParticleModel`, PMMH, SMC², particle Gibbs.

use amatsuki::{OpenClosed01, Rng, StandardNormal};
use faer::{Col, Mat};

use crate::error::{Error, Result};
use crate::filter::particle::{resample_indices, ParticleConfig, ParticleFilter, ResamplingScheme};
use crate::filter::ssm::{
    ess, log_sum_exp, mvn_logpdf, normalize_log_weights, observation_cov_at, predict_obs,
    predict_state, process_cov_at, sample_mvn, slice_from_col, weighted_moments, DiscreteSsm,
};
use crate::linalg::col_from_slice;
use crate::path::Path;

/// Observation / transition densities that need not be Gaussian.
pub trait ParticleModel {
    fn state_dim(&self) -> usize;
    fn obs_dim(&self) -> usize;

    fn sample_prior<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>>;

    fn sample_transition<R: Rng + ?Sized>(
        &self,
        t: f64,
        dt: f64,
        x: &[f64],
        rng: &mut R,
    ) -> Result<Vec<f64>>;

    fn log_obs_density(&self, t: f64, x: &[f64], y: &[f64]) -> Result<f64>;

    /// Transition log-density. Required by PMMH / particle Gibbs.
    fn log_transition_density(&self, t: f64, dt: f64, x: &[f64], x_next: &[f64]) -> Result<f64>;
}

/// Wrap a [`DiscreteSsm`] as a Gaussian [`ParticleModel`].
pub struct GaussianParticle<'a, M: DiscreteSsm> {
    pub model: &'a M,
    pub x0: Col<f64>,
    pub p0: Mat<f64>,
}

impl<'a, M: DiscreteSsm> ParticleModel for GaussianParticle<'a, M> {
    fn state_dim(&self) -> usize {
        self.model.state_dim()
    }
    fn obs_dim(&self) -> usize {
        self.model.obs_dim()
    }
    fn sample_prior<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>> {
        Ok(slice_from_col(&sample_mvn(&self.x0, &self.p0, rng)?))
    }
    fn sample_transition<R: Rng + ?Sized>(
        &self,
        t: f64,
        dt: f64,
        x: &[f64],
        rng: &mut R,
    ) -> Result<Vec<f64>> {
        let xf = predict_state(self.model, t, dt, &col_from_slice(x));
        let q = process_cov_at(self.model, t, dt, x);
        Ok(slice_from_col(&sample_mvn(&xf, &q, rng)?))
    }
    fn log_obs_density(&self, t: f64, x: &[f64], y: &[f64]) -> Result<f64> {
        let yhat = predict_obs(self.model, t, &col_from_slice(x));
        let r = observation_cov_at(self.model, t);
        mvn_logpdf(&col_from_slice(y), &yhat, &r)
    }
    fn log_transition_density(&self, t: f64, dt: f64, x: &[f64], x_next: &[f64]) -> Result<f64> {
        let xf = predict_state(self.model, t, dt, &col_from_slice(x));
        let q = process_cov_at(self.model, t, dt, x);
        mvn_logpdf(&col_from_slice(x_next), &xf, &q)
    }
}

/// Left-censored (Tobit) Gaussian observation on the first coordinate.
#[derive(Clone, Debug)]
pub struct TobitObs {
    pub loc: f64,
    pub scale: f64,
    pub cutoff: f64,
}

impl TobitObs {
    pub fn new(loc: f64, scale: f64, cutoff: f64) -> Result<Self> {
        if !(scale > 0.0) {
            return Err(Error::param("Tobit scale must be positive"));
        }
        Ok(Self { loc, scale, cutoff })
    }

    pub fn log_density(&self, y: f64, mean: f64) -> f64 {
        let z = (y - mean - self.loc) / self.scale;
        if y > self.cutoff {
            -0.5 * z * z - self.scale.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
        } else {
            // Φ((c − μ)/σ)
            let c = (self.cutoff - mean - self.loc) / self.scale;
            crate::finance::special::norm_cdf(c).max(1e-300).ln()
        }
    }
}

/// Student-t observation (location-scale).
#[derive(Clone, Copy, Debug)]
pub struct StudentTObs {
    pub df: f64,
    pub scale: f64,
}

impl StudentTObs {
    pub fn new(df: f64, scale: f64) -> Result<Self> {
        if !(df > 0.0 && scale > 0.0) {
            return Err(Error::param("Student-t obs needs df>0, scale>0"));
        }
        Ok(Self { df, scale })
    }
    pub fn log_density(&self, y: f64, mean: f64) -> f64 {
        let z = (y - mean) / self.scale;
        let nu = self.df;
        // Student-t location-scale:
        // ln Γ((ν+1)/2) − ln Γ(ν/2) − ½ ln(νπ) − ln σ − (ν+1)/2 ln(1+z²/ν)
        -0.5 * (nu + 1.0) * (1.0 + z * z / nu).ln()
            - self.scale.ln()
            - 0.5 * (nu * std::f64::consts::PI).ln()
            - ln_gamma(0.5 * nu)
            + ln_gamma(0.5 * (nu + 1.0))
    }
}

fn ln_gamma(z: f64) -> f64 {
    // Lanczos (g=7) on the principal strip.
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma(1.0 - z);
    }
    const P: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = z - 1.0;
    let mut x = P[0];
    for (i, &p) in P.iter().enumerate().skip(1) {
        x += p / (z + i as f64);
    }
    let t = z + 7.5;
    (2.0 * std::f64::consts::PI).sqrt().ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Count observation `y ~ Poisson(exp(x))` or `Poisson(x⁺)`.
#[derive(Clone, Copy, Debug)]
pub struct PoissonObs {
    pub log_link: bool,
}

impl PoissonObs {
    pub fn log_density(&self, y: f64, x: f64) -> f64 {
        let lam = if self.log_link { x.exp() } else { x.max(0.0) };
        if y < 0.0 || !y.is_finite() {
            return f64::NEG_INFINITY;
        }
        if lam <= 0.0 {
            return if y == 0.0 { 0.0 } else { f64::NEG_INFINITY };
        }
        y * lam.ln() - lam - ln_fact(y as u64)
    }
}

fn ln_fact(n: u64) -> f64 {
    if n < 2 {
        0.0
    } else {
        (1..=n).map(|k| (k as f64).ln()).sum()
    }
}

fn log_uniform_weights(n: usize) -> Vec<f64> {
    vec![-(n as f64).ln(); n]
}

/// Bootstrap PF on a generic [`ParticleModel`].
pub fn particle_filter_model<M, R>(
    model: &M,
    observations: &Path,
    cfg: ParticleConfig,
    rng: &mut R,
) -> Result<ParticleFilter>
where
    M: ParticleModel,
    R: Rng + ?Sized,
{
    if cfg.n_particles == 0 {
        return Err(Error::param("need a positive number of particles"));
    }
    if observations.dim() != model.obs_dim() {
        return Err(Error::dim("observation dim != model obs dim"));
    }
    let mut particles: Vec<Col<f64>> = Vec::with_capacity(cfg.n_particles);
    for _ in 0..cfg.n_particles {
        particles.push(col_from_slice(&model.sample_prior(rng)?));
    }
    let mut logw = log_uniform_weights(cfg.n_particles);
    let (w0, _) = normalize_log_weights(&logw)?;
    let (m0, c0) = weighted_moments(&particles, &w0);
    let mut out = ParticleFilter {
        filtered: vec![m0],
        filtered_cov: vec![c0],
        loglik: 0.0,
        log_increments: Vec::new(),
        n_resamples: 0,
        ess: vec![cfg.n_particles as f64],
        particles: if cfg.store_particles {
            Some(vec![particles.clone()])
        } else {
            None
        },
        weights: if cfg.store_particles {
            Some(vec![w0])
        } else {
            None
        },
    };
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        let t1 = observations.times()[i + 1];
        let y = observations.state(i + 1);
        for p in &mut particles {
            let xs = slice_from_col(p);
            *p = col_from_slice(&model.sample_transition(t, dt, &xs, rng)?);
        }
        for (j, p) in particles.iter().enumerate() {
            logw[j] += model.log_obs_density(t1, &slice_from_col(p), y)?;
        }
        let (w, lse) = normalize_log_weights(&logw)?;
        out.loglik += lse;
        out.log_increments.push(lse);
        let e = ess(&w);
        out.ess.push(e);
        if cfg.ess_ratio > 0.0 && e < cfg.ess_ratio * cfg.n_particles as f64 {
            let idx = resample_indices(&w, cfg.resampling, rng);
            particles = idx.iter().map(|&k| particles[k].clone()).collect();
            logw = log_uniform_weights(cfg.n_particles);
            out.n_resamples += 1;
            let wequal = vec![1.0 / cfg.n_particles as f64; cfg.n_particles];
            let (mean, cov) = weighted_moments(&particles, &wequal);
            out.filtered.push(mean);
            out.filtered_cov.push(cov);
            if let Some(store) = out.particles.as_mut() {
                store.push(particles.clone());
            }
            if let Some(store) = out.weights.as_mut() {
                store.push(wequal);
            }
        } else {
            let (mean, cov) = weighted_moments(&particles, &w);
            out.filtered.push(mean);
            out.filtered_cov.push(cov);
            if let Some(store) = out.particles.as_mut() {
                store.push(particles.clone());
            }
            if let Some(store) = out.weights.as_mut() {
                store.push(w);
            }
            logw = {
                let (wn, _) = normalize_log_weights(&logw)?;
                wn.into_iter().map(|wi| wi.ln()).collect()
            };
        }
    }
    Ok(out)
}

/// Particle marginal Metropolis–Hastings (Andrieu–Doucet–Holenstein).
#[derive(Clone, Debug)]
pub struct PmmhFit {
    pub mean: Vec<f64>,
    pub map: Vec<f64>,
    pub samples: Vec<Vec<f64>>,
    pub accept_rate: f64,
    pub loglik_map: f64,
}

pub fn pmmh<F, M, R>(
    model_at: F,
    log_prior: impl Fn(&[f64]) -> f64,
    observations: &Path,
    start: &[f64],
    step_sd: &[f64],
    n_samples: usize,
    n_burn: usize,
    pf_cfg: ParticleConfig,
    rng: &mut R,
) -> Result<PmmhFit>
where
    F: Fn(&[f64]) -> Result<M>,
    M: ParticleModel,
    R: Rng + ?Sized,
{
    if start.len() != step_sd.len() || n_samples == 0 {
        return Err(Error::dim("PMMH start / step_sd / n_samples"));
    }
    let mut theta = start.to_vec();
    let mut ll = particle_filter_model(&model_at(&theta)?, observations, pf_cfg, rng)?.loglik
        + log_prior(&theta);
    let mut acc = 0usize;
    let mut samples = Vec::with_capacity(n_samples);
    let mut map = theta.clone();
    let mut ll_map = ll;
    let total = n_burn + n_samples;
    for it in 0..total {
        let mut prop = theta.clone();
        for i in 0..prop.len() {
            prop[i] += step_sd[i] * rng.sample(StandardNormal);
        }
        let lp = match model_at(&prop) {
            Ok(m) => match particle_filter_model(&m, observations, pf_cfg, rng) {
                Ok(pf) => pf.loglik + log_prior(&prop),
                Err(_) => f64::NEG_INFINITY,
            },
            Err(_) => f64::NEG_INFINITY,
        };
        let u: f64 = rng.sample(OpenClosed01);
        if lp.is_finite() && (lp - ll).exp() >= u {
            theta = prop;
            ll = lp;
            acc += 1;
        }
        if ll > ll_map {
            ll_map = ll;
            map = theta.clone();
        }
        if it >= n_burn {
            samples.push(theta.clone());
        }
    }
    let d = start.len();
    let mut mean = vec![0.0; d];
    for s in &samples {
        for i in 0..d {
            mean[i] += s[i];
        }
    }
    for m in &mut mean {
        *m /= samples.len() as f64;
    }
    Ok(PmmhFit {
        mean,
        map,
        samples,
        accept_rate: acc as f64 / total as f64,
        loglik_map: ll_map,
    })
}

/// Lightweight SMC²: outer particles on `θ`, each carrying an inner PF likelihood.
#[derive(Clone, Debug)]
pub struct Smc2Fit {
    pub theta: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
    pub loglik: f64,
}

pub fn smc2<F, M, R>(
    model_at: F,
    log_prior: impl Fn(&[f64]) -> f64,
    observations: &Path,
    prior_draw: impl Fn(&mut R) -> Result<Vec<f64>>,
    n_theta: usize,
    pf_cfg: ParticleConfig,
    rng: &mut R,
) -> Result<Smc2Fit>
where
    F: Fn(&[f64]) -> Result<M>,
    M: ParticleModel,
    R: Rng + ?Sized,
{
    if n_theta == 0 {
        return Err(Error::param("SMC² needs a positive number of θ-particles"));
    }
    let mut theta = Vec::with_capacity(n_theta);
    let mut logw = Vec::with_capacity(n_theta);
    for _ in 0..n_theta {
        let th = prior_draw(rng)?;
        let ll = match model_at(&th)
            .and_then(|m| particle_filter_model(&m, observations, pf_cfg, rng))
        {
            Ok(pf) => pf.loglik + log_prior(&th),
            Err(_) => f64::NEG_INFINITY,
        };
        theta.push(th);
        logw.push(ll);
    }
    let lse = log_sum_exp(&logw);
    let w: Vec<f64> = logw.iter().map(|&v| (v - lse).exp()).collect();
    Ok(Smc2Fit {
        theta,
        weights: w,
        loglik: lse - (n_theta as f64).ln(),
    })
}

/// Conditional SMC (particle Gibbs): one trajectory is forced through the cloud.
pub fn conditional_smc<M, R>(
    model: &M,
    observations: &Path,
    reference: &[Vec<f64>],
    cfg: ParticleConfig,
    rng: &mut R,
) -> Result<(ParticleFilter, Vec<Vec<f64>>)>
where
    M: ParticleModel,
    R: Rng + ?Sized,
{
    if reference.len() != observations.n_nodes() {
        return Err(Error::dim("reference trajectory length"));
    }
    let n = cfg.n_particles.max(2);
    let mut particles: Vec<Col<f64>> = Vec::with_capacity(n);
    particles.push(col_from_slice(&reference[0]));
    for _ in 1..n {
        particles.push(col_from_slice(&model.sample_prior(rng)?));
    }
    let mut logw = log_uniform_weights(n);
    let mut ancestors: Vec<Vec<usize>> = Vec::new();
    let mut history = vec![particles.clone()];
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        let t1 = observations.times()[i + 1];
        let y = observations.state(i + 1);
        let (w, _) = normalize_log_weights(&logw)?;
        let mut idx = resample_indices(&w, ResamplingScheme::Systematic, rng);
        idx[0] = 0; // keep the reference ancestor
        ancestors.push(idx.clone());
        let mut nxt = Vec::with_capacity(n);
        nxt.push(col_from_slice(&reference[i + 1]));
        for &a in idx.iter().skip(1) {
            let xs = slice_from_col(&particles[a]);
            nxt.push(col_from_slice(&model.sample_transition(t, dt, &xs, rng)?));
        }
        particles = nxt;
        logw = log_uniform_weights(n);
        for (j, p) in particles.iter().enumerate() {
            logw[j] += model.log_obs_density(t1, &slice_from_col(p), y)?;
        }
        history.push(particles.clone());
    }
    let (w, _) = normalize_log_weights(&logw)?;
    // Trace a new trajectory (not the frozen one) by ancestor sampling.
    let mut k = {
        let u: f64 = rng.sample(OpenClosed01);
        let mut c = 0.0;
        let mut pick = n - 1;
        for (i, &wi) in w.iter().enumerate() {
            c += wi;
            if u <= c {
                pick = i;
                break;
            }
        }
        if pick == 0 && n > 1 {
            1
        } else {
            pick
        }
    };
    let mut traj = vec![slice_from_col(&history[history.len() - 1][k])];
    for step in (0..ancestors.len()).rev() {
        k = ancestors[step][k];
        traj.push(slice_from_col(&history[step][k]));
    }
    traj.reverse();
    let pf = particle_filter_model(model, observations, cfg, rng)?;
    Ok((pf, traj))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::ssm::LinearGaussian;
    use crate::linalg::mat_from_row_slice;
    use crate::path::Path;
    use crate::rng::seed_rng;

    #[test]
    fn tobit_and_pmmh_smoke() {
        let tob = TobitObs::new(0.0, 1.0, 0.0).unwrap();
        // Censoring mass Φ((c−μ)/σ) shrinks as the latent mean moves above c.
        assert!(tob.log_density(-1.0, 2.0) < tob.log_density(-1.0, 0.0));
        assert!(tob.log_density(0.5, 0.5) > tob.log_density(0.5, 3.0));
        let model = LinearGaussian::new(
            mat_from_row_slice(1, 1, &[0.8]),
            mat_from_row_slice(1, 1, &[0.04]),
            mat_from_row_slice(1, 1, &[1.0]),
            mat_from_row_slice(1, 1, &[0.09]),
        )
        .unwrap();
        let times: Vec<f64> = (0..=12).map(|i| i as f64).collect();
        let mut vals = vec![0.0; 13];
        let mut rng = seed_rng(3);
        let mut x = 0.0;
        for i in 1..=12 {
            x = 0.8 * x + 0.2 * rng.sample(StandardNormal);
            vals[i] = x + 0.3 * rng.sample(StandardNormal);
        }
        let obs = Path::new(times, vals, 1).unwrap();
        let wrap = GaussianParticle {
            model: &model,
            x0: col_from_slice(&[0.0]),
            p0: mat_from_row_slice(1, 1, &[1.0]),
        };
        let pf = particle_filter_model(
            &wrap,
            &obs,
            ParticleConfig {
                n_particles: 80,
                ..ParticleConfig::default()
            },
            &mut rng,
        )
        .unwrap();
        assert!(pf.loglik.is_finite());
        let t = StudentTObs::new(5.0, 1.0).unwrap();
        assert!(t.log_density(0.0, 0.0).is_finite());
    }
}
