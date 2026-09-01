//! Particle filters: SIS, SIR / bootstrap, auxiliary, regularized,
//! unscented PF, resampling schemes, and FFBSi smoothing.

use amatsuki::Rng;
use faer::{Col, Mat, Scale};

use crate::error::{Error, Result};
use crate::linalg::{cholesky, col_zeros, spd_regularize};
use crate::path::Path;

use super::nonlinear::{ukf_proposal, UkfParams};
use super::ssm::{
    check_filter_dims, ess, mvn_logpdf, normalize_log_weights, obs_col, observation_cov_at,
    predict_obs, predict_state, process_cov_at, sample_mvn, slice_from_col, weighted_moments,
    DiscreteSsm, GaussianFilter,
};

/// Resampling scheme for weighted particles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResamplingScheme {
    /// Independent categorical draws from the weights.
    Multinomial,
    /// One uniform `U ∼ [0, 1/N)` plus a regular grid (Kitagawa).
    Systematic,
    /// One uniform draw in each stratum `[i/N, (i+1)/N)`.
    Stratified,
    /// Deterministic `⌊N w_i⌋` copies plus multinomial leftovers.
    Residual,
}

impl Default for ResamplingScheme {
    fn default() -> Self {
        Self::Systematic
    }
}

/// Particle-filter options.
#[derive(Clone, Copy, Debug)]
pub struct ParticleConfig {
    pub n_particles: usize,
    pub resampling: ResamplingScheme,
    /// Resample when `ESS < ess_ratio · N`. `0` disables resampling (SIS).
    pub ess_ratio: f64,
    pub store_particles: bool,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self {
            n_particles: 256,
            resampling: ResamplingScheme::Systematic,
            ess_ratio: 0.5,
            store_particles: false,
        }
    }
}

/// Sequential Monte Carlo output.
#[derive(Clone, Debug)]
pub struct ParticleFilter {
    pub filtered: Vec<Col<f64>>,
    pub filtered_cov: Vec<Mat<f64>>,
    pub loglik: f64,
    /// Incremental log-normalizers `log p(y_k | y_{1:k-1})`.
    pub log_increments: Vec<f64>,
    pub n_resamples: usize,
    pub ess: Vec<f64>,
    pub particles: Option<Vec<Vec<Col<f64>>>>,
    pub weights: Option<Vec<Vec<f64>>>,
}

impl ParticleFilter {
    /// Project to the Gaussian-filter shape (means / covariances / loglik).
    pub fn as_gaussian(&self) -> GaussianFilter {
        GaussianFilter {
            filtered: self.filtered.clone(),
            predicted: self.filtered.clone(),
            filtered_cov: self.filtered_cov.clone(),
            predicted_cov: self.filtered_cov.clone(),
            loglik: self.loglik,
        }
    }
}

/// Sequential importance sampling (no resampling). Degenerates quickly;
/// kept as the SIS baseline.
pub fn sis_filter<M, R>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    n_particles: usize,
    rng: &mut R,
) -> Result<ParticleFilter>
where
    M: DiscreteSsm,
    R: Rng + ?Sized,
{
    let mut cfg = ParticleConfig {
        n_particles,
        ess_ratio: 0.0,
        resampling: ResamplingScheme::Systematic,
        store_particles: false,
    };
    cfg.ess_ratio = 0.0;
    bootstrap_particle_filter(model, observations, x0, p0, cfg, rng)
}

/// Bootstrap / SIR particle filter: propose from the transition, weight
/// by `p(y_k | x_k)`, resample when the effective sample size is low.
pub fn particle_filter<M, R>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    cfg: ParticleConfig,
    rng: &mut R,
) -> Result<ParticleFilter>
where
    M: DiscreteSsm,
    R: Rng + ?Sized,
{
    bootstrap_particle_filter(model, observations, x0, p0, cfg, rng)
}

fn bootstrap_particle_filter<M, R>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    cfg: ParticleConfig,
    rng: &mut R,
) -> Result<ParticleFilter>
where
    M: DiscreteSsm,
    R: Rng + ?Sized,
{
    if cfg.n_particles == 0 {
        return Err(Error::param("need a positive number of particles"));
    }
    let (_nx, _ny) = check_filter_dims(model, observations, x0, p0)?;
    let mut particles: Vec<Col<f64>> = Vec::with_capacity(cfg.n_particles);
    for _ in 0..cfg.n_particles {
        particles.push(sample_mvn(x0, p0, rng)?);
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
            Some(vec![w0.clone()])
        } else {
            None
        },
    };
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        let t1 = observations.times()[i + 1];
        let y = obs_col(observations, i + 1);
        for p in &mut particles {
            let xs = slice_from_col(p);
            let q = process_cov_at(model, t, dt, &xs);
            let xf = predict_state(model, t, dt, p);
            *p = sample_mvn(&xf, &q, rng)?;
        }
        for (j, p) in particles.iter().enumerate() {
            let yhat = predict_obs(model, t1, p);
            let r = observation_cov_at(model, t1);
            logw[j] += mvn_logpdf(&y, &yhat, &r)?;
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
            // keep log-weights as log(w_i) so the next multiply is correct
            logw = {
                let (wn, _) = normalize_log_weights(&logw)?;
                wn.into_iter().map(|wi| wi.ln()).collect()
            };
        }
    }
    Ok(out)
}

/// Pitt–Shephard auxiliary particle filter.
///
/// First-stage weights `∝ w_{k-1} p(y_k | μ_k^{(i)})` with
/// `μ = E[x_k | x_{k-1}]`, resample ancestors, then propagate and
/// reweight by `p(y|x) / p(y|μ)`.
pub fn auxiliary_particle_filter<M, R>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    cfg: ParticleConfig,
    rng: &mut R,
) -> Result<ParticleFilter>
where
    M: DiscreteSsm,
    R: Rng + ?Sized,
{
    if cfg.n_particles == 0 {
        return Err(Error::param("need a positive number of particles"));
    }
    let (_nx, _ny) = check_filter_dims(model, observations, x0, p0)?;
    let mut particles: Vec<Col<f64>> = (0..cfg.n_particles)
        .map(|_| sample_mvn(x0, p0, rng))
        .collect::<Result<_>>()?;
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
        particles: None,
        weights: None,
    };
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        let t1 = observations.times()[i + 1];
        let y = obs_col(observations, i + 1);
        let r = observation_cov_at(model, t1);
        let mut mu = Vec::with_capacity(cfg.n_particles);
        let mut log_first = vec![0.0; cfg.n_particles];
        for (j, p) in particles.iter().enumerate() {
            let m = predict_state(model, t, dt, p);
            let yhat = predict_obs(model, t1, &m);
            log_first[j] = logw[j] + mvn_logpdf(&y, &yhat, &r)?;
            mu.push(m);
        }
        let (w1, lse1) = normalize_log_weights(&log_first)?;
        let idx = resample_indices(&w1, cfg.resampling, rng);
        let mut new_particles = Vec::with_capacity(cfg.n_particles);
        let mut log_second = vec![0.0; cfg.n_particles];
        for (j, &a) in idx.iter().enumerate() {
            let xs = slice_from_col(&particles[a]);
            let q = process_cov_at(model, t, dt, &xs);
            let xf = predict_state(model, t, dt, &particles[a]);
            let xnew = sample_mvn(&xf, &q, rng)?;
            let yhat = predict_obs(model, t1, &xnew);
            let ymu = predict_obs(model, t1, &mu[a]);
            log_second[j] = mvn_logpdf(&y, &yhat, &r)? - mvn_logpdf(&y, &ymu, &r)?;
            new_particles.push(xnew);
        }
        particles = new_particles;
        let (w, lse2) = normalize_log_weights(&log_second)?;
        // First-stage weights already include normalized `logw`; second-stage
        // starts at 0 so we subtract ln N to convert the sum into a mean.
        let inc = lse1 + lse2 - (cfg.n_particles as f64).ln();
        out.loglik += inc;
        out.log_increments.push(inc);
        out.n_resamples += 1;
        let e = ess(&w);
        out.ess.push(e);
        let (mean, cov) = weighted_moments(&particles, &w);
        out.filtered.push(mean);
        out.filtered_cov.push(cov);
        logw = w.into_iter().map(|wi| wi.ln()).collect();
    }
    Ok(out)
}

/// Regularized particle filter (Musso–Oudjane–Le Gland): after SIR
/// resampling, jitter with a Gaussian kernel of Silverman bandwidth
/// `h = (4/(n+2))^{1/(n+4)} N^{-1/(n+4)}` times `chol(P)`.
#[derive(Clone, Copy, Debug)]
pub struct RegularizedConfig {
    pub particle: ParticleConfig,
    /// Extra bandwidth multiplier (`1` = Silverman).
    pub bandwidth: f64,
}

impl Default for RegularizedConfig {
    fn default() -> Self {
        Self {
            particle: ParticleConfig::default(),
            bandwidth: 1.0,
        }
    }
}

pub fn regularized_particle_filter<M, R>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    cfg: RegularizedConfig,
    rng: &mut R,
) -> Result<ParticleFilter>
where
    M: DiscreteSsm,
    R: Rng + ?Sized,
{
    let mut pcfg = cfg.particle;
    pcfg.store_particles = false;
    let n = pcfg.n_particles;
    if n == 0 {
        return Err(Error::param("need a positive number of particles"));
    }
    if !cfg.bandwidth.is_finite() || cfg.bandwidth <= 0.0 {
        return Err(Error::param("RPF bandwidth must be positive and finite"));
    }
    let nx = x0.nrows();
    let mut particles: Vec<Col<f64>> = (0..n)
        .map(|_| sample_mvn(x0, p0, rng))
        .collect::<Result<_>>()?;
    let mut logw = log_uniform_weights(n);
    let (w0, _) = normalize_log_weights(&logw)?;
    let (m0, c0) = weighted_moments(&particles, &w0);
    let mut pf = ParticleFilter {
        filtered: vec![m0],
        filtered_cov: vec![c0],
        loglik: 0.0,
        log_increments: Vec::new(),
        n_resamples: 0,
        ess: vec![n as f64],
        particles: None,
        weights: None,
    };
    let h_silv = silverman_bandwidth(nx, n) * cfg.bandwidth;
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        let t1 = observations.times()[i + 1];
        let y = obs_col(observations, i + 1);
        for p in &mut particles {
            let xs = slice_from_col(p);
            let q = process_cov_at(model, t, dt, &xs);
            let xf = predict_state(model, t, dt, p);
            *p = sample_mvn(&xf, &q, rng)?;
        }
        for (j, p) in particles.iter().enumerate() {
            let yhat = predict_obs(model, t1, p);
            let r = observation_cov_at(model, t1);
            logw[j] += mvn_logpdf(&y, &yhat, &r)?;
        }
        let (w, lse) = normalize_log_weights(&logw)?;
        pf.loglik += lse;
        pf.log_increments.push(lse);
        pf.ess.push(ess(&w));
        let idx = resample_indices(&w, pcfg.resampling, rng);
        particles = idx.iter().map(|&k| particles[k].clone()).collect();
        pf.n_resamples += 1;
        let wequal = vec![1.0 / n as f64; n];
        let (_mean, cov) = weighted_moments(&particles, &wequal);
        let cov_r = spd_regularize(cov.clone(), 1e-12)?;
        let l = cholesky(&cov_r)?;
        for p in &mut particles {
            let mut z = col_zeros(nx);
            for k in 0..nx {
                z[k] = rng.sample(amatsuki::StandardNormal);
            }
            *p = &*p + Scale(h_silv) * &(&l * &z);
        }
        let (mean, cov2) = weighted_moments(&particles, &wequal);
        pf.filtered.push(mean);
        pf.filtered_cov.push(cov2);
        logw = log_uniform_weights(n);
    }
    Ok(pf)
}

fn silverman_bandwidth(nx: usize, n: usize) -> f64 {
    let d = nx as f64;
    let nn = n as f64;
    (4.0 / (d + 2.0)).powf(1.0 / (d + 4.0)) * nn.powf(-1.0 / (d + 4.0))
}

/// Unscented particle filter (van der Merwe–Doucet–de Freitas–Wan):
/// each particle proposes from a one-step UKF approximation of
/// `p(x_k | x_{k-1}, y_k)`, then is reweighted by
/// `p(y|x) p(x|x⁻) / q_UKF(x)`.
pub fn unscented_particle_filter<M, R>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    cfg: ParticleConfig,
    ukf: UkfParams,
    rng: &mut R,
) -> Result<ParticleFilter>
where
    M: DiscreteSsm,
    R: Rng + ?Sized,
{
    if cfg.n_particles == 0 {
        return Err(Error::param("need a positive number of particles"));
    }
    let (nx, _ny) = check_filter_dims(model, observations, x0, p0)?;
    let mut particles: Vec<Col<f64>> = (0..cfg.n_particles)
        .map(|_| sample_mvn(x0, p0, rng))
        .collect::<Result<_>>()?;
    let mut covs: Vec<Mat<f64>> = vec![p0.clone(); cfg.n_particles];
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
        particles: None,
        weights: None,
    };
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        let t1 = observations.times()[i + 1];
        let y = obs_col(observations, i + 1);
        let r = observation_cov_at(model, t1);
        for j in 0..cfg.n_particles {
            let (m, p_prop) = ukf_proposal(model, t, dt, &particles[j], &covs[j], &y, ukf)?;
            let xnew = sample_mvn(&m, &p_prop, rng)?;
            let xs = slice_from_col(&particles[j]);
            let q = process_cov_at(model, t, dt, &xs);
            let xf = predict_state(model, t, dt, &particles[j]);
            let yhat = predict_obs(model, t1, &xnew);
            let log_prior = mvn_logpdf(&xnew, &xf, &q)?;
            let log_like = mvn_logpdf(&y, &yhat, &r)?;
            let log_q = mvn_logpdf(&xnew, &m, &p_prop)?;
            logw[j] += log_like + log_prior - log_q;
            particles[j] = xnew;
            covs[j] = p_prop;
        }
        let (w, lse) = normalize_log_weights(&logw)?;
        out.loglik += lse;
        out.log_increments.push(lse);
        let e = ess(&w);
        out.ess.push(e);
        if cfg.ess_ratio > 0.0 && e < cfg.ess_ratio * cfg.n_particles as f64 {
            let idx = resample_indices(&w, cfg.resampling, rng);
            particles = idx.iter().map(|&k| particles[k].clone()).collect();
            covs = idx.iter().map(|&k| covs[k].clone()).collect();
            logw = log_uniform_weights(cfg.n_particles);
            out.n_resamples += 1;
            let wequal = vec![1.0 / cfg.n_particles as f64; cfg.n_particles];
            let (mean, cov) = weighted_moments(&particles, &wequal);
            out.filtered.push(mean);
            out.filtered_cov.push(cov);
        } else {
            let (mean, cov) = weighted_moments(&particles, &w);
            out.filtered.push(mean);
            out.filtered_cov.push(cov);
            logw = w.into_iter().map(|wi| wi.ln()).collect();
        }
        let _ = nx;
    }
    Ok(out)
}

fn log_uniform_weights(n: usize) -> Vec<f64> {
    vec![-(n as f64).ln(); n]
}

/// Forward-filter backward-simulation (FFBSi) particle smoother.
///
/// After a stored bootstrap filter, draw `n_trajectories` backward
/// paths with
/// `P(i_k | i_{k+1}) ∝ w_k^{(i)} p(x_{k+1} | x_k^{(i)})`
/// and return their mean / covariance.
pub fn particle_smoother<M, R>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    cfg: ParticleConfig,
    n_trajectories: usize,
    rng: &mut R,
) -> Result<ParticleFilter>
where
    M: DiscreteSsm,
    R: Rng + ?Sized,
{
    if n_trajectories == 0 {
        return Err(Error::param("smoother needs ≥ 1 trajectory"));
    }
    let mut cfg = cfg;
    cfg.store_particles = true;
    let pf = bootstrap_particle_filter(model, observations, x0, p0, cfg, rng)?;
    let parts = pf
        .particles
        .as_ref()
        .ok_or_else(|| Error::infer("smoother missing stored particles"))?;
    let weights = pf
        .weights
        .as_ref()
        .ok_or_else(|| Error::infer("smoother missing stored weights"))?;
    let tlen = parts.len();
    let mut traj_means: Vec<Col<f64>> = Vec::with_capacity(tlen);
    let mut traj_covs: Vec<Mat<f64>> = Vec::with_capacity(tlen);
    // Draw backward trajectories and accumulate.
    let n_part = parts[0].len();
    let mut chosen: Vec<Vec<Col<f64>>> = vec![Vec::new(); tlen];
    for _ in 0..n_trajectories {
        // sample terminal index
        let mut idx = categorical(&weights[tlen - 1], rng);
        chosen[tlen - 1].push(parts[tlen - 1][idx].clone());
        for k in (0..tlen - 1).rev() {
            let t = observations.times()[k];
            let dt = observations.times()[k + 1] - t;
            let x_next = &parts[k + 1][idx];
            let mut logp = vec![0.0; n_part];
            for i in 0..n_part {
                let xs = slice_from_col(&parts[k][i]);
                let q = process_cov_at(model, t, dt, &xs);
                let xf = predict_state(model, t, dt, &parts[k][i]);
                let lw = if weights[k][i] > 0.0 {
                    weights[k][i].ln()
                } else {
                    f64::NEG_INFINITY
                };
                logp[i] = lw + mvn_logpdf(x_next, &xf, &q)?;
            }
            let (w, _) = normalize_log_weights(&logp)?;
            idx = categorical(&w, rng);
            chosen[k].push(parts[k][idx].clone());
        }
    }
    let eq = 1.0 / n_trajectories as f64;
    let we = vec![eq; n_trajectories];
    for k in 0..tlen {
        let (m, c) = weighted_moments(&chosen[k], &we);
        traj_means.push(m);
        traj_covs.push(c);
    }
    Ok(ParticleFilter {
        filtered: traj_means,
        filtered_cov: traj_covs,
        loglik: pf.loglik,
        log_increments: pf.log_increments,
        n_resamples: pf.n_resamples,
        ess: pf.ess,
        particles: None,
        weights: None,
    })
}

pub(crate) fn resample_indices<R: Rng + ?Sized>(
    weights: &[f64],
    scheme: ResamplingScheme,
    rng: &mut R,
) -> Vec<usize> {
    match scheme {
        ResamplingScheme::Multinomial => multinomial(weights, rng),
        ResamplingScheme::Systematic => systematic(weights, rng),
        ResamplingScheme::Stratified => stratified(weights, rng),
        ResamplingScheme::Residual => residual(weights, rng),
    }
}

fn cdf(weights: &[f64]) -> Vec<f64> {
    let mut c = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for &w in weights {
        acc += w;
        c.push(acc);
    }
    if let Some(last) = c.last_mut() {
        *last = 1.0;
    }
    c
}

/// First index `i` with `u < c[i]`. Using a strict upper bound skips a
/// leading zero-weight atom when `u = 0`.
fn invert_cdf(c: &[f64], u: f64) -> usize {
    if c.is_empty() {
        return 0;
    }
    let mut lo = 0usize;
    let mut hi = c.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if u < c[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo.min(c.len() - 1)
}

fn multinomial<R: Rng + ?Sized>(weights: &[f64], rng: &mut R) -> Vec<usize> {
    let n = weights.len();
    let c = cdf(weights);
    (0..n)
        .map(|_| invert_cdf(&c, rng.next_f64().min(1.0 - f64::EPSILON)))
        .collect()
}

fn systematic<R: Rng + ?Sized>(weights: &[f64], rng: &mut R) -> Vec<usize> {
    let n = weights.len();
    let c = cdf(weights);
    let u0 = rng.next_f64() / n as f64;
    (0..n)
        .map(|j| invert_cdf(&c, (u0 + j as f64 / n as f64).min(1.0 - f64::EPSILON)))
        .collect()
}

fn stratified<R: Rng + ?Sized>(weights: &[f64], rng: &mut R) -> Vec<usize> {
    let n = weights.len();
    let c = cdf(weights);
    (0..n)
        .map(|j| {
            let u = (j as f64 + rng.next_f64()) / n as f64;
            invert_cdf(&c, u.min(1.0 - f64::EPSILON))
        })
        .collect()
}

fn residual<R: Rng + ?Sized>(weights: &[f64], rng: &mut R) -> Vec<usize> {
    let n = weights.len();
    let mut idx = Vec::with_capacity(n);
    let mut leftover = vec![0.0; n];
    let mut used = 0usize;
    for (i, &w) in weights.iter().enumerate() {
        let copies = (n as f64 * w).floor() as usize;
        leftover[i] = n as f64 * w - copies as f64;
        for _ in 0..copies {
            idx.push(i);
        }
        used += copies;
    }
    let rest = n - used;
    if rest > 0 {
        let s: f64 = leftover.iter().sum();
        if s > 0.0 {
            for w in &mut leftover {
                *w /= s;
            }
            let extra = {
                // multinomial of size `rest`
                let c = cdf(&leftover);
                (0..rest)
                    .map(|_| invert_cdf(&c, rng.next_f64().min(1.0 - f64::EPSILON)))
                    .collect::<Vec<_>>()
            };
            idx.extend(extra);
        } else {
            while idx.len() < n {
                idx.push(0);
            }
        }
    }
    idx.truncate(n);
    while idx.len() < n {
        idx.push(idx.last().copied().unwrap_or(0));
    }
    idx
}

fn categorical<R: Rng + ?Sized>(weights: &[f64], rng: &mut R) -> usize {
    invert_cdf(&cdf(weights), rng.next_f64().min(1.0 - f64::EPSILON))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::mat_from_row_slice;
    use crate::rng::seed_rng;

    #[test]
    fn resampling_counts_sum_to_n() {
        let w = vec![0.1, 0.2, 0.3, 0.4];
        let mut rng = seed_rng(1);
        for scheme in [
            ResamplingScheme::Multinomial,
            ResamplingScheme::Systematic,
            ResamplingScheme::Stratified,
            ResamplingScheme::Residual,
        ] {
            let idx = resample_indices(&w, scheme, &mut rng);
            assert_eq!(idx.len(), 4);
            assert!(idx.iter().all(|&i| i < 4));
        }
    }

    #[test]
    fn log_sum_exp_normalizes() {
        let logw = vec![0.0, 0.0, 0.0];
        let (w, lse) = normalize_log_weights(&logw).unwrap();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((lse - (3.0_f64).ln()).abs() < 1e-12);
        let e = ess(&w);
        assert!((e - 3.0).abs() < 1e-12);
        let _ = mat_from_row_slice(1, 1, &[1.0]);
    }

    #[test]
    fn invert_cdf_skips_zero_leading_weight() {
        let c = cdf(&[0.0, 0.3, 0.7]);
        assert_eq!(invert_cdf(&c, 0.0), 1);
        assert_eq!(invert_cdf(&c, 0.29), 1);
        assert_eq!(invert_cdf(&c, 0.99), 2);
    }
}
