//! Statistical inference: QMLE, LSE, information criteria, change-point,
//! LASSO, adaptive Bayes (YUIMA `qmle`, `lse`, `CPoint`, `lasso`, `adaBayes`).

use amatsuki::{OpenClosed01, Rng, StandardNormal};
use faer::{Col, Mat, Scale};

use crate::error::{Error, Result};
use crate::linalg::{gram_rowmajor, logdet_spd, solve_spd, spd_regularize};
use crate::model::{ParametricSde, Sde};
use crate::optimize::{nelder_mead, OptOptions, OptResult};
use crate::path::Path;

const LN_2PI: f64 = 1.8378770664093453; // ln(2π)

/// Euler Gaussian quasi-log-likelihood of a frozen model on a path.
///
/// Restricted to Brownian Markov diffusions: jump measures and
/// `H ≠ 1/2` are rejected (those parameters do not enter this contrast).
/// A non-SPD instantaneous covariance yields `−∞` rather than εI jitter.
pub fn quasi_loglik_frozen<M: Sde + ?Sized>(model: &M, path: &Path) -> Result<f64> {
    if path.dim() != model.dim() {
        return Err(Error::dim("path dim != model dim"));
    }
    if model.levy().is_some() {
        return Err(Error::infer(
            "Euler QMLE is for Brownian Markov diffusions; jump models need a jump likelihood",
        ));
    }
    if let Some(h) = model.hurst() {
        if (h - 0.5).abs() > 1e-14 {
            return Err(Error::infer(
                "Euler QMLE ignores Hurst; use a fractional covariance likelihood",
            ));
        }
    }
    let n = model.dim();
    let m = model.n_noise();
    let mut a = vec![0.0; n];
    let mut s = vec![0.0; n * m];
    let mut ql = 0.0;
    for i in 0..path.n_steps() {
        let t = path.times()[i];
        let dt = path.times()[i + 1] - path.times()[i];
        if dt <= 0.0 {
            return Err(Error::sampling("non-positive time step"));
        }
        let x = path.state(i);
        model.drift(t, x, &mut a);
        model.diffusion(t, x, &mut s);
        // Scalar path: one multiply, no Mat / Cholesky.
        if n == 1 {
            let mut var = 0.0;
            for j in 0..m {
                var += s[j] * s[j];
            }
            var *= dt;
            if !(var > 0.0 && var.is_finite()) {
                return Ok(f64::NEG_INFINITY);
            }
            let innov = path.state(i + 1)[0] - x[0] - a[0] * dt;
            ql += -0.5 * (LN_2PI + var.ln() + innov * innov / var);
            continue;
        }
        let mut sigma = gram_rowmajor(&s, n, m);
        sigma = Scale(dt) * &sigma;
        let mut innov = crate::linalg::col_zeros(n);
        for k in 0..n {
            innov[k] = path.state(i + 1)[k] - x[k] - a[k] * dt;
        }
        // One Cholesky yields both log-det and the quadratic form.
        let l = match crate::linalg::cholesky(&sigma) {
            Ok(v) => v,
            Err(_) => return Ok(f64::NEG_INFINITY),
        };
        let mut ld = 0.0;
        for k in 0..n {
            let d = l[(k, k)];
            if !(d > 0.0) {
                return Ok(f64::NEG_INFINITY);
            }
            ld += d.ln();
        }
        ld *= 2.0;
        let y = match crate::linalg::solve_lower(&l, &innov) {
            Ok(v) => v,
            Err(_) => return Ok(f64::NEG_INFINITY),
        };
        let quad = crate::linalg::dot(&y, &y);
        ql += -0.5 * (n as f64 * LN_2PI + ld + quad);
    }
    Ok(ql)
}

/// Quasi-log-likelihood of a parametric family at `params`.
pub fn quasi_loglik<F: ParametricSde>(family: &F, path: &Path, params: &[f64]) -> Result<f64> {
    let frozen = family.with_params(params)?.freeze()?;
    quasi_loglik_frozen(&frozen, path)
}

/// Euler residuals: `(ΔX − a Δt) / √(diag(σσᵀ Δt))` (univariate) or
/// Mahalanobis innovations (multivariate). Useful for residual plots.
pub fn euler_residuals<M: Sde + ?Sized>(model: &M, path: &Path) -> Result<Vec<Col<f64>>> {
    if path.dim() != model.dim() {
        return Err(Error::dim("path dim != model dim"));
    }
    let n = model.dim();
    let m = model.n_noise();
    let mut a = vec![0.0; n];
    let mut s = vec![0.0; n * m];
    let mut out = Vec::with_capacity(path.n_steps());
    for i in 0..path.n_steps() {
        let t = path.times()[i];
        let dt = path.times()[i + 1] - path.times()[i];
        let x = path.state(i);
        model.drift(t, x, &mut a);
        model.diffusion(t, x, &mut s);
        let mut sigma = gram_rowmajor(&s, n, m);
        sigma = Scale(dt) * &sigma;
        sigma = spd_regularize(sigma, 1e-12 * dt)?;
        let mut innov = crate::linalg::col_zeros(n);
        for k in 0..n {
            innov[k] = path.state(i + 1)[k] - x[k] - a[k] * dt;
        }
        let sol = solve_spd(&sigma, &innov)?;
        // Whitened residual R such that Rᵀ R = innovᵀ Σ⁻¹ innov.
        match crate::linalg::cholesky(&sigma) {
            Ok(l) => out.push(crate::linalg::solve_lower(&l, &innov)?),
            Err(_) => out.push(sol),
        }
    }
    Ok(out)
}

/// Least-squares contrast for the drift (YUIMA `lse`).
///
/// `Σ_i ‖ΔX_i − a(t_i, X_i) Δt_i‖² / Δt_i`
pub fn lse_contrast_frozen<M: Sde + ?Sized>(model: &M, path: &Path) -> Result<f64> {
    if path.dim() != model.dim() {
        return Err(Error::dim("path dim != model dim"));
    }
    let n = model.dim();
    let mut a = vec![0.0; n];
    let mut s = 0.0;
    for i in 0..path.n_steps() {
        let t = path.times()[i];
        let dt = path.times()[i + 1] - path.times()[i];
        let x = path.state(i);
        model.drift(t, x, &mut a);
        for k in 0..n {
            let r = path.state(i + 1)[k] - x[k] - a[k] * dt;
            s += r * r / dt;
        }
    }
    Ok(s)
}

/// Fit result shared by QMLE / LSE / LASSO.
#[derive(Clone, Debug)]
pub struct Fit {
    pub params: Vec<f64>,
    pub names: Vec<String>,
    pub quasi_loglik: f64,
    pub n_obs: usize,
    pub converged: bool,
    pub iters: usize,
    pub contrast: f64,
    /// Hessian of the quasi-log-likelihood at `params`, if computed.
    pub hessian: Option<Mat<f64>>,
}

impl Fit {
    pub fn aic(&self) -> f64 {
        -2.0 * self.quasi_loglik + 2.0 * self.params.len() as f64
    }
    pub fn bic(&self) -> f64 {
        -2.0 * self.quasi_loglik + (self.params.len() as f64) * (self.n_obs as f64).ln()
    }
    /// BIC that uses the number of increments (`n_obs − 1`), not QBIC.
    pub fn bic_increments(&self) -> f64 {
        -2.0 * self.quasi_loglik + (self.params.len() as f64) * ((self.n_obs - 1) as f64).ln()
    }
    /// Eguchi–Masuda QBIC: `−2 QL + log det(−H)` when a Hessian is stored.
    ///
    /// Falls back to [`Self::bic_increments`] if the Hessian is missing or
    /// not negative definite.
    pub fn qbic(&self) -> f64 {
        if let Some(h) = &self.hessian {
            let negh = Scale(-1.0) * h;
            if let Ok(ld) = logdet_spd(&negh) {
                return -2.0 * self.quasi_loglik + ld;
            }
        }
        self.bic_increments()
    }
    pub fn coef(&self, name: &str) -> Result<f64> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.params[i])
            .ok_or_else(|| Error::infer(format!("unknown parameter {name}")))
    }
}

fn names_of<F: ParametricSde>(f: &F) -> Vec<String> {
    f.param_names().iter().map(|s| (*s).to_string()).collect()
}

/// Quasi-maximum likelihood (`qmle`).
pub fn qmle<F: ParametricSde>(
    family: &F,
    path: &Path,
    start: &[f64],
    lower: Option<&[f64]>,
    upper: Option<&[f64]>,
    opt: OptOptions,
) -> Result<Fit> {
    let obj = |p: &[f64]| match quasi_loglik(family, path, p) {
        Ok(ll) if ll.is_finite() => -ll,
        _ => 1e16,
    };
    let r: OptResult = nelder_mead(&obj, start, lower, upper, opt)?;
    let ql = quasi_loglik(family, path, &r.x)?;
    let hessian = fd_hessian(family, path, &r.x, 1e-4).ok();
    Ok(Fit {
        params: r.x,
        names: names_of(family),
        quasi_loglik: ql,
        n_obs: path.n_nodes(),
        converged: r.converged,
        iters: r.iters,
        contrast: r.f,
        hessian,
    })
}

/// Least-squares estimator of the drift (`lse`).
pub fn lse<F: ParametricSde>(
    family: &F,
    path: &Path,
    start: &[f64],
    lower: Option<&[f64]>,
    upper: Option<&[f64]>,
    opt: OptOptions,
) -> Result<Fit> {
    let obj = |p: &[f64]| match family
        .with_params(p)
        .and_then(|g| g.freeze())
        .and_then(|m| lse_contrast_frozen(&m, path))
    {
        Ok(c) if c.is_finite() => c,
        _ => 1e16,
    };
    let r = nelder_mead(&obj, start, lower, upper, opt)?;
    let ql = quasi_loglik(family, path, &r.x).unwrap_or(f64::NAN);
    Ok(Fit {
        params: r.x,
        names: names_of(family),
        quasi_loglik: ql,
        n_obs: path.n_nodes(),
        converged: r.converged,
        iters: r.iters,
        contrast: r.f,
        hessian: None,
    })
}

/// L1-penalized QMLE (`lasso` in YUIMA).
pub fn lasso_qmle<F: ParametricSde>(
    family: &F,
    path: &Path,
    start: &[f64],
    lambda: f64,
    lower: Option<&[f64]>,
    upper: Option<&[f64]>,
    opt: OptOptions,
) -> Result<Fit> {
    if lambda < 0.0 {
        return Err(Error::param("lasso λ must be non-negative"));
    }
    let obj = |p: &[f64]| match quasi_loglik(family, path, p) {
        Ok(ll) if ll.is_finite() => -ll + lambda * p.iter().map(|x| x.abs()).sum::<f64>(),
        _ => 1e16,
    };
    let r = nelder_mead(&obj, start, lower, upper, opt)?;
    let ql = quasi_loglik(family, path, &r.x)?;
    Ok(Fit {
        params: r.x,
        names: names_of(family),
        quasi_loglik: ql,
        n_obs: path.n_nodes(),
        converged: r.converged,
        iters: r.iters,
        contrast: r.f,
        hessian: None,
    })
}

/// Information criteria for a fitted model (YUIMA `IC`).
#[derive(Clone, Debug)]
pub struct InformationCriteria {
    pub aic: f64,
    pub bic: f64,
    pub qbic: f64,
    pub k: usize,
    pub n: usize,
    pub quasi_loglik: f64,
}

impl InformationCriteria {
    pub fn from_fit(fit: &Fit) -> Self {
        Self {
            aic: fit.aic(),
            bic: fit.bic(),
            qbic: fit.qbic(),
            k: fit.params.len(),
            n: fit.n_obs,
            quasi_loglik: fit.quasi_loglik,
        }
    }
}

/// Select the candidate with smallest AIC.
pub fn select_aic(fits: &[Fit]) -> Result<usize> {
    fits.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.aic()
                .partial_cmp(&b.aic())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .ok_or_else(|| Error::infer("no candidate models"))
}

/// Non-parametric volatility change-point: maximize the Gaussian QV contrast.
///
/// For a univariate path, if `σ` jumps at `τ`, the contrast
/// `n₁ log(QV₁/n₁) + n₂ log(QV₂/n₂)` is minimized (equivalently the two-regime
/// quasi-likelihood is maximized).
pub fn change_point_qv(path: &Path, component: usize) -> Result<ChangePoint> {
    if path.n_steps() < 8 {
        return Err(Error::infer("need more increments for a change-point"));
    }
    let dx = path.increments(component)?;
    let n = dx.len();
    let mut prefix = vec![0.0; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + dx[i] * dx[i];
    }
    let qv = prefix[n];
    let mut best_k = n / 2;
    let mut best = f64::INFINITY;
    for k in 2..n - 2 {
        let q1 = prefix[k];
        let q2 = qv - q1;
        if q1 <= 0.0 || q2 <= 0.0 {
            continue;
        }
        let c = k as f64 * (q1 / k as f64).ln() + (n - k) as f64 * (q2 / (n - k) as f64).ln();
        if c < best {
            best = c;
            best_k = k;
        }
    }
    Ok(ChangePoint {
        index: best_k,
        time: path.times()[best_k],
        contrast: best,
        qv_left: prefix[best_k],
        qv_right: qv - prefix[best_k],
    })
}

/// Parametric CPoint: scan grid times and maximize `QL(θ₁; left) + QL(θ₂; right)`.
pub fn change_point_qmle<F: ParametricSde>(
    family: &F,
    path: &Path,
    param_left: &[f64],
    param_right: &[f64],
) -> Result<ChangePoint> {
    let m1 = family.with_params(param_left)?.freeze()?;
    let m2 = family.with_params(param_right)?.freeze()?;
    let n = path.n_nodes();
    if n < 10 {
        return Err(Error::infer("path too short for CPoint"));
    }
    let mut best_i = n / 2;
    let mut best = f64::NEG_INFINITY;
    // Evaluate on a stride to keep this O(n²) feasible; refine locally.
    let stride = ((n / 80).max(1)).min(8);
    let mut candidates = Vec::new();
    let mut i = 4;
    while i + 4 < n {
        candidates.push(i);
        i += stride;
    }
    for i in candidates {
        let left = path.window(path.times()[0], path.times()[i])?;
        let right = path.window(path.times()[i], path.times()[n - 1])?;
        if left.n_steps() < 2 || right.n_steps() < 2 {
            continue;
        }
        let ql = match (
            quasi_loglik_frozen(&m1, &left),
            quasi_loglik_frozen(&m2, &right),
        ) {
            (Ok(a), Ok(b)) => a + b,
            _ => continue,
        };
        if ql > best {
            best = ql;
            best_i = i;
        }
    }
    let left = path.window(path.times()[0], path.times()[best_i])?;
    Ok(ChangePoint {
        index: best_i,
        time: path.times()[best_i],
        contrast: best,
        qv_left: left.quadratic_variation(0).unwrap_or(f64::NAN),
        qv_right: f64::NAN,
    })
}

#[derive(Clone, Debug)]
pub struct ChangePoint {
    pub index: usize,
    pub time: f64,
    pub contrast: f64,
    pub qv_left: f64,
    pub qv_right: f64,
}

/// Random-walk Metropolis–Hastings on the quasi-posterior
/// `π(θ | data) ∝ exp(QL(θ)) N(θ; μ₀, diag(σ)²)`.
///
/// This is **not** Uchida–Yoshida adaptive Bayes: the proposal width is
/// fixed (`step_sd`) and does not adapt. `start` is the chain initial
/// value; `prior_mean` is the Gaussian prior mean (defaults to `start`
/// only when omitted — pass it explicitly so moving `start` does not
/// change the target).
#[derive(Clone, Debug)]
pub struct BayesFit {
    pub mean: Vec<f64>,
    pub map: Vec<f64>,
    pub samples: Vec<Vec<f64>>,
    pub accept_rate: f64,
}

pub fn adaptive_bayes<F, R>(
    family: &F,
    path: &Path,
    start: &[f64],
    prior_sd: &[f64],
    n_samples: usize,
    n_burn: usize,
    step_sd: &[f64],
    prior_mean: Option<&[f64]>,
    rng: &mut R,
) -> Result<BayesFit>
where
    F: ParametricSde,
    R: Rng + ?Sized,
{
    if start.len() != prior_sd.len() || start.len() != step_sd.len() {
        return Err(Error::dim("Bayes start / prior_sd / step_sd length"));
    }
    if let Some(mu) = prior_mean {
        if mu.len() != start.len() {
            return Err(Error::dim("prior_mean length must match start"));
        }
    }
    if n_samples == 0 {
        return Err(Error::infer("n_samples must be positive"));
    }
    let mu0: Vec<f64> = prior_mean.unwrap_or(start).to_vec();
    let log_post = |p: &[f64]| -> f64 {
        let ql = match quasi_loglik(family, path, p) {
            Ok(v) if v.is_finite() => v,
            _ => return f64::NEG_INFINITY,
        };
        let mut lp = ql;
        for i in 0..p.len() {
            let z = p[i] - mu0[i];
            lp += -0.5 * (z / prior_sd[i]).powi(2);
        }
        lp
    };
    let mut x = start.to_vec();
    let mut lx = log_post(&x);
    let mut acc = 0usize;
    let mut samples = Vec::with_capacity(n_samples);
    let total = n_burn + n_samples;
    for it in 0..total {
        let mut y = x.clone();
        for i in 0..y.len() {
            let z: f64 = rng.sample(StandardNormal);
            y[i] += step_sd[i] * z;
        }
        let ly = log_post(&y);
        let u: f64 = rng.sample(OpenClosed01);
        if ly.is_finite() && (ly - lx).exp() >= u {
            x = y;
            lx = ly;
            acc += 1;
        }
        if it >= n_burn {
            samples.push(x.clone());
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
    let map = samples
        .iter()
        .max_by(|a, b| {
            log_post(a)
                .partial_cmp(&log_post(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
        .unwrap_or_else(|| start.to_vec());
    Ok(BayesFit {
        mean,
        map,
        samples,
        accept_rate: acc as f64 / total as f64,
    })
}

/// Wald-type test `H₀: θ_i = θ0` using a finite-difference Hessian of QL.
pub fn wald_test<F: ParametricSde>(
    family: &F,
    path: &Path,
    fit: &Fit,
    index: usize,
    theta0: f64,
) -> Result<Wald> {
    if index >= fit.params.len() {
        return Err(Error::infer("parameter index out of range"));
    }
    let h = fd_hessian(family, path, &fit.params, 1e-4)?;
    // Covariance ≈ (−H)⁻¹
    let negh = Scale(-1.0) * &h;
    let cov = crate::linalg::try_inverse(&negh).ok_or_else(|| {
        Error::numeric("Hessian is not negative definite; Wald variance unavailable")
    })?;
    let var = cov[(index, index)].max(0.0);
    let se = var.sqrt();
    let z = if se > 0.0 {
        (fit.params[index] - theta0) / se
    } else {
        f64::NAN
    };
    Ok(Wald {
        estimate: fit.params[index],
        se,
        z,
        pvalue: 2.0 * std_normal_sf(z.abs()),
    })
}

#[derive(Clone, Debug)]
pub struct Wald {
    pub estimate: f64,
    pub se: f64,
    pub z: f64,
    pub pvalue: f64,
}

fn std_normal_sf(z: f64) -> f64 {
    // erfc(z / √2) / 2
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

fn erfc(x: f64) -> f64 {
    // Abramowitz–Stegun 7.1.26
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let a = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = a * (-z * z).exp();
    if x >= 0.0 {
        y
    } else {
        2.0 - y
    }
}

fn fd_hessian<F: ParametricSde>(family: &F, path: &Path, p: &[f64], eps: f64) -> Result<Mat<f64>> {
    let k = p.len();
    let mut h = crate::linalg::mat_zeros(k, k);
    let f0 = quasi_loglik(family, path, p)?;
    for i in 0..k {
        let mut pp = p.to_vec();
        let mut pm = p.to_vec();
        pp[i] += eps;
        pm[i] -= eps;
        let fp = quasi_loglik(family, path, &pp)?;
        let fm = quasi_loglik(family, path, &pm)?;
        h[(i, i)] = (fp - 2.0 * f0 + fm) / (eps * eps);
        for j in (i + 1)..k {
            let mut pp2 = p.to_vec();
            let mut pm2 = p.to_vec();
            let mut pmp = p.to_vec();
            let mut ppm = p.to_vec();
            pp2[i] += eps;
            pp2[j] += eps;
            pm2[i] -= eps;
            pm2[j] -= eps;
            pmp[i] -= eps;
            pmp[j] += eps;
            ppm[i] += eps;
            ppm[j] -= eps;
            let v = (quasi_loglik(family, path, &pp2)?
                - quasi_loglik(family, path, &ppm)?
                - quasi_loglik(family, path, &pmp)?
                + quasi_loglik(family, path, &pm2)?)
                / (4.0 * eps * eps);
            h[(i, j)] = v;
            h[(j, i)] = v;
        }
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::OrnsteinUhlenbeck;
    use crate::optimize::OptOptions;
    use crate::rng::seed_rng;
    use crate::sampling::Sampling;
    use crate::simulate::{simulate_ou_exact, SimConfig};

    #[test]
    fn ou_qmle_recovers_sigma() {
        let true_m = OrnsteinUhlenbeck::new(1.2, 0.0, 0.4).unwrap();
        let samp = Sampling::from_terminal(20.0, 4000).unwrap();
        let mut rng = seed_rng(9);
        let path = simulate_ou_exact(&true_m, &samp, 0.0, &mut rng).unwrap();
        let start = [0.8, 0.1, 0.7];
        let fit = qmle(
            &true_m,
            &path,
            &start,
            Some(&[0.05, -2.0, 0.05]),
            Some(&[4.0, 2.0, 2.0]),
            OptOptions {
                max_iter: 250,
                ..OptOptions::default()
            },
        )
        .unwrap();
        assert!((fit.params[2] - 0.4).abs() < 0.08, "sigma {:?}", fit.params);
        let _ = SimConfig::default();
    }
}
