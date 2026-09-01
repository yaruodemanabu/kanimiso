//! Nonlinear Kalman family: EKF, IEKF, second-order EKF, UKF, CKF,
//! EnKF, Gaussian-sum, continuous–discrete EKF, and extended RTS.

use amatsuki::Rng;
use faer::{Col, Mat, Scale};

use crate::error::{Error, Result};
use crate::linalg::{cholesky, col_zeros, mat_identity, mat_zeros, spd_regularize, try_inverse};
use crate::model::Sde;
use crate::path::Path;

use super::ssm::{
    add_scaled_outer, check_filter_dims, empty_gaussian, innovation_update, obs_col,
    observation_cov_at, observe_matrix, predict_obs, predict_state, process_cov_at, sample_mvn,
    slice_from_col, symmetrize, transition_matrix, weighted_moments, DiscreteSsm, GaussianFilter,
    GaussianSmoother,
};

/// Extended Kalman filter: linearize `f` and `h` (analytic Jacobian or
/// central differences), then run the Joseph-form Kalman update.
pub fn extended_kalman<M: DiscreteSsm>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
) -> Result<GaussianFilter> {
    ekf_loop(
        model,
        observations,
        x0,
        p0,
        IekfConfig {
            max_iter: 1,
            tol: 0.0,
        },
        false,
    )
}

/// Iterated EKF (Bell–Cathey): Gauss–Newton refinement of the measurement
/// update,
///
/// `x ← x⁻ + K (y − h(x) − H (x⁻ − x))`,
///
/// re-linearizing `h` at the current iterate.
#[derive(Clone, Copy, Debug)]
pub struct IekfConfig {
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for IekfConfig {
    fn default() -> Self {
        Self {
            max_iter: 5,
            tol: 1e-8,
        }
    }
}

pub fn iterated_ekf<M: DiscreteSsm>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    cfg: IekfConfig,
) -> Result<GaussianFilter> {
    let mut cfg = cfg;
    if cfg.max_iter == 0 {
        cfg.max_iter = 1;
    }
    ekf_loop(model, observations, x0, p0, cfg, false)
}

/// Second-order EKF: first-order covariances plus the observation-mean
/// correction `ĥ_i += ½ tr(P⁻ ∇² h_i)`.
pub fn second_order_ekf<M: DiscreteSsm>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
) -> Result<GaussianFilter> {
    ekf_loop(
        model,
        observations,
        x0,
        p0,
        IekfConfig {
            max_iter: 1,
            tol: 0.0,
        },
        true,
    )
}

fn ekf_loop<M: DiscreteSsm>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    cfg: IekfConfig,
    second_order: bool,
) -> Result<GaussianFilter> {
    let (nx, ny) = check_filter_dims(model, observations, x0, p0)?;
    let mut x = x0.clone();
    let mut pmat = p0.clone();
    let mut run = empty_gaussian(x0, p0);
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        let xs = slice_from_col(&x);
        let f = transition_matrix(model, t, dt, &xs)?;
        let q = process_cov_at(model, t, dt, &xs);
        let x_pred = predict_state(model, t, dt, &x);
        let mut p_pred = &f * &pmat * f.transpose() + q;
        symmetrize(&mut p_pred);
        p_pred = spd_regularize(p_pred, 1e-14)?;
        run.predicted.push(x_pred.clone());
        run.predicted_cov.push(p_pred.clone());

        let t1 = observations.times()[i + 1];
        let y = obs_col(observations, i + 1);
        let r = observation_cov_at(model, t1);
        let mut x_lin = x_pred.clone();
        let mut p_new = p_pred.clone();
        let mut ll = 0.0;
        for it in 0..cfg.max_iter {
            let xlin_s = slice_from_col(&x_lin);
            let h = observe_matrix(model, t1, &xlin_s)?;
            let mut yhat = predict_obs(model, t1, &x_lin);
            if second_order {
                yhat = soekf_mean_correction(model, t1, &x_pred, &p_pred, &yhat)?;
            }
            // IEKF residual: y − h(x) − H (x⁻ − x), i.e. yhat = h(x) + H (x⁻ − x).
            let yhat_i = &yhat + &h * (&x_pred - &x_lin);
            let (dx, p_u, _k, step_ll) = innovation_update(&y, &yhat_i, &p_pred, &h, &r)?;
            let x_new = &x_pred + &dx;
            ll = step_ll;
            p_new = p_u;
            let mut delta = 0.0;
            for j in 0..nx {
                delta += (x_new[j] - x_lin[j]).abs();
            }
            x_lin = x_new;
            if delta < cfg.tol {
                let _ = it;
                break;
            }
        }
        x = x_lin;
        pmat = p_new;
        run.loglik += ll;
        run.filtered.push(x.clone());
        run.filtered_cov.push(pmat.clone());
        let _ = ny;
    }
    Ok(run)
}

fn soekf_mean_correction<M: DiscreteSsm>(
    model: &M,
    t: f64,
    x: &Col<f64>,
    p: &Mat<f64>,
    yhat: &Col<f64>,
) -> Result<Col<f64>> {
    let n = model.state_dim();
    let ny = model.obs_dim();
    let xs = slice_from_col(x);
    let mut y = yhat.clone();
    let h0 = observe_matrix(model, t, &xs)?;
    for j in 0..n {
        let eps = 1e-5 * (1.0 + xs[j].abs());
        let mut xp = xs.clone();
        let mut xm = xs.clone();
        xp[j] += eps;
        xm[j] -= eps;
        let hp = observe_matrix(model, t, &xp)?;
        let hm = observe_matrix(model, t, &xm)?;
        for i in 0..ny {
            // ∂²h_i / ∂x_j² ≈ (H⁺_{i j} − 2 H_{i j} + H⁻_{i j}) / ε² is not
            // quite right; use the directional Hessian via (H⁺ − H⁻)/(2ε)
            // as the j-th column of ∇(∇h_i), then tr(P Hess) ≈ Σ_k P_{jk} ∂²h_i/∂x_j∂x_k
            // with a diagonal approximation Σ_j P_{jj} ∂²h_i/∂x_j².
            let hxx = (hp[(i, j)] - hm[(i, j)]) / (2.0 * eps);
            y[i] += 0.5 * p[(j, j)] * hxx;
        }
    }
    let _ = h0;
    Ok(y)
}

/// Julier–Uhlmann / van der Merwe unscented transform parameters
///
/// `λ = α² (n+κ) − n`, `W_m⁰ = λ/(n+λ)`,
/// `W_c⁰ = λ/(n+λ) + (1−α²+β)`, `W^{i} = 1 / (2(n+λ))`.
#[derive(Clone, Copy, Debug)]
pub struct UkfParams {
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
}

impl Default for UkfParams {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 2.0,
            kappa: 0.0,
        }
    }
}

/// Unscented Kalman filter (additive-noise form).
pub fn unscented_kalman<M: DiscreteSsm>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    params: UkfParams,
) -> Result<GaussianFilter> {
    let (_nx, _ny) = check_filter_dims(model, observations, x0, p0)?;
    let mut x = x0.clone();
    let mut pmat = p0.clone();
    let mut run = empty_gaussian(x0, p0);
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        let xs = slice_from_col(&x);
        let q = process_cov_at(model, t, dt, &xs);
        let (x_pred, p_pred) = unscented_predict(model, t, dt, &x, &pmat, &q, params)?;
        run.predicted.push(x_pred.clone());
        run.predicted_cov.push(p_pred.clone());

        let t1 = observations.times()[i + 1];
        let y = obs_col(observations, i + 1);
        let r = observation_cov_at(model, t1);
        let (x_new, p_new, ll) = unscented_update(model, t1, &x_pred, &p_pred, &y, &r, params)?;
        x = x_new;
        pmat = p_new;
        run.loglik += ll;
        run.filtered.push(x.clone());
        run.filtered_cov.push(pmat.clone());
    }
    Ok(run)
}

pub(crate) fn unscented_predict<M: DiscreteSsm + ?Sized>(
    model: &M,
    t: f64,
    dt: f64,
    x: &Col<f64>,
    p: &Mat<f64>,
    q: &Mat<f64>,
    params: UkfParams,
) -> Result<(Col<f64>, Mat<f64>)> {
    let (pts, wm, wc) = sigma_points(x, p, params)?;
    let mut prop = Vec::with_capacity(pts.len());
    for pt in &pts {
        prop.push(predict_state(model, t, dt, pt));
    }
    let (m, mut cov) = unscented_moments(&prop, &wm, &wc);
    cov = cov + q;
    symmetrize(&mut cov);
    cov = spd_regularize(cov, 1e-14)?;
    Ok((m, cov))
}

pub(crate) fn unscented_update<M: DiscreteSsm + ?Sized>(
    model: &M,
    t: f64,
    x: &Col<f64>,
    p: &Mat<f64>,
    y: &Col<f64>,
    r: &Mat<f64>,
    params: UkfParams,
) -> Result<(Col<f64>, Mat<f64>, f64)> {
    let (pts, wm, wc) = sigma_points(x, p, params)?;
    let yhat_pts: Vec<Col<f64>> = pts.iter().map(|pt| predict_obs(model, t, pt)).collect();
    let (y_mean, mut pyy) = unscented_moments(&yhat_pts, &wm, &wc);
    pyy = pyy + r;
    symmetrize(&mut pyy);
    pyy = spd_regularize(pyy, 1e-14)?;
    let nx = x.nrows();
    let ny = y.nrows();
    let mut pxy = mat_zeros(nx, ny);
    for i in 0..pts.len() {
        let dx = &pts[i] - x;
        let dy = &yhat_pts[i] - &y_mean;
        for a in 0..nx {
            for b in 0..ny {
                pxy[(a, b)] += wc[i] * dx[a] * dy[b];
            }
        }
    }
    let pyy_inv = try_inverse(&pyy).ok_or_else(|| Error::numeric("UKF Pyy not invertible"))?;
    let k = &pxy * pyy_inv;
    let innov = y - &y_mean;
    let x_new = x + &k * &innov;
    let mut p_new = p - &k * &pyy * k.transpose();
    symmetrize(&mut p_new);
    p_new = spd_regularize(p_new, 1e-14)?;
    let ll = super::ssm::mvn_logpdf(y, &y_mean, &pyy)?;
    Ok((x_new, p_new, ll))
}

fn sigma_points(
    x: &Col<f64>,
    p: &Mat<f64>,
    params: UkfParams,
) -> Result<(Vec<Col<f64>>, Vec<f64>, Vec<f64>)> {
    let n = x.nrows();
    if !(params.alpha > 0.0) {
        return Err(Error::param("UKF α must be positive"));
    }
    let lambda = params.alpha * params.alpha * (n as f64 + params.kappa) - n as f64;
    let scale = n as f64 + lambda;
    if scale <= 0.0 {
        return Err(Error::param("UKF n+λ must be positive (adjust α, κ)"));
    }
    let p_s = spd_regularize(Scale(scale) * p, 1e-12)?;
    let l = cholesky(&p_s)?;
    let mut pts = Vec::with_capacity(2 * n + 1);
    pts.push(x.clone());
    for j in 0..n {
        let mut col = col_zeros(n);
        for i in 0..n {
            col[i] = l[(i, j)];
        }
        pts.push(x + &col);
        pts.push(x - &col);
    }
    let wm0 = lambda / scale;
    let wc0 = wm0 + (1.0 - params.alpha * params.alpha + params.beta);
    let wi = 1.0 / (2.0 * scale);
    let mut wm = vec![wi; 2 * n + 1];
    let mut wc = vec![wi; 2 * n + 1];
    wm[0] = wm0;
    wc[0] = wc0;
    Ok((pts, wm, wc))
}

fn unscented_moments(pts: &[Col<f64>], wm: &[f64], wc: &[f64]) -> (Col<f64>, Mat<f64>) {
    let n = pts[0].nrows();
    let mut mean = col_zeros(n);
    for (pt, &w) in pts.iter().zip(wm.iter()) {
        for i in 0..n {
            mean[i] += w * pt[i];
        }
    }
    let mut cov = mat_zeros(n, n);
    for (pt, &w) in pts.iter().zip(wc.iter()) {
        let d = pt - &mean;
        add_scaled_outer(&mut cov, w, &d);
    }
    symmetrize(&mut cov);
    (mean, cov)
}

/// Cubature Kalman filter (Arasaratnam–Haykin): `2n` points
/// `x ± √n (chol P)_j` with equal weights `1/(2n)`.
pub fn cubature_kalman<M: DiscreteSsm>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
) -> Result<GaussianFilter> {
    let (nx, _ny) = check_filter_dims(model, observations, x0, p0)?;
    let mut x = x0.clone();
    let mut pmat = p0.clone();
    let mut run = empty_gaussian(x0, p0);
    let w = 1.0 / (2.0 * nx as f64);
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        let xs = slice_from_col(&x);
        let q = process_cov_at(model, t, dt, &xs);
        let pts = cubature_points(&x, &pmat)?;
        let prop: Vec<Col<f64>> = pts
            .iter()
            .map(|pt| predict_state(model, t, dt, pt))
            .collect();
        let (x_pred, mut p_pred) = equal_moments(&prop, w);
        p_pred = p_pred + q;
        symmetrize(&mut p_pred);
        p_pred = spd_regularize(p_pred, 1e-14)?;
        run.predicted.push(x_pred.clone());
        run.predicted_cov.push(p_pred.clone());

        let t1 = observations.times()[i + 1];
        let y = obs_col(observations, i + 1);
        let r = observation_cov_at(model, t1);
        let pts2 = cubature_points(&x_pred, &p_pred)?;
        let ypts: Vec<Col<f64>> = pts2.iter().map(|pt| predict_obs(model, t1, pt)).collect();
        let (y_mean, mut pyy) = equal_moments(&ypts, w);
        pyy = pyy + &r;
        symmetrize(&mut pyy);
        pyy = spd_regularize(pyy, 1e-14)?;
        let mut pxy = mat_zeros(nx, y.nrows());
        for (pt, yp) in pts2.iter().zip(ypts.iter()) {
            let dx = pt - &x_pred;
            let dy = yp - &y_mean;
            for a in 0..nx {
                for b in 0..y.nrows() {
                    pxy[(a, b)] += w * dx[a] * dy[b];
                }
            }
        }
        let pyy_inv = try_inverse(&pyy).ok_or_else(|| Error::numeric("CKF Pyy not invertible"))?;
        let k = &pxy * pyy_inv;
        let innov = &y - &y_mean;
        x = &x_pred + &k * &innov;
        pmat = &p_pred - &k * &pyy * k.transpose();
        symmetrize(&mut pmat);
        pmat = spd_regularize(pmat, 1e-14)?;
        run.loglik += super::ssm::mvn_logpdf(&y, &y_mean, &pyy)?;
        run.filtered.push(x.clone());
        run.filtered_cov.push(pmat.clone());
    }
    Ok(run)
}

fn cubature_points(x: &Col<f64>, p: &Mat<f64>) -> Result<Vec<Col<f64>>> {
    let n = x.nrows();
    let p_s = spd_regularize(Scale(n as f64) * p, 1e-12)?;
    let l = cholesky(&p_s)?;
    let mut pts = Vec::with_capacity(2 * n);
    for j in 0..n {
        let mut col = col_zeros(n);
        for i in 0..n {
            col[i] = l[(i, j)];
        }
        pts.push(x + &col);
        pts.push(x - &col);
    }
    Ok(pts)
}

fn equal_moments(pts: &[Col<f64>], w: f64) -> (Col<f64>, Mat<f64>) {
    let n = pts[0].nrows();
    let mut mean = col_zeros(n);
    for pt in pts {
        for i in 0..n {
            mean[i] += w * pt[i];
        }
    }
    let mut cov = mat_zeros(n, n);
    for pt in pts {
        let d = pt - &mean;
        add_scaled_outer(&mut cov, w, &d);
    }
    symmetrize(&mut cov);
    (mean, cov)
}

/// Stochastic ensemble Kalman filter (perturbed observations) with
/// optional multiplicative covariance inflation after the forecast.
#[derive(Clone, Copy, Debug)]
pub struct EnkfConfig {
    pub n_ensemble: usize,
    pub inflation: f64,
}

impl Default for EnkfConfig {
    fn default() -> Self {
        Self {
            n_ensemble: 64,
            inflation: 1.0,
        }
    }
}

pub fn ensemble_kalman<M, R>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    cfg: EnkfConfig,
    rng: &mut R,
) -> Result<GaussianFilter>
where
    M: DiscreteSsm,
    R: Rng + ?Sized,
{
    if cfg.n_ensemble < 2 {
        return Err(Error::param("EnKF needs at least 2 members"));
    }
    if cfg.inflation <= 0.0 {
        return Err(Error::param("EnKF inflation must be positive"));
    }
    let (nx, ny) = check_filter_dims(model, observations, x0, p0)?;
    let mut ens: Vec<Col<f64>> = Vec::with_capacity(cfg.n_ensemble);
    for _ in 0..cfg.n_ensemble {
        ens.push(sample_mvn(x0, p0, rng)?);
    }
    let w0 = vec![1.0 / cfg.n_ensemble as f64; cfg.n_ensemble];
    let (m0, c0) = weighted_moments(&ens, &w0);
    let mut run = empty_gaussian(&m0, &c0);
    run.filtered[0] = m0;
    run.filtered_cov[0] = c0;
    let ones = 1.0 / cfg.n_ensemble as f64;
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        for member in &mut ens {
            let xs = slice_from_col(member);
            let q = process_cov_at(model, t, dt, &xs);
            let xf = predict_state(model, t, dt, member);
            *member = sample_mvn(&xf, &q, rng)?;
        }
        let (mut x_pred, mut p_pred) = weighted_moments(&ens, &vec![ones; cfg.n_ensemble]);
        if (cfg.inflation - 1.0).abs() > 0.0 {
            p_pred = Scale(cfg.inflation) * &p_pred;
            // inflate members about the mean
            for member in &mut ens {
                *member = &x_pred + Scale(cfg.inflation.sqrt()) * &(member.clone() - &x_pred);
            }
        }
        symmetrize(&mut p_pred);
        run.predicted.push(x_pred.clone());
        run.predicted_cov.push(p_pred.clone());

        let t1 = observations.times()[i + 1];
        let y = obs_col(observations, i + 1);
        let r = observation_cov_at(model, t1);
        let ypts: Vec<Col<f64>> = ens.iter().map(|m| predict_obs(model, t1, m)).collect();
        let (y_mean, mut pyy) = weighted_moments(&ypts, &vec![ones; cfg.n_ensemble]);
        pyy = pyy + &r;
        symmetrize(&mut pyy);
        pyy = spd_regularize(pyy, 1e-12)?;
        let mut pxy = mat_zeros(nx, ny);
        for (m, yp) in ens.iter().zip(ypts.iter()) {
            let dx = m - &x_pred;
            let dy = yp - &y_mean;
            for a in 0..nx {
                for b in 0..ny {
                    pxy[(a, b)] += ones * dx[a] * dy[b];
                }
            }
        }
        let pyy_inv = try_inverse(&pyy).ok_or_else(|| Error::numeric("EnKF Pyy not invertible"))?;
        let k = &pxy * pyy_inv;
        for (member, yp) in ens.iter_mut().zip(ypts.iter()) {
            let y_pert = sample_mvn(&y, &r, rng)?;
            let innov = &y_pert - yp;
            *member = &*member + &k * &innov;
        }
        let (mean, cov) = weighted_moments(&ens, &vec![ones; cfg.n_ensemble]);
        x_pred = mean.clone();
        run.loglik += super::ssm::mvn_logpdf(&y, &y_mean, &pyy)?;
        run.filtered.push(mean);
        run.filtered_cov.push(cov);
        let _ = x_pred;
    }
    Ok(run)
}

/// Gaussian-sum filter: a mixture of `M` EKFs whose weights are updated
/// by the innovation likelihood (`w_i ∝ w_i N(y; ĥ_i, S_i)`).
pub fn gaussian_sum_filter<M: DiscreteSsm>(
    model: &M,
    observations: &Path,
    components: &[(Col<f64>, Mat<f64>, f64)],
) -> Result<GaussianFilter> {
    if components.is_empty() {
        return Err(Error::param("Gaussian-sum filter needs ≥ 1 component"));
    }
    let mut means: Vec<Col<f64>> = components.iter().map(|c| c.0.clone()).collect();
    let mut covs: Vec<Mat<f64>> = components.iter().map(|c| c.1.clone()).collect();
    let mut logw: Vec<f64> = components.iter().map(|c| c.2.max(0.0).ln()).collect();
    let (w0, _) = super::ssm::normalize_log_weights(&logw)?;
    let (m0, c0) = mix_gaussians(&means, &covs, &w0);
    let mut run = empty_gaussian(&m0, &c0);
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        let t1 = observations.times()[i + 1];
        let y = obs_col(observations, i + 1);
        let mut pred_means = Vec::new();
        let mut pred_covs = Vec::new();
        for c in 0..means.len() {
            let xs = slice_from_col(&means[c]);
            let f = transition_matrix(model, t, dt, &xs)?;
            let q = process_cov_at(model, t, dt, &xs);
            let xp = predict_state(model, t, dt, &means[c]);
            let mut pp = &f * &covs[c] * f.transpose() + q;
            symmetrize(&mut pp);
            pp = spd_regularize(pp, 1e-14)?;
            pred_means.push(xp);
            pred_covs.push(pp);
        }
        let (wcur, _) = super::ssm::normalize_log_weights(&logw)?;
        let (x_pred, p_pred) = mix_gaussians(&pred_means, &pred_covs, &wcur);
        run.predicted.push(x_pred);
        run.predicted_cov.push(p_pred);

        let r = observation_cov_at(model, t1);
        for c in 0..means.len() {
            let xs = slice_from_col(&pred_means[c]);
            let h = observe_matrix(model, t1, &xs)?;
            let yhat = predict_obs(model, t1, &pred_means[c]);
            let (dx, p_new, _k, ll) = innovation_update(&y, &yhat, &pred_covs[c], &h, &r)?;
            means[c] = &pred_means[c] + &dx;
            covs[c] = p_new;
            logw[c] += ll;
        }
        let (w, lse) = super::ssm::normalize_log_weights(&logw)?;
        logw = w.iter().map(|wi| wi.ln()).collect();
        let (mean, cov) = mix_gaussians(&means, &covs, &w);
        run.loglik += lse;
        run.filtered.push(mean);
        run.filtered_cov.push(cov);
    }
    Ok(run)
}

fn mix_gaussians(means: &[Col<f64>], covs: &[Mat<f64>], w: &[f64]) -> (Col<f64>, Mat<f64>) {
    let n = means[0].nrows();
    let mut mean = col_zeros(n);
    for (m, &wi) in means.iter().zip(w.iter()) {
        for i in 0..n {
            mean[i] += wi * m[i];
        }
    }
    let mut cov = mat_zeros(n, n);
    for ((m, c), &wi) in means.iter().zip(covs.iter()).zip(w.iter()) {
        let d = m - &mean;
        add_scaled_outer(&mut cov, wi, &d);
        cov = cov + Scale(wi) * c;
    }
    symmetrize(&mut cov);
    (mean, cov)
}

/// Continuous–discrete EKF: integrate the moment ODEs between observations.
///
/// `ẋ = a(t,x)`, `Ṗ = A P + P Aᵀ + σσᵀ` (`A = ∂a/∂x`), then a discrete
/// EKF measurement update. Each observation interval is split into
/// `max(8, ⌈20 |Δt|⌉)` Euler substeps (capped at 256) so this is not the
/// same as one `SdeSsm` EKF step of size `Δt_obs`.
pub fn continuous_discrete_ekf<M, H>(
    sde: &M,
    observe: H,
    observations: &Path,
    r: &Mat<f64>,
    x0: &Col<f64>,
    p0: &Mat<f64>,
) -> Result<GaussianFilter>
where
    M: Sde + ?Sized,
    H: Fn(f64, &[f64], &mut [f64]),
{
    sde.validate()?;
    let nx = sde.dim();
    let ny = observations.dim();
    if x0.nrows() != nx || p0.nrows() != nx {
        return Err(Error::dim("CD-EKF prior dimension mismatch"));
    }
    if r.nrows() != ny || r.ncols() != ny {
        return Err(Error::dim("CD-EKF R dimension mismatch"));
    }
    let mut x = x0.clone();
    let mut pmat = p0.clone();
    let mut run = empty_gaussian(x0, p0);
    let mut a = vec![0.0; nx];
    let mut sig = vec![0.0; nx * sde.n_noise()];
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        // Enough substeps that the local Euler increment stays small; the
        // old one-step-per-observation scheme was the same as `SdeSsm` EKF.
        let nsub = ((dt.abs() * 20.0).ceil() as usize).clamp(8, 256);
        let h = dt / nsub as f64;
        let mut x_pred = x.clone();
        let mut p_pred = pmat.clone();
        for s in 0..nsub {
            let ts = t + h * s as f64;
            let xs = slice_from_col(&x_pred);
            sde.drift(ts, &xs, &mut a);
            sde.diffusion(ts, &xs, &mut sig);
            let jac = drift_jacobian(sde, ts, &xs);
            let qq = crate::linalg::gram_rowmajor(&sig, nx, sde.n_noise());
            for j in 0..nx {
                x_pred[j] += a[j] * h;
            }
            p_pred = &p_pred + Scale(h) * &(&jac * &p_pred + &p_pred * jac.transpose() + qq);
            symmetrize(&mut p_pred);
        }
        p_pred = spd_regularize(p_pred, 1e-14)?;
        run.predicted.push(x_pred.clone());
        run.predicted_cov.push(p_pred.clone());

        let t1 = observations.times()[i + 1];
        let y = obs_col(observations, i + 1);
        let xs1 = slice_from_col(&x_pred);
        let h = fd_observe_only(&observe, ny, t1, &xs1);
        let mut yhat_buf = vec![0.0; ny];
        observe(t1, &xs1, &mut yhat_buf);
        let yhat = crate::linalg::col_from_slice(&yhat_buf);
        let (dx, p_new, _k, ll) = innovation_update(&y, &yhat, &p_pred, &h, r)?;
        x = &x_pred + &dx;
        pmat = p_new;
        run.loglik += ll;
        run.filtered.push(x.clone());
        run.filtered_cov.push(pmat.clone());
    }
    Ok(run)
}

fn drift_jacobian<M: Sde + ?Sized>(sde: &M, t: f64, x: &[f64]) -> Mat<f64> {
    let n = sde.dim();
    let mut j = mat_zeros(n, n);
    let mut a0 = vec![0.0; n];
    let mut ap = vec![0.0; n];
    let mut am = vec![0.0; n];
    sde.drift(t, x, &mut a0);
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    for col in 0..n {
        let eps = 1e-6 * (1.0 + x[col].abs());
        xp[col] = x[col] + eps;
        xm[col] = x[col] - eps;
        sde.drift(t, &xp, &mut ap);
        sde.drift(t, &xm, &mut am);
        for row in 0..n {
            j[(row, col)] = (ap[row] - am[row]) / (2.0 * eps);
        }
        xp[col] = x[col];
        xm[col] = x[col];
    }
    let _ = a0;
    j
}

fn fd_observe_only<H>(observe: &H, ny: usize, t: f64, x: &[f64]) -> Mat<f64>
where
    H: Fn(f64, &[f64], &mut [f64]),
{
    let n = x.len();
    let mut h = mat_zeros(ny, n);
    let mut hp = vec![0.0; ny];
    let mut hm = vec![0.0; ny];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    for j in 0..n {
        let eps = 1e-6 * (1.0 + x[j].abs());
        xp[j] = x[j] + eps;
        xm[j] = x[j] - eps;
        observe(t, &xp, &mut hp);
        observe(t, &xm, &mut hm);
        for i in 0..ny {
            h[(i, j)] = (hp[i] - hm[i]) / (2.0 * eps);
        }
        xp[j] = x[j];
        xm[j] = x[j];
    }
    h
}

/// Extended RTS smoother: same backward pass as [`super::rts_smoother`],
/// with `F_k = ∂f/∂x` evaluated at the filtered mean.
pub fn extended_rts_smoother<M: DiscreteSsm>(
    model: &M,
    observations: &Path,
    filter: &GaussianFilter,
) -> Result<GaussianSmoother> {
    let n = filter.filtered.len();
    if n != observations.n_nodes() {
        return Err(Error::dim("extended RTS length mismatch"));
    }
    let mut xs = filter.filtered.clone();
    let mut ps = filter.filtered_cov.clone();
    for k in (0..n - 1).rev() {
        let t = observations.times()[k];
        let dt = observations.times()[k + 1] - t;
        let xf = slice_from_col(&filter.filtered[k]);
        let f = transition_matrix(model, t, dt, &xf)?;
        let p_p_inv = try_inverse(&filter.predicted_cov[k + 1])
            .ok_or_else(|| Error::numeric("extended RTS predicted P not invertible"))?;
        let c = &filter.filtered_cov[k] * f.transpose() * p_p_inv;
        let d = &xs[k + 1] - &filter.predicted[k + 1];
        xs[k] = &filter.filtered[k] + &c * &d;
        let dp = &ps[k + 1] - &filter.predicted_cov[k + 1];
        ps[k] = &filter.filtered_cov[k] + &c * dp * c.transpose();
        symmetrize(&mut ps[k]);
    }
    let _ = mat_identity(1);
    Ok(GaussianSmoother {
        smoothed: xs,
        smoothed_cov: ps,
        loglik: filter.loglik,
    })
}

/// One-step UKF proposal `p(x_k | x_{k-1}, y_k) ≈ N(m, P)` used by the
/// unscented particle filter.
pub(crate) fn ukf_proposal<M: DiscreteSsm + ?Sized>(
    model: &M,
    t: f64,
    dt: f64,
    x: &Col<f64>,
    p: &Mat<f64>,
    y: &Col<f64>,
    params: UkfParams,
) -> Result<(Col<f64>, Mat<f64>)> {
    let xs = slice_from_col(x);
    let q = process_cov_at(model, t, dt, &xs);
    let (xp, pp) = unscented_predict(model, t, dt, x, p, &q, params)?;
    let t1 = t + dt;
    let r = observation_cov_at(model, t1);
    let (m, p_post, _ll) = unscented_update(model, t1, &xp, &pp, y, &r, params)?;
    Ok((m, p_post))
}
