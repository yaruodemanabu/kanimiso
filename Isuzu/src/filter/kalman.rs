//! Linear Kalman family: discrete KF, Kalman–Bucy, RTS, information,
//! square-root, and Sage–Husa adaptive forms.

use faer::{Col, Mat, Scale};

use crate::error::{Error, Result};
use crate::linalg::{
    cholesky, col_from_slice, mat_identity, mat_zeros, spd_regularize, try_inverse,
    van_loan_discretize,
};
use crate::model::LinearStateSpace;
use crate::path::Path;

use super::ssm::{
    check_filter_dims, empty_gaussian, innovation_update, obs_col, observation_cov_at, predict_obs,
    predict_state, process_cov_at, slice_from_col, symmetrize, transition_matrix, DiscreteSsm,
    GaussianFilter, GaussianSmoother, LinearGaussian,
};

/// Discrete Kalman filter on the Euler / exact linear transition of
/// `dX = (A X + b) dt + sigma dW`, observation `Y_k = H X_{t_k} + eta_k`.
#[derive(Clone, Debug)]
pub struct KalmanBucy {
    pub filtered: Vec<Col<f64>>,
    pub predicted: Vec<Col<f64>>,
    pub loglik: f64,
}

/// Discrete Kalman-Bucy filter (YUIMA `KalmanBucy`) using the Van Loan
/// exact discretisation of `F`, `u`, and `Q` on each observation interval.
pub fn kalman_bucy(
    model: &LinearStateSpace,
    observations: &Path,
    r_obs: &Mat<f64>,
    x0: &Col<f64>,
    p0: &Mat<f64>,
) -> Result<KalmanBucy> {
    let h = model
        .h
        .as_ref()
        .ok_or_else(|| Error::param("LinearStateSpace needs an observation matrix H"))?;
    let n = model.a.nrows();
    let p = h.nrows();
    if observations.dim() != p {
        return Err(Error::dim("observation dim != H rows"));
    }
    if r_obs.nrows() != p || x0.nrows() != n || p0.nrows() != n {
        return Err(Error::dim("Kalman dimension mismatch"));
    }
    let mut x = x0.clone();
    let mut pmat = p0.clone();
    let mut filtered = Vec::with_capacity(observations.n_nodes());
    let mut predicted = Vec::with_capacity(observations.n_nodes());
    let mut loglik = 0.0;
    filtered.push(x.clone());
    predicted.push(x.clone());
    for i in 0..observations.n_steps() {
        let dt = observations.times()[i + 1] - observations.times()[i];
        let (f, u, q) = van_loan_discretize(&model.a, &model.b, &model.sigma, dt)?;
        let i_n = mat_identity(n);
        let x_pred = &f * &x + &u;
        let p_pred = &f * &pmat * f.transpose() + q;
        predicted.push(x_pred.clone());

        let y = col_from_slice(observations.state(i + 1));
        let innov = &y - h * &x_pred;
        let s = h * &p_pred * h.transpose() + r_obs;
        let s = spd_regularize(s, 1e-12)?;
        let s_inv = try_inverse(&s)
            .ok_or_else(|| Error::numeric("Kalman innovation covariance not invertible"))?;
        let k = &p_pred * h.transpose() * s_inv;
        x = x_pred + &k * &innov;
        pmat = (&i_n - &k * h) * p_pred;
        if let Ok(l) = crate::linalg::cholesky(&s) {
            let ld = 2.0 * (0..p).map(|j| l[(j, j)].ln()).sum::<f64>();
            let sol = crate::linalg::solve_spd(&s, &innov)?;
            loglik += -0.5
                * (p as f64 * (2.0 * std::f64::consts::PI).ln()
                    + ld
                    + crate::linalg::dot(&innov, &sol));
        }
        filtered.push(x.clone());
    }
    Ok(KalmanBucy {
        filtered,
        predicted,
        loglik,
    })
}

/// Classical discrete-time Kalman filter for [`LinearGaussian`].
///
/// Predict: `x⁻ = F x + u`, `P⁻ = F P Fᵀ + Q`.
/// Update: Joseph-stabilized covariance, Gaussian innovation likelihood.
pub fn kalman(
    model: &LinearGaussian,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
) -> Result<GaussianFilter> {
    gaussian_kalman(model, observations, x0, p0, KalmanFlavour::Joseph)
}

/// Joseph-form Kalman filter followed by a Cholesky check of `P`.
///
/// This is **not** a QR / Potter / Bierman square-root filter: the
/// covariance is still propagated as a full matrix. The name is kept for
/// API stability; prefer [`kalman`] unless you want the extra SPD check.
pub fn square_root_kalman(
    model: &LinearGaussian,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
) -> Result<GaussianFilter> {
    gaussian_kalman(model, observations, x0, p0, KalmanFlavour::SquareRoot)
}

/// Information-form Kalman filter (`Y = P⁻¹`, `y = Y x`).
///
/// Prediction is done in covariance form (stable when `Q` is SPD);
/// the measurement update is `Y⁺ = Y⁻ + Hᵀ R⁻¹ H`, `y⁺ = y⁻ + Hᵀ R⁻¹ z`.
pub fn information_filter(
    model: &LinearGaussian,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
) -> Result<GaussianFilter> {
    gaussian_kalman(model, observations, x0, p0, KalmanFlavour::Information)
}

#[derive(Clone, Copy)]
enum KalmanFlavour {
    Joseph,
    SquareRoot,
    Information,
}

fn gaussian_kalman<M: DiscreteSsm>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    flavour: KalmanFlavour,
) -> Result<GaussianFilter> {
    let (nx, _ny) = check_filter_dims(model, observations, x0, p0)?;
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
        if (0..y.nrows()).any(|k| y[k].is_nan()) {
            // Durbin–Koopman missing observation: skip the update.
            x = x_pred;
            pmat = p_pred;
            run.filtered.push(x.clone());
            run.filtered_cov.push(pmat.clone());
            continue;
        }
        let h = super::ssm::observe_matrix(model, t1, &slice_from_col(&x_pred))?;
        let r = observation_cov_at(model, t1);
        let yhat = predict_obs(model, t1, &x_pred);

        match flavour {
            KalmanFlavour::Information => {
                let pinv = try_inverse(&p_pred)
                    .ok_or_else(|| Error::numeric("predicted P not invertible"))?;
                let y_info_pred = &pinv * &x_pred;
                let rinv = try_inverse(&r).ok_or_else(|| Error::numeric("R not invertible"))?;
                let y_info = &pinv + h.transpose() * &rinv * &h;
                let y_vec = &y_info_pred + h.transpose() * &rinv * &y;
                let p_new = try_inverse(&y_info)
                    .ok_or_else(|| Error::numeric("information matrix not invertible"))?;
                x = &p_new * &y_vec;
                pmat = p_new;
                symmetrize(&mut pmat);
                run.loglik +=
                    super::ssm::mvn_logpdf(&y, &yhat, &(&h * &p_pred * h.transpose() + &r))?;
            }
            KalmanFlavour::Joseph | KalmanFlavour::SquareRoot => {
                let (dx, p_new, _k, ll) = innovation_update(&y, &yhat, &p_pred, &h, &r)?;
                x = &x_pred + &dx;
                pmat = p_new;
                if matches!(flavour, KalmanFlavour::SquareRoot) {
                    let _ = cholesky(&pmat)?;
                    pmat = spd_regularize(pmat, 1e-16)?;
                }
                run.loglik += ll;
            }
        }
        let _ = nx;
        run.filtered.push(x.clone());
        run.filtered_cov.push(pmat.clone());
    }
    Ok(run)
}

/// Rauch–Tung–Striebel smoother for a linear Gaussian model.
///
/// Backward: `C = P_f Fᵀ (P⁻)⁻¹`, `x_s = x_f + C (x_s⁺ − x⁻)`,
/// `P_s = P_f + C (P_s⁺ − P⁻) Cᵀ`.
pub fn rts_smoother(model: &LinearGaussian, filter: &GaussianFilter) -> Result<GaussianSmoother> {
    let n = filter.filtered.len();
    if n == 0 || filter.predicted.len() != n {
        return Err(Error::dim("RTS needs a complete filter trajectory"));
    }
    let mut xs = filter.filtered.clone();
    let mut ps = filter.filtered_cov.clone();
    for k in (0..n - 1).rev() {
        let p_f = &filter.filtered_cov[k];
        let p_p = &filter.predicted_cov[k + 1];
        let p_p_inv =
            try_inverse(p_p).ok_or_else(|| Error::numeric("RTS predicted P not invertible"))?;
        let c = p_f * model.f.transpose() * p_p_inv;
        let d = &xs[k + 1] - &filter.predicted[k + 1];
        xs[k] = &filter.filtered[k] + &c * &d;
        let dp = &ps[k + 1] - p_p;
        ps[k] = p_f + &c * dp * c.transpose();
        symmetrize(&mut ps[k]);
    }
    Ok(GaussianSmoother {
        smoothed: xs,
        smoothed_cov: ps,
        loglik: filter.loglik,
    })
}

/// Sage–Husa adaptive Kalman filter: innovation-based matching of `Q` and `R`
/// with forgetting factor `b ∈ (0,1)`
///
/// `d_k = (1−b)/(1−b^{k+1})`,
/// `R ← (1−d) R + d (ννᵀ − H P⁻ Hᵀ)`,
/// `Q ← (1−d) Q + d (K ννᵀ Kᵀ)`.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveKalmanConfig {
    pub forgetting: f64,
}

impl Default for AdaptiveKalmanConfig {
    fn default() -> Self {
        Self { forgetting: 0.96 }
    }
}

pub fn adaptive_kalman(
    model: &LinearGaussian,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    cfg: AdaptiveKalmanConfig,
) -> Result<GaussianFilter> {
    if !(cfg.forgetting > 0.0 && cfg.forgetting < 1.0) {
        return Err(Error::param("Sage–Husa forgetting must lie in (0, 1)"));
    }
    let (_nx, _ny) = check_filter_dims(model, observations, x0, p0)?;
    let mut x = x0.clone();
    let mut pmat = p0.clone();
    let mut q = model.q.clone();
    let mut r = model.r.clone();
    let mut run = empty_gaussian(x0, p0);
    let b = cfg.forgetting;
    for i in 0..observations.n_steps() {
        let t = observations.times()[i];
        let dt = observations.times()[i + 1] - t;
        let xs = slice_from_col(&x);
        let f = transition_matrix(model, t, dt, &xs)?;
        let x_pred = predict_state(model, t, dt, &x);
        let mut p_pred = &f * &pmat * f.transpose() + &q;
        symmetrize(&mut p_pred);
        p_pred = spd_regularize(p_pred, 1e-14)?;
        run.predicted.push(x_pred.clone());
        run.predicted_cov.push(p_pred.clone());

        let t1 = observations.times()[i + 1];
        let y = obs_col(observations, i + 1);
        let h = super::ssm::observe_matrix(model, t1, &slice_from_col(&x_pred))?;
        let yhat = predict_obs(model, t1, &x_pred);
        let (dx, p_new, k, ll) = innovation_update(&y, &yhat, &p_pred, &h, &r)?;
        let innov = &y - &yhat;
        let dk = (1.0 - b) / (1.0 - b.powi((i + 1) as i32));
        let mut r_hat = &innov * innov.transpose() - &h * &p_pred * h.transpose();
        symmetrize(&mut r_hat);
        r = Scale(1.0 - dk) * &r + Scale(dk) * &r_hat;
        r = spd_regularize(r, 1e-10)?;
        let mut q_hat = &k * &innov * innov.transpose() * k.transpose();
        symmetrize(&mut q_hat);
        q = Scale(1.0 - dk) * &q + Scale(dk) * &q_hat;
        q = spd_regularize(q, 1e-10)?;
        x = &x_pred + &dx;
        pmat = p_new;
        run.loglik += ll;
        run.filtered.push(x.clone());
        run.filtered_cov.push(pmat.clone());
    }
    let _ = mat_zeros(1, 1);
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{col_from_slice, mat_from_row_slice};
    use crate::rng::seed_rng;
    use crate::sampling::Sampling;
    use crate::simulate::{simulate, SimConfig};

    #[test]
    fn kalman_tracks_ou_like() {
        let a = mat_from_row_slice(1, 1, &[-1.0]);
        let b = col_from_slice(&[0.0]);
        let sigma = mat_from_row_slice(1, 1, &[1.0]);
        let mut model = LinearStateSpace::new(a, b, sigma).unwrap();
        model = model
            .with_observation(mat_from_row_slice(1, 1, &[1.0]))
            .unwrap();
        let samp = Sampling::from_terminal(2.0, 200).unwrap();
        let mut rng = seed_rng(2);
        let latent = simulate(&model, &samp, &[0.0], &mut rng, &SimConfig::default()).unwrap();
        let r = mat_from_row_slice(1, 1, &[0.05]);
        let kb = kalman_bucy(
            &model,
            &latent,
            &r,
            &col_from_slice(&[0.0]),
            &mat_from_row_slice(1, 1, &[1.0]),
        )
        .unwrap();
        let err = (kb.filtered.last().unwrap()[0] - latent.terminal()[0]).abs();
        assert!(err < 0.5);
    }

    #[test]
    fn discrete_matches_information_and_sqrt() {
        let model = LinearGaussian::new(
            mat_from_row_slice(1, 1, &[0.9]),
            mat_from_row_slice(1, 1, &[0.04]),
            mat_from_row_slice(1, 1, &[1.0]),
            mat_from_row_slice(1, 1, &[0.16]),
        )
        .unwrap();
        let times: Vec<f64> = (0..21).map(|i| i as f64).collect();
        let mut vals = vec![0.0; 21];
        let mut x = 0.0;
        let mut rng = seed_rng(3);
        use amatsuki::{Rng, StandardNormal};
        for i in 1..21 {
            x = 0.9 * x + 0.2 * rng.sample(StandardNormal);
            vals[i] = x + 0.4 * rng.sample(StandardNormal);
        }
        let obs = Path::new(times, vals, 1).unwrap();
        let x0 = col_from_slice(&[0.0]);
        let p0 = mat_from_row_slice(1, 1, &[1.0]);
        let kf = kalman(&model, &obs, &x0, &p0).unwrap();
        let inf = information_filter(&model, &obs, &x0, &p0).unwrap();
        let sr = square_root_kalman(&model, &obs, &x0, &p0).unwrap();
        for i in 0..kf.filtered.len() {
            assert!((kf.filtered[i][0] - inf.filtered[i][0]).abs() < 1e-8);
            assert!((kf.filtered[i][0] - sr.filtered[i][0]).abs() < 1e-8);
        }
        let sm = rts_smoother(&model, &kf).unwrap();
        assert!((sm.smoothed.last().unwrap()[0] - kf.filtered.last().unwrap()[0]).abs() < 1e-12);
        let ad = adaptive_kalman(&model, &obs, &x0, &p0, AdaptiveKalmanConfig::default()).unwrap();
        assert!(ad.filtered.len() == kf.filtered.len());
        assert!(ad.loglik.is_finite());
    }
}
