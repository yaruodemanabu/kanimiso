//! Linear-Gaussian parameter estimation: KF MLE and Shumway–Stoffer EM.

use faer::{Col, Mat, Scale};

use crate::error::{Error, Result};
use crate::filter::kalman::{kalman, rts_smoother};
use crate::filter::ssm::{DiscreteSsm, LinearGaussian};
use crate::linalg::{col_zeros, mat_from_row_slice, mat_identity, mat_zeros, try_inverse};
use crate::optimize::{lbfgs_b, nelder_mead, LbfgsOptions, OptOptions};
use crate::path::Path;

/// Durbin–Koopman diffuse prior `P₀ = κ I`.
pub fn diffuse_prior(n: usize, kappa: f64) -> Result<Mat<f64>> {
    if n == 0 || !(kappa > 0.0) {
        return Err(Error::param("diffuse prior needs n>0 and κ>0"));
    }
    Ok(Scale(kappa) * &mat_identity(n))
}

/// Pack `(F, Q, H, R)` (row-major, with `Q,R` stored as lower logs of a
/// diagonal for the default scalar/diagonal case).
///
/// The 1-D laboratory is `[F, log Q, H, log R]`, or `[F, log Q, log R]`
/// when [`Self::freeze_h`] is set (the usual AR(1) + noise identification).
#[derive(Clone, Debug)]
pub struct LgParametrization {
    pub nx: usize,
    pub ny: usize,
    pub freeze_h: Option<f64>,
}

impl LgParametrization {
    pub fn scalar() -> Self {
        Self {
            nx: 1,
            ny: 1,
            freeze_h: None,
        }
    }

    /// Scalar AR(1) with observation loading fixed at `h` (typically 1).
    pub fn scalar_ar(h: f64) -> Self {
        Self {
            nx: 1,
            ny: 1,
            freeze_h: Some(h),
        }
    }

    pub fn unpack(&self, p: &[f64]) -> Result<LinearGaussian> {
        if self.nx == 1 && self.ny == 1 {
            let (f, q, h, r) = if let Some(h0) = self.freeze_h {
                if p.len() != 3 {
                    return Err(Error::dim("scalar AR needs [F, logQ, logR]"));
                }
                (p[0], p[1].exp(), h0, p[2].exp())
            } else {
                if p.len() != 4 {
                    return Err(Error::dim("scalar LG needs [F, logQ, H, logR]"));
                }
                (p[0], p[1].exp(), p[2], p[3].exp())
            };
            return LinearGaussian::new(
                mat_from_row_slice(1, 1, &[f]),
                mat_from_row_slice(1, 1, &[q]),
                mat_from_row_slice(1, 1, &[h]),
                mat_from_row_slice(1, 1, &[r]),
            );
        }
        Err(Error::unsupported(
            "ssm_mle unpack is implemented for the scalar laboratory; use EM for free matrices",
        ))
    }
}

/// Maximize the Kalman innovation likelihood in a parametrized LG model.
pub fn ssm_mle(
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    start: &[f64],
    spec: &LgParametrization,
    use_lbfgs: bool,
) -> Result<(LinearGaussian, f64)> {
    let obj = |p: &[f64]| -> f64 {
        match spec
            .unpack(p)
            .and_then(|m| kalman(&m, observations, x0, p0))
        {
            Ok(f) if f.loglik.is_finite() => -f.loglik,
            _ => 1e16,
        }
    };
    let opt = OptOptions {
        max_iter: 150,
        ..OptOptions::default()
    };
    let r = if use_lbfgs {
        let lb = lbfgs_b(&obj, None, start, None, None, LbfgsOptions::default())?;
        if lb.f <= obj(start) * 0.999 || lb.converged {
            lb
        } else {
            nelder_mead(&obj, start, None, None, opt)?
        }
    } else {
        nelder_mead(&obj, start, None, None, opt)?
    };
    let model = spec.unpack(&r.x)?;
    let ll = kalman(&model, observations, x0, p0)?.loglik;
    Ok((model, ll))
}

/// Shumway–Stoffer EM output.
#[derive(Clone, Debug)]
pub struct SsmEmFit {
    pub model: LinearGaussian,
    pub loglik: f64,
    pub iters: usize,
}

/// Time-invariant linear Gaussian EM (Shumway–Stoffer) with lag-1
/// cross-covariances from the RTS smoother.
pub fn shumway_stoffer_em(
    observations: &Path,
    mut model: LinearGaussian,
    x0: &Col<f64>,
    p0: &Mat<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<SsmEmFit> {
    let nx = model.state_dim();
    let ny = model.obs_dim();
    if nx == 0 {
        return Err(Error::dim("empty state"));
    }
    let mut ll_prev = f64::NEG_INFINITY;
    let mut iters = 0;
    for it in 0..max_iter {
        iters = it + 1;
        let filt = kalman(&model, observations, x0, p0)?;
        let sm = rts_smoother(&model, &filt)?;
        let n = sm.smoothed.len();
        if n < 2 {
            return Err(Error::infer("EM needs at least two nodes"));
        }
        let lag1 = rts_lag1(&model, &filt, &sm)?;
        // M-step: F = C / A, Q = (B − F Cᵀ) / (n−1), H = D / B, R = (E − H Dᵀ)/n
        // with A = Σ (P_{t-1} + x_{t-1} x_{t-1}ᵀ), etc.
        let mut a = mat_zeros(nx, nx);
        let mut b = mat_zeros(nx, nx);
        let mut c = mat_zeros(nx, nx);
        let mut d = mat_zeros(ny, nx);
        let mut e = mat_zeros(ny, ny);
        for t in 1..n {
            let xt = &sm.smoothed[t];
            let xs = &sm.smoothed[t - 1];
            let pt = &sm.smoothed_cov[t];
            let ps = &sm.smoothed_cov[t - 1];
            add_outer(&mut a, xs, ps);
            add_outer(&mut b, xt, pt);
            for i in 0..nx {
                for j in 0..nx {
                    c[(i, j)] += lag1[t][(i, j)] + xt[i] * xs[j];
                }
            }
            if t >= 1 {
                let y = crate::filter::ssm::obs_col(observations, t);
                if (0..y.nrows()).all(|k| !y[k].is_nan()) {
                    for i in 0..ny {
                        for j in 0..nx {
                            d[(i, j)] += y[i] * xt[j];
                        }
                    }
                    for i in 0..ny {
                        for j in 0..ny {
                            e[(i, j)] += y[i] * y[j];
                        }
                    }
                }
            }
        }
        let a_inv = try_inverse(&a).ok_or_else(|| Error::numeric("EM A not invertible"))?;
        let f = &c * &a_inv;
        let mut q = Scale(1.0 / (n as f64 - 1.0)) * &(&b - &f * c.transpose());
        symmetrize_in(&mut q);
        let b_inv = try_inverse(&b).ok_or_else(|| Error::numeric("EM B not invertible"))?;
        let h = &d * &b_inv;
        let mut r = Scale(1.0 / n as f64) * &(&e - &h * d.transpose());
        symmetrize_in(&mut r);
        model.f = f;
        model.q = q;
        model.h = h;
        model.r = r;
        let ll = filt.loglik;
        if (ll - ll_prev).abs() < tol * (1.0 + ll.abs()) {
            return Ok(SsmEmFit {
                model,
                loglik: ll,
                iters,
            });
        }
        ll_prev = ll;
    }
    let ll = kalman(&model, observations, x0, p0)?.loglik;
    Ok(SsmEmFit {
        model,
        loglik: ll,
        iters,
    })
}

fn add_outer(acc: &mut Mat<f64>, x: &Col<f64>, p: &Mat<f64>) {
    let n = x.nrows();
    for i in 0..n {
        for j in 0..n {
            acc[(i, j)] += p[(i, j)] + x[i] * x[j];
        }
    }
}

fn symmetrize_in(a: &mut Mat<f64>) {
    let n = a.nrows();
    for i in 0..n {
        for j in i..n {
            let v = 0.5 * (a[(i, j)] + a[(j, i)]);
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
    }
}

/// Lag-1 smoothed covariances `P_{t,t−1}^n = J_{t−1} P_t^n`.
fn rts_lag1(
    model: &LinearGaussian,
    filter: &crate::filter::ssm::GaussianFilter,
    sm: &crate::filter::ssm::GaussianSmoother,
) -> Result<Vec<Mat<f64>>> {
    let n = sm.smoothed.len();
    let mut out = vec![mat_zeros(model.state_dim(), model.state_dim()); n];
    for t in 1..n {
        let p_f = &filter.filtered_cov[t - 1];
        let p_p = &filter.predicted_cov[t];
        let p_p_inv = try_inverse(p_p).ok_or_else(|| Error::numeric("lag-1 predicted P"))?;
        let j = p_f * model.f.transpose() * p_p_inv;
        out[t] = &j * &sm.smoothed_cov[t];
    }
    Ok(out)
}

/// CARMA driven by Gaussian Lévy → linear SSM (observation `bᵀ X`).
pub fn carma_as_linear_gaussian(
    carma: &crate::models::Carma,
    dt: f64,
    levy_var: f64,
    obs_var: f64,
) -> Result<LinearGaussian> {
    let p = carma.p;
    let a = carma.companion().clone();
    let mut g = mat_zeros(p, 1);
    g[(p - 1, 0)] = levy_var.sqrt();
    let b = col_zeros(p);
    let mut h = mat_zeros(1, p);
    // Y = bᵀ X + loc. The MA vector is stored as `observe`.
    let dummy = vec![0.0; p];
    let _ = dummy;
    for j in 0..p {
        // Reconstruct b from a unit impulse in each coordinate.
        let mut e = vec![0.0; p];
        e[j] = 1.0;
        h[(0, j)] = carma.observe(&e) - carma.loc;
    }
    let ls = crate::model::LinearStateSpace::new(a, b, g)?.with_observation(h)?;
    LinearGaussian::from_linear_sde(&ls, dt, mat_from_row_slice(1, 1, &[obs_var]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::kalman::kalman;
    use crate::linalg::col_from_slice;
    use crate::rng::seed_rng;
    use amatsuki::{Rng, StandardNormal};

    #[test]
    fn ssm_mle_recovers_scalar_f() {
        let truth = LinearGaussian::new(
            mat_from_row_slice(1, 1, &[0.8]),
            mat_from_row_slice(1, 1, &[0.04]),
            mat_from_row_slice(1, 1, &[1.0]),
            mat_from_row_slice(1, 1, &[0.09]),
        )
        .unwrap();
        let n = 160;
        let times: Vec<f64> = (0..=n).map(|i| i as f64).collect();
        let mut vals = vec![0.0; n + 1];
        let mut x = 0.0;
        let mut rng = seed_rng(6);
        for i in 1..=n {
            x = 0.8 * x + 0.2 * rng.sample(StandardNormal);
            vals[i] = x + 0.3 * rng.sample(StandardNormal);
        }
        let obs = Path::new(times, vals, 1).unwrap();
        let x0 = col_from_slice(&[0.0]);
        let p0 = mat_from_row_slice(1, 1, &[1.0]);
        let (fit, ll) = ssm_mle(
            &obs,
            &x0,
            &p0,
            &[0.5, (0.06_f64).ln(), (0.12_f64).ln()],
            &LgParametrization::scalar_ar(1.0),
            false,
        )
        .unwrap();
        assert!(ll.is_finite());
        assert!((fit.f[(0, 0)] - 0.8).abs() < 0.25, "F {}", fit.f[(0, 0)]);
        let kf = kalman(&truth, &obs, &x0, &p0).unwrap();
        assert!(ll + 8.0 >= kf.loglik);
        let em = shumway_stoffer_em(&obs, fit, &x0, &p0, 12, 1e-6).unwrap();
        assert!(em.loglik.is_finite());
        assert!(
            (em.model.f[(0, 0)] - 0.8).abs() < 0.3,
            "EM F {}",
            em.model.f[(0, 0)]
        );
    }

    #[test]
    fn missing_obs_skips_update() {
        let model = LinearGaussian::new(
            mat_from_row_slice(1, 1, &[0.9]),
            mat_from_row_slice(1, 1, &[0.01]),
            mat_from_row_slice(1, 1, &[1.0]),
            mat_from_row_slice(1, 1, &[0.01]),
        )
        .unwrap();
        let obs = Path::new(vec![0.0, 1.0, 2.0], vec![0.0, f64::NAN, 0.5], 1).unwrap();
        let x0 = col_from_slice(&[0.0]);
        let p0 = mat_from_row_slice(1, 1, &[1.0]);
        let kf = kalman(&model, &obs, &x0, &p0).unwrap();
        assert!(kf.loglik.is_finite());
        assert_eq!(kf.filtered.len(), 3);
    }
}
