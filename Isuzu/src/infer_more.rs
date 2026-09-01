//! Extra contrasts: CIC, two-stage / threshold / Kessler QMLE, Hurst.

use faer::{Mat, Scale};

use crate::error::{Error, Result};
use crate::infer::{quasi_loglik, Fit};
use crate::linalg::{mat_zeros, try_inverse};
use crate::model::{ParametricSde, Sde};
use crate::optimize::{lbfgs_b, nelder_mead, LbfgsOptions, OptOptions};
use crate::path::Path;

impl Fit {
    /// `vcov = (−H)⁻¹` from the stored Hessian.
    pub fn vcov(&self) -> Result<Mat<f64>> {
        let h = self
            .hessian
            .as_ref()
            .ok_or_else(|| Error::infer("Fit has no Hessian; cannot form vcov"))?;
        try_inverse(&(Scale(-1.0) * h))
            .ok_or_else(|| Error::numeric("Hessian is not negative definite"))
    }

    /// Coordinate-wise standard errors `√diag(vcov)`.
    pub fn se(&self) -> Result<Vec<f64>> {
        let v = self.vcov()?;
        Ok((0..self.params.len())
            .map(|i| v[(i, i)].max(0.0).sqrt())
            .collect())
    }
}

/// Uchida CIC: `−2 QL + 2 tr(Ĥ⁻¹ Ĝ)` with `G` the outer product of scores.
pub fn cic<F: ParametricSde>(family: &F, path: &Path, fit: &Fit) -> Result<f64> {
    let h = fit
        .hessian
        .as_ref()
        .ok_or_else(|| Error::infer("CIC needs a Hessian"))?;
    let g = score_outer(family, path, &fit.params)?;
    let hinv = try_inverse(h).ok_or_else(|| Error::numeric("CIC Hessian not invertible"))?;
    let hg = &hinv * &g;
    let mut tr = 0.0;
    for i in 0..hg.nrows() {
        tr += hg[(i, i)];
    }
    Ok(-2.0 * fit.quasi_loglik + 2.0 * tr.abs())
}

fn score_outer<F: ParametricSde>(family: &F, path: &Path, p: &[f64]) -> Result<Mat<f64>> {
    let k = p.len();
    let mut g = mat_zeros(k, k);
    // Score of the whole-path QL by central differences, then G = s sᵀ
    // (one-path observed information). For increment-level sandwich we
    // accumulate per-step scores.
    let n = path.n_steps();
    if n == 0 {
        return Ok(g);
    }
    for i in 0..n {
        let left = path.window(path.times()[i], path.times()[i + 1])?;
        let mut s = vec![0.0; k];
        for j in 0..k {
            let eps = 1e-5 * (1.0 + p[j].abs());
            let mut pp = p.to_vec();
            let mut pm = p.to_vec();
            pp[j] += eps;
            pm[j] -= eps;
            let fp = quasi_loglik(family, &left, &pp).unwrap_or(f64::NEG_INFINITY);
            let fm = quasi_loglik(family, &left, &pm).unwrap_or(f64::NEG_INFINITY);
            s[j] = (fp - fm) / (2.0 * eps);
        }
        for a in 0..k {
            for b in 0..k {
                g[(a, b)] += s[a] * s[b];
            }
        }
    }
    Ok(g)
}

/// Two-stage QMLE: estimate the diffusion from quadratic variation, then
/// the drift with that scale held fixed (when the last parameter is `σ`).
pub fn two_stage_qmle<F: ParametricSde>(
    family: &F,
    path: &Path,
    start: &[f64],
    sigma_index: usize,
    lower: Option<&[f64]>,
    upper: Option<&[f64]>,
    opt: OptOptions,
) -> Result<Fit> {
    if sigma_index >= start.len() {
        return Err(Error::infer("sigma_index out of range"));
    }
    let qv = path.quadratic_variation(0)?;
    let t = path.times()[path.n_nodes() - 1] - path.times()[0];
    if t <= 0.0 {
        return Err(Error::sampling("two-stage needs a positive horizon"));
    }
    let mut p0 = start.to_vec();
    p0[sigma_index] = (qv / t).max(1e-12).sqrt();
    let frozen_sig = p0[sigma_index];
    let obj = |p: &[f64]| {
        let mut q = p.to_vec();
        q[sigma_index] = frozen_sig;
        match quasi_loglik(family, path, &q) {
            Ok(ll) if ll.is_finite() => -ll,
            _ => 1e16,
        }
    };
    let r = nelder_mead(&obj, &p0, lower, upper, opt)?;
    let mut params = r.x;
    params[sigma_index] = frozen_sig;
    let ql = quasi_loglik(family, path, &params)?;
    Ok(Fit {
        params,
        names: family
            .param_names()
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        quasi_loglik: ql,
        n_obs: path.n_nodes(),
        converged: r.converged,
        iters: r.iters,
        contrast: r.f,
        hessian: None,
    })
}

/// Shimizu–Yoshida threshold QMLE: drop increments with `|ΔX| > u_n`.
pub fn threshold_qmle<F: ParametricSde>(
    family: &F,
    path: &Path,
    start: &[f64],
    threshold: f64,
    lower: Option<&[f64]>,
    upper: Option<&[f64]>,
    opt: OptOptions,
) -> Result<Fit> {
    if !(threshold > 0.0) {
        return Err(Error::param("threshold must be positive"));
    }
    let kept = threshold_path(path, threshold)?;
    crate::infer::qmle(family, &kept, start, lower, upper, opt)
}

fn threshold_path(path: &Path, u: f64) -> Result<Path> {
    let mut times = vec![path.times()[0]];
    let mut values = path.state(0).to_vec();
    for i in 0..path.n_steps() {
        let mut big = false;
        for j in 0..path.dim() {
            if (path.state(i + 1)[j] - path.state(i)[j]).abs() > u {
                big = true;
            }
        }
        if !big {
            times.push(path.times()[i + 1]);
            values.extend_from_slice(path.state(i + 1));
        }
    }
    if times.len() < 2 {
        return Err(Error::infer("threshold removed every increment"));
    }
    Path::new(times, values, path.dim())
}

/// Kessler (1997) second-order Gaussian contrast for a scalar diffusion:
/// mean `x + a Δ + (a a' + ½ σ² a'') Δ²/2` reduced to
/// `x + a Δ + ½ (a ∂_x a + ½ σ² ∂_{xx} a) Δ²` with FD derivatives.
pub fn kessler_loglik_frozen<M: Sde + ?Sized>(model: &M, path: &Path) -> Result<f64> {
    if model.dim() != 1 || path.dim() != 1 {
        return Err(Error::unsupported(
            "Kessler contrast is scalar in this crate",
        ));
    }
    const LN_2PI: f64 = 1.8378770664093453;
    let mut ll = 0.0;
    let mut a = [0.0];
    let mut s = [0.0];
    for i in 0..path.n_steps() {
        let t = path.times()[i];
        let dt = path.times()[i + 1] - t;
        let x = path.state(i)[0];
        model.drift(t, &[x], &mut a);
        model.diffusion(t, &[x], &mut s);
        let eps = 1e-5 * (1.0 + x.abs());
        let mut ap = [0.0];
        let mut am = [0.0];
        model.drift(t, &[x + eps], &mut ap);
        model.drift(t, &[x - eps], &mut am);
        let da = (ap[0] - am[0]) / (2.0 * eps);
        let dda = (ap[0] - 2.0 * a[0] + am[0]) / (eps * eps);
        let mean = x + a[0] * dt + 0.5 * (a[0] * da + 0.5 * s[0] * s[0] * dda) * dt * dt;
        let var = (s[0] * s[0]) * dt * (1.0 + (da + 0.0) * dt);
        if !(var > 0.0) {
            return Ok(f64::NEG_INFINITY);
        }
        let innov = path.state(i + 1)[0] - mean;
        ll += -0.5 * (LN_2PI + var.ln() + innov * innov / var);
    }
    Ok(ll)
}

pub fn kessler_qmle<F: ParametricSde>(
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
        .and_then(|m| kessler_loglik_frozen(&m, path))
    {
        Ok(ll) if ll.is_finite() => -ll,
        _ => 1e16,
    };
    let r = lbfgs_b(&obj, None, start, lower, upper, LbfgsOptions::default())
        .or_else(|_| nelder_mead(&obj, start, lower, upper, opt))?;
    let frozen = family.with_params(&r.x)?.freeze()?;
    let ql = kessler_loglik_frozen(&frozen, path)?;
    Ok(Fit {
        params: r.x,
        names: family
            .param_names()
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        quasi_loglik: ql,
        n_obs: path.n_nodes(),
        converged: r.converged,
        iters: r.iters,
        contrast: r.f,
        hessian: None,
    })
}

/// Quadratic generalized variation Hurst estimator
/// `Ĥ = ½ log₂( QV_{2Δ} / QV_Δ ) + ½` on a regular grid.
pub fn hurst_qgv(path: &Path, component: usize) -> Result<f64> {
    let dx = path.increments(component)?;
    if dx.len() < 8 {
        return Err(Error::infer("qgv needs more increments"));
    }
    let q1: f64 = dx.iter().map(|d| d * d).sum();
    let mut q2 = 0.0;
    let mut n2 = 0usize;
    let mut i = 0;
    while i + 1 < dx.len() {
        let s = dx[i] + dx[i + 1];
        q2 += s * s;
        n2 += 1;
        i += 2;
    }
    if q1 <= 0.0 || q2 <= 0.0 || n2 == 0 {
        return Err(Error::numeric("qgv quadratic variation vanished"));
    }
    // Un-normalized: QV_{2Δ}/QV_Δ ≈ 2^{2H−1}, so H = ½ log₂(QV₂/QV₁) + ½.
    // (Averaging each QV by its increment count would drop the +½.)
    let _ = n2;
    let r = q2 / q1;
    Ok(0.5 * r.log2() + 0.5)
}

/// Method-of-moments fBM increment variance: `Var(ΔX) ∝ Δ^{2H}`.
pub fn hurst_mmfrac(path: &Path, component: usize) -> Result<f64> {
    let x = path.component(component)?;
    if x.len() < 16 {
        return Err(Error::infer("mmfrac needs a longer path"));
    }
    let dt = path.times()[1] - path.times()[0];
    let v1 = increment_var(&x, 1);
    let v2 = increment_var(&x, 2);
    if v1 <= 0.0 || v2 <= 0.0 {
        return Err(Error::numeric("mmfrac variance vanished"));
    }
    let h = 0.5 * (v2 / v1).ln() / 2.0_f64.ln();
    let _ = dt;
    Ok(h)
}

fn increment_var(x: &[f64], step: usize) -> f64 {
    let mut s = 0.0;
    let mut n = 0usize;
    let mut i = 0;
    while i + step < x.len() {
        let d = x[i + step] - x[i];
        s += d * d;
        n += 1;
        i += step;
    }
    if n == 0 {
        0.0
    } else {
        s / n as f64
    }
}

/// Iacus–Mercuri–Rroji style GMM for COGARCH(1,1): match `E[ΔG]`,
/// `E[(ΔG)²]`, and `Cov((ΔG_t)²,(ΔG_{t−1})²)`.
pub fn cogarch_gmm(path: &Path, start: &[f64; 3]) -> Result<(crate::models::Cogarch, f64)> {
    if path.dim() < 1 {
        return Err(Error::dim("COGARCH GMM needs a G path"));
    }
    let g = path.component(0)?;
    let mut dg = Vec::new();
    for w in g.windows(2) {
        dg.push(w[1] - w[0]);
    }
    if dg.len() < 8 {
        return Err(Error::infer("COGARCH GMM needs more increments"));
    }
    let m1 = dg.iter().sum::<f64>() / dg.len() as f64;
    let m2 = dg.iter().map(|x| x * x).sum::<f64>() / dg.len() as f64;
    let mut ac = 0.0;
    for i in 1..dg.len() {
        ac += (dg[i] * dg[i]) * (dg[i - 1] * dg[i - 1]);
    }
    ac /= (dg.len() - 1) as f64;
    let obj = |p: &[f64]| {
        if p[0] <= 0.0 || p[1] <= 0.0 || p[2] <= 0.0 {
            return 1e16;
        }
        // Stationary E[V] = β / (η − φ E[(ΔL)²]) ≈ β/η for a small-jump driver.
        let ev = p[0] / p[1];
        let e2 = ev; // E[(ΔG)²] ≈ E[V] Δt, Δt absorbed into identification
        let eac = ev * ev * (p[2] / p[1]).max(0.0);
        (m1).powi(2) + (m2 - e2).powi(2) + (ac - eac).powi(2)
    };
    let r = nelder_mead(
        &obj,
        start,
        Some(&[1e-6, 1e-6, 1e-6]),
        None,
        OptOptions::default(),
    )?;
    let levy = crate::noise::LevyMeasure::CompoundPoisson {
        intensity: 1.0,
        law: crate::noise::JumpLaw::Normal {
            mu: 0.0,
            sigma: 1.0,
        },
    };
    let model = crate::models::Cogarch::cogarch11(r.x[0], r.x[1], r.x[2], levy)?;
    Ok((model, r.f))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::OrnsteinUhlenbeck;
    use crate::optimize::OptOptions;
    use crate::rng::seed_rng;
    use crate::sampling::Sampling;
    use crate::simulate::simulate_ou_exact;

    #[test]
    fn hurst_bm_near_half_and_two_stage() {
        let ou = OrnsteinUhlenbeck::new(1.2, 0.0, 0.4).unwrap();
        let samp = Sampling::from_terminal(8.0, 2000).unwrap();
        let mut rng = seed_rng(2);
        let path = simulate_ou_exact(&ou, &samp, 0.0, &mut rng).unwrap();
        let h = hurst_qgv(&path, 0).unwrap();
        assert!((h - 0.5).abs() < 0.15, "qgv H={h}");
        let fit = two_stage_qmle(
            &ou,
            &path,
            &[0.8, 0.0, 0.7],
            2,
            Some(&[0.05, -2.0, 0.05]),
            Some(&[4.0, 2.0, 2.0]),
            OptOptions {
                max_iter: 80,
                ..OptOptions::default()
            },
        )
        .unwrap();
        assert!((fit.params[2] - 0.4).abs() < 0.15);
    }
}
