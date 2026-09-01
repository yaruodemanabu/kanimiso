//! Numerical schemes for SDEs (`method` in YUIMA `simulate`).

use crate::error::Result;
use crate::model::Sde;

/// Discretization scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    /// Euler–Maruyama (default in YUIMA).
    EulerMaruyama,
    /// Strong order 1.0 Milstein (commutative / 1-D noise, or Jacobian).
    Milstein,
    /// Kloeden–Platen order 1.5 (scalar SDE).
    KloedenPlaten15,
    /// Closed-form transition when the model implements one (GBM, OU).
    Exact,
}

impl Default for Scheme {
    fn default() -> Self {
        Self::EulerMaruyama
    }
}

impl Scheme {
    pub fn name(self) -> &'static str {
        match self {
            Self::EulerMaruyama => "euler",
            Self::Milstein => "milstein",
            Self::KloedenPlaten15 => "kp15",
            Self::Exact => "exact",
        }
    }
}

/// One Euler–Maruyama step: `x += a dt + σ ΔW` (+ jump handled outside).
pub fn euler_step<M: Sde + ?Sized>(
    model: &M,
    t: f64,
    x: &mut [f64],
    dt: f64,
    dw: &[f64],
    scratch_a: &mut [f64],
    scratch_s: &mut [f64],
) {
    let n = model.dim();
    let m = model.n_noise();
    model.drift(t, x, scratch_a);
    model.diffusion(t, x, scratch_s);
    for i in 0..n {
        let mut inc = scratch_a[i] * dt;
        for j in 0..m {
            inc += scratch_s[i * m + j] * dw[j];
        }
        x[i] += inc;
    }
}

/// Milstein step. For `m = 1` uses `½ σ ∂ₓσ ((ΔW)² − dt)`.
///
/// Multivariate commutative noise uses the same formula per column when a
/// Jacobian is available; otherwise falls back to Euler.
pub fn milstein_step<M: Sde + ?Sized>(
    model: &M,
    t: f64,
    x: &mut [f64],
    dt: f64,
    dw: &[f64],
    scratch_a: &mut [f64],
    scratch_s: &mut [f64],
    scratch_j: &mut [f64],
) {
    let n = model.dim();
    let m = model.n_noise();
    model.drift(t, x, scratch_a);
    model.diffusion(t, x, scratch_s);
    let has_j = model.diffusion_jacobian(t, x, scratch_j);
    if !has_j {
        // Finite-difference Jacobian of σ (central, step ε).
        fd_diffusion_jacobian(model, t, x, scratch_s, scratch_j);
    }
    for i in 0..n {
        let mut inc = scratch_a[i] * dt;
        for j in 0..m {
            inc += scratch_s[i * m + j] * dw[j];
        }
        // Commutative Milstein correction: Σ_{j,p,k} σ_{p k} ∂_{x_p} σ_{i j} * ½ (ΔW_j ΔW_k − δ_{jk} dt)
        for j in 0..m {
            for k in 0..m {
                let levy = if j == k {
                    0.5 * (dw[j] * dw[k] - dt)
                } else {
                    0.5 * dw[j] * dw[k]
                };
                let mut dsig = 0.0;
                for p in 0..n {
                    let ds_ij_dxp = scratch_j[i * n * m + p * m + j];
                    dsig += scratch_s[p * m + k] * ds_ij_dxp;
                }
                inc += dsig * levy;
            }
        }
        x[i] += inc;
    }
}

fn fd_diffusion_jacobian<M: Sde + ?Sized>(
    model: &M,
    t: f64,
    x: &[f64],
    sigma0: &[f64],
    out: &mut [f64],
) {
    let n = model.dim();
    let m = model.n_noise();
    let mut xp = x.to_vec();
    let mut sp = vec![0.0; n * m];
    let eps = 1e-6;
    for p in 0..n {
        xp[p] = x[p] + eps;
        model.diffusion(t, &xp, &mut sp);
        for i in 0..n {
            for j in 0..m {
                out[i * n * m + p * m + j] = (sp[i * m + j] - sigma0[i * m + j]) / eps;
            }
        }
        xp[p] = x[p];
    }
}

/// Kloeden–Platen 1.5 for a **scalar** SDE (finite-difference derivatives).
pub fn kp15_scalar_step<M: Sde + ?Sized>(
    model: &M,
    t: f64,
    x: &mut [f64],
    dt: f64,
    dw: f64,
    z: f64,
) -> Result<()> {
    // z ~ N(0,1) independent; ΔZ = ½ dt (ΔW + √(dt/3) z)  (space-time Lévy area)
    let dz = 0.5 * dt * (dw + (dt / 3.0).sqrt() * z);
    // ε ~ 10^{-4}(1+|x|) so ε² is well above roundoff (ε=1e-6 gave ε²=1e-12).
    let eps = (1e-4 * (1.0 + x[0].abs())).max(1e-8);
    let mut a = [0.0];
    let mut s = [0.0];
    model.drift(t, x, &mut a);
    model.diffusion(t, x, &mut s);
    let xp = [x[0] + eps];
    let xm = [x[0] - eps];
    let mut ap = [0.0];
    let mut am = [0.0];
    let mut sp = [0.0];
    let mut sm = [0.0];
    model.drift(t, &xp, &mut ap);
    model.drift(t, &xm, &mut am);
    model.diffusion(t, &xp, &mut sp);
    model.diffusion(t, &xm, &mut sm);
    let ax = (ap[0] - am[0]) / (2.0 * eps);
    let axx = (ap[0] - 2.0 * a[0] + am[0]) / (eps * eps);
    let sx = (sp[0] - sm[0]) / (2.0 * eps);
    let sxx = (sp[0] - 2.0 * s[0] + sm[0]) / (eps * eps);
    let mut atp = [0.0];
    let mut stp = [0.0];
    model.drift(t + eps, x, &mut atp);
    model.diffusion(t + eps, x, &mut stp);
    let at = (atp[0] - a[0]) / eps;
    let st = (stp[0] - s[0]) / eps;

    let l0_a = at + a[0] * ax + 0.5 * s[0] * s[0] * axx;
    let l0_s = st + a[0] * sx + 0.5 * s[0] * s[0] * sxx;
    let l1_a = s[0] * ax;
    let l1_s = s[0] * sx;
    // L¹L¹ σ = σ(σ σ″ + (σ′)²); I_{(1,1,1)} = (ΔW³ − 3 Δt ΔW)/6
    // (Kloeden–Platen 1992, (10.4.1) / the missing triple Itô integral).
    let l1l1_s = s[0] * (s[0] * sxx + sx * sx);
    let i111 = (dw * dw * dw - 3.0 * dt * dw) / 6.0;

    x[0] += a[0] * dt
        + s[0] * dw
        + l1_s * 0.5 * (dw * dw - dt)
        + l1_a * dz
        + l0_s * (dw * dt - dz)
        + l0_a * 0.5 * dt * dt
        + l1l1_s * i111;
    Ok(())
}
