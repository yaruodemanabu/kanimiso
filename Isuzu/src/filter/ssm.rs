//! Discrete-time state-space models shared by Kalman and particle filters.

use amatsuki::{Rng, StandardNormal};
use faer::{Col, Mat, Scale};

use crate::error::{Error, Result};
use crate::linalg::{
    cholesky, col_from_slice, col_zeros, dot, gram_rowmajor, logdet_spd, mat_identity, mat_zeros,
    solve_spd, spd_regularize, try_inverse, van_loan_discretize,
};
use crate::model::{LinearStateSpace, Sde};
use crate::path::Path;

pub(crate) const LN_2PI: f64 = 1.8378770664093453;

/// Discrete-time (possibly nonlinear, possibly time-varying) state space
///
/// ```text
/// x_{k+1} = f(t_k, Δt, x_k) + w_k,   w_k ∼ N(0, Q(t_k, Δt, x_k))
/// y_k     = h(t_k, x_k) + v_k,       v_k ∼ N(0, R(t_k))
/// ```
///
/// Filters in this module treat the first observation node as the prior
/// time (no measurement update at `t₀`), then predict–update on each
/// subsequent increment — the same convention as [`super::kalman_bucy`].
pub trait DiscreteSsm {
    fn state_dim(&self) -> usize;
    fn obs_dim(&self) -> usize;

    fn transition(&self, t: f64, dt: f64, x: &[f64], out: &mut [f64]);
    fn process_cov(&self, t: f64, dt: f64, x: &[f64], q: &mut Mat<f64>);
    fn observe(&self, t: f64, x: &[f64], out: &mut [f64]);
    fn observation_cov(&self, t: f64, r: &mut Mat<f64>);

    /// Analytic `∂f/∂x`. Return `false` to use central differences.
    fn transition_jacobian(&self, _t: f64, _dt: f64, _x: &[f64], _f: &mut Mat<f64>) -> bool {
        false
    }
    /// Analytic `∂h/∂x`. Return `false` to use central differences.
    fn observe_jacobian(&self, _t: f64, _x: &[f64], _h: &mut Mat<f64>) -> bool {
        false
    }
}

/// Continuous-time linear SDE used to rebuild `F,u,Q` at each `dt`.
#[derive(Clone, Debug)]
struct LinearSdeDisc {
    a: Mat<f64>,
    b: Col<f64>,
    sigma: Mat<f64>,
}

/// Time-invariant linear Gaussian model `x⁺ = F x + u + w`, `y = H x + v`.
#[derive(Clone, Debug)]
pub struct LinearGaussian {
    pub f: Mat<f64>,
    pub u: Col<f64>,
    pub q: Mat<f64>,
    pub h: Mat<f64>,
    pub r: Mat<f64>,
    continuous: Option<LinearSdeDisc>,
}

impl LinearGaussian {
    pub fn new(f: Mat<f64>, q: Mat<f64>, h: Mat<f64>, r: Mat<f64>) -> Result<Self> {
        let n = f.nrows();
        if f.ncols() != n || q.nrows() != n || q.ncols() != n {
            return Err(Error::dim("F and Q must be n × n"));
        }
        if h.ncols() != n {
            return Err(Error::dim("H must have n columns"));
        }
        let p = h.nrows();
        if r.nrows() != p || r.ncols() != p {
            return Err(Error::dim("R must be p × p"));
        }
        Ok(Self {
            f,
            u: col_zeros(n),
            q,
            h,
            r,
            continuous: None,
        })
    }

    pub fn with_input(mut self, u: Col<f64>) -> Result<Self> {
        if u.nrows() != self.f.nrows() {
            return Err(Error::dim("u length must equal state dim"));
        }
        self.u = u;
        Ok(self)
    }

    /// Exact linear transition of [`LinearStateSpace`] over a step `dt`
    /// using Van Loan (1978) for `F`, `u`, and `Q`.
    ///
    /// The continuous generator is stored, so irregular observation grids
    /// rebuild `F,u,Q` at each `Δt` rather than reusing a single Euler `Q`.
    pub fn from_linear_sde(model: &LinearStateSpace, dt: f64, r: Mat<f64>) -> Result<Self> {
        let h = model
            .h
            .as_ref()
            .ok_or_else(|| Error::param("LinearStateSpace needs an observation matrix H"))?
            .clone();
        let (f, u, q) = van_loan_discretize(&model.a, &model.b, &model.sigma, dt)?;
        let mut lg = Self::new(f, q, h, r)?.with_input(u)?;
        lg.continuous = Some(LinearSdeDisc {
            a: model.a.clone(),
            b: model.b.clone(),
            sigma: model.sigma.clone(),
        });
        Ok(lg)
    }

    fn disc_at(&self, dt: f64) -> (Mat<f64>, Col<f64>, Mat<f64>) {
        if let Some(c) = &self.continuous {
            van_loan_discretize(&c.a, &c.b, &c.sigma, dt)
                .unwrap_or_else(|_| (self.f.clone(), self.u.clone(), self.q.clone()))
        } else {
            (self.f.clone(), self.u.clone(), self.q.clone())
        }
    }
}

impl DiscreteSsm for LinearGaussian {
    fn state_dim(&self) -> usize {
        self.f.nrows()
    }
    fn obs_dim(&self) -> usize {
        self.h.nrows()
    }
    fn transition(&self, _t: f64, dt: f64, x: &[f64], out: &mut [f64]) {
        let (f, u, _) = self.disc_at(dt);
        let xv = col_from_slice(x);
        let y = &f * &xv + &u;
        copy_col_to_slice(&y, out);
    }
    fn process_cov(&self, _t: f64, dt: f64, _x: &[f64], q: &mut Mat<f64>) {
        let (_, _, qq) = self.disc_at(dt);
        copy_mat(&qq, q);
    }
    fn observe(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let xv = col_from_slice(x);
        let y = &self.h * &xv;
        copy_col_to_slice(&y, out);
    }
    fn observation_cov(&self, _t: f64, r: &mut Mat<f64>) {
        copy_mat(&self.r, r);
    }
    fn transition_jacobian(&self, _t: f64, dt: f64, _x: &[f64], f: &mut Mat<f64>) -> bool {
        let (ff, _, _) = self.disc_at(dt);
        copy_mat(&ff, f);
        true
    }
    fn observe_jacobian(&self, _t: f64, _x: &[f64], h: &mut Mat<f64>) -> bool {
        copy_mat(&self.h, h);
        true
    }
}

/// Closure-based nonlinear state space with additive Gaussian noise.
pub struct FnSsm<F, H>
where
    F: Fn(f64, f64, &[f64], &mut [f64]),
    H: Fn(f64, &[f64], &mut [f64]),
{
    nx: usize,
    ny: usize,
    f: F,
    h: H,
    q: Mat<f64>,
    r: Mat<f64>,
}

impl<F, H> FnSsm<F, H>
where
    F: Fn(f64, f64, &[f64], &mut [f64]),
    H: Fn(f64, &[f64], &mut [f64]),
{
    pub fn new(nx: usize, ny: usize, f: F, h: H, q: Mat<f64>, r: Mat<f64>) -> Result<Self> {
        if nx == 0 || ny == 0 {
            return Err(Error::dim("state and observation dims must be positive"));
        }
        if q.nrows() != nx || q.ncols() != nx {
            return Err(Error::dim("Q must be nx × nx"));
        }
        if r.nrows() != ny || r.ncols() != ny {
            return Err(Error::dim("R must be ny × ny"));
        }
        Ok(Self { nx, ny, f, h, q, r })
    }
}

impl<F, H> DiscreteSsm for FnSsm<F, H>
where
    F: Fn(f64, f64, &[f64], &mut [f64]),
    H: Fn(f64, &[f64], &mut [f64]),
{
    fn state_dim(&self) -> usize {
        self.nx
    }
    fn obs_dim(&self) -> usize {
        self.ny
    }
    fn transition(&self, t: f64, dt: f64, x: &[f64], out: &mut [f64]) {
        (self.f)(t, dt, x, out);
    }
    fn process_cov(&self, _t: f64, _dt: f64, _x: &[f64], q: &mut Mat<f64>) {
        copy_mat(&self.q, q);
    }
    fn observe(&self, t: f64, x: &[f64], out: &mut [f64]) {
        (self.h)(t, x, out);
    }
    fn observation_cov(&self, _t: f64, r: &mut Mat<f64>) {
        copy_mat(&self.r, r);
    }
}

/// Euler discretisation of an [`Sde`] with a nonlinear observation map.
///
/// `f(x) = x + a(t,x) Δt`, `Q = σσᵀ Δt`.
pub struct SdeSsm<M, H>
where
    M: Sde,
    H: Fn(f64, &[f64], &mut [f64]),
{
    sde: M,
    observe_fn: H,
    ny: usize,
    r: Mat<f64>,
}

impl<M, H> SdeSsm<M, H>
where
    M: Sde,
    H: Fn(f64, &[f64], &mut [f64]),
{
    pub fn new(sde: M, ny: usize, observe: H, r: Mat<f64>) -> Result<Self> {
        sde.validate()?;
        if ny == 0 {
            return Err(Error::dim("observation dim must be positive"));
        }
        if r.nrows() != ny || r.ncols() != ny {
            return Err(Error::dim("R must be ny × ny"));
        }
        Ok(Self {
            sde,
            observe_fn: observe,
            ny,
            r,
        })
    }
}

impl<M, H> DiscreteSsm for SdeSsm<M, H>
where
    M: Sde,
    H: Fn(f64, &[f64], &mut [f64]),
{
    fn state_dim(&self) -> usize {
        self.sde.dim()
    }
    fn obs_dim(&self) -> usize {
        self.ny
    }
    fn transition(&self, t: f64, dt: f64, x: &[f64], out: &mut [f64]) {
        let n = self.sde.dim();
        let mut a = vec![0.0; n];
        self.sde.drift(t, x, &mut a);
        for i in 0..n {
            out[i] = x[i] + a[i] * dt;
        }
    }
    fn process_cov(&self, t: f64, dt: f64, x: &[f64], q: &mut Mat<f64>) {
        let n = self.sde.dim();
        let m = self.sde.n_noise();
        let mut s = vec![0.0; n * m];
        self.sde.diffusion(t, x, &mut s);
        let qq = Scale(dt) * &gram_rowmajor(&s, n, m);
        copy_mat(&qq, q);
    }
    fn observe(&self, t: f64, x: &[f64], out: &mut [f64]) {
        (self.observe_fn)(t, x, out);
    }
    fn observation_cov(&self, _t: f64, r: &mut Mat<f64>) {
        copy_mat(&self.r, r);
    }
}

/// Gaussian filter / smoother trajectory.
#[derive(Clone, Debug)]
pub struct GaussianFilter {
    pub filtered: Vec<Col<f64>>,
    pub predicted: Vec<Col<f64>>,
    pub filtered_cov: Vec<Mat<f64>>,
    pub predicted_cov: Vec<Mat<f64>>,
    pub loglik: f64,
}

/// Rauch–Tung–Striebel (or extended RTS) smoother output.
#[derive(Clone, Debug)]
pub struct GaussianSmoother {
    pub smoothed: Vec<Col<f64>>,
    pub smoothed_cov: Vec<Mat<f64>>,
    pub loglik: f64,
}

pub(crate) fn check_filter_dims<M: DiscreteSsm + ?Sized>(
    model: &M,
    observations: &Path,
    x0: &Col<f64>,
    p0: &Mat<f64>,
) -> Result<(usize, usize)> {
    let nx = model.state_dim();
    let ny = model.obs_dim();
    if observations.dim() != ny {
        return Err(Error::dim("observation dim != model obs dim"));
    }
    if x0.nrows() != nx || p0.nrows() != nx || p0.ncols() != nx {
        return Err(Error::dim("prior dimension mismatch"));
    }
    if observations.n_nodes() < 2 {
        return Err(Error::sampling(
            "filter needs at least two observation nodes",
        ));
    }
    Ok((nx, ny))
}

pub(crate) fn copy_mat(src: &Mat<f64>, dst: &mut Mat<f64>) {
    debug_assert_eq!(src.nrows(), dst.nrows());
    debug_assert_eq!(src.ncols(), dst.ncols());
    for i in 0..src.nrows() {
        for j in 0..src.ncols() {
            dst[(i, j)] = src[(i, j)];
        }
    }
}

pub(crate) fn copy_col_to_slice(x: &Col<f64>, out: &mut [f64]) {
    debug_assert_eq!(x.nrows(), out.len());
    for i in 0..out.len() {
        out[i] = x[i];
    }
}

pub(crate) fn col_from_buf(buf: &[f64]) -> Col<f64> {
    col_from_slice(buf)
}

pub(crate) fn slice_from_col(x: &Col<f64>) -> Vec<f64> {
    (0..x.nrows()).map(|i| x[i]).collect()
}

pub(crate) fn symmetrize(p: &mut Mat<f64>) {
    let n = p.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (p[(i, j)] + p[(j, i)]);
            p[(i, j)] = v;
            p[(j, i)] = v;
        }
    }
}

pub(crate) fn add_scaled_outer(p: &mut Mat<f64>, w: f64, d: &Col<f64>) {
    let n = d.nrows();
    for i in 0..n {
        for j in 0..n {
            p[(i, j)] += w * d[i] * d[j];
        }
    }
}

pub(crate) fn joseph_update(
    p_pred: &Mat<f64>,
    k: &Mat<f64>,
    h: &Mat<f64>,
    r: &Mat<f64>,
) -> Mat<f64> {
    let n = p_pred.nrows();
    let i_n = mat_identity(n);
    let ikh = &i_n - k * h;
    let mut p = &ikh * p_pred * ikh.transpose() + k * r * k.transpose();
    symmetrize(&mut p);
    p
}

pub(crate) fn mvn_logpdf(y: &Col<f64>, mean: &Col<f64>, cov: &Mat<f64>) -> Result<f64> {
    let innov = y - mean;
    let s = spd_regularize(cov.clone(), 1e-12)?;
    let ld = logdet_spd(&s)?;
    let sol = solve_spd(&s, &innov)?;
    let p = y.nrows() as f64;
    Ok(-0.5 * (p * LN_2PI + ld + dot(&innov, &sol)))
}

pub(crate) fn sample_mvn<R: Rng + ?Sized>(
    mean: &Col<f64>,
    cov: &Mat<f64>,
    rng: &mut R,
) -> Result<Col<f64>> {
    let s = spd_regularize(cov.clone(), 1e-12)?;
    let l = cholesky(&s)?;
    let n = mean.nrows();
    let mut z = col_zeros(n);
    for i in 0..n {
        z[i] = rng.sample(StandardNormal);
    }
    Ok(mean + &l * &z)
}

pub(crate) fn transition_matrix<M: DiscreteSsm + ?Sized>(
    model: &M,
    t: f64,
    dt: f64,
    x: &[f64],
) -> Result<Mat<f64>> {
    let n = model.state_dim();
    let mut f = mat_zeros(n, n);
    if model.transition_jacobian(t, dt, x, &mut f) {
        return Ok(f);
    }
    fd_transition_jac(model, t, dt, x, &mut f);
    Ok(f)
}

pub(crate) fn observe_matrix<M: DiscreteSsm + ?Sized>(
    model: &M,
    t: f64,
    x: &[f64],
) -> Result<Mat<f64>> {
    let n = model.state_dim();
    let p = model.obs_dim();
    let mut h = mat_zeros(p, n);
    if model.observe_jacobian(t, x, &mut h) {
        return Ok(h);
    }
    fd_observe_jac(model, t, x, &mut h);
    Ok(h)
}

fn fd_transition_jac<M: DiscreteSsm + ?Sized>(
    model: &M,
    t: f64,
    dt: f64,
    x: &[f64],
    f: &mut Mat<f64>,
) {
    let n = model.state_dim();
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    let mut fp = vec![0.0; n];
    let mut fm = vec![0.0; n];
    for j in 0..n {
        let eps = 1e-6 * (1.0 + x[j].abs());
        xp[j] = x[j] + eps;
        xm[j] = x[j] - eps;
        model.transition(t, dt, &xp, &mut fp);
        model.transition(t, dt, &xm, &mut fm);
        for i in 0..n {
            f[(i, j)] = (fp[i] - fm[i]) / (2.0 * eps);
        }
        xp[j] = x[j];
        xm[j] = x[j];
    }
}

fn fd_observe_jac<M: DiscreteSsm + ?Sized>(model: &M, t: f64, x: &[f64], h: &mut Mat<f64>) {
    let n = model.state_dim();
    let p = model.obs_dim();
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    let mut hp = vec![0.0; p];
    let mut hm = vec![0.0; p];
    for j in 0..n {
        let eps = 1e-6 * (1.0 + x[j].abs());
        xp[j] = x[j] + eps;
        xm[j] = x[j] - eps;
        model.observe(t, &xp, &mut hp);
        model.observe(t, &xm, &mut hm);
        for i in 0..p {
            h[(i, j)] = (hp[i] - hm[i]) / (2.0 * eps);
        }
        xp[j] = x[j];
        xm[j] = x[j];
    }
}

pub(crate) fn process_cov_at<M: DiscreteSsm + ?Sized>(
    model: &M,
    t: f64,
    dt: f64,
    x: &[f64],
) -> Mat<f64> {
    let n = model.state_dim();
    let mut q = mat_zeros(n, n);
    model.process_cov(t, dt, x, &mut q);
    q
}

pub(crate) fn observation_cov_at<M: DiscreteSsm + ?Sized>(model: &M, t: f64) -> Mat<f64> {
    let p = model.obs_dim();
    let mut r = mat_zeros(p, p);
    model.observation_cov(t, &mut r);
    r
}

pub(crate) fn predict_state<M: DiscreteSsm + ?Sized>(
    model: &M,
    t: f64,
    dt: f64,
    x: &Col<f64>,
) -> Col<f64> {
    let n = model.state_dim();
    let xs = slice_from_col(x);
    let mut out = vec![0.0; n];
    model.transition(t, dt, &xs, &mut out);
    col_from_buf(&out)
}

pub(crate) fn predict_obs<M: DiscreteSsm + ?Sized>(model: &M, t: f64, x: &Col<f64>) -> Col<f64> {
    let p = model.obs_dim();
    let xs = slice_from_col(x);
    let mut out = vec![0.0; p];
    model.observe(t, &xs, &mut out);
    col_from_buf(&out)
}

pub(crate) fn innovation_update(
    y: &Col<f64>,
    yhat: &Col<f64>,
    p_pred: &Mat<f64>,
    h: &Mat<f64>,
    r: &Mat<f64>,
) -> Result<(Col<f64>, Mat<f64>, Mat<f64>, f64)> {
    let innov = y - yhat;
    let mut s = h * p_pred * h.transpose() + r;
    s = spd_regularize(s, 1e-12)?;
    let s_inv =
        try_inverse(&s).ok_or_else(|| Error::numeric("innovation covariance not invertible"))?;
    let k = p_pred * h.transpose() * s_inv;
    let x_gain = &k * &innov;
    let p_new = joseph_update(p_pred, &k, h, r);
    let ll = mvn_logpdf(y, yhat, &s)?;
    Ok((x_gain, p_new, k, ll))
}

pub(crate) fn log_sum_exp(logw: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &v in logw {
        if v > m {
            m = v;
        }
    }
    if !m.is_finite() {
        return f64::NEG_INFINITY;
    }
    let mut s = 0.0;
    for &v in logw {
        s += (v - m).exp();
    }
    m + s.ln()
}

pub(crate) fn normalize_log_weights(logw: &[f64]) -> Result<(Vec<f64>, f64)> {
    let lse = log_sum_exp(logw);
    if !lse.is_finite() {
        return Err(Error::numeric("particle weights underflowed"));
    }
    let w: Vec<f64> = logw.iter().map(|&v| (v - lse).exp()).collect();
    Ok((w, lse))
}

pub(crate) fn weighted_moments(particles: &[Col<f64>], weights: &[f64]) -> (Col<f64>, Mat<f64>) {
    let n = particles[0].nrows();
    let mut mean = col_zeros(n);
    for (x, &w) in particles.iter().zip(weights.iter()) {
        for i in 0..n {
            mean[i] += w * x[i];
        }
    }
    let mut cov = mat_zeros(n, n);
    for (x, &w) in particles.iter().zip(weights.iter()) {
        let d = x - &mean;
        add_scaled_outer(&mut cov, w, &d);
    }
    symmetrize(&mut cov);
    (mean, cov)
}

pub(crate) fn ess(weights: &[f64]) -> f64 {
    let s: f64 = weights.iter().map(|w| w * w).sum();
    if s > 0.0 {
        1.0 / s
    } else {
        0.0
    }
}

pub(crate) fn empty_gaussian(x0: &Col<f64>, p0: &Mat<f64>) -> GaussianFilter {
    GaussianFilter {
        filtered: vec![x0.clone()],
        predicted: vec![x0.clone()],
        filtered_cov: vec![p0.clone()],
        predicted_cov: vec![p0.clone()],
        loglik: 0.0,
    }
}

pub(crate) fn obs_col(observations: &Path, i: usize) -> Col<f64> {
    col_from_slice(observations.state(i))
}
