//! Broader diffusion catalogue (CKLS family, SABR, Jacobi, Bessel, bridges, …).

use crate::error::{Error, Result};
use crate::model::{ParametricSde, Sde};
use crate::models::util::{require_pos, require_unit};
use crate::noise::{JumpLaw, LevyMeasure};

/// Chan–Karolyi–Longstaff–Sanders: `dX = (α + β X) dt + σ X^γ dW`.
///
/// Covers Vasicek (`γ=0`), CIR (`γ=½`), CEV / Brennan–Schwartz (`γ=1`).
#[derive(Clone, Debug)]
pub struct Ckls {
    pub alpha: f64,
    pub beta: f64,
    pub sigma: f64,
    pub gamma: f64,
    p: [f64; 4],
}

impl Ckls {
    pub fn new(alpha: f64, beta: f64, sigma: f64, gamma: f64) -> Result<Self> {
        require_pos("sigma", sigma)?;
        if !(0.0..=2.0).contains(&gamma) {
            return Err(Error::param("CKLS gamma should lie in [0, 2]"));
        }
        Ok(Self {
            alpha,
            beta,
            sigma,
            gamma,
            p: [alpha, beta, sigma, gamma],
        })
    }
    pub fn cev(mu: f64, sigma: f64, gamma: f64) -> Result<Self> {
        Self::new(0.0, mu, sigma, gamma)
    }
    pub fn brennan_schwartz(alpha: f64, beta: f64, sigma: f64) -> Result<Self> {
        Self::new(alpha, beta, sigma, 1.0)
    }
}

impl Sde for Ckls {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.alpha + self.beta * x[0];
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let xa = x[0].abs();
        out[0] = self.sigma * xa.powf(self.gamma);
    }
}

impl ParametricSde for Ckls {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["alpha", "beta", "sigma", "gamma"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        *self = Self::new(p[0], p[1], p[2], p[3])?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// Constant-elasticity-of-variance (alias of CKLS with `α = 0`).
pub type Cev = Ckls;

/// Hull–White (constant θ): `dX = (θ − κ X) dt + σ dW`.
#[derive(Clone, Debug)]
pub struct HullWhite {
    pub theta: f64,
    pub kappa: f64,
    pub sigma: f64,
    p: [f64; 3],
}

impl HullWhite {
    pub fn new(theta: f64, kappa: f64, sigma: f64) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("sigma", sigma)?;
        Ok(Self {
            theta,
            kappa,
            sigma,
            p: [theta, kappa, sigma],
        })
    }
}

impl Sde for HullWhite {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.theta - self.kappa * x[0];
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma;
    }
}

impl ParametricSde for HullWhite {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["theta", "kappa", "sigma"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        *self = Self::new(p[0], p[1], p[2])?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// Black–Karasinski: `d log r = κ (θ − log r) dt + σ dW` on state `r > 0`.
#[derive(Clone, Debug)]
pub struct BlackKarasinski {
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    p: [f64; 3],
}

impl BlackKarasinski {
    pub fn new(kappa: f64, theta: f64, sigma: f64) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("sigma", sigma)?;
        Ok(Self {
            kappa,
            theta,
            sigma,
            p: [kappa, theta, sigma],
        })
    }
}

impl Sde for BlackKarasinski {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let r = x[0].max(1e-12);
        // Itô: dr = r [κ(θ−log r) + ½σ²] dt + σ r dW
        out[0] = r * (self.kappa * (self.theta - r.ln()) + 0.5 * self.sigma * self.sigma);
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma * x[0].max(1e-12);
    }
}

impl ParametricSde for BlackKarasinski {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["kappa", "theta", "sigma"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        *self = Self::new(p[0], p[1], p[2])?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// Jacobi diffusion on (0, 1): `dX = κ(θ−X) dt + σ √(X(1−X)) dW`.
#[derive(Clone, Debug)]
pub struct Jacobi {
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    p: [f64; 3],
}

impl Jacobi {
    pub fn new(kappa: f64, theta: f64, sigma: f64) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("sigma", sigma)?;
        require_unit("theta", theta)?;
        Ok(Self {
            kappa,
            theta,
            sigma,
            p: [kappa, theta, sigma],
        })
    }
}

impl Sde for Jacobi {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.kappa * (self.theta - x[0].clamp(0.0, 1.0));
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let z = x[0].clamp(0.0, 1.0);
        out[0] = self.sigma * (z * (1.0 - z)).max(0.0).sqrt();
    }
}

impl ParametricSde for Jacobi {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["kappa", "theta", "sigma"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        *self = Self::new(p[0], p[1], p[2])?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// Bessel process of dimension `δ`: `dX = (δ−1)/(2X) dt + dW`.
#[derive(Clone, Debug)]
pub struct Bessel {
    pub delta: f64,
    p: [f64; 1],
}

impl Bessel {
    pub fn new(delta: f64) -> Result<Self> {
        require_pos("delta", delta)?;
        Ok(Self { delta, p: [delta] })
    }
}

impl Sde for Bessel {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let r = x[0].abs().max(1e-8);
        out[0] = (self.delta - 1.0) / (2.0 * r);
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        out[0] = 1.0;
    }
}

impl ParametricSde for Bessel {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["delta"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        *self = Self::new(p[0])?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// Brownian bridge from `a` to `b` on `[0, T]`: `dX = (b−X)/(T−t) dt + σ dW`.
#[derive(Clone, Debug)]
pub struct BrownianBridge {
    pub a: f64,
    pub b: f64,
    pub t_end: f64,
    pub sigma: f64,
    p: [f64; 4],
}

impl BrownianBridge {
    pub fn new(a: f64, b: f64, t_end: f64, sigma: f64) -> Result<Self> {
        require_pos("T", t_end)?;
        require_pos("sigma", sigma)?;
        Ok(Self {
            a,
            b,
            t_end,
            sigma,
            p: [a, b, t_end, sigma],
        })
    }
}

impl Sde for BrownianBridge {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, t: f64, x: &[f64], out: &mut [f64]) {
        let den = (self.t_end - t).max(1e-10);
        out[0] = (self.b - x[0]) / den;
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma;
    }
    fn exact_step(&self, t: f64, x: &[f64], dt: f64, dw: &[f64], out: &mut [f64]) -> bool {
        let z = if dt > 0.0 { dw[0] / dt.sqrt() } else { 0.0 };
        let rem = (self.t_end - t).max(1e-16);
        let mean = x[0] * (rem - dt).max(0.0) / rem + self.b * dt / rem;
        let var = self.sigma * self.sigma * dt * (rem - dt).max(0.0) / rem;
        out[0] = mean + var.max(0.0).sqrt() * z;
        true
    }
}

impl ParametricSde for BrownianBridge {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["a", "b", "t_end", "sigma"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        *self = Self::new(p[0], p[1], p[2], p[3])?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// SABR: `dF = α F^β dW1`, `dα = ν α dW2`, `d⟨W1,W2⟩ = ρ dt`. State `[F, α]`.
#[derive(Clone, Debug)]
pub struct Sabr {
    pub beta: f64,
    pub nu: f64,
    pub rho: f64,
    p: [f64; 3],
}

impl Sabr {
    pub fn new(beta: f64, nu: f64, rho: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&beta) {
            return Err(Error::param("SABR beta in [0, 1]"));
        }
        require_pos("nu", nu)?;
        if !(-1.0..=1.0).contains(&rho) {
            return Err(Error::param("rho in [-1, 1]"));
        }
        Ok(Self {
            beta,
            nu,
            rho,
            p: [beta, nu, rho],
        })
    }
}

impl Sde for Sabr {
    fn dim(&self) -> usize {
        2
    }
    fn n_noise(&self) -> usize {
        2
    }
    fn drift(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        out[0] = 0.0;
        out[1] = 0.0;
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let f = x[0].max(0.0);
        let a = x[1].max(1e-12);
        let srho = (1.0 - self.rho * self.rho).max(0.0).sqrt();
        out[0] = a * f.powf(self.beta);
        out[1] = 0.0;
        out[2] = self.nu * a * self.rho;
        out[3] = self.nu * a * srho;
    }
}

impl ParametricSde for Sabr {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["beta", "nu", "rho"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        *self = Self::new(p[0], p[1], p[2])?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// 3/2 volatility: `dv = κ v (θ − v) dt + ξ v^{3/2} dW`.
#[derive(Clone, Debug)]
pub struct ThreeHalves {
    pub kappa: f64,
    pub theta: f64,
    pub xi: f64,
    p: [f64; 3],
}

impl ThreeHalves {
    pub fn new(kappa: f64, theta: f64, xi: f64) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("theta", theta)?;
        require_pos("xi", xi)?;
        Ok(Self {
            kappa,
            theta,
            xi,
            p: [kappa, theta, xi],
        })
    }
}

impl Sde for ThreeHalves {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let v = x[0].max(0.0);
        out[0] = self.kappa * v * (self.theta - v);
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let v = x[0].max(0.0);
        out[0] = self.xi * v.powf(1.5);
    }
}

impl ParametricSde for ThreeHalves {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["kappa", "theta", "xi"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        *self = Self::new(p[0], p[1], p[2])?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// Stein–Stein: OU stochastic volatility on the price `dS = μS dt + |v| S dW1`.
#[derive(Clone, Debug)]
pub struct SteinStein {
    pub mu: f64,
    pub kappa: f64,
    pub theta: f64,
    pub xi: f64,
    pub rho: f64,
    p: [f64; 5],
}

impl SteinStein {
    pub fn new(mu: f64, kappa: f64, theta: f64, xi: f64, rho: f64) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("xi", xi)?;
        if !(-1.0..=1.0).contains(&rho) {
            return Err(Error::param("rho in [-1, 1]"));
        }
        Ok(Self {
            mu,
            kappa,
            theta,
            xi,
            rho,
            p: [mu, kappa, theta, xi, rho],
        })
    }
}

impl Sde for SteinStein {
    fn dim(&self) -> usize {
        2
    }
    fn n_noise(&self) -> usize {
        2
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.mu * x[0];
        out[1] = self.kappa * (self.theta - x[1]);
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let s = x[0];
        let v = x[1].abs();
        let srho = (1.0 - self.rho * self.rho).max(0.0).sqrt();
        out[0] = v * s;
        out[1] = 0.0;
        out[2] = self.xi * self.rho;
        out[3] = self.xi * srho;
    }
}

impl ParametricSde for SteinStein {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["mu", "kappa", "theta", "xi", "rho"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        *self = Self::new(p[0], p[1], p[2], p[3], p[4])?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// Bates: Heston plus Merton jumps on the spot. State `[S, v]`.
#[derive(Clone, Debug)]
pub struct Bates {
    pub mu: f64,
    pub kappa: f64,
    pub theta: f64,
    pub xi: f64,
    pub rho: f64,
    pub intensity: f64,
    pub jump_mu: f64,
    pub jump_sigma: f64,
    levy: LevyMeasure,
    p: [f64; 8],
}

impl Bates {
    pub fn new(
        mu: f64,
        kappa: f64,
        theta: f64,
        xi: f64,
        rho: f64,
        intensity: f64,
        jump_mu: f64,
        jump_sigma: f64,
    ) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("theta", theta)?;
        require_pos("xi", xi)?;
        require_pos("jump_sigma", jump_sigma)?;
        if intensity < 0.0 || !(-1.0..=1.0).contains(&rho) {
            return Err(Error::param("invalid Bates intensity or rho"));
        }
        Ok(Self {
            mu,
            kappa,
            theta,
            xi,
            rho,
            intensity,
            jump_mu,
            jump_sigma,
            levy: LevyMeasure::CompoundPoisson {
                intensity,
                law: JumpLaw::Normal {
                    mu: jump_mu,
                    sigma: jump_sigma,
                },
            },
            p: [mu, kappa, theta, xi, rho, intensity, jump_mu, jump_sigma],
        })
    }
}

impl Sde for Bates {
    fn dim(&self) -> usize {
        2
    }
    fn n_noise(&self) -> usize {
        2
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let s = x[0];
        let v = x[1].max(0.0);
        out[0] = self.mu * s;
        out[1] = self.kappa * (self.theta - v);
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let s = x[0];
        let sv = x[1].max(0.0).sqrt();
        let srho = (1.0 - self.rho * self.rho).max(0.0).sqrt();
        out[0] = sv * s;
        out[1] = 0.0;
        out[2] = self.xi * sv * self.rho;
        out[3] = self.xi * sv * srho;
    }
    fn jump_coeff(&self, _t: f64, x: &[f64], out: &mut [f64]) -> bool {
        out[0] = x[0];
        out[1] = 0.0;
        true
    }
    fn levy(&self) -> Option<&LevyMeasure> {
        Some(&self.levy)
    }
    fn multiplicative_jumps(&self) -> bool {
        true
    }
}

impl ParametricSde for Bates {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &[
            "mu",
            "kappa",
            "theta",
            "xi",
            "rho",
            "intensity",
            "jump_mu",
            "jump_sigma",
        ]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        *self = Self::new(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// Fractional OU: OU coefficients driven by fBM (`H` stored on the model).
#[derive(Clone, Debug)]
pub struct FractionalOu {
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    pub hurst: f64,
    p: [f64; 4],
}

impl FractionalOu {
    pub fn new(kappa: f64, theta: f64, sigma: f64, hurst: f64) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("sigma", sigma)?;
        if !(0.0 < hurst && hurst < 1.0) {
            return Err(Error::param("Hurst in (0, 1)"));
        }
        Ok(Self {
            kappa,
            theta,
            sigma,
            hurst,
            p: [kappa, theta, sigma, hurst],
        })
    }
}

impl Sde for FractionalOu {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.kappa * (self.theta - x[0]);
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma;
    }
    fn hurst(&self) -> Option<f64> {
        Some(self.hurst)
    }
}

impl ParametricSde for FractionalOu {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["kappa", "theta", "sigma", "hurst"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        *self = Self::new(p[0], p[1], p[2], p[3])?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ckls_cev_diffusion() {
        let m = Ckls::cev(0.1, 0.2, 0.5).unwrap();
        let mut s = [0.0];
        m.diffusion(0.0, &[4.0], &mut s);
        assert!((s[0] - 0.4).abs() < 1e-14);
    }
}
