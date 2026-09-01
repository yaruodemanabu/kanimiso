//! Energy / commodity spot models (Schwartz–Smith, Lucia–Schwartz,
//! Cartea–Figueroa, Gibson–Schwartz, regime-switching).
//!
//! These are the workhorse SDEs of energy-risk books: a short-term
//! deviation plus a long-term factor, a seasonal level, and spikes.

use faer::Mat;

use crate::error::{Error, Result};
use crate::linalg::{col_from_slice, mat_from_row_slice};
use crate::model::{LinearStateSpace, ParametricSde, Sde};
use crate::noise::{JumpLaw, LevyMeasure};

fn require_pos(name: &str, x: f64) -> Result<()> {
    if x > 0.0 && x.is_finite() {
        Ok(())
    } else {
        Err(Error::param(format!("{name} must be positive")))
    }
}

/// Schwartz–Smith two-factor: `log S = χ + ξ`,
/// `dχ = −κ χ dt + σ_χ dW^χ`, `dξ = μ_ξ dt + σ_ξ dW^ξ`.
///
/// State is `[χ, ξ]`. Observation of `log S` is `H = [1, 1]`.
#[derive(Clone, Debug)]
pub struct SchwartzSmith {
    pub kappa: f64,
    pub mu_xi: f64,
    pub sigma_chi: f64,
    pub sigma_xi: f64,
    pub rho: f64,
    p: [f64; 5],
}

impl SchwartzSmith {
    pub fn new(kappa: f64, mu_xi: f64, sigma_chi: f64, sigma_xi: f64, rho: f64) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("sigma_chi", sigma_chi)?;
        require_pos("sigma_xi", sigma_xi)?;
        if !(-1.0..=1.0).contains(&rho) || !mu_xi.is_finite() {
            return Err(Error::param("Schwartz–Smith ρ in [-1,1], μ_ξ finite"));
        }
        Ok(Self {
            kappa,
            mu_xi,
            sigma_chi,
            sigma_xi,
            rho,
            p: [kappa, mu_xi, sigma_chi, sigma_xi, rho],
        })
    }

    /// Linear SDE `dX = (A X + b) dt + G dW` on `[χ, ξ]`.
    pub fn linear_state(&self) -> Result<LinearStateSpace> {
        let a = mat_from_row_slice(2, 2, &[-self.kappa, 0.0, 0.0, 0.0]);
        let b = col_from_slice(&[0.0, self.mu_xi]);
        let srho = (1.0 - self.rho * self.rho).max(0.0).sqrt();
        let g = mat_from_row_slice(
            2,
            2,
            &[
                self.sigma_chi,
                0.0,
                self.sigma_xi * self.rho,
                self.sigma_xi * srho,
            ],
        );
        LinearStateSpace::new(a, b, g)?.with_observation(mat_from_row_slice(1, 2, &[1.0, 1.0]))
    }

    pub fn log_spot(&self, chi: f64, xi: f64) -> f64 {
        chi + xi
    }

    /// Futures `F(t,T)=E[S_T | χ_t, ξ_t]` under the physical measure
    /// (risk premia already absorbed into `μ_ξ` if the caller wants Q).
    ///
    /// `log F = e^{−κτ} χ + ξ + μ_ξ τ + ½ Var(log S_T)`.
    pub fn futures(&self, chi: f64, xi: f64, tau: f64) -> Result<f64> {
        if !(tau >= 0.0 && tau.is_finite()) {
            return Err(Error::param("futures tenor must be ≥ 0"));
        }
        let e = (-self.kappa * tau).exp();
        let mean = e * chi + xi + self.mu_xi * tau;
        let vchi = self.sigma_chi * self.sigma_chi * (1.0 - e * e) / (2.0 * self.kappa);
        let vxi = self.sigma_xi * self.sigma_xi * tau;
        let cov = self.rho * self.sigma_chi * self.sigma_xi * (1.0 - e) / self.kappa;
        Ok((mean + 0.5 * (vchi + vxi + 2.0 * cov)).exp())
    }
}

impl Sde for SchwartzSmith {
    fn dim(&self) -> usize {
        2
    }
    fn n_noise(&self) -> usize {
        2
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = -self.kappa * x[0];
        out[1] = self.mu_xi;
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        let srho = (1.0 - self.rho * self.rho).max(0.0).sqrt();
        out[0] = self.sigma_chi;
        out[1] = 0.0;
        out[2] = self.sigma_xi * self.rho;
        out[3] = self.sigma_xi * srho;
    }
}

impl ParametricSde for SchwartzSmith {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["kappa", "mu_xi", "sigma_chi", "sigma_xi", "rho"]
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

/// Lucia–Schwartz: `log S_t = f(t) + X_t` with OU residual and a Fourier season.
///
/// `f(t) = a + b t + Σ_k (γ_k cos(2π k t / period) + δ_k sin(…))`.
/// State is the residual `X`.
#[derive(Clone, Debug)]
pub struct LuciaSchwartz {
    pub kappa: f64,
    pub sigma: f64,
    pub level: f64,
    pub trend: f64,
    pub period: f64,
    pub cos: Vec<f64>,
    pub sin: Vec<f64>,
    p: Vec<f64>,
}

impl LuciaSchwartz {
    pub fn new(
        kappa: f64,
        sigma: f64,
        level: f64,
        trend: f64,
        period: f64,
        cos: Vec<f64>,
        sin: Vec<f64>,
    ) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("sigma", sigma)?;
        require_pos("period", period)?;
        if cos.len() != sin.len() {
            return Err(Error::dim("Lucia–Schwartz Fourier coefficients must pair"));
        }
        let mut p = vec![kappa, sigma, level, trend, period];
        p.extend_from_slice(&cos);
        p.extend_from_slice(&sin);
        Ok(Self {
            kappa,
            sigma,
            level,
            trend,
            period,
            cos,
            sin,
            p,
        })
    }

    pub fn season(&self, t: f64) -> f64 {
        let mut f = self.level + self.trend * t;
        for (k, (&c, &s)) in self.cos.iter().zip(self.sin.iter()).enumerate() {
            let w = 2.0 * std::f64::consts::PI * (k + 1) as f64 * t / self.period;
            f += c * w.cos() + s * w.sin();
        }
        f
    }

    pub fn log_spot(&self, t: f64, x: f64) -> f64 {
        self.season(t) + x
    }
}

impl Sde for LuciaSchwartz {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = -self.kappa * x[0];
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma;
    }
    fn exact_step(&self, _t: f64, x: &[f64], dt: f64, dw: &[f64], out: &mut [f64]) -> bool {
        let e = (-self.kappa * dt).exp();
        let var = self.sigma * self.sigma * (1.0 - e * e) / (2.0 * self.kappa);
        let z = if dt > 0.0 { dw[0] / dt.sqrt() } else { 0.0 };
        out[0] = x[0] * e + var.max(0.0).sqrt() * z;
        true
    }
}

impl ParametricSde for LuciaSchwartz {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["kappa", "sigma", "level", "trend", "period"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        if p.len() < 5 || (p.len() - 5) % 2 != 0 {
            return Err(Error::param("Lucia–Schwartz parameter length"));
        }
        if p.iter().any(|x| !x.is_finite()) {
            return Err(Error::param("parameters must be finite"));
        }
        let n = (p.len() - 5) / 2;
        *self = Self::new(
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            p[5..5 + n].to_vec(),
            p[5 + n..].to_vec(),
        )?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// Cartea–Figueroa: mean-reverting log-price plus a spike compound Poisson.
///
/// `dX = −α X dt + σ dW + J dN`, `S = exp(X)`.
#[derive(Clone, Debug)]
pub struct CarteaFigueroa {
    pub alpha: f64,
    pub sigma: f64,
    pub intensity: f64,
    pub jump_mu: f64,
    pub jump_sigma: f64,
    levy: LevyMeasure,
    p: [f64; 5],
}

impl CarteaFigueroa {
    pub fn new(
        alpha: f64,
        sigma: f64,
        intensity: f64,
        jump_mu: f64,
        jump_sigma: f64,
    ) -> Result<Self> {
        require_pos("alpha", alpha)?;
        require_pos("sigma", sigma)?;
        if intensity < 0.0 {
            return Err(Error::param("intensity must be ≥ 0"));
        }
        require_pos("jump_sigma", jump_sigma)?;
        Ok(Self {
            alpha,
            sigma,
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
            p: [alpha, sigma, intensity, jump_mu, jump_sigma],
        })
    }
}

impl Sde for CarteaFigueroa {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = -self.alpha * x[0];
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma;
    }
    fn jump_coeff(&self, _t: f64, _x: &[f64], out: &mut [f64]) -> bool {
        out[0] = 1.0;
        true
    }
    fn levy(&self) -> Option<&LevyMeasure> {
        Some(&self.levy)
    }
}

impl ParametricSde for CarteaFigueroa {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["alpha", "sigma", "intensity", "jump_mu", "jump_sigma"]
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

/// Gibson–Schwartz: spot + stochastic convenience yield.
///
/// `dS = (r − δ) S dt + σ_s S dW¹`, `dδ = κ(α − δ) dt + σ_δ dW²`.
/// State `[S, δ]`.
#[derive(Clone, Debug)]
pub struct GibsonSchwartz {
    pub rate: f64,
    pub kappa: f64,
    pub alpha: f64,
    pub sigma_s: f64,
    pub sigma_delta: f64,
    pub rho: f64,
    p: [f64; 6],
}

impl GibsonSchwartz {
    pub fn new(
        rate: f64,
        kappa: f64,
        alpha: f64,
        sigma_s: f64,
        sigma_delta: f64,
        rho: f64,
    ) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("sigma_s", sigma_s)?;
        require_pos("sigma_delta", sigma_delta)?;
        if !(-1.0..=1.0).contains(&rho) || !rate.is_finite() || !alpha.is_finite() {
            return Err(Error::param("Gibson–Schwartz parameters"));
        }
        Ok(Self {
            rate,
            kappa,
            alpha,
            sigma_s,
            sigma_delta,
            rho,
            p: [rate, kappa, alpha, sigma_s, sigma_delta, rho],
        })
    }

    /// Gibson–Schwartz futures `F = S exp(−H(τ) δ + A(τ))`.
    pub fn futures(&self, spot: f64, delta: f64, tau: f64) -> Result<f64> {
        if !(spot > 0.0 && tau >= 0.0 && tau.is_finite()) {
            return Err(Error::param("Gibson–Schwartz futures needs S>0, τ≥0"));
        }
        let k = self.kappa;
        let h = (1.0 - (-k * tau).exp()) / k;
        let a = (self.rate - self.alpha + 0.5 * self.sigma_delta * self.sigma_delta / (k * k)
            - self.sigma_s * self.sigma_delta * self.rho / k)
            * tau
            + self.sigma_delta * self.sigma_delta * (1.0 - (-2.0 * k * tau).exp())
                / (4.0 * k * k * k)
            + (self.alpha * k + self.sigma_s * self.sigma_delta * self.rho
                - self.sigma_delta * self.sigma_delta / k)
                * (1.0 - (-k * tau).exp())
                / (k * k);
        Ok(spot * (-h * delta + a).exp())
    }
}

impl Sde for GibsonSchwartz {
    fn dim(&self) -> usize {
        2
    }
    fn n_noise(&self) -> usize {
        2
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = (self.rate - x[1]) * x[0];
        out[1] = self.kappa * (self.alpha - x[1]);
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let srho = (1.0 - self.rho * self.rho).max(0.0).sqrt();
        out[0] = self.sigma_s * x[0];
        out[1] = 0.0;
        out[2] = self.sigma_delta * self.rho;
        out[3] = self.sigma_delta * srho;
    }
}

impl ParametricSde for GibsonSchwartz {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["rate", "kappa", "alpha", "sigma_s", "sigma_delta", "rho"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        *self = Self::new(p[0], p[1], p[2], p[3], p[4], p[5])?;
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// Two-regime GBM (or OU) switched by a continuous-time Markov chain.
#[derive(Clone, Debug)]
pub struct RegimeSwitchingDiffusion {
    pub mu: [f64; 2],
    pub sigma: [f64; 2],
    /// Off-diagonal rates `λ_{01}, λ_{10}`.
    pub lambda: [f64; 2],
    pub ou: bool,
}

impl RegimeSwitchingDiffusion {
    pub fn gbm(mu: [f64; 2], sigma: [f64; 2], lambda: [f64; 2]) -> Result<Self> {
        if sigma.iter().any(|s| *s <= 0.0) || lambda.iter().any(|l| *l < 0.0) {
            return Err(Error::param("regime-switching needs σ>0 and λ≥0"));
        }
        Ok(Self {
            mu,
            sigma,
            lambda,
            ou: false,
        })
    }

    pub fn ou(kappa_theta: [f64; 2], sigma: [f64; 2], lambda: [f64; 2]) -> Result<Self> {
        Self::gbm(kappa_theta, sigma, lambda).map(|mut s| {
            s.ou = true;
            s
        })
    }

    pub fn generator(&self) -> Mat<f64> {
        mat_from_row_slice(
            2,
            2,
            &[
                -self.lambda[0],
                self.lambda[0],
                self.lambda[1],
                -self.lambda[1],
            ],
        )
    }

    /// Simulate `(X, regime)` on a time grid. Returns a 2-D path `[X, regime]`.
    pub fn simulate<R: amatsuki::Rng + ?Sized>(
        &self,
        sampling: &crate::sampling::Sampling,
        x0: f64,
        regime0: usize,
        rng: &mut R,
    ) -> Result<crate::path::Path> {
        use amatsuki::{Exp1, StandardNormal};
        let n = sampling.n_nodes();
        let mut values = vec![0.0; n * 2];
        values[0] = x0;
        values[1] = regime0 as f64;
        let mut x = x0;
        let mut r = regime0.min(1);
        let mut t = sampling.times()[0];
        let mut next_jump = t + rng.sample(Exp1) / self.lambda[r].max(1e-16);
        for i in 1..n {
            let t1 = sampling.times()[i];
            while next_jump < t1 {
                let dt = next_jump - t;
                x = self.step(x, r, dt, rng.sample(StandardNormal));
                r = 1 - r;
                t = next_jump;
                next_jump = t + rng.sample(Exp1) / self.lambda[r].max(1e-16);
            }
            let dt = t1 - t;
            x = self.step(x, r, dt, rng.sample(StandardNormal));
            t = t1;
            values[i * 2] = x;
            values[i * 2 + 1] = r as f64;
        }
        crate::path::Path::new(sampling.times().to_vec(), values, 2)
    }

    fn step(&self, x: f64, r: usize, dt: f64, z: f64) -> f64 {
        if self.ou {
            let e = (-self.mu[r] * dt).exp();
            x * e + self.sigma[r] * ((1.0 - e * e) / (2.0 * self.mu[r].max(1e-12))).sqrt() * z
        } else {
            x * ((self.mu[r] - 0.5 * self.sigma[r] * self.sigma[r]) * dt
                + self.sigma[r] * dt.sqrt() * z)
                .exp()
        }
    }
}

/// Schwartz (1997) one-factor: `X = log S` is an OU,
/// `dX = κ(α − X) dt + σ dW`.
///
/// Futures `F = exp(e^{-κτ} X + (1−e^{-κτ})α + ½ Var(X_T))`.
#[derive(Clone, Debug)]
pub struct SchwartzOneFactor {
    pub kappa: f64,
    pub alpha: f64,
    pub sigma: f64,
    p: [f64; 3],
}

impl SchwartzOneFactor {
    pub fn new(kappa: f64, alpha: f64, sigma: f64) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("sigma", sigma)?;
        if !alpha.is_finite() {
            return Err(Error::param("Schwartz one-factor α must be finite"));
        }
        Ok(Self {
            kappa,
            alpha,
            sigma,
            p: [kappa, alpha, sigma],
        })
    }

    pub fn spot(&self, log_spot: f64) -> f64 {
        log_spot.exp()
    }

    /// `E[S_T | X_t]`.
    pub fn futures(&self, log_spot: f64, tau: f64) -> Result<f64> {
        if !(tau >= 0.0 && tau.is_finite()) {
            return Err(Error::param("Schwartz futures tenor must be ≥ 0"));
        }
        let e = (-self.kappa * tau).exp();
        let mean = log_spot * e + self.alpha * (1.0 - e);
        let var = self.sigma * self.sigma * (1.0 - e * e) / (2.0 * self.kappa);
        Ok((mean + 0.5 * var).exp())
    }
}

impl Sde for SchwartzOneFactor {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.kappa * (self.alpha - x[0]);
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma;
    }
    fn exact_step(&self, _t: f64, x: &[f64], dt: f64, dw: &[f64], out: &mut [f64]) -> bool {
        let e = (-self.kappa * dt).exp();
        let var = self.sigma * self.sigma * (1.0 - e * e) / (2.0 * self.kappa);
        let z = if dt > 0.0 { dw[0] / dt.sqrt() } else { 0.0 };
        out[0] = self.alpha + (x[0] - self.alpha) * e + var.max(0.0).sqrt() * z;
        true
    }
}

impl ParametricSde for SchwartzOneFactor {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["kappa", "alpha", "sigma"]
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

/// Spark-spread payoff `(S_power − h S_fuel)⁺` (heat rate `h`).
#[derive(Clone, Copy, Debug)]
pub struct SparkSpread {
    pub heat_rate: f64,
}

impl SparkSpread {
    pub fn new(heat_rate: f64) -> Result<Self> {
        if !(heat_rate > 0.0 && heat_rate.is_finite()) {
            return Err(Error::param("heat rate must be positive"));
        }
        Ok(Self { heat_rate })
    }
    pub fn payoff(&self, power: f64, fuel: f64) -> f64 {
        (power - self.heat_rate * fuel).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schwartz_smith_linear_and_season() {
        let m = SchwartzSmith::new(1.2, 0.01, 0.3, 0.1, 0.3).unwrap();
        let ls = m.linear_state().unwrap();
        assert_eq!(ls.a.nrows(), 2);
        assert!(ls.h.is_some());
        let ls = LuciaSchwartz::new(2.0, 0.4, 3.0, 0.0, 365.0, vec![0.2], vec![0.1]).unwrap();
        assert!(ls.season(0.0).is_finite());
        let cf = CarteaFigueroa::new(0.5, 0.3, 12.0, 0.5, 0.2).unwrap();
        assert!(cf.levy().is_some());
        let sp = SparkSpread::new(8.0).unwrap();
        assert!((sp.payoff(80.0, 5.0) - 40.0).abs() < 1e-12);
        let f0 = m.futures(0.0, 1.0, 0.0).unwrap();
        assert!((f0 - 1.0_f64.exp()).abs() < 1e-12);
        let gs = GibsonSchwartz::new(0.05, 1.0, 0.02, 0.3, 0.1, 0.3).unwrap();
        let f = gs.futures(50.0, 0.02, 0.0).unwrap();
        assert!((f - 50.0).abs() < 1e-9);
        let rs = RegimeSwitchingDiffusion::gbm([0.05, -0.1], [0.2, 0.5], [2.0, 4.0]).unwrap();
        assert_eq!(rs.generator().nrows(), 2);
        let s1 = SchwartzOneFactor::new(1.5, 3.5, 0.3).unwrap();
        let f0 = s1.futures(3.5, 0.0).unwrap();
        assert!((f0 - 3.5_f64.exp()).abs() < 1e-12);
        assert!(s1.futures(3.5, 1.0).unwrap() > 0.0);
    }
}
