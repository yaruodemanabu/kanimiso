//! Classical diffusion and jump-diffusion models.

use crate::error::{Error, Result};
use crate::model::{ParametricSde, Sde};
use crate::noise::{JumpLaw, LevyMeasure};

fn require_pos(name: &str, x: f64) -> Result<()> {
    if x > 0.0 && x.is_finite() {
        Ok(())
    } else {
        Err(Error::param(format!("{name} must be positive")))
    }
}

/// Geometric Brownian motion / Black–Scholes: `dX = μ X dt + σ X dW`.
#[derive(Clone, Debug)]
pub struct GeometricBrownianMotion {
    pub mu: f64,
    pub sigma: f64,
    p: [f64; 2],
}

impl GeometricBrownianMotion {
    pub fn new(mu: f64, sigma: f64) -> Result<Self> {
        require_pos("sigma", sigma)?;
        Ok(Self {
            mu,
            sigma,
            p: [mu, sigma],
        })
    }
}

impl Sde for GeometricBrownianMotion {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.mu * x[0];
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma * x[0];
    }
    fn diffusion_jacobian(&self, _t: f64, _x: &[f64], out: &mut [f64]) -> bool {
        out[0] = self.sigma;
        true
    }
    fn exact_step(&self, _t: f64, x: &[f64], dt: f64, dw: &[f64], out: &mut [f64]) -> bool {
        if x[0] <= 0.0 {
            return false;
        }
        out[0] = x[0] * ((self.mu - 0.5 * self.sigma * self.sigma) * dt + self.sigma * dw[0]).exp();
        true
    }
}

impl ParametricSde for GeometricBrownianMotion {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["mu", "sigma"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        require_pos("sigma", p[1])?;
        self.mu = p[0];
        self.sigma = p[1];
        self.p = [p[0], p[1]];
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

// ParametricSde::params is awkward for struct fields. We'll store params in a
// small array on each named model to make the trait ergonomic.

/// Black–Scholes alias.
pub type BlackScholes = GeometricBrownianMotion;

/// Ornstein–Uhlenbeck: `dX = κ (θ − X) dt + σ dW`.
#[derive(Clone, Debug)]
pub struct OrnsteinUhlenbeck {
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    p: [f64; 3],
}

impl OrnsteinUhlenbeck {
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

    /// Exact Gaussian transition (used by the `Exact` scheme).
    pub fn exact_step(&self, x: f64, dt: f64, z: f64) -> f64 {
        let e = (-self.kappa * dt).exp();
        let var = self.sigma * self.sigma * (1.0 - e * e) / (2.0 * self.kappa);
        self.theta + (x - self.theta) * e + var.max(0.0).sqrt() * z
    }
}

impl Sde for OrnsteinUhlenbeck {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.kappa * (self.theta - x[0]);
    }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma;
    }
    fn diffusion_jacobian(&self, _t: f64, _x: &[f64], out: &mut [f64]) -> bool {
        out[0] = 0.0;
        true
    }
    fn exact_step(&self, _t: f64, x: &[f64], dt: f64, dw: &[f64], out: &mut [f64]) -> bool {
        let z = if dt > 0.0 { dw[0] / dt.sqrt() } else { 0.0 };
        out[0] = self.exact_step(x[0], dt, z);
        true
    }
}

impl ParametricSde for OrnsteinUhlenbeck {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["kappa", "theta", "sigma"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        require_pos("kappa", p[0])?;
        require_pos("sigma", p[2])?;
        self.kappa = p[0];
        self.theta = p[1];
        self.sigma = p[2];
        self.p = [p[0], p[1], p[2]];
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// Vasicek short-rate model (OU with interest-rate interpretation).
pub type Vasicek = OrnsteinUhlenbeck;

/// Cox–Ingersoll–Ross: `dX = κ (θ − X) dt + σ √X dW`.
#[derive(Clone, Debug)]
pub struct Cir {
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    p: [f64; 3],
}

impl Cir {
    pub fn new(kappa: f64, theta: f64, sigma: f64) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("theta", theta)?;
        require_pos("sigma", sigma)?;
        Ok(Self {
            kappa,
            theta,
            sigma,
            p: [kappa, theta, sigma],
        })
    }

    /// Feller condition `2 κ θ ≥ σ²` (positivity).
    pub fn feller_holds(&self) -> bool {
        2.0 * self.kappa * self.theta + 1e-15 >= self.sigma * self.sigma
    }

    /// Exact CIR transition via a non-central χ² draw.
    ///
    /// `c = 2κ / (σ² (1−e^{−κΔt}))`, `X' = χ²_{4κθ/σ²}(c X e^{−κΔt}) / c`.
    pub fn sample_exact<R: amatsuki::Rng + ?Sized>(
        &self,
        x: f64,
        dt: f64,
        rng: &mut R,
    ) -> crate::error::Result<f64> {
        if dt <= 0.0 {
            return Ok(x.max(0.0));
        }
        // Glasserman: X' = c χ²_d(λ) with
        // c = σ²(1−e^{−κΔ})/(4κ), d = 4κθ/σ², λ = X e^{−κΔ}/c.
        // The mixed convention X' = χ² / (2κ/(σ²(1−e^{−κΔ}))) doubles the mean.
        let decay = (-self.kappa * dt).exp();
        let c = self.sigma * self.sigma * (1.0 - decay) / (4.0 * self.kappa);
        let df = 4.0 * self.kappa * self.theta / (self.sigma * self.sigma);
        let ncp = if c > 0.0 { x.max(0.0) * decay / c } else { 0.0 };
        let z = amatsuki::sample_noncentral_chi2(df, ncp, rng)
            .map_err(|e| crate::error::Error::numeric(e.to_string()))?;
        Ok(c * z)
    }
}

impl Sde for Cir {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.kappa * (self.theta - x[0].max(0.0));
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma * x[0].max(0.0).sqrt();
    }
    fn diffusion_jacobian(&self, _t: f64, x: &[f64], out: &mut [f64]) -> bool {
        let xp = x[0].max(0.0);
        out[0] = if xp > 0.0 {
            0.5 * self.sigma / xp.sqrt()
        } else {
            0.0
        };
        true
    }
}

impl ParametricSde for Cir {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["kappa", "theta", "sigma"]
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn set_params(&mut self, p: &[f64]) -> Result<()> {
        Self::check_params(p, self.param_names())?;
        require_pos("kappa", p[0])?;
        require_pos("theta", p[1])?;
        require_pos("sigma", p[2])?;
        self.kappa = p[0];
        self.theta = p[1];
        self.sigma = p[2];
        self.p = [p[0], p[1], p[2]];
        Ok(())
    }
    fn freeze(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

/// Heston stochastic volatility
///
/// ```text
/// dS = μ S dt + √v S dW¹
/// dv = κ (θ − v) dt + ξ √v dW²
/// d⟨W¹, W²⟩ = ρ dt
/// ```
///
/// State is `[S, v]`. Noise is already Cholesky-mixed: the diffusion matrix
/// embeds `ρ`.
#[derive(Clone, Debug)]
pub struct Heston {
    pub mu: f64,
    pub kappa: f64,
    pub theta: f64,
    pub xi: f64,
    pub rho: f64,
    p: [f64; 5],
}

impl Heston {
    pub fn new(mu: f64, kappa: f64, theta: f64, xi: f64, rho: f64) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("theta", theta)?;
        require_pos("xi", xi)?;
        if !(-1.0..=1.0).contains(&rho) {
            return Err(Error::param("rho must lie in [-1, 1]"));
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

    /// Andersen (2008) quadratic-exponential step for `(S, v)`.
    ///
    /// `z_v, z_s` are independent `N(0,1)` draws. The variance uses the
    /// moment-matched quadratic / exponential switch; the spot is a
    /// log-Euler step with the realized `v` average.
    pub fn qe_step(&self, s: f64, v: f64, dt: f64, z_v: f64, z_s: f64) -> (f64, f64) {
        let k = self.kappa;
        let th = self.theta;
        let xi = self.xi;
        let e = (-k * dt).exp();
        let m = th + (v - th) * e;
        let s2 = v * xi * xi * e * (1.0 - e) / k + th * xi * xi * (1.0 - e).powi(2) / (2.0 * k);
        let psi = if m > 0.0 { s2 / (m * m) } else { f64::INFINITY };
        let v_next = if psi <= 1.5 && psi.is_finite() {
            let b2 = 2.0 / psi - 1.0 + (2.0 / psi * (2.0 / psi - 1.0)).sqrt();
            let a = m / (1.0 + b2);
            a * (b2.sqrt() + z_v).powi(2)
        } else {
            let p = (psi - 1.0) / (psi + 1.0);
            let beta = (1.0 - p) / m.max(1e-16);
            let u = {
                let z = z_v.abs();
                let t = 1.0 / (1.0 + 0.3275911 * z);
                let a = t
                    * (0.254829592
                        + t * (-0.284496736
                            + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
                let erf = if z_v >= 0.0 {
                    1.0 - a * (-z * z).exp()
                } else {
                    a * (-z * z).exp() - 1.0
                };
                (0.5 * (1.0 + erf)).clamp(1e-12, 1.0 - 1e-12)
            };
            if u <= p {
                0.0
            } else {
                ((1.0 - p) / (1.0 - u)).ln() / beta
            }
        };
        let v_bar = 0.5 * (v + v_next);
        let drift =
            (self.mu - 0.5 * v_bar) * dt + self.rho / xi * (v_next - v - k * (th - v_bar) * dt);
        let vol = ((1.0 - self.rho * self.rho).max(0.0) * v_bar * dt).sqrt();
        let s_next = (s.max(1e-16) * (drift + vol * z_s).exp()).max(1e-16);
        (s_next, v_next.max(0.0))
    }
}

impl Sde for Heston {
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
        let rho = self.rho;
        let srho = (1.0 - rho * rho).max(0.0).sqrt();
        // row 0: [√v S, 0] after mixing: [√v S, 0] * chol([[1,ρ],[ρ,1]])
        // We put the Cholesky of the correlation into σ:
        // σ = [[ √v S, 0 ], [ ξ √v ρ, ξ √v √(1-ρ²) ]]
        out[0] = sv * s;
        out[1] = 0.0;
        out[2] = self.xi * sv * rho;
        out[3] = self.xi * sv * srho;
    }
}

impl ParametricSde for Heston {
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

/// Merton jump-diffusion: `dX = μ X dt + σ X dW + X dJ`, `J` compound Poisson
/// of log-jumps `N(α, δ²)` (the increment applied is `e^Z − 1` when
/// `multiplicative_jumps` is combined with `γ = X` and raw `Z` increments —
/// here we use additive `γ dJ` with `γ = X` and `dJ = e^Z − 1` sampled in
/// the Lévy measure via a transformed law).
#[derive(Clone, Debug)]
pub struct MertonJumpDiffusion {
    pub mu: f64,
    pub sigma: f64,
    pub intensity: f64,
    pub jump_mu: f64,
    pub jump_sigma: f64,
    levy: LevyMeasure,
    p: [f64; 5],
}

impl MertonJumpDiffusion {
    pub fn new(mu: f64, sigma: f64, intensity: f64, jump_mu: f64, jump_sigma: f64) -> Result<Self> {
        require_pos("sigma", sigma)?;
        if intensity < 0.0 {
            return Err(Error::param("intensity must be non-negative"));
        }
        require_pos("jump_sigma", jump_sigma)?;
        Ok(Self {
            mu,
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
            p: [mu, sigma, intensity, jump_mu, jump_sigma],
        })
    }
}

impl Sde for MertonJumpDiffusion {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.mu * x[0];
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma * x[0];
    }
    fn jump_coeff(&self, _t: f64, x: &[f64], out: &mut [f64]) -> bool {
        out[0] = x[0];
        true
    }
    fn levy(&self) -> Option<&LevyMeasure> {
        Some(&self.levy)
    }
    fn multiplicative_jumps(&self) -> bool {
        // increment is log-jump Z; apply X * (e^Z − 1) = X * (exp(dJ) − 1)
        true
    }
}

impl ParametricSde for MertonJumpDiffusion {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["mu", "sigma", "intensity", "jump_mu", "jump_sigma"]
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

/// Kou double-exponential jump-diffusion.
#[derive(Clone, Debug)]
pub struct KouJumpDiffusion {
    pub mu: f64,
    pub sigma: f64,
    pub intensity: f64,
    pub p: f64,
    pub eta_plus: f64,
    pub eta_minus: f64,
    levy: LevyMeasure,
    params: [f64; 6],
}

impl KouJumpDiffusion {
    pub fn new(
        mu: f64,
        sigma: f64,
        intensity: f64,
        p: f64,
        eta_plus: f64,
        eta_minus: f64,
    ) -> Result<Self> {
        require_pos("sigma", sigma)?;
        if intensity < 0.0 || !(0.0..=1.0).contains(&p) {
            return Err(Error::param("invalid Kou intensity or p"));
        }
        require_pos("eta_plus", eta_plus)?;
        require_pos("eta_minus", eta_minus)?;
        Ok(Self {
            mu,
            sigma,
            intensity,
            p,
            eta_plus,
            eta_minus,
            levy: LevyMeasure::CompoundPoisson {
                intensity,
                law: JumpLaw::DoubleExponential {
                    p,
                    eta_plus,
                    eta_minus,
                },
            },
            params: [mu, sigma, intensity, p, eta_plus, eta_minus],
        })
    }
}

impl Sde for KouJumpDiffusion {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.mu * x[0];
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma * x[0];
    }
    fn jump_coeff(&self, _t: f64, x: &[f64], out: &mut [f64]) -> bool {
        out[0] = x[0];
        true
    }
    fn levy(&self) -> Option<&LevyMeasure> {
        Some(&self.levy)
    }
    fn multiplicative_jumps(&self) -> bool {
        true
    }
}

impl ParametricSde for KouJumpDiffusion {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["mu", "sigma", "intensity", "p", "eta_plus", "eta_minus"]
    }
    fn params(&self) -> &[f64] {
        &self.params
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

/// Geometric Brownian motion driven by fractional Brownian motion
/// (`dX = μ X dt + σ X dBᴴ`).
#[derive(Clone, Debug)]
pub struct FractionalGbm {
    pub mu: f64,
    pub sigma: f64,
    pub hurst: f64,
    p: [f64; 3],
}

impl FractionalGbm {
    pub fn new(mu: f64, sigma: f64, hurst: f64) -> Result<Self> {
        require_pos("sigma", sigma)?;
        if !(0.0 < hurst && hurst < 1.0) {
            return Err(Error::param("Hurst must lie in (0, 1)"));
        }
        Ok(Self {
            mu,
            sigma,
            hurst,
            p: [mu, sigma, hurst],
        })
    }
}

impl Sde for FractionalGbm {
    fn dim(&self) -> usize {
        1
    }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.mu * x[0];
    }
    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        out[0] = self.sigma * x[0];
    }
    fn hurst(&self) -> Option<f64> {
        Some(self.hurst)
    }
}

impl ParametricSde for FractionalGbm {
    type Frozen = Self;
    fn param_names(&self) -> &[&'static str] {
        &["mu", "sigma", "hurst"]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cir_feller() {
        let c = Cir::new(2.0, 1.0, 0.5).unwrap();
        assert!(c.feller_holds());
        let bad = Cir::new(0.1, 0.1, 2.0).unwrap();
        assert!(!bad.feller_holds());
        let mut rng = crate::rng::seed_rng(3);
        let mut x = 1.0;
        for _ in 0..80 {
            x = c.sample_exact(x, 0.05, &mut rng).unwrap();
            assert!(x >= 0.0 && x.is_finite());
        }
        let mut s = 0.0;
        let n = 4_000;
        let dt = 0.5;
        let e = (-2.0_f64 * dt).exp();
        let expect = 0.4 * e + (1.0 - e);
        for _ in 0..n {
            s += c.sample_exact(0.4, dt, &mut rng).unwrap();
        }
        assert!(
            (s / n as f64 - expect).abs() < 0.06,
            "CIR exact mean {} vs {expect}",
            s / n as f64
        );
        let h = crate::models::Heston::new(0.05, 2.0, 0.04, 0.3, -0.5).unwrap();
        let (s, v) = h.qe_step(100.0, 0.04, 0.01, 0.2, -0.1);
        assert!(s > 0.0 && v >= 0.0);
    }
}
