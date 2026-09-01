//! Extra point processes: multivariate / power-law Hawkes, renewal, Cox,
//! self-correcting, marked Hawkes, CARMA-Hawkes.

use amatsuki::{Distribution, Exp, Exp1, Gamma, Rng, Uniform};

use crate::error::{Error, Result};
use crate::linalg::is_hurwitz;
use crate::models::util::{require_nonneg, require_pos};
use crate::noise::poisson_arrivals;
use crate::optimize::{nelder_mead, OptOptions};

/// Two-dimensional exponential Hawkes (mutually exciting).
///
/// `λᵢ(t) = μᵢ + Σⱼ Σ_{tₖʲ < t} αᵢⱼ exp(−βᵢⱼ (t − tₖʲ))`.
#[derive(Clone, Debug)]
pub struct MultivariateHawkes2 {
    pub mu: [f64; 2],
    pub alpha: [[f64; 2]; 2],
    pub beta: [[f64; 2]; 2],
}

impl MultivariateHawkes2 {
    pub fn new(mu: [f64; 2], alpha: [[f64; 2]; 2], beta: [[f64; 2]; 2]) -> Result<Self> {
        for &m in &mu {
            require_nonneg("mu", m)?;
        }
        for i in 0..2 {
            for j in 0..2 {
                require_nonneg("alpha", alpha[i][j])?;
                require_pos("beta", beta[i][j])?;
            }
        }
        Ok(Self { mu, alpha, beta })
    }

    pub fn intensity(&self, t: f64, arrivals: &[Vec<f64>]) -> [f64; 2] {
        let mut lam = self.mu;
        for i in 0..2 {
            for (j, times) in arrivals.iter().enumerate().take(2) {
                for &tk in times {
                    if tk < t {
                        lam[i] += self.alpha[i][j] * (-self.beta[i][j] * (t - tk)).exp();
                    }
                }
            }
        }
        lam
    }

    pub fn simulate<R: Rng + ?Sized>(
        &self,
        t0: f64,
        t1: f64,
        rng: &mut R,
    ) -> Result<[Vec<f64>; 2]> {
        if t1 <= t0 {
            return Err(Error::sampling("empty Hawkes window"));
        }
        let mut t = t0;
        let mut arr = [Vec::new(), Vec::new()];
        // recursive states R_{ij} = Σ exp(−βᵢⱼ(t−tₖʲ))
        let mut r = [[0.0; 2]; 2];
        loop {
            let lam0 = self.mu[0] + self.alpha[0][0] * r[0][0] + self.alpha[0][1] * r[0][1];
            let lam1 = self.mu[1] + self.alpha[1][0] * r[1][0] + self.alpha[1][1] * r[1][1];
            let bar = (lam0 + lam1).max(0.0);
            if bar <= 0.0 {
                break;
            }
            let e: f64 = rng.sample(Exp1);
            let dt = e / bar;
            t += dt;
            if t >= t1 {
                break;
            }
            for i in 0..2 {
                for j in 0..2 {
                    r[i][j] *= (-self.beta[i][j] * dt).exp();
                }
            }
            let l0 = self.mu[0] + self.alpha[0][0] * r[0][0] + self.alpha[0][1] * r[0][1];
            let l1 = self.mu[1] + self.alpha[1][0] * r[1][0] + self.alpha[1][1] * r[1][1];
            let u: f64 = rng.sample(Uniform::new(0.0, 1.0));
            if u * bar <= l0.max(0.0) {
                arr[0].push(t);
                r[0][0] += 1.0;
                r[1][0] += 1.0;
            } else if u * bar <= (l0.max(0.0) + l1.max(0.0)) {
                arr[1].push(t);
                r[0][1] += 1.0;
                r[1][1] += 1.0;
            }
        }
        Ok(arr)
    }
}

/// Power-law / Omori Hawkes: `λ(t) = μ + Σ α (c + t − tᵢ)^{-p}`.
#[derive(Clone, Debug)]
pub struct PowerLawHawkes {
    pub mu: f64,
    pub alpha: f64,
    pub c: f64,
    pub p: f64,
}

impl PowerLawHawkes {
    pub fn new(mu: f64, alpha: f64, c: f64, p: f64) -> Result<Self> {
        require_nonneg("mu", mu)?;
        require_nonneg("alpha", alpha)?;
        require_pos("c", c)?;
        if p <= 1.0 {
            return Err(Error::param(
                "power-law Hawkes needs p > 1 for integrability",
            ));
        }
        Ok(Self { mu, alpha, c, p })
    }

    pub fn intensity(&self, t: f64, arrivals: &[f64]) -> f64 {
        let mut s = self.mu;
        for &ti in arrivals {
            if ti < t {
                s += self.alpha * (self.c + t - ti).powf(-self.p);
            }
        }
        s
    }

    pub fn simulate<R: Rng + ?Sized>(&self, t0: f64, t1: f64, rng: &mut R) -> Result<Vec<f64>> {
        let mut t = t0;
        let mut arr = Vec::new();
        while t < t1 {
            let lam = self.intensity(t, &arr);
            // bound: each event contributes at most α c^{-p}
            let bar = self.mu + self.alpha * (arr.len() as f64) * self.c.powf(-self.p) + 1e-12;
            let e: f64 = rng.sample(Exp1);
            t += e / bar.max(lam);
            if t >= t1 {
                break;
            }
            let u: f64 = rng.sample(Uniform::new(0.0, 1.0));
            if u * bar.max(lam) <= self.intensity(t, &arr) {
                arr.push(t);
            }
        }
        Ok(arr)
    }
}

/// Isham–Westcott self-correcting process: `λ(t) = exp(μ + β t − α N_{t−})`.
#[derive(Clone, Debug)]
pub struct SelfCorrecting {
    pub mu: f64,
    pub beta: f64,
    pub alpha: f64,
}

impl SelfCorrecting {
    pub fn new(mu: f64, beta: f64, alpha: f64) -> Result<Self> {
        require_pos("beta", beta)?;
        require_pos("alpha", alpha)?;
        Ok(Self { mu, beta, alpha })
    }

    pub fn intensity(&self, t: f64, n: usize) -> f64 {
        (self.mu + self.beta * t - self.alpha * n as f64).exp()
    }

    pub fn simulate<R: Rng + ?Sized>(&self, t0: f64, t1: f64, rng: &mut R) -> Result<Vec<f64>> {
        let mut t = t0;
        let mut arr = Vec::new();
        while t < t1 {
            let n = arr.len();
            // λ increases in t; bound on a short horizon
            let bar = self.intensity(t1, n);
            let e: f64 = rng.sample(Exp1);
            t += e / bar.max(1e-12);
            if t >= t1 {
                break;
            }
            let u: f64 = rng.sample(Uniform::new(0.0, 1.0));
            if u * bar <= self.intensity(t, n) {
                arr.push(t);
            }
        }
        Ok(arr)
    }
}

/// Weibull renewal process (interarrivals `Weibull(k, λ)`).
#[derive(Clone, Debug)]
pub struct WeibullRenewal {
    pub shape: f64,
    pub scale: f64,
}

impl WeibullRenewal {
    pub fn new(shape: f64, scale: f64) -> Result<Self> {
        require_pos("shape", shape)?;
        require_pos("scale", scale)?;
        Ok(Self { shape, scale })
    }

    pub fn simulate<R: Rng + ?Sized>(&self, t0: f64, t1: f64, rng: &mut R) -> Result<Vec<f64>> {
        let mut t = t0;
        let mut arr = Vec::new();
        let u = Uniform::new(0.0, 1.0);
        while t < t1 {
            let uni: f64 = rng.sample(u);
            let w = self.scale * (-uni.ln()).powf(1.0 / self.shape);
            t += w;
            if t >= t1 {
                break;
            }
            arr.push(t);
        }
        Ok(arr)
    }

    /// Log-likelihood on `[t0, t1]` including the right-censoring survival
    /// term from the last event (or `t0`) to `t1`.
    pub fn loglik(&self, arrivals: &[f64], t0: f64, t1: f64) -> Result<f64> {
        if t1 < t0 {
            return Err(Error::sampling("Weibull window must satisfy t1 ≥ t0"));
        }
        let mut prev = t0;
        let mut ll = 0.0;
        let k = self.shape;
        let lam = 1.0 / self.scale;
        for &ti in arrivals {
            if ti < t0 || ti > t1 {
                return Err(Error::sampling("arrival outside window"));
            }
            let x = ti - prev;
            if x <= 0.0 {
                return Err(Error::sampling("non-positive interarrival"));
            }
            ll += k.ln() + k * lam.ln() + (k - 1.0) * x.ln() - (lam * x).powf(k);
            prev = ti;
        }
        let rem = t1 - prev;
        if rem < 0.0 {
            return Err(Error::sampling("last arrival after t1"));
        }
        // log S(rem) = −(λ rem)^k
        ll += -(lam * rem).powf(k);
        Ok(ll)
    }

    /// Nelder–Mead MLE of `(shape, scale)` on `[t0, t1]`.
    pub fn mle(arrivals: &[f64], t0: f64, t1: f64, start: [f64; 2]) -> Result<(Self, f64)> {
        use crate::optimize::{nelder_mead, OptOptions};
        let f = |p: &[f64]| {
            if p[0] <= 0.0 || p[1] <= 0.0 {
                return 1e16;
            }
            match Self::new(p[0], p[1]).and_then(|w| w.loglik(arrivals, t0, t1)) {
                Ok(ll) if ll.is_finite() => -ll,
                _ => 1e16,
            }
        };
        let opt = nelder_mead(&f, &start, None, None, OptOptions::default())?;
        let w = Self::new(opt.x[0], opt.x[1])?;
        let ll = w.loglik(arrivals, t0, t1)?;
        Ok((w, ll))
    }
}

/// Gamma renewal (interarrivals `Gamma(shape, scale)`).
#[derive(Clone, Debug)]
pub struct GammaRenewal {
    pub shape: f64,
    pub scale: f64,
}

impl GammaRenewal {
    pub fn new(shape: f64, scale: f64) -> Result<Self> {
        require_pos("shape", shape)?;
        require_pos("scale", scale)?;
        Ok(Self { shape, scale })
    }

    pub fn simulate<R: Rng + ?Sized>(&self, t0: f64, t1: f64, rng: &mut R) -> Result<Vec<f64>> {
        let g = Gamma::new(self.shape, self.scale).map_err(|e| Error::param(e.to_string()))?;
        let mut t = t0;
        let mut arr = Vec::new();
        while t < t1 {
            t += g.sample(rng);
            if t >= t1 {
                break;
            }
            arr.push(t);
        }
        Ok(arr)
    }
}

/// Cox process with CIR intensity (doubly stochastic Poisson).
#[derive(Clone, Debug)]
pub struct CoxCir {
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    pub lambda0: f64,
}

impl CoxCir {
    pub fn new(kappa: f64, theta: f64, sigma: f64, lambda0: f64) -> Result<Self> {
        require_pos("kappa", kappa)?;
        require_pos("theta", theta)?;
        require_pos("sigma", sigma)?;
        require_nonneg("lambda0", lambda0)?;
        Ok(Self {
            kappa,
            theta,
            sigma,
            lambda0,
        })
    }

    pub fn simulate<R: Rng + ?Sized>(
        &self,
        t0: f64,
        t1: f64,
        n_grid: usize,
        rng: &mut R,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        use crate::models::Cir;
        use crate::sampling::Sampling;
        use crate::simulate::{simulate, SimConfig};
        let cir = Cir::new(self.kappa, self.theta, self.sigma)?;
        let samp = Sampling::regular(t0, t1, n_grid)?;
        let path = simulate(&cir, &samp, &[self.lambda0], rng, &SimConfig::default())?;
        let lam = path.as_univariate()?;
        let mut arr = Vec::new();
        for i in 0..n_grid {
            let dt = samp.delta(i);
            let mid = 0.5 * (lam[i].max(0.0) + lam[i + 1].max(0.0));
            for t in poisson_arrivals(samp.times()[i], samp.times()[i + 1], mid, rng)? {
                let _ = dt;
                arr.push(t);
            }
        }
        Ok((arr, lam))
    }
}

/// Marked exponential Hawkes: each event carries a mark `z` that scales α.
#[derive(Clone, Debug)]
pub struct MarkedHawkes {
    pub mu: f64,
    pub alpha: f64,
    pub beta: f64,
    pub mark_mean: f64,
}

impl MarkedHawkes {
    pub fn new(mu: f64, alpha: f64, beta: f64, mark_mean: f64) -> Result<Self> {
        require_nonneg("mu", mu)?;
        require_nonneg("alpha", alpha)?;
        require_pos("beta", beta)?;
        require_pos("mark_mean", mark_mean)?;
        Ok(Self {
            mu,
            alpha,
            beta,
            mark_mean,
        })
    }

    pub fn simulate<R: Rng + ?Sized>(
        &self,
        t0: f64,
        t1: f64,
        rng: &mut R,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut t = t0;
        let mut times = Vec::new();
        let mut marks = Vec::new();
        let mut r = 0.0;
        let exp = Exp::new(1.0 / self.mark_mean).map_err(|e| Error::param(e.to_string()))?;
        loop {
            let bar = self.mu + self.alpha * r;
            if bar <= 0.0 {
                break;
            }
            let e: f64 = rng.sample(Exp1);
            t += e / bar;
            if t >= t1 {
                break;
            }
            r *= (-self.beta * e / bar).exp();
            let lam = self.mu + self.alpha * r;
            let u: f64 = rng.sample(Uniform::new(0.0, 1.0));
            if u * bar <= lam {
                let z = exp.sample(rng);
                times.push(t);
                marks.push(z);
                r += z;
            }
        }
        Ok((times, marks))
    }
}

/// CARMA(p, q)–Hawkes counting process.
///
/// `λ(t) = μ + bᵀ X_{t−}`, `dX = A X dt + e dN`.
#[derive(Clone, Debug)]
pub struct CarmaHawkes {
    pub mu: f64,
    pub ar: Vec<f64>,
    pub ma: Vec<f64>,
}

impl CarmaHawkes {
    pub fn new(mu: f64, ar: Vec<f64>, ma: Vec<f64>) -> Result<Self> {
        require_nonneg("mu", mu)?;
        if ar.is_empty() || ma.is_empty() || ma.len() > ar.len() {
            return Err(Error::param("CARMA-Hawkes needs p ≥ 1, q < p"));
        }
        if ar.iter().any(|a| *a <= 0.0) || ma.iter().any(|b| *b < 0.0) {
            return Err(Error::param("CARMA-Hawkes AR > 0, MA ≥ 0"));
        }
        Ok(Self { mu, ar, ma })
    }

    pub fn p(&self) -> usize {
        self.ar.len()
    }

    fn companion(&self) -> faer::Mat<f64> {
        let p = self.p();
        let mut a = crate::linalg::mat_zeros(p, p);
        for i in 0..p.saturating_sub(1) {
            a[(i, i + 1)] = 1.0;
        }
        for j in 0..p {
            a[(p - 1, j)] = -self.ar[p - 1 - j];
        }
        a
    }

    pub fn is_stable(&self) -> bool {
        is_hurwitz(&self.companion())
    }

    fn intensity_raw(&self, x: &[f64]) -> f64 {
        let mut s = self.mu;
        for (i, &bi) in self.ma.iter().enumerate() {
            if i < x.len() {
                s += bi * x[i];
            }
        }
        s
    }

    fn observe(&self, x: &[f64]) -> f64 {
        self.intensity_raw(x).max(1e-16)
    }

    fn integrated_lambda(&self, a: &faer::Mat<f64>, x: &faer::Col<f64>, dt: f64) -> Result<f64> {
        // ∫_0^Δ (μ + bᵀ e^{As} x) ds = μΔ + bᵀ ∫ e^{As} x ds
        let u = crate::linalg::integrate_expm(a, x, dt)?;
        let mut s = self.mu * dt;
        for (i, &bi) in self.ma.iter().enumerate() {
            if i < u.nrows() {
                s += bi * u[i];
            }
        }
        Ok(s)
    }

    fn local_intensity_bound(&self, a: &faer::Mat<f64>, x: &faer::Col<f64>, horizon: f64) -> f64 {
        // |bᵀ e^{As} x| ≤ ‖b‖₁ ‖x‖₂ exp(‖A‖_F s) for s ≤ horizon.
        let bx: f64 = self.ma.iter().map(|b| b.abs()).sum();
        let mut xn = 0.0;
        for i in 0..x.nrows() {
            xn += x[i] * x[i];
        }
        xn = xn.sqrt();
        let mut an = 0.0;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                an += a[(i, j)] * a[(i, j)];
            }
        }
        an = an.sqrt();
        (self.mu + bx * xn * (an * horizon).exp()).max(1e-16)
    }

    /// Ogata thinning. For `p = 1` the intensity is monotone between jumps
    /// so the current value is a valid bound. For `p ≥ 2` the kernel is not
    /// monotone; a local dominating bound on a short horizon is used.
    pub fn simulate<R: Rng + ?Sized>(&self, t0: f64, t1: f64, rng: &mut R) -> Result<Vec<f64>> {
        use crate::linalg::{col_to_vec, col_zeros, expm};
        use faer::Scale;
        if t1 <= t0 {
            return Err(Error::sampling("CARMA-Hawkes interval must be non-empty"));
        }
        let p = self.p();
        let a = self.companion();
        let mut t = t0;
        let mut x = col_zeros(p);
        let mut arr = Vec::new();
        const MAX_EVENTS: usize = 1_000_000;
        let mut an = 0.0;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                an += a[(i, j)] * a[(i, j)];
            }
        }
        an = an.sqrt();
        let horizon = (1.0 / (an + 1.0)).clamp(1e-4, 0.5);
        while t < t1 {
            let bar = if p == 1 {
                self.observe(&col_to_vec(&x))
            } else {
                self.local_intensity_bound(&a, &x, horizon)
            };
            let e: f64 = rng.sample(Exp1);
            let wait = e / bar;
            if p > 1 && wait > horizon {
                let step = (t1 - t).min(horizon);
                let f = expm(&(Scale(step) * &a));
                x = &f * &x;
                t += step;
                continue;
            }
            let t_cand = t + wait;
            if t_cand >= t1 {
                break;
            }
            let f = expm(&(Scale(wait) * &a));
            x = &f * &x;
            let lam2 = self.intensity_raw(&col_to_vec(&x)).max(0.0);
            let u: f64 = rng.sample(Uniform::new(0.0, 1.0));
            if u * bar <= lam2 {
                arr.push(t_cand);
                x[p - 1] += 1.0;
            }
            t = t_cand;
            if arr.len() > MAX_EVENTS {
                return Err(Error::sim("CARMA-Hawkes event cap exceeded"));
            }
        }
        Ok(arr)
    }

    /// Exact log-likelihood: compensator `μΔ + bᵀ ∫ e^{As} x ds` via the
    /// affine block exponential (no trapezoid / `nsub` time-unit dependence).
    pub fn loglik(&self, arrivals: &[f64], t0: f64, t1: f64) -> Result<f64> {
        use crate::linalg::{col_to_vec, col_zeros, expm};
        use faer::Scale;
        let p = self.p();
        let a = self.companion();
        let mut x = col_zeros(p);
        let mut prev = t0;
        let mut ll = 0.0;
        let mut integral = 0.0;
        for &ti in arrivals {
            if ti < prev {
                return Err(Error::sampling("arrivals must be sorted"));
            }
            let dt = ti - prev;
            if dt > 0.0 {
                integral += self.integrated_lambda(&a, &x, dt)?;
                let f = expm(&(Scale(dt) * &a));
                x = &f * &x;
            }
            let lam = self.intensity_raw(&col_to_vec(&x));
            if lam <= 0.0 {
                return Ok(f64::NEG_INFINITY);
            }
            ll += lam.ln();
            x[p - 1] += 1.0;
            prev = ti;
        }
        let dt = t1 - prev;
        if dt > 0.0 {
            integral += self.integrated_lambda(&a, &x, dt)?;
        }
        Ok(ll - integral)
    }

    /// Dedicated QMLE / MLE for `(μ, a…, b…)`.
    pub fn mle(
        arrivals: &[f64],
        t0: f64,
        t1: f64,
        p: usize,
        q: usize,
        start: &[f64],
    ) -> Result<(Self, f64)> {
        if start.len() != 1 + p + (q + 1) {
            return Err(Error::param("start length must be 1+p+(q+1)"));
        }
        let obj = |theta: &[f64]| {
            let mu = theta[0];
            let ar = theta[1..=p].to_vec();
            let ma = theta[p + 1..].to_vec();
            match Self::new(mu, ar, ma).and_then(|m| m.loglik(arrivals, t0, t1)) {
                Ok(ll) if ll.is_finite() => -ll,
                _ => 1e16,
            }
        };
        let opt = nelder_mead(&obj, start, None, None, OptOptions::default())?;
        let mu = opt.x[0];
        let ar = opt.x[1..=p].to_vec();
        let ma = opt.x[p + 1..].to_vec();
        let model = Self::new(mu, ar, ma)?;
        let ll = model.loglik(arrivals, t0, t1)?;
        Ok((model, ll))
    }
}

/// Inhibiting exponential Hawkes (`α < 0`) with a positive floor.
#[derive(Clone, Debug)]
pub struct InhibitingHawkes {
    pub mu: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl InhibitingHawkes {
    pub fn new(mu: f64, alpha: f64, beta: f64) -> Result<Self> {
        require_pos("mu", mu)?;
        if alpha >= 0.0 {
            return Err(Error::param("inhibiting Hawkes needs α < 0"));
        }
        require_pos("beta", beta)?;
        Ok(Self { mu, alpha, beta })
    }

    pub fn intensity(&self, r: f64) -> f64 {
        (self.mu + self.alpha * r).max(1e-12)
    }

    pub fn simulate<R: Rng + ?Sized>(&self, t0: f64, t1: f64, rng: &mut R) -> Result<Vec<f64>> {
        let mut t = t0;
        let mut r = 0.0;
        let mut arr = Vec::new();
        while t < t1 {
            let bar = self.mu; // μ is an upper bound when α < 0
            let e: f64 = rng.sample(Exp1);
            t += e / bar;
            if t >= t1 {
                break;
            }
            r *= (-self.beta * e / bar).exp();
            let lam = self.intensity(r);
            let u: f64 = rng.sample(Uniform::new(0.0, 1.0));
            if u * bar <= lam {
                arr.push(t);
                r += 1.0;
            }
        }
        Ok(arr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed_rng;

    #[test]
    fn carma_hawkes11_stable() {
        let m = CarmaHawkes::new(0.4, vec![1.2], vec![0.5]).unwrap();
        assert!(m.is_stable());
        let mut rng = seed_rng(2);
        let arr = m.simulate(0.0, 5.0, &mut rng).unwrap();
        assert!(arr.len() < 200);
    }

    #[test]
    fn weibull_mean_interarrival() {
        let w = WeibullRenewal::new(1.0, 0.5).unwrap(); // Exp(2)
        let mut rng = seed_rng(3);
        let arr = w.simulate(0.0, 20.0, &mut rng).unwrap();
        assert!(arr.len() > 20);
    }
}
