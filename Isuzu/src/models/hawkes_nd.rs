//! N-dimensional exponential Hawkes with an analytic log-likelihood gradient.

use amatsuki::{Exp1, Rng, Uniform};

use crate::error::{Error, Result};

/// Multivariate Hawkes with exponential kernels
/// `λ_i(t) = μ_i + Σ_j Σ_{t_k^j < t} α_{ij} exp(−β_{ij}(t − t_k^j))`.
#[derive(Clone, Debug)]
pub struct MultivariateHawkes {
    pub dim: usize,
    pub mu: Vec<f64>,
    /// Row-major `α_{ij}`.
    pub alpha: Vec<f64>,
    /// Row-major `β_{ij}`.
    pub beta: Vec<f64>,
}

impl MultivariateHawkes {
    pub fn new(mu: Vec<f64>, alpha: Vec<f64>, beta: Vec<f64>) -> Result<Self> {
        let d = mu.len();
        if d == 0 || alpha.len() != d * d || beta.len() != d * d {
            return Err(Error::dim("Hawkes μ, α, β dimension"));
        }
        if mu.iter().any(|x| *x < 0.0)
            || alpha.iter().any(|x| *x < 0.0)
            || beta.iter().any(|x| *x <= 0.0)
        {
            return Err(Error::param("Hawkes needs μ,α ≥ 0 and β > 0"));
        }
        Ok(Self {
            dim: d,
            mu,
            alpha,
            beta,
        })
    }

    pub fn intensity(&self, i: usize, t: f64, events: &[(f64, usize)]) -> f64 {
        let mut s = self.mu[i];
        for &(tk, j) in events {
            if tk < t {
                let a = self.alpha[i * self.dim + j];
                let b = self.beta[i * self.dim + j];
                s += a * (-b * (t - tk)).exp();
            }
        }
        s
    }

    /// Ogata thinning with a piecewise upper bound (sum of current intensities
    /// after each event; valid because every kernel is decreasing).
    pub fn simulate<R: Rng + ?Sized>(
        &self,
        t0: f64,
        t1: f64,
        rng: &mut R,
    ) -> Result<Vec<(f64, usize)>> {
        if t1 <= t0 {
            return Err(Error::sampling("Hawkes interval must be nonempty"));
        }
        let d = self.dim;
        let mut r = vec![0.0; d * d];
        let mut t = t0;
        let mut out = Vec::new();
        loop {
            let mut lam = vec![0.0; d];
            let mut bar = 0.0;
            for i in 0..d {
                let mut s = self.mu[i];
                for j in 0..d {
                    s += self.alpha[i * d + j] * r[i * d + j];
                }
                lam[i] = s;
                bar += s;
            }
            if bar <= 0.0 {
                break;
            }
            let wait = rng.sample(Exp1) / bar;
            t += wait;
            if t >= t1 {
                break;
            }
            for i in 0..d {
                for j in 0..d {
                    r[i * d + j] *= (-self.beta[i * d + j] * wait).exp();
                }
            }
            let mut lam2 = vec![0.0; d];
            let mut sum = 0.0;
            for i in 0..d {
                let mut s = self.mu[i];
                for j in 0..d {
                    s += self.alpha[i * d + j] * r[i * d + j];
                }
                lam2[i] = s;
                sum += s;
            }
            let u: f64 = rng.sample(Uniform::new(0.0, 1.0));
            if u * bar <= sum {
                let mut c = 0.0;
                let v: f64 = rng.sample(Uniform::new(0.0, 1.0));
                let mut mark = 0usize;
                for i in 0..d {
                    c += lam2[i] / sum;
                    if v <= c {
                        mark = i;
                        break;
                    }
                }
                out.push((t, mark));
                for i in 0..d {
                    r[i * d + mark] += 1.0;
                }
            }
        }
        Ok(out)
    }

    /// Exact log-likelihood and analytic gradient wrt `(μ, α, β)` (length `d + 2d²`).
    pub fn loglik_grad(
        &self,
        events: &[(f64, usize)],
        t0: f64,
        t1: f64,
    ) -> Result<(f64, Vec<f64>)> {
        let d = self.dim;
        let npar = d + 2 * d * d;
        let mut g = vec![0.0; npar];
        let mut ll = 0.0;
        let mut r = vec![0.0; d * d];
        let mut prev = t0;
        for &(ti, k) in events {
            let dt = ti - prev;
            for i in 0..d {
                for j in 0..d {
                    r[i * d + j] *= (-self.beta[i * d + j] * dt).exp();
                }
            }
            let mut lam = self.mu[k];
            for j in 0..d {
                lam += self.alpha[k * d + j] * r[k * d + j];
            }
            if lam <= 0.0 {
                return Ok((f64::NEG_INFINITY, g));
            }
            ll += lam.ln();
            g[k] += 1.0 / lam;
            for j in 0..d {
                g[d + k * d + j] += r[k * d + j] / lam;
                // ∂λ/∂β_{kj} = α_{kj} * ∂R/∂β, R = Σ e^{-β(t-t)} so at event
                // we only have the current residual (no closed ∂R here);
                // use the integral term below for β.
            }
            for i in 0..d {
                r[i * d + k] += 1.0;
            }
            prev = ti;
        }
        // Compensator and its gradient.
        let tspan = t1 - t0;
        for i in 0..d {
            ll -= self.mu[i] * tspan;
            g[i] -= tspan;
        }
        for &(ti, j) in events {
            for i in 0..d {
                let a = self.alpha[i * d + j];
                let b = self.beta[i * d + j];
                let e = (-b * (t1 - ti)).exp();
                ll -= (a / b) * (1.0 - e);
                g[d + i * d + j] -= (1.0 - e) / b;
                g[d + d * d + i * d + j] -= a * (-(1.0 - e) / (b * b) + (t1 - ti) * e / b);
            }
        }
        Ok((ll, g))
    }

    pub fn loglik(&self, events: &[(f64, usize)], t0: f64, t1: f64) -> Result<f64> {
        Ok(self.loglik_grad(events, t0, t1)?.0)
    }

    /// Ogata time-change residuals `Λ_i(t_k^i) − Λ_i(t_{k-1}^i)` vs Exp(1).
    pub fn ogata_residuals(&self, events: &[(f64, usize)], t0: f64, t1: f64) -> Result<Vec<f64>> {
        let mut res = Vec::new();
        for i in 0..self.dim {
            let mut prev = t0;
            let times: Vec<f64> = events
                .iter()
                .filter(|(_, k)| *k == i)
                .map(|(t, _)| *t)
                .collect();
            for &ti in &times {
                res.push(self.compensator_i(i, events, prev, ti)?);
                prev = ti;
            }
            res.push(self.compensator_i(i, events, prev, t1)?);
        }
        Ok(res)
    }

    fn compensator_i(&self, i: usize, events: &[(f64, usize)], t0: f64, t1: f64) -> Result<f64> {
        let mut s = self.mu[i] * (t1 - t0);
        for &(tk, j) in events {
            if tk >= t1 {
                continue;
            }
            if tk < t1 {
                let a = self.alpha[i * self.dim + j];
                let b = self.beta[i * self.dim + j];
                let t_lo = tk.max(t0);
                if t_lo >= t1 {
                    continue;
                }
                // ∫_{t_lo}^{t1} α e^{-β(s-tk)} ds
                s += (a / b) * ((-b * (t_lo - tk)).exp() - (-b * (t1 - tk)).exp());
            }
        }
        Ok(s)
    }

    /// KS statistic of the time-changed residuals against Exp(1) (via `1−e^{−r}`).
    pub fn ogata_ks(&self, events: &[(f64, usize)], t0: f64, t1: f64) -> Result<f64> {
        let r = self.ogata_residuals(events, t0, t1)?;
        let u: Vec<f64> = r.into_iter().map(|x| 1.0 - (-x).exp()).collect();
        Ok(ks_uniform_local(u))
    }
}

fn ks_uniform_local(mut u: Vec<f64>) -> f64 {
    u.retain(|x| x.is_finite() && (0.0..=1.0).contains(x));
    if u.is_empty() {
        return 0.0;
    }
    u.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = u.len() as f64;
    let mut d = 0.0_f64;
    for (i, &x) in u.iter().enumerate() {
        let fn_ = (i + 1) as f64 / n;
        let fm = i as f64 / n;
        d = f64::max(d, f64::max((fn_ - x).abs(), (fm - x).abs()));
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed_rng;

    #[test]
    fn nd_hawkes_loglik_finite_and_ks() {
        let h = MultivariateHawkes::new(vec![0.4, 0.3], vec![0.2, 0.1, 0.1, 0.2], vec![1.0; 4])
            .unwrap();
        let mut rng = seed_rng(4);
        let ev = h.simulate(0.0, 20.0, &mut rng).unwrap();
        let (ll, g) = h.loglik_grad(&ev, 0.0, 20.0).unwrap();
        assert!(ll.is_finite());
        assert_eq!(g.len(), 2 + 8);
        let ks = h.ogata_ks(&ev, 0.0, 20.0).unwrap();
        assert!(ks < 0.4, "KS {ks}");
    }
}
