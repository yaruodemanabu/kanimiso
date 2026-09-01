//! Compensated Poisson processes and Merton jump-diffusion prices.

use amatsuki::{Exp1, Rng, StandardNormal};

use crate::error::{Error, Result};
use crate::finance::black_scholes::{call as bs_call, BlackScholesMarket};
use crate::finance::monte_carlo::{MonteCarloEstimate, OnlineMoments};
use crate::finance::pde::solve_tridiagonal;
use crate::finance::special::{norm_cdf, norm_pdf};
use crate::models::MertonJumpDiffusion;
use crate::sampling::Sampling;
use crate::simulate::{simulate, SimConfig};

/// Homogeneous Poisson counting path on a grid, plus the compensator `λt`.
#[derive(Clone, Debug)]
pub struct CompensatedPoisson {
    pub times: Vec<f64>,
    pub n: Vec<f64>,
    pub compensator: Vec<f64>,
    pub martingale: Vec<f64>,
}

/// Simulate `N_t − λ t` on `times`.
pub fn compensated_poisson<R: Rng + ?Sized>(
    times: &[f64],
    intensity: f64,
    rng: &mut R,
) -> Result<CompensatedPoisson> {
    if times.len() < 2 || intensity < 0.0 {
        return Err(Error::param("compensated Poisson needs a grid and λ≥0"));
    }
    let mut arrivals = Vec::new();
    let t0 = times[0];
    let t1 = *times.last().unwrap();
    let mut t = t0;
    if intensity > 0.0 {
        loop {
            t += rng.sample(Exp1) / intensity;
            if t >= t1 {
                break;
            }
            arrivals.push(t);
        }
    }
    let mut n = Vec::with_capacity(times.len());
    let mut k = 0usize;
    for &s in times {
        while k < arrivals.len() && arrivals[k] <= s {
            k += 1;
        }
        n.push(k as f64);
    }
    let compensator: Vec<f64> = times.iter().map(|s| intensity * (s - t0)).collect();
    let martingale: Vec<f64> = n
        .iter()
        .zip(compensator.iter())
        .map(|(a, b)| a - b)
        .collect();
    Ok(CompensatedPoisson {
        times: times.to_vec(),
        n,
        compensator,
        martingale,
    })
}

/// Risk-neutral drift correction for a Merton jump-diffusion:
/// `μ = r − q − λ k` with `k = E[e^Z − 1] = exp(α + δ²/2) − 1`.
pub fn merton_compensator(intensity: f64, jump_mu: f64, jump_sigma: f64) -> f64 {
    let k = (jump_mu + 0.5 * jump_sigma * jump_sigma).exp() - 1.0;
    intensity * k
}

/// Merton (1976) series price of a European call.
pub fn merton_call(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    time: f64,
    intensity: f64,
    jump_mu: f64,
    jump_sigma: f64,
    n_terms: usize,
) -> Result<f64> {
    if n_terms == 0 {
        return Err(Error::param("Merton series needs at least one term"));
    }
    let kbar = (jump_mu + 0.5 * jump_sigma * jump_sigma).exp() - 1.0;
    let lam_p = intensity * (1.0 + kbar);
    let mut price = 0.0;
    let mut w = (-lam_p * time).exp();
    for n in 0..n_terms {
        let sigma_n = (vol * vol + (n as f64) * jump_sigma * jump_sigma / time).sqrt();
        let r_n = rate - intensity * kbar + (n as f64) * (1.0 + kbar).ln() / time;
        let mkt = BlackScholesMarket::new(spot, r_n, dividend, sigma_n, time)?;
        price += w * bs_call(&mkt, strike)?.price;
        w *= lam_p * time / (n as f64 + 1.0);
    }
    Ok(price)
}

/// Kou (2002) European call via the truncated double-exponential series
/// (Cai–Kou / original transform; here the first-term BS + jump mixture).
///
/// This is the Poisson-mixture form that reduces to Merton when the jump
/// law is replaced by its first two moments — used as a benchmark, not a
/// full PIDE solver.
pub fn kou_call_mixture(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    intensity: f64,
    p: f64,
    eta_plus: f64,
    eta_minus: f64,
    n_terms: usize,
) -> Result<f64> {
    if !(0.0..=1.0).contains(&p) || eta_plus <= 1.0 || eta_minus <= 0.0 {
        return Err(Error::param("Kou mixture needs p in [0,1], η+>1, η−>0"));
    }
    let kbar = p * eta_plus / (eta_plus - 1.0) + (1.0 - p) * eta_minus / (eta_minus + 1.0) - 1.0;
    let mut price = 0.0;
    let mut w = (-intensity * time).exp();
    for n in 0..n_terms {
        // Moment-matched Gaussian jump of n i.i.d. double-exponentials.
        let mean = n as f64 * (p / eta_plus - (1.0 - p) / eta_minus);
        let var = n as f64
            * (p * 2.0 / (eta_plus * eta_plus) + (1.0 - p) * 2.0 / (eta_minus * eta_minus));
        let sigma_n = (vol * vol + var / time).sqrt();
        let r_n = rate - intensity * kbar + mean / time;
        let mkt = BlackScholesMarket::new(spot, r_n, 0.0, sigma_n, time)?;
        price += w * bs_call(&mkt, strike)?.price;
        w *= intensity * time / (n as f64 + 1.0);
        let _ = norm_cdf;
    }
    Ok(price)
}

/// Sample a compound-Poisson increment over `dt` (number of jumps × marks).
pub fn compound_poisson_increment<R, F>(
    intensity: f64,
    dt: f64,
    mut mark: F,
    rng: &mut R,
) -> Result<f64>
where
    R: Rng + ?Sized,
    F: FnMut(&mut R) -> f64,
{
    if intensity < 0.0 || dt < 0.0 {
        return Err(Error::param("compound Poisson needs λ,Δt ≥ 0"));
    }
    let mut t = 0.0;
    let mut s = 0.0;
    if intensity == 0.0 {
        return Ok(0.0);
    }
    loop {
        t += rng.sample(Exp1) / intensity;
        if t > dt {
            break;
        }
        s += mark(rng);
    }
    Ok(s)
}

/// Standard normal mark, for tests.
pub fn standard_normal_mark<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    rng.sample(StandardNormal)
}

/// Esscher tilt of a Normal jump: intensity `λ ↦ λ E[e^{θY}]` and
/// `Y ∼ N(μ,δ²) ↦ N(μ+θδ², δ²)`.
pub fn esscher_normal_jump(
    intensity: f64,
    jump_mu: f64,
    jump_sigma: f64,
    theta: f64,
) -> Result<(f64, f64)> {
    if intensity < 0.0 || jump_sigma < 0.0 || !theta.is_finite() {
        return Err(Error::param("Esscher needs λ≥0, δ≥0, finite θ"));
    }
    let mgf = (theta * jump_mu + 0.5 * theta * theta * jump_sigma * jump_sigma).exp();
    Ok((intensity * mgf, jump_mu + theta * jump_sigma * jump_sigma))
}

/// Merton European call by Euler + compound-Poisson Monte Carlo.
///
/// Drift is the risk-neutral `r − q − λκ` so that `e^{-rT} S_T` is a
/// martingale after the jump compensator.
pub fn merton_call_mc<R: Rng + ?Sized>(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    time: f64,
    intensity: f64,
    jump_mu: f64,
    jump_sigma: f64,
    n_steps: usize,
    n_paths: usize,
    rng: &mut R,
) -> Result<MonteCarloEstimate> {
    if n_steps == 0 || n_paths == 0 || !(spot > 0.0 && time > 0.0) {
        return Err(Error::param("Merton MC needs positive paths / time / spot"));
    }
    let kappa = (jump_mu + 0.5 * jump_sigma * jump_sigma).exp() - 1.0;
    let mu = rate - dividend - intensity * kappa;
    let model = MertonJumpDiffusion::new(mu, vol, intensity, jump_mu, jump_sigma)?;
    let samp = Sampling::from_terminal(time, n_steps)?;
    let df = (-rate * time).exp();
    let mut acc = OnlineMoments::default();
    let cfg = SimConfig::default();
    for _ in 0..n_paths {
        let path = simulate(&model, &samp, &[spot], rng, &cfg)?;
        acc.push(df * (path.terminal()[0] - strike).max(0.0));
    }
    Ok(MonteCarloEstimate::from_moments(&acc, 1.96))
}

/// Merton European call by an IMEX PIDE on a uniform log-spot grid.
///
/// Local diffusion / reaction is implicit; the jump convolution
/// `λ ∫ V(x+y) φ(y) dy` is explicit (Cont–Voltchkova). The log-drift
/// uses the compensator `μ = r − q − σ²/2 − λκ`.
pub fn merton_call_pide(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    time: f64,
    intensity: f64,
    jump_mu: f64,
    jump_sigma: f64,
    n_space: usize,
    n_time: usize,
) -> Result<f64> {
    if n_space < 8 || n_time == 0 || !(spot > 0.0 && strike > 0.0 && time > 0.0) {
        return Err(Error::param("Merton PIDE grid invalid"));
    }
    if vol < 0.0 || intensity < 0.0 || jump_sigma < 0.0 {
        return Err(Error::param("Merton PIDE needs σ,λ,δ ≥ 0"));
    }
    let kappa = (jump_mu + 0.5 * jump_sigma * jump_sigma).exp() - 1.0;
    let mu = rate - dividend - 0.5 * vol * vol - intensity * kappa;
    let x0 = spot.ln();
    let width =
        (6.0 * vol * time.sqrt() + 6.0 * jump_sigma.max(0.05) + jump_mu.abs() + 1.5).max(2.5);
    let x_min = x0 - width;
    let x_max = x0 + width;
    let h = (x_max - x_min) / n_space as f64;
    let dt = time / n_time as f64;
    let xs: Vec<f64> = (0..=n_space).map(|i| x_min + i as f64 * h).collect();
    let mut v: Vec<f64> = xs.iter().map(|&x| (x.exp() - strike).max(0.0)).collect();

    let jmax = ((6.0 * jump_sigma.max(1e-8) + jump_mu.abs()) / h).ceil() as i32 + 1;
    let mut kern = Vec::new();
    let mut wsum = 0.0;
    for j in -jmax..=jmax {
        let y = j as f64 * h;
        let z = if jump_sigma > 1e-14 {
            (y - jump_mu) / jump_sigma
        } else if (y - jump_mu).abs() < 0.5 * h {
            0.0
        } else {
            continue;
        };
        let w = if jump_sigma > 1e-14 {
            norm_pdf(z) / jump_sigma * h
        } else {
            1.0
        };
        kern.push((j, w));
        wsum += w;
    }
    if wsum > 0.0 {
        for k in &mut kern {
            k.1 /= wsum;
        }
    }

    let n_int = n_space - 1;
    let sig2h = 0.5 * vol * vol / (h * h);
    let a_left = -mu / (2.0 * h) + sig2h;
    let a_diag = -2.0 * sig2h - (rate + intensity);
    let a_right = mu / (2.0 * h) + sig2h;
    let sub = -dt * a_left;
    let diag = 1.0 - dt * a_diag;
    let sup = -dt * a_right;

    for step in 0..n_time {
        let tau = time - (step + 1) as f64 * dt;
        let vlo = 0.0;
        let vhi = xs[n_space].exp() * (-dividend * tau).exp() - strike * (-rate * tau).exp();
        let mut conv = vec![0.0; n_space + 1];
        if intensity > 0.0 {
            for i in 0..=n_space {
                let mut s = 0.0;
                for &(j, w) in &kern {
                    let k = i as i32 + j;
                    let vk = if k < 0 {
                        0.0
                    } else if k as usize > n_space {
                        let x = x_min + k as f64 * h;
                        (x.exp() * (-dividend * (tau + dt)).exp()
                            - strike * (-rate * (tau + dt)).exp())
                        .max(0.0)
                    } else {
                        v[k as usize]
                    };
                    s += w * vk;
                }
                conv[i] = s;
            }
        }
        let mut a = vec![0.0; n_int];
        let mut b = vec![0.0; n_int];
        let mut c = vec![0.0; n_int];
        let mut rhs = vec![0.0; n_int];
        for i in 1..n_space {
            let idx = i - 1;
            a[idx] = sub;
            b[idx] = diag;
            c[idx] = sup;
            rhs[idx] = v[i] + dt * intensity * conv[i];
        }
        rhs[0] -= sub * vlo;
        a[0] = 0.0;
        rhs[n_int - 1] -= sup * vhi;
        c[n_int - 1] = 0.0;
        let inner = solve_tridiagonal(&a, &b, &c, &rhs)?;
        v[0] = vlo;
        v[n_space] = vhi;
        for i in 1..n_space {
            v[i] = inner[i - 1];
        }
    }

    // Linear interpolate at ln S0.
    for i in 1..=n_space {
        if x0 <= xs[i] {
            let w = (x0 - xs[i - 1]) / (xs[i] - xs[i - 1]);
            return Ok(v[i - 1] * (1.0 - w) + v[i] * w);
        }
    }
    Ok(*v.last().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed_rng;

    #[test]
    fn merton_reduces_to_bs_when_no_jumps() {
        let m = merton_call(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 0.0, 0.0, 0.1, 8).unwrap();
        let bs = bs_call(
            &BlackScholesMarket::new(100.0, 0.05, 0.0, 0.2, 1.0).unwrap(),
            100.0,
        )
        .unwrap()
        .price;
        assert!((m - bs).abs() < 1e-8);
        let mut rng = seed_rng(2);
        let times: Vec<f64> = (0..=50).map(|i| i as f64 * 0.02).collect();
        let p = compensated_poisson(&times, 3.0, &mut rng).unwrap();
        let last = *p.martingale.last().unwrap();
        assert!(last.abs() < 8.0);
        let pide0 =
            merton_call_pide(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 0.0, 0.0, 0.1, 80, 40).unwrap();
        assert!((pide0 - bs).abs() < 0.2, "λ=0 PIDE {pide0} vs BS {bs}");
        let series = merton_call(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 1.0, -0.08, 0.12, 24).unwrap();
        let pide =
            merton_call_pide(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 1.0, -0.08, 0.12, 100, 50).unwrap();
        assert!(
            (pide - series).abs() < 0.25,
            "PIDE {pide} vs Merton series {series}"
        );
        let (lp, mp) = esscher_normal_jump(1.0, 0.0, 0.2, 1.0).unwrap();
        assert!((mp - 0.04).abs() < 1e-14);
        assert!(lp > 1.0);
    }
}
