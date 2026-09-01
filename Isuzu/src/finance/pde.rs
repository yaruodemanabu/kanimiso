//! Black–Scholes PDE: explicit / implicit / Crank–Nicolson + American PSOR.

use crate::error::{Error, Result};
use crate::finance::black_scholes::{call as bs_call, BlackScholesMarket};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryCondition {
    Dirichlet,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeScheme {
    Explicit,
    Implicit,
    CrankNicolson,
}

#[derive(Clone, Debug)]
pub struct PdeGrid {
    pub space: Vec<f64>,
    pub time: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct StabilityDiagnostics {
    pub cfl: f64,
    pub stable: bool,
}

#[derive(Clone, Debug)]
pub struct PdeSolution {
    pub values: Vec<Vec<f64>>,
    pub grid: PdeGrid,
    pub stability: StabilityDiagnostics,
}

/// Thomas algorithm for a tridiagonal system.
pub fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();
    if a.len() != n || c.len() != n || d.len() != n || n < 2 {
        return Err(Error::dim("tridiagonal length"));
    }
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    if b[0].abs() < 1e-18 {
        return Err(Error::numeric("tridiagonal pivot"));
    }
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for i in 1..n {
        let den = b[i] - a[i] * cp[i - 1];
        if den.abs() < 1e-18 {
            return Err(Error::numeric("tridiagonal pivot"));
        }
        cp[i] = c[i] / den;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    Ok(x)
}

fn payoff_call(s: f64, k: f64) -> f64 {
    (s - k).max(0.0)
}
fn payoff_put(s: f64, k: f64) -> f64 {
    (k - s).max(0.0)
}

/// European Black–Scholes PDE on a uniform log-or-spot grid (spot grid).
pub fn black_scholes_fd(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    s_max: f64,
    n_space: usize,
    n_time: usize,
    is_call: bool,
    scheme: TimeScheme,
    rannacher: usize,
) -> Result<PdeSolution> {
    if n_space < 3 || n_time == 0 || !(s_max > spot) || vol < 0.0 || time <= 0.0 {
        return Err(Error::param("PDE grid invalid"));
    }
    let ds = s_max / n_space as f64;
    let dt = time / n_time as f64;
    let space: Vec<f64> = (0..=n_space).map(|i| i as f64 * ds).collect();
    let times: Vec<f64> = (0..=n_time).map(|i| i as f64 * dt).collect();
    let mut v: Vec<f64> = space
        .iter()
        .map(|&s| {
            if is_call {
                payoff_call(s, strike)
            } else {
                payoff_put(s, strike)
            }
        })
        .collect();
    let mut values = vec![v.clone()];
    let mut max_cfl = 0.0_f64;
    for step in 0..n_time {
        let use_implicit = match scheme {
            TimeScheme::Explicit => false,
            TimeScheme::Implicit => true,
            TimeScheme::CrankNicolson => step < rannacher,
        };
        let theta = if use_implicit {
            1.0
        } else if matches!(scheme, TimeScheme::CrankNicolson) {
            0.5
        } else {
            0.0
        };
        // Interior points 1..n-1
        let n = n_space - 1;
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        for i in 1..n_space {
            let s = space[i];
            let sig2 = vol * vol * s * s;
            let mu = rate * s;
            let cfl = dt * sig2 / (ds * ds);
            max_cfl = max_cfl.max(cfl);
            if matches!(scheme, TimeScheme::Explicit) && cfl > 0.5 {
                return Err(Error::numeric(format!("explicit BS PDE CFL {cfl} > 1/2")));
            }
            let alpha = 0.5 * dt * (sig2 / (ds * ds) - mu / ds);
            let beta = dt * (sig2 / (ds * ds) + rate);
            let gamma = 0.5 * dt * (sig2 / (ds * ds) + mu / ds);
            // (I + θ L) v^{n} = (I - (1-θ) L) v^{n+1}  (time runs backward)
            let idx = i - 1;
            a[idx] = -theta * alpha;
            b[idx] = 1.0 + theta * beta;
            c[idx] = -theta * gamma;
            let mut r = v[i];
            r += (1.0 - theta) * alpha * v[i - 1];
            r += -(1.0 - theta) * beta * v[i];
            r += (1.0 - theta) * gamma * v[i + 1];
            rhs[idx] = r;
        }
        // Dirichlet boundaries at the new time level.
        let t_rem = time - (step + 1) as f64 * dt;
        let (vlo, vhi) = if is_call {
            (0.0, s_max - strike * (-rate * t_rem).exp())
        } else {
            (strike * (-rate * t_rem).exp(), 0.0)
        };
        rhs[0] -= a[0] * vlo;
        a[0] = 0.0;
        rhs[n - 1] -= c[n - 1] * vhi;
        c[n - 1] = 0.0;
        let inner = solve_tridiagonal(&a, &b, &c, &rhs)?;
        v[0] = vlo;
        v[n_space] = vhi;
        for i in 1..n_space {
            v[i] = inner[i - 1];
        }
        values.push(v.clone());
    }
    values.reverse();
    Ok(PdeSolution {
        values,
        grid: PdeGrid { space, time: times },
        stability: StabilityDiagnostics {
            cfl: max_cfl,
            stable: max_cfl <= 0.5 || !matches!(scheme, TimeScheme::Explicit),
        },
    })
}

/// Interpolate the PDE solution at `spot`.
pub fn pde_value_at(sol: &PdeSolution, spot: f64) -> Result<f64> {
    let xs = &sol.grid.space;
    // `black_scholes_fd` reverses the backward layers so `values[0]` is t = 0.
    let v0 = sol
        .values
        .first()
        .ok_or_else(|| Error::numeric("empty PDE"))?;
    if spot <= xs[0] {
        return Ok(v0[0]);
    }
    if spot >= *xs.last().unwrap() {
        return Ok(*v0.last().unwrap());
    }
    for i in 1..xs.len() {
        if spot <= xs[i] {
            let w = (spot - xs[i - 1]) / (xs[i] - xs[i - 1]);
            return Ok(v0[i - 1] * (1.0 - w) + v0[i] * w);
        }
    }
    Ok(*v0.last().unwrap())
}

/// American put via projected SOR (Cryer).
pub fn american_put_psor(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    s_max: f64,
    n_space: usize,
    n_time: usize,
    omega: f64,
    tol: f64,
    max_iter: usize,
) -> Result<(PdeSolution, f64)> {
    let mut sol = black_scholes_fd(
        spot,
        strike,
        rate,
        vol,
        time,
        s_max,
        n_space,
        n_time,
        false,
        TimeScheme::Implicit,
        0,
    )?;
    let ds = s_max / n_space as f64;
    let dt = time / n_time as f64;
    let mut v: Vec<f64> = sol
        .grid
        .space
        .iter()
        .map(|&s| payoff_put(s, strike))
        .collect();
    let mut values = vec![v.clone()];
    let mut residual = 0.0_f64;
    for step in 0..n_time {
        let t_rem = time - (step + 1) as f64 * dt;
        v[0] = strike * (-rate * t_rem).exp();
        v[n_space] = 0.0;
        for _it in 0..max_iter {
            let mut ch = 0.0;
            residual = 0.0;
            for i in 1..n_space {
                let s = sol.grid.space[i];
                let sig2 = vol * vol * s * s;
                let mu = rate * s;
                let alpha = 0.5 * dt * (sig2 / (ds * ds) - mu / ds);
                let beta = dt * (sig2 / (ds * ds) + rate);
                let gamma = 0.5 * dt * (sig2 / (ds * ds) + mu / ds);
                let rhs = values.last().unwrap()[i];
                let mid = 1.0 + beta;
                let y = (rhs + alpha * v[i - 1] + gamma * v[i + 1]) / mid;
                let ex = payoff_put(s, strike);
                let nxt = f64::max(v[i] + omega * (y - v[i]), ex);
                ch = f64::max(ch, (nxt - v[i]).abs());
                residual = f64::max(residual, (nxt - ex).min(0.0).abs());
                v[i] = nxt;
            }
            if ch < tol {
                break;
            }
        }
        values.push(v.clone());
    }
    values.reverse();
    sol.values = values;
    Ok((sol, residual))
}

/// Convenience: CN European call vs analytic.
pub fn cn_call_error(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    n_space: usize,
    n_time: usize,
) -> Result<f64> {
    let sol = black_scholes_fd(
        spot,
        strike,
        rate,
        vol,
        time,
        spot * 4.0,
        n_space,
        n_time,
        true,
        TimeScheme::CrankNicolson,
        2,
    )?;
    let price = pde_value_at(&sol, spot)?;
    let bs = bs_call(
        &BlackScholesMarket::new(spot, rate, 0.0, vol, time)?,
        strike,
    )?
    .price;
    Ok(price - bs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cn_near_bs_and_explicit_cfl() {
        let err = cn_call_error(100.0, 100.0, 0.05, 0.2, 1.0, 80, 80).unwrap();
        assert!(err.abs() < 0.15, "cn error {err}");
        let bad = black_scholes_fd(
            100.0,
            100.0,
            0.05,
            0.4,
            1.0,
            400.0,
            20,
            2,
            true,
            TimeScheme::Explicit,
            0,
        );
        assert!(bad.is_err());
    }
}
