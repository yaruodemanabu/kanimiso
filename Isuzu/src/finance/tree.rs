//! Cox–Ross–Rubinstein binomial tree with replication holdings.

use crate::error::{Error, Result};
use crate::finance::black_scholes::{call as bs_call, BlackScholesMarket};

/// One-period finite-state market (Shreve I).
#[derive(Clone, Debug)]
pub struct OnePeriodMarket {
    pub bank_growth: f64,
    pub stock_now: f64,
    pub stock_next: Vec<f64>,
}

/// Replicating holdings `(Δ, Γ_bond)` in the stock and the money-market.
#[derive(Clone, Debug)]
pub struct Portfolio {
    pub stock: f64,
    pub bond: f64,
    pub value: f64,
}

/// State prices `ψ_i` solving `Σ ψ_i = 1/R` and `Σ ψ_i S_i = S_0`.
pub fn state_prices(market: &OnePeriodMarket) -> Result<Vec<f64>> {
    if market.stock_next.len() != 2 {
        return Err(Error::unsupported(
            "state_prices is implemented for two states",
        ));
    }
    if !(market.bank_growth > 0.0 && market.stock_now > 0.0) {
        return Err(Error::param(
            "one-period market needs positive bank growth and spot",
        ));
    }
    let su = market.stock_next[0];
    let sd = market.stock_next[1];
    if (su - sd).abs() < 1e-15 {
        return Err(Error::numeric("one-period market is degenerate"));
    }
    // ψ_u + ψ_d = 1/R,  ψ_u S_u + ψ_d S_d = S_0
    let rinv = 1.0 / market.bank_growth;
    let psi_u = (market.stock_now - rinv * sd) / (su - sd);
    let psi_d = rinv - psi_u;
    if psi_u < -1e-12 || psi_d < -1e-12 {
        return Err(Error::numeric("one-period market admits arbitrage"));
    }
    Ok(vec![psi_u.max(0.0), psi_d.max(0.0)])
}

/// Replicate a two-state contingent claim.
pub fn replicate(claim: &[f64], market: &OnePeriodMarket) -> Result<Portfolio> {
    if claim.len() != 2 {
        return Err(Error::dim("two-state claim"));
    }
    let su = market.stock_next[0];
    let sd = market.stock_next[1];
    let delta = (claim[0] - claim[1]) / (su - sd);
    let bond = (claim[0] - delta * su) / market.bank_growth;
    Ok(Portfolio {
        stock: delta,
        bond,
        value: delta * market.stock_now + bond,
    })
}

/// CRR tree price (European or American) plus the initial delta / bond.
#[derive(Clone, Debug)]
pub struct CrrPrice {
    pub price: f64,
    pub delta: f64,
    pub bond: f64,
    pub risk_neutral_up: f64,
    pub tree: Vec<Vec<f64>>,
}

/// Cox–Ross–Rubinstein backward induction.
pub fn crr_price(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    n_steps: usize,
    is_call: bool,
    american: bool,
) -> Result<CrrPrice> {
    if n_steps == 0 || !(spot > 0.0 && strike >= 0.0 && vol >= 0.0 && time > 0.0) {
        return Err(Error::param("CRR inputs invalid"));
    }
    let dt = time / n_steps as f64;
    let u = (vol * dt.sqrt()).exp();
    let d = 1.0 / u;
    let growth = (rate * dt).exp();
    if d > growth + 1e-14 || growth > u + 1e-14 {
        return Err(Error::numeric("CRR parameters admit arbitrage"));
    }
    let p = (growth - d) / (u - d);
    if !(0.0..=1.0).contains(&p) {
        return Err(Error::numeric("CRR risk-neutral probability outside [0,1]"));
    }
    let mut stock = vec![0.0; n_steps + 1];
    for j in 0..=n_steps {
        stock[j] = spot * u.powi(j as i32) * d.powi((n_steps - j) as i32);
    }
    let mut val: Vec<f64> = stock
        .iter()
        .map(|&s| {
            if is_call {
                (s - strike).max(0.0)
            } else {
                (strike - s).max(0.0)
            }
        })
        .collect();
    let mut tree = vec![val.clone()];
    for step in (0..n_steps).rev() {
        let mut nxt = vec![0.0; step + 1];
        for j in 0..=step {
            let cont = (p * val[j + 1] + (1.0 - p) * val[j]) / growth;
            let s = spot * u.powi(j as i32) * d.powi((step - j) as i32);
            let ex = if is_call {
                (s - strike).max(0.0)
            } else {
                (strike - s).max(0.0)
            };
            nxt[j] = if american { cont.max(ex) } else { cont };
        }
        val = nxt;
        tree.push(val.clone());
    }
    tree.reverse();
    let s_up = spot * u;
    let s_dn = spot * d;
    let v_up = if tree.len() > 1 && tree[1].len() > 1 {
        tree[1][1]
    } else {
        val[0]
    };
    let v_dn = if tree.len() > 1 { tree[1][0] } else { val[0] };
    let delta = (v_up - v_dn) / (s_up - s_dn);
    let bond = (v_up - delta * s_up) / growth;
    Ok(CrrPrice {
        price: val[0],
        delta,
        bond,
        risk_neutral_up: p,
        tree,
    })
}

/// CRR European call should approach Black–Scholes as `n → ∞`.
pub fn crr_vs_bs_call(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    n_steps: usize,
) -> Result<(f64, f64)> {
    let tree = crr_price(spot, strike, rate, vol, time, n_steps, true, false)?;
    let bs = bs_call(
        &BlackScholesMarket::new(spot, rate, 0.0, vol, time)?,
        strike,
    )?;
    Ok((tree.price, bs.price))
}

/// Kamrad–Ritchken trinomial (`λ = √3`) European or American vanilla.
pub fn trinomial_price(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    n_steps: usize,
    is_call: bool,
    american: bool,
) -> Result<f64> {
    if n_steps == 0 || !(spot > 0.0 && strike >= 0.0 && vol >= 0.0 && time > 0.0) {
        return Err(Error::param("trinomial inputs invalid"));
    }
    let dt = time / n_steps as f64;
    let nu = rate - 0.5 * vol * vol;
    let dx = vol * (3.0 * dt).sqrt().max(1e-18);
    let pu = 0.5 * ((vol * vol * dt + nu * nu * dt * dt) / (dx * dx) + nu * dt / dx);
    let pd = 0.5 * ((vol * vol * dt + nu * nu * dt * dt) / (dx * dx) - nu * dt / dx);
    let pm = 1.0 - pu - pd;
    if pu < -1e-12 || pd < -1e-12 || pm < -1e-12 {
        return Err(Error::numeric("trinomial probabilities negative"));
    }
    let disc = (-rate * dt).exp();
    // Nodes at step k: j = 0..=2k, log-spot = ln S + (j-k) dx.
    let mut val = vec![0.0; 2 * n_steps + 1];
    for j in 0..=2 * n_steps {
        let s = (x0_plus(spot, (j as i32 - n_steps as i32) as f64 * dx)).exp();
        val[j] = if is_call {
            (s - strike).max(0.0)
        } else {
            (strike - s).max(0.0)
        };
    }
    for step in (0..n_steps).rev() {
        let mut nxt = vec![0.0; 2 * step + 1];
        for j in 0..=2 * step {
            let cont = disc * (pu * val[j + 2] + pm * val[j + 1] + pd * val[j]);
            let s = (x0_plus(spot, (j as i32 - step as i32) as f64 * dx)).exp();
            let ex = if is_call {
                (s - strike).max(0.0)
            } else {
                (strike - s).max(0.0)
            };
            nxt[j] = if american { cont.max(ex) } else { cont };
        }
        val = nxt;
    }
    Ok(val[0])
}

fn x0_plus(spot: f64, dx: f64) -> f64 {
    spot.ln() + dx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_period_replication_and_crr_limit() {
        let m = OnePeriodMarket {
            bank_growth: 1.1,
            stock_now: 4.0,
            stock_next: vec![8.0, 2.0],
        };
        let psi = state_prices(&m).unwrap();
        assert!((psi[0] + psi[1] - 1.0 / 1.1).abs() < 1e-12);
        let port = replicate(&[1.0, 0.0], &m).unwrap();
        assert!((port.stock - 1.0 / 6.0).abs() < 1e-12);
        let (tree, bs) = crr_vs_bs_call(100.0, 100.0, 0.05, 0.2, 1.0, 400).unwrap();
        assert!((tree - bs).abs() < 0.05, "tree {tree} bs {bs}");
        let euro = crr_price(100.0, 100.0, 0.05, 0.2, 1.0, 80, false, false).unwrap();
        let amer = crr_price(100.0, 100.0, 0.05, 0.2, 1.0, 80, false, true).unwrap();
        assert!(amer.price + 1e-12 >= euro.price);
        let ac = crr_price(100.0, 100.0, 0.05, 0.2, 1.0, 80, true, true).unwrap();
        let ec = crr_price(100.0, 100.0, 0.05, 0.2, 1.0, 80, true, false).unwrap();
        assert!((ac.price - ec.price).abs() < 1e-8);
        let tri = trinomial_price(100.0, 100.0, 0.05, 0.2, 1.0, 80, true, false).unwrap();
        assert!((tri - bs).abs() < 0.08, "trinomial {tri} bs {bs}");
        let tac = trinomial_price(100.0, 100.0, 0.05, 0.2, 1.0, 60, true, true).unwrap();
        let tec = trinomial_price(100.0, 100.0, 0.05, 0.2, 1.0, 60, true, false).unwrap();
        assert!((tac - tec).abs() < 1e-8);
    }
}
