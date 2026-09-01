//! Affine short-rate bond prices, yields, bond options, and caplets.

use crate::error::{Error, Result};
use crate::finance::black_scholes::{call as bs_call, put as bs_put, BlackScholesMarket};
use crate::finance::special::norm_cdf;
use crate::models::{Cir, OrnsteinUhlenbeck};

/// Zero-coupon bond quote.
#[derive(Clone, Copy, Debug)]
pub struct BondQuote {
    pub price: f64,
    pub yield_: f64,
    pub duration: f64,
}

fn bond_quote(price: f64, tau: f64) -> Result<BondQuote> {
    if !(price > 0.0 && price.is_finite()) || !(tau > 0.0) {
        return Err(Error::numeric("bond price must be positive"));
    }
    Ok(BondQuote {
        price,
        yield_: -price.ln() / tau,
        duration: tau,
    })
}

/// Vasicek / Hull–White (constant θ) zero-coupon `P(t,T) = exp(A − B r)`.
pub fn vasicek_bond(model: &OrnsteinUhlenbeck, r0: f64, tau: f64) -> Result<BondQuote> {
    if tau < 0.0 || !r0.is_finite() {
        return Err(Error::param("vasicek bond needs finite r0 and τ≥0"));
    }
    if tau == 0.0 {
        return Ok(BondQuote {
            price: 1.0,
            yield_: r0,
            duration: 0.0,
        });
    }
    let k = model.kappa;
    let th = model.theta;
    let sig = model.sigma;
    let b = (1.0 - (-k * tau).exp()) / k;
    let a = (th - sig * sig / (2.0 * k * k)) * (b - tau) - sig * sig * b * b / (4.0 * k);
    bond_quote((a - b * r0).exp(), tau)
}

/// CIR (Feller) affine bond.
pub fn cir_bond(model: &Cir, r0: f64, tau: f64) -> Result<BondQuote> {
    if tau < 0.0 || !(r0 >= 0.0 && r0.is_finite()) {
        return Err(Error::param("CIR bond needs r0≥0 and τ≥0"));
    }
    if tau == 0.0 {
        return Ok(BondQuote {
            price: 1.0,
            yield_: r0,
            duration: 0.0,
        });
    }
    let k = model.kappa;
    let th = model.theta;
    let sig = model.sigma;
    let phi = (k * k + 2.0 * sig * sig).sqrt();
    let e = (phi * tau).exp();
    let den = (phi + k) * (e - 1.0) + 2.0 * phi;
    let b = 2.0 * (e - 1.0) / den;
    let a = (2.0 * k * th / (sig * sig)) * (2.0 * phi * ((k + phi) * tau * 0.5).exp() / den).ln();
    bond_quote((a - b * r0).exp(), tau)
}

/// Instantaneous forward `f(t,T) = −∂_T log P(t,T)` by a one-sided difference.
pub fn forward_rate(price: impl Fn(f64) -> Result<f64>, t: f64, h: f64) -> Result<f64> {
    let p0 = price(t)?;
    let p1 = price(t + h)?;
    if p0 <= 0.0 || p1 <= 0.0 {
        return Err(Error::numeric("forward rate: non-positive bond"));
    }
    Ok(-(p1.ln() - p0.ln()) / h)
}

/// Jamshidian / affine bond option under Vasicek (European call on `P(T,S)`).
///
/// The bond is log-normal, so the price is Black on the forward bond.
pub fn vasicek_bond_option(
    model: &OrnsteinUhlenbeck,
    r0: f64,
    expiry: f64,
    bond_mat: f64,
    strike: f64,
    is_call: bool,
) -> Result<f64> {
    if bond_mat <= expiry {
        return Err(Error::param("bond option needs S > T"));
    }
    let p_t = vasicek_bond(model, r0, expiry)?.price;
    let p_s = vasicek_bond(model, r0, bond_mat)?.price;
    let k = model.kappa;
    let sig = model.sigma;
    let b_ts = (1.0 - (-k * (bond_mat - expiry)).exp()) / k;
    let var = (sig * sig / (2.0 * k)) * (1.0 - (-2.0 * k * expiry).exp()) * b_ts * b_ts;
    let vol = var.sqrt() / expiry.sqrt();
    let fwd = p_s / p_t;
    let mkt = BlackScholesMarket::new(fwd, 0.0, 0.0, vol, expiry)?;
    let black = if is_call {
        bs_call(&mkt, strike)?.price
    } else {
        bs_put(&mkt, strike)?.price
    };
    Ok(p_t * black)
}

/// Hull–White time-dependent drift `θ(t)` that fits a discount curve.
///
/// `θ(t) = ∂f/∂T + κ f(0,t) + σ²(1−e^{−2κt})/(2κ)` with
/// `f(0,t) = −∂_t log P(0,t)` by central differences on `discounts`.
pub fn hull_white_theta(
    kappa: f64,
    sigma: f64,
    times: &[f64],
    discounts: &[f64],
) -> Result<Vec<f64>> {
    if times.len() != discounts.len() || times.len() < 3 {
        return Err(Error::dim("Hull–White fit needs ≥ 3 (t, P) nodes"));
    }
    if !(kappa > 0.0 && sigma >= 0.0) {
        return Err(Error::param("Hull–White needs κ>0, σ≥0"));
    }
    let n = times.len();
    let mut fwd = vec![0.0; n];
    for i in 0..n {
        if discounts[i] <= 0.0 || !discounts[i].is_finite() {
            return Err(Error::numeric("discount must be positive"));
        }
        if i + 1 < n && times[i + 1] <= times[i] {
            return Err(Error::sampling("Hull–White times must increase"));
        }
    }
    for i in 0..n {
        let (il, ir) = if i == 0 {
            (0, 1)
        } else if i + 1 == n {
            (n - 2, n - 1)
        } else {
            (i - 1, i + 1)
        };
        let dt = times[ir] - times[il];
        if dt <= 0.0 {
            return Err(Error::numeric("Hull–White dt"));
        }
        fwd[i] = -(discounts[ir].ln() - discounts[il].ln()) / dt;
    }
    let mut theta = vec![0.0; n];
    for i in 0..n {
        let (il, ir) = if i == 0 {
            (0, 1)
        } else if i + 1 == n {
            (n - 2, n - 1)
        } else {
            (i - 1, i + 1)
        };
        let dfdt = (fwd[ir] - fwd[il]) / (times[ir] - times[il]);
        let t = times[i];
        theta[i] = dfdt
            + kappa * fwd[i]
            + sigma * sigma * (1.0 - (-2.0 * kappa * t).exp()) / (2.0 * kappa);
    }
    Ok(theta)
}

/// Black swaption (unit notional) with annuity `A`, forward swap rate `S`.
pub fn black_swaption(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    annuity: f64,
    is_payer: bool,
) -> Result<f64> {
    if !(forward > 0.0 && strike > 0.0 && vol >= 0.0 && expiry >= 0.0 && annuity > 0.0) {
        return Err(Error::param("swaption inputs invalid"));
    }
    let black = black_caplet(forward, strike, vol, expiry, 1.0, 1.0)?;
    let put = black_caplet(strike, forward, vol, expiry, 1.0, 1.0)?;
    Ok(annuity * if is_payer { black } else { put })
}

/// Caplet under Black with a simply-compounded forward `F` and year-fraction `δ`.
pub fn black_caplet(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    delta: f64,
    df: f64,
) -> Result<f64> {
    if !(forward > 0.0 && strike > 0.0 && vol >= 0.0 && expiry >= 0.0 && delta > 0.0) {
        return Err(Error::param("caplet inputs invalid"));
    }
    if expiry == 0.0 || vol == 0.0 {
        return Ok(df * delta * (forward - strike).max(0.0));
    }
    let vs = vol * expiry.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * vol * vol * expiry) / vs;
    let d2 = d1 - vs;
    Ok(df * delta * (forward * norm_cdf(d1) - strike * norm_cdf(d2)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Cir, OrnsteinUhlenbeck};

    #[test]
    fn vasicek_cir_bonds_positive() {
        let v = OrnsteinUhlenbeck::new(0.5, 0.03, 0.02).unwrap();
        let b = vasicek_bond(&v, 0.03, 1.0).unwrap();
        assert!(b.price > 0.0 && b.price < 1.0);
        assert!((b.yield_ - 0.03).abs() < 0.01);
        let c = Cir::new(0.5, 0.03, 0.05).unwrap();
        let pb = cir_bond(&c, 0.03, 1.0).unwrap();
        assert!(pb.price > 0.0 && pb.price < 1.0);
        let cap = black_caplet(0.03, 0.03, 0.2, 1.0, 0.25, 0.97).unwrap();
        assert!(cap > 0.0);
        let opt = vasicek_bond_option(&v, 0.03, 0.5, 1.0, 0.97, true).unwrap();
        assert!(opt > 0.0);
        let ts = [0.0_f64, 1.0, 2.0, 3.0];
        let ps: Vec<f64> = ts.iter().map(|t| (-0.03_f64 * t).exp()).collect();
        let th = hull_white_theta(0.5, 0.01, &ts, &ps).unwrap();
        assert_eq!(th.len(), 4);
        assert!(th.iter().all(|x| x.is_finite()));
        let sw = black_swaption(0.03, 0.03, 0.2, 1.0, 4.0, true).unwrap();
        assert!(sw > 0.0);
    }
}
