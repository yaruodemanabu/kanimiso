//! Black–Scholes–Merton analytic prices and Greeks.

use crate::error::{Error, Result};
use crate::finance::special::{norm_cdf, norm_pdf};

/// Flat Black–Scholes market: spot, rate, dividend, vol, maturity.
#[derive(Clone, Copy, Debug)]
pub struct BlackScholesMarket {
    pub spot: f64,
    pub rate: f64,
    pub dividend: f64,
    pub vol: f64,
    pub time: f64,
}

impl BlackScholesMarket {
    pub fn new(spot: f64, rate: f64, dividend: f64, vol: f64, time: f64) -> Result<Self> {
        if !(spot > 0.0 && spot.is_finite()) {
            return Err(Error::param("spot must be positive and finite"));
        }
        if !rate.is_finite() || !dividend.is_finite() {
            return Err(Error::param("rate and dividend must be finite"));
        }
        if !(vol >= 0.0 && vol.is_finite()) {
            return Err(Error::param("vol must be finite and ≥ 0"));
        }
        if !(time >= 0.0 && time.is_finite()) {
            return Err(Error::param("time must be finite and ≥ 0"));
        }
        Ok(Self {
            spot,
            rate,
            dividend,
            vol,
            time,
        })
    }

    pub fn forward(&self) -> f64 {
        self.spot * ((self.rate - self.dividend) * self.time).exp()
    }

    pub fn discount(&self) -> f64 {
        (-self.rate * self.time).exp()
    }

    pub fn d1d2(&self, strike: f64) -> Result<(f64, f64)> {
        if !(strike > 0.0 && strike.is_finite()) {
            return Err(Error::param("strike must be positive and finite"));
        }
        if self.time == 0.0 || self.vol == 0.0 {
            let fwd = self.forward();
            let inf = if fwd > strike {
                f64::INFINITY
            } else if fwd < strike {
                f64::NEG_INFINITY
            } else {
                0.0
            };
            return Ok((inf, inf));
        }
        let vs = self.vol * self.time.sqrt();
        let d1 = ((self.spot / strike).ln()
            + (self.rate - self.dividend + 0.5 * self.vol * self.vol) * self.time)
            / vs;
        Ok((d1, d1 - vs))
    }
}

/// Analytic European call / put and first-order Greeks.
#[derive(Clone, Copy, Debug)]
pub struct BlackScholesPrice {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

fn intrinsic_call(m: &BlackScholesMarket, strike: f64) -> f64 {
    (m.spot * (-m.dividend * m.time).exp() - strike * m.discount()).max(0.0)
}

fn intrinsic_put(m: &BlackScholesMarket, strike: f64) -> f64 {
    (strike * m.discount() - m.spot * (-m.dividend * m.time).exp()).max(0.0)
}

/// European call.
pub fn call(m: &BlackScholesMarket, strike: f64) -> Result<BlackScholesPrice> {
    if m.time == 0.0 || m.vol == 0.0 {
        let price = intrinsic_call(m, strike);
        let itm = m.forward() > strike;
        return Ok(BlackScholesPrice {
            price,
            delta: if itm {
                (-m.dividend * m.time).exp()
            } else {
                0.0
            },
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: if itm {
                strike * m.time * m.discount()
            } else {
                0.0
            },
        });
    }
    let (d1, d2) = m.d1d2(strike)?;
    let df_q = (-m.dividend * m.time).exp();
    let df_r = m.discount();
    let price = m.spot * df_q * norm_cdf(d1) - strike * df_r * norm_cdf(d2);
    Ok(greeks_from_d(m, strike, d1, d2, price, true))
}

/// European put.
pub fn put(m: &BlackScholesMarket, strike: f64) -> Result<BlackScholesPrice> {
    if m.time == 0.0 || m.vol == 0.0 {
        let price = intrinsic_put(m, strike);
        let itm = m.forward() < strike;
        return Ok(BlackScholesPrice {
            price,
            delta: if itm {
                -(-m.dividend * m.time).exp()
            } else {
                0.0
            },
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: if itm {
                -strike * m.time * m.discount()
            } else {
                0.0
            },
        });
    }
    let (d1, d2) = m.d1d2(strike)?;
    let df_q = (-m.dividend * m.time).exp();
    let df_r = m.discount();
    let price = strike * df_r * norm_cdf(-d2) - m.spot * df_q * norm_cdf(-d1);
    Ok(greeks_from_d(m, strike, d1, d2, price, false))
}

fn greeks_from_d(
    m: &BlackScholesMarket,
    strike: f64,
    d1: f64,
    d2: f64,
    price: f64,
    is_call: bool,
) -> BlackScholesPrice {
    let df_q = (-m.dividend * m.time).exp();
    let df_r = m.discount();
    let sqrt_t = m.time.sqrt();
    let nd1 = norm_pdf(d1);
    let delta = if is_call {
        df_q * norm_cdf(d1)
    } else {
        -df_q * norm_cdf(-d1)
    };
    let gamma = df_q * nd1 / (m.spot * m.vol * sqrt_t);
    let vega = m.spot * df_q * nd1 * sqrt_t;
    let theta = if is_call {
        -m.spot * df_q * nd1 * m.vol / (2.0 * sqrt_t) + m.dividend * m.spot * df_q * norm_cdf(d1)
            - m.rate * strike * df_r * norm_cdf(d2)
    } else {
        -m.spot * df_q * nd1 * m.vol / (2.0 * sqrt_t) - m.dividend * m.spot * df_q * norm_cdf(-d1)
            + m.rate * strike * df_r * norm_cdf(-d2)
    };
    let rho = if is_call {
        strike * m.time * df_r * norm_cdf(d2)
    } else {
        -strike * m.time * df_r * norm_cdf(-d2)
    };
    BlackScholesPrice {
        price,
        delta,
        gamma,
        vega,
        theta,
        rho,
    }
}

/// Put–call parity residual `C − P − (S e^{−qT} − K e^{−rT})`.
pub fn put_call_parity_gap(m: &BlackScholesMarket, strike: f64) -> Result<f64> {
    let c = call(m, strike)?;
    let p = put(m, strike)?;
    Ok(c.price - p.price - (m.spot * (-m.dividend * m.time).exp() - strike * m.discount()))
}

/// Cash-or-nothing digital call `e^{−rT} Φ(d₂)`.
pub fn digital_call(m: &BlackScholesMarket, strike: f64, cash: f64) -> Result<f64> {
    if m.time == 0.0 || m.vol == 0.0 {
        return Ok(if m.forward() > strike {
            cash * m.discount()
        } else {
            0.0
        });
    }
    let (_, d2) = m.d1d2(strike)?;
    Ok(cash * m.discount() * norm_cdf(d2))
}

/// Margrabe (1978) exchange: the value of receiving `S1` and paying `S2`.
///
/// `C = S_1 e^{-q_1 T} Φ(d_1) − S_2 e^{-q_2 T} Φ(d_2)` with
/// `σ = √(σ₁² + σ₂² − 2ρσ₁σ₂)`.
pub fn margrabe(
    s1: f64,
    s2: f64,
    q1: f64,
    q2: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    time: f64,
) -> Result<f64> {
    if !(s1 > 0.0 && s2 > 0.0 && time >= 0.0)
        || !(-1.0..=1.0).contains(&rho)
        || vol1 < 0.0
        || vol2 < 0.0
    {
        return Err(Error::param("Margrabe needs S>0, T≥0, σ≥0, ρ∈[-1,1]"));
    }
    let sig2 = vol1 * vol1 + vol2 * vol2 - 2.0 * rho * vol1 * vol2;
    let sig = sig2.max(0.0).sqrt();
    let m = BlackScholesMarket::new(s1, q2, q1, sig, time)?;
    Ok(call(&m, s2)?.price)
}

/// Continuous geometric Asian call (Kemna–Vorst / closed lognormal).
pub fn geometric_asian_call(m: &BlackScholesMarket, strike: f64) -> Result<f64> {
    if m.time <= 0.0 {
        return Ok((m.spot - strike).max(0.0));
    }
    let sig_a = m.vol / 3.0_f64.sqrt();
    let b = 0.5 * (m.rate - m.dividend - m.vol * m.vol / 6.0);
    let adj = BlackScholesMarket::new(m.spot, m.rate, m.rate - b, sig_a, m.time)?;
    Ok(call(&adj, strike)?.price)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_and_atm_call() {
        let m = BlackScholesMarket::new(100.0, 0.05, 0.0, 0.2, 1.0).unwrap();
        let c = call(&m, 100.0).unwrap();
        // Haug / standard ATM ≈ 10.4506
        assert!((c.price - 10.450583572185565).abs() < 1e-8, "{}", c.price);
        assert!(put_call_parity_gap(&m, 100.0).unwrap().abs() < 1e-12);
        assert!(c.gamma > 0.0 && c.vega > 0.0);
        let t0 = BlackScholesMarket::new(110.0, 0.05, 0.0, 0.2, 0.0).unwrap();
        assert!((call(&t0, 100.0).unwrap().price - 10.0).abs() < 1e-12);
        let z = BlackScholesMarket::new(100.0, 0.05, 0.0, 0.0, 1.0).unwrap();
        let intrinsic = (100.0 * 1.0 - 100.0 * (-0.05_f64).exp()).max(0.0);
        assert!((call(&z, 100.0).unwrap().price - intrinsic).abs() < 1e-12);
        let ex = margrabe(100.0, 100.0, 0.0, 0.0, 0.2, 0.2, 1.0, 1.0).unwrap();
        assert!(ex.abs() < 1e-10, "perfectly correlated equal assets {ex}");
        let spark = margrabe(80.0, 40.0, 0.0, 0.0, 0.4, 0.3, 0.3, 1.0).unwrap();
        assert!(spark > 0.0);
    }
}
