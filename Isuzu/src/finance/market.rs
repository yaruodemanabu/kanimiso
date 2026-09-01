//! Discount curves, numeraires, and pricing measures.

use crate::error::{Error, Result};

/// Pricing measure. Physical and risk-neutral drifts are not the same `μ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PricingMeasure {
    Physical,
    RiskNeutral,
    Forward { maturity: f64 },
}

/// Deterministic discount curve.
pub trait DiscountCurve: Send + Sync {
    fn discount(&self, t: f64) -> Result<f64>;
    fn zero_rate(&self, t: f64) -> Result<f64>;
}

/// Flat continuously compounded rate.
#[derive(Clone, Copy, Debug)]
pub struct FlatCurve {
    pub rate: f64,
}

impl FlatCurve {
    pub fn new(rate: f64) -> Result<Self> {
        if !rate.is_finite() {
            return Err(Error::param("flat rate must be finite"));
        }
        Ok(Self { rate })
    }
}

impl DiscountCurve for FlatCurve {
    fn discount(&self, t: f64) -> Result<f64> {
        if !t.is_finite() || t < 0.0 {
            return Err(Error::param("discount time must be finite and ≥ 0"));
        }
        Ok((-self.rate * t).exp())
    }
    fn zero_rate(&self, t: f64) -> Result<f64> {
        let _ = t;
        Ok(self.rate)
    }
}

/// Numeraire process `N_t`.
pub trait Numeraire {
    fn value(&self, t: f64, state: &[f64]) -> Result<f64>;
}

/// Money-market numeraire `B_t = exp(∫ r)` for a flat curve.
#[derive(Clone, Copy, Debug)]
pub struct MoneyMarket {
    pub rate: f64,
}

impl MoneyMarket {
    pub fn new(rate: f64) -> Result<Self> {
        if !rate.is_finite() {
            return Err(Error::param("money-market rate must be finite"));
        }
        Ok(Self { rate })
    }
}

impl Numeraire for MoneyMarket {
    fn value(&self, t: f64, _state: &[f64]) -> Result<f64> {
        if !t.is_finite() || t < 0.0 {
            return Err(Error::param("numeraire time must be finite and ≥ 0"));
        }
        Ok((self.rate * t).exp())
    }
}

/// Zero-coupon bond numeraire `P(t,T)` under a flat curve.
#[derive(Clone, Copy, Debug)]
pub struct BondNumeraire {
    pub rate: f64,
    pub maturity: f64,
}

impl BondNumeraire {
    pub fn new(rate: f64, maturity: f64) -> Result<Self> {
        if !rate.is_finite() || !(maturity > 0.0 && maturity.is_finite()) {
            return Err(Error::param("bond numeraire needs finite rate and T>0"));
        }
        Ok(Self { rate, maturity })
    }
}

impl Numeraire for BondNumeraire {
    fn value(&self, t: f64, _state: &[f64]) -> Result<f64> {
        if t > self.maturity + 1e-14 {
            return Err(Error::param("bond numeraire evaluated after maturity"));
        }
        Ok(((-self.rate) * (self.maturity - t).max(0.0)).exp())
    }
}

/// Girsanov exponential martingale `Z_t = exp(−θ W_t − ½ θ² t)` for a constant
/// market price of risk (one-dimensional).
pub fn exponential_martingale(theta: f64, w: f64, t: f64) -> Result<f64> {
    if !theta.is_finite() || !w.is_finite() || !(t >= 0.0 && t.is_finite()) {
        return Err(Error::param("exponential martingale inputs must be finite"));
    }
    Ok((-theta * w - 0.5 * theta * theta * t).exp())
}

/// T-forward price of a cash payoff paid at `T`: `cash / P(0,T)`.
///
/// For deterministic rates this equals `E^T[X_T] = E[X_T]`.
pub fn t_forward_price(cash: f64, p0t: f64) -> Result<f64> {
    if !cash.is_finite() || !(p0t > 0.0 && p0t.is_finite()) {
        return Err(Error::param(
            "T-forward needs a finite cash price and P(0,T)>0",
        ));
    }
    Ok(cash / p0t)
}

/// Asset forward `F(0,T) = S_0 e^{(r−q)T}` under a flat curve
/// (equivalently `S_0 e^{−qT} / P(0,T)`).
pub fn asset_forward(spot: f64, rate: f64, dividend: f64, time: f64) -> Result<f64> {
    if !(spot > 0.0 && time >= 0.0) || !rate.is_finite() || !dividend.is_finite() {
        return Err(Error::param("asset forward needs S>0, T≥0, finite r,q"));
    }
    Ok(spot * ((rate - dividend) * time).exp())
}

/// Constant market price of risk `θ = (μ − r)/σ` on GBM.
pub fn market_price_of_risk(mu: f64, rate: f64, sigma: f64) -> Result<f64> {
    if !(sigma > 0.0 && sigma.is_finite()) || !mu.is_finite() || !rate.is_finite() {
        return Err(Error::param(
            "market price of risk needs σ>0 and finite μ,r",
        ));
    }
    Ok((mu - rate) / sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_discount_and_girsanov() {
        let c = FlatCurve::new(0.05).unwrap();
        assert!((c.discount(1.0).unwrap() - (-0.05_f64).exp()).abs() < 1e-14);
        let z = exponential_martingale(0.0, 1.0, 2.0).unwrap();
        assert!((z - 1.0).abs() < 1e-14);
        let th = market_price_of_risk(0.1, 0.02, 0.2).unwrap();
        assert!((th - 0.4).abs() < 1e-14);
        let p = c.discount(1.0).unwrap();
        let fwd = t_forward_price(10.450583572185565, p).unwrap();
        assert!((fwd - 10.450583572185565 / p).abs() < 1e-12);
        assert!(
            (asset_forward(100.0, 0.05, 0.0, 1.0).unwrap() - 100.0 * (0.05_f64).exp()).abs()
                < 1e-12
        );
    }
}
