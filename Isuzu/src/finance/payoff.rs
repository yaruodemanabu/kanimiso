//! Terminal and path-dependent payoffs.

use crate::error::{Error, Result};
use crate::path::Path;

/// Payoff that depends only on the terminal state.
pub trait TerminalPayoff: Send + Sync {
    fn payoff(&self, terminal_state: &[f64]) -> Result<f64>;
}

/// Payoff that may depend on the whole path.
pub trait PathPayoff: Send + Sync {
    fn payoff(&self, path: &Path) -> Result<f64>;
}

fn check_strike(strike: f64) -> Result<()> {
    if !strike.is_finite() || strike < 0.0 {
        return Err(Error::param("strike must be finite and ≥ 0"));
    }
    Ok(())
}

/// European call `(S_T − K)⁺` on coordinate `asset`.
#[derive(Clone, Debug)]
pub struct EuropeanCall {
    pub asset: usize,
    pub strike: f64,
}

impl EuropeanCall {
    pub fn new(asset: usize, strike: f64) -> Result<Self> {
        check_strike(strike)?;
        Ok(Self { asset, strike })
    }
}

impl TerminalPayoff for EuropeanCall {
    fn payoff(&self, terminal_state: &[f64]) -> Result<f64> {
        let s = *terminal_state
            .get(self.asset)
            .ok_or_else(|| Error::dim("call asset index out of range"))?;
        Ok((s - self.strike).max(0.0))
    }
}

impl PathPayoff for EuropeanCall {
    fn payoff(&self, path: &Path) -> Result<f64> {
        TerminalPayoff::payoff(self, path.terminal())
    }
}

/// European put `(K − S_T)⁺`.
#[derive(Clone, Debug)]
pub struct EuropeanPut {
    pub asset: usize,
    pub strike: f64,
}

impl EuropeanPut {
    pub fn new(asset: usize, strike: f64) -> Result<Self> {
        check_strike(strike)?;
        Ok(Self { asset, strike })
    }
}

impl TerminalPayoff for EuropeanPut {
    fn payoff(&self, terminal_state: &[f64]) -> Result<f64> {
        let s = *terminal_state
            .get(self.asset)
            .ok_or_else(|| Error::dim("put asset index out of range"))?;
        Ok((self.strike - s).max(0.0))
    }
}

impl PathPayoff for EuropeanPut {
    fn payoff(&self, path: &Path) -> Result<f64> {
        TerminalPayoff::payoff(self, path.terminal())
    }
}

/// Cash-or-nothing digital: `1_{S_T > K}` (call) or `1_{S_T < K}` (put).
#[derive(Clone, Debug)]
pub struct Digital {
    pub asset: usize,
    pub strike: f64,
    pub cash: f64,
    pub is_call: bool,
}

impl Digital {
    pub fn new(asset: usize, strike: f64, cash: f64, is_call: bool) -> Result<Self> {
        check_strike(strike)?;
        if !cash.is_finite() {
            return Err(Error::param("digital cash must be finite"));
        }
        Ok(Self {
            asset,
            strike,
            cash,
            is_call,
        })
    }
}

impl TerminalPayoff for Digital {
    fn payoff(&self, terminal_state: &[f64]) -> Result<f64> {
        let s = *terminal_state
            .get(self.asset)
            .ok_or_else(|| Error::dim("digital asset index out of range"))?;
        let hit = if self.is_call {
            s > self.strike
        } else {
            s < self.strike
        };
        Ok(if hit { self.cash } else { 0.0 })
    }
}

impl PathPayoff for Digital {
    fn payoff(&self, path: &Path) -> Result<f64> {
        TerminalPayoff::payoff(self, path.terminal())
    }
}

/// Arithmetic or geometric Asian call / put on the discrete path mean.
#[derive(Clone, Debug)]
pub struct AsianOption {
    pub asset: usize,
    pub strike: f64,
    pub is_call: bool,
    pub geometric: bool,
}

impl AsianOption {
    pub fn new(asset: usize, strike: f64, is_call: bool, geometric: bool) -> Result<Self> {
        check_strike(strike)?;
        Ok(Self {
            asset,
            strike,
            is_call,
            geometric,
        })
    }
}

impl PathPayoff for AsianOption {
    fn payoff(&self, path: &Path) -> Result<f64> {
        if self.asset >= path.dim() {
            return Err(Error::dim("asian asset index out of range"));
        }
        let xs = path.component(self.asset)?;
        let avg = if self.geometric {
            let mut s = 0.0;
            for &x in &xs {
                if x <= 0.0 {
                    return Ok(0.0);
                }
                s += x.ln();
            }
            (s / xs.len() as f64).exp()
        } else {
            xs.iter().sum::<f64>() / xs.len() as f64
        };
        Ok(if self.is_call {
            (avg - self.strike).max(0.0)
        } else {
            (self.strike - avg).max(0.0)
        })
    }
}

/// Fixed-strike lookback: `(M − K)⁺` or `(K − m)⁺`.
#[derive(Clone, Debug)]
pub struct LookbackOption {
    pub asset: usize,
    pub strike: f64,
    pub is_call: bool,
}

impl LookbackOption {
    pub fn new(asset: usize, strike: f64, is_call: bool) -> Result<Self> {
        check_strike(strike)?;
        Ok(Self {
            asset,
            strike,
            is_call,
        })
    }
}

impl PathPayoff for LookbackOption {
    fn payoff(&self, path: &Path) -> Result<f64> {
        let xs = path.component(self.asset)?;
        if self.is_call {
            let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Ok((m - self.strike).max(0.0))
        } else {
            let m = xs.iter().copied().fold(f64::INFINITY, f64::min);
            Ok((self.strike - m).max(0.0))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierDirection {
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierActivation {
    KnockOut,
    KnockIn,
}

/// Barrier wrapper around a terminal payoff, with an optional rebate.
#[derive(Clone, Debug)]
pub struct BarrierOption<P> {
    pub underlying: P,
    pub asset: usize,
    pub barrier: f64,
    pub direction: BarrierDirection,
    pub activation: BarrierActivation,
    pub rebate: f64,
}

impl<P: TerminalPayoff> BarrierOption<P> {
    pub fn new(
        underlying: P,
        asset: usize,
        barrier: f64,
        direction: BarrierDirection,
        activation: BarrierActivation,
        rebate: f64,
    ) -> Result<Self> {
        if !barrier.is_finite() || barrier <= 0.0 {
            return Err(Error::param("barrier must be positive and finite"));
        }
        if !rebate.is_finite() {
            return Err(Error::param("rebate must be finite"));
        }
        Ok(Self {
            underlying,
            asset,
            barrier,
            direction,
            activation,
            rebate,
        })
    }
}

impl<P: TerminalPayoff> PathPayoff for BarrierOption<P> {
    fn payoff(&self, path: &Path) -> Result<f64> {
        let xs = path.component(self.asset)?;
        let hit = match self.direction {
            BarrierDirection::Up => xs.iter().any(|&s| s >= self.barrier),
            BarrierDirection::Down => xs.iter().any(|&s| s <= self.barrier),
        };
        match self.activation {
            BarrierActivation::KnockOut => {
                if hit {
                    Ok(self.rebate)
                } else {
                    self.underlying.payoff(path.terminal())
                }
            }
            BarrierActivation::KnockIn => {
                if hit {
                    self.underlying.payoff(path.terminal())
                } else {
                    Ok(self.rebate)
                }
            }
        }
    }
}

/// Margrabe exchange `(S¹_T − S²_T)⁺` or a weighted basket call.
#[derive(Clone, Debug)]
pub struct BasketCall {
    pub weights: Vec<f64>,
    pub strike: f64,
}

impl BasketCall {
    pub fn new(weights: Vec<f64>, strike: f64) -> Result<Self> {
        check_strike(strike)?;
        if weights.is_empty() || weights.iter().any(|w| !w.is_finite()) {
            return Err(Error::param("basket weights must be finite and nonempty"));
        }
        Ok(Self { weights, strike })
    }

    pub fn exchange(asset_long: usize, asset_short: usize, n_assets: usize) -> Result<Self> {
        if asset_long >= n_assets || asset_short >= n_assets {
            return Err(Error::dim("exchange asset index"));
        }
        let mut w = vec![0.0; n_assets];
        w[asset_long] = 1.0;
        w[asset_short] = -1.0;
        Self::new(w, 0.0)
    }
}

impl TerminalPayoff for BasketCall {
    fn payoff(&self, terminal_state: &[f64]) -> Result<f64> {
        if terminal_state.len() < self.weights.len() {
            return Err(Error::dim("basket state shorter than weights"));
        }
        let mut s = 0.0;
        for (w, x) in self.weights.iter().zip(terminal_state.iter()) {
            s += w * x;
        }
        Ok((s - self.strike).max(0.0))
    }
}

impl PathPayoff for BasketCall {
    fn payoff(&self, path: &Path) -> Result<f64> {
        TerminalPayoff::payoff(self, path.terminal())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_put_and_reject_bad_strike() {
        let c = EuropeanCall::new(0, 100.0).unwrap();
        assert!((TerminalPayoff::payoff(&c, &[110.0]).unwrap() - 10.0).abs() < 1e-14);
        assert_eq!(TerminalPayoff::payoff(&c, &[90.0]).unwrap(), 0.0);
        assert!(EuropeanCall::new(0, f64::NAN).is_err());
        let p = EuropeanPut::new(0, 100.0).unwrap();
        assert!((TerminalPayoff::payoff(&p, &[90.0]).unwrap() - 10.0).abs() < 1e-14);
    }
}
