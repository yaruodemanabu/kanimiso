//! Real options (Dixit–Pindyck / McDonald–Siegel) on GBM cash-flows.
//!
//! The energy / commodity versions use the same thresholds with a
//! convenience yield `δ` in place of a dividend.

use crate::error::{Error, Result};

/// Perpetual American call (the investment opportunity `V(S) = A S^β`).
#[derive(Clone, Copy, Debug)]
pub struct PerpetualCall {
    pub beta: f64,
    pub threshold: f64,
    pub scale: f64,
    pub investment: f64,
}

/// `β` root of `½ σ² β(β−1) + (r−δ)β − r = 0` (the positive one).
pub fn investment_beta(rate: f64, dividend: f64, vol: f64) -> Result<f64> {
    if !(vol > 0.0 && rate > 0.0) {
        return Err(Error::param("perpetual option needs r>0, σ>0"));
    }
    let mu = rate - dividend;
    let a = 0.5 * vol * vol;
    let b = mu - a;
    let disc = b * b + 2.0 * rate * vol * vol;
    if disc < 0.0 {
        return Err(Error::numeric("investment beta discriminant negative"));
    }
    Ok((-b + disc.sqrt()) / (2.0 * a))
}

/// McDonald–Siegel / Dixit–Pindyck irreversible investment.
///
/// Invest `I` to receive a project worth `S` (GBM). The threshold is
/// `S* = β/(β−1) I` and `V(S) = (S* − I) (S/S*)^β` for `S < S*`.
pub fn mcdonald_siegel(
    rate: f64,
    dividend: f64,
    vol: f64,
    investment: f64,
) -> Result<PerpetualCall> {
    if !(investment > 0.0) {
        return Err(Error::param("investment cost must be positive"));
    }
    let beta = investment_beta(rate, dividend, vol)?;
    if beta <= 1.0 {
        return Err(Error::numeric("investment beta must exceed 1 (need δ>0)"));
    }
    let threshold = beta / (beta - 1.0) * investment;
    let scale = (threshold - investment) / threshold.powf(beta);
    Ok(PerpetualCall {
        beta,
        threshold,
        scale,
        investment,
    })
}

impl PerpetualCall {
    pub fn value(&self, spot: f64) -> Result<f64> {
        if !(spot > 0.0 && spot.is_finite()) {
            return Err(Error::param("spot must be positive"));
        }
        if spot >= self.threshold {
            Ok(spot - self.investment)
        } else {
            Ok(self.scale * spot.powf(self.beta))
        }
    }
}

/// Perpetual American put (abandonment / scrap value `E`).
#[derive(Clone, Copy, Debug)]
pub struct PerpetualPut {
    pub beta: f64,
    pub threshold: f64,
    pub scale: f64,
    pub scrap: f64,
}

/// Negative root of the same fundamental quadratic.
pub fn abandonment_beta(rate: f64, dividend: f64, vol: f64) -> Result<f64> {
    if !(vol > 0.0 && rate > 0.0) {
        return Err(Error::param("abandonment needs r>0, σ>0"));
    }
    let mu = rate - dividend;
    let disc = (mu / vol / vol - 0.5).powi(2) + 2.0 * rate / (vol * vol);
    Ok(0.5 - mu / (vol * vol) - disc.sqrt())
}

/// Abandon a project worth `S` for scrap `E`.
pub fn abandonment_option(rate: f64, dividend: f64, vol: f64, scrap: f64) -> Result<PerpetualPut> {
    if !(scrap > 0.0) {
        return Err(Error::param("scrap value must be positive"));
    }
    let beta = abandonment_beta(rate, dividend, vol)?;
    if beta >= 0.0 {
        return Err(Error::numeric("abandonment beta must be negative"));
    }
    let threshold = beta / (beta - 1.0) * scrap;
    let scale = (scrap - threshold) / threshold.powf(beta);
    Ok(PerpetualPut {
        beta,
        threshold,
        scale,
        scrap,
    })
}

impl PerpetualPut {
    pub fn value(&self, spot: f64) -> Result<f64> {
        if !(spot > 0.0 && spot.is_finite()) {
            return Err(Error::param("spot must be positive"));
        }
        if spot <= self.threshold {
            Ok(self.scrap - spot)
        } else {
            Ok(self.scale * spot.powf(self.beta))
        }
    }
}

/// Entry / exit pair of thresholds (Dixit 1989, two-sided).
#[derive(Clone, Copy, Debug)]
pub struct EntryExit {
    pub enter: f64,
    pub exit: f64,
}

/// Approximate Dixit entry/exit for unit operating profit `S−C` with
/// entry cost `k` and exit cost `l` (closed form when `C=0` is the
/// investment/abandonment pair; here `C` shifts the flow).
pub fn entry_exit(
    rate: f64,
    dividend: f64,
    vol: f64,
    cost_flow: f64,
    entry_cost: f64,
    exit_cost: f64,
) -> Result<EntryExit> {
    let inv = mcdonald_siegel(
        rate,
        dividend,
        vol,
        entry_cost + cost_flow / rate.max(1e-12),
    )?;
    let abd = abandonment_option(
        rate,
        dividend,
        vol,
        (cost_flow / rate.max(1e-12) - exit_cost).max(1e-12),
    )?;
    Ok(EntryExit {
        enter: inv.threshold,
        exit: abd.threshold,
    })
}

/// Finite-horizon investment as a European call on the project value
/// (McDonald–Siegel with a deadline). The Black–Scholes call with
/// dividend `δ` and strike `I` is the value of a now-or-at-T option;
/// early exercise is an American call (only if `δ>0`).
pub fn finite_horizon_investment(
    project: f64,
    investment: f64,
    rate: f64,
    dividend: f64,
    vol: f64,
    time: f64,
) -> Result<f64> {
    use crate::finance::black_scholes::{call, BlackScholesMarket};
    let m = BlackScholesMarket::new(project, rate, dividend, vol, time)?;
    Ok(call(&m, investment)?.price)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn investment_threshold_above_cost() {
        let opt = mcdonald_siegel(0.05, 0.03, 0.2, 1.0).unwrap();
        assert!(opt.threshold > 1.0);
        assert!(opt.value(0.5).unwrap() < opt.value(opt.threshold).unwrap());
        let at = opt.value(opt.threshold).unwrap();
        assert!((at - (opt.threshold - 1.0)).abs() < 1e-12);
        let abd = abandonment_option(0.05, 0.03, 0.2, 1.0).unwrap();
        assert!(abd.threshold < 1.0);
        let ee = entry_exit(0.05, 0.03, 0.2, 0.1, 1.0, 0.2).unwrap();
        assert!(ee.enter > ee.exit);
    }
}
