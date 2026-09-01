//! Self-financing delta hedges and replication error.

use crate::error::{Error, Result};
use crate::finance::black_scholes::{call, BlackScholesMarket};
use crate::path::Path;

/// Discrete self-financing strategy along a stock / cash path.
#[derive(Clone, Debug)]
pub struct HedgePath {
    pub holdings: Vec<f64>,
    pub cash: Vec<f64>,
    pub value: Vec<f64>,
    pub terminal_error: f64,
}

/// Rebalance a BS call delta at every node; financing is at a flat rate.
pub fn delta_hedge_call(
    path: &Path,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
) -> Result<HedgePath> {
    if path.dim() != 1 {
        return Err(Error::dim("delta hedge expects a univariate spot path"));
    }
    let n = path.n_nodes();
    let t_end = time;
    let mut holdings = vec![0.0; n];
    let mut cash = vec![0.0; n];
    let mut value = vec![0.0; n];
    let s0 = path.state(0)[0];
    let m0 = BlackScholesMarket::new(s0, rate, 0.0, vol, t_end)?;
    let p0 = call(&m0, strike)?;
    holdings[0] = p0.delta;
    cash[0] = p0.price - holdings[0] * s0;
    value[0] = p0.price;
    for i in 1..n {
        let dt = path.times()[i] - path.times()[i - 1];
        cash[i] = cash[i - 1] * (rate * dt).exp();
        let s = path.state(i)[0];
        let t_rem = (t_end - path.times()[i]).max(0.0);
        let m = BlackScholesMarket::new(s.max(1e-12), rate, 0.0, vol, t_rem.max(1e-16))?;
        let pr = call(&m, strike)?;
        let d = if t_rem <= 0.0 {
            if s > strike {
                1.0
            } else {
                0.0
            }
        } else {
            pr.delta
        };
        cash[i] -= (d - holdings[i - 1]) * s;
        holdings[i] = d;
        value[i] = holdings[i] * s + cash[i];
    }
    let claim = (path.terminal()[0] - strike).max(0.0);
    Ok(HedgePath {
        holdings,
        cash,
        terminal_error: value[n - 1] - claim,
        value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::GeometricBrownianMotion;
    use crate::rng::seed_rng;
    use crate::sampling::Sampling;
    use crate::scheme::Scheme;
    use crate::simulate::{simulate, SimConfig};

    #[test]
    fn hedge_error_small_on_fine_grid() {
        let model = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let samp = Sampling::from_terminal(1.0, 250).unwrap();
        let mut rng = seed_rng(4);
        let path = simulate(
            &model,
            &samp,
            &[100.0],
            &mut rng,
            &SimConfig {
                scheme: Scheme::Exact,
                ..SimConfig::default()
            },
        )
        .unwrap();
        let h = delta_hedge_call(&path, 100.0, 0.05, 0.2, 1.0).unwrap();
        assert!(h.terminal_error.abs() < 3.0, "err {}", h.terminal_error);
    }
}
