//! Malliavin weights for a call on GBM (toy).

use isuzu::error::Result;
use isuzu::malliavin::{bs_call_payoff, malliavin_greeks, moment_summary};
use isuzu::models::GeometricBrownianMotion;
use isuzu::rng::seed_rng;
use isuzu::sampling::Sampling;

fn main() -> Result<()> {
    let model = GeometricBrownianMotion::new(0.05, 0.2)?;
    let sampling = Sampling::from_terminal(1.0, 80)?;
    let mut rng = seed_rng(2);
    let g = malliavin_greeks(
        &model,
        &sampling,
        100.0,
        2000,
        &mut rng,
        bs_call_payoff(100.0),
    )?;
    println!(
        "MC+Malliavin call: price≈{:.3}  Δ≈{:.3}  Γ≈{:.4}  (n={})",
        g.price, g.delta, g.gamma, g.nsim
    );
    let mut rng = seed_rng(2);
    let m = moment_summary(&model, &sampling, 100.0, 1500, &mut rng)?;
    println!(
        "GBM moments: E[S]≈{:.2}  (closed form {:.2})  var≈{:.1}",
        m.mean,
        100.0 * 0.05_f64.exp(),
        m.variance
    );
    Ok(())
}
