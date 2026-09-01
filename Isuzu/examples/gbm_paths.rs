//! Simulate geometric Brownian motion and print a short summary.

use isuzu::prelude::*;

fn main() -> Result<()> {
    let model = GeometricBrownianMotion::new(0.08, 0.25)?;
    let sampling = Sampling::from_terminal(1.0, 500)?;
    let mut rng = seed_rng(2026);
    let ens = simulate_n(
        &model,
        &sampling,
        &[100.0],
        200,
        &mut rng,
        &SimConfig::default(),
    )?;
    println!(
        "GBM ensemble: n={}  E[X_1]≈{:.3}  Var[X_1]≈{:.3}  (closed form E = {:.3})",
        ens.n_paths(),
        ens.terminal_mean(0)?,
        ens.terminal_var(0)?,
        100.0 * 0.08_f64.exp()
    );
    Ok(())
}
