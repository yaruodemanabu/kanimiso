//! Exact OU simulation followed by quasi-maximum likelihood.

use isuzu::prelude::*;

fn main() -> Result<()> {
    let truth = OrnsteinUhlenbeck::new(1.5, 0.2, 0.35)?;
    let sampling = Sampling::from_terminal(15.0, 3000)?;
    let mut rng = seed_rng(7);
    let path = simulate_ou_exact(&truth, &sampling, 0.0, &mut rng)?;
    let fit = qmle(
        &truth,
        &path,
        &[1.0, 0.0, 0.5],
        Some(&[0.05, -2.0, 0.05]),
        Some(&[5.0, 2.0, 2.0]),
        OptOptions::default(),
    )?;
    println!(
        "true  (κ, θ, σ) = ({:.3}, {:.3}, {:.3})",
        truth.kappa, truth.theta, truth.sigma
    );
    println!(
        "qmle  (κ, θ, σ) = ({:.3}, {:.3}, {:.3})  AIC = {:.1}",
        fit.params[0],
        fit.params[1],
        fit.params[2],
        fit.aic()
    );
    Ok(())
}
