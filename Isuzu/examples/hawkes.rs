//! Simulate an exponential Hawkes process and recover parameters by MLE.

use isuzu::prelude::*;

fn main() -> Result<()> {
    let truth = ExponentialHawkes::new(0.6, 0.5, 1.4)?;
    let mut rng = seed_rng(3);
    let arrivals = truth.simulate(0.0, 200.0, &mut rng)?;
    println!(
        "simulated {} events; stationary intensity ≈ {:.3}",
        arrivals.len(),
        truth.stationary_intensity()?
    );
    let (fit, ll) = ExponentialHawkes::mle(&arrivals, 0.0, 200.0, [0.5, 0.4, 1.2])?;
    println!(
        "true (μ, α, β) = ({:.3}, {:.3}, {:.3})",
        truth.mu, truth.alpha, truth.beta
    );
    println!(
        "mle  (μ, α, β) = ({:.3}, {:.3}, {:.3})  loglik = {:.1}",
        fit.mu, fit.alpha, fit.beta, ll
    );
    Ok(())
}
