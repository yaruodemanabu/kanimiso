//! Simulate a CARMA(1,0)-Hawkes process and evaluate its dedicated likelihood.

use isuzu::datasets::make_carma_hawkes;
use isuzu::error::Result;
use isuzu::models::CarmaHawkes;

fn main() -> Result<()> {
    let (model, arr) = make_carma_hawkes(0.55, vec![1.3], vec![0.45], 40.0, 4)?;
    let ll = model.loglik(&arr, 0.0, 40.0)?;
    println!(
        "events = {}  loglik = {:.2}  stable = {}",
        arr.len(),
        ll,
        model.is_stable()
    );
    let (fit, ll2) = CarmaHawkes::mle(&arr, 0.0, 40.0, 1, 0, &[0.4, 1.0, 0.3])?;
    println!(
        "mle μ={:.3} a={:.3} b={:.3}  loglik={:.2}",
        fit.mu, fit.ar[0], fit.ma[0], ll2
    );
    Ok(())
}
