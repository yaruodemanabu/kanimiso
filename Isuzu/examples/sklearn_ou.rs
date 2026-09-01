//! scikit-learn-style QMLE estimator on a labelled OU toy.

use isuzu::api::{recover, QmleEstimator};
use isuzu::datasets::make_ou;
use isuzu::error::Result;

fn main() -> Result<()> {
    let toy = make_ou(1.3, 0.15, 0.4, 10.0, 2000, 7)?;
    let mut est = QmleEstimator::new(toy.model.clone(), vec![0.8, 0.0, 0.6])
        .bounds(vec![0.05, -1.0, 0.05], vec![4.0, 2.0, 2.0]);
    let report = recover(&mut est, &toy.path, &toy.truth)?;
    println!("names  {:?}", report.names);
    println!("truth  {:?}", report.truth);
    println!("fitted {:?}", report.fitted);
    println!("|err|  {:?}", report.abs_error);
    Ok(())
}
