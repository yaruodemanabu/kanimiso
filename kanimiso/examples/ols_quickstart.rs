use kanimiso::data::{Matrix, Vector};
use kanimiso::linear_model::LinearRegression;
use kanimiso::log::Session;
use kanimiso::traits::Fit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let noise = [
        0.14, 0.31, -0.07, 0.22, -0.29, -0.11, 0.05, -0.35, 0.28, 0.09, -0.18, 0.34, -0.03, -0.24,
        0.12, -0.32, 0.26, -0.08, 0.19, -0.15,
    ];
    let x = Matrix::from_fn(noise.len(), 1, |i, _| i as f64);
    let y = Vector::from_iter(
        noise
            .iter()
            .enumerate()
            .map(|(i, e)| 1.0 + 2.0 * i as f64 + e),
    );

    let session = Session::new("linear_regression", "fit");
    let mut estimator = LinearRegression::new();
    let fitted = estimator.fit(&x, &y, &session)?;

    println!("intercept = {:.6}", fitted.value.intercept);
    println!("slope     = {:.6}", fitted.value.coef[0]);
    println!("R²        = {:.6}", fitted.value.r2);
    println!("quality issues = {}", fitted.report.issues().len());
    println!("session events = {}", session.ledger().len());

    for issue in fitted.report.issues() {
        eprintln!("quality: {issue}");
    }

    Ok(())
}
