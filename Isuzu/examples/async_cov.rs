//! Hayashi-Yoshida covariance and lead-lag on Poisson-sampled Brownian motions.

use isuzu::prelude::*;

fn main() -> Result<()> {
    let model = FnSde::new(
        2,
        2,
        |_t, _x, a| {
            a[0] = 0.0;
            a[1] = 0.0;
        },
        |_t, _x, s| {
            // σ σᵀ has off-diagonal 0.6 (unit vols)
            s[0] = 1.0;
            s[1] = 0.0;
            s[2] = 0.6;
            s[3] = 0.8;
        },
    );
    let sampling = Sampling::from_terminal(1.0, 2000)?;
    let mut rng = seed_rng(11);
    let path = simulate(
        &model,
        &sampling,
        &[0.0, 0.0],
        &mut rng,
        &SimConfig::default(),
    )?;
    let data = poisson_random_sampling(&path, &[0.25, 0.3], &mut rng)?;
    let est = cce(&data)?;
    println!("Hayashi-Yoshida covariance:");
    print_mat(&est.cov);
    println!("correlation:");
    print_mat(&est.corr);

    let grid = lead_lag_grid(-0.05, 0.05, 21)?;
    let ll = lead_lag(&data.series[0], &data.series[1], &grid)?;
    println!(
        "lead-lag θ̂ = {:.4}  (HY = {:.4}, corr = {:.3})",
        ll.theta, ll.hy_at_theta, ll.corr_at_theta
    );
    Ok(())
}

fn print_mat(a: &Mat<f64>) {
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            print!(" {:8.4}", a[(i, j)]);
        }
        println!();
    }
}
