//! Dirichlet process mixture and Indian buffet process on toy data.

use isuzu::datasets::{make_dp_gaussians, make_ibp_linear_gaussian};
use isuzu::npbayes::{
    dp_gaussian_mixture_gibbs, sample_ibp_sequential, sample_stick_breaking, IbpParams,
    StickBreakingKind,
};
use isuzu::prelude::*;

fn main() -> isuzu::Result<()> {
    let mut rng = seed_rng(7);
    let sb = sample_stick_breaking(StickBreakingKind::Dirichlet { alpha: 1.5 }, 12, &mut rng)?;
    println!(
        "Sethuraman stick-breaking (K=12) first 5 weights: {:?}",
        &sb.weights[..5]
    );

    let z = sample_ibp_sequential(20, IbpParams::new(2.0)?, &mut rng)?;
    println!("IBP(20, α=2) features K = {}", z.k);

    let (x, truth) = make_dp_gaussians(30, &[-3.0, 3.0], 0.35, 3)?;
    let fit = dp_gaussian_mixture_gibbs(&x, 0.5, 0.35, 0.0, 3.0, 30, &mut rng)?;
    println!(
        "DP mixture: true 2 blobs, inferred K = {}, means = {:?}",
        fit.n_clusters, fit.means
    );
    let _ = truth;

    let (xx, zz, _) = make_ibp_linear_gaussian(24, 1.0, 0.2, 1.0, 2, 5)?;
    println!("linear-Gaussian IBP toy: N={}, true K={}", xx.nrows(), zz.k);
    Ok(())
}
