//! Integration tests for the nonparametric Bayes catalogue.

use isuzu::datasets::{make_dp_gaussians, make_ibp_linear_gaussian};
use isuzu::npbayes::{
    dp_gaussian_mixture_gibbs, expected_crp_tables, expected_ibp_features,
    sample_bernoulli_process, sample_beta_process_finite, sample_crp_assignments,
    sample_ibp_sequential, sample_pitman_yor_crp, sample_stick_breaking, BetaProcessParams,
    IbpParams, PitmanYorParams, StickBreakingKind,
};
use isuzu::prelude::*;

#[test]
fn stick_breaking_is_a_probability_vector() {
    let mut rng = seed_rng(1);
    let sb =
        sample_stick_breaking(StickBreakingKind::Dirichlet { alpha: 2.0 }, 16, &mut rng).unwrap();
    let s: f64 = sb.weights.iter().sum();
    assert!((s - 1.0).abs() < 1e-12);
}

#[test]
fn crp_and_py_run() {
    let mut rng = seed_rng(2);
    let c = sample_crp_assignments(50, 1.2, &mut rng).unwrap();
    assert_eq!(c.assignments.len(), 50);
    assert!(!c.sizes.is_empty());
    let py = sample_pitman_yor_crp(50, PitmanYorParams::new(0.3, 1.0).unwrap(), &mut rng).unwrap();
    assert_eq!(py.assignments.len(), 50);
}

#[test]
fn beta_bernoulli_finite_and_ibp() {
    let mut rng = seed_rng(3);
    let (pi, _) =
        sample_beta_process_finite(BetaProcessParams::ibp(1.4).unwrap(), 10, &mut rng).unwrap();
    let z = sample_bernoulli_process(12, &pi, &mut rng).unwrap();
    assert_eq!(z.n, 12);
    let ibp = sample_ibp_sequential(15, IbpParams::new(1.5).unwrap(), &mut rng).unwrap();
    assert_eq!(ibp.n, 15);
    let ek = expected_crp_tables(20, 1.0).unwrap();
    let ef = expected_ibp_features(20, 1.5).unwrap();
    assert!(ek > 1.0 && ef > 1.0);
}

#[test]
fn dp_mixture_on_toy_blobs() {
    let (x, truth) = make_dp_gaussians(25, &[-3.5, 3.5], 0.3, 4).unwrap();
    let mut rng = seed_rng(4);
    let fit = dp_gaussian_mixture_gibbs(&x, 0.4, 0.3, 0.0, 3.0, 25, &mut rng).unwrap();
    assert!(fit.n_clusters >= 2);
    assert_eq!(fit.assignments.len(), truth.len());
}

#[test]
fn ibp_toy_dataset_shapes() {
    let (x, z, a) = make_ibp_linear_gaussian(18, 1.0, 0.2, 1.0, 2, 6).unwrap();
    assert_eq!(x.nrows(), 18);
    assert_eq!(x.ncols(), 2);
    assert_eq!(z.n, 18);
    assert_eq!(a.ncols(), 2);
}
