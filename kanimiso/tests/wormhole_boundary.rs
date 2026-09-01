//! Port of wormhole `tests/kanimiso_integration.rs` (AGENTS.md §3.1).
//!
//! `Matrix::inner()` is passed to `wormhole::solve_samples` and the plan is
//! wrapped with `Matrix::from_faer`. No `Vec<Vec<f64>>` copy.

use kanimiso::{Matrix, Vector};
use wormhole::{solve_samples, Metric, SolveOptions};

#[test]
fn kanimiso_faer_storage_is_a_zero_copy_wormhole_boundary() {
    let samples = Matrix::from_fn(3, 2, |i, j| [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]][i][j]);
    let weights = Vector::from_slice(&[0.2, 0.3, 0.5]);

    let solved = solve_samples(
        samples.inner(),
        samples.inner(),
        Some(weights.as_slice()),
        Some(weights.as_slice()),
        Metric::SquaredEuclidean,
        SolveOptions::default(),
    )
    .unwrap();

    assert!(solved.value.abs() < 1e-12);
    let plan = Matrix::from_faer(solved.plan);
    assert_eq!(plan.shape(), (3, 3));
    for i in 0..3 {
        assert!((plan.get(i, i) - weights[i]).abs() < 1e-12);
    }
}
