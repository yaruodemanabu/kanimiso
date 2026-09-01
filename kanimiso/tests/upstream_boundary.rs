//! Wrappers must reproduce wormhole / coronel / jelly-wave values (AGENTS.md §3.4).

use kanimiso::{data::Matrix, data::Vector, metrics, tslearn};
use ojizou_san::Session;

#[test]
fn dtw_matches_jelly_wave() {
    // measured 2026-09-01: |wrapper − jelly_wave::dtw| = 0 (tol 1e-15)
    let a = Vector::from_slice(&[0.0, 1.0, 2.0, 1.0]);
    let b = Vector::from_slice(&[0.0, 2.0, 1.0]);
    let got = tslearn::dtw(&a, &b, &Session::new("wave", "dtw"))
        .unwrap()
        .value;
    let want = jelly_wave::dtw(a.as_slice(), b.as_slice()).unwrap();
    assert!((got - want).abs() <= 1e-15, "{got} vs {want}");
}

#[test]
fn erp_matches_jelly_wave() {
    let a = Vector::from_slice(&[0.0, 1.0, 2.0]);
    let b = Vector::from_slice(&[0.0, 2.0]);
    let got = tslearn::erp(&a, &b, 0.0, &Session::new("wave", "erp"))
        .unwrap()
        .value;
    let want = jelly_wave::erp_distance(a.as_slice(), b.as_slice(), 0.0).unwrap();
    assert!((got - want).abs() <= 1e-15, "{got} vs {want}");
}

#[test]
fn manhattan_matches_wormhole() {
    let a = Matrix::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
    let b = Matrix::from_fn(2, 3, |i, j| ((i + 1) * j) as f64);
    let got = metrics::manhattan_distances(&a, &b, &Session::new("m", "l1"))
        .unwrap()
        .value;
    let want =
        wormhole::metrics::pairwise(a.inner(), b.inner(), wormhole::metrics::Metric::Manhattan)
            .unwrap();
    assert!((got.get(0, 1) - want[(0, 1)]).abs() <= 1e-15);
}
