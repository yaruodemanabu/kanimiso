//! Tier 2 properties for the verified numeric kernels (AGENTS.md §6 / §8 PR 10).

use kanimiso::data::Matrix;
use kanimiso::hmm::{CosinePower, Emission, Gaussian, HiddenMarkovModel};
use kanimiso::metrics;
use kanimiso::special::{betainc_reg, chi2_cdf, f_cdf, norm_cdf, student_t_cdf};
use ojizou_san::Session;
use proptest::prelude::*;

fn finite_unit() -> impl Strategy<Value = f64> {
    (-8.0_f64..8.0).prop_filter("finite", |x| x.is_finite())
}

fn positive_shape() -> impl Strategy<Value = f64> {
    (0.2_f64..12.0).prop_filter("finite", |x| x.is_finite())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn norm_cdf_is_monotone_and_in_unit_interval(z1 in finite_unit(), z2 in finite_unit()) {
        let (lo, hi) = if z1 <= z2 { (z1, z2) } else { (z2, z1) };
        let a = norm_cdf(lo);
        let b = norm_cdf(hi);
        prop_assert!(a.is_finite() && b.is_finite());
        prop_assert!((0.0..=1.0).contains(&a));
        prop_assert!((0.0..=1.0).contains(&b));
        prop_assert!(a <= b + 1e-12);
    }

    #[test]
    fn betainc_reg_stays_in_unit_interval(
        a in positive_shape(),
        b in positive_shape(),
        x in 0.0_f64..1.0,
    ) {
        let f = betainc_reg(a, b, x);
        prop_assert!(f.is_finite());
        prop_assert!((-1e-12..=1.0 + 1e-12).contains(&f));
    }

    #[test]
    fn chi2_and_f_and_t_cdfs_stay_in_unit_interval(
        x in 0.0_f64..40.0,
        df in 1.0_f64..30.0,
        t in -8.0_f64..8.0,
    ) {
        let c = chi2_cdf(x, df);
        let f = f_cdf(x, df, df + 1.0);
        let s = student_t_cdf(t, df);
        prop_assert!((0.0..=1.0).contains(&c));
        prop_assert!((0.0..=1.0).contains(&f));
        prop_assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn gaussian_log_prob_is_finite_or_neg_inf(
        mean in -5.0_f64..5.0,
        var in 0.05_f64..4.0,
        y in -10.0_f64..10.0,
    ) {
        let e = Gaussian::univariate(mean, var);
        let lp = e.log_prob(&vec![y]);
        prop_assert!(lp.is_finite() || lp == f64::NEG_INFINITY);
        prop_assert!(!lp.is_nan());
        let outside = e.log_prob(&vec![y, y]);
        prop_assert_eq!(outside, f64::NEG_INFINITY);
    }

    #[test]
    fn cosine_power_is_neg_inf_outside_support(n in 0.0_f64..6.0, y in 1.1_f64..4.0) {
        let law = CosinePower::new(0.0, 1.0, n.max(0.0));
        prop_assert_eq!(law.log_prob(&y), f64::NEG_INFINITY);
        prop_assert_eq!(law.log_prob(&-y), f64::NEG_INFINITY);
    }

    #[test]
    fn hmm_transition_rows_stay_stochastic_after_one_em_step(
        p in 0.15_f64..0.85,
    ) {
        let start = vec![p, 1.0 - p];
        let trans = vec![vec![p, 1.0 - p], vec![1.0 - p, p]];
        let m = HiddenMarkovModel::new(
            start,
            trans,
            vec![
                Gaussian::univariate(-1.0, 0.4),
                Gaussian::univariate(1.0, 0.4),
            ],
        );
        let obs: Vec<Vec<f64>> = (-4..=4).map(|i| vec![i as f64 * 0.4]).collect();
        let fitted = m
            .fit(&obs, 1, &Session::new("prop", "em"))
            .unwrap()
            .value;
        for row in &fitted.transition {
            let s: f64 = row.iter().sum();
            prop_assert!((s - 1.0).abs() < 1e-9, "row sum {s}");
            prop_assert!(row.iter().all(|v| v.is_finite() && *v >= 0.0));
        }
    }

    #[test]
    fn euclidean_distances_are_nonnegative_and_self_zero(seed in 0u32..50) {
        let n = 3 + (seed % 3) as usize;
        let a = Matrix::from_fn(n, 2, |i, j| (i + j + seed as usize) as f64 * 0.1);
        let d = metrics::euclidean_distances(&a, &a, &Session::new("prop", "l2"))
            .unwrap()
            .value;
        for i in 0..n {
            prop_assert!(d.get(i, i).abs() <= 1e-12);
            for j in 0..n {
                prop_assert!(d.get(i, j) >= -1e-15);
            }
        }
    }
}
