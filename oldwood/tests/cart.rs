#![allow(clippy::float_cmp)]

use oldwood::{
    ClassificationCriterion, DecisionTreeClassifier, DecisionTreeRegressor, DenseMatrix, Error,
    MatrixView, NodeKind, RegressionCriterion, SplitContext, SplitStrategy, TreeOptions,
};

fn matrix(rows: usize, columns: usize, values: &[f64]) -> DenseMatrix {
    DenseMatrix::from_row_major(rows, columns, values.to_vec()).unwrap()
}

fn assert_same(left: f64, right: f64) {
    // Initial measurements on 2026-09-04 found zero ULP discrepancy for every
    // cross-path fixture, so these checks intentionally use no tolerance.
    assert_eq!(
        left.to_bits(),
        right.to_bits(),
        "{left:.17e} != {right:.17e}"
    );
}

// The oracle fixtures use small, finite values whose sum cannot overflow. Keep
// this deliberately simpler than the production midpoint so the brute-force
// checks do not share the implementation under test. `f64::midpoint` is newer
// than the crate's Rust 1.85 MSRV.
#[allow(unknown_lints, clippy::manual_midpoint)]
fn oracle_midpoint(lower: f64, upper: f64) -> f64 {
    (lower + upper) / 2.0
}

fn root_split<T>(nodes: &[oldwood::ArenaNode<T>]) -> (usize, f64, usize, usize) {
    match nodes[0].kind() {
        NodeKind::Split {
            feature,
            threshold,
            left,
            right,
        } => (*feature, *threshold, *left, *right),
        NodeKind::Leaf => panic!("expected root split"),
    }
}

#[test]
fn dense_matrix_checks_storage_and_supports_borrowed_views() {
    let dense = matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let borrowed = &dense;
    assert_eq!(borrowed.nrows(), 2);
    assert_eq!(borrowed.ncols(), 2);
    assert_eq!(borrowed.get(1, 0), 3.0);
    assert_eq!(dense.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert!(matches!(
        DenseMatrix::from_row_major(2, 3, vec![0.0; 5]),
        Err(Error::InvalidMatrixStorage { .. })
    ));
}

struct ColumnMajorView<'a> {
    rows: usize,
    columns: usize,
    values: &'a [f64],
}

impl MatrixView for ColumnMajorView<'_> {
    fn nrows(&self) -> usize {
        self.rows
    }

    fn ncols(&self) -> usize {
        self.columns
    }

    fn get(&self, row: usize, column: usize) -> f64 {
        self.values[column * self.rows + row]
    }
}

#[test]
fn custom_matrix_view_fits_without_copying_into_dense_storage() {
    let x = ColumnMajorView {
        rows: 4,
        columns: 2,
        values: &[0.0, 1.0, 2.0, 3.0, 8.0, 8.0, 8.0, 8.0],
    };
    let fitted = DecisionTreeClassifier::default()
        .fit(&x, &[3, 3, 9, 9], None)
        .unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), [3, 3, 9, 9]);
    assert_eq!(root_split(fitted.nodes()).0, 0);
}

#[test]
fn classifier_matches_analytical_weighted_leaf_probabilities() {
    let x = matrix(4, 1, &[0.0, 0.0, 1.0, 1.0]);
    let y = [7, 7, 9, 9];
    let weights = [1.0, 3.0, 2.0, 2.0];
    let fitted = DecisionTreeClassifier::default()
        .fit(&x, &y, Some(&weights))
        .unwrap();
    assert_eq!(fitted.classes(), &[7, 9]);
    assert_eq!(fitted.predict(&x).unwrap(), y);
    let probabilities = fitted.predict_proba(&x).unwrap();
    assert_eq!(
        probabilities.as_slice(),
        &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    );
    for row in 0..probabilities.nrows() {
        assert_eq!(probabilities.get(row, 0) + probabilities.get(row, 1), 1.0);
    }
}

#[test]
fn regressor_matches_analytical_weighted_means() {
    let x = matrix(4, 1, &[0.0, 0.0, 1.0, 1.0]);
    let y = [1.0, 3.0, 10.0, 14.0];
    let weights = [1.0, 3.0, 1.0, 1.0];
    let fitted = DecisionTreeRegressor::default()
        .fit(&x, &y, Some(&weights))
        .unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), vec![2.5, 2.5, 12.0, 12.0]);
    let (_, _, left, right) = root_split(fitted.nodes());
    assert_eq!(fitted.nodes()[left].weighted_sample_count(), 4.0);
    assert_eq!(fitted.nodes()[right].weighted_sample_count(), 2.0);
}

#[test]
fn classifier_root_matches_independent_brute_force() {
    let x = matrix(
        6,
        2,
        &[0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 3.0, 5.0, 4.0, 4.0, 5.0, 3.0],
    );
    let y = [0, 0, 1, 1, 1, 0];
    let weights = [1.0, 2.0, 1.0, 3.0, 1.0, 2.0];
    let options = TreeOptions {
        max_depth: Some(1),
        ..TreeOptions::default()
    };
    let fitted = DecisionTreeClassifier::new(ClassificationCriterion::Gini, options)
        .fit(&x, &y, Some(&weights))
        .unwrap();
    let expected = brute_classification_root(&x, &y, &weights);
    let actual = root_split(fitted.nodes());
    assert_eq!(actual.0, expected.0);
    assert_eq!(actual.1, expected.1);
    assert_same(fitted.nodes()[0].impurity_decrease(), expected.2);
}

#[test]
fn regressor_root_matches_independent_brute_force() {
    let x = matrix(
        6,
        2,
        &[0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0, 4.0, 5.0, 5.0, 4.0],
    );
    let y = [0.0, 0.5, 1.0, 8.0, 9.0, 9.5];
    let weights = [1.0, 1.0, 2.0, 1.0, 2.0, 1.0];
    let options = TreeOptions {
        max_depth: Some(1),
        ..TreeOptions::default()
    };
    let fitted = DecisionTreeRegressor::new(RegressionCriterion::SquaredError, options)
        .fit(&x, &y, Some(&weights))
        .unwrap();
    let expected = brute_regression_root(&x, &y, &weights);
    let actual = root_split(fitted.nodes());
    assert_eq!(actual.0, expected.0);
    assert_eq!(actual.1, expected.1);
    assert_same(fitted.nodes()[0].impurity_decrease(), expected.2);
}

#[test]
fn row_permutation_preserves_classifier_and_regressor() {
    let x = matrix(
        8,
        2,
        &[
            0.0, 4.0, 0.0, 4.0, 1.0, 3.0, 1.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0, 1.0,
        ],
    );
    let yc = [0, 0, 0, 1, 1, 1, 2, 2];
    let yr = [0.0, 0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 10.0];
    let weights = [1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0];
    let permutation = [6, 2, 7, 0, 4, 1, 5, 3];
    let xp = permute_matrix(&x, &permutation);
    let permuted_classes: Vec<usize> = permutation.iter().map(|&row| yc[row]).collect();
    let permuted_targets: Vec<f64> = permutation.iter().map(|&row| yr[row]).collect();
    let wp: Vec<f64> = permutation.iter().map(|&row| weights[row]).collect();

    let classifier = DecisionTreeClassifier::default();
    let left = classifier.fit(&x, &yc, Some(&weights)).unwrap();
    let right = classifier.fit(&xp, &permuted_classes, Some(&wp)).unwrap();
    assert_eq!(left.nodes(), right.nodes());
    assert_eq!(left.predict(&x).unwrap(), right.predict(&x).unwrap());

    let regressor = DecisionTreeRegressor::default();
    let left = regressor.fit(&x, &yr, Some(&weights)).unwrap();
    let right = regressor.fit(&xp, &permuted_targets, Some(&wp)).unwrap();
    assert_eq!(left.nodes(), right.nodes());
    assert_eq!(left.predict(&x).unwrap(), right.predict(&x).unwrap());
}

#[test]
fn positive_weight_scaling_preserves_fit() {
    let x = matrix(6, 1, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = [0, 0, 1, 1, 2, 2];
    let weights = [1.0, 2.0, 1.0, 3.0, 2.0, 1.0];
    let scaled: Vec<f64> = weights.iter().map(|weight| weight * 8.0).collect();
    let model = DecisionTreeClassifier::default();
    let base = model.fit(&x, &y, Some(&weights)).unwrap();
    let rescaled = model.fit(&x, &y, Some(&scaled)).unwrap();
    assert_eq!(base.predict(&x).unwrap(), rescaled.predict(&x).unwrap());
    for (left, right) in base
        .predict_proba(&x)
        .unwrap()
        .as_slice()
        .iter()
        .zip(rescaled.predict_proba(&x).unwrap().as_slice())
    {
        assert_same(*left, *right);
    }
    for (left, right) in base
        .feature_importances()
        .iter()
        .zip(rescaled.feature_importances())
    {
        assert_same(*left, *right);
    }
}

#[test]
fn integer_weights_match_duplicated_rows() {
    let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
    let y = [0.0, 1.0, 9.0, 10.0];
    let weights = [2.0, 1.0, 3.0, 1.0];
    let duplicated_x = matrix(7, 1, &[0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0]);
    let duplicated_y = [0.0, 0.0, 1.0, 9.0, 9.0, 9.0, 10.0];
    let model = DecisionTreeRegressor::default();
    let weighted = model.fit(&x, &y, Some(&weights)).unwrap();
    let duplicated = model.fit(&duplicated_x, &duplicated_y, None).unwrap();
    let probe = matrix(7, 1, &[-1.0, 0.0, 0.5, 1.5, 2.0, 2.5, 4.0]);
    for (left, right) in weighted
        .predict(&probe)
        .unwrap()
        .iter()
        .zip(duplicated.predict(&probe).unwrap())
    {
        assert_same(*left, right);
    }
}

#[test]
fn zero_weight_rows_have_no_training_effect() {
    let x = matrix(5, 1, &[-100.0, 0.0, 1.0, 2.0, 100.0]);
    let y = [99, 0, 0, 1, 88];
    let weights = [0.0, 1.0, 1.0, 1.0, 0.0];
    let fitted = DecisionTreeClassifier::default()
        .fit(&x, &y, Some(&weights))
        .unwrap();
    assert_eq!(fitted.classes(), &[0, 1]);
    assert_eq!(fitted.nodes()[0].sample_count(), 3);
}

#[test]
fn entropy_runtime_criterion_is_used() {
    let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
    let y = [0, 0, 1, 1];
    let options = TreeOptions {
        max_depth: Some(1),
        ..TreeOptions::default()
    };
    let fitted = DecisionTreeClassifier::new(ClassificationCriterion::Entropy, options)
        .fit(&x, &y, None)
        .unwrap();
    assert_eq!(fitted.nodes()[0].impurity(), 1.0);
    assert_eq!(fitted.nodes()[0].impurity_decrease(), 1.0);
}

#[test]
fn lexicographic_tie_break_prefers_lower_feature_then_threshold() {
    let x = matrix(4, 2, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let y = [0, 0, 1, 1];
    let options = TreeOptions {
        max_depth: Some(1),
        ..TreeOptions::default()
    };
    let fitted = DecisionTreeClassifier::new(ClassificationCriterion::Gini, options)
        .fit(&x, &y, None)
        .unwrap();
    let (feature, threshold, _, _) = root_split(fitted.nodes());
    assert_eq!(feature, 0);
    assert_eq!(threshold, 1.5);
}

#[test]
fn options_control_leaf_constraints_and_feature_prefix() {
    let x = matrix(4, 2, &[0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
    let y = [0, 0, 1, 1];
    let prefix_only = TreeOptions {
        max_features: Some(1),
        ..TreeOptions::default()
    };
    let fitted = DecisionTreeClassifier::new(ClassificationCriterion::Gini, prefix_only)
        .fit(&x, &y, None)
        .unwrap();
    assert!(matches!(fitted.nodes()[0].kind(), NodeKind::Leaf));

    let too_heavy = TreeOptions {
        min_weight_leaf: 3.0,
        ..TreeOptions::default()
    };
    let fitted = DecisionTreeClassifier::new(ClassificationCriterion::Gini, too_heavy)
        .fit(&x, &y, None)
        .unwrap();
    assert!(matches!(fitted.nodes()[0].kind(), NodeKind::Leaf));

    for options in [
        TreeOptions {
            max_depth: Some(0),
            ..TreeOptions::default()
        },
        TreeOptions {
            min_samples_split: 5,
            ..TreeOptions::default()
        },
        TreeOptions {
            min_samples_leaf: 3,
            ..TreeOptions::default()
        },
        TreeOptions {
            min_impurity_decrease: 0.500_000_000_000_000_1,
            ..TreeOptions::default()
        },
    ] {
        let fitted = DecisionTreeClassifier::new(ClassificationCriterion::Gini, options)
            .fit(&x, &y, None)
            .unwrap();
        assert!(matches!(fitted.nodes()[0].kind(), NodeKind::Leaf));
    }
}

#[test]
fn minimum_impurity_decrease_accepts_the_exact_boundary() {
    let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
    let options = TreeOptions {
        min_impurity_decrease: 0.5,
        ..TreeOptions::default()
    };
    let fitted = DecisionTreeClassifier::new(ClassificationCriterion::Gini, options)
        .fit(&x, &[0, 0, 1, 1], None)
        .unwrap();
    assert!(matches!(fitted.nodes()[0].kind(), NodeKind::Split { .. }));
    assert_eq!(fitted.nodes()[0].impurity_decrease(), 0.5);
}

#[test]
fn arena_apply_and_feature_importance_are_consistent() {
    let x = matrix(
        6,
        2,
        &[0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0, 5.0, 4.0],
    );
    let y = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
    let fitted = DecisionTreeRegressor::default().fit(&x, &y, None).unwrap();
    let leaves = fitted.apply(&x).unwrap();
    assert!(leaves
        .iter()
        .all(|&leaf| matches!(fitted.nodes()[leaf].kind(), NodeKind::Leaf)));
    let importance_sum: f64 = fitted.feature_importances().iter().sum();
    assert_same(importance_sum, 1.0);
    assert_eq!(fitted.feature_importances()[0], 1.0);
    assert_eq!(fitted.feature_importances()[1], 0.0);
}

#[test]
fn safe_midpoint_handles_extreme_opposite_sign_features() {
    let x = matrix(4, 1, &[-f64::MAX, -1.0, 1.0, f64::MAX]);
    let y = [0, 0, 1, 1];
    let options = TreeOptions {
        max_depth: Some(1),
        ..TreeOptions::default()
    };
    let fitted = DecisionTreeClassifier::new(ClassificationCriterion::Gini, options)
        .fit(&x, &y, None)
        .unwrap();
    let (_, threshold, _, _) = root_split(fitted.nodes());
    assert_eq!(threshold, 0.0);
    assert!(threshold.is_finite());
}

#[test]
fn adjacent_floats_are_separated_by_a_half_open_classification_threshold() {
    let lower = 1.0_f64;
    let upper = f64::from_bits(lower.to_bits() + 1);
    let x = matrix(2, 1, &[lower, upper]);
    let fitted = DecisionTreeClassifier::default()
        .fit(&x, &[0, 1], None)
        .unwrap();
    let (_, threshold, _, _) = root_split(fitted.nodes());
    assert_eq!(threshold, lower);
    assert_eq!(fitted.predict(&x).unwrap(), vec![0, 1]);
}

#[test]
fn adjacent_floats_are_separated_by_a_half_open_regression_threshold() {
    let lower = 1.0_f64;
    let upper = f64::from_bits(lower.to_bits() + 1);
    let x = matrix(2, 1, &[lower, upper]);
    let fitted = DecisionTreeRegressor::default()
        .fit(&x, &[-2.0, 3.0], None)
        .unwrap();
    let (_, threshold, _, _) = root_split(fitted.nodes());
    assert_eq!(threshold, lower);
    assert_eq!(fitted.predict(&x).unwrap(), vec![-2.0, 3.0]);
}

#[test]
fn leaf_class_tie_break_respects_sub_ulp_weight_mass() {
    let x = matrix(3, 1, &[0.0, 0.0, 0.0]);
    let tail = 2.0f64.powi(-53);
    let fitted = DecisionTreeClassifier::default()
        .fit(&x, &[0, 1, 1], Some(&[1.0, 1.0, tail]))
        .expect("fit weighted leaf");
    assert_eq!(fitted.predict(&x).expect("predict"), vec![1; 3]);
    let probabilities = fitted.predict_proba(&x).expect("probability");
    for row in 0..3 {
        assert!(probabilities.get(row, 1) > probabilities.get(row, 0));
        assert_eq!(probabilities.get(row, 0) + probabilities.get(row, 1), 1.0);
    }
}

#[test]
fn regression_leaf_preserves_a_finite_cancellation_mean() {
    let large = 8.0e153;
    let x = matrix(3, 1, &[0.0, 0.0, 0.0]);
    let fitted = DecisionTreeRegressor::default()
        .fit(&x, &[large, 1.0, -large], None)
        .expect("finite weighted moments");
    assert_eq!(fitted.predict(&x).expect("predict"), vec![1.0 / 3.0; 3]);
}

#[test]
fn regression_leaf_keeps_a_subnormal_weighted_mean() {
    let tiny = f64::from_bits(1);
    let x = matrix(2, 1, &[0.0, 0.0]);
    let fitted = DecisionTreeRegressor::default()
        .fit(&x, &[0.0, f64::MAX], Some(&[f64::MAX, tiny]))
        .expect("representable extreme weighted moments");
    assert_eq!(fitted.predict(&x).expect("predict"), vec![tiny; 2]);
    assert_eq!(fitted.nodes()[0].impurity(), f64::MAX * tiny);
}

#[test]
fn large_offset_regression_remains_finite() {
    let x = matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
    let y = [1e15 + 1.0, 1e15 + 2.0, 1e15 + 3.0, 1e15 + 4.0];
    let fitted = DecisionTreeRegressor::default().fit(&x, &y, None).unwrap();
    assert!(fitted
        .nodes()
        .iter()
        .all(|node| node.impurity().is_finite() && node.value().is_finite()));
}

#[test]
fn invalid_inputs_are_reported_before_training() {
    let good = matrix(2, 1, &[0.0, 1.0]);
    assert!(matches!(
        DecisionTreeClassifier::default().fit(&good, &[0], None),
        Err(Error::TargetLength { .. })
    ));
    assert!(matches!(
        DecisionTreeClassifier::default().fit(&good, &[0, 1], Some(&[1.0])),
        Err(Error::WeightLength { .. })
    ));
    assert!(matches!(
        DecisionTreeClassifier::default().fit(&good, &[0, 1], Some(&[1.0, -1.0])),
        Err(Error::InvalidWeight { row: 1 })
    ));
    assert_eq!(
        DecisionTreeClassifier::default()
            .fit(&good, &[0, 1], Some(&[0.0, 0.0]))
            .unwrap_err(),
        Error::NoPositiveWeight
    );
    let non_finite = matrix(2, 1, &[0.0, f64::NAN]);
    assert!(matches!(
        DecisionTreeClassifier::default().fit(&non_finite, &[0, 1], None),
        Err(Error::NonFiniteFeature { row: 1, column: 0 })
    ));
    assert!(matches!(
        DecisionTreeRegressor::default().fit(&good, &[0.0, f64::INFINITY], None),
        Err(Error::NonFiniteTarget { row: 1 })
    ));
    assert!(matches!(
        DecisionTreeClassifier::default().fit(
            &DenseMatrix::from_row_major(0, 1, vec![]).unwrap(),
            &[],
            None
        ),
        Err(Error::EmptyTrainingData)
    ));
    assert!(matches!(
        DecisionTreeClassifier::default().fit(
            &DenseMatrix::from_row_major(2, 0, vec![]).unwrap(),
            &[0, 1],
            None
        ),
        Err(Error::EmptyFeatures)
    ));
    assert!(matches!(
        DecisionTreeClassifier::default().fit(&good, &[0, 1], Some(&[1.0, f64::INFINITY])),
        Err(Error::InvalidWeight { row: 1 })
    ));
}

fn assert_invalid_option(options: TreeOptions, name: &'static str) {
    let x = matrix(2, 1, &[0.0, 1.0]);
    assert!(matches!(
        DecisionTreeClassifier::new(ClassificationCriterion::Gini, options).fit(
            &x,
            &[0, 1],
            None
        ),
        Err(Error::InvalidOption { name: actual, .. }) if actual == name
    ));
}

#[test]
fn every_tree_option_domain_is_validated() {
    assert_invalid_option(
        TreeOptions {
            min_samples_split: 1,
            ..TreeOptions::default()
        },
        "min_samples_split",
    );
    assert_invalid_option(
        TreeOptions {
            min_samples_leaf: 0,
            ..TreeOptions::default()
        },
        "min_samples_leaf",
    );
    assert_invalid_option(
        TreeOptions {
            min_weight_leaf: f64::NAN,
            ..TreeOptions::default()
        },
        "min_weight_leaf",
    );
    assert_invalid_option(
        TreeOptions {
            min_impurity_decrease: -1.0,
            ..TreeOptions::default()
        },
        "min_impurity_decrease",
    );
    assert_invalid_option(
        TreeOptions {
            max_features: Some(0),
            ..TreeOptions::default()
        },
        "max_features",
    );
    assert_invalid_option(
        TreeOptions {
            max_features: Some(2),
            ..TreeOptions::default()
        },
        "max_features",
    );
}

#[test]
fn numerical_overflow_is_an_error_not_a_non_finite_model() {
    let x = matrix(2, 1, &[0.0, 1.0]);
    let error = DecisionTreeRegressor::default()
        .fit(&x, &[-f64::MAX, f64::MAX], None)
        .unwrap_err();
    assert!(matches!(error, Error::NumericalOverflow { .. }));
}

#[derive(Default)]
struct RecordingStrategy {
    contexts: Vec<SplitContext>,
}

impl SplitStrategy for RecordingStrategy {
    fn features(&mut self, context: SplitContext, total_features: usize, output: &mut Vec<usize>) {
        self.contexts.push(context);
        output.extend((0..total_features).rev());
        output.push(0);
    }

    fn thresholds(
        &mut self,
        _context: SplitContext,
        _feature: usize,
        unique_values: &[f64],
        output: &mut Vec<f64>,
    ) {
        for values in unique_values.windows(2).rev() {
            output.push(oracle_midpoint(values[0], values[1]));
        }
        if let Some(&threshold) = output.first() {
            output.push(threshold);
        }
    }
}

#[test]
fn strategy_candidates_are_canonicalized_and_context_ids_are_stable() {
    let x = matrix(4, 2, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let y = [0, 0, 1, 1];
    let options = TreeOptions {
        max_depth: Some(1),
        ..TreeOptions::default()
    };
    let mut strategy = RecordingStrategy::default();
    let fitted = DecisionTreeClassifier::new(ClassificationCriterion::Gini, options)
        .fit_with_strategy(&x, &y, None, &mut strategy)
        .unwrap();
    assert_eq!(root_split(fitted.nodes()).0, 0);
    assert_eq!(
        strategy.contexts,
        vec![SplitContext {
            node_id: 0,
            depth: 0,
            sample_count: 4
        }]
    );
}

struct BadFeature;

impl SplitStrategy for BadFeature {
    fn features(&mut self, _context: SplitContext, total_features: usize, output: &mut Vec<usize>) {
        output.push(total_features);
    }

    fn thresholds(
        &mut self,
        _context: SplitContext,
        _feature: usize,
        _unique_values: &[f64],
        _output: &mut Vec<f64>,
    ) {
    }
}

struct BadThreshold;

impl SplitStrategy for BadThreshold {
    fn features(
        &mut self,
        _context: SplitContext,
        _total_features: usize,
        output: &mut Vec<usize>,
    ) {
        output.push(0);
    }

    fn thresholds(
        &mut self,
        _context: SplitContext,
        _feature: usize,
        _unique_values: &[f64],
        output: &mut Vec<f64>,
    ) {
        output.push(f64::NAN);
    }
}

#[test]
fn invalid_strategy_output_is_rejected() {
    let x = matrix(2, 1, &[0.0, 1.0]);
    let y = [0, 1];
    assert!(matches!(
        DecisionTreeClassifier::default().fit_with_strategy(&x, &y, None, &mut BadFeature),
        Err(Error::InvalidStrategyFeature {
            feature: 1,
            columns: 1
        })
    ));
    assert!(matches!(
        DecisionTreeClassifier::default().fit_with_strategy(&x, &y, None, &mut BadThreshold),
        Err(Error::InvalidStrategyThreshold { feature: 0 })
    ));
}

#[test]
fn prediction_validates_shape_and_values() {
    let x = matrix(2, 1, &[0.0, 1.0]);
    let fitted = DecisionTreeClassifier::default()
        .fit(&x, &[0, 1], None)
        .unwrap();
    let wrong_shape = matrix(1, 2, &[0.0, 0.0]);
    assert!(matches!(
        fitted.predict(&wrong_shape),
        Err(Error::FeatureCount {
            expected: 1,
            actual: 2
        })
    ));
    let non_finite = matrix(1, 1, &[f64::NAN]);
    assert!(matches!(
        fitted.predict(&non_finite),
        Err(Error::NonFiniteFeature { .. })
    ));
}

fn permute_matrix(source: &DenseMatrix, permutation: &[usize]) -> DenseMatrix {
    let mut values = Vec::with_capacity(source.ncols() * permutation.len());
    for &row in permutation {
        for column in 0..source.ncols() {
            values.push(source.get(row, column));
        }
    }
    matrix(permutation.len(), source.ncols(), &values)
}

fn brute_classification_root(
    matrix: &DenseMatrix,
    targets: &[usize],
    weights: &[f64],
) -> (usize, f64, f64) {
    let parent = direct_gini(targets, weights);
    let mut best: Option<(usize, f64, f64)> = None;
    for feature in 0..matrix.ncols() {
        let mut unique: Vec<f64> = (0..matrix.nrows())
            .map(|row| matrix.get(row, feature))
            .collect();
        unique.sort_by(f64::total_cmp);
        unique.dedup();
        for pair in unique.windows(2) {
            let threshold = oracle_midpoint(pair[0], pair[1]);
            let mut left_targets = Vec::new();
            let mut left_weights = Vec::new();
            let mut right_targets = Vec::new();
            let mut right_weights = Vec::new();
            for row in 0..matrix.nrows() {
                if matrix.get(row, feature) <= threshold {
                    left_targets.push(targets[row]);
                    left_weights.push(weights[row]);
                } else {
                    right_targets.push(targets[row]);
                    right_weights.push(weights[row]);
                }
            }
            let left_weight: f64 = left_weights.iter().sum();
            let right_weight: f64 = right_weights.iter().sum();
            let total = left_weight + right_weight;
            let gain = parent
                - left_weight / total * direct_gini(&left_targets, &left_weights)
                - right_weight / total * direct_gini(&right_targets, &right_weights);
            let candidate = (feature, threshold, gain);
            if best.is_none_or(|current| {
                gain > current.2
                    || (gain == current.2 && (feature, threshold) < (current.0, current.1))
            }) {
                best = Some(candidate);
            }
        }
    }
    best.unwrap()
}

fn direct_gini(targets: &[usize], weights: &[f64]) -> f64 {
    let mut classes = targets.to_vec();
    classes.sort_unstable();
    classes.dedup();
    let total: f64 = weights.iter().sum();
    1.0 - classes
        .iter()
        .map(|class| {
            let weight: f64 = targets
                .iter()
                .zip(weights)
                .filter_map(|(target, weight)| (target == class).then_some(*weight))
                .sum();
            (weight / total).powi(2)
        })
        .sum::<f64>()
}

fn brute_regression_root(
    matrix: &DenseMatrix,
    targets: &[f64],
    weights: &[f64],
) -> (usize, f64, f64) {
    let parent = direct_variance(targets, weights);
    let mut best: Option<(usize, f64, f64)> = None;
    for feature in 0..matrix.ncols() {
        let mut unique: Vec<f64> = (0..matrix.nrows())
            .map(|row| matrix.get(row, feature))
            .collect();
        unique.sort_by(f64::total_cmp);
        unique.dedup();
        for pair in unique.windows(2) {
            let threshold = oracle_midpoint(pair[0], pair[1]);
            let mut left_targets = Vec::new();
            let mut left_weights = Vec::new();
            let mut right_targets = Vec::new();
            let mut right_weights = Vec::new();
            for row in 0..matrix.nrows() {
                if matrix.get(row, feature) <= threshold {
                    left_targets.push(targets[row]);
                    left_weights.push(weights[row]);
                } else {
                    right_targets.push(targets[row]);
                    right_weights.push(weights[row]);
                }
            }
            let left_weight: f64 = left_weights.iter().sum();
            let right_weight: f64 = right_weights.iter().sum();
            let total = left_weight + right_weight;
            let gain = parent
                - left_weight / total * direct_variance(&left_targets, &left_weights)
                - right_weight / total * direct_variance(&right_targets, &right_weights);
            let candidate = (feature, threshold, gain);
            if best.is_none_or(|current| {
                gain > current.2
                    || (gain == current.2 && (feature, threshold) < (current.0, current.1))
            }) {
                best = Some(candidate);
            }
        }
    }
    best.unwrap()
}

fn direct_variance(targets: &[f64], weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    let mean: f64 = targets
        .iter()
        .zip(weights)
        .map(|(target, weight)| target * weight)
        .sum::<f64>()
        / total;
    targets
        .iter()
        .zip(weights)
        .map(|(target, weight)| weight * (target - mean).powi(2))
        .sum::<f64>()
        / total
}
