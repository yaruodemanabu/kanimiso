use super::common::checked_sample_variance_from_m2;
use super::*;
use crate::data::{Matrix, Vector};
use crate::traits::{PartialFit, Predict, Transform};
use ojizou_san::{EventKind, Session};
use signlred::IssueCode;

fn has_incremental(session: &Session) -> bool {
    session
        .ledger()
        .events()
        .iter()
        .any(|event| event.kind == EventKind::IncrementalExplanation)
}

fn mean_state(model: &OnlineMean) -> (u64, u64, u64) {
    (model.n, model.mean.to_bits(), model.updates)
}

fn sum_state(model: &OnlineSum) -> (u64, u64, u64, u64) {
    (
        model.n,
        model.sum.to_bits(),
        model.compensation.to_bits(),
        model.updates,
    )
}

fn variance_state(model: &OnlineVar) -> (u64, u64, u64, u64, u64) {
    (
        model.n,
        model.mean.to_bits(),
        model.m2.to_bits(),
        model.m2_compensation.to_bits(),
        model.updates,
    )
}

fn covariance_state(model: &OnlineCovariance) -> (u64, u64, u64, u64, u64, u64) {
    (
        model.n,
        model.mean_x.to_bits(),
        model.mean_y.to_bits(),
        model.cross.to_bits(),
        model.cross_compensation.to_bits(),
        model.updates,
    )
}

fn autocorrelation_state(model: &OnlineAutoCorr) -> (Option<u64>, [u64; 10]) {
    (
        model.last.map(f64::to_bits),
        [
            model.n,
            model.mean_lagged.to_bits(),
            model.mean_current.to_bits(),
            model.cross.to_bits(),
            model.cross_compensation.to_bits(),
            model.lagged_m2.to_bits(),
            model.lagged_m2_compensation.to_bits(),
            model.current_m2.to_bits(),
            model.current_m2_compensation.to_bits(),
            model.updates,
        ],
    )
}

fn autocorrelation_numeric_state(model: &OnlineAutoCorr) -> (Option<u64>, [u64; 9]) {
    let (last, state) = autocorrelation_state(model);
    (
        last,
        [
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
            state[8],
        ],
    )
}

fn variance_threshold_state(
    model: &OnlineVarianceThreshold,
) -> (u64, u64, Vec<u64>, Vec<u64>, Vec<u64>, usize, u64) {
    (
        model.threshold.to_bits(),
        model.n_seen,
        model.mean.iter().map(|value| value.to_bits()).collect(),
        model.m2.iter().map(|value| value.to_bits()).collect(),
        model
            .m2_compensation
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        model.n_features,
        model.updates,
    )
}

fn variance_threshold_variance(model: &OnlineVarianceThreshold, column: usize) -> f64 {
    checked_sample_variance_from_m2(
        model.n_seen,
        model.m2[column],
        model.m2_compensation[column],
    )
    .unwrap()
    .unwrap()
}

#[test]
fn online_autocorrelation_matches_exact_rational_oracle() {
    let observations = [1.0, 4.0, 2.0, 8.0, 5.0, 7.0];
    let x = Matrix::from_row_major(observations.len(), 1, &observations);
    let session = Session::new("online-autocorrelation", "rational-oracle");
    let mut model = OnlineAutoCorr::new();
    assert!(model.score().is_nan());

    let result = model.partial_fit(&x, None, &session).unwrap();
    // Exact integer oracle: Sxx=30, Syy=114/5, Sxy=-1.
    let expected = -1.0 / 684.0_f64.sqrt();
    let error = (model.score() - expected).abs();
    eprintln!("online autocorrelation rational oracle error={error:.17e}");
    // Measured on 2026-09-03: 6.9389e-18; R9 limit is 4.1×.
    assert!(error <= 2.85e-17);
    assert_eq!(model.n, observations.len() as u64);
    assert_eq!(result.value.quality.effective_sample_size, 5.0);
    assert!(result.value.quality.still_identified);
    assert!(!result.value.quality.warmup);
}

#[test]
fn online_autocorrelation_warmup_and_constant_semantics_are_explicit() {
    let session = Session::new("online-autocorrelation", "warmup-and-constant");
    let mut model = OnlineAutoCorr::new();
    for (value, pairs) in [(1.0, 0.0), (2.0, 1.0)] {
        let result = model
            .partial_fit(&Matrix::from_row_major(1, 1, &[value]), None, &session)
            .unwrap();
        assert!(model.score().is_nan());
        assert_eq!(result.value.quality.effective_sample_size, pairs);
        assert!(result.value.quality.warmup);
        assert!(!result.value.quality.still_identified);
    }
    let third = model
        .partial_fit(&Matrix::from_row_major(1, 1, &[4.0]), None, &session)
        .unwrap();
    let perfect_error = (model.score() - 1.0).abs();
    eprintln!("online autocorrelation two-pair error={perfect_error:.17e}");
    // Measured on 2026-09-03: 1.1102e-16; R9 limit is 4.1×.
    assert!(perfect_error <= 4.55e-16);
    assert_eq!(third.value.quality.effective_sample_size, 2.0);
    assert!(!third.value.quality.warmup);
    assert!(third.value.quality.still_identified);

    let constant = Matrix::from_row_major(4, 1, &[f64::MAX; 4]);
    let mut constant_model = OnlineAutoCorr::new();
    let result = constant_model
        .partial_fit(&constant, None, &session)
        .unwrap();
    assert!(constant_model.score().is_nan());
    assert!(!result.value.quality.warmup);
    assert!(!result.value.quality.still_identified);
    assert!(result.report.contains(IssueCode::NearZeroVariance));
}

#[test]
fn online_autocorrelation_is_bitwise_partition_invariant() {
    let observations = [1.0, 4.0, 2.0, 8.0, 5.0, 7.0, -3.0];
    let all = Matrix::from_row_major(observations.len(), 1, &observations);
    let session = Session::new("online-autocorrelation", "partition-invariance");
    let mut whole = OnlineAutoCorr::new();
    let _ = whole.partial_fit(&all, None, &session).unwrap();
    let mut partitioned = OnlineAutoCorr::new();
    for values in [&observations[..1], &observations[1..4], &observations[4..]] {
        let batch = Matrix::from_row_major(values.len(), 1, values);
        let _ = partitioned.partial_fit(&batch, None, &session).unwrap();
    }
    assert_eq!(
        autocorrelation_numeric_state(&whole),
        autocorrelation_numeric_state(&partitioned)
    );
    assert_eq!(whole.score().to_bits(), partitioned.score().to_bits());
    assert_eq!(whole.updates, 1);
    assert_eq!(partitioned.updates, 3);
}

#[test]
fn online_autocorrelation_handles_representable_extreme_scale() {
    let base = [1.0, 2.0, 0.0, 3.0, -1.0];
    let scale = f64::MAX.sqrt() / 16.0;
    let scaled = base.map(|value| value * scale);
    let session = Session::new("online-autocorrelation", "extreme-scale");
    let mut base_model = OnlineAutoCorr::new();
    let mut scaled_model = OnlineAutoCorr::new();
    let _ = base_model
        .partial_fit(
            &Matrix::from_row_major(base.len(), 1, &base),
            None,
            &session,
        )
        .unwrap();
    let _ = scaled_model
        .partial_fit(
            &Matrix::from_row_major(scaled.len(), 1, &scaled),
            None,
            &session,
        )
        .unwrap();
    let error = (scaled_model.score() - base_model.score()).abs();
    eprintln!("online autocorrelation extreme-scale error={error:.17e}");
    assert!(scaled_model.score().is_finite());
    // Measured on 2026-09-03: exact agreement; R9 tolerance is zero.
    assert_eq!(error, 0.0);
}

#[test]
fn online_autocorrelation_failures_are_transactional() {
    let session = Session::new("online-autocorrelation", "transactional-failures");
    let mut model = OnlineAutoCorr::new();
    let _ = model
        .partial_fit(
            &Matrix::from_row_major(4, 1, &[1.0, 4.0, 2.0, 8.0]),
            None,
            &session,
        )
        .unwrap();
    let state = autocorrelation_state(&model);
    for (input, code) in [
        (
            Matrix::from_row_major(3, 1, &[5.0, f64::NAN, 6.0]),
            IssueCode::NonFiniteInput,
        ),
        (Matrix::zeros(0, 1), IssueCode::EmptyMatrix),
        (Matrix::zeros(1, 0), IssueCode::EmptyMatrix),
        (
            Matrix::from_row_major(2, 1, &[f64::MAX, -f64::MAX]),
            IssueCode::NumericalOverflow,
        ),
    ] {
        let error = model.partial_fit(&input, None, &session).unwrap_err();
        assert_eq!(error.primary.code, code);
        assert_eq!(autocorrelation_state(&model), state);
    }

    let mut underflow = OnlineAutoCorr::new();
    let tiny = Matrix::from_row_major(3, 1, &[0.0, f64::MIN_POSITIVE, 2.0 * f64::MIN_POSITIVE]);
    let error = underflow.partial_fit(&tiny, None, &session).unwrap_err();
    assert_eq!(error.primary.code, IssueCode::NumericalUnderflow);
    assert_eq!(
        autocorrelation_state(&underflow),
        autocorrelation_state(&OnlineAutoCorr::new())
    );

    let mut update_overflow = model.clone();
    update_overflow.updates = u64::MAX;
    let overflow_state = autocorrelation_state(&update_overflow);
    let error = update_overflow
        .partial_fit(&Matrix::from_row_major(1, 1, &[9.0]), None, &session)
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::InvalidParameter);
    assert_eq!(autocorrelation_state(&update_overflow), overflow_state);

    let mut observation_overflow = model.clone();
    observation_overflow.n = u64::MAX;
    let overflow_state = autocorrelation_state(&observation_overflow);
    let error = observation_overflow
        .partial_fit(&Matrix::from_row_major(1, 1, &[9.0]), None, &session)
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::InvalidParameter);
    assert_eq!(autocorrelation_state(&observation_overflow), overflow_state);
}

#[test]
fn online_variance_threshold_matches_exact_all_column_oracle() {
    let values = [
        1.0, 2.0, 0.0, 2.0, 2.0, 2.0, 3.0, 2.0, 0.0, 4.0, 2.0, 2.0, 5.0, 2.0, 0.0,
    ];
    let x = Matrix::from_row_major(5, 3, &values);
    let session = Session::new("online-variance-threshold", "rational-oracle");
    let mut model = OnlineVarianceThreshold::new(1.5);
    let result = model.partial_fit(&x, None, &session).unwrap();
    let expected = [5.0 / 2.0, 0.0, 6.0 / 5.0];
    let maximum_error = expected
        .into_iter()
        .enumerate()
        .map(|(column, expected)| (variance_threshold_variance(&model, column) - expected).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("online variance-threshold rational oracle max_error={maximum_error:.17e}");
    // Measured on 2026-09-03: exact agreement; R9 tolerance is zero.
    assert_eq!(maximum_error, 0.0);
    assert_eq!(model.n_seen, 5);
    assert_eq!(result.value.quality.effective_sample_size, 5.0);
    assert!(result.value.quality.still_identified);

    let transformed = model.transform(&x, &session).unwrap().value;
    assert_eq!(transformed.shape(), (5, 1));
    for row in 0..5 {
        assert_eq!(transformed.get(row, 0), x.get(row, 0));
    }
    let one_column = Matrix::from_fn(5, 1, |row, _| x.get(row, 2));
    let mut one_column_model = OnlineVarianceThreshold::new(1.0);
    let _ = one_column_model
        .partial_fit(&one_column, None, &session)
        .unwrap();
    assert_eq!(
        variance_threshold_variance(&one_column_model, 0).to_bits(),
        variance_threshold_variance(&model, 2).to_bits()
    );

    let first_column = Matrix::from_fn(5, 1, |row, _| x.get(row, 0));
    let mut strict = OnlineVarianceThreshold::new(2.5);
    let update = strict.partial_fit(&first_column, None, &session).unwrap();
    assert!(update.report.contains(IssueCode::NearZeroVariance));
    assert_eq!(
        strict
            .transform(&first_column, &session)
            .unwrap_err()
            .primary
            .code,
        IssueCode::MeaninglessFit
    );
}

#[test]
fn online_variance_threshold_is_bitwise_partition_invariant() {
    let values = [
        3.0, -1.0, 4.0, -1.0, 2.0, 0.0, 7.0, 5.0, -3.0, 2.5, -4.0, 8.0, -8.0, 3.0, 6.0, 11.0, 1.0,
        -2.0, 0.25, 9.25, 5.5,
    ];
    let all = Matrix::from_row_major(7, 3, &values);
    let batches = [
        Matrix::from_fn(2, 3, |row, column| all.get(row, column)),
        Matrix::from_fn(1, 3, |_, column| all.get(2, column)),
        Matrix::from_fn(4, 3, |row, column| all.get(row + 3, column)),
    ];
    let session = Session::new("online-variance-threshold", "partition-invariance");
    let mut whole = OnlineVarianceThreshold::new(0.5);
    let _ = whole.partial_fit(&all, None, &session).unwrap();
    let mut partitioned = OnlineVarianceThreshold::new(0.5);
    for batch in &batches {
        let _ = partitioned.partial_fit(batch, None, &session).unwrap();
    }
    let whole_updates = whole.updates;
    let partitioned_updates = partitioned.updates;
    whole.updates = 0;
    partitioned.updates = 0;
    assert_eq!(
        variance_threshold_state(&whole),
        variance_threshold_state(&partitioned)
    );
    assert_eq!(whole_updates, 1);
    assert_eq!(partitioned_updates, 3);

    let whole_output = whole.transform(&all, &session).unwrap().value;
    let partitioned_output = partitioned.transform(&all, &session).unwrap().value;
    assert_eq!(whole_output.shape(), partitioned_output.shape());
    for row in 0..whole_output.nrows() {
        for column in 0..whole_output.ncols() {
            assert_eq!(
                whole_output.get(row, column).to_bits(),
                partitioned_output.get(row, column).to_bits()
            );
        }
    }
}

#[test]
fn online_variance_threshold_warmup_constant_and_extreme_semantics() {
    let session = Session::new("online-variance-threshold", "edge-semantics");
    let mut warming = OnlineVarianceThreshold::new(0.0);
    let first = warming
        .partial_fit(&Matrix::from_row_major(1, 2, &[3.0, 4.0]), None, &session)
        .unwrap();
    assert!(first.value.quality.warmup);
    assert_eq!(
        warming
            .transform(&Matrix::from_row_major(1, 2, &[3.0, 4.0]), &session)
            .unwrap_err()
            .primary
            .code,
        IssueCode::InsufficientSample
    );
    let _ = warming
        .partial_fit(&Matrix::from_row_major(1, 2, &[3.0, 6.0]), None, &session)
        .unwrap();
    let observed = Matrix::from_row_major(2, 2, &[3.0, 4.0, 3.0, 6.0]);
    assert_eq!(
        warming
            .transform(&observed, &session)
            .unwrap()
            .value
            .shape(),
        (2, 1)
    );

    let constant = Matrix::from_fn(4, 2, |_, _| f64::MAX);
    let mut constant_model = OnlineVarianceThreshold::new(0.0);
    let result = constant_model
        .partial_fit(&constant, None, &session)
        .unwrap();
    assert!(!result.value.quality.still_identified);
    assert!(result.report.contains(IssueCode::NearZeroVariance));
    assert_eq!(
        constant_model
            .transform(&constant, &session)
            .unwrap_err()
            .primary
            .code,
        IssueCode::MeaninglessFit
    );

    let base = [1.0, 2.0, 0.0, 3.0, -1.0];
    let scale = f64::MAX.sqrt() / 16.0;
    let scaled = Matrix::from_fn(base.len(), 2, |row, column| {
        if column == 0 {
            base[row] * scale
        } else {
            f64::MAX
        }
    });
    let mut extreme = OnlineVarianceThreshold::new(0.0);
    let _ = extreme.partial_fit(&scaled, None, &session).unwrap();
    let error = (variance_threshold_variance(&extreme, 0) / (scale * scale) - 2.5).abs();
    eprintln!("online variance-threshold extreme-scale error={error:.17e}");
    // Measured on 2026-09-03: exact agreement; R9 tolerance is zero.
    assert_eq!(error, 0.0);
    assert_eq!(
        extreme.transform(&scaled, &session).unwrap().value.shape(),
        (5, 1)
    );
}

#[test]
fn online_variance_threshold_failures_are_transactional() {
    let session = Session::new("online-variance-threshold", "transactional-failures");
    let valid = Matrix::from_row_major(3, 2, &[1.0, 4.0, 2.0, 2.0, 3.0, 8.0]);
    let mut model = OnlineVarianceThreshold::new(0.0);
    let _ = model.partial_fit(&valid, None, &session).unwrap();
    let state = variance_threshold_state(&model);
    for (input, code) in [
        (
            Matrix::from_row_major(1, 2, &[4.0, f64::NAN]),
            IssueCode::NonFiniteInput,
        ),
        (Matrix::zeros(0, 2), IssueCode::EmptyMatrix),
        (Matrix::zeros(1, 0), IssueCode::EmptyMatrix),
        (
            Matrix::from_row_major(1, 1, &[4.0]),
            IssueCode::FeatureSpaceChangedOnline,
        ),
        (
            Matrix::from_row_major(1, 2, &[f64::MAX, 0.0]),
            IssueCode::NumericalOverflow,
        ),
    ] {
        let error = model.partial_fit(&input, None, &session).unwrap_err();
        assert_eq!(error.primary.code, code);
        assert_eq!(variance_threshold_state(&model), state);
    }

    let mut underflow = OnlineVarianceThreshold::new(0.0);
    let tiny = Matrix::from_row_major(2, 1, &[0.0, f64::MIN_POSITIVE]);
    let error = underflow.partial_fit(&tiny, None, &session).unwrap_err();
    assert_eq!(error.primary.code, IssueCode::NumericalUnderflow);
    assert_eq!(
        variance_threshold_state(&underflow),
        variance_threshold_state(&OnlineVarianceThreshold::new(0.0))
    );

    for invalid_threshold in [-1.0, f64::NAN, f64::INFINITY] {
        let mut invalid = model.clone();
        invalid.threshold = invalid_threshold;
        let invalid_state = variance_threshold_state(&invalid);
        let error = invalid
            .partial_fit(&Matrix::from_row_major(1, 2, &[4.0, 6.0]), None, &session)
            .unwrap_err();
        assert_eq!(error.primary.code, IssueCode::InvalidParameter);
        assert_eq!(variance_threshold_state(&invalid), invalid_state);
    }

    let mut update_overflow = model.clone();
    update_overflow.updates = u64::MAX;
    let overflow_state = variance_threshold_state(&update_overflow);
    let error = update_overflow
        .partial_fit(&Matrix::from_row_major(1, 2, &[4.0, 6.0]), None, &session)
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::InvalidParameter);
    assert_eq!(variance_threshold_state(&update_overflow), overflow_state);

    let mut observation_overflow = model.clone();
    observation_overflow.n_seen = u64::MAX;
    let overflow_state = variance_threshold_state(&observation_overflow);
    let error = observation_overflow
        .partial_fit(&Matrix::from_row_major(1, 2, &[4.0, 6.0]), None, &session)
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::InvalidParameter);
    assert_eq!(
        variance_threshold_state(&observation_overflow),
        overflow_state
    );

    assert_eq!(
        model
            .transform(&Matrix::from_row_major(1, 1, &[1.0]), &session)
            .unwrap_err()
            .primary
            .code,
        IssueCode::FeatureSpaceChangedOnline
    );
    assert_eq!(
        model
            .transform(
                &Matrix::from_row_major(1, 2, &[1.0, f64::NEG_INFINITY]),
                &session,
            )
            .unwrap_err()
            .primary
            .code,
        IssueCode::NonFiniteInput
    );
    assert_eq!(variance_threshold_state(&model), state);
}

#[test]
fn unweighted_online_moments_match_integer_scaled_decimal_oracle() {
    let observations = [-2.5, 1.25, 4.75, 9.5, -3.0, 8.125];
    let paired = [7.0, -1.5, 2.25, 10.0, 4.0, -6.75];
    let x = Matrix::from_row_major(observations.len(), 1, &observations);
    let xy = Matrix::from_fn(observations.len(), 2, |row, column| {
        if column == 0 {
            observations[row]
        } else {
            paired[row]
        }
    });
    let targets = Vector::from_slice(&paired);
    let session = Session::new("online-moments", "decimal-oracle");
    let mut mean = OnlineMean::new();
    let mut sum = OnlineSum::new();
    let mut variance = OnlineVar::new();
    let mut covariance_columns = OnlineCovariance::new();
    let mut covariance_target = OnlineCovariance::new();
    let mut count = OnlineCount::new();
    let _ = mean.partial_fit(&x, None, &session).unwrap();
    let _ = sum.partial_fit(&x, None, &session).unwrap();
    let _ = variance.partial_fit(&x, None, &session).unwrap();
    let _ = covariance_columns.partial_fit(&xy, None, &session).unwrap();
    let _ = covariance_target
        .partial_fit(&x, Some(&targets), &session)
        .unwrap();
    let _ = count.partial_fit(&x, None, &session).unwrap();

    // Independent integer-scaled oracle: mean=145/48, variance=54101/1920,
    // covariance=-827/160, and sum=145/8.
    let mean_error = (mean.score() - 145.0 / 48.0).abs();
    let sum_error = (sum.score() - 145.0 / 8.0).abs();
    let variance_error = (variance.score() - 54_101.0 / 1_920.0).abs();
    let covariance_error = (covariance_columns.score() + 827.0 / 160.0).abs();
    eprintln!(
        "unweighted moment oracle: mean={mean_error:.17e}, sum={sum_error:.17e}, var={variance_error:.17e}, cov={covariance_error:.17e}"
    );
    // Measured on 2026-09-03: mean=4.4409e-16, variance=3.5527e-15;
    // R9 limits are about 4.1×. Sum and covariance errors were zero.
    assert!(mean_error <= 1.8e-15);
    assert_eq!(sum_error, 0.0);
    assert!(variance_error <= 1.45e-14);
    assert_eq!(covariance_error, 0.0);
    assert_eq!(covariance_columns.score(), covariance_target.score());
    assert_eq!(count.score(), observations.len() as f64);

    let one = Matrix::from_row_major(1, 1, &[4.0]);
    let one_pair = Matrix::from_row_major(1, 2, &[4.0, -2.0]);
    let mut warming_variance = OnlineVar::new();
    let mut warming_covariance = OnlineCovariance::new();
    let _ = warming_variance.partial_fit(&one, None, &session).unwrap();
    let _ = warming_covariance
        .partial_fit(&one_pair, None, &session)
        .unwrap();
    assert!(warming_variance.score().is_nan());
    assert!(warming_covariance.score().is_nan());
}

#[test]
fn unweighted_online_moments_are_bitwise_partition_invariant() {
    let observations = [3.0, -1.0, 7.0, 2.5, -8.0, 11.0, 0.25];
    let paired = [-4.0, 2.0, 8.0, -3.5, 6.0, 1.0, 9.25];
    let all = Matrix::from_row_major(observations.len(), 1, &observations);
    let all_xy = Matrix::from_fn(observations.len(), 2, |row, column| {
        if column == 0 {
            observations[row]
        } else {
            paired[row]
        }
    });
    let split = 3;
    let first = Matrix::from_row_major(split, 1, &observations[..split]);
    let second = Matrix::from_row_major(observations.len() - split, 1, &observations[split..]);
    let first_xy = Matrix::from_fn(split, 2, |row, column| {
        if column == 0 {
            observations[row]
        } else {
            paired[row]
        }
    });
    let second_xy = Matrix::from_fn(observations.len() - split, 2, |row, column| {
        if column == 0 {
            observations[row + split]
        } else {
            paired[row + split]
        }
    });
    let session = Session::new("online-moments", "partition-invariance");

    let mut whole_mean = OnlineMean::new();
    let mut split_mean = OnlineMean::new();
    let _ = whole_mean.partial_fit(&all, None, &session).unwrap();
    let _ = split_mean.partial_fit(&first, None, &session).unwrap();
    let _ = split_mean.partial_fit(&second, None, &session).unwrap();
    assert_eq!(mean_state(&whole_mean).0, mean_state(&split_mean).0);
    assert_eq!(whole_mean.mean.to_bits(), split_mean.mean.to_bits());

    let mut whole_sum = OnlineSum::new();
    let mut split_sum = OnlineSum::new();
    let _ = whole_sum.partial_fit(&all, None, &session).unwrap();
    let _ = split_sum.partial_fit(&first, None, &session).unwrap();
    let _ = split_sum.partial_fit(&second, None, &session).unwrap();
    assert_eq!(whole_sum.n, split_sum.n);
    assert_eq!(whole_sum.sum.to_bits(), split_sum.sum.to_bits());
    assert_eq!(
        whole_sum.compensation.to_bits(),
        split_sum.compensation.to_bits()
    );

    let mut whole_variance = OnlineVar::new();
    let mut split_variance = OnlineVar::new();
    let _ = whole_variance.partial_fit(&all, None, &session).unwrap();
    let _ = split_variance.partial_fit(&first, None, &session).unwrap();
    let _ = split_variance.partial_fit(&second, None, &session).unwrap();
    assert_eq!(whole_variance.n, split_variance.n);
    assert_eq!(whole_variance.mean.to_bits(), split_variance.mean.to_bits());
    assert_eq!(whole_variance.m2.to_bits(), split_variance.m2.to_bits());
    assert_eq!(
        whole_variance.m2_compensation.to_bits(),
        split_variance.m2_compensation.to_bits()
    );

    let mut whole_covariance = OnlineCovariance::new();
    let mut split_covariance = OnlineCovariance::new();
    let _ = whole_covariance
        .partial_fit(&all_xy, None, &session)
        .unwrap();
    let _ = split_covariance
        .partial_fit(&first_xy, None, &session)
        .unwrap();
    let _ = split_covariance
        .partial_fit(&second_xy, None, &session)
        .unwrap();
    let mut whole_state = covariance_state(&whole_covariance);
    let mut split_state = covariance_state(&split_covariance);
    whole_state.5 = 0;
    split_state.5 = 0;
    assert_eq!(whole_state, split_state);

    let mut whole_count = OnlineCount::new();
    let mut split_count = OnlineCount::new();
    let _ = whole_count.partial_fit(&all, None, &session).unwrap();
    let _ = split_count.partial_fit(&first, None, &session).unwrap();
    let _ = split_count.partial_fit(&second, None, &session).unwrap();
    assert_eq!(whole_count.n, split_count.n);
}

#[test]
fn unweighted_online_moments_handle_representable_extremes() {
    let session = Session::new("online-moments", "representable-extremes");
    let cancelling = Matrix::from_row_major(2, 1, &[f64::MAX, -f64::MAX]);
    let constant_max = Matrix::from_row_major(2, 1, &[f64::MAX, f64::MAX]);
    let mut mean = OnlineMean::new();
    let _ = mean.partial_fit(&cancelling, None, &session).unwrap();
    assert_eq!(mean.score(), 0.0);
    let mut sum = OnlineSum::new();
    let _ = sum.partial_fit(&cancelling, None, &session).unwrap();
    assert_eq!(sum.score(), 0.0);

    let mut compensated_sum = OnlineSum::new();
    let _ = compensated_sum
        .partial_fit(
            &Matrix::from_row_major(3, 1, &[1.0e16, 1.0, -1.0e16]),
            None,
            &session,
        )
        .unwrap();
    assert_eq!(compensated_sum.score(), 1.0);
    let mut variance = OnlineVar::new();
    let _ = variance.partial_fit(&constant_max, None, &session).unwrap();
    assert_eq!(variance.score(), 0.0);

    let cross_scale = Matrix::from_row_major(2, 2, &[f64::MAX, f64::MIN_POSITIVE, -f64::MAX, 0.0]);
    let mut covariance = OnlineCovariance::new();
    let _ = covariance
        .partial_fit(&cross_scale, None, &session)
        .unwrap();
    let error = (covariance.score() - f64::MAX * f64::MIN_POSITIVE).abs();
    eprintln!("extreme covariance oracle error={error:.17e}");
    // Measured on 2026-09-03: exact agreement; R9 tolerance is zero.
    assert_eq!(error, 0.0);
}

#[test]
fn unweighted_online_moment_failures_are_transactional() {
    let session = Session::new("online-moments", "transactional-failures");
    let valid = Matrix::from_row_major(2, 1, &[1.0, 3.0]);
    let valid_xy = Matrix::from_row_major(2, 2, &[1.0, 4.0, 3.0, -2.0]);
    let non_finite = Matrix::from_row_major(2, 1, &[5.0, f64::NAN]);

    let mut mean = OnlineMean::new();
    let _ = mean.partial_fit(&valid, None, &session).unwrap();
    let state = mean_state(&mean);
    for (input, code) in [
        (non_finite.clone(), IssueCode::NonFiniteInput),
        (Matrix::zeros(0, 1), IssueCode::EmptyMatrix),
    ] {
        let error = mean.partial_fit(&input, None, &session).unwrap_err();
        assert_eq!(error.primary.code, code);
        assert_eq!(mean_state(&mean), state);
    }

    let mut sum = OnlineSum::new();
    let _ = sum.partial_fit(&valid, None, &session).unwrap();
    let state = sum_state(&sum);
    let error = sum
        .partial_fit(
            &Matrix::from_row_major(2, 1, &[f64::MAX, f64::MAX]),
            None,
            &session,
        )
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::NumericalOverflow);
    assert_eq!(sum_state(&sum), state);
    let mut underflow_sum = OnlineSum {
        n: 2,
        sum: f64::MAX,
        compensation: -f64::MAX,
        updates: 1,
    };
    let state = sum_state(&underflow_sum);
    let error = underflow_sum
        .partial_fit(&Matrix::from_row_major(1, 1, &[1.0]), None, &session)
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::NumericalUnderflow);
    assert_eq!(sum_state(&underflow_sum), state);

    let mut variance = OnlineVar::new();
    let _ = variance.partial_fit(&valid, None, &session).unwrap();
    let state = variance_state(&variance);
    let error = variance
        .partial_fit(
            &Matrix::from_row_major(2, 1, &[f64::MAX, -f64::MAX]),
            None,
            &session,
        )
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::NumericalOverflow);
    assert_eq!(variance_state(&variance), state);
    let mut underflow_variance = OnlineVar::new();
    let error = underflow_variance
        .partial_fit(
            &Matrix::from_row_major(2, 1, &[0.0, f64::MIN_POSITIVE]),
            None,
            &session,
        )
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::NumericalUnderflow);
    assert_eq!(variance_state(&underflow_variance), (0, 0, 0, 0, 0));

    let mut covariance = OnlineCovariance::new();
    let _ = covariance.partial_fit(&valid_xy, None, &session).unwrap();
    let state = covariance_state(&covariance);
    let error = covariance
        .partial_fit(&valid, Some(&Vector::from_slice(&[1.0])), &session)
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::DimensionMismatch);
    assert_eq!(covariance_state(&covariance), state);
    let error = covariance.partial_fit(&valid, None, &session).unwrap_err();
    assert_eq!(error.primary.code, IssueCode::MissingTarget);
    assert_eq!(covariance_state(&covariance), state);

    let mut count = OnlineCount::new();
    let _ = count.partial_fit(&valid, None, &session).unwrap();
    let count_state = (count.n, count.updates);
    let error = count.partial_fit(&non_finite, None, &session).unwrap_err();
    assert_eq!(error.primary.code, IssueCode::NonFiniteInput);
    assert_eq!((count.n, count.updates), count_state);

    mean.updates = u64::MAX;
    let state = mean_state(&mean);
    let error = mean
        .partial_fit(&Matrix::from_row_major(1, 1, &[1.0]), None, &session)
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::InvalidParameter);
    assert_eq!(mean_state(&mean), state);
    count.n = u64::MAX;
    let count_state = (count.n, count.updates);
    let error = count
        .partial_fit(&Matrix::from_row_major(1, 1, &[1.0]), None, &session)
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::InvalidParameter);
    assert_eq!((count.n, count.updates), count_state);
}

#[test]
fn ew_moments_match_finite_weight_closed_form() {
    let values = [2.0, -1.0, 4.0, 8.0, 0.5];
    let alpha = 0.3_f64;
    let decay = 1.0 - alpha;
    let count = values.len();
    let weights: Vec<f64> = (0..count)
        .map(|index| {
            if index == 0 {
                decay.powi((count - 1) as i32)
            } else {
                alpha * decay.powi((count - 1 - index) as i32)
            }
        })
        .collect();
    let expected_mean = values
        .iter()
        .zip(&weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>();
    let expected_variance = values
        .iter()
        .zip(&weights)
        .map(|(value, weight)| weight * (value - expected_mean).powi(2))
        .sum::<f64>();
    let expected_effective_sample_size =
        1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>();
    let x = Matrix::from_row_major(count, 1, &values);
    let session = Session::new("online-ew", "closed-form-oracle");
    let mut mean = OnlineEwMean::new(alpha);
    let mean_result = mean.partial_fit(&x, None, &session).unwrap();
    let mut variance = OnlineEwVar::new(alpha);
    let variance_result = variance.partial_fit(&x, None, &session).unwrap();

    let errors = [
        (mean.score() - expected_mean).abs(),
        (variance.mean - expected_mean).abs(),
        (variance.score() - expected_variance).abs(),
        (mean_result.value.quality.effective_sample_size - expected_effective_sample_size).abs(),
        (variance_result.value.quality.effective_sample_size - expected_effective_sample_size)
            .abs(),
    ];
    let maximum_error = errors.into_iter().fold(0.0_f64, f64::max);
    eprintln!("EW finite closed form max_abs={maximum_error:.17e}");
    // Measured on 2026-09-03: 1.7764e-15; R9 limit is about 4.1×.
    assert!(maximum_error <= 7.2e-15);
    assert_eq!(mean_result.value.quality.forgetting_factor, Some(decay));
    assert_eq!(variance_result.value.quality.forgetting_factor, Some(decay));
}

#[test]
fn ew_moments_are_partition_invariant() {
    let values = [3.0, -1.0, 7.0, 2.5, -8.0, 11.0, 0.25];
    let all = Matrix::from_row_major(values.len(), 1, &values);
    let session = Session::new("online-ew", "partition-property");
    let mut whole_mean = OnlineEwMean::new(0.37);
    let mut whole_variance = OnlineEwVar::new(0.37);
    let _ = whole_mean.partial_fit(&all, None, &session).unwrap();
    let _ = whole_variance.partial_fit(&all, None, &session).unwrap();

    let mut split_mean = OnlineEwMean::new(0.37);
    let mut split_variance = OnlineEwVar::new(0.37);
    for values in [&values[..3], &values[3..]] {
        let batch = Matrix::from_row_major(values.len(), 1, values);
        let _ = split_mean.partial_fit(&batch, None, &session).unwrap();
        let _ = split_variance.partial_fit(&batch, None, &session).unwrap();
    }
    assert_eq!(whole_mean.mean.to_bits(), split_mean.mean.to_bits());
    assert_eq!(
        whole_mean.weight_square_sum.to_bits(),
        split_mean.weight_square_sum.to_bits()
    );
    assert_eq!(whole_mean.n_seen, split_mean.n_seen);
    assert_eq!(whole_variance.mean.to_bits(), split_variance.mean.to_bits());
    assert_eq!(whole_variance.var.to_bits(), split_variance.var.to_bits());
    assert_eq!(
        whole_variance.weight_square_sum.to_bits(),
        split_variance.weight_square_sum.to_bits()
    );
    assert_eq!(whole_variance.n_seen, split_variance.n_seen);
}

#[test]
fn ew_moments_handle_constant_and_extreme_finite_values() {
    let session = Session::new("online-ew", "finite-extremes");
    let constant = Matrix::from_row_major(4, 1, &[7.0, 7.0, 7.0, 7.0]);
    let mut mean = OnlineEwMean::new(0.4);
    let mut variance = OnlineEwVar::new(0.4);
    let _ = mean.partial_fit(&constant, None, &session).unwrap();
    let _ = variance.partial_fit(&constant, None, &session).unwrap();
    assert_eq!(mean.score(), 7.0);
    assert_eq!(variance.mean, 7.0);
    assert_eq!(variance.score(), 0.0);

    let extreme_value = f64::MAX / 2.0;
    let extreme = Matrix::from_row_major(4, 1, &[extreme_value; 4]);
    let mut extreme_mean = OnlineEwMean::new(0.4);
    let mut extreme_variance = OnlineEwVar::new(0.4);
    let _ = extreme_mean.partial_fit(&extreme, None, &session).unwrap();
    let _ = extreme_variance
        .partial_fit(&extreme, None, &session)
        .unwrap();
    assert_eq!(extreme_mean.score(), extreme_value);
    assert_eq!(extreme_variance.mean, extreme_value);
    assert_eq!(extreme_variance.score(), 0.0);

    let changing = Matrix::from_row_major(3, 1, &[1.0, 2.0, 3.0]);
    let mut replace_mean = OnlineEwMean::new(1.0);
    let mut replace_variance = OnlineEwVar::new(1.0);
    let mean_result = replace_mean.partial_fit(&changing, None, &session).unwrap();
    let variance_result = replace_variance
        .partial_fit(&changing, None, &session)
        .unwrap();
    assert_eq!(replace_mean.score(), 3.0);
    assert_eq!(replace_variance.score(), 0.0);
    assert_eq!(mean_result.value.quality.effective_sample_size, 1.0);
    assert_eq!(variance_result.value.quality.effective_sample_size, 1.0);
    assert!(!variance_result.value.quality.still_identified);

    let opposing = Matrix::from_row_major(2, 1, &[-f64::MAX, f64::MAX]);
    let mut opposing_mean = OnlineEwMean::new(1.0);
    let mut opposing_variance = OnlineEwVar::new(1.0);
    let _ = opposing_mean
        .partial_fit(&opposing, None, &session)
        .unwrap();
    let _ = opposing_variance
        .partial_fit(&opposing, None, &session)
        .unwrap();
    assert_eq!(opposing_mean.score(), f64::MAX);
    assert_eq!(opposing_variance.mean, f64::MAX);
    assert_eq!(opposing_variance.score(), 0.0);
    assert_eq!(opposing_variance.effective_sample_size(), 1.0);
}

#[test]
fn ew_moments_reject_invalid_batches_transactionally() {
    let session = Session::new("online-ew", "transactional-validation");
    let x = Matrix::from_row_major(2, 1, &[1.0, 3.0]);
    let mut mean = OnlineEwMean::new(0.4);
    let _ = mean.partial_fit(&x, None, &session).unwrap();
    let state = (
        mean.mean.to_bits(),
        mean.weight_square_sum.to_bits(),
        mean.n_seen,
        mean.updates,
    );
    for alpha in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
        let mut invalid = mean.clone();
        invalid.alpha = alpha;
        let before = (
            invalid.alpha.to_bits(),
            invalid.mean.to_bits(),
            invalid.weight_square_sum.to_bits(),
            invalid.n_seen,
            invalid.updates,
        );
        let error = invalid.partial_fit(&x, None, &session).unwrap_err();
        assert_eq!(error.primary.code, IssueCode::InvalidWeight);
        assert_eq!(
            (
                invalid.alpha.to_bits(),
                invalid.mean.to_bits(),
                invalid.weight_square_sum.to_bits(),
                invalid.n_seen,
                invalid.updates,
            ),
            before
        );
    }
    let mut underflowing = mean.clone();
    underflowing.alpha = f64::from_bits(1);
    let before = (
        underflowing.alpha.to_bits(),
        underflowing.mean.to_bits(),
        underflowing.weight_square_sum.to_bits(),
        underflowing.n_seen,
        underflowing.updates,
    );
    let error = underflowing.partial_fit(&x, None, &session).unwrap_err();
    assert_eq!(error.primary.code, IssueCode::NumericalUnderflow);
    assert_eq!(
        (
            underflowing.alpha.to_bits(),
            underflowing.mean.to_bits(),
            underflowing.weight_square_sum.to_bits(),
            underflowing.n_seen,
            underflowing.updates,
        ),
        before
    );
    for (batch, code) in [
        (Matrix::zeros(0, 1), IssueCode::EmptyMatrix),
        (
            Matrix::from_row_major(2, 1, &[1.0, f64::NAN]),
            IssueCode::NonFiniteInput,
        ),
    ] {
        let error = mean.partial_fit(&batch, None, &session).unwrap_err();
        assert_eq!(error.primary.code, code);
        assert_eq!(
            (
                mean.mean.to_bits(),
                mean.weight_square_sum.to_bits(),
                mean.n_seen,
                mean.updates,
            ),
            state
        );
    }
    let mut count_overflow = mean.clone();
    count_overflow.n_seen = u64::MAX;
    let before = count_overflow.clone();
    let error = count_overflow
        .partial_fit(&Matrix::from_row_major(1, 1, &[2.0]), None, &session)
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::InvalidParameter);
    assert_eq!(count_overflow.mean.to_bits(), before.mean.to_bits());
    assert_eq!(count_overflow.n_seen, before.n_seen);

    let mut variance = OnlineEwVar::new(0.5);
    let _ = variance.partial_fit(&x, None, &session).unwrap();
    let state = (
        variance.mean.to_bits(),
        variance.var.to_bits(),
        variance.weight_square_sum.to_bits(),
        variance.n_seen,
        variance.updates,
    );
    let error = variance
        .partial_fit(&Matrix::from_row_major(1, 1, &[f64::MAX]), None, &session)
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::NumericalOverflow);
    assert_eq!(
        (
            variance.mean.to_bits(),
            variance.var.to_bits(),
            variance.weight_square_sum.to_bits(),
            variance.n_seen,
            variance.updates,
        ),
        state
    );
    variance.updates = u64::MAX;
    let state = variance.clone();
    let error = variance
        .partial_fit(&Matrix::from_row_major(1, 1, &[2.0]), None, &session)
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::InvalidParameter);
    assert_eq!(variance.mean.to_bits(), state.mean.to_bits());
    assert_eq!(variance.updates, state.updates);
}

#[test]
fn weighted_mean_matches_hand_oracle_and_is_weight_scale_invariant() {
    let x = Matrix::from_row_major(3, 1, &[2.0, 4.0, 10.0]);
    let weights = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let session = Session::new("online-weighted-mean", "hand-oracle");
    let mut model = OnlineWeightedMean::new();
    let result = model.partial_fit(&x, Some(&weights), &session).unwrap();
    let mean_error = (model.score() - 40.0 / 6.0).abs();
    let effective_error = (result.value.quality.effective_sample_size - 36.0 / 14.0).abs();
    eprintln!(
        "OnlineWeightedMean hand oracle: mean_abs={mean_error:.17e}, Kish_abs={effective_error:.17e}"
    );
    // Measured on 2026-09-03: 8.8818e-16 and 4.4409e-16;
    // R9 limits are approximately 4.1×.
    assert!(mean_error <= 3.6e-15);
    assert!(effective_error <= 1.8e-15);

    let scaled_weights = Vector::from_slice(&[8.0, 16.0, 24.0]);
    let mut scaled = OnlineWeightedMean::new();
    let scaled_result = scaled
        .partial_fit(&x, Some(&scaled_weights), &session)
        .unwrap();
    assert_eq!(model.score().to_bits(), scaled.score().to_bits());
    assert_eq!(
        result.value.quality.effective_sample_size.to_bits(),
        scaled_result.value.quality.effective_sample_size.to_bits()
    );

    let huge_x = Matrix::from_row_major(2, 1, &[2.0, 4.0]);
    let huge_weights = Vector::from_slice(&[f64::MAX / 2.0, f64::MAX / 4.0]);
    let relative_weights = Vector::from_slice(&[2.0, 1.0]);
    let mut huge = OnlineWeightedMean::new();
    let mut relative = OnlineWeightedMean::new();
    let huge_result = huge
        .partial_fit(&huge_x, Some(&huge_weights), &session)
        .unwrap();
    let relative_result = relative
        .partial_fit(&huge_x, Some(&relative_weights), &session)
        .unwrap();
    assert_eq!(huge.score().to_bits(), relative.score().to_bits());
    assert_eq!(
        huge_result.value.quality.effective_sample_size.to_bits(),
        relative_result
            .value
            .quality
            .effective_sample_size
            .to_bits()
    );
}

#[test]
fn weighted_mean_is_partition_invariant() {
    let values = [3.0, -1.0, 7.0, 2.5, -8.0, 11.0, 0.25];
    let weights = [0.5, 4.0, 1.25, 8.0, 2.0, 0.75, 16.0];
    let x = Matrix::from_row_major(values.len(), 1, &values);
    let w = Vector::from_slice(&weights);
    let session = Session::new("online-weighted-mean", "partition-property");
    let mut whole = OnlineWeightedMean::new();
    let _ = whole.partial_fit(&x, Some(&w), &session).unwrap();
    let mut partitioned = OnlineWeightedMean::new();
    for (values, weights) in [(&values[..3], &weights[..3]), (&values[3..], &weights[3..])] {
        let x = Matrix::from_row_major(values.len(), 1, values);
        let weights = Vector::from_slice(weights);
        let _ = partitioned
            .partial_fit(&x, Some(&weights), &session)
            .unwrap();
    }
    assert_eq!(whole.mean.to_bits(), partitioned.mean.to_bits());
    assert_eq!(
        whole.weight_scale.to_bits(),
        partitioned.weight_scale.to_bits()
    );
    assert_eq!(
        whole.scaled_weight_sum.to_bits(),
        partitioned.scaled_weight_sum.to_bits()
    );
    assert_eq!(
        whole.scaled_weight_square_sum.to_bits(),
        partitioned.scaled_weight_square_sum.to_bits()
    );
    assert_eq!(whole.n_seen, partitioned.n_seen);
    assert_eq!(
        whole.effective_sample_size().to_bits(),
        partitioned.effective_sample_size().to_bits()
    );
}

#[test]
fn weighted_mean_rejects_invalid_batches_transactionally() {
    let session = Session::new("online-weighted-mean", "transactional-validation");
    let x = Matrix::from_row_major(2, 1, &[1.0, 3.0]);
    let weights = Vector::from_slice(&[1.0, 2.0]);
    let mut model = OnlineWeightedMean::new();
    let _ = model.partial_fit(&x, Some(&weights), &session).unwrap();
    let state = (
        model.mean.to_bits(),
        model.weight_scale.to_bits(),
        model.scaled_weight_sum.to_bits(),
        model.scaled_weight_square_sum.to_bits(),
        model.n_seen,
        model.updates,
    );
    let assert_unchanged = |actual: &OnlineWeightedMean| {
        assert_eq!(
            (
                actual.mean.to_bits(),
                actual.weight_scale.to_bits(),
                actual.scaled_weight_sum.to_bits(),
                actual.scaled_weight_square_sum.to_bits(),
                actual.n_seen,
                actual.updates,
            ),
            state
        );
    };

    let error = model.partial_fit(&x, None, &session).unwrap_err();
    assert_eq!(error.primary.code, IssueCode::MissingTarget);
    assert_unchanged(&model);
    let error = model
        .partial_fit(&x, Some(&Vector::from_slice(&[1.0])), &session)
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::DimensionMismatch);
    assert_unchanged(&model);
    for weight in [0.0, -1.0] {
        let error = model
            .partial_fit(&x, Some(&Vector::from_slice(&[1.0, weight])), &session)
            .unwrap_err();
        assert_eq!(error.primary.code, IssueCode::InvalidWeight);
        assert_unchanged(&model);
    }
    for weight in [f64::NAN, f64::INFINITY] {
        let error = model
            .partial_fit(&x, Some(&Vector::from_slice(&[1.0, weight])), &session)
            .unwrap_err();
        assert_eq!(error.primary.code, IssueCode::NonFiniteInput);
        assert_unchanged(&model);
    }
    let error = model
        .partial_fit(
            &Matrix::from_row_major(2, 1, &[1.0, f64::NEG_INFINITY]),
            Some(&weights),
            &session,
        )
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::NonFiniteInput);
    assert_unchanged(&model);
    let error = model
        .partial_fit(
            &x,
            Some(&Vector::from_slice(&[1.0, f64::MIN_POSITIVE])),
            &session,
        )
        .unwrap_err();
    assert_eq!(error.primary.code, IssueCode::NumericalUnderflow);
    assert_unchanged(&model);

    model.updates = u64::MAX;
    let before = model.updates;
    let error = model.partial_fit(&x, Some(&weights), &session).unwrap_err();
    assert_eq!(error.primary.code, IssueCode::InvalidParameter);
    assert_eq!(model.updates, before);
}

#[test]
fn rls_recovers_slope() {
    let n = 80;
    let x = Matrix::from_fn(n, 1, |row, _| (row as f64) + 1.0);
    let y = Vector::from_iter((0..n).map(|row| 2.0 * ((row as f64) + 1.0)));
    let session = Session::new("rls", "partial_fit");
    let mut model = LinearRegression {
        forgetting_factor: 1.0,
        fit_intercept: false,
        p0: 100.0,
        ..LinearRegression::default()
    };
    let result = model.partial_fit(&x, Some(&y), &session).expect("RLS fit");
    assert!(!result.value.narrative.is_empty());
    assert!(has_incremental(&session));
    let slope = model.coef()[0];
    assert!((slope - 2.0).abs() < 0.05, "slope={slope}");
}

fn rls_json_number(value: &serde_json::Value) -> f64 {
    value
        .as_str()
        .expect("RLS oracle numbers are encoded as strings")
        .parse::<f64>()
        .expect("valid RLS oracle number")
}

fn rls_json_matrix(value: &serde_json::Value) -> Matrix {
    let rows = value.as_array().expect("RLS oracle matrix rows");
    let columns = rows
        .first()
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    Matrix::from_fn(rows.len(), columns, |row, column| {
        rls_json_number(&rows[row][column])
    })
}

#[test]
fn rls_matches_independent_decimal_batch_oracle() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../golden/online_rls.json"
    )))
    .expect("parse online RLS fixture");
    let cases = fixture["cases"].as_array().expect("RLS oracle cases");
    let mut max_absolute_error = 0.0_f64;
    let mut max_relative_error = 0.0_f64;

    for case in cases {
        let x = rls_json_matrix(&case["x"]);
        let prediction_x = rls_json_matrix(&case["prediction_x"]);
        let y = Vector::from_iter(
            case["y"]
                .as_array()
                .expect("RLS oracle targets")
                .iter()
                .map(rls_json_number),
        );
        let mut model = LinearRegression {
            forgetting_factor: rls_json_number(&case["forgetting_factor"]),
            fit_intercept: case["fit_intercept"]
                .as_bool()
                .expect("RLS oracle fit_intercept"),
            p0: rls_json_number(&case["p0"]),
            ..LinearRegression::default()
        };
        let session = Session::new("online-rls", "decimal-oracle");
        let _ = model
            .partial_fit(&x, Some(&y), &session)
            .expect("RLS oracle update");

        let expected_theta = case["theta"].as_array().expect("RLS oracle theta");
        for (actual, expected) in model.theta.as_slice().iter().zip(expected_theta) {
            let expected = rls_json_number(expected);
            let absolute = (actual - expected).abs();
            max_absolute_error = max_absolute_error.max(absolute);
            max_relative_error =
                max_relative_error.max(absolute / expected.abs().max(f64::MIN_POSITIVE));
        }
        let expected_p = case["inverse_gram"]
            .as_array()
            .expect("RLS oracle inverse Gram");
        let actual_p = model.p_matrix().expect("fitted inverse Gram");
        for row in 0..actual_p.nrows() {
            for column in 0..actual_p.ncols() {
                let expected = rls_json_number(&expected_p[row][column]);
                let absolute = (actual_p[(row, column)] - expected).abs();
                max_absolute_error = max_absolute_error.max(absolute);
                max_relative_error =
                    max_relative_error.max(absolute / expected.abs().max(f64::MIN_POSITIVE));
            }
        }
        let predictions = model
            .predict(&prediction_x, &session)
            .expect("RLS oracle prediction")
            .value;
        let expected_predictions = case["predictions"]
            .as_array()
            .expect("RLS oracle predictions");
        for (actual, expected) in predictions.as_slice().iter().zip(expected_predictions) {
            let expected = rls_json_number(expected);
            let absolute = (actual - expected).abs();
            max_absolute_error = max_absolute_error.max(absolute);
            max_relative_error =
                max_relative_error.max(absolute / expected.abs().max(f64::MIN_POSITIVE));
        }
        let expected_n_eff = rls_json_number(&case["effective_sample_size"]);
        let absolute = (model.effective_sample_size - expected_n_eff).abs();
        max_absolute_error = max_absolute_error.max(absolute);
        max_relative_error =
            max_relative_error.max(absolute / expected_n_eff.abs().max(f64::MIN_POSITIVE));
    }

    assert!(max_absolute_error.is_finite());
    assert!(max_relative_error.is_finite());
    eprintln!(
        "online RLS Decimal oracle: max_abs={max_absolute_error:.17e}, max_rel={max_relative_error:.17e}"
    );
    // Measured on 2026-09-03: max_abs=6.2173e-15 and
    // max_rel=2.6153e-14; the R9 limits are approximately 4.0× each.
    assert!(
        max_absolute_error <= 2.5e-14,
        "max_abs={max_absolute_error:e}"
    );
    assert!(
        max_relative_error <= 1.05e-13,
        "max_rel={max_relative_error:e}"
    );
}

#[test]
fn rls_rejects_invalid_updates_transactionally() {
    let session = Session::new("online-rls", "transactional-validation");
    let x = Matrix::from_row_major(3, 1, &[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);
    let mut model = LinearRegression::new(1.0);
    let _ = model
        .partial_fit(&x, Some(&y), &session)
        .expect("valid RLS update");
    let theta_before = model.theta.clone();
    let p_before = model.p_matrix().expect("fitted inverse Gram").clone();
    let n_before = model.n_seen;
    let updates_before = model.updates;

    let short_y = Vector::from_slice(&[2.0]);
    let error = model.partial_fit(&x, Some(&short_y), &session).unwrap_err();
    assert_eq!(error.primary.code, IssueCode::DimensionMismatch);
    assert_eq!(model.theta, theta_before);
    assert_eq!(model.n_seen, n_before);
    assert_eq!(model.updates, updates_before);
    for row in 0..p_before.nrows() {
        for column in 0..p_before.ncols() {
            assert_eq!(
                model.p_matrix().unwrap()[(row, column)],
                p_before[(row, column)]
            );
        }
    }

    model.forgetting_factor = f64::NAN;
    let error = model.partial_fit(&x, Some(&y), &session).unwrap_err();
    assert_eq!(error.primary.code, IssueCode::InvalidWeight);
    assert_eq!(model.theta, theta_before);
    assert_eq!(model.n_seen, n_before);
    assert_eq!(model.updates, updates_before);

    model.forgetting_factor = 1.0;
    model.fit_intercept = false;
    let error = model.partial_fit(&x, Some(&y), &session).unwrap_err();
    assert_eq!(error.primary.code, IssueCode::FeatureSpaceChangedOnline);
    assert_eq!(model.theta, theta_before);
    assert_eq!(model.n_seen, n_before);
    assert_eq!(model.updates, updates_before);
    model.fit_intercept = true;

    let overflow_x = Matrix::from_row_major(2, 1, &[1.0, f64::MAX]);
    let overflow_y = Vector::from_slice(&[2.0, 2.0]);
    let error = model
        .partial_fit(&overflow_x, Some(&overflow_y), &session)
        .unwrap_err();
    assert!(matches!(
        error.primary.code,
        IssueCode::NearSingular | IssueCode::NonFiniteOutput
    ));
    assert_eq!(model.theta, theta_before);
    assert_eq!(model.n_seen, n_before);
    assert_eq!(model.updates, updates_before);
}

#[test]
fn rls_is_partition_invariant_and_response_scale_equivariant() {
    let x = Matrix::from_fn(17, 2, |row, column| {
        let t = row as f64 - 7.0;
        if column == 0 {
            t / 3.0
        } else {
            ((row * 5 + 2) % 11) as f64 - 4.0
        }
    });
    let y = Vector::from_iter(
        (0..x.nrows()).map(|row| 0.75 + 1.25 * x.get(row, 0) - 0.4 * x.get(row, 1)),
    );
    let session = Session::new("online-rls", "algebraic-properties");
    let mut all_at_once = LinearRegression::new(0.96);
    all_at_once.p0 = 12.0;
    let _ = all_at_once.partial_fit(&x, Some(&y), &session).unwrap();

    let split = 6;
    let first_x = Matrix::from_fn(split, x.ncols(), |row, column| x.get(row, column));
    let first_y = Vector::from_iter((0..split).map(|row| y[row]));
    let second_x = Matrix::from_fn(x.nrows() - split, x.ncols(), |row, column| {
        x.get(row + split, column)
    });
    let second_y = Vector::from_iter((split..x.nrows()).map(|row| y[row]));
    let mut partitioned = LinearRegression::new(0.96);
    partitioned.p0 = 12.0;
    let _ = partitioned
        .partial_fit(&first_x, Some(&first_y), &session)
        .unwrap();
    let _ = partitioned
        .partial_fit(&second_x, Some(&second_y), &session)
        .unwrap();
    assert_eq!(all_at_once.theta, partitioned.theta);
    assert_eq!(
        all_at_once.effective_sample_size,
        partitioned.effective_sample_size
    );
    let all_p = all_at_once.p_matrix().unwrap();
    let partitioned_p = partitioned.p_matrix().unwrap();
    for row in 0..all_p.nrows() {
        for column in 0..all_p.ncols() {
            assert_eq!(all_p[(row, column)], partitioned_p[(row, column)]);
        }
    }

    let response_scale = 3.5;
    let scaled_y = Vector::from_iter(y.as_slice().iter().map(|value| response_scale * value));
    let mut scaled = LinearRegression::new(0.96);
    scaled.p0 = 12.0;
    let _ = scaled.partial_fit(&x, Some(&scaled_y), &session).unwrap();
    let mut max_scale_error = 0.0_f64;
    for (scaled_value, base_value) in scaled
        .theta
        .as_slice()
        .iter()
        .zip(all_at_once.theta.as_slice())
    {
        max_scale_error = max_scale_error.max((scaled_value - response_scale * base_value).abs());
    }
    eprintln!("online RLS response-scale max_abs={max_scale_error:.17e}");
    // Measured on 2026-09-03: 8.8818e-16; the R9 limit is about 4.1×.
    assert!(max_scale_error <= 3.6e-15);
    let scaled_p = scaled.p_matrix().unwrap();
    for row in 0..all_p.nrows() {
        for column in 0..all_p.ncols() {
            assert_eq!(all_p[(row, column)], scaled_p[(row, column)]);
        }
    }
}

#[test]
fn rls_prediction_rejects_shape_and_non_finite_input() {
    let session = Session::new("online-rls", "predict-validation");
    let x = Matrix::from_row_major(3, 1, &[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);
    let mut model = LinearRegression::new(1.0);
    let _ = model.partial_fit(&x, Some(&y), &session).unwrap();

    let wrong_shape = Matrix::zeros(1, 2);
    let error = model.predict(&wrong_shape, &session).unwrap_err();
    assert_eq!(error.primary.code, IssueCode::DimensionMismatch);

    let non_finite = Matrix::from_row_major(1, 1, &[f64::INFINITY]);
    let error = model.predict(&non_finite, &session).unwrap_err();
    assert_eq!(error.primary.code, IssueCode::NonFiniteInput);
}
