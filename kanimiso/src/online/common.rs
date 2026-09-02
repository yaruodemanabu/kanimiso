use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use ojizou_san::{IncrementalExplain, Session};
use signlred::{Failure, IncrementalQuality, Issue, IssueCode, Qualified, Result};

/// Apply the shape checks shared by incremental estimators.
pub(crate) fn inspect_online_xy(ctx: &mut FitCtx, x: &Matrix, y: Option<&Vector>) {
    let (n, p) = x.shape();
    ctx.report.set_sample_shape(n, p);
    if n == 0 || p == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message(format!("online design is {n}×{p}"))
                .build(),
        );
        return;
    }
    if let Some(y) = y {
        if y.len() != n {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!("y.len()={} but X has {n} rows", y.len()))
                    .build(),
            );
        }
    }
}

/// Build the standard explanation for a rejected incremental batch.
pub(crate) fn reject_explain(
    update: u64,
    batch: usize,
    n_seen: u64,
    why: &str,
) -> IncrementalExplain {
    IncrementalExplain::from_quality(
        IncrementalQuality::new(update, batch, n_seen),
        "nothing",
        why,
        "invalid",
        "invalid",
    )
}

/// Record the standard zero-information issue for an incremental update.
pub(crate) fn flag_info(ctx: &mut FitCtx, quality: &IncrementalQuality) {
    if quality.is_uninformative(ctx.policy.uninformative_info_eps) {
        ctx.push(
            Issue::builder(IssueCode::UpdateWithZeroInformation)
                .incremental(quality.clone())
                .message("this online update added no usable information")
                .build(),
        );
    }
}

/// Record an incremental explanation before finishing its qualified result.
pub(crate) fn finish_explain(
    ctx: FitCtx,
    explanation: IncrementalExplain,
) -> Result<Qualified<IncrementalExplain>> {
    ctx.session.record_incremental(explanation.clone());
    ctx.finish(explanation)
}

pub(super) fn online_moment_preflight(
    session: &Session,
    x: &Matrix,
    y: Option<&Vector>,
    pairwise: bool,
    name: &str,
    updates: u64,
    n_seen: u64,
) -> std::result::Result<(FitCtx, u64, u64), Failure> {
    let mut ctx = FitCtx::with_session(session.child("partial_fit"));
    ctx.report.set_sample_shape(x.nrows(), x.ncols());
    let mut invalid = false;
    if x.nrows() == 0 || x.ncols() == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message(format!(
                    "{name} requires a non-empty first column; got {}×{}",
                    x.nrows(),
                    x.ncols()
                ))
                .build(),
        );
        invalid = true;
    }
    if pairwise {
        if let Some(target) = y {
            if target.len() != x.nrows() {
                ctx.push(
                    Issue::builder(IssueCode::DimensionMismatch)
                        .message(format!(
                            "{name} target length {} does not match {} rows",
                            target.len(),
                            x.nrows()
                        ))
                        .build(),
                );
                invalid = true;
            }
        }
        if x.ncols() < 2 && y.is_none() {
            ctx.push(
                Issue::builder(IssueCode::MissingTarget)
                    .message(format!(
                        "{name} needs column 1 or one explicit target per row"
                    ))
                    .build(),
            );
            invalid = true;
        }
    }

    if !invalid {
        let non_finite_row = (0..x.nrows()).find(|&row| {
            let first = x.get(row, 0);
            let second_is_non_finite = if pairwise {
                if x.ncols() >= 2 {
                    !x.get(row, 1).is_finite()
                } else {
                    !y.expect("validated pairwise target")[row].is_finite()
                }
            } else {
                false
            };
            !first.is_finite() || second_is_non_finite
        });
        if let Some(row) = non_finite_row {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteInput)
                    .message(format!(
                        "{name} observation at row {row} contains a non-finite value"
                    ))
                    .build(),
            );
            invalid = true;
        }
    }

    let batch_rows = match u64::try_from(x.nrows()) {
        Ok(rows) => Some(rows),
        Err(_) => {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message(format!(
                        "{name} batch row count cannot be represented by its counter"
                    ))
                    .build(),
            );
            invalid = true;
            None
        }
    };
    let new_n_seen = batch_rows.and_then(|rows| n_seen.checked_add(rows));
    if new_n_seen.is_none() {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message(format!("{name} observation counter overflowed"))
                .build(),
        );
        invalid = true;
    }
    let new_updates = updates.checked_add(1);
    if new_updates.is_none() {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message(format!("{name} update counter overflowed"))
                .build(),
        );
        invalid = true;
    }

    match (invalid, new_n_seen, new_updates) {
        (false, Some(new_n_seen), Some(new_updates)) => Ok((ctx, new_n_seen, new_updates)),
        _ => Err(reject_metric_batch(ctx, updates, x.nrows(), n_seen)),
    }
}

pub(super) fn online_mean_step(previous: f64, observation: f64, new_count: u64) -> f64 {
    let new_share = 1.0 / new_count as f64;
    if previous.is_sign_negative() == observation.is_sign_negative() {
        previous + (observation - previous) * new_share
    } else {
        (1.0 - new_share) * previous + new_share * observation
    }
}

pub(super) fn checked_online_covariance_step(
    count: u64,
    left_mean: f64,
    right_mean: f64,
    left: f64,
    right: f64,
) -> std::result::Result<(f64, f64, f64), IssueCode> {
    let next_count = count.checked_add(1).ok_or(IssueCode::InvalidParameter)?;
    if count == 0 {
        return Ok((left, right, 0.0));
    }

    let next_left_mean = online_mean_step(left_mean, left, next_count);
    let next_right_mean = online_mean_step(right_mean, right, next_count);
    if !next_left_mean.is_finite() || !next_right_mean.is_finite() {
        return Err(IssueCode::NumericalOverflow);
    }
    let cross_increment = checked_difference_product(left, left_mean, right, next_right_mean)?;
    Ok((next_left_mean, next_right_mean, cross_increment))
}

pub(super) fn autocorrelation_from_moments(cross: f64, left_m2: f64, right_m2: f64) -> f64 {
    if !cross.is_finite()
        || !left_m2.is_finite()
        || !right_m2.is_finite()
        || left_m2 <= 0.0
        || right_m2 <= 0.0
    {
        return f64::NAN;
    }

    let left_scale = left_m2.sqrt();
    let right_scale = right_m2.sqrt();
    if left_scale <= right_scale {
        (cross / left_scale) / right_scale
    } else {
        (cross / right_scale) / left_scale
    }
}

pub(super) fn checked_sample_variance_from_m2(
    count: u64,
    m2: f64,
    compensation: f64,
) -> std::result::Result<Option<f64>, IssueCode> {
    let total = m2 + compensation;
    if !m2.is_finite() || !compensation.is_finite() || !total.is_finite() || total < 0.0 {
        return Err(IssueCode::NonFiniteOutput);
    }
    if count < 2 {
        return Ok(None);
    }

    let variance = total / (count - 1) as f64;
    if !variance.is_finite() {
        Err(IssueCode::NumericalOverflow)
    } else if total != 0.0 && variance == 0.0 {
        Err(IssueCode::NumericalUnderflow)
    } else {
        Ok(Some(variance))
    }
}

pub(super) fn checked_difference_product(
    left_value: f64,
    left_center: f64,
    right_value: f64,
    right_center: f64,
) -> std::result::Result<f64, IssueCode> {
    let left = left_value - left_center;
    let right = right_value - right_center;
    if left == 0.0 || right == 0.0 {
        return Ok(0.0);
    }

    let product = match (left.is_finite(), right.is_finite()) {
        (true, true) => left * right,
        (false, true) => {
            let half_left = left_value * 0.5 - left_center * 0.5;
            (half_left * right) * 2.0
        }
        (true, false) => {
            let half_right = right_value * 0.5 - right_center * 0.5;
            (left * half_right) * 2.0
        }
        (false, false) => return Err(IssueCode::NumericalOverflow),
    };
    if !product.is_finite() {
        Err(IssueCode::NumericalOverflow)
    } else if product == 0.0 {
        Err(IssueCode::NumericalUnderflow)
    } else {
        Ok(product)
    }
}

pub(super) fn checked_compensated_add(
    sum: f64,
    compensation: f64,
    value: f64,
) -> std::result::Result<(f64, f64), IssueCode> {
    let next_sum = sum + value;
    if !next_sum.is_finite() {
        return Err(IssueCode::NumericalOverflow);
    }
    let correction = if sum.abs() >= value.abs() {
        (sum - next_sum) + value
    } else {
        (value - next_sum) + sum
    };
    let next_compensation = compensation + correction;
    if correction != 0.0 && next_compensation == compensation {
        return Err(IssueCode::NumericalUnderflow);
    }
    let next_total = next_sum + next_compensation;
    if !correction.is_finite() || !next_compensation.is_finite() || !next_total.is_finite() {
        Err(IssueCode::NumericalOverflow)
    } else {
        Ok((next_sum, next_compensation))
    }
}

pub(super) fn finite_score_delta(before: f64, after: f64) -> Option<f64> {
    if before.is_finite() && after.is_finite() {
        let delta = (after - before).abs();
        delta.is_finite().then_some(delta)
    } else {
        None
    }
}

pub(super) fn reject_metric_batch(ctx: FitCtx, updates: u64, batch: usize, n_seen: u64) -> Failure {
    ctx.session.record_incremental(reject_explain(
        updates,
        batch,
        n_seen,
        "invalid online metric batch",
    ));
    ctx.finish_failure()
}
