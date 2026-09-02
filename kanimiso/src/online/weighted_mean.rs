use super::common::{finish_explain, flag_info, reject_metric_batch};
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::PartialFit;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{Failure, IncrementalQuality, Issue, IssueCode, Qualified, Result};

/// Streaming weighted mean (river `stats.WeightedMean`).
///
/// Column 0 holds observations and `y` supplies one finite, strictly positive
/// weight per row. Weight sums are stored relative to the largest weight to
/// avoid avoidable overflow while retaining the Kish effective sample size.
#[derive(Clone, Debug, Default)]
pub struct OnlineWeightedMean {
    pub(super) mean: f64,
    pub(super) weight_scale: f64,
    pub(super) scaled_weight_sum: f64,
    pub(super) scaled_weight_square_sum: f64,
    pub(super) n_seen: u64,
    pub(super) updates: u64,
}

impl OnlineWeightedMean {
    /// Construct an empty weighted mean.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the current weighted mean, or NaN before the first pair.
    pub fn score(&self) -> f64 {
        if self.n_seen == 0 {
            f64::NAN
        } else {
            self.mean
        }
    }

    pub(super) fn effective_sample_size(&self) -> f64 {
        if self.n_seen == 0 {
            0.0
        } else {
            self.scaled_weight_sum * self.scaled_weight_sum / self.scaled_weight_square_sum
        }
    }
}

impl PartialFit for OnlineWeightedMean {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let (mut ctx, weights, new_n_seen) =
            match preflight(session, x, y, self.updates, self.n_seen) {
                Ok(validated) => validated,
                Err(failure) => return Err(failure),
            };
        if let Some((row, weight)) = weights
            .as_slice()
            .iter()
            .copied()
            .enumerate()
            .find(|(_, weight)| *weight <= 0.0)
        {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!(
                        "OnlineWeightedMean weight at row {row} must be strictly positive; got {weight}"
                    ))
                    .build(),
            );
            return Err(reject_metric_batch(
                ctx,
                self.updates,
                x.nrows(),
                self.n_seen,
            ));
        }
        let Some(new_updates) = self.updates.checked_add(1) else {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("OnlineWeightedMean update counter overflowed")
                    .build(),
            );
            return Err(reject_metric_batch(
                ctx,
                self.updates,
                x.nrows(),
                self.n_seen,
            ));
        };
        let stored_state_is_valid = if self.n_seen == 0 {
            self.mean == 0.0
                && self.weight_scale == 0.0
                && self.scaled_weight_sum == 0.0
                && self.scaled_weight_square_sum == 0.0
        } else {
            self.mean.is_finite()
                && self.weight_scale.is_finite()
                && self.weight_scale > 0.0
                && self.scaled_weight_sum.is_finite()
                && self.scaled_weight_sum > 0.0
                && self.scaled_weight_square_sum.is_finite()
                && self.scaled_weight_square_sum > 0.0
                && self.effective_sample_size().is_finite()
        };
        if !stored_state_is_valid {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("stored OnlineWeightedMean state is invalid")
                    .build(),
            );
            return Err(reject_metric_batch(
                ctx,
                self.updates,
                x.nrows(),
                self.n_seen,
            ));
        }

        let before = self.score();
        let before_effective_sample_size = self.effective_sample_size();
        let mut candidate_mean = self.mean;
        let mut candidate_weight_scale = self.weight_scale;
        let mut candidate_scaled_weight_sum = self.scaled_weight_sum;
        let mut candidate_scaled_weight_square_sum = self.scaled_weight_square_sum;
        for row in 0..x.nrows() {
            let value = x.get(row, 0);
            let weight = weights[row];
            if candidate_weight_scale == 0.0 {
                candidate_mean = value;
                candidate_weight_scale = weight;
                candidate_scaled_weight_sum = 1.0;
                candidate_scaled_weight_square_sum = 1.0;
                continue;
            }

            let (old_mass, new_mass, next_scale, old_square_scale) =
                if weight > candidate_weight_scale {
                    let scale_ratio = candidate_weight_scale / weight;
                    (
                        candidate_scaled_weight_sum * scale_ratio,
                        1.0,
                        weight,
                        scale_ratio * scale_ratio,
                    )
                } else {
                    let relative_weight = weight / candidate_weight_scale;
                    (
                        candidate_scaled_weight_sum,
                        relative_weight,
                        candidate_weight_scale,
                        1.0,
                    )
                };
            let new_square = new_mass * new_mass;
            let old_square_mass = candidate_scaled_weight_square_sum * old_square_scale;
            if old_mass == 0.0 || new_mass == 0.0 || old_square_mass == 0.0 || new_square == 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NumericalUnderflow)
                        .message(format!(
                            "OnlineWeightedMean relative weight underflowed at row {row}"
                        ))
                        .build(),
                );
                return Err(reject_metric_batch(
                    ctx,
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                ));
            }
            let next_weight_sum = old_mass + new_mass;
            let next_weight_square_sum = old_square_mass + new_square;
            let old_share = old_mass / next_weight_sum;
            let new_share = new_mass / next_weight_sum;
            let next_mean = old_share * candidate_mean + new_share * value;
            let finite_update = next_scale.is_finite()
                && next_scale > 0.0
                && next_weight_sum.is_finite()
                && next_weight_sum > 0.0
                && next_weight_square_sum.is_finite()
                && next_weight_square_sum > 0.0
                && next_mean.is_finite();
            if !finite_update {
                ctx.push(
                    Issue::builder(IssueCode::NumericalOverflow)
                        .message(format!(
                            "OnlineWeightedMean update produced a non-finite state at row {row}"
                        ))
                        .build(),
                );
                return Err(reject_metric_batch(
                    ctx,
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                ));
            }
            candidate_mean = next_mean;
            candidate_weight_scale = next_scale;
            candidate_scaled_weight_sum = next_weight_sum;
            candidate_scaled_weight_square_sum = next_weight_square_sum;
        }

        self.mean = candidate_mean;
        self.weight_scale = candidate_weight_scale;
        self.scaled_weight_sum = candidate_scaled_weight_sum;
        self.scaled_weight_square_sum = candidate_scaled_weight_square_sum;
        self.n_seen = new_n_seen;
        self.updates = new_updates;
        let after = self.score();
        let after_effective_sample_size = self.effective_sample_size();
        let mut quality = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        quality.effective_sample_size = after_effective_sample_size;
        quality.parameter_delta_norm = before.is_finite().then(|| (after - before).abs());
        quality.information_gain =
            Some((after_effective_sample_size - before_effective_sample_size).abs());
        quality.still_identified = self.n_seen >= 1;
        quality.warmup = self.n_seen < 1;
        quality.explanation =
            format!("OnlineWeightedMean={after:.6e}, Kish n_eff={after_effective_sample_size:.6e}");
        flag_info(&mut ctx, &quality);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                quality,
                "weighted mean",
                "positive weights in y; scale-normalized weighted recurrence",
                format!("m={before:.6e}"),
                format!("m={after:.6e}, Kish n_eff={after_effective_sample_size:.6e}"),
            ),
        )
    }
}

fn preflight<'a>(
    session: &Session,
    x: &Matrix,
    y: Option<&'a Vector>,
    updates: u64,
    n_seen: u64,
) -> std::result::Result<(FitCtx, &'a Vector, u64), Failure> {
    let mut ctx = FitCtx::with_session(session.child("partial_fit"));
    ctx.report.set_sample_shape(x.nrows(), x.ncols());
    let Some(weights) = y else {
        ctx.push(Issue::builder(IssueCode::MissingTarget).build());
        return Err(reject_metric_batch(ctx, updates, x.nrows(), n_seen));
    };

    let mut invalid = false;
    if x.nrows() == 0 || x.ncols() == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message(format!(
                    "online metric predictions must be non-empty; got shape {}×{}",
                    x.nrows(),
                    x.ncols()
                ))
                .build(),
        );
        invalid = true;
    }
    if weights.len() != x.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "metric target length {} does not match {} prediction rows",
                    weights.len(),
                    x.nrows()
                ))
                .build(),
        );
        invalid = true;
    }
    let non_finite_row = if invalid {
        None
    } else {
        (0..x.nrows()).find(|&row| !x.get(row, 0).is_finite() || !weights[row].is_finite())
    };
    if let Some(row) = non_finite_row {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message(format!(
                    "online metric pair at row {row} contains a non-finite prediction or target"
                ))
                .build(),
        );
        invalid = true;
    }

    let after_n = u64::try_from(x.nrows())
        .ok()
        .and_then(|batch| n_seen.checked_add(batch));
    if after_n.is_none() {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message("online metric observation counter overflowed")
                .build(),
        );
        invalid = true;
    }

    if invalid {
        Err(reject_metric_batch(ctx, updates, x.nrows(), n_seen))
    } else {
        Ok((ctx, weights, after_n.expect("validated metric counter")))
    }
}
