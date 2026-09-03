use super::common::{finish_explain, flag_info, reject_metric_batch};
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::PartialFit;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{Failure, IncrementalQuality, Issue, IssueCode, Qualified, Result};

/// Exponentially weighted mean (river `stats.EWMean`).
///
/// The first observation receives unit mass. Each later observation scales
/// old weights by `1 - alpha` and receives mass `alpha`.
#[derive(Clone, Debug)]
pub struct OnlineEwMean {
    /// Smoothing parameter `alpha` in `(0, 1]`.
    pub alpha: f64,
    pub(super) mean: f64,
    pub(super) weight_square_sum: f64,
    pub(super) n_seen: u64,
    pub(super) updates: u64,
}

impl Default for OnlineEwMean {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            mean: 0.0,
            weight_square_sum: 0.0,
            n_seen: 0,
            updates: 0,
        }
    }
}

impl OnlineEwMean {
    /// Construct an exponentially weighted mean with the given `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::default()
        }
    }

    /// Return the current mean, or NaN before the first observation.
    pub fn score(&self) -> f64 {
        if self.n_seen == 0 {
            f64::NAN
        } else {
            self.mean
        }
    }

    pub(super) fn effective_sample_size(&self) -> f64 {
        effective_sample_size(self.n_seen, self.weight_square_sum)
    }
}

impl PartialFit for OnlineEwMean {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let (mut ctx, new_n_seen, new_updates) =
            match preflight(session, x, self.alpha, self.updates, self.n_seen) {
                Ok(validated) => validated,
                Err(failure) => return Err(failure),
            };
        let stored_state_is_valid = if self.n_seen == 0 {
            self.mean == 0.0 && self.weight_square_sum == 0.0
        } else {
            self.mean.is_finite()
                && self.weight_square_sum.is_finite()
                && self.weight_square_sum > 0.0
                && self.effective_sample_size().is_finite()
        };
        if !stored_state_is_valid {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("stored OnlineEwMean state is invalid")
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
        let mut candidate_weight_square_sum = self.weight_square_sum;
        let mut initialized = self.n_seen > 0;
        for row in 0..x.nrows() {
            let value = x.get(row, 0);
            candidate_mean = if initialized {
                mean_step(candidate_mean, value, self.alpha)
            } else {
                value
            };
            candidate_weight_square_sum =
                weight_square_step(candidate_weight_square_sum, initialized, self.alpha);
            initialized = true;
            if !candidate_mean.is_finite()
                || !candidate_weight_square_sum.is_finite()
                || candidate_weight_square_sum <= 0.0
            {
                ctx.push(
                    Issue::builder(IssueCode::NumericalOverflow)
                        .message(format!(
                            "OnlineEwMean update produced a non-finite state at row {row}"
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
        }

        self.mean = candidate_mean;
        self.weight_square_sum = candidate_weight_square_sum;
        self.n_seen = new_n_seen;
        self.updates = new_updates;
        let after = self.score();
        let after_effective_sample_size = self.effective_sample_size();
        let mut quality = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        quality.effective_sample_size = after_effective_sample_size;
        quality.forgetting_factor = Some(1.0 - self.alpha);
        quality.parameter_delta_norm = before.is_finite().then(|| (after - before).abs());
        quality.information_gain =
            Some((after_effective_sample_size - before_effective_sample_size).abs());
        quality.still_identified = self.n_seen >= 1;
        quality.warmup = self.n_seen < 1;
        quality.explanation = format!(
            "OnlineEwMean={after:.6e}, alpha={:.6e}, Kish n_eff={after_effective_sample_size:.6e}",
            self.alpha
        );
        flag_info(&mut ctx, &quality);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                quality,
                "exponentially weighted mean",
                "finite-stream EW recurrence on column 0",
                format!("ew={before:.6e}"),
                format!("ew={after:.6e}, Kish n_eff={after_effective_sample_size:.6e}"),
            ),
        )
    }
}

/// Exponentially weighted population variance (river `stats.EWVar`).
#[derive(Clone, Debug)]
pub struct OnlineEwVar {
    /// Smoothing parameter `alpha` in `(0, 1]`.
    pub alpha: f64,
    pub(super) mean: f64,
    pub(super) var: f64,
    pub(super) weight_square_sum: f64,
    pub(super) n_seen: u64,
    pub(super) updates: u64,
}

impl Default for OnlineEwVar {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            mean: 0.0,
            var: 0.0,
            weight_square_sum: 0.0,
            n_seen: 0,
            updates: 0,
        }
    }
}

impl OnlineEwVar {
    /// Construct an exponentially weighted variance with the given `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::default()
        }
    }

    /// Return the current variance, or NaN during warmup.
    pub fn score(&self) -> f64 {
        if self.n_seen < 2 {
            f64::NAN
        } else {
            self.var
        }
    }

    pub(super) fn effective_sample_size(&self) -> f64 {
        effective_sample_size(self.n_seen, self.weight_square_sum)
    }
}

impl PartialFit for OnlineEwVar {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let (mut ctx, new_n_seen, new_updates) =
            match preflight(session, x, self.alpha, self.updates, self.n_seen) {
                Ok(validated) => validated,
                Err(failure) => return Err(failure),
            };
        let stored_state_is_valid = if self.n_seen == 0 {
            self.mean == 0.0 && self.var == 0.0 && self.weight_square_sum == 0.0
        } else {
            self.mean.is_finite()
                && self.var.is_finite()
                && self.var >= 0.0
                && self.weight_square_sum.is_finite()
                && self.weight_square_sum > 0.0
                && self.effective_sample_size().is_finite()
        };
        if !stored_state_is_valid {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("stored OnlineEwVar state is invalid")
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
        let mut candidate_var = self.var;
        let mut candidate_weight_square_sum = self.weight_square_sum;
        let mut initialized = self.n_seen > 0;
        let decay = 1.0 - self.alpha;
        let sqrt_alpha = self.alpha.sqrt();
        for row in 0..x.nrows() {
            let value = x.get(row, 0);
            if initialized {
                if decay == 0.0 {
                    candidate_mean = value;
                    candidate_var = 0.0;
                } else {
                    let scaled_difference = sqrt_alpha * value - sqrt_alpha * candidate_mean;
                    let innovation_variance = scaled_difference * scaled_difference;
                    candidate_var = decay * candidate_var + decay * innovation_variance;
                    candidate_mean = mean_step(candidate_mean, value, self.alpha);
                }
            } else {
                candidate_mean = value;
                candidate_var = 0.0;
            }
            candidate_weight_square_sum =
                weight_square_step(candidate_weight_square_sum, initialized, self.alpha);
            initialized = true;
            if !candidate_mean.is_finite()
                || !candidate_var.is_finite()
                || candidate_var < 0.0
                || !candidate_weight_square_sum.is_finite()
                || candidate_weight_square_sum <= 0.0
            {
                ctx.push(
                    Issue::builder(IssueCode::NumericalOverflow)
                        .message(format!(
                            "OnlineEwVar update produced a non-finite state at row {row}"
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
        }

        self.mean = candidate_mean;
        self.var = candidate_var;
        self.weight_square_sum = candidate_weight_square_sum;
        self.n_seen = new_n_seen;
        self.updates = new_updates;
        let after = self.score();
        let after_effective_sample_size = self.effective_sample_size();
        let mut quality = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        quality.effective_sample_size = after_effective_sample_size;
        quality.forgetting_factor = Some(1.0 - self.alpha);
        quality.parameter_delta_norm = before.is_finite().then(|| (after - before).abs());
        quality.information_gain =
            Some((after_effective_sample_size - before_effective_sample_size).abs());
        quality.still_identified = self.n_seen >= 2 && after_effective_sample_size > 1.0;
        quality.warmup = !quality.still_identified;
        quality.explanation = format!(
            "OnlineEwVar={after:.6e}, alpha={:.6e}, Kish n_eff={after_effective_sample_size:.6e}",
            self.alpha
        );
        flag_info(&mut ctx, &quality);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                quality,
                "exponentially weighted variance",
                "normalized EW population moment on column 0",
                format!("v={before:.6e}"),
                format!("v={after:.6e}, Kish n_eff={after_effective_sample_size:.6e}"),
            ),
        )
    }
}

fn preflight(
    session: &Session,
    x: &Matrix,
    alpha: f64,
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
                    "exponentially weighted update requires a non-empty column; got {}×{}",
                    x.nrows(),
                    x.ncols()
                ))
                .build(),
        );
        invalid = true;
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!(
                    "exponentially weighted smoothing alpha must be finite and in (0, 1]; got {alpha}"
                ))
                .build(),
        );
        invalid = true;
    } else if alpha < 1.0 && 1.0 - alpha == 1.0 {
        ctx.push(
            Issue::builder(IssueCode::NumericalUnderflow)
                .message(format!(
                    "EW alpha={alpha} is positive but rounds 1 - alpha to 1; new observations would receive no representable weight"
                ))
                .build(),
        );
        invalid = true;
    }
    if x.ncols() > 0 && (0..x.nrows()).any(|row| !x.get(row, 0).is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("exponentially weighted observations in column 0 must be finite")
                .build(),
        );
        invalid = true;
    }

    let batch_rows = match u64::try_from(x.nrows()) {
        Ok(rows) => Some(rows),
        Err(_) => {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("EW batch row count cannot be represented by its counter")
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
                .message("EW observation counter overflowed")
                .build(),
        );
        invalid = true;
    }
    let new_updates = updates.checked_add(1);
    if new_updates.is_none() {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message("EW update counter overflowed")
                .build(),
        );
        invalid = true;
    }

    match (invalid, new_n_seen, new_updates) {
        (false, Some(new_n_seen), Some(new_updates)) => Ok((ctx, new_n_seen, new_updates)),
        _ => Err(reject_metric_batch(ctx, updates, x.nrows(), n_seen)),
    }
}

fn mean_step(previous: f64, observation: f64, alpha: f64) -> f64 {
    alpha.mul_add(observation, (1.0 - alpha) * previous)
}

fn weight_square_step(current: f64, initialized: bool, alpha: f64) -> f64 {
    if initialized {
        let decay = 1.0 - alpha;
        (decay * decay).mul_add(current, alpha * alpha)
    } else {
        1.0
    }
}

fn effective_sample_size(n_seen: u64, weight_square_sum: f64) -> f64 {
    if n_seen == 0 {
        0.0
    } else {
        1.0 / weight_square_sum
    }
}
