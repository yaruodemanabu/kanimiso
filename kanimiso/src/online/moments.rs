use super::common::{
    checked_compensated_add, checked_difference_product, checked_online_covariance_step,
    finish_explain, finite_score_delta, flag_info, online_mean_step, online_moment_preflight,
    reject_metric_batch,
};
use crate::data::{Matrix, Vector};
use crate::traits::PartialFit;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{IncrementalQuality, Issue, IssueCode, Qualified, Result};

/// Streaming pairwise sample covariance (river `stats.Cov`).
///
/// Column 1 takes precedence when present; otherwise `y` supplies the paired
/// observations. Invalid batches and unrepresentable arithmetic leave the
/// state unchanged.
#[derive(Clone, Debug, Default)]
pub struct OnlineCovariance {
    pub(super) n: u64,
    pub(super) mean_x: f64,
    pub(super) mean_y: f64,
    pub(super) cross: f64,
    pub(super) cross_compensation: f64,
    pub(super) updates: u64,
}

impl OnlineCovariance {
    /// Construct an empty covariance accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the current sample covariance, or NaN during warmup.
    pub fn score(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            (self.cross + self.cross_compensation) / (self.n - 1) as f64
        }
    }
}

impl PartialFit for OnlineCovariance {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let (mut ctx, new_n, new_updates) = match online_moment_preflight(
            session,
            x,
            y,
            true,
            "OnlineCovariance",
            self.updates,
            self.n,
        ) {
            Ok(validated) => validated,
            Err(failure) => return Err(failure),
        };
        let stored_state_is_valid = if self.n == 0 {
            self.mean_x == 0.0
                && self.mean_y == 0.0
                && self.cross == 0.0
                && self.cross_compensation == 0.0
        } else {
            self.mean_x.is_finite()
                && self.mean_y.is_finite()
                && self.cross.is_finite()
                && self.cross_compensation.is_finite()
                && (self.cross + self.cross_compensation).is_finite()
        };
        if !stored_state_is_valid {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("stored OnlineCovariance state is invalid")
                    .build(),
            );
            return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
        }

        let before = self.score();
        let mut candidate_n = self.n;
        let mut candidate_mean_x = self.mean_x;
        let mut candidate_mean_y = self.mean_y;
        let mut candidate_cross = self.cross;
        let mut candidate_cross_compensation = self.cross_compensation;
        for row in 0..x.nrows() {
            let left = x.get(row, 0);
            let right = if x.ncols() >= 2 {
                x.get(row, 1)
            } else {
                y.expect("validated pairwise target")[row]
            };
            let (next_mean_x, next_mean_y, increment) = match checked_online_covariance_step(
                candidate_n,
                candidate_mean_x,
                candidate_mean_y,
                left,
                right,
            ) {
                Ok(step) => step,
                Err(code) => {
                    ctx.push(
                        Issue::builder(code)
                            .message(format!(
                                "OnlineCovariance cross-moment is not representable at row {row}"
                            ))
                            .build(),
                    );
                    return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                }
            };
            let Some(row_count) = candidate_n.checked_add(1) else {
                ctx.push(
                    Issue::builder(IssueCode::InvalidParameter)
                        .message("OnlineCovariance observation counter overflowed")
                        .build(),
                );
                return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
            };
            let (next_cross, next_compensation) = match checked_compensated_add(
                candidate_cross,
                candidate_cross_compensation,
                increment,
            ) {
                Ok(candidate) => candidate,
                Err(code) => {
                    ctx.push(
                        Issue::builder(code)
                            .message(format!(
                                "OnlineCovariance could not accumulate its cross-moment at row {row}"
                            ))
                            .build(),
                    );
                    return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                }
            };
            candidate_n = row_count;
            candidate_mean_x = next_mean_x;
            candidate_mean_y = next_mean_y;
            candidate_cross = next_cross;
            candidate_cross_compensation = next_compensation;
        }

        debug_assert_eq!(candidate_n, new_n);
        self.n = new_n;
        self.mean_x = candidate_mean_x;
        self.mean_y = candidate_mean_y;
        self.cross = candidate_cross;
        self.cross_compensation = candidate_cross_compensation;
        self.updates = new_updates;
        let after = self.score();
        let mut quality = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n);
        quality.effective_sample_size = self.n as f64;
        quality.parameter_delta_norm = finite_score_delta(before, after);
        quality.information_gain = Some(x.nrows() as f64);
        quality.still_identified = self.n >= 2;
        quality.warmup = self.n < 2;
        quality.explanation = format!("OnlineCovariance={after:.6e}");
        flag_info(&mut ctx, &quality);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                quality,
                "Welford covariance",
                "pairwise covariance of column 0 with column 1 or y",
                format!("cov={before:.6e}"),
                format!("cov={after:.6e}"),
            ),
        )
    }
}

/// Streaming arithmetic mean (river `stats.Mean`).
///
/// Opposite-signed extreme values use a convex-combination update. Invalid
/// batches and unrepresentable arithmetic leave the state unchanged.
#[derive(Clone, Debug, Default)]
pub struct OnlineMean {
    pub(super) n: u64,
    pub(super) mean: f64,
    pub(super) updates: u64,
}

impl OnlineMean {
    /// Construct an empty mean accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the current mean, or NaN when empty.
    pub fn score(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.mean
        }
    }
}

impl PartialFit for OnlineMean {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let (mut ctx, new_n, new_updates) = match online_moment_preflight(
            session,
            x,
            None,
            false,
            "OnlineMean",
            self.updates,
            self.n,
        ) {
            Ok(validated) => validated,
            Err(failure) => return Err(failure),
        };
        let stored_state_is_valid = if self.n == 0 {
            self.mean == 0.0
        } else {
            self.mean.is_finite()
        };
        if !stored_state_is_valid {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("stored OnlineMean state is invalid")
                    .build(),
            );
            return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
        }

        let before = self.score();
        let mut candidate_n = self.n;
        let mut candidate_mean = self.mean;
        for row in 0..x.nrows() {
            let value = x.get(row, 0);
            let row_count = candidate_n + 1;
            let next_mean = if candidate_n == 0 {
                value
            } else {
                online_mean_step(candidate_mean, value, row_count)
            };
            if !next_mean.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NumericalOverflow)
                        .message(format!(
                            "OnlineMean update produced a non-finite state at row {row}"
                        ))
                        .build(),
                );
                return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
            }
            candidate_n = row_count;
            candidate_mean = next_mean;
        }

        debug_assert_eq!(candidate_n, new_n);
        self.n = new_n;
        self.mean = candidate_mean;
        self.updates = new_updates;
        let after = self.score();
        finish_scalar_update(
            ctx,
            self.updates,
            x.nrows(),
            self.n,
            before,
            after,
            self.n >= 1,
            self.n < 1,
            "OnlineMean",
            "Welford mean",
            "running mean of column 0",
            "mean",
        )
    }
}

/// Streaming compensated sum (river `stats.Sum`).
#[derive(Clone, Debug, Default)]
pub struct OnlineSum {
    pub(super) n: u64,
    pub(super) sum: f64,
    pub(super) compensation: f64,
    pub(super) updates: u64,
}

impl OnlineSum {
    /// Construct an empty sum.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the current compensated sum.
    pub fn score(&self) -> f64 {
        self.sum + self.compensation
    }
}

impl PartialFit for OnlineSum {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let (mut ctx, new_n, new_updates) = match online_moment_preflight(
            session,
            x,
            None,
            false,
            "OnlineSum",
            self.updates,
            self.n,
        ) {
            Ok(validated) => validated,
            Err(failure) => return Err(failure),
        };
        let stored_state_is_valid = if self.n == 0 {
            self.sum == 0.0 && self.compensation == 0.0
        } else {
            self.sum.is_finite() && self.compensation.is_finite() && self.score().is_finite()
        };
        if !stored_state_is_valid {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("stored OnlineSum state is invalid")
                    .build(),
            );
            return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
        }

        let before = self.score();
        let mut candidate_sum = self.sum;
        let mut candidate_compensation = self.compensation;
        for row in 0..x.nrows() {
            match checked_compensated_add(candidate_sum, candidate_compensation, x.get(row, 0)) {
                Ok((next_sum, next_compensation)) => {
                    candidate_sum = next_sum;
                    candidate_compensation = next_compensation;
                }
                Err(code) => {
                    ctx.push(
                        Issue::builder(code)
                            .message(format!(
                                "OnlineSum compensated total is not representable at row {row}"
                            ))
                            .build(),
                    );
                    return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                }
            }
        }

        self.n = new_n;
        self.sum = candidate_sum;
        self.compensation = candidate_compensation;
        self.updates = new_updates;
        let after = self.score();
        finish_scalar_update(
            ctx,
            self.updates,
            x.nrows(),
            self.n,
            before,
            after,
            self.n > 0,
            self.n == 0,
            "OnlineSum",
            "running sum",
            "compensated sum of column 0",
            "sum",
        )
    }
}

/// Streaming sample variance (river `stats.Var`).
#[derive(Clone, Debug, Default)]
pub struct OnlineVar {
    pub(super) n: u64,
    pub(super) mean: f64,
    pub(super) m2: f64,
    pub(super) m2_compensation: f64,
    pub(super) updates: u64,
}

impl OnlineVar {
    /// Construct an empty variance accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the current sample variance, or NaN during warmup.
    pub fn score(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            (self.m2 + self.m2_compensation) / (self.n - 1) as f64
        }
    }
}

impl PartialFit for OnlineVar {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let (mut ctx, new_n, new_updates) = match online_moment_preflight(
            session,
            x,
            None,
            false,
            "OnlineVar",
            self.updates,
            self.n,
        ) {
            Ok(validated) => validated,
            Err(failure) => return Err(failure),
        };
        let stored_state_is_valid = if self.n == 0 {
            self.mean == 0.0 && self.m2 == 0.0 && self.m2_compensation == 0.0
        } else {
            self.mean.is_finite()
                && self.m2.is_finite()
                && self.m2_compensation.is_finite()
                && (self.m2 + self.m2_compensation).is_finite()
                && self.m2 + self.m2_compensation >= 0.0
        };
        if !stored_state_is_valid {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("stored OnlineVar state is invalid")
                    .build(),
            );
            return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
        }

        let before = self.score();
        let mut candidate_n = self.n;
        let mut candidate_mean = self.mean;
        let mut candidate_m2 = self.m2;
        let mut candidate_compensation = self.m2_compensation;
        for row in 0..x.nrows() {
            let value = x.get(row, 0);
            let row_count = candidate_n + 1;
            let next_mean = if candidate_n == 0 {
                value
            } else {
                online_mean_step(candidate_mean, value, row_count)
            };
            let increment = if candidate_n == 0 {
                0.0
            } else {
                match checked_difference_product(value, candidate_mean, value, next_mean) {
                    Ok(increment) => increment,
                    Err(code) => {
                        ctx.push(
                            Issue::builder(code)
                                .message(format!(
                                    "OnlineVar second central moment is not representable at row {row}"
                                ))
                                .build(),
                        );
                        return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                    }
                }
            };
            if increment < 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message(format!(
                            "OnlineVar produced a negative second-moment increment at row {row}"
                        ))
                        .build(),
                );
                return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
            }
            let (next_m2, next_compensation) = match checked_compensated_add(
                candidate_m2,
                candidate_compensation,
                increment,
            ) {
                Ok(candidate) => candidate,
                Err(code) => {
                    ctx.push(
                        Issue::builder(code)
                            .message(format!(
                                "OnlineVar could not accumulate its second central moment at row {row}"
                            ))
                            .build(),
                    );
                    return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                }
            };
            if !next_mean.is_finite() || next_m2 + next_compensation < 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message(format!(
                            "OnlineVar update produced an invalid state at row {row}"
                        ))
                        .build(),
                );
                return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
            }
            candidate_n = row_count;
            candidate_mean = next_mean;
            candidate_m2 = next_m2;
            candidate_compensation = next_compensation;
        }

        debug_assert_eq!(candidate_n, new_n);
        self.n = new_n;
        self.mean = candidate_mean;
        self.m2 = candidate_m2;
        self.m2_compensation = candidate_compensation;
        self.updates = new_updates;
        let after = self.score();
        finish_scalar_update(
            ctx,
            self.updates,
            x.nrows(),
            self.n,
            before,
            after,
            self.n >= 2,
            self.n < 2,
            "OnlineVar",
            "Welford variance",
            "running sample variance of column 0",
            "var",
        )
    }
}

/// Streaming finite observation count (river `stats.Count`).
#[derive(Clone, Debug, Default)]
pub struct OnlineCount {
    pub(super) n: u64,
    pub(super) updates: u64,
}

impl OnlineCount {
    /// Construct an empty counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the current count as a score.
    pub fn score(&self) -> f64 {
        self.n as f64
    }
}

impl PartialFit for OnlineCount {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let (ctx, new_n, new_updates) = match online_moment_preflight(
            session,
            x,
            None,
            false,
            "OnlineCount",
            self.updates,
            self.n,
        ) {
            Ok(validated) => validated,
            Err(failure) => return Err(failure),
        };
        let before = self.score();
        self.n = new_n;
        self.updates = new_updates;
        let after = self.score();
        finish_scalar_update(
            ctx,
            self.updates,
            x.nrows(),
            self.n,
            before,
            after,
            self.n > 0,
            self.n == 0,
            "OnlineCount",
            "count update",
            "finite entries in column 0",
            "n",
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_scalar_update(
    mut ctx: crate::context::FitCtx,
    updates: u64,
    batch: usize,
    n_seen: u64,
    before: f64,
    after: f64,
    identified: bool,
    warmup: bool,
    name: &str,
    what: &str,
    why: &str,
    score_name: &str,
) -> Result<Qualified<IncrementalExplain>> {
    let mut quality = IncrementalQuality::new(updates - 1, batch, n_seen);
    quality.effective_sample_size = n_seen as f64;
    quality.parameter_delta_norm = finite_score_delta(before, after);
    quality.information_gain = Some(batch as f64);
    quality.still_identified = identified;
    quality.warmup = warmup;
    quality.explanation = format!("{name}={after:.6e}");
    flag_info(&mut ctx, &quality);
    finish_explain(
        ctx,
        IncrementalExplain::from_quality(
            quality,
            what,
            why,
            format!("{score_name}={before:.6e}"),
            format!("{score_name}={after:.6e}"),
        ),
    )
}
