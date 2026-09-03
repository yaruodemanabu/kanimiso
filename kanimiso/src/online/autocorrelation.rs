use super::common::{
    autocorrelation_from_moments, checked_compensated_add, checked_online_covariance_step,
    finish_explain, finite_score_delta, flag_info, online_moment_preflight, reject_metric_batch,
};
use crate::data::{Matrix, Vector};
use crate::traits::PartialFit;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{IncrementalQuality, Issue, IssueCode, Qualified, Result};

/// Streaming lag-1 sample correlation of consecutive observations.
///
/// This is the Pearson correlation of `z[..n-1]` with `z[1..]`. A batch
/// boundary contributes `(previous_last, current_first)` exactly once.
#[derive(Clone, Debug, Default)]
pub struct OnlineAutoCorr {
    pub(super) last: Option<f64>,
    pub(super) n: u64,
    pub(super) mean_lagged: f64,
    pub(super) mean_current: f64,
    pub(super) cross: f64,
    pub(super) cross_compensation: f64,
    pub(super) lagged_m2: f64,
    pub(super) lagged_m2_compensation: f64,
    pub(super) current_m2: f64,
    pub(super) current_m2_compensation: f64,
    pub(super) updates: u64,
}

impl OnlineAutoCorr {
    /// Construct an empty lag-1 accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the lag-1 correlation.
    ///
    /// Returns NaN before three observations or when either paired marginal
    /// is constant. No denominator floor or result clamp is applied.
    pub fn score(&self) -> f64 {
        if self.n < 3 {
            return f64::NAN;
        }
        autocorrelation_from_moments(
            self.cross + self.cross_compensation,
            self.lagged_m2 + self.lagged_m2_compensation,
            self.current_m2 + self.current_m2_compensation,
        )
    }
}

impl PartialFit for OnlineAutoCorr {
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
            "OnlineAutoCorr",
            self.updates,
            self.n,
        ) {
            Ok(validated) => validated,
            Err(failure) => return Err(failure),
        };
        let stored_state_is_valid = match (self.n, self.last) {
            (0, None) => self.zero_moments(),
            (1, Some(last)) => last.is_finite() && self.zero_moments(),
            (n, Some(last)) if n >= 2 => {
                last.is_finite()
                    && self.mean_lagged.is_finite()
                    && self.mean_current.is_finite()
                    && self.cross.is_finite()
                    && self.cross_compensation.is_finite()
                    && (self.cross + self.cross_compensation).is_finite()
                    && self.lagged_m2.is_finite()
                    && self.lagged_m2_compensation.is_finite()
                    && (self.lagged_m2 + self.lagged_m2_compensation).is_finite()
                    && self.lagged_m2 + self.lagged_m2_compensation >= 0.0
                    && self.current_m2.is_finite()
                    && self.current_m2_compensation.is_finite()
                    && (self.current_m2 + self.current_m2_compensation).is_finite()
                    && self.current_m2 + self.current_m2_compensation >= 0.0
            }
            _ => false,
        };
        if !stored_state_is_valid {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("stored OnlineAutoCorr state is invalid")
                    .build(),
            );
            return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
        }

        let before = self.score();
        let before_pairs = self.n.saturating_sub(1);
        let mut candidate_last = self.last;
        let mut candidate_n = self.n;
        let mut candidate_mean_lagged = self.mean_lagged;
        let mut candidate_mean_current = self.mean_current;
        let mut candidate_cross = self.cross;
        let mut candidate_cross_compensation = self.cross_compensation;
        let mut candidate_lagged_m2 = self.lagged_m2;
        let mut candidate_lagged_compensation = self.lagged_m2_compensation;
        let mut candidate_current_m2 = self.current_m2;
        let mut candidate_current_compensation = self.current_m2_compensation;

        for row in 0..x.nrows() {
            let current = x.get(row, 0);
            if let Some(previous) = candidate_last {
                let pair_count = candidate_n - 1;
                let (next_mean_lagged, next_mean_current, cross_increment) =
                    match checked_online_covariance_step(
                        pair_count,
                        candidate_mean_lagged,
                        candidate_mean_current,
                        previous,
                        current,
                    ) {
                        Ok(step) => step,
                        Err(code) => {
                            ctx.push(
                                Issue::builder(code)
                                    .message(format!(
                                        "OnlineAutoCorr cross-moment is not representable at row {row}"
                                    ))
                                    .build(),
                            );
                            return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                        }
                    };
                let (_, _, lagged_increment) = match checked_online_covariance_step(
                    pair_count,
                    candidate_mean_lagged,
                    candidate_mean_lagged,
                    previous,
                    previous,
                ) {
                    Ok(step) => step,
                    Err(code) => {
                        ctx.push(
                            Issue::builder(code)
                                .message(format!(
                                    "OnlineAutoCorr lagged variance is not representable at row {row}"
                                ))
                                .build(),
                        );
                        return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                    }
                };
                let (_, _, current_increment) = match checked_online_covariance_step(
                    pair_count,
                    candidate_mean_current,
                    candidate_mean_current,
                    current,
                    current,
                ) {
                    Ok(step) => step,
                    Err(code) => {
                        ctx.push(
                            Issue::builder(code)
                                .message(format!(
                                    "OnlineAutoCorr current variance is not representable at row {row}"
                                ))
                                .build(),
                        );
                        return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                    }
                };
                if lagged_increment < 0.0 || current_increment < 0.0 {
                    ctx.push(
                        Issue::builder(IssueCode::NonFiniteOutput)
                            .message(format!(
                                "OnlineAutoCorr produced a negative variance increment at row {row}"
                            ))
                            .build(),
                    );
                    return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                }

                let (next_cross, next_cross_compensation) = match checked_compensated_add(
                    candidate_cross,
                    candidate_cross_compensation,
                    cross_increment,
                ) {
                    Ok(next) => next,
                    Err(code) => {
                        ctx.push(
                                Issue::builder(code)
                                    .message(format!(
                                        "OnlineAutoCorr could not accumulate its cross-moment at row {row}"
                                    ))
                                    .build(),
                            );
                        return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                    }
                };
                let (next_lagged_m2, next_lagged_compensation) = match checked_compensated_add(
                    candidate_lagged_m2,
                    candidate_lagged_compensation,
                    lagged_increment,
                ) {
                    Ok(next) => next,
                    Err(code) => {
                        ctx.push(
                                Issue::builder(code)
                                    .message(format!(
                                        "OnlineAutoCorr could not accumulate its lagged variance at row {row}"
                                    ))
                                    .build(),
                            );
                        return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                    }
                };
                let (next_current_m2, next_current_compensation) = match checked_compensated_add(
                    candidate_current_m2,
                    candidate_current_compensation,
                    current_increment,
                ) {
                    Ok(next) => next,
                    Err(code) => {
                        ctx.push(
                                Issue::builder(code)
                                    .message(format!(
                                        "OnlineAutoCorr could not accumulate its current variance at row {row}"
                                    ))
                                    .build(),
                            );
                        return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                    }
                };
                if next_lagged_m2 + next_lagged_compensation < 0.0
                    || next_current_m2 + next_current_compensation < 0.0
                {
                    ctx.push(
                        Issue::builder(IssueCode::NonFiniteOutput)
                            .message(format!(
                                "OnlineAutoCorr update produced an invalid variance at row {row}"
                            ))
                            .build(),
                    );
                    return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                }

                candidate_mean_lagged = next_mean_lagged;
                candidate_mean_current = next_mean_current;
                candidate_cross = next_cross;
                candidate_cross_compensation = next_cross_compensation;
                candidate_lagged_m2 = next_lagged_m2;
                candidate_lagged_compensation = next_lagged_compensation;
                candidate_current_m2 = next_current_m2;
                candidate_current_compensation = next_current_compensation;
            }
            candidate_last = Some(current);
            candidate_n = match candidate_n.checked_add(1) {
                Some(next) => next,
                None => {
                    ctx.push(
                        Issue::builder(IssueCode::InvalidParameter)
                            .message("OnlineAutoCorr observation counter overflowed")
                            .build(),
                    );
                    return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
                }
            };
        }

        let candidate_lagged_total = candidate_lagged_m2 + candidate_lagged_compensation;
        let candidate_current_total = candidate_current_m2 + candidate_current_compensation;
        if candidate_n >= 3 && candidate_lagged_total > 0.0 && candidate_current_total > 0.0 {
            let candidate_correlation = autocorrelation_from_moments(
                candidate_cross + candidate_cross_compensation,
                candidate_lagged_total,
                candidate_current_total,
            );
            if !candidate_correlation.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NumericalOverflow)
                        .message("OnlineAutoCorr correlation is not representable")
                        .build(),
                );
                return Err(reject_metric_batch(ctx, self.updates, x.nrows(), self.n));
            }
        }

        debug_assert_eq!(candidate_n, new_n);
        self.last = candidate_last;
        self.n = new_n;
        self.mean_lagged = candidate_mean_lagged;
        self.mean_current = candidate_mean_current;
        self.cross = candidate_cross;
        self.cross_compensation = candidate_cross_compensation;
        self.lagged_m2 = candidate_lagged_m2;
        self.lagged_m2_compensation = candidate_lagged_compensation;
        self.current_m2 = candidate_current_m2;
        self.current_m2_compensation = candidate_current_compensation;
        self.updates = new_updates;

        let after = self.score();
        let pair_count = self.n - 1;
        let new_pairs = pair_count - before_pairs;
        let mut quality = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n);
        quality.effective_sample_size = pair_count as f64;
        quality.parameter_delta_norm = finite_score_delta(before, after);
        quality.information_gain = Some(new_pairs as f64);
        quality.still_identified = after.is_finite();
        quality.warmup = pair_count < 2;
        quality.explanation = format!("OnlineAutoCorr={after:.6e}");
        if !quality.warmup
            && (self.lagged_m2 + self.lagged_m2_compensation == 0.0
                || self.current_m2 + self.current_m2_compensation == 0.0)
        {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .incremental(quality.clone())
                    .message("lag-1 correlation is undefined because a paired marginal is constant")
                    .build(),
            );
        }
        flag_info(&mut ctx, &quality);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                quality,
                "lag-1 correlation",
                "Welford Pearson correlation of every consecutive column-0 pair",
                format!("rho={before:.6e}"),
                format!("rho={after:.6e}"),
            ),
        )
    }
}

impl OnlineAutoCorr {
    fn zero_moments(&self) -> bool {
        self.mean_lagged == 0.0
            && self.mean_current == 0.0
            && self.cross == 0.0
            && self.cross_compensation == 0.0
            && self.lagged_m2 == 0.0
            && self.lagged_m2_compensation == 0.0
            && self.current_m2 == 0.0
            && self.current_m2_compensation == 0.0
    }
}
