use super::common::{
    checked_compensated_add, checked_online_covariance_step, checked_sample_variance_from_m2,
    finish_explain, finite_score_delta, flag_info, online_moment_preflight, reject_metric_batch,
};
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::{PartialFit, Transform};
use ojizou_san::{IncrementalExplain, Session};
use signlred::{IncrementalQuality, Issue, IssueCode, Qualified, Result};

/// Streaming sample-variance threshold (river `feature_selection.VarianceThreshold`).
///
/// Every accepted row updates every column. A non-finite cell rejects the
/// complete batch. Later batches and transforms must have the initialized
/// column count. A column is retained exactly when its sample variance is
/// strictly greater than `threshold`.
#[derive(Clone, Debug)]
pub struct OnlineVarianceThreshold {
    /// Finite non-negative sample-variance cutoff.
    pub threshold: f64,
    pub(super) n_seen: u64,
    pub(super) mean: Vec<f64>,
    pub(super) m2: Vec<f64>,
    pub(super) m2_compensation: Vec<f64>,
    pub(super) n_features: usize,
    pub(super) updates: u64,
}

impl Default for OnlineVarianceThreshold {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            n_seen: 0,
            mean: Vec::new(),
            m2: Vec::new(),
            m2_compensation: Vec::new(),
            n_features: 0,
            updates: 0,
        }
    }
}

impl OnlineVarianceThreshold {
    /// Construct a selector with sample-variance cutoff `threshold`.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Self::default()
        }
    }
}

impl PartialFit for OnlineVarianceThreshold {
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
            "OnlineVarianceThreshold",
            self.updates,
            self.n_seen,
        ) {
            Ok(validated) => validated,
            Err(failure) => return Err(failure),
        };
        if !self.threshold.is_finite() || self.threshold < 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message(format!(
                        "OnlineVarianceThreshold.threshold must be finite and non-negative; got {}",
                        self.threshold
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
        if self.n_features != 0 && x.ncols() != self.n_features {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message(format!(
                        "OnlineVarianceThreshold saw {} columns after init with {}",
                        x.ncols(),
                        self.n_features
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
        if let Some((row, column)) = (0..x.nrows()).find_map(|row| {
            (0..x.ncols())
                .find(|&column| !x.get(row, column).is_finite())
                .map(|column| (row, column))
        }) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteInput)
                    .message(format!(
                        "OnlineVarianceThreshold input at row {row}, column {column} is non-finite"
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

        let stored_state_is_valid = if self.n_features == 0 {
            self.n_seen == 0
                && self.mean.is_empty()
                && self.m2.is_empty()
                && self.m2_compensation.is_empty()
                && self.updates == 0
        } else {
            self.n_seen > 0
                && self.mean.len() == self.n_features
                && self.m2.len() == self.n_features
                && self.m2_compensation.len() == self.n_features
                && self.mean.iter().all(|value| value.is_finite())
                && (0..self.n_features).all(|column| {
                    checked_sample_variance_from_m2(
                        self.n_seen,
                        self.m2[column],
                        self.m2_compensation[column],
                    )
                    .is_ok()
                        && (self.n_seen != 1
                            || self.m2[column] + self.m2_compensation[column] == 0.0)
                })
        };
        if !stored_state_is_valid {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("stored OnlineVarianceThreshold state is invalid")
                    .build(),
            );
            return Err(reject_metric_batch(
                ctx,
                self.updates,
                x.nrows(),
                self.n_seen,
            ));
        }

        let before = if self.n_seen < 2 {
            None
        } else {
            let mut maximum = 0.0_f64;
            for column in 0..self.n_features {
                let variance = checked_sample_variance_from_m2(
                    self.n_seen,
                    self.m2[column],
                    self.m2_compensation[column],
                )
                .expect("stored variance state was validated")
                .expect("validated count");
                maximum = maximum.max(variance);
            }
            Some(maximum)
        };

        let candidate_n_features = if self.n_features == 0 {
            x.ncols()
        } else {
            self.n_features
        };
        let mut candidate_mean = if self.n_features == 0 {
            vec![0.0; candidate_n_features]
        } else {
            self.mean.clone()
        };
        let mut candidate_m2 = if self.n_features == 0 {
            vec![0.0; candidate_n_features]
        } else {
            self.m2.clone()
        };
        let mut candidate_compensation = if self.n_features == 0 {
            vec![0.0; candidate_n_features]
        } else {
            self.m2_compensation.clone()
        };

        for column in 0..x.ncols() {
            let mut candidate_count = self.n_seen;
            for row in 0..x.nrows() {
                let value = x.get(row, column);
                let (next_mean, _, increment) = match checked_online_covariance_step(
                    candidate_count,
                    candidate_mean[column],
                    candidate_mean[column],
                    value,
                    value,
                ) {
                    Ok(step) => step,
                    Err(code) => {
                        ctx.push(
                            Issue::builder(code)
                                .message(format!(
                                    "OnlineVarianceThreshold variance is not representable at row {row}, column {column}"
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
                };
                if increment < 0.0 {
                    ctx.push(
                        Issue::builder(IssueCode::NonFiniteOutput)
                            .message(format!(
                                "OnlineVarianceThreshold produced a negative variance increment at row {row}, column {column}"
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
                let (next_m2, next_compensation) = match checked_compensated_add(
                    candidate_m2[column],
                    candidate_compensation[column],
                    increment,
                ) {
                    Ok(next) => next,
                    Err(code) => {
                        ctx.push(
                            Issue::builder(code)
                                .message(format!(
                                    "OnlineVarianceThreshold could not accumulate variance at row {row}, column {column}"
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
                };
                candidate_mean[column] = next_mean;
                candidate_m2[column] = next_m2;
                candidate_compensation[column] = next_compensation;
                candidate_count = match candidate_count.checked_add(1) {
                    Some(next) => next,
                    None => {
                        ctx.push(
                            Issue::builder(IssueCode::InvalidParameter)
                                .message("OnlineVarianceThreshold observation counter overflowed")
                                .build(),
                        );
                        return Err(reject_metric_batch(
                            ctx,
                            self.updates,
                            x.nrows(),
                            self.n_seen,
                        ));
                    }
                };
            }
            debug_assert_eq!(candidate_count, new_n);
        }

        let mut after = None;
        let mut kept = 0;
        if new_n >= 2 {
            let mut maximum = 0.0_f64;
            for column in 0..candidate_n_features {
                let variance = match checked_sample_variance_from_m2(
                    new_n,
                    candidate_m2[column],
                    candidate_compensation[column],
                ) {
                    Ok(Some(variance)) => variance,
                    Ok(None) => unreachable!("new_n was checked"),
                    Err(code) => {
                        ctx.push(
                            Issue::builder(code)
                                .message(format!(
                                    "OnlineVarianceThreshold sample variance is not representable for column {column}"
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
                };
                maximum = maximum.max(variance);
                if variance > self.threshold {
                    kept += 1;
                }
            }
            after = Some(maximum);
        }

        self.n_seen = new_n;
        self.mean = candidate_mean;
        self.m2 = candidate_m2;
        self.m2_compensation = candidate_compensation;
        self.n_features = candidate_n_features;
        self.updates = new_updates;
        let mut quality = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        quality.effective_sample_size = self.n_seen as f64;
        quality.parameter_delta_norm = match (before, after) {
            (Some(before), Some(after)) => finite_score_delta(before, after),
            _ => None,
        };
        quality.information_gain = Some(x.nrows() as f64);
        quality.still_identified = self.n_seen >= 2 && kept >= 1;
        quality.warmup = self.n_seen < 2;
        quality.explanation = format!("OnlineVarianceThreshold kept={kept}");
        if !quality.warmup && kept == 0 {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .incremental(quality.clone())
                    .message(
                        "no column has sample variance strictly above the configured threshold",
                    )
                    .build(),
            );
        }
        flag_info(&mut ctx, &quality);
        let before_text = before
            .map(|value| format!("maxvar={value:.6e}"))
            .unwrap_or_else(|| "maxvar=undefined".to_string());
        let after_text = after
            .map(|value| format!("maxvar={value:.6e} kept={kept}"))
            .unwrap_or_else(|| format!("maxvar=undefined kept={kept}"));
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                quality,
                "column variance update",
                "shared Welford sample variance for every input column",
                before_text,
                after_text,
            ),
        )
    }
}

impl Transform for OnlineVarianceThreshold {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        ctx.report.set_sample_shape(x.nrows(), x.ncols());
        let mut invalid = false;
        if !self.threshold.is_finite() || self.threshold < 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message(format!(
                        "OnlineVarianceThreshold.threshold must be finite and non-negative; got {}",
                        self.threshold
                    ))
                    .build(),
            );
            invalid = true;
        }
        if x.nrows() == 0 || x.ncols() == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message(format!(
                        "OnlineVarianceThreshold transform requires a non-empty matrix; got {}×{}",
                        x.nrows(),
                        x.ncols()
                    ))
                    .build(),
            );
            invalid = true;
        }
        if self.n_features == 0 {
            ctx.push(
                Issue::builder(IssueCode::PartialFitBeforeInit)
                    .message("OnlineVarianceThreshold must observe a batch before transform")
                    .build(),
            );
            invalid = true;
        } else if x.ncols() != self.n_features {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message(format!(
                        "OnlineVarianceThreshold transform received {} columns after init with {}",
                        x.ncols(),
                        self.n_features
                    ))
                    .build(),
            );
            invalid = true;
        }
        if !invalid {
            if let Some((row, column)) = (0..x.nrows()).find_map(|row| {
                (0..x.ncols())
                    .find(|&column| !x.get(row, column).is_finite())
                    .map(|column| (row, column))
            }) {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteInput)
                        .message(format!(
                            "OnlineVarianceThreshold transform input at row {row}, column {column} is non-finite"
                        ))
                        .build(),
                );
                invalid = true;
            }
        }
        if !invalid
            && (self.n_seen == 0
                || self.mean.len() != self.n_features
                || self.m2.len() != self.n_features
                || self.m2_compensation.len() != self.n_features
                || self.mean.iter().any(|value| !value.is_finite()))
        {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("stored OnlineVarianceThreshold state is invalid")
                    .build(),
            );
            invalid = true;
        }
        if invalid {
            return Err(ctx.finish_failure());
        }
        if self.n_seen < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("sample variance requires at least two accepted rows")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }

        let mut keep = Vec::new();
        for column in 0..self.n_features {
            let variance = match checked_sample_variance_from_m2(
                self.n_seen,
                self.m2[column],
                self.m2_compensation[column],
            ) {
                Ok(Some(variance)) => variance,
                Ok(None) => unreachable!("n_seen was checked"),
                Err(code) => {
                    ctx.push(
                        Issue::builder(code)
                            .message(format!(
                                "stored sample variance for column {column} is not representable"
                            ))
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                }
            };
            if variance > self.threshold {
                keep.push(column);
            }
        }
        if keep.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("OnlineVarianceThreshold kept no columns")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        ctx.finish(Matrix::from_fn(x.nrows(), keep.len(), |row, column| {
            x.get(row, keep[column])
        }))
    }
}
