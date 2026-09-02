use super::common::{finish_explain, inspect_online_xy, reject_explain};
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::{PartialFit, Predict};
use faer::Mat;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{IncrementalQuality, Issue, IssueCode, Qualified, Result};

/// Online recursive least squares with a forgetting factor `λ`.
///
/// Sufficient statistics are the coefficient vector `θ` and inverse Gram
/// matrix `P`. Validation and numerical failures are transactional: no part
/// of the estimator state is committed.
#[derive(Clone, Debug)]
pub struct LinearRegression {
    /// Forgetting factor `λ ∈ (0, 1]`. `λ = 1` is ordinary growing-window RLS.
    pub forgetting_factor: f64,
    /// Prepend a column of ones.
    pub fit_intercept: bool,
    /// Initial `P = p0 I`.
    pub p0: f64,
    pub(super) theta: Vector,
    pub(super) p_mat: Option<Mat<f64>>,
    pub(super) n_features: usize,
    pub(super) initialized_fit_intercept: bool,
    pub(super) effective_sample_size: f64,
    pub(super) n_seen: u64,
    pub(super) updates: u64,
    pub(super) initialized: bool,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            forgetting_factor: 1.0,
            fit_intercept: true,
            p0: 1_000.0,
            theta: Vector::zeros(0),
            p_mat: None,
            n_features: 0,
            initialized_fit_intercept: true,
            effective_sample_size: 0.0,
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl LinearRegression {
    /// Construct RLS with forgetting factor `λ`.
    pub fn new(forgetting_factor: f64) -> Self {
        Self {
            forgetting_factor,
            ..Self::default()
        }
    }

    /// Slope coefficients, excluding the intercept.
    pub fn coef(&self) -> Vector {
        if self.fit_intercept && !self.theta.is_empty() {
            Vector::from_iter((1..self.theta.len()).map(|index| self.theta[index]))
        } else {
            self.theta.clone()
        }
    }

    /// Intercept, or zero when `fit_intercept` is false.
    pub fn intercept(&self) -> f64 {
        if self.fit_intercept && !self.theta.is_empty() {
            self.theta[0]
        } else {
            0.0
        }
    }

    /// Inverse Gram matrix `P`, proportional to the coefficient covariance.
    pub fn p_matrix(&self) -> Option<&Mat<f64>> {
        self.p_mat.as_ref()
    }

    fn dim(&self, features: usize) -> usize {
        features + usize::from(self.fit_intercept)
    }

    fn row_vec(&self, x: &Matrix, row: usize) -> Vector {
        let mut values = Vector::zeros(self.dim(x.ncols()));
        if self.fit_intercept {
            values[0] = 1.0;
            for column in 0..x.ncols() {
                values[column + 1] = x.get(row, column);
            }
        } else {
            for column in 0..x.ncols() {
                values[column] = x.get(row, column);
            }
        }
        values
    }

    fn predict_row(&self, x: &Matrix, row: usize) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        self.row_vec(x, row).dot(&self.theta)
    }
}

impl PartialFit for LinearRegression {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(
                Issue::builder(IssueCode::MissingTarget)
                    .message("online LinearRegression.partial_fit requires y")
                    .build(),
            );
            return fail_update(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "missing y"),
            );
        };

        let mut invalid = false;
        if x.nrows() == 0 || x.ncols() == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message(format!(
                        "online RLS requires a non-empty design; got {}×{}",
                        x.nrows(),
                        x.ncols()
                    ))
                    .build(),
            );
            if !self.initialized {
                ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            }
            invalid = true;
        }
        inspect_online_xy(&mut ctx, x, Some(y));
        if y.len() != x.nrows() {
            invalid = true;
        }
        if (0..x.nrows()).any(|row| (0..x.ncols()).any(|column| !x.get(row, column).is_finite()))
            || y.as_slice().iter().any(|value| !value.is_finite())
        {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteInput)
                    .message("online RLS inputs must contain only finite values")
                    .build(),
            );
            invalid = true;
        }
        if !(self.forgetting_factor > 0.0 && self.forgetting_factor <= 1.0) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!(
                        "forgetting λ={} is not in (0, 1]",
                        self.forgetting_factor
                    ))
                    .build(),
            );
            invalid = true;
        }
        if !self.p0.is_finite() || self.p0 <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message(format!(
                        "RLS initial covariance scale p0={} must be finite and positive",
                        self.p0
                    ))
                    .build(),
            );
            invalid = true;
        }

        let parameter_count = self.dim(x.ncols());
        if self.initialized
            && (x.ncols() != self.n_features
                || self.fit_intercept != self.initialized_fit_intercept
                || parameter_count != self.theta.len())
        {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message(format!(
                        "RLS was initialized with {} features and fit_intercept={}; got {} features and fit_intercept={} ({} versus {} design columns)",
                        self.n_features,
                        self.initialized_fit_intercept,
                        x.ncols(),
                        self.fit_intercept,
                        self.theta.len(),
                        parameter_count
                    ))
                    .build(),
            );
            invalid = true;
        }
        if invalid {
            return fail_update(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "invalid RLS update"),
            );
        }

        let mut candidate_theta = if self.initialized {
            self.theta.clone()
        } else {
            Vector::zeros(parameter_count)
        };
        let mut candidate_p = if self.initialized {
            let Some(matrix) = self.p_mat.as_ref() else {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message("initialized online RLS has no inverse Gram matrix")
                        .build(),
                );
                return fail_update(
                    ctx,
                    reject_explain(
                        self.updates,
                        x.nrows(),
                        self.n_seen,
                        "invalid stored RLS state",
                    ),
                );
            };
            if matrix.nrows() != parameter_count || matrix.ncols() != parameter_count {
                ctx.push(
                    Issue::builder(IssueCode::DimensionMismatch)
                        .message(format!(
                            "stored RLS inverse Gram is {}×{}, expected {parameter_count}×{parameter_count}",
                            matrix.nrows(),
                            matrix.ncols()
                        ))
                        .build(),
                );
                return fail_update(
                    ctx,
                    reject_explain(
                        self.updates,
                        x.nrows(),
                        self.n_seen,
                        "invalid stored RLS state",
                    ),
                );
            }
            matrix.clone()
        } else {
            Mat::<f64>::from_fn(parameter_count, parameter_count, |row, column| {
                if row == column {
                    self.p0
                } else {
                    0.0
                }
            })
        };

        let stored_state_is_finite = self.effective_sample_size.is_finite()
            && self.effective_sample_size >= 0.0
            && candidate_theta
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
            && (0..parameter_count).all(|row| {
                candidate_p[(row, row)] > 0.0
                    && (0..parameter_count).all(|column| {
                        candidate_p[(row, column)].is_finite()
                            && candidate_p[(row, column)] == candidate_p[(column, row)]
                    })
            });
        if !stored_state_is_finite {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("stored RLS state is non-finite, non-positive, or asymmetric")
                    .build(),
            );
            return fail_update(
                ctx,
                reject_explain(
                    self.updates,
                    x.nrows(),
                    self.n_seen,
                    "invalid stored RLS state",
                ),
            );
        }

        let before = candidate_theta.clone();
        let Some(loss_before) = mse_with_theta(x, y, &candidate_theta, self.fit_intercept) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("RLS pre-update loss is non-finite")
                    .build(),
            );
            return fail_update(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "non-finite RLS loss"),
            );
        };

        let mut information = 0.0;
        let mut candidate_effective_sample_size = self.effective_sample_size;
        let lambda = self.forgetting_factor;
        for row in 0..x.nrows() {
            let design = self.row_vec(x, row);
            let mut p_times_design = Vector::zeros(design.len());
            for output_row in 0..design.len() {
                let mut sum = 0.0;
                for column in 0..design.len() {
                    sum += candidate_p[(output_row, column)] * design[column];
                }
                p_times_design[output_row] = sum;
            }
            let denominator = lambda + design.dot(&p_times_design);
            if !denominator.is_finite() || denominator <= 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NearSingular)
                        .message(format!(
                            "RLS gain denominator at row {row} is not finite and positive ({denominator})"
                        ))
                        .build(),
                );
                return fail_update(
                    ctx,
                    reject_explain(
                        self.updates,
                        x.nrows(),
                        self.n_seen,
                        "RLS gain computation failed",
                    ),
                );
            }

            let prediction = design.dot(&candidate_theta);
            let residual = y[row] - prediction;
            information += residual * residual;
            candidate_effective_sample_size = lambda * candidate_effective_sample_size + 1.0;
            for index in 0..candidate_theta.len() {
                candidate_theta[index] += (p_times_design[index] / denominator) * residual;
            }

            let mut next_p = Mat::<f64>::zeros(design.len(), design.len());
            for output_row in 0..design.len() {
                for column in output_row..design.len() {
                    let value = (candidate_p[(output_row, column)]
                        - p_times_design[output_row] * p_times_design[column] / denominator)
                        / lambda;
                    next_p[(output_row, column)] = value;
                    next_p[(column, output_row)] = value;
                }
            }
            let finite_update = information.is_finite()
                && candidate_effective_sample_size.is_finite()
                && candidate_theta
                    .as_slice()
                    .iter()
                    .all(|value| value.is_finite())
                && (0..design.len()).all(|output_row| {
                    next_p[(output_row, output_row)] > 0.0
                        && (output_row..design.len())
                            .all(|column| next_p[(output_row, column)].is_finite())
                });
            if !finite_update {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message(format!(
                            "RLS update at row {row} produced a non-finite or non-positive state"
                        ))
                        .build(),
                );
                return fail_update(
                    ctx,
                    reject_explain(
                        self.updates,
                        x.nrows(),
                        self.n_seen,
                        "RLS numerical update failed",
                    ),
                );
            }
            candidate_p = next_p;
        }

        let Some(loss_after) = mse_with_theta(x, y, &candidate_theta, self.fit_intercept) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("RLS post-update loss is non-finite")
                    .build(),
            );
            return fail_update(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "non-finite RLS loss"),
            );
        };
        let Ok(batch_rows) = u64::try_from(x.nrows()) else {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("RLS batch row count cannot be represented by its counter")
                    .build(),
            );
            return fail_update(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "RLS counter overflow"),
            );
        };
        let Some(new_n_seen) = self.n_seen.checked_add(batch_rows) else {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("RLS observation counter overflowed")
                    .build(),
            );
            return fail_update(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "RLS counter overflow"),
            );
        };
        let Some(new_updates) = self.updates.checked_add(1) else {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("RLS update counter overflowed")
                    .build(),
            );
            return fail_update(
                ctx,
                reject_explain(self.updates, x.nrows(), self.n_seen, "RLS counter overflow"),
            );
        };

        let delta = candidate_theta.sub(&before);
        self.theta = candidate_theta;
        self.p_mat = Some(candidate_p);
        self.n_features = x.ncols();
        self.initialized_fit_intercept = self.fit_intercept;
        self.effective_sample_size = candidate_effective_sample_size;
        self.n_seen = new_n_seen;
        self.updates = new_updates;
        self.initialized = true;

        let effective_sample_size = self.effective_sample_size;
        let parameter_count = self.theta.len();
        let mut quality = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        quality.effective_sample_size = effective_sample_size;
        quality.forgetting_factor = Some(self.forgetting_factor);
        quality.parameter_delta_norm = Some(delta.norm());
        quality.parameter_delta_max = Some(delta.max_abs());
        quality.loss_before = Some(loss_before);
        quality.loss_after = Some(loss_after);
        quality.information_gain = Some(
            (loss_before - loss_after)
                .abs()
                .max(information / x.nrows().max(1) as f64),
        );
        quality.still_identified =
            effective_sample_size > parameter_count as f64 && self.n_seen > parameter_count as u64;
        quality.warmup = self.n_seen < 5 || self.n_seen <= parameter_count as u64;
        quality.explanation = format!(
            "RLS λ={:.4}: {} rows, n_eff={effective_sample_size:.3}, ||Δθ||={:.4e}, mse {:.6e} → {:.6e}",
            self.forgetting_factor,
            x.nrows(),
            delta.norm(),
            loss_before,
            loss_after
        );
        quality.top_moved_parameters = top_moved(&delta, 3);

        if quality.is_uninformative(ctx.policy.uninformative_info_eps)
            || (loss_before - loss_after).abs() <= ctx.policy.uninformative_info_eps
                && delta.norm() <= ctx.policy.uninformative_info_eps
        {
            ctx.push(
                Issue::builder(IssueCode::UpdateWithZeroInformation)
                    .incremental(quality.clone())
                    .message("this RLS batch did not change the residual or θ")
                    .build(),
            );
        }
        if effective_sample_size < ctx.policy.min_effective_sample
            || effective_sample_size <= parameter_count as f64
        {
            ctx.push(
                Issue::builder(IssueCode::ForgettingErasedIdentification)
                    .incremental(quality.clone())
                    .message(format!(
                        "n_eff={effective_sample_size:.3} (asymptotic 1/(1−λ)={:.3}) is too small to identify {parameter_count} parameters",
                        if self.forgetting_factor == 1.0 {
                            f64::INFINITY
                        } else {
                            1.0 / (1.0 - self.forgetting_factor)
                        }
                    ))
                    .metric("n_eff", effective_sample_size)
                    .build(),
            );
        }
        if quality.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(quality.clone())
                    .message("RLS is still in warmup (n_seen ≤ p or n_seen < 5)")
                    .build(),
            );
        }
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                quality,
                format!("θ[{parameter_count}] and inverse Gram P"),
                "recursive least squares / Kalman gain on the new rows",
                format!("mse={loss_before:.6e}"),
                format!("mse={loss_after:.6e}"),
            ),
        )
    }
}

impl Predict for LinearRegression {
    type Output = Vector;

    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return Err(ctx.finish_failure());
        }
        if self.fit_intercept != self.initialized_fit_intercept {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message(format!(
                        "online RLS was initialized with fit_intercept={}, but it is now {}",
                        self.initialized_fit_intercept, self.fit_intercept
                    ))
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if x.ncols() != self.n_features {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!(
                        "online RLS predict expected {} columns, got {}",
                        self.n_features,
                        x.ncols()
                    ))
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if (0..x.nrows()).any(|row| (0..x.ncols()).any(|column| !x.get(row, column).is_finite())) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteInput)
                    .message("online RLS prediction input must be finite")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let output = Vector::from_iter((0..x.nrows()).map(|row| self.predict_row(x, row)));
        if output.as_slice().iter().any(|value| !value.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("online RLS prediction produced a non-finite value")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        ctx.finish(output)
    }
}

fn mse_with_theta(x: &Matrix, y: &Vector, theta: &Vector, fit_intercept: bool) -> Option<f64> {
    let mut squared_error = 0.0;
    for row in 0..x.nrows() {
        let mut prediction = if fit_intercept { theta[0] } else { 0.0 };
        let offset = usize::from(fit_intercept);
        for column in 0..x.ncols() {
            prediction += theta[column + offset] * x.get(row, column);
        }
        let residual = prediction - y[row];
        squared_error += residual * residual;
    }
    let mse = squared_error / x.nrows() as f64;
    mse.is_finite().then_some(mse)
}

fn fail_update(
    ctx: FitCtx,
    explanation: IncrementalExplain,
) -> Result<Qualified<IncrementalExplain>> {
    ctx.session.record_incremental(explanation);
    Err(ctx.finish_failure())
}

fn top_moved(delta: &Vector, count: usize) -> Vec<(String, f64)> {
    let mut indices: Vec<usize> = (0..delta.len()).collect();
    indices.sort_by(|left, right| {
        delta[*right]
            .abs()
            .partial_cmp(&delta[*left].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
        .into_iter()
        .take(count)
        .map(|index| (format!("theta[{index}]"), delta[index]))
        .collect()
}
