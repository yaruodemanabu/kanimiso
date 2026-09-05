//! Linear and canonical generalized linear regression with result annotations.

use crate::annotation::{basic_annotations, context, note, stopped};
use crate::{AnalysisOptions, Annotation, Matrix, Qualified, Result, Session, Topic, Vector};
use signlred::{Issue, IssueCode};
use tsutsumi::linalg::{least_squares_with_diagnostics, ThinSvd};
use tsutsumi::{special, FitCtx};

/// Supported distribution and canonical link, selected at runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Family {
    /// Gaussian observations and identity link.
    #[default]
    Gaussian,
    /// Independent Bernoulli observations in {0, 1} and logit link.
    Binomial,
    /// Non-negative integer counts and log link, with unit exposure.
    Poisson,
}

impl Family {
    /// Inverse canonical link. Extreme Poisson predictors can overflow.
    pub fn mean(self, eta: f64) -> f64 {
        match self {
            Self::Gaussian => eta,
            Self::Binomial if eta >= 0.0 => 1.0 / (1.0 + (-eta).exp()),
            Self::Binomial => {
                let exp = eta.exp();
                exp / (1.0 + exp)
            }
            Self::Poisson => eta.exp(),
        }
    }

    pub(crate) fn variance(self, mu: f64) -> f64 {
        match self {
            Self::Gaussian => 1.0,
            Self::Binomial => mu * (1.0 - mu),
            Self::Poisson => mu,
        }
    }

    pub(crate) fn log_density(self, y: f64, eta: f64) -> f64 {
        match self {
            Self::Gaussian => -0.5 * (y - eta).powi(2),
            Self::Binomial => y * eta - eta.max(0.0) - (-eta.abs()).exp().ln_1p(),
            Self::Poisson => y * eta - eta.exp() - special::ln_gamma(y + 1.0),
        }
    }

    pub(crate) fn valid(self, y: f64) -> bool {
        y.is_finite()
            && match self {
                Self::Gaussian => true,
                Self::Binomial => y == 0.0 || y == 1.0,
                Self::Poisson => y >= 0.0 && y.fract() == 0.0,
            }
    }
}

/// One coefficient and its optional model-based asymptotic or t inference.
#[derive(Clone, Debug)]
pub struct Coefficient {
    /// Original feature name or generated column name.
    pub name: String,
    /// Estimate on the linear-predictor scale.
    pub estimate: f64,
    /// Standard error, absent when inference is not justified.
    pub standard_error: Option<f64>,
    /// Two-sided nominal p-value, unadjusted for multiplicity.
    pub p_value: Option<f64>,
}

/// Measured fitting and sensitivity evidence.
#[derive(Clone, Debug)]
pub struct RegressionDiagnostics {
    /// Number of observations used.
    pub observations: usize,
    /// Numerical design rank.
    pub rank: usize,
    /// Residual degrees of freedom: n-rank, or n-trace(H) for penalized fits.
    pub residual_degrees_of_freedom: f64,
    /// Condition number of the solved (possibly weighted/penalized) design.
    pub condition_number: f64,
    /// Number of completed IRLS updates (one for OLS).
    pub iterations: usize,
    /// Whether the requested convergence criterion was met.
    pub converged: bool,
    /// Squared-error or canonical GLM deviance.
    pub deviance: f64,
    /// Pearson residual sum divided by residual degrees of freedom, when defined.
    pub pearson_dispersion: Option<f64>,
    /// Hat diagonals from the final weighted design.
    pub leverage: Vector,
    /// Cook distances for Gaussian OLS; absent for other objectives.
    pub cooks_distance: Option<Vector>,
    /// Coefficient table in design-column order, including any intercept.
    pub coefficients: Vec<Coefficient>,
}

/// Fitted linear predictor, inference evidence, and detailed interpretation.
#[derive(Clone, Debug)]
pub struct FittedRegression {
    /// Observation family and link.
    pub family: Family,
    /// Design coefficients, including the intercept when requested.
    pub beta: Vector,
    /// Whether `beta[0]` is an added intercept.
    pub fit_intercept: bool,
    /// In-sample fitted response means.
    pub fitted_values: Vector,
    /// Raw response residuals.
    pub residuals: Vector,
    /// Model-based covariance; absent for penalized or failed inference.
    pub covariance: Option<Matrix>,
    /// Numerical and statistical measurements.
    pub diagnostics: RegressionDiagnostics,
    /// Assumptions, limitations, and suggested next checks.
    pub annotations: Vec<Annotation>,
    pub(crate) bounds: Vec<(f64, f64)>,
    pub(crate) options: AnalysisOptions,
}

impl FittedRegression {
    /// Predict response means and record any feature-wise extrapolation.
    pub fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = context(session, &self.options);
        validate_prediction(&mut ctx, x, self.bounds.len());
        if stopped(&ctx) {
            return Err(ctx.finish_failure());
        }
        for (column, &(low, high)) in self.bounds.iter().enumerate() {
            let outside = (0..x.nrows())
                .filter(|&row| x.get(row, column) < low || x.get(row, column) > high)
                .count();
            if outside > 0 {
                note(&mut ctx, IssueCode::PValueUnreliable, format!("{outside} prediction rows lie outside training range [{low}, {high}] for feature {column}; uncertainty from model extrapolation is not included"));
            }
        }
        let eta = Vector::from_iter((0..x.nrows()).map(|row| self.linear_predictor(x, row)));
        if eta.as_slice().iter().any(|v| !v.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("linear predictor overflowed before the inverse link")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let prediction =
            Vector::from_iter(eta.as_slice().iter().map(|&value| self.family.mean(value)));
        if prediction.as_slice().iter().any(|v| !v.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("inverse link overflowed during prediction")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        ctx.finish(prediction)
    }

    pub(crate) fn linear_predictor(&self, x: &Matrix, row: usize) -> f64 {
        let offset = usize::from(self.fit_intercept);
        let mut result = if self.fit_intercept {
            self.beta[0]
        } else {
            0.0
        };
        for column in 0..x.ncols() {
            result += x.get(row, column) * self.beta[column + offset];
        }
        result
    }
}

/// Ordinary least squares for an annotated first analysis.
#[derive(Clone, Debug, Default)]
pub struct LinearModel {
    /// Intercept and quality controls.
    pub options: AnalysisOptions,
}

impl LinearModel {
    /// Fit the independently checked OLS kernel and attach analysis notes.
    pub fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRegression>> {
        let mut ctx = context(session, &self.options);
        validate_training(&mut ctx, x, y, Family::Gaussian, self.options.fit_intercept);
        if stopped(&ctx) {
            return Err(ctx.finish_failure());
        }
        let core = crate::linear::LinearRegression {
            fit_intercept: self.options.fit_intercept,
        }
        .fit_with_policy(x, y, &self.options.policy, &session.child("ols"))?;
        ctx.report.merge(core.report);
        let core = core.value;
        let design = design_matrix(x, self.options.fit_intercept);
        let svd = tsutsumi::linalg::thin_svd(&mut ctx.report, &design, &ctx.policy)
            .ok_or_else(|| failure(&mut ctx, "OLS covariance decomposition failed"))?;
        let covariance = if core.se.as_slice().iter().all(|v| v.is_finite()) {
            Some(covariance_from_svd(&svd, core.sigma2))
        } else {
            None
        };
        let mut annotations = basic_annotations();
        annotations.push(Annotation::new(Topic::Assumptions,
            "OLS t/F inference assumes an identified linear conditional mean and independent, homoscedastic errors; exact finite-sample tests additionally use Gaussian errors.",
            "Inspect residual-versus-fitted plots, leverage, Cook distances, and sampling order; use robust or cluster covariance when the design requires it."));
        annotations.push(Annotation::new(Topic::Inference,
            format!("{} observations, {} residual degrees of freedom; all p-values in the table are nominal and unadjusted.", core.n, core.df_resid),
            "Report coefficient uncertainty together with the specification and selection procedure."));
        annotations.push(sensitivity_note(
            &core.leverage,
            Some(&core.cooks),
            svd.condition_number(),
        ));
        let table = coefficient_table(
            x,
            &core.beta,
            self.options.fit_intercept,
            Some((&core.se, &core.p_values)),
        );
        let model = FittedRegression {
            family: Family::Gaussian,
            beta: core.beta,
            fit_intercept: core.used_intercept,
            fitted_values: core.fitted,
            residuals: core.resid,
            covariance,
            diagnostics: RegressionDiagnostics {
                observations: core.n,
                rank: svd.rank(ctx.policy.rank_tol_relative),
                residual_degrees_of_freedom: core.df_resid,
                condition_number: svd.condition_number(),
                iterations: 1,
                converged: true,
                deviance: core.sigma2 * core.df_resid,
                pearson_dispersion: core.sigma2.is_finite().then_some(core.sigma2),
                leverage: core.leverage,
                cooks_distance: Some(core.cooks),
                coefficients: table,
            },
            annotations,
            bounds: feature_bounds(x),
            options: self.options.clone(),
        };
        note(&mut ctx, IssueCode::CausalClaimUnidentified, "OLS coefficients are associations; causal identification and post-selection validity have not been established");
        ctx.finish(model)
    }
}

/// Canonical Gaussian, Bernoulli-logit, or Poisson-log GLM.
#[derive(Clone, Debug, Default)]
pub struct GeneralizedLinearModel {
    /// Runtime response distribution and canonical link.
    pub family: Family,
    /// Intercept, iterations, and quality controls.
    pub options: AnalysisOptions,
}

impl GeneralizedLinearModel {
    /// Fit by step-halved Newton/IRLS with rank and convergence checks.
    pub fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRegression>> {
        if self.family == Family::Gaussian {
            return LinearModel {
                options: self.options.clone(),
            }
            .fit(x, y, session);
        }
        self.fit_penalized(x, y, 0.0, session)
    }

    pub(crate) fn fit_penalized(
        &self,
        x: &Matrix,
        y: &Vector,
        penalty: f64,
        session: &Session,
    ) -> Result<Qualified<FittedRegression>> {
        let mut ctx = context(session, &self.options);
        validate_training(&mut ctx, x, y, self.family, self.options.fit_intercept);
        if !penalty.is_finite() || penalty < 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("penalty must be finite and non-negative")
                    .build(),
            );
        }
        if stopped(&ctx) {
            return Err(ctx.finish_failure());
        }
        let design = design_matrix(x, self.options.fit_intercept);
        let n = y.len();
        let p = design.ncols();
        let offset = usize::from(self.options.fit_intercept);
        let mut beta = Vector::zeros(p);
        if self.options.fit_intercept {
            let mean = y.mean();
            beta[0] = match self.family {
                Family::Gaussian => mean,
                Family::Binomial => mean.ln() - (-mean).ln_1p(),
                Family::Poisson => mean.ln(),
            };
        }
        let objective = |b: &Vector| {
            let eta = design.matvec(b);
            let loss: f64 = (0..n).map(|i| -self.family.log_density(y[i], eta[i])).sum();
            loss + 0.5 * penalty * b.as_slice()[offset..].iter().map(|v| v * v).sum::<f64>()
        };
        let mut old_loss = objective(&beta);
        let mut iterations = 0;
        let mut converged = false;
        for iteration in 0..self.options.max_iterations {
            let eta = design.matvec(&beta);
            let mu = Vector::from_iter(eta.as_slice().iter().map(|&v| self.family.mean(v)));
            let variance: Vec<f64> = mu
                .as_slice()
                .iter()
                .map(|&v| self.family.variance(v))
                .collect();
            if variance.iter().any(|&v| !v.is_finite() || v <= 0.0) {
                ctx.push(Issue::builder(if self.family == Family::Binomial { IssueCode::PerfectSeparation } else { IssueCode::NonFiniteOutput })
                    .message("IRLS information vanished or overflowed; a finite regular interior MLE is not established").build());
                return Err(ctx.finish_failure());
            }
            let extra = if penalty > 0.0 { p - offset } else { 0 };
            let weighted = Matrix::from_fn(n + extra, p, |i, j| {
                if i < n {
                    design.get(i, j) * variance[i].sqrt()
                } else if j == i - n + offset {
                    penalty.sqrt()
                } else {
                    0.0
                }
            });
            let rhs = Vector::from_iter((0..n + extra).map(|i| {
                if i < n {
                    (y[i] - mu[i]) / variance[i].sqrt()
                } else {
                    -penalty.sqrt() * beta[i - n + offset]
                }
            }));
            let solution =
                least_squares_with_diagnostics(&mut ctx.report, &weighted, &rhs, &ctx.policy)
                    .ok_or_else(|| failure(&mut ctx, "IRLS weighted solve failed"))?;
            if solution.rank < p {
                return Err(failure(
                    &mut ctx,
                    "IRLS design does not identify all coefficients",
                ));
            }
            let mut fraction = 1.0;
            let mut accepted = None;
            for _ in 0..ctx.policy.cf_max_iter {
                let trial = beta.add(&solution.coefficients.scale(fraction));
                let trial_loss = objective(&trial);
                if trial_loss.is_finite() && trial_loss <= old_loss {
                    accepted = Some((trial, trial_loss));
                    break;
                }
                fraction *= 0.5;
            }
            let Some((trial, loss)) = accepted else {
                return Err(failure(
                    &mut ctx,
                    "IRLS line search could not establish descent",
                ));
            };
            let change = trial.sub(&beta).max_abs();
            let relative_loss = (old_loss - loss).abs() / (1.0 + old_loss.abs());
            beta = trial;
            old_loss = loss;
            iterations = iteration + 1;
            if solution.coefficients.max_abs()
                <= ctx.policy.optimizer_parameter_tol * (1.0 + beta.max_abs())
                && relative_loss <= ctx.policy.optimizer_objective_tol
            {
                converged = true;
                break;
            }
            if change == 0.0 {
                return Err(failure(
                    &mut ctx,
                    "IRLS step stalled before the full Newton correction converged",
                ));
            }
        }
        if !converged {
            return Err(failure(
                &mut ctx,
                "IRLS reached the iteration limit; inference is withheld",
            ));
        }
        let eta = design.matvec(&beta);
        let fitted = Vector::from_iter(eta.as_slice().iter().map(|&v| self.family.mean(v)));
        let residuals = y.sub(&fitted);
        let extra = if penalty > 0.0 { p - offset } else { 0 };
        let weighted = Matrix::from_fn(n + extra, p, |i, j| {
            if i < n {
                design.get(i, j) * self.family.variance(fitted[i]).sqrt()
            } else if j == i - n + offset {
                penalty.sqrt()
            } else {
                0.0
            }
        });
        let svd = tsutsumi::linalg::thin_svd(&mut ctx.report, &weighted, &ctx.policy)
            .ok_or_else(|| failure(&mut ctx, "final GLM information decomposition failed"))?;
        let rank = svd.rank(ctx.policy.rank_tol_relative);
        if rank != p || stopped(&ctx) {
            return Err(failure(
                &mut ctx,
                "final GLM information is not identified under the requested policy",
            ));
        }
        let leverage =
            Vector::from_iter((0..n).map(|i| (0..rank).map(|k| svd.u[(i, k)].powi(2)).sum()));
        let df = n as f64 - leverage.as_slice().iter().sum::<f64>();
        let pearson: f64 = (0..n)
            .map(|i| residuals[i].powi(2) / self.family.variance(fitted[i]))
            .sum();
        let dispersion = (df > 0.0).then_some(pearson / df);
        let covariance = if penalty == 0.0 && df > 0.0 {
            Some(covariance_from_svd(
                &svd,
                if self.family == Family::Gaussian {
                    dispersion.unwrap()
                } else {
                    1.0
                },
            ))
        } else {
            None
        };
        if covariance
            .as_ref()
            .is_some_and(|cov| !tsutsumi::linalg::matrix_is_finite(cov))
        {
            return Err(failure(
                &mut ctx,
                "GLM covariance is not representable; inference is withheld",
            ));
        }
        let mut se = Vector::filled(p, f64::NAN);
        let mut pv = se.clone();
        if let Some(cov) = &covariance {
            for j in 0..p {
                se[j] = cov.get(j, j).sqrt();
                pv[j] = special::norm_pvalue_two_sided(beta[j] / se[j]);
            }
        }
        let deviance: f64 = (0..n)
            .map(|i| match self.family {
                Family::Gaussian => residuals[i].powi(2),
                Family::Binomial => -2.0 * self.family.log_density(y[i], eta[i]),
                Family::Poisson => {
                    2.0 * (if y[i] == 0.0 {
                        fitted[i]
                    } else {
                        y[i] * (y[i].ln() - eta[i]) - y[i] + fitted[i]
                    })
                }
            })
            .sum();
        if !deviance.is_finite()
            || !pearson.is_finite()
            || beta.as_slice().iter().any(|v| !v.is_finite())
        {
            return Err(failure(&mut ctx, "GLM diagnostics are not representable"));
        }
        let mut annotations = basic_annotations();
        annotations.push(Annotation::new(Topic::Assumptions, format!("{:?} canonical GLM; observations are independent, the link is correctly specified, and Poisson exposure is one. Dispersion estimate: {dispersion:?}.",self.family),
            "Check sampling dependence, class balance/count support, exposure, and overdispersion; use an appropriate mixed model if rows are clustered."));
        annotations.push(Annotation::new(Topic::Inference,
            if penalty > 0.0 { "Penalized fitting: ordinary GLM coefficient tests are withheld." } else { "Standard errors and p-values use local model-based information and asymptotic normality; they can be unreliable in small samples or near separation." },
            "Inspect convergence, design rank, leverage, sensitivity to specification, and a suitable resampling analysis."));
        annotations.push(Annotation::new(Topic::Computation, format!("IRLS converged after {iterations} updates; objective={old_loss}, numerical rank={rank}/{p}, penalty={penalty}."),
            "Recheck the result under tighter numerical tolerances when coefficient interpretation is consequential."));
        annotations.push(sensitivity_note(&leverage, None, svd.condition_number()));
        note(&mut ctx, IssueCode::PValueUnreliable, "GLM inference is model-based and asymptotic; annotation fields state sampling and distribution assumptions");
        let coefficients =
            coefficient_table(x, &beta, self.options.fit_intercept, Some((&se, &pv)));
        ctx.finish(FittedRegression {
            family: self.family,
            beta,
            fit_intercept: self.options.fit_intercept,
            fitted_values: fitted,
            residuals,
            covariance,
            diagnostics: RegressionDiagnostics {
                observations: n,
                rank,
                residual_degrees_of_freedom: df,
                condition_number: svd.condition_number(),
                iterations,
                converged,
                deviance,
                pearson_dispersion: dispersion,
                leverage,
                cooks_distance: None,
                coefficients,
            },
            annotations,
            bounds: feature_bounds(x),
            options: self.options.clone(),
        })
    }
}

pub(crate) fn failure(ctx: &mut FitCtx, message: &str) -> signlred::Failure {
    ctx.push(
        Issue::builder(IssueCode::DidNotConverge)
            .severity(signlred::Severity::Error)
            .message(message)
            .build(),
    );
    let failure = signlred::Failure {
        primary: ctx.report.issues().last().unwrap().clone(),
        report: ctx.report.clone(),
    };
    ctx.session.finish_err(&failure);
    failure
}

fn sensitivity_note(leverage: &Vector, cooks: Option<&Vector>, condition: f64) -> Annotation {
    let largest = |values: &Vector| {
        values
            .as_slice()
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(row, value)| (row, *value))
    };
    Annotation::new(Topic::Sensitivity,
        format!("Largest leverage (zero-based row, value)={:?}; largest Cook distance={:?}; solved-design condition number={condition}. These diagnostics are not automatic outlier-removal rules.", largest(leverage), cooks.and_then(largest)),
        "Inspect the corresponding observations, covariate scaling and residuals; compare a principled sensitivity fit without selecting removals by the desired conclusion.")
}

pub(crate) fn validate_training(
    ctx: &mut FitCtx,
    x: &Matrix,
    y: &Vector,
    family: Family,
    intercept: bool,
) {
    tsutsumi::validate::inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    if y.as_slice().iter().any(|&v| !family.valid(v)) {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message(format!(
                    "target is outside the {:?} response domain",
                    family
                ))
                .build(),
        );
    }
    tsutsumi::validate::inspect_identification(
        &mut ctx.report,
        x.nrows(),
        x.ncols() + usize::from(intercept),
        &ctx.policy,
    );
    tsutsumi::validate::inspect_collinearity(&mut ctx.report, x, &ctx.policy);
}

pub(crate) fn validate_prediction(ctx: &mut FitCtx, x: &Matrix, columns: usize) {
    if x.ncols() != columns {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("prediction feature count differs from training")
                .build(),
        );
    }
    if !tsutsumi::linalg::matrix_is_finite(x) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("prediction features contain NaN or infinity")
                .build(),
        );
    }
}

pub(crate) fn design_matrix(x: &Matrix, intercept: bool) -> Matrix {
    if intercept {
        x.with_intercept()
    } else {
        x.clone()
    }
}
pub(crate) fn feature_bounds(x: &Matrix) -> Vec<(f64, f64)> {
    (0..x.ncols())
        .map(|j| {
            (0..x.nrows()).fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), i| {
                (low.min(x.get(i, j)), high.max(x.get(i, j)))
            })
        })
        .collect()
}
fn covariance_from_svd(svd: &ThinSvd, scale: f64) -> Matrix {
    let p = svd.v.nrows();
    Matrix::from_fn(p, p, |i, j| {
        (0..p)
            .map(|k| {
                (svd.v[(i, k)] / svd.singular_values[k])
                    * (svd.v[(j, k)] / svd.singular_values[k])
                    * scale
            })
            .sum()
    })
}
fn coefficient_table(
    x: &Matrix,
    beta: &Vector,
    intercept: bool,
    inference: Option<(&Vector, &Vector)>,
) -> Vec<Coefficient> {
    (0..beta.len())
        .map(|j| {
            let name = if intercept && j == 0 {
                "intercept".into()
            } else {
                let column = j - usize::from(intercept);
                x.col_names
                    .as_ref()
                    .and_then(|names| names.get(column))
                    .cloned()
                    .unwrap_or_else(|| format!("x{column}"))
            };
            Coefficient {
                name,
                estimate: beta[j],
                standard_error: inference.and_then(|(se, _)| se[j].is_finite().then_some(se[j])),
                p_value: inference.and_then(|(_, pv)| pv[j].is_finite().then_some(pv[j])),
            }
        })
        .collect()
}
