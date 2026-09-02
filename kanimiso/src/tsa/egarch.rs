//! Fixed-order asymmetric EGARCH Gaussian QMLE.

use super::common::{
    gaussian_qml_profile, inspect_scale_invariant_univariate, normalized_log_squares,
    select_ranked_objective_candidate, NormalizedLogSquares,
};
use crate::context::FitCtx;
use crate::data::Vector;
use crate::optimize::{decode_open_interval, encode_open_interval, NelderMead};
use crate::traits::FitSeries;
use ojizou_san::Session;
use signlred::{
    insufficient_sample, Issue, IssueCode, NumericalCompromise, Qualified, Result, Severity,
};

pub(super) const EGARCH_CENTERED_ABSOLUTE_NORMAL: f64 = 0.797_884_560_802_865_4;
const EGARCH_INTERIOR_PARAMETER_COUNT: usize = 5;
const EGARCH_DEFAULT_SIMPLEX_STEP: f64 = 0.25;
const EGARCH_BETA_LOWER: f64 = 0.0;
const EGARCH_BETA_UPPER: f64 = 1.0;
const EGARCH_BETA_SEEDS: [f64; 6] = [0.03, 0.30, 0.60, 0.82, 0.94, 0.98];
const EGARCH_ALPHA_SEEDS: [f64; 5] = [-0.30, 0.0, 0.12, 0.30, 0.60];
const EGARCH_GAMMA_SEEDS: [f64; 5] = [-0.40, -0.15, 0.0, 0.15, 0.40];
const EGARCH_LONG_RUN_OFFSETS: [f64; 5] = [-4.0, -2.0, 0.0, 2.0, 4.0];

#[derive(Clone, Copy, Debug)]
pub(super) struct EgarchQmlParameters {
    pub(super) omega_normalized: f64,
    pub(super) alpha: f64,
    pub(super) gamma: f64,
    pub(super) beta: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EgarchQmlFace {
    Interior,
    BetaZero,
}

impl EgarchQmlFace {
    const ALL: [Self; 2] = [Self::Interior, Self::BetaZero];

    fn index(self) -> usize {
        match self {
            Self::Interior => 0,
            Self::BetaZero => 1,
        }
    }

    fn dimension(self) -> usize {
        match self {
            Self::Interior => 4,
            Self::BetaZero => 3,
        }
    }

    fn selection_rank(self) -> usize {
        match self {
            Self::BetaZero => 0,
            Self::Interior => 1,
        }
    }

    fn session_name(self) -> &'static str {
        match self {
            Self::Interior => "qml-interior",
            Self::BetaZero => "qml-beta-zero",
        }
    }
}

struct EgarchQmlCandidate {
    face: EgarchQmlFace,
    point: Vector,
    objective: f64,
}

fn decode_egarch_qml_point(face: EgarchQmlFace, point: &[f64]) -> Option<EgarchQmlParameters> {
    let (long_run_log_variance_normalized, alpha, gamma, beta) = match (face, point) {
        (EgarchQmlFace::Interior, [long_run, alpha, gamma, beta_coordinate]) => (
            *long_run,
            *alpha,
            *gamma,
            decode_open_interval(*beta_coordinate, EGARCH_BETA_LOWER, EGARCH_BETA_UPPER)?,
        ),
        (EgarchQmlFace::BetaZero, [long_run, alpha, gamma]) => {
            (*long_run, *alpha, *gamma, EGARCH_BETA_LOWER)
        }
        _ => return None,
    };
    let omega_normalized = (1.0 - beta) * long_run_log_variance_normalized;
    [
        long_run_log_variance_normalized,
        omega_normalized,
        alpha,
        gamma,
        beta,
    ]
    .iter()
    .all(|value| value.is_finite())
    .then_some(EgarchQmlParameters {
        omega_normalized,
        alpha,
        gamma,
        beta,
    })
}

pub(super) fn standardized_residual_from_logs(
    residual: f64,
    log_squared_residual: f64,
    log_variance: f64,
) -> Option<f64> {
    if log_squared_residual == f64::NEG_INFINITY {
        return Some(0.0);
    }
    if !residual.is_finite() || !log_squared_residual.is_finite() || !log_variance.is_finite() {
        return None;
    }
    let magnitude = (0.5 * (log_squared_residual - log_variance)).exp();
    magnitude
        .is_finite()
        .then_some(magnitude.copysign(residual))
}

pub(super) fn egarch_centered_news(standardized_residual: f64, alpha: f64, gamma: f64) -> f64 {
    alpha * (standardized_residual.abs() - EGARCH_CENTERED_ABSOLUTE_NORMAL)
        + gamma * standardized_residual
}

pub(super) fn egarch_qml_log_variances(
    data: &NormalizedLogSquares,
    parameters: EgarchQmlParameters,
) -> Option<Vec<f64>> {
    if data.scale == 0.0
        || data.values.is_empty()
        || data.normalized_values.len() != data.values.len()
        || !data.log_mean_square.is_finite()
    {
        return None;
    }
    let mut log_variances = vec![data.log_mean_square; data.values.len()];
    for time in 1..data.values.len() {
        let standardized_residual = standardized_residual_from_logs(
            data.normalized_values[time - 1],
            data.values[time - 1],
            log_variances[time - 1],
        )?;
        let next = parameters.omega_normalized
            + egarch_centered_news(standardized_residual, parameters.alpha, parameters.gamma)
            + parameters.beta * log_variances[time - 1];
        if !next.is_finite() {
            return None;
        }
        log_variances[time] = next;
    }
    Some(log_variances)
}

fn egarch_qml_objective(data: &NormalizedLogSquares, parameters: EgarchQmlParameters) -> f64 {
    let Some(log_variances) = egarch_qml_log_variances(data, parameters) else {
        return f64::INFINITY;
    };
    gaussian_qml_profile(&data.values, &log_variances)
}

#[cfg(test)]
fn egarch_qml_gradient(data: &NormalizedLogSquares, parameters: EgarchQmlParameters) -> [f64; 4] {
    let Some(log_variances) = egarch_qml_log_variances(data, parameters) else {
        return [f64::NAN; 4];
    };
    let mut log_variance_derivative = [0.0; 4];
    let mut gradient = [0.0; 4];
    for time in 0..log_variances.len() {
        let standardized_square = if data.values[time] == f64::NEG_INFINITY {
            0.0
        } else {
            (data.values[time] - log_variances[time]).exp()
        };
        let contribution = 0.5 * (1.0 - standardized_square);
        for coordinate in 0..gradient.len() {
            gradient[coordinate] += contribution * log_variance_derivative[coordinate];
        }
        if time + 1 < log_variances.len() {
            let Some(standardized_residual) = standardized_residual_from_logs(
                data.normalized_values[time],
                data.values[time],
                log_variances[time],
            ) else {
                return [f64::NAN; 4];
            };
            let magnitude_news = standardized_residual.abs() - EGARCH_CENTERED_ABSOLUTE_NORMAL;
            let feedback = parameters.beta
                - 0.5
                    * (parameters.alpha * standardized_residual.abs()
                        + parameters.gamma * standardized_residual);
            log_variance_derivative = [
                1.0 + feedback * log_variance_derivative[0],
                magnitude_news + feedback * log_variance_derivative[1],
                standardized_residual + feedback * log_variance_derivative[2],
                log_variances[time] + feedback * log_variance_derivative[3],
            ];
        }
    }
    gradient
}

fn tracked_egarch_qml_objective(
    data: &NormalizedLogSquares,
    parameters: EgarchQmlParameters,
    overflowed: &mut usize,
) -> f64 {
    let objective = egarch_qml_objective(data, parameters);
    if objective == f64::INFINITY {
        *overflowed += 1;
    }
    objective
}

fn select_egarch_candidate(
    candidates: Vec<EgarchQmlCandidate>,
    objective_tie_ulps: usize,
) -> Option<EgarchQmlCandidate> {
    select_ranked_objective_candidate(
        candidates,
        objective_tie_ulps,
        |candidate| candidate.objective,
        |candidate| candidate.face.selection_rank(),
    )
}

/// Fixed-order asymmetric EGARCH with `p = o = q = 1`.
///
/// The recurrence is
/// `log(h_t) = omega + alpha * (|z_(t-1)| - sqrt(2/pi)) + gamma * z_(t-1)
/// + beta * log(h_(t-1))`. Residuals are max-absolute normalized before
/// optimization, `log(h_0)` is the parameter-independent log mean square,
/// and Gaussian QML is evaluated wholly in the log domain. The domain
/// `0 <= beta < 1` matches `arch.univariate.EGARCH`; the exact `beta = 0`
/// face is searched separately from the open interior.
#[derive(Clone, Debug)]
pub struct Egarch {
    /// Shared derivative-free solver and numerical-quality policy.
    pub optimizer: NelderMead,
    /// Dimensionless edge length of the deterministic initial simplex.
    pub simplex_step: f64,
}

impl Default for Egarch {
    fn default() -> Self {
        Self {
            optimizer: NelderMead::default(),
            simplex_step: EGARCH_DEFAULT_SIMPLEX_STEP,
        }
    }
}

impl Egarch {
    /// Default Gaussian-QML settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted asymmetric EGARCH log-variance recursion.
#[derive(Clone, Debug)]
pub struct FittedEgarch {
    /// Physical-scale log-variance intercept.
    pub omega: f64,
    /// Centered absolute-news coefficient.
    pub alpha: f64,
    /// Signed-news (leverage) coefficient.
    pub gamma: f64,
    /// Log-variance persistence in `[0, 1)`.
    pub beta: f64,
    /// In-sample conditional variances on the observation scale.
    pub sigma2: Vector,
    /// In-sample log conditional variances on the observation scale.
    pub log_sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

impl FittedEgarch {
    fn next_log_variance(&self) -> Option<f64> {
        let log_variance = self.log_sigma2.as_slice().last().copied()?;
        let residual = self.resid.as_slice().last().copied()?;
        if !self.omega.is_finite()
            || !self.alpha.is_finite()
            || !self.gamma.is_finite()
            || !self.beta.is_finite()
            || self.beta < EGARCH_BETA_LOWER
            || self.beta >= EGARCH_BETA_UPPER
            || !log_variance.is_finite()
            || !residual.is_finite()
        {
            return None;
        }
        let log_squared_residual = if residual == 0.0 {
            f64::NEG_INFINITY
        } else {
            2.0 * residual.abs().ln()
        };
        let standardized_residual = if residual == 0.0 {
            0.0
        } else {
            standardized_residual_from_logs(residual, log_squared_residual, log_variance)?
        };
        let next = self.omega
            + egarch_centered_news(standardized_residual, self.alpha, self.gamma)
            + self.beta * log_variance;
        next.is_finite().then_some(next)
    }

    /// Forecast the conditional log variance.
    ///
    /// The first entry is exact given the final observed innovation. Later
    /// entries are conditional expectations of **log variance** under centered
    /// unit-variance Gaussian innovations. They must not be exponentiated and
    /// interpreted as arithmetic expected variances.
    pub fn forecast_log_variance(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast-log-variance"));
        if h == 0 {
            return ctx.finish(Vector::zeros(0));
        }
        let Some(mut next) = self.next_log_variance() else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("EGARCH cannot continue an invalid or empty fitted log-variance path")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        let mut output = Vector::zeros(h);
        for step in 0..h {
            if !next.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NumericalOverflow)
                        .message("EGARCH expected log-variance forecast overflowed")
                        .metric("forecast_step", (step + 1) as f64)
                        .build(),
                );
                return Err(ctx.finish_failure());
            }
            output[step] = next;
            next = self.omega + self.beta * next;
        }
        ctx.finish(output)
    }

    /// Exact one-step conditional variance forecast.
    ///
    /// This API does not provide arithmetic-variance forecasts beyond one step.
    /// Use [`Self::forecast_log_variance`] for the explicitly different
    /// expected-log-variance quantity, or a distribution-specific
    /// simulation/bootstrap forecast for a longer arithmetic-variance horizon.
    pub fn forecast_variance(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast-variance"));
        if h == 0 {
            return ctx.finish(Vector::zeros(0));
        }
        if h != 1 {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("analytic EGARCH arithmetic-variance forecasting supports only h = 1")
                    .metric("h", h as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let Some(log_variance) = self.next_log_variance() else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("EGARCH cannot continue an invalid or empty fitted log-variance path")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        let variance = log_variance.exp();
        if !variance.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("EGARCH one-step physical variance forecast overflowed")
                    .metric("log_variance", log_variance)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if variance == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalUnderflow)
                    .message("EGARCH one-step physical variance forecast underflowed to zero")
                    .metric("log_variance", log_variance)
                    .compromise(NumericalCompromise::new(
                        "return a positive EGARCH variance in binary64",
                        "return zero while preserving the finite log-variance forecast",
                        "the physical-scale exponential is below binary64 range",
                        "zero is representational underflow, not zero process variance",
                    ))
                    .build(),
            );
        }
        ctx.finish(Vector::from_slice(&[variance]))
    }
}

impl FitSeries for Egarch {
    type Fitted = FittedEgarch;

    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedEgarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.policy = self.optimizer.policy.clone();
        inspect_scale_invariant_univariate(&mut ctx, y);
        if y.is_empty() || y.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(ctx.finish_failure());
        }
        if y.len() <= EGARCH_INTERIOR_PARAMETER_COUNT {
            ctx.push(
                Issue::builder(IssueCode::SampleSmallerThanFeatures)
                    .message("EGARCH Gaussian QMLE needs more observations than fitted parameters")
                    .metric("n", y.len() as f64)
                    .metric("p", EGARCH_INTERIOR_PARAMETER_COUNT as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if let Some(issue) =
            insufficient_sample(y.len(), EGARCH_INTERIOR_PARAMETER_COUNT, &ctx.policy)
        {
            ctx.push(issue);
            return Err(ctx.finish_failure());
        }
        if !self.simplex_step.is_finite() || self.simplex_step <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("EGARCH simplex_step must be finite and positive")
                    .metric("simplex_step", self.simplex_step)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let boundary_tolerance = self.optimizer.policy.model_parameter_tol;
        if !boundary_tolerance.is_finite()
            || boundary_tolerance <= 0.0
            || boundary_tolerance >= EGARCH_BETA_UPPER
        {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message(
                        "Policy::model_parameter_tol must be finite, positive, and smaller than the EGARCH beta upper bound",
                    )
                    .metric("model_parameter_tol", boundary_tolerance)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }

        let mean = y.mean();
        let residuals = Vector::from_iter(y.as_slice().iter().map(|value| value - mean));
        if !mean.is_finite() || residuals.as_slice().iter().any(|value| !value.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("EGARCH demeaning produced a non-finite residual")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let Some(data) = normalized_log_squares(residuals.as_slice()) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("EGARCH residual normalization failed")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        if data.scale == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message(
                        "EGARCH parameters are unidentified because every demeaned residual is zero",
                    )
                    .build(),
            );
            return Err(ctx.finish_failure());
        }

        let mut overflowed_objectives = 0usize;
        let mut best_seeds: [Option<(Vector, f64)>; 2] = std::array::from_fn(|_| None);
        for long_run_offset in EGARCH_LONG_RUN_OFFSETS {
            let long_run = data.log_mean_square + long_run_offset;
            for alpha in EGARCH_ALPHA_SEEDS {
                for gamma in EGARCH_GAMMA_SEEDS {
                    let zero_point = Vector::from_slice(&[long_run, alpha, gamma]);
                    let zero_parameters =
                        decode_egarch_qml_point(EgarchQmlFace::BetaZero, zero_point.as_slice())
                            .expect("finite exact-boundary EGARCH seed");
                    let zero_objective = tracked_egarch_qml_objective(
                        &data,
                        zero_parameters,
                        &mut overflowed_objectives,
                    );
                    if zero_objective.is_nan() || zero_objective == f64::NEG_INFINITY {
                        ctx.push(
                            Issue::builder(IssueCode::LossIsNan)
                                .message(
                                    "an exact-boundary EGARCH seed produced an invalid objective",
                                )
                                .build(),
                        );
                        return Err(ctx.finish_failure());
                    }
                    if zero_objective.is_finite()
                        && best_seeds[EgarchQmlFace::BetaZero.index()]
                            .as_ref()
                            .map(|(_, best)| zero_objective < *best)
                            .unwrap_or(true)
                    {
                        best_seeds[EgarchQmlFace::BetaZero.index()] =
                            Some((zero_point, zero_objective));
                    }

                    for beta in EGARCH_BETA_SEEDS {
                        let beta_coordinate =
                            encode_open_interval(beta, EGARCH_BETA_LOWER, EGARCH_BETA_UPPER)
                                .expect("strictly interior EGARCH beta seed");
                        let point = Vector::from_slice(&[long_run, alpha, gamma, beta_coordinate]);
                        let parameters =
                            decode_egarch_qml_point(EgarchQmlFace::Interior, point.as_slice())
                                .expect("finite interior EGARCH seed");
                        let objective = tracked_egarch_qml_objective(
                            &data,
                            parameters,
                            &mut overflowed_objectives,
                        );
                        if objective.is_nan() || objective == f64::NEG_INFINITY {
                            ctx.push(
                                Issue::builder(IssueCode::LossIsNan)
                                    .message(
                                        "an interior EGARCH seed produced an invalid objective",
                                    )
                                    .build(),
                            );
                            return Err(ctx.finish_failure());
                        }
                        if objective.is_finite()
                            && best_seeds[EgarchQmlFace::Interior.index()]
                                .as_ref()
                                .map(|(_, best)| objective < *best)
                                .unwrap_or(true)
                        {
                            best_seeds[EgarchQmlFace::Interior.index()] = Some((point, objective));
                        }
                    }
                }
            }
        }
        if best_seeds.iter().all(Option::is_none) {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .severity(Severity::Fatal)
                    .message("every deterministic EGARCH Gaussian-QML seed overflowed")
                    .metric("overflowed_objectives", overflowed_objectives as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let initial_objective = best_seeds
            .iter()
            .filter_map(|seed| seed.as_ref().map(|(_, objective)| *objective))
            .fold(f64::INFINITY, f64::min);
        ctx.session.step(0, initial_objective, None);

        let mut candidates = Vec::with_capacity(EgarchQmlFace::ALL.len());
        let missing_seed_faces = best_seeds.iter().filter(|seed| seed.is_none()).count();
        let mut max_iteration_faces = 0usize;
        let mut collapsed_faces = 0usize;
        for face in EgarchQmlFace::ALL {
            let Some((seed_point, seed_objective)) = best_seeds[face.index()].take() else {
                continue;
            };
            let mut simplex = Vec::with_capacity(seed_point.len() + 1);
            simplex.push(seed_point.clone());
            for coordinate in 0..seed_point.len() {
                let mut vertex = seed_point.as_slice().to_vec();
                vertex[coordinate] += self.simplex_step;
                simplex.push(Vector::from_iter(vertex));
            }
            let optimized = match self.optimizer.minimize_nested(
                &simplex,
                |point| {
                    let Some(parameters) = decode_egarch_qml_point(face, point) else {
                        return f64::INFINITY;
                    };
                    tracked_egarch_qml_objective(&data, parameters, &mut overflowed_objectives)
                },
                &ctx.session.child(face.session_name()),
            ) {
                Ok(qualified) => qualified,
                Err(failure) => return Err(ctx.merge_failure(failure)),
            };
            let (optimization, optimizer_report) = optimized.into_parts();
            max_iteration_faces +=
                usize::from(optimizer_report.contains(IssueCode::MaxIterReached));
            collapsed_faces += usize::from(optimizer_report.contains(IssueCode::StepSizeCollapsed));
            let (point, objective) = if seed_objective < optimization.value {
                (seed_point, seed_objective)
            } else {
                (optimization.point, optimization.value)
            };
            if objective.is_finite() {
                candidates.push(EgarchQmlCandidate {
                    face,
                    point,
                    objective,
                });
            }
        }
        let Some(selected) = select_egarch_candidate(
            candidates,
            self.optimizer.policy.optimizer_objective_tie_ulps,
        ) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("EGARCH Gaussian QMLE did not produce a finite face candidate")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        // The volatility face has four (or three) free coordinates and the
        // demeaned likelihood also estimates one nuisance mean.
        ctx.report.set_n_parameters(selected.face.dimension() + 1);
        if missing_seed_faces > 0 || max_iteration_faces > 0 || collapsed_faces > 0 {
            let code = match (missing_seed_faces, max_iteration_faces, collapsed_faces) {
                (0, positive, 0) if positive > 0 => IssueCode::MaxIterReached,
                (0, 0, positive) if positive > 0 => IssueCode::StepSizeCollapsed,
                _ => IssueCode::DidNotConverge,
            };
            ctx.push(
                Issue::builder(code)
                    .message(
                        "one or more EGARCH parameter-face searches were incomplete; the reported fit is the best finite candidate and may come from a partial search",
                    )
                    .metric("missing_seed_faces", missing_seed_faces as f64)
                    .metric("max_iteration_faces", max_iteration_faces as f64)
                    .metric("collapsed_faces", collapsed_faces as f64)
                    .metric(
                        "searched_faces",
                        (EgarchQmlFace::ALL.len() - missing_seed_faces) as f64,
                    )
                    .metric("total_faces", EgarchQmlFace::ALL.len() as f64)
                    .build(),
            );
        }
        let selected_face = selected.face;
        let selected_objective = selected.objective;
        let Some(parameters) = decode_egarch_qml_point(selected_face, selected.point.as_slice())
        else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("EGARCH could not decode the selected optimizer point")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        if !selected_objective.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("EGARCH selected a non-finite Gaussian-QML objective")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if overflowed_objectives > 0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .severity(Severity::Advisory)
                    .message(format!(
                        "{overflowed_objectives} dominated EGARCH candidate objectives overflowed to positive infinity"
                    ))
                    .metric("overflowed_objectives", overflowed_objectives as f64)
                    .build(),
            );
        }

        let stationarity_slack = 1.0 - parameters.beta;
        if stationarity_slack <= boundary_tolerance {
            ctx.push(
                Issue::builder(IssueCode::NonStationary)
                    .message("EGARCH beta converged too close to the open stationarity boundary")
                    .metric("beta", parameters.beta)
                    .metric("stationarity_slack", stationarity_slack)
                    .metric("parameter_tolerance", boundary_tolerance)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if selected_face == EgarchQmlFace::BetaZero || parameters.beta <= boundary_tolerance {
            ctx.push(
                Issue::builder(IssueCode::ParameterAtBoundary)
                    .message("EGARCH Gaussian QMLE selected or approached the beta = 0 face")
                    .metric("beta", parameters.beta)
                    .metric("parameter_tolerance", boundary_tolerance)
                    .build(),
            );
        }

        let log_physical_scale_square = 2.0 * data.scale.ln();
        let omega = parameters.omega_normalized + stationarity_slack * log_physical_scale_square;
        if !omega.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("EGARCH physical-scale log-variance intercept is not representable")
                    .metric("omega_normalized", parameters.omega_normalized)
                    .metric("scale", data.scale)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let Some(normalized_log_variances) = egarch_qml_log_variances(&data, parameters) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("EGARCH selected parameters produced an invalid log-variance path")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        let log_sigma2 = normalized_log_variances
            .iter()
            .map(|log_variance| log_physical_scale_square + log_variance)
            .collect::<Vec<_>>();
        if log_sigma2
            .iter()
            .any(|log_variance| !log_variance.is_finite())
        {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("EGARCH physical log-variance path is not representable")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let sigma2 = log_sigma2
            .iter()
            .map(|log_variance| log_variance.exp())
            .collect::<Vec<_>>();
        if sigma2.iter().any(|variance| !variance.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("EGARCH physical conditional variance overflowed")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let underflowed = sigma2.iter().filter(|variance| **variance == 0.0).count();
        if underflowed > 0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalUnderflow)
                    .message(format!(
                        "{underflowed} EGARCH conditional variances underflowed to zero"
                    ))
                    .metric("underflowed_variances", underflowed as f64)
                    .compromise(NumericalCompromise::new(
                        "return every positive EGARCH conditional variance in binary64",
                        "return zero while preserving every finite physical log variance",
                        "one or more physical-scale exponentials are below binary64 range",
                        "zero entries are representational underflow, not zero process variance",
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedEgarch {
            omega,
            alpha: parameters.alpha,
            gamma: parameters.gamma,
            beta: parameters.beta,
            sigma2: Vector::from_iter(sigma2),
            log_sigma2: Vector::from_iter(log_sigma2),
            resid: residuals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn egarch_golden() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../golden/egarch_qml.json"))
            .expect("golden/egarch_qml.json")
    }

    fn egarch_golden_case(name: &str) -> serde_json::Value {
        egarch_golden()["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("missing EGARCH golden case {name}"))
            .clone()
    }

    fn decimal_value(value: &serde_json::Value) -> f64 {
        value
            .as_str()
            .expect("decimal string")
            .parse::<f64>()
            .expect("binary64 value")
    }

    fn golden_observations(case: &serde_json::Value) -> Vec<f64> {
        case["observations"]
            .as_array()
            .expect("observations")
            .iter()
            .map(decimal_value)
            .collect()
    }

    fn leveraged_observations() -> Vec<f64> {
        golden_observations(&egarch_golden_case("leveraged"))
    }

    #[test]
    fn egarch_beta_transform_and_exact_boundary_enforce_the_documented_domain() {
        for beta in EGARCH_BETA_SEEDS {
            let coordinate = encode_open_interval(beta, EGARCH_BETA_LOWER, EGARCH_BETA_UPPER)
                .expect("interior beta");
            let parameters =
                decode_egarch_qml_point(EgarchQmlFace::Interior, &[-0.3, 0.2, -0.1, coordinate])
                    .expect("decoded EGARCH point");
            assert!(parameters.beta > 0.0 && parameters.beta < 1.0);
            assert!((parameters.beta - beta).abs() <= 4.0 * f64::EPSILON);
            assert!(
                (parameters.omega_normalized - (1.0 - beta) * -0.3).abs() <= 4.0 * f64::EPSILON
            );
        }
        let boundary = decode_egarch_qml_point(EgarchQmlFace::BetaZero, &[-0.3, 0.2, -0.1])
            .expect("exact beta-zero face");
        assert_eq!(boundary.beta, 0.0);
        assert_eq!(boundary.omega_normalized, -0.3);
        assert!(decode_egarch_qml_point(EgarchQmlFace::Interior, &[0.0; 3]).is_none());
        assert!(decode_egarch_qml_point(EgarchQmlFace::BetaZero, &[0.0; 4]).is_none());
    }

    #[test]
    fn egarch_log_recurrence_matches_the_direct_equation_without_floors() {
        let residuals = [1.0, -2.0, 0.5, -0.25, 1.5, -0.75, 0.1, -0.1];
        let data = normalized_log_squares(&residuals).expect("finite residuals");
        let parameters = EgarchQmlParameters {
            omega_normalized: -0.1,
            alpha: 0.2,
            gamma: -0.15,
            beta: 0.7,
        };
        let actual = egarch_qml_log_variances(&data, parameters).expect("log recurrence");
        let mut expected = vec![data.log_mean_square; residuals.len()];
        for time in 1..expected.len() {
            let z = data.normalized_values[time - 1] * (-0.5 * expected[time - 1]).exp();
            expected[time] = parameters.omega_normalized
                + parameters.alpha * (z.abs() - EGARCH_CENTERED_ABSOLUTE_NORMAL)
                + parameters.gamma * z
                + parameters.beta * expected[time - 1];
        }
        let maximum_log_error = actual
            .iter()
            .zip(&expected)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(maximum_log_error <= 8.0 * f64::EPSILON);
        let direct_objective = expected
            .iter()
            .zip(&data.normalized_values)
            .map(|(log_variance, residual)| {
                0.5 * (log_variance + residual * residual * (-log_variance).exp())
            })
            .sum::<f64>();
        assert!(
            (egarch_qml_objective(&data, parameters) - direct_objective).abs()
                <= 16.0 * f64::EPSILON
        );
    }

    #[test]
    fn egarch_log_standardization_preserves_a_nonzero_subnormal_sign() {
        let minimum_subnormal = f64::from_bits(1);
        for tiny in [minimum_subnormal, -minimum_subnormal] {
            let data =
                normalized_log_squares(&[2.0, -2.0, tiny]).expect("finite extreme residuals");
            assert_eq!(data.normalized_values[2], 0.0);
            assert_eq!(
                data.normalized_values[2].is_sign_negative(),
                tiny.is_sign_negative()
            );
            assert!(data.values[2].is_finite());

            let standardized = standardized_residual_from_logs(
                data.normalized_values[2],
                data.values[2],
                data.values[2],
            )
            .expect("log-domain standardized residual");
            assert_eq!(standardized, 1.0_f64.copysign(tiny));
        }
    }

    #[test]
    fn egarch_signed_news_has_the_exact_leverage_response() {
        let magnitude = 1.7;
        let alpha = 0.23;
        let gamma = -0.14;
        let positive = egarch_centered_news(magnitude, alpha, gamma);
        let negative = egarch_centered_news(-magnitude, alpha, gamma);
        let expected_difference = -2.0 * gamma * magnitude;
        assert!((negative - positive - expected_difference).abs() <= 2.0 * f64::EPSILON);
        assert_eq!(
            egarch_centered_news(magnitude, alpha, 0.0),
            egarch_centered_news(-magnitude, alpha, 0.0)
        );
    }

    #[test]
    fn egarch_analytic_gradient_matches_centered_differences() {
        let residuals = [1.0, -2.0, 0.5, -0.25, 1.5, -0.75, 0.1, -0.1];
        let data = normalized_log_squares(&residuals).expect("finite residuals");
        let base = [-0.1_f64, 0.2, -0.15, 0.7];
        let parameters = |point: [f64; 4]| EgarchQmlParameters {
            omega_normalized: point[0],
            alpha: point[1],
            gamma: point[2],
            beta: point[3],
        };
        let analytic = egarch_qml_gradient(&data, parameters(base));
        let step = f64::EPSILON.cbrt();
        let mut maximum_error = 0.0_f64;
        for coordinate in 0..base.len() {
            let mut plus = base;
            let mut minus = base;
            plus[coordinate] += step;
            minus[coordinate] -= step;
            let numerical = (egarch_qml_objective(&data, parameters(plus))
                - egarch_qml_objective(&data, parameters(minus)))
                / (2.0 * step);
            maximum_error = maximum_error.max((analytic[coordinate] - numerical).abs());
        }
        eprintln!("EGARCH analytic gradient max centered-difference error={maximum_error:e}");
        // Measured 1.835e-10 on 2026-09-02; tolerance is about 4x.
        assert!(maximum_error <= 7.34e-10);
    }

    #[test]
    fn egarch_transformed_gradient_matches_the_long_run_coordinate() {
        let residuals = [1.0, -2.0, 0.5, -0.25, 1.5, -0.75, 0.1, -0.1];
        let data = normalized_log_squares(&residuals).expect("finite residuals");
        let beta = 0.7;
        let beta_coordinate = encode_open_interval(beta, EGARCH_BETA_LOWER, EGARCH_BETA_UPPER)
            .expect("interior beta");
        let point = [-0.3_f64, 0.2, -0.15, beta_coordinate];
        let parameters =
            decode_egarch_qml_point(EgarchQmlFace::Interior, &point).expect("interior point");
        let physical = egarch_qml_gradient(&data, parameters);
        let analytic = [
            (1.0 - beta) * physical[0],
            physical[1],
            physical[2],
            beta * (1.0 - beta) * (physical[3] - point[0] * physical[0]),
        ];
        let step = f64::EPSILON.cbrt();
        let mut maximum_error = 0.0_f64;
        for coordinate in 0..point.len() {
            let mut plus = point;
            let mut minus = point;
            plus[coordinate] += step;
            minus[coordinate] -= step;
            let plus_parameters =
                decode_egarch_qml_point(EgarchQmlFace::Interior, &plus).expect("plus point");
            let minus_parameters =
                decode_egarch_qml_point(EgarchQmlFace::Interior, &minus).expect("minus point");
            let numerical = (egarch_qml_objective(&data, plus_parameters)
                - egarch_qml_objective(&data, minus_parameters))
                / (2.0 * step);
            maximum_error = maximum_error.max((analytic[coordinate] - numerical).abs());
        }
        eprintln!("EGARCH transformed-gradient max error={maximum_error:e}");
        // Measured 5.489e-11 on 2026-09-02; tolerance is about four times that.
        assert!(maximum_error <= 2.20e-10);
    }

    #[test]
    fn fitted_egarch_forecast_separates_variance_from_expected_log_variance() {
        let fitted = FittedEgarch {
            omega: -0.2,
            alpha: 0.3,
            gamma: -0.1,
            beta: 0.8,
            sigma2: Vector::from_iter([(-0.5_f64).exp(), (-0.2_f64).exp()]),
            log_sigma2: Vector::from_slice(&[-0.5, -0.2]),
            resid: Vector::from_slice(&[0.3, -0.4]),
        };
        let z = -0.4 * 0.1_f64.exp();
        let first_log =
            -0.2 + 0.3 * (z.abs() - EGARCH_CENTERED_ABSOLUTE_NORMAL) - 0.1 * z + 0.8 * -0.2;
        let one_step = fitted
            .forecast_variance(1, &Session::new("egarch", "variance"))
            .expect("one-step variance")
            .value;
        assert!((one_step[0].ln() - first_log).abs() <= 4.0 * f64::EPSILON);
        let log_forecast = fitted
            .forecast_log_variance(3, &Session::new("egarch", "log-variance"))
            .expect("expected log variance")
            .value;
        assert!((log_forecast[0] - first_log).abs() <= 4.0 * f64::EPSILON);
        assert_eq!(
            log_forecast[1],
            fitted.omega + fitted.beta * log_forecast[0]
        );
        assert_eq!(
            log_forecast[2],
            fitted.omega + fitted.beta * log_forecast[1]
        );
        let unsupported = fitted
            .forecast_variance(2, &Session::new("egarch", "unsupported"))
            .unwrap_err();
        assert_eq!(unsupported.primary.code, IssueCode::InvalidParameter);
    }

    #[test]
    fn egarch_rejects_empty_short_nonfinite_constant_and_invalid_config() {
        let empty = Egarch::new()
            .fit_series(&Vector::zeros(0), &Session::new("egarch", "empty"))
            .unwrap_err();
        assert_eq!(empty.primary.code, IssueCode::EmptyMatrix);

        let short = Egarch::new()
            .fit_series(
                &Vector::from_slice(&[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0]),
                &Session::new("egarch", "short"),
            )
            .unwrap_err();
        assert_eq!(short.primary.code, IssueCode::InsufficientSample);

        let nonfinite = Egarch::new()
            .fit_series(
                &Vector::from_slice(&[1.0, -1.0, 0.5, -0.5, f64::NAN, 0.25, -0.25, 0.0]),
                &Session::new("egarch", "nonfinite"),
            )
            .unwrap_err();
        assert_eq!(nonfinite.primary.code, IssueCode::NonFiniteInput);

        let constant = Egarch::new()
            .fit_series(
                &Vector::from_slice(&[7.0; 25]),
                &Session::new("egarch", "constant"),
            )
            .unwrap_err();
        assert_eq!(constant.primary.code, IssueCode::UnidentifiedModel);

        let mut invalid = Egarch::new();
        invalid.simplex_step = 0.0;
        let invalid_observations = leveraged_observations();
        let invalid = invalid
            .fit_series(
                &Vector::from_slice(&invalid_observations[..25]),
                &Session::new("egarch", "invalid-config"),
            )
            .unwrap_err();
        assert_eq!(invalid.primary.code, IssueCode::InvalidParameter);

        let mut invalid_boundary_policy = Egarch::new();
        invalid_boundary_policy.optimizer.policy.model_parameter_tol = EGARCH_BETA_UPPER;
        let invalid_boundary_policy = invalid_boundary_policy
            .fit_series(
                &Vector::from_slice(&invalid_observations[..25]),
                &Session::new("egarch", "invalid-boundary-policy"),
            )
            .unwrap_err();
        assert_eq!(
            invalid_boundary_policy.primary.code,
            IssueCode::InvalidParameter
        );
    }

    #[test]
    fn egarch_enforces_the_policy_sample_budget_and_counts_the_mean() {
        let observations = leveraged_observations();
        let below = Egarch::new()
            .fit_series(
                &Vector::from_slice(&observations[..24]),
                &Session::new("egarch", "below-policy-budget"),
            )
            .unwrap_err();
        assert_eq!(below.primary.code, IssueCode::InsufficientSample);

        let at_budget = Egarch::new()
            .fit_series(
                &Vector::from_slice(&observations[..25]),
                &Session::new("egarch", "at-policy-budget"),
            )
            .expect("five observations per fitted interior parameter");
        assert!(matches!(at_budget.report.n_parameters, Some(4 | 5)));
    }

    #[test]
    fn egarch_decimal_golden_replays_the_numerical_kernel() {
        let payload = egarch_golden();
        let cases = payload["cases"].as_array().expect("cases");
        assert_eq!(cases.len(), 7, "oracle generator documents seven cases");
        let tolerances = &payload["rust_kernel_tolerances"];
        let objective_tolerance = decimal_value(&tolerances["objective_absolute"]);
        let log_variance_tolerance = decimal_value(&tolerances["log_variance_absolute"]);
        let variance_tolerance = decimal_value(&tolerances["physical_variance_relative"]);
        let forecast_tolerance = decimal_value(&tolerances["one_step_forecast_relative"]);
        let gradient_tolerance = decimal_value(&tolerances["physical_gradient_absolute"]);
        let mut maximum_objective_error = 0.0_f64;
        let mut maximum_log_variance_error = 0.0_f64;
        let mut maximum_variance_relative_error = 0.0_f64;
        let mut maximum_forecast_relative_error = 0.0_f64;
        let mut maximum_gradient_error = 0.0_f64;

        for case in cases {
            let name = case["name"].as_str().expect("case name");
            let observations = golden_observations(case);
            let y = Vector::from_slice(&observations);
            let mean = y.mean();
            let residuals = observations
                .iter()
                .map(|observation| observation - mean)
                .collect::<Vec<_>>();
            let data = normalized_log_squares(&residuals).expect("finite golden observations");
            match case["outcome"].as_str().expect("outcome") {
                "unidentified_constant_series" => {
                    assert_eq!(data.scale, 0.0, "{name}");
                }
                "extended_real_probe" => {
                    let expected = &case["parameters"];
                    let parameters = EgarchQmlParameters {
                        omega_normalized: decimal_value(&expected["omega_normalized"]),
                        alpha: decimal_value(&expected["alpha"]),
                        gamma: decimal_value(&expected["gamma"]),
                        beta: decimal_value(&expected["beta"]),
                    };
                    let objective = egarch_qml_objective(&data, parameters);
                    assert_eq!(objective, f64::INFINITY, "{name}");
                    assert!(!objective.is_nan(), "{name}");
                    assert!(case["decimal_objective_is_finite"]
                        .as_bool()
                        .expect("Decimal finiteness"));
                }
                "success" => {
                    let expected = &case["fit"];
                    let parameters = EgarchQmlParameters {
                        omega_normalized: decimal_value(&expected["omega_normalized"]),
                        alpha: decimal_value(&expected["alpha"]),
                        gamma: decimal_value(&expected["gamma"]),
                        beta: decimal_value(&expected["beta"]),
                    };
                    let objective = egarch_qml_objective(&data, parameters);
                    maximum_objective_error = maximum_objective_error
                        .max((objective - decimal_value(&expected["objective"])).abs());

                    let log_variances =
                        egarch_qml_log_variances(&data, parameters).expect("finite oracle path");
                    let expected_log_variances = case["normalized_log_variances"]
                        .as_array()
                        .expect("normalized log variances");
                    let expected_variances = case["physical_variances"]
                        .as_array()
                        .expect("physical variances");
                    assert_eq!(log_variances.len(), expected_log_variances.len(), "{name}");
                    assert_eq!(log_variances.len(), expected_variances.len(), "{name}");
                    let log_scale_square = 2.0 * data.scale.ln();
                    for ((actual_log, expected_log), expected_variance) in log_variances
                        .iter()
                        .zip(expected_log_variances)
                        .zip(expected_variances)
                    {
                        maximum_log_variance_error = maximum_log_variance_error
                            .max((actual_log - decimal_value(expected_log)).abs());
                        let actual_variance = (log_scale_square + actual_log).exp();
                        maximum_variance_relative_error = maximum_variance_relative_error
                            .max((actual_variance / decimal_value(expected_variance) - 1.0).abs());
                    }

                    let last = log_variances.len() - 1;
                    let standardized_residual = standardized_residual_from_logs(
                        data.normalized_values[last],
                        data.values[last],
                        log_variances[last],
                    )
                    .expect("finite final standardized residual");
                    let next_log_variance = parameters.omega_normalized
                        + egarch_centered_news(
                            standardized_residual,
                            parameters.alpha,
                            parameters.gamma,
                        )
                        + parameters.beta * log_variances[last];
                    let actual_forecast = (log_scale_square + next_log_variance).exp();
                    let expected_forecast =
                        decimal_value(&case["one_step_forecast"]["physical_variance"]);
                    maximum_forecast_relative_error = maximum_forecast_relative_error
                        .max((actual_forecast / expected_forecast - 1.0).abs());

                    let expected_gradient = &expected["physical_gradient"];
                    let actual_gradient = egarch_qml_gradient(&data, parameters);
                    for (actual, key) in actual_gradient
                        .into_iter()
                        .zip(["omega", "alpha", "gamma", "beta"])
                    {
                        maximum_gradient_error = maximum_gradient_error
                            .max((actual - decimal_value(&expected_gradient[key])).abs());
                    }
                    if expected["selection"] == "beta_zero" {
                        let mu = decimal_value(&expected["mu_normalized"]);
                        let direction = actual_gradient[3] - mu * actual_gradient[0];
                        assert!(
                            direction >= -gradient_tolerance,
                            "{name}: beta=0 KKT direction"
                        );
                    }
                }
                outcome => panic!("unknown EGARCH golden outcome {outcome}"),
            }
        }
        eprintln!(
            "EGARCH Decimal kernel max errors: objective={maximum_objective_error:e}, log-variance={maximum_log_variance_error:e}, variance-relative={maximum_variance_relative_error:e}, forecast-relative={maximum_forecast_relative_error:e}, gradient={maximum_gradient_error:e}"
        );
        // The golden stores Rust-side measured maxima and tolerances with
        // approximately four times the observed errors.
        assert!(maximum_objective_error <= objective_tolerance);
        assert!(maximum_log_variance_error <= log_variance_tolerance);
        assert!(maximum_variance_relative_error <= variance_tolerance);
        assert!(maximum_forecast_relative_error <= forecast_tolerance);
        // Measured Rust-versus-Decimal physical-gradient error 1.192e-12 on
        // 2026-09-02; tolerance is about four times that maximum.
        assert!(maximum_gradient_error <= gradient_tolerance);
    }

    #[test]
    fn egarch_qmle_matches_decimal_golden_faces_and_paths() {
        let payload = egarch_golden();
        let cases = payload["cases"].as_array().expect("cases");
        let tolerances = &payload["rust_fit_tolerances"];
        let mut maximum_coefficient_error = 0.0_f64;
        let mut maximum_omega_error = 0.0_f64;
        let mut maximum_objective_error = 0.0_f64;
        let mut maximum_log_variance_error = 0.0_f64;
        let mut maximum_variance_relative_error = 0.0_f64;
        let mut maximum_forecast_relative_error = 0.0_f64;

        for case in cases.iter().filter(|case| case["outcome"] == "success") {
            let name = case["name"].as_str().expect("case name");
            let observations = golden_observations(case);
            let fitted = Egarch::new()
                .fit_series(
                    &Vector::from_slice(&observations),
                    &Session::new("egarch-decimal", name),
                )
                .expect("oracle expects an attained EGARCH fit");
            let expected = &case["fit"];
            let expected_alpha = decimal_value(&expected["alpha"]);
            let expected_gamma = decimal_value(&expected["gamma"]);
            let expected_beta = decimal_value(&expected["beta"]);
            maximum_coefficient_error = maximum_coefficient_error
                .max((fitted.value.alpha - expected_alpha).abs())
                .max((fitted.value.gamma - expected_gamma).abs())
                .max((fitted.value.beta - expected_beta).abs());
            maximum_omega_error = maximum_omega_error
                .max((fitted.value.omega - decimal_value(&expected["omega_physical"])).abs());

            let data = normalized_log_squares(fitted.value.resid.as_slice())
                .expect("fitted residual normalization");
            let parameters = EgarchQmlParameters {
                omega_normalized: fitted.value.omega
                    - (1.0 - fitted.value.beta) * 2.0 * data.scale.ln(),
                alpha: fitted.value.alpha,
                gamma: fitted.value.gamma,
                beta: fitted.value.beta,
            };
            maximum_objective_error = maximum_objective_error.max(
                (egarch_qml_objective(&data, parameters) - decimal_value(&expected["objective"]))
                    .abs(),
            );

            let expected_log_variances = case["normalized_log_variances"]
                .as_array()
                .expect("normalized log variances");
            let expected_variances = case["physical_variances"]
                .as_array()
                .expect("physical variances");
            assert_eq!(fitted.value.log_sigma2.len(), expected_log_variances.len());
            for (((actual_log, actual_variance), expected_log), expected_variance) in fitted
                .value
                .log_sigma2
                .as_slice()
                .iter()
                .zip(fitted.value.sigma2.as_slice())
                .zip(expected_log_variances)
                .zip(expected_variances)
            {
                let expected_physical_log = 2.0 * data.scale.ln() + decimal_value(expected_log);
                maximum_log_variance_error =
                    maximum_log_variance_error.max((actual_log - expected_physical_log).abs());
                maximum_variance_relative_error = maximum_variance_relative_error
                    .max((actual_variance / decimal_value(expected_variance) - 1.0).abs());
            }
            let forecast = fitted
                .value
                .forecast_variance(1, &Session::new("egarch-decimal", "forecast"))
                .expect("one-step forecast")
                .value[0];
            maximum_forecast_relative_error = maximum_forecast_relative_error.max(
                (forecast / decimal_value(&case["one_step_forecast"]["physical_variance"]) - 1.0)
                    .abs(),
            );

            match expected["selection"].as_str().expect("selection") {
                "beta_zero" => {
                    assert_eq!(fitted.value.beta, 0.0, "{name}");
                    assert_eq!(fitted.report.n_parameters, Some(4), "{name}");
                    assert!(fitted.report.contains(IssueCode::ParameterAtBoundary));
                }
                "interior" => {
                    assert!(fitted.value.beta > 0.0 && fitted.value.beta < 1.0, "{name}");
                    assert_eq!(fitted.report.n_parameters, Some(5), "{name}");
                    assert!(
                        !fitted.report.contains(IssueCode::ParameterAtBoundary),
                        "{name}"
                    );
                }
                selection => panic!("unknown EGARCH selection {selection}"),
            }
        }
        eprintln!(
            "EGARCH Decimal fit max errors: coefficient={maximum_coefficient_error:e}, omega={maximum_omega_error:e}, objective={maximum_objective_error:e}, log-variance={maximum_log_variance_error:e}, variance-relative={maximum_variance_relative_error:e}, forecast-relative={maximum_forecast_relative_error:e}"
        );
        // Measured on 2026-09-02: coefficient 2.247e-8, physical omega
        // 9.489e-9, objective 2.274e-13, physical log variance 7.843e-8,
        // variance-relative 7.843e-8, and one-step-relative 1.937e-8.
        // Tolerances below are about four times those maxima.
        assert!(maximum_coefficient_error <= decimal_value(&tolerances["coefficient_absolute"]));
        assert!(maximum_omega_error <= decimal_value(&tolerances["omega_absolute"]));
        assert!(maximum_objective_error <= decimal_value(&tolerances["objective_absolute"]));
        assert!(maximum_log_variance_error <= decimal_value(&tolerances["log_variance_absolute"]));
        assert!(
            maximum_variance_relative_error
                <= decimal_value(&tolerances["physical_variance_relative"])
        );
        assert!(
            maximum_forecast_relative_error
                <= decimal_value(&tolerances["one_step_forecast_relative"])
        );
    }

    #[test]
    fn egarch_preserves_partial_optimizer_reports_and_strict_failures() {
        let observations = leveraged_observations();
        let y = Vector::from_slice(&observations);
        let mut capped = Egarch::new();
        capped.optimizer.max_iterations = 1;
        let capped_session = Session::new("egarch", "capped");
        let capped = capped
            .fit_series(&y, &capped_session)
            .expect("iteration cap returns a qualified EGARCH fit");
        assert!(capped.report.contains(IssueCode::MaxIterReached));
        assert!(capped.is_compromised());
        assert_eq!(
            capped_session
                .ledger()
                .events()
                .iter()
                .filter(|event| {
                    event.kind == ojizou_san::EventKind::QualityWarning
                        && event.issue.as_ref().map(|issue| issue.code)
                            == Some(IssueCode::MaxIterReached)
                })
                .count(),
            1,
            "the aggregate partial-search issue must be ingested exactly once"
        );
        assert_eq!(
            capped_session
                .ledger()
                .of_kind(ojizou_san::EventKind::FitFinished)
                .len(),
            1,
            "only the enclosing EGARCH fit records a terminal event"
        );

        let mut strict = Egarch::new();
        strict.optimizer.max_iterations = 1;
        strict.optimizer.policy.abort_at = Severity::Warning;
        let failure = strict
            .fit_series(&y, &Session::new("egarch", "strict-capped"))
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::MaxIterReached);
        assert!(failure.report.contains(IssueCode::MaxIterReached));
    }

    #[test]
    fn egarch_reports_physical_variance_underflow_and_overflow() {
        let observations = leveraged_observations();
        let tiny = Vector::from_iter(observations.iter().map(|value| value * 1e-200));
        let underflow = Egarch::new()
            .fit_series(&tiny, &Session::new("egarch", "physical-underflow"))
            .expect("finite log-variance path survives physical underflow");
        assert!(underflow.report.contains(IssueCode::NumericalUnderflow));
        assert!(underflow
            .value
            .sigma2
            .as_slice()
            .iter()
            .any(|variance| *variance == 0.0));
        assert!(underflow
            .value
            .log_sigma2
            .as_slice()
            .iter()
            .all(|log_variance| log_variance.is_finite()));

        let huge = Vector::from_iter(observations.iter().map(|value| value * 1e155));
        let overflow = Egarch::new()
            .fit_series(&huge, &Session::new("egarch", "physical-overflow"))
            .unwrap_err();
        assert_eq!(overflow.primary.code, IssueCode::NumericalOverflow);
    }

    #[test]
    fn egarch_fit_replays_gamma_and_is_sign_and_scale_equivariant() {
        let observations = leveraged_observations();
        let base = Egarch::new()
            .fit_series(
                &Vector::from_slice(&observations),
                &Session::new("egarch", "base"),
            )
            .expect("base EGARCH");
        assert_eq!(base.value.sigma2.len(), observations.len());
        assert_eq!(base.value.log_sigma2.len(), observations.len());
        assert!(base.value.beta >= 0.0 && base.value.beta < 1.0);
        assert!(base.value.gamma.is_finite());
        let mut maximum_recurrence_error = 0.0_f64;
        for time in 1..base.value.log_sigma2.len() {
            let previous_log_variance = base.value.log_sigma2[time - 1];
            let previous_residual = base.value.resid[time - 1];
            let standardized_residual = if previous_residual == 0.0 {
                0.0
            } else {
                previous_residual * (-0.5 * previous_log_variance).exp()
            };
            let expected = base.value.omega
                + egarch_centered_news(standardized_residual, base.value.alpha, base.value.gamma)
                + base.value.beta * previous_log_variance;
            maximum_recurrence_error =
                maximum_recurrence_error.max((base.value.log_sigma2[time] - expected).abs());
        }

        let reflected_observations = observations.iter().map(|value| -*value).collect::<Vec<_>>();
        let reflected = Egarch::new()
            .fit_series(
                &Vector::from_slice(&reflected_observations),
                &Session::new("egarch", "reflected"),
            )
            .expect("sign-reflected EGARCH");
        let sign_parameter_error = (base.value.omega - reflected.value.omega)
            .abs()
            .max((base.value.alpha - reflected.value.alpha).abs())
            .max((base.value.gamma + reflected.value.gamma).abs())
            .max((base.value.beta - reflected.value.beta).abs());
        let sign_path_error = base
            .value
            .log_sigma2
            .as_slice()
            .iter()
            .zip(reflected.value.log_sigma2.as_slice())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);

        let mut maximum_scale_coefficient_error = 0.0_f64;
        let base_residual_scale = base
            .value
            .resid
            .as_slice()
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        let base_omega_normalized =
            base.value.omega - (1.0 - base.value.beta) * 2.0 * base_residual_scale.ln();
        let mut maximum_scale_normalized_omega_error = 0.0_f64;
        let mut maximum_scale_log_variance_error = 0.0_f64;
        for factor in [1e-100_f64, 1e100_f64] {
            let scaled_observations = observations
                .iter()
                .map(|value| value * factor)
                .collect::<Vec<_>>();
            let scaled = Egarch::new()
                .fit_series(
                    &Vector::from_slice(&scaled_observations),
                    &Session::new("egarch", "scaled"),
                )
                .expect("scaled EGARCH");
            maximum_scale_coefficient_error = maximum_scale_coefficient_error
                .max((scaled.value.alpha - base.value.alpha).abs())
                .max((scaled.value.gamma - base.value.gamma).abs())
                .max((scaled.value.beta - base.value.beta).abs());
            let scaled_residual_scale = scaled
                .value
                .resid
                .as_slice()
                .iter()
                .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
            let scaled_omega_normalized =
                scaled.value.omega - (1.0 - scaled.value.beta) * 2.0 * scaled_residual_scale.ln();
            maximum_scale_normalized_omega_error = maximum_scale_normalized_omega_error
                .max((scaled_omega_normalized - base_omega_normalized).abs());
            let log_scale_square = 2.0 * factor.abs().ln();
            for (scaled_log_variance, base_log_variance) in scaled
                .value
                .log_sigma2
                .as_slice()
                .iter()
                .zip(base.value.log_sigma2.as_slice())
            {
                maximum_scale_log_variance_error = maximum_scale_log_variance_error
                    .max((scaled_log_variance - base_log_variance - log_scale_square).abs());
            }
        }
        eprintln!(
            "EGARCH invariants: recurrence={maximum_recurrence_error:e}, sign-parameters={sign_parameter_error:e}, sign-path={sign_path_error:e}, scale-coefficients={maximum_scale_coefficient_error:e}, scale-normalized-omega={maximum_scale_normalized_omega_error:e}, scale-log-path={maximum_scale_log_variance_error:e}"
        );
        // Measured on 2026-09-02: recurrence 5.552e-16, sign parameters
        // 2.023e-8, sign path 5.401e-8, scale coefficients 1.935e-8,
        // normalized omega 3.011e-8, and scale log path 7.710e-8.
        // Tolerances below are about 4x.
        assert!(maximum_recurrence_error <= 2.23e-15);
        assert!(sign_parameter_error <= 8.09e-8);
        assert!(sign_path_error <= 2.17e-7);
        assert!(maximum_scale_coefficient_error <= 7.74e-8);
        assert!(maximum_scale_normalized_omega_error <= 1.21e-7);
        assert!(maximum_scale_log_variance_error <= 3.09e-7);
    }
}
