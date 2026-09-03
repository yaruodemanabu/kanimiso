//! Fixed-order fractionally integrated EGARCH Gaussian QMLE.

use super::common::{
    compensated_sum, gaussian_qml_profile, inspect_scale_invariant_univariate,
    normalized_log_squares, select_ranked_objective_candidate, NormalizedLogSquares,
};
#[cfg(test)]
use super::egarch::EGARCH_CENTERED_ABSOLUTE_NORMAL;
use super::egarch::{egarch_centered_news, standardized_residual_from_logs};
use crate::context::FitCtx;
use crate::data::Vector;
use crate::optimize::{decode_open_interval, encode_open_interval, NelderMead};
use crate::traits::FitSeries;
use ojizou_san::Session;
use signlred::{
    insufficient_sample, Issue, IssueCode, NumericalCompromise, Qualified, Result, Severity,
};

const FIEGARCH_DEFAULT_SIMPLEX_STEP: f64 = 0.25;
const FIEGARCH_DEFAULT_TRUNCATION: usize = 1_000;
const FIEGARCH_INTERIOR_PARAMETER_COUNT: usize = 6;
const FIEGARCH_BETA_LOWER: f64 = -1.0;
const FIEGARCH_BETA_UPPER: f64 = 1.0;
const FIEGARCH_D_LOWER: f64 = 0.0;
const FIEGARCH_D_UPPER: f64 = 0.5;
const FIEGARCH_OMEGA_OFFSETS: [f64; 5] = [-4.0, -2.0, 0.0, 2.0, 4.0];
const FIEGARCH_ALPHA_SEEDS: [f64; 4] = [-0.25, 0.0, 0.15, 0.4];
const FIEGARCH_GAMMA_SEEDS: [f64; 3] = [-0.3, 0.0, 0.3];
const FIEGARCH_BETA_SEEDS: [f64; 6] = [-0.75, -0.25, 0.0, 0.45, 0.8, 0.95];
const FIEGARCH_D_SEEDS: [f64; 4] = [0.04, 0.16, 0.30, 0.44];

#[derive(Clone, Copy, Debug)]
struct FiegarchQmlParameters {
    mu_normalized: f64,
    alpha: f64,
    gamma: f64,
    beta: f64,
    d: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FiegarchQmlFace {
    Interior,
    DZero,
}

impl FiegarchQmlFace {
    const ALL: [Self; 2] = [Self::Interior, Self::DZero];

    fn index(self) -> usize {
        match self {
            Self::Interior => 0,
            Self::DZero => 1,
        }
    }

    fn dimension(self) -> usize {
        match self {
            Self::Interior => 5,
            Self::DZero => 4,
        }
    }

    fn selection_rank(self) -> usize {
        match self {
            Self::DZero => 0,
            Self::Interior => 1,
        }
    }

    fn session_name(self) -> &'static str {
        match self {
            Self::Interior => "qml-interior",
            Self::DZero => "qml-d-zero",
        }
    }
}

struct FiegarchQmlCandidate {
    face: FiegarchQmlFace,
    point: Vector,
    objective: f64,
}

struct FiegarchQmlPath {
    log_variances: Vec<f64>,
    #[cfg(test)]
    news: Vec<f64>,
    #[cfg(test)]
    weights: Vec<f64>,
}

fn decode_fiegarch_qml_point(
    face: FiegarchQmlFace,
    point: &[f64],
) -> Option<FiegarchQmlParameters> {
    let (mu_normalized, alpha, gamma, beta_coordinate, d) = match (face, point) {
        (FiegarchQmlFace::Interior, [omega, alpha, gamma, beta, d_coordinate]) => (
            *omega,
            *alpha,
            *gamma,
            *beta,
            decode_open_interval(*d_coordinate, FIEGARCH_D_LOWER, FIEGARCH_D_UPPER)?,
        ),
        (FiegarchQmlFace::DZero, [omega, alpha, gamma, beta]) => {
            (*omega, *alpha, *gamma, *beta, FIEGARCH_D_LOWER)
        }
        _ => return None,
    };
    let beta = decode_open_interval(beta_coordinate, FIEGARCH_BETA_LOWER, FIEGARCH_BETA_UPPER)?;
    [mu_normalized, alpha, gamma, beta, d]
        .iter()
        .all(|value| value.is_finite())
        .then_some(FiegarchQmlParameters {
            mu_normalized,
            alpha,
            gamma,
            beta,
            d,
        })
}

fn fractional_integration_weights(d: f64, length: usize) -> Option<Vec<f64>> {
    if !d.is_finite() || !(FIEGARCH_D_LOWER..FIEGARCH_D_UPPER).contains(&d) {
        return None;
    }
    if length == 0 {
        return Some(Vec::new());
    }
    let mut weights = vec![0.0; length];
    weights[0] = 1.0;
    for lag in 1..length {
        let previous = weights[lag - 1];
        let next = previous * (lag as f64 - 1.0 + d) / lag as f64;
        if !next.is_finite() {
            return None;
        }
        weights[lag] = next;
    }
    Some(weights)
}

fn fiegarch_qml_path(
    data: &NormalizedLogSquares,
    parameters: FiegarchQmlParameters,
    truncation: usize,
) -> Option<FiegarchQmlPath> {
    if data.scale == 0.0
        || data.values.is_empty()
        || data.normalized_values.len() != data.values.len()
        || !data.log_mean_square.is_finite()
        || truncation == 0
        || !parameters.mu_normalized.is_finite()
        || !parameters.alpha.is_finite()
        || !parameters.gamma.is_finite()
        || !parameters.beta.is_finite()
        || !(FIEGARCH_BETA_LOWER..FIEGARCH_BETA_UPPER).contains(&parameters.beta)
    {
        return None;
    }
    let effective_truncation = truncation.min(data.values.len());
    let filter_length = if parameters.d == FIEGARCH_D_LOWER {
        1
    } else {
        effective_truncation
    };
    let weights = fractional_integration_weights(parameters.d, filter_length)?;
    let mut log_variances = vec![data.log_mean_square; data.values.len()];
    let mut news = vec![0.0; data.values.len()];
    for time in 0..data.values.len() {
        let standardized_residual = standardized_residual_from_logs(
            data.normalized_values[time],
            data.values[time],
            log_variances[time],
        )?;
        news[time] =
            egarch_centered_news(standardized_residual, parameters.alpha, parameters.gamma);
        if !news[time].is_finite() {
            return None;
        }
        if time + 1 < data.values.len() {
            let included = (time + 1).min(weights.len());
            let fractional_news = compensated_sum(
                weights[..included]
                    .iter()
                    .enumerate()
                    .map(|(lag, weight)| *weight * news[time - lag]),
            )?;
            let next = (1.0 - parameters.beta) * parameters.mu_normalized
                + parameters.beta * log_variances[time]
                + fractional_news;
            if !next.is_finite() {
                return None;
            }
            log_variances[time + 1] = next;
        }
    }
    Some(FiegarchQmlPath {
        log_variances,
        #[cfg(test)]
        news,
        #[cfg(test)]
        weights,
    })
}

fn fiegarch_qml_objective(
    data: &NormalizedLogSquares,
    parameters: FiegarchQmlParameters,
    truncation: usize,
) -> f64 {
    let Some(path) = fiegarch_qml_path(data, parameters, truncation) else {
        return f64::INFINITY;
    };
    gaussian_qml_profile(&data.values, &path.log_variances)
}

fn tracked_fiegarch_qml_objective(
    data: &NormalizedLogSquares,
    parameters: FiegarchQmlParameters,
    truncation: usize,
    overflowed: &mut usize,
) -> f64 {
    let objective = fiegarch_qml_objective(data, parameters, truncation);
    if objective == f64::INFINITY {
        *overflowed += 1;
    }
    objective
}

fn select_fiegarch_candidate(
    candidates: Vec<FiegarchQmlCandidate>,
    objective_tie_ulps: usize,
) -> Option<FiegarchQmlCandidate> {
    select_ranked_objective_candidate(
        candidates,
        objective_tie_ulps,
        |candidate| candidate.objective,
        |candidate| candidate.face.selection_rank(),
    )
}

fn admissible_fiegarch_candidate(
    face: FiegarchQmlFace,
    point: &[f64],
    lower_boundary_guard: f64,
) -> bool {
    let Some(parameters) = decode_fiegarch_qml_point(face, point) else {
        return false;
    };
    face != FiegarchQmlFace::Interior || parameters.d > lower_boundary_guard
}

/// Fractionally integrated exponential GARCH with one short-memory AR term.
///
/// This is the fixed-order specialization of Bollerslev--Mikkelsen Eq. 11
/// without an additional news-MA polynomial:
/// `(1 - beta * L)(log(h[t]) - omega) = (1 - L)^(-d) g(z[t-1])`,
/// where `g(z) = alpha * (|z| - sqrt(2/pi)) + gamma * z`.
/// `omega` is the long-run mean log variance, `-1 < beta < 1`, and
/// `0 <= d < 1/2`. Only the inverse-fractional filter is truncated; the AR
/// recursion remains exact. The exact `d = 0` EGARCH face is searched
/// separately from the open long-memory interior.
#[derive(Clone, Debug)]
pub struct Fiegarch {
    /// Shared derivative-free solver and numerical-quality policy.
    pub optimizer: NelderMead,
    /// Dimensionless edge length of the deterministic initial simplex.
    pub simplex_step: f64,
    /// Runtime lag count for the inverse-fractional filter; must be at least 2.
    pub truncation: usize,
}

impl Default for Fiegarch {
    fn default() -> Self {
        Self {
            optimizer: NelderMead::default(),
            simplex_step: FIEGARCH_DEFAULT_SIMPLEX_STEP,
            truncation: FIEGARCH_DEFAULT_TRUNCATION,
        }
    }
}

impl Fiegarch {
    /// Default Gaussian-QML and finite-filter settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted fixed-order FIEGARCH log-variance recursion.
#[derive(Clone, Debug)]
pub struct FittedFiegarch {
    /// Physical-scale long-run mean log variance (paper Eq. 11 omega).
    pub omega: f64,
    /// Centered absolute-news coefficient.
    pub alpha: f64,
    /// Signed-news (leverage) coefficient.
    pub gamma: f64,
    /// Short-memory log-variance AR coefficient in `(-1, 1)`.
    pub beta: f64,
    /// Fractional integration order in `[0, 1/2)`.
    pub d: f64,
    /// Configured maximum lag count for the inverse-fractional filter.
    /// In-sample fitting uses `min(truncation, n_observations)` coefficients.
    pub truncation: usize,
    /// In-sample conditional variances on the observation scale.
    pub sigma2: Vector,
    /// In-sample log conditional variances on the observation scale.
    pub log_sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

impl FitSeries for Fiegarch {
    type Fitted = FittedFiegarch;

    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedFiegarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.policy = self.optimizer.policy.clone();
        inspect_scale_invariant_univariate(&mut ctx, y);
        if y.is_empty() || y.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(ctx.finish_failure());
        }
        if y.len() <= FIEGARCH_INTERIOR_PARAMETER_COUNT {
            ctx.push(
                Issue::builder(IssueCode::SampleSmallerThanFeatures)
                    .message(
                        "FIEGARCH Gaussian QMLE needs more observations than fitted parameters",
                    )
                    .metric("n", y.len() as f64)
                    .metric("p", FIEGARCH_INTERIOR_PARAMETER_COUNT as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if !self.simplex_step.is_finite() || self.simplex_step <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("FIEGARCH simplex_step must be finite and positive")
                    .metric("simplex_step", self.simplex_step)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if self.truncation < 2 {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("FIEGARCH truncation must be at least 2 so d is identified")
                    .metric("truncation", self.truncation as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if self.truncation > self.optimizer.policy.max_infinite_filter_terms {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("FIEGARCH truncation exceeds Policy::max_infinite_filter_terms")
                    .metric("truncation", self.truncation as f64)
                    .metric(
                        "max_infinite_filter_terms",
                        self.optimizer.policy.max_infinite_filter_terms as f64,
                    )
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let boundary_tolerance = self.optimizer.policy.model_parameter_tol;
        if !boundary_tolerance.is_finite()
            || boundary_tolerance <= 0.0
            || boundary_tolerance >= FIEGARCH_D_UPPER
        {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message(
                        "Policy::model_parameter_tol must be finite, positive, and smaller than the FIEGARCH d upper bound",
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
                    .message("FIEGARCH demeaning produced a non-finite residual")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let Some(data) = normalized_log_squares(residuals.as_slice()) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("FIEGARCH residual normalization failed")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        if data.scale == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message(
                        "FIEGARCH parameters are unidentified because every demeaned residual is zero",
                    )
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if let Some(issue) =
            insufficient_sample(y.len(), FIEGARCH_INTERIOR_PARAMETER_COUNT, &ctx.policy)
        {
            ctx.push(issue);
            return Err(ctx.finish_failure());
        }

        let effective_truncation = self.truncation.min(y.len());
        let mut overflowed_objectives = 0usize;
        let mut best_seeds: [Option<(Vector, f64)>; 2] = std::array::from_fn(|_| None);
        for omega_offset in FIEGARCH_OMEGA_OFFSETS {
            let omega = data.log_mean_square + omega_offset;
            for alpha in FIEGARCH_ALPHA_SEEDS {
                for gamma in FIEGARCH_GAMMA_SEEDS {
                    for beta in FIEGARCH_BETA_SEEDS {
                        let beta_coordinate =
                            encode_open_interval(beta, FIEGARCH_BETA_LOWER, FIEGARCH_BETA_UPPER)
                                .expect("strictly interior FIEGARCH beta seed");
                        let zero_point =
                            Vector::from_slice(&[omega, alpha, gamma, beta_coordinate]);
                        let zero_parameters = decode_fiegarch_qml_point(
                            FiegarchQmlFace::DZero,
                            zero_point.as_slice(),
                        )
                        .expect("finite exact-boundary FIEGARCH seed");
                        let zero_objective = tracked_fiegarch_qml_objective(
                            &data,
                            zero_parameters,
                            effective_truncation,
                            &mut overflowed_objectives,
                        );
                        if zero_objective.is_nan() || zero_objective == f64::NEG_INFINITY {
                            ctx.push(
                                Issue::builder(IssueCode::LossIsNan)
                                    .message(
                                        "an exact-boundary FIEGARCH seed produced an invalid objective",
                                    )
                                    .build(),
                            );
                            return Err(ctx.finish_failure());
                        }
                        if zero_objective.is_finite()
                            && best_seeds[FiegarchQmlFace::DZero.index()]
                                .as_ref()
                                .map(|(_, best)| zero_objective < *best)
                                .unwrap_or(true)
                        {
                            best_seeds[FiegarchQmlFace::DZero.index()] =
                                Some((zero_point, zero_objective));
                        }

                        for d in FIEGARCH_D_SEEDS {
                            let d_coordinate =
                                encode_open_interval(d, FIEGARCH_D_LOWER, FIEGARCH_D_UPPER)
                                    .expect("strictly interior FIEGARCH d seed");
                            let point = Vector::from_slice(&[
                                omega,
                                alpha,
                                gamma,
                                beta_coordinate,
                                d_coordinate,
                            ]);
                            let parameters = decode_fiegarch_qml_point(
                                FiegarchQmlFace::Interior,
                                point.as_slice(),
                            )
                            .expect("finite interior FIEGARCH seed");
                            let objective = tracked_fiegarch_qml_objective(
                                &data,
                                parameters,
                                effective_truncation,
                                &mut overflowed_objectives,
                            );
                            if objective.is_nan() || objective == f64::NEG_INFINITY {
                                ctx.push(
                                    Issue::builder(IssueCode::LossIsNan)
                                        .message(
                                            "an interior FIEGARCH seed produced an invalid objective",
                                        )
                                        .build(),
                                );
                                return Err(ctx.finish_failure());
                            }
                            if objective.is_finite()
                                && best_seeds[FiegarchQmlFace::Interior.index()]
                                    .as_ref()
                                    .map(|(_, best)| objective < *best)
                                    .unwrap_or(true)
                            {
                                best_seeds[FiegarchQmlFace::Interior.index()] =
                                    Some((point, objective));
                            }
                        }
                    }
                }
            }
        }
        if best_seeds.iter().all(Option::is_none) {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .severity(Severity::Fatal)
                    .message("every deterministic FIEGARCH Gaussian-QML seed overflowed")
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

        let mut candidates = Vec::with_capacity(FiegarchQmlFace::ALL.len());
        let missing_seed_faces = best_seeds.iter().filter(|seed| seed.is_none()).count();
        let mut max_iteration_faces = 0usize;
        let mut collapsed_faces = 0usize;
        for face in FiegarchQmlFace::ALL {
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
                    let Some(parameters) = decode_fiegarch_qml_point(face, point) else {
                        return f64::INFINITY;
                    };
                    tracked_fiegarch_qml_objective(
                        &data,
                        parameters,
                        effective_truncation,
                        &mut overflowed_objectives,
                    )
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
            if objective.is_finite()
                && admissible_fiegarch_candidate(face, point.as_slice(), boundary_tolerance)
            {
                candidates.push(FiegarchQmlCandidate {
                    face,
                    point,
                    objective,
                });
            }
        }
        let Some(selected) = select_fiegarch_candidate(
            candidates,
            self.optimizer.policy.optimizer_objective_tie_ulps,
        ) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("FIEGARCH Gaussian QMLE did not produce a finite face candidate")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
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
                        "one or more FIEGARCH parameter-face searches were incomplete; the reported fit is the best finite candidate and may come from a partial search",
                    )
                    .metric("missing_seed_faces", missing_seed_faces as f64)
                    .metric("max_iteration_faces", max_iteration_faces as f64)
                    .metric("collapsed_faces", collapsed_faces as f64)
                    .metric(
                        "searched_faces",
                        (FiegarchQmlFace::ALL.len() - missing_seed_faces) as f64,
                    )
                    .metric("total_faces", FiegarchQmlFace::ALL.len() as f64)
                    .build(),
            );
        }

        let selected_face = selected.face;
        let selected_objective = selected.objective;
        let Some(parameters) = decode_fiegarch_qml_point(selected_face, selected.point.as_slice())
        else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("FIEGARCH could not decode the selected optimizer point")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        if !selected_objective.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("FIEGARCH selected a non-finite Gaussian-QML objective")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if overflowed_objectives > 0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .severity(Severity::Advisory)
                    .message(format!(
                        "{overflowed_objectives} dominated FIEGARCH candidate objectives overflowed to positive infinity"
                    ))
                    .metric("overflowed_objectives", overflowed_objectives as f64)
                    .build(),
            );
        }

        let beta_slack = 1.0 - parameters.beta.abs();
        if beta_slack <= boundary_tolerance {
            ctx.push(
                Issue::builder(IssueCode::NonStationary)
                    .message("FIEGARCH beta converged too close to the open causality boundary")
                    .metric("beta", parameters.beta)
                    .metric("causality_slack", beta_slack)
                    .metric("parameter_tolerance", boundary_tolerance)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let d_slack = FIEGARCH_D_UPPER - parameters.d;
        if d_slack <= boundary_tolerance {
            ctx.push(
                Issue::builder(IssueCode::NonStationary)
                    .message(
                        "FIEGARCH d converged too close to the open covariance-stationarity boundary",
                    )
                    .metric("d", parameters.d)
                    .metric("stationarity_slack", d_slack)
                    .metric("parameter_tolerance", boundary_tolerance)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if selected_face == FiegarchQmlFace::DZero || parameters.d <= boundary_tolerance {
            ctx.push(
                Issue::builder(IssueCode::ParameterAtBoundary)
                    .message("FIEGARCH Gaussian QMLE selected or approached the d = 0 EGARCH face")
                    .metric("d", parameters.d)
                    .metric("parameter_tolerance", boundary_tolerance)
                    .build(),
            );
        }
        if parameters.alpha.abs().max(parameters.gamma.abs()) <= boundary_tolerance {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message(
                        "FIEGARCH d is unidentified when both centered-news coefficients are numerically zero",
                    )
                    .metric("alpha", parameters.alpha)
                    .metric("gamma", parameters.gamma)
                    .metric("parameter_tolerance", boundary_tolerance)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if parameters.d > FIEGARCH_D_LOWER && self.truncation < y.len() {
            ctx.push(
                Issue::builder(IssueCode::InfiniteFilterTruncated)
                    .message(
                        "FIEGARCH used the configured finite inverse-fractional lag expansion",
                    )
                    .metric("truncation", self.truncation as f64)
                    .metric("available_history", y.len() as f64)
                    .compromise(
                        NumericalCompromise::new(
                            "evaluate the infinite inverse-fractional FIEGARCH filter",
                            format!(
                                "truncate (1-L)^(-d) after {} coefficients while retaining the AR recursion",
                                self.truncation
                            ),
                            "an infinite distributed lag cannot be evaluated in finite time",
                            "the fitted path is the explicitly finite-filter approximation; long-memory tail effects are omitted",
                        )
                        .violate("the inverse-fractional expansion has infinite support"),
                    )
                    .build(),
            );
        }

        let log_physical_scale_square = 2.0 * data.scale.ln();
        let omega = parameters.mu_normalized + log_physical_scale_square;
        if !omega.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("FIEGARCH physical long-run log-variance level is not representable")
                    .metric("mu_normalized", parameters.mu_normalized)
                    .metric("scale", data.scale)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let Some(path) = fiegarch_qml_path(&data, parameters, effective_truncation) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("FIEGARCH selected parameters produced an invalid log-variance path")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        let log_sigma2 = path
            .log_variances
            .iter()
            .map(|log_variance| log_physical_scale_square + log_variance)
            .collect::<Vec<_>>();
        if log_sigma2
            .iter()
            .any(|log_variance| !log_variance.is_finite())
        {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("FIEGARCH physical log-variance path is not representable")
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
                    .message("FIEGARCH physical conditional variance overflowed")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let underflowed = sigma2.iter().filter(|variance| **variance == 0.0).count();
        if underflowed > 0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalUnderflow)
                    .message(format!(
                        "{underflowed} FIEGARCH conditional variances underflowed to zero"
                    ))
                    .metric("underflowed_variances", underflowed as f64)
                    .compromise(NumericalCompromise::new(
                        "return every positive FIEGARCH conditional variance in binary64",
                        "return zero while preserving every finite physical log variance",
                        "one or more physical-scale exponentials are below binary64 range",
                        "zero entries are representational underflow, not zero process variance",
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedFiegarch {
            omega,
            alpha: parameters.alpha,
            gamma: parameters.gamma,
            beta: parameters.beta,
            d: parameters.d,
            truncation: self.truncation,
            sigma2: Vector::from_iter(sigma2),
            log_sigma2: Vector::from_iter(log_sigma2),
            resid: residuals,
        })
    }
}

#[cfg(test)]
fn fractional_weights_with_derivative(d: f64, length: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let weights = fractional_integration_weights(d, length)?;
    let mut derivatives = vec![0.0; length];
    for lag in 1..length {
        let factor = (lag as f64 - 1.0 + d) / lag as f64;
        derivatives[lag] = derivatives[lag - 1] * factor + weights[lag - 1] / lag as f64;
        if !derivatives[lag].is_finite() {
            return None;
        }
    }
    Some((weights, derivatives))
}

#[cfg(test)]
fn fiegarch_qml_gradient(
    data: &NormalizedLogSquares,
    parameters: FiegarchQmlParameters,
    truncation: usize,
) -> [f64; 5] {
    let effective_truncation = truncation.min(data.values.len());
    let Some((weights, weight_derivatives)) =
        fractional_weights_with_derivative(parameters.d, effective_truncation)
    else {
        return [f64::NAN; 5];
    };
    if data.scale == 0.0
        || data.values.is_empty()
        || data.normalized_values.len() != data.values.len()
        || !data.log_mean_square.is_finite()
    {
        return [f64::NAN; 5];
    }
    let mut log_variance = data.log_mean_square;
    let mut log_variance_derivative = [0.0; 5];
    let mut news = vec![0.0; data.values.len()];
    let mut news_derivatives = vec![[0.0; 5]; data.values.len()];
    let mut gradient = [0.0; 5];
    for time in 0..data.values.len() {
        let standardized_square = if data.values[time] == f64::NEG_INFINITY {
            0.0
        } else {
            (data.values[time] - log_variance).exp()
        };
        let contribution = 0.5 * (1.0 - standardized_square);
        for coordinate in 0..gradient.len() {
            gradient[coordinate] += contribution * log_variance_derivative[coordinate];
        }
        let Some(standardized_residual) = standardized_residual_from_logs(
            data.normalized_values[time],
            data.values[time],
            log_variance,
        ) else {
            return [f64::NAN; 5];
        };
        news[time] =
            egarch_centered_news(standardized_residual, parameters.alpha, parameters.gamma);
        let feedback = -0.5
            * (parameters.alpha * standardized_residual.abs()
                + parameters.gamma * standardized_residual);
        for coordinate in 0..5 {
            news_derivatives[time][coordinate] = feedback * log_variance_derivative[coordinate];
        }
        news_derivatives[time][1] += standardized_residual.abs() - EGARCH_CENTERED_ABSOLUTE_NORMAL;
        news_derivatives[time][2] += standardized_residual;

        if time + 1 < data.values.len() {
            let included = (time + 1).min(weights.len());
            let mut fractional_derivative = [0.0; 5];
            for coordinate in 0..5 {
                let Some(value) = compensated_sum((0..included).map(|lag| {
                    weights[lag] * news_derivatives[time - lag][coordinate]
                        + if coordinate == 4 {
                            weight_derivatives[lag] * news[time - lag]
                        } else {
                            0.0
                        }
                })) else {
                    return [f64::NAN; 5];
                };
                fractional_derivative[coordinate] = value;
            }
            let Some(fractional_news) = compensated_sum(
                weights[..included]
                    .iter()
                    .enumerate()
                    .map(|(lag, weight)| *weight * news[time - lag]),
            ) else {
                return [f64::NAN; 5];
            };
            let next_log_variance = (1.0 - parameters.beta) * parameters.mu_normalized
                + parameters.beta * log_variance
                + fractional_news;
            log_variance_derivative = [
                (1.0 - parameters.beta)
                    + parameters.beta * log_variance_derivative[0]
                    + fractional_derivative[0],
                parameters.beta * log_variance_derivative[1] + fractional_derivative[1],
                parameters.beta * log_variance_derivative[2] + fractional_derivative[2],
                -parameters.mu_normalized
                    + log_variance
                    + parameters.beta * log_variance_derivative[3]
                    + fractional_derivative[3],
                parameters.beta * log_variance_derivative[4] + fractional_derivative[4],
            ];
            log_variance = next_log_variance;
        }
    }
    gradient
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tsa::egarch::{egarch_qml_log_variances, EgarchQmlParameters};

    fn fiegarch_golden() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../golden/fiegarch_qml.json"))
            .expect("golden/fiegarch_qml.json")
    }

    fn decimal_value(value: &serde_json::Value) -> f64 {
        match value.as_str().expect("decimal string") {
            "+inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            text => text.parse::<f64>().expect("binary64 value"),
        }
    }

    fn golden_observations(case: &serde_json::Value) -> Vec<f64> {
        case["observations"]
            .as_array()
            .expect("observations")
            .iter()
            .map(decimal_value)
            .collect()
    }

    fn golden_parameters(case: &serde_json::Value) -> FiegarchQmlParameters {
        let source = if case["outcome"] == "success" {
            &case["fit"]
        } else {
            &case["parameters"]
        };
        FiegarchQmlParameters {
            mu_normalized: decimal_value(&source["mu_normalized"]),
            alpha: decimal_value(&source["alpha"]),
            gamma: decimal_value(&source["gamma"]),
            beta: decimal_value(&source["beta"]),
            d: decimal_value(&source["d"]),
        }
    }

    fn fractional_terms(path: &FiegarchQmlPath) -> Vec<f64> {
        (0..path.news.len())
            .map(|time| {
                let included = (time + 1).min(path.weights.len());
                compensated_sum(
                    path.weights[..included]
                        .iter()
                        .enumerate()
                        .map(|(lag, weight)| *weight * path.news[time - lag]),
                )
                .expect("finite fractional term")
            })
            .collect()
    }

    fn one_step_normalized_log_variance(
        path: &FiegarchQmlPath,
        parameters: FiegarchQmlParameters,
    ) -> f64 {
        let time = path.news.len() - 1;
        let included = (time + 1).min(path.weights.len());
        let filtered = compensated_sum(
            path.weights[..included]
                .iter()
                .enumerate()
                .map(|(lag, weight)| *weight * path.news[time - lag]),
        )
        .expect("finite one-step fractional term");
        (1.0 - parameters.beta) * parameters.mu_normalized
            + parameters.beta * path.log_variances[time]
            + filtered
    }

    fn fixture_residuals() -> [f64; 12] {
        [
            1.0, -2.0, 0.5, -0.25, 1.5, -0.75, 0.1, -0.4, 0.9, -1.2, 0.35, -0.15,
        ]
    }

    fn fixture_parameters() -> FiegarchQmlParameters {
        FiegarchQmlParameters {
            mu_normalized: -0.35,
            alpha: 0.22,
            gamma: -0.13,
            beta: 0.63,
            d: 0.27,
        }
    }

    fn maximum_absolute_difference(left: &[f64], right: &[f64]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn inverse_fractional_weights_have_the_required_positive_hyperbolic_tail() {
        let d = 0.25;
        let weights = fractional_integration_weights(d, 12).expect("fractional weights");
        assert_eq!(weights[0], 1.0);
        assert_eq!(weights[1], d);
        assert!((weights[2] - d * (d + 1.0) / 2.0).abs() <= f64::EPSILON);
        assert!(weights
            .iter()
            .all(|weight| weight.is_finite() && *weight > 0.0));
        assert!(weights.windows(2).all(|pair| pair[1] < pair[0]));

        let mut product = 1.0;
        let mut maximum_error = 0.0_f64;
        for lag in 1..weights.len() {
            product *= (d + lag as f64 - 1.0) / lag as f64;
            maximum_error = maximum_error.max((weights[lag] - product).abs());
        }
        eprintln!("FIEGARCH recurrence/product weight error={maximum_error:e}");
        assert!(maximum_error <= f64::EPSILON);

        let zero = fractional_integration_weights(0.0, 8).expect("d-zero weights");
        assert_eq!(zero[0], 1.0);
        assert!(zero[1..].iter().all(|weight| *weight == 0.0));
        assert!(fractional_integration_weights(-f64::EPSILON, 4).is_none());
        assert!(fractional_integration_weights(0.5, 4).is_none());
    }

    #[test]
    fn fiegarch_path_matches_the_factorized_manual_recurrence_without_floors() {
        let residuals = fixture_residuals();
        let data = normalized_log_squares(&residuals).expect("normalized residuals");
        let parameters = fixture_parameters();
        let path = fiegarch_qml_path(&data, parameters, residuals.len()).expect("FIEGARCH path");
        let weights = fractional_integration_weights(parameters.d, residuals.len())
            .expect("inverse-fractional weights");
        let mut expected_log_variances = vec![data.log_mean_square; residuals.len()];
        let mut expected_news = vec![0.0; residuals.len()];
        for time in 0..residuals.len() {
            let z = data.normalized_values[time] * (-0.5 * expected_log_variances[time]).exp();
            expected_news[time] = parameters.alpha * (z.abs() - EGARCH_CENTERED_ABSOLUTE_NORMAL)
                + parameters.gamma * z;
            if time + 1 < residuals.len() {
                let fractional_news = (0..=time)
                    .map(|lag| weights[lag] * expected_news[time - lag])
                    .sum::<f64>();
                expected_log_variances[time + 1] = (1.0 - parameters.beta)
                    * parameters.mu_normalized
                    + parameters.beta * expected_log_variances[time]
                    + fractional_news;
            }
        }
        let log_error = maximum_absolute_difference(&path.log_variances, &expected_log_variances);
        let news_error = maximum_absolute_difference(&path.news, &expected_news);
        eprintln!("FIEGARCH manual path error: log={log_error:e}, news={news_error:e}");
        // Measured 1.111e-16 (log) and 4.164e-17 (news) on 2026-09-02;
        // tolerances are approximately four times the observed errors.
        assert!(log_error <= 4.45e-16);
        assert!(news_error <= 1.67e-16);
        assert_eq!(path.weights, weights);

        let direct_objective = expected_log_variances
            .iter()
            .zip(&data.normalized_values)
            .map(|(log_variance, residual)| {
                0.5 * (log_variance + residual * residual * (-log_variance).exp())
            })
            .sum::<f64>();
        let objective_error =
            (fiegarch_qml_objective(&data, parameters, residuals.len()) - direct_objective).abs();
        eprintln!("FIEGARCH manual objective error={objective_error:e}");
        // Measured 1.111e-16 on 2026-09-02; tolerance is approximately 4x.
        assert!(objective_error <= 4.45e-16);
    }

    #[test]
    fn d_zero_is_the_verified_egarch_recurrence_for_every_truncation() {
        let residuals = fixture_residuals();
        let data = normalized_log_squares(&residuals).expect("normalized residuals");
        let omega_level = -0.42;
        let alpha = 0.19;
        let gamma = -0.11;
        let beta = 0.71;
        let egarch = EgarchQmlParameters {
            omega_normalized: (1.0 - beta) * omega_level,
            alpha,
            gamma,
            beta,
        };
        let expected = egarch_qml_log_variances(&data, egarch).expect("EGARCH path");
        for truncation in [2, 3, residuals.len()] {
            let parameters = FiegarchQmlParameters {
                mu_normalized: omega_level,
                alpha,
                gamma,
                beta,
                d: 0.0,
            };
            let actual = fiegarch_qml_path(&data, parameters, truncation)
                .expect("d-zero FIEGARCH path")
                .log_variances;
            let maximum_error = maximum_absolute_difference(&actual, &expected);
            eprintln!("FIEGARCH d-zero/EGARCH error at truncation {truncation}={maximum_error:e}");
            // Measured 1.111e-16 on 2026-09-02; tolerance is approximately 4x.
            assert!(maximum_error <= 4.45e-16);
        }
    }

    #[test]
    fn beta_zero_is_the_pure_inverse_fractional_filter() {
        let residuals = fixture_residuals();
        let data = normalized_log_squares(&residuals).expect("normalized residuals");
        let parameters = FiegarchQmlParameters {
            beta: 0.0,
            ..fixture_parameters()
        };
        let path = fiegarch_qml_path(&data, parameters, residuals.len()).expect("beta-zero path");
        let mut maximum_error = 0.0_f64;
        for time in 1..residuals.len() {
            let expected = parameters.mu_normalized
                + (0..time)
                    .map(|lag| path.weights[lag] * path.news[time - 1 - lag])
                    .sum::<f64>();
            maximum_error = maximum_error.max((path.log_variances[time] - expected).abs());
        }
        eprintln!("FIEGARCH beta-zero direct-filter error={maximum_error:e}");
        // Measured 1.111e-16 on 2026-09-02; tolerance is approximately 4x.
        assert!(maximum_error <= 4.45e-16);
    }

    #[test]
    fn domains_zero_news_and_extended_real_barriers_are_explicit() {
        let beta_coordinate = encode_open_interval(-0.6, FIEGARCH_BETA_LOWER, FIEGARCH_BETA_UPPER)
            .expect("negative beta coordinate");
        let d_coordinate = encode_open_interval(0.31, FIEGARCH_D_LOWER, FIEGARCH_D_UPPER)
            .expect("interior d coordinate");
        let decoded = decode_fiegarch_qml_point(
            FiegarchQmlFace::Interior,
            &[-0.2, 0.3, -0.1, beta_coordinate, d_coordinate],
        )
        .expect("interior parameters");
        assert!((decoded.beta + 0.6).abs() <= f64::EPSILON);
        assert!((decoded.d - 0.31).abs() <= f64::EPSILON);
        assert!(decode_fiegarch_qml_point(FiegarchQmlFace::Interior, &[0.0; 4]).is_none());
        assert!(decode_fiegarch_qml_point(FiegarchQmlFace::DZero, &[0.0; 5]).is_none());

        let lower_guard = f64::EPSILON.sqrt();
        let saturated_d_coordinate =
            encode_open_interval(lower_guard / 2.0, FIEGARCH_D_LOWER, FIEGARCH_D_UPPER)
                .expect("saturated interior d coordinate");
        assert!(!admissible_fiegarch_candidate(
            FiegarchQmlFace::Interior,
            &[0.0, 0.1, -0.1, beta_coordinate, saturated_d_coordinate],
            lower_guard,
        ));
        assert!(admissible_fiegarch_candidate(
            FiegarchQmlFace::DZero,
            &[0.0, 0.1, -0.1, beta_coordinate],
            lower_guard,
        ));

        let residuals = fixture_residuals();
        let data = normalized_log_squares(&residuals).expect("normalized residuals");
        let no_news_zero = FiegarchQmlParameters {
            mu_normalized: -0.4,
            alpha: 0.0,
            gamma: 0.0,
            beta: -0.25,
            d: 0.0,
        };
        let no_news_long_memory = FiegarchQmlParameters {
            d: 0.49,
            ..no_news_zero
        };
        let zero_path =
            fiegarch_qml_path(&data, no_news_zero, residuals.len()).expect("zero-news d-zero path");
        let long_memory_path = fiegarch_qml_path(&data, no_news_long_memory, residuals.len())
            .expect("zero-news long-memory path");
        assert_eq!(zero_path.log_variances, long_memory_path.log_variances);

        let barrier = FiegarchQmlParameters {
            mu_normalized: -2_000.0,
            alpha: 0.2,
            gamma: -0.1,
            beta: 0.0,
            d: 0.2,
        };
        let objective = fiegarch_qml_objective(&data, barrier, residuals.len());
        assert_eq!(objective, f64::INFINITY);
        assert!(!objective.is_nan());
    }

    #[test]
    fn fiegarch_analytic_gradient_matches_centered_differences() {
        let residuals = fixture_residuals();
        let data = normalized_log_squares(&residuals).expect("normalized residuals");
        let base = [-0.35_f64, 0.22, -0.13, 0.63, 0.27];
        let parameters = |point: [f64; 5]| FiegarchQmlParameters {
            mu_normalized: point[0],
            alpha: point[1],
            gamma: point[2],
            beta: point[3],
            d: point[4],
        };
        let analytic = fiegarch_qml_gradient(&data, parameters(base), residuals.len());
        let step = f64::EPSILON.cbrt();
        let mut maximum_error = 0.0_f64;
        for coordinate in 0..base.len() {
            let mut plus = base;
            let mut minus = base;
            plus[coordinate] += step;
            minus[coordinate] -= step;
            let numerical = (fiegarch_qml_objective(&data, parameters(plus), residuals.len())
                - fiegarch_qml_objective(&data, parameters(minus), residuals.len()))
                / (2.0 * step);
            maximum_error = maximum_error.max((analytic[coordinate] - numerical).abs());
        }
        eprintln!("FIEGARCH analytic-gradient max error={maximum_error:e}");
        // Measured 2.073e-10 on 2026-09-02; tolerance is approximately 4x.
        assert!(maximum_error <= 8.30e-10);
    }

    #[test]
    fn transformed_gradient_matches_open_beta_and_d_coordinates() {
        let residuals = fixture_residuals();
        let data = normalized_log_squares(&residuals).expect("normalized residuals");
        let physical = fixture_parameters();
        let beta_coordinate =
            encode_open_interval(physical.beta, FIEGARCH_BETA_LOWER, FIEGARCH_BETA_UPPER)
                .expect("beta coordinate");
        let d_coordinate = encode_open_interval(physical.d, FIEGARCH_D_LOWER, FIEGARCH_D_UPPER)
            .expect("d coordinate");
        let point = [
            physical.mu_normalized,
            physical.alpha,
            physical.gamma,
            beta_coordinate,
            d_coordinate,
        ];
        let direct = fiegarch_qml_gradient(&data, physical, residuals.len());
        let beta_jacobian = (physical.beta - FIEGARCH_BETA_LOWER)
            * (FIEGARCH_BETA_UPPER - physical.beta)
            / (FIEGARCH_BETA_UPPER - FIEGARCH_BETA_LOWER);
        let d_jacobian = (physical.d - FIEGARCH_D_LOWER) * (FIEGARCH_D_UPPER - physical.d)
            / (FIEGARCH_D_UPPER - FIEGARCH_D_LOWER);
        let analytic = [
            direct[0],
            direct[1],
            direct[2],
            beta_jacobian * direct[3],
            d_jacobian * direct[4],
        ];
        let step = f64::EPSILON.cbrt();
        let mut maximum_error = 0.0_f64;
        for coordinate in 0..point.len() {
            let mut plus = point;
            let mut minus = point;
            plus[coordinate] += step;
            minus[coordinate] -= step;
            let plus_parameters =
                decode_fiegarch_qml_point(FiegarchQmlFace::Interior, &plus).expect("plus");
            let minus_parameters =
                decode_fiegarch_qml_point(FiegarchQmlFace::Interior, &minus).expect("minus");
            let numerical = (fiegarch_qml_objective(&data, plus_parameters, residuals.len())
                - fiegarch_qml_objective(&data, minus_parameters, residuals.len()))
                / (2.0 * step);
            maximum_error = maximum_error.max((analytic[coordinate] - numerical).abs());
        }
        eprintln!("FIEGARCH transformed-gradient max error={maximum_error:e}");
        // Measured 2.870e-11 on 2026-09-02; tolerance is approximately 4x.
        assert!(maximum_error <= 1.15e-10);
    }

    #[test]
    fn fiegarch_decimal_golden_replays_the_fixed_parameter_kernel() {
        let payload = fiegarch_golden();
        let cases = payload["cases"].as_array().expect("cases");
        assert_eq!(
            cases.len(),
            payload["case_count"].as_u64().expect("case count") as usize
        );
        let mut maximum_mean_error = 0.0_f64;
        let mut maximum_scale_error = 0.0_f64;
        let mut maximum_initial_error = 0.0_f64;
        let mut maximum_objective_error = 0.0_f64;
        let mut maximum_log_variance_error = 0.0_f64;
        let mut maximum_fractional_error = 0.0_f64;
        let mut maximum_variance_relative_error = 0.0_f64;
        let mut maximum_forecast_log_error = 0.0_f64;
        let mut maximum_forecast_relative_error = 0.0_f64;

        for case in cases {
            let name = case["name"].as_str().expect("case name");
            let observations = golden_observations(case);
            let y = Vector::from_slice(&observations);
            let mean = y.mean();
            maximum_mean_error =
                maximum_mean_error.max((mean - decimal_value(&case["mean"])).abs());
            let residuals = observations
                .iter()
                .map(|observation| observation - mean)
                .collect::<Vec<_>>();
            let data = normalized_log_squares(&residuals).expect("golden normalization");
            maximum_scale_error =
                maximum_scale_error.max((data.scale - decimal_value(&case["scale"])).abs());

            match case["outcome"].as_str().expect("outcome") {
                "unidentified_constant_series" => {
                    assert_eq!(data.scale, 0.0, "{name}");
                }
                "extended_real_probe" => {
                    maximum_initial_error = maximum_initial_error.max(
                        (data.log_mean_square - decimal_value(&case["initial_log_variance"])).abs(),
                    );
                    let parameters = golden_parameters(case);
                    let truncation = case["truncation"].as_u64().expect("truncation") as usize;
                    let path = fiegarch_qml_path(&data, parameters, truncation)
                        .expect("the Decimal recurrence remains representable in binary64");
                    let objective = fiegarch_qml_objective(&data, parameters, truncation);
                    assert_eq!(objective, f64::INFINITY, "{name}");
                    assert!(!objective.is_nan(), "{name}");
                    assert!(case["decimal_objective_is_finite"]
                        .as_bool()
                        .expect("Decimal finiteness"));
                    let forecast = one_step_normalized_log_variance(&path, parameters);
                    assert!(forecast.is_finite(), "{name}");
                    assert_eq!(
                        (2.0 * data.scale.ln() + forecast).exp(),
                        f64::INFINITY,
                        "{name}"
                    );
                }
                "success" | "parameter_probe" => {
                    maximum_initial_error = maximum_initial_error.max(
                        (data.log_mean_square - decimal_value(&case["initial_log_variance"])).abs(),
                    );
                    let parameters = golden_parameters(case);
                    let truncation = case["truncation"].as_u64().expect("truncation") as usize;
                    let path = fiegarch_qml_path(&data, parameters, truncation)
                        .expect("finite golden FIEGARCH path");
                    let expected_objective = if case["outcome"] == "success" {
                        decimal_value(&case["fit"]["objective"])
                    } else {
                        decimal_value(&case["decimal_objective"])
                    };
                    maximum_objective_error = maximum_objective_error.max(
                        (fiegarch_qml_objective(&data, parameters, truncation)
                            - expected_objective)
                            .abs(),
                    );
                    let expected_log_variances = case["normalized_log_variances"]
                        .as_array()
                        .expect("normalized log variances");
                    assert_eq!(
                        path.log_variances.len(),
                        expected_log_variances.len(),
                        "{name}"
                    );
                    for (actual, expected) in path.log_variances.iter().zip(expected_log_variances)
                    {
                        maximum_log_variance_error = maximum_log_variance_error
                            .max((actual - decimal_value(expected)).abs());
                    }

                    if let Some(expected_terms) = case["fractional_terms"].as_array() {
                        let actual_terms = fractional_terms(&path);
                        assert_eq!(actual_terms.len(), expected_terms.len(), "{name}");
                        for (actual, expected) in actual_terms.iter().zip(expected_terms) {
                            maximum_fractional_error = maximum_fractional_error
                                .max((actual - decimal_value(expected)).abs());
                        }
                    }
                    if let Some(expected_variances) = case["physical_variances"].as_array() {
                        assert_eq!(path.log_variances.len(), expected_variances.len(), "{name}");
                        let log_scale_square = 2.0 * data.scale.ln();
                        for (actual_log, expected) in
                            path.log_variances.iter().zip(expected_variances)
                        {
                            let actual = (log_scale_square + actual_log).exp();
                            maximum_variance_relative_error = maximum_variance_relative_error
                                .max((actual / decimal_value(expected) - 1.0).abs());
                        }
                    }

                    let forecast = one_step_normalized_log_variance(&path, parameters);
                    maximum_forecast_log_error = maximum_forecast_log_error.max(
                        (forecast
                            - decimal_value(&case["one_step_forecast"]["normalized_log_variance"]))
                        .abs(),
                    );
                    let actual_physical_forecast = (2.0 * data.scale.ln() + forecast).exp();
                    let expected_physical_forecast =
                        decimal_value(&case["one_step_forecast"]["physical_variance"]);
                    maximum_forecast_relative_error = maximum_forecast_relative_error
                        .max((actual_physical_forecast / expected_physical_forecast - 1.0).abs());
                }
                outcome => panic!("unknown FIEGARCH golden outcome {outcome}"),
            }
        }

        eprintln!(
            "FIEGARCH Decimal kernel max errors: mean={maximum_mean_error:e}, scale={maximum_scale_error:e}, initial={maximum_initial_error:e}, objective={maximum_objective_error:e}, log-variance={maximum_log_variance_error:e}, fractional={maximum_fractional_error:e}, variance-relative={maximum_variance_relative_error:e}, forecast-log={maximum_forecast_log_error:e}, forecast-relative={maximum_forecast_relative_error:e}"
        );
        // Measured on 2026-09-02: mean 1.106e-17, scale 0, initial
        // 4.441e-16, objective 2.842e-14, log path 8.882e-16,
        // fractional term 4.441e-16, variance relative 8.882e-16,
        // forecast log 4.441e-16, forecast relative 2.220e-16.
        // Nonzero tolerances below are approximately four times those maxima.
        assert!(maximum_mean_error <= 4.43e-17);
        assert_eq!(maximum_scale_error, 0.0);
        assert!(maximum_initial_error <= 1.78e-15);
        assert!(maximum_objective_error <= 1.14e-13);
        assert!(maximum_log_variance_error <= 3.56e-15);
        assert!(maximum_fractional_error <= 1.78e-15);
        assert!(maximum_variance_relative_error <= 3.56e-15);
        assert!(maximum_forecast_log_error <= 1.78e-15);
        assert!(maximum_forecast_relative_error <= 8.89e-16);
    }

    #[test]
    fn fiegarch_analytic_gradient_matches_decimal_nonstationary_probes() {
        let payload = fiegarch_golden();
        let cases = payload["cases"].as_array().expect("cases");
        let probes = payload["cross_case_checks"]["nonstationary_gradient_probes"]
            .as_object()
            .expect("gradient probes");
        let mut maximum_error = 0.0_f64;
        for (probe_name, probe) in probes {
            let series = probe["series"].as_str().expect("probe series");
            let case = cases
                .iter()
                .find(|case| case["name"] == series)
                .unwrap_or_else(|| panic!("missing probe series {series}"));
            let observations = golden_observations(case);
            let y = Vector::from_slice(&observations);
            let mean = y.mean();
            let residuals = observations
                .iter()
                .map(|observation| observation - mean)
                .collect::<Vec<_>>();
            let data = normalized_log_squares(&residuals).expect("probe normalization");
            let source = &probe["parameters"];
            let parameters = FiegarchQmlParameters {
                mu_normalized: decimal_value(&source["mu"]),
                alpha: decimal_value(&source["alpha"]),
                gamma: decimal_value(&source["gamma"]),
                beta: decimal_value(&source["beta"]),
                d: decimal_value(&source["d"]),
            };
            let actual = fiegarch_qml_gradient(
                &data,
                parameters,
                probe["truncation"].as_u64().expect("probe truncation") as usize,
            );
            let expected = &probe["physical_gradient"];
            let recurrence_intercept_gradient = decimal_value(&expected["omega"]);
            let expected_long_run_gradient = [
                (1.0 - parameters.beta) * recurrence_intercept_gradient,
                decimal_value(&expected["alpha"]),
                decimal_value(&expected["gamma"]),
                decimal_value(&expected["beta"])
                    - parameters.mu_normalized * recurrence_intercept_gradient,
                decimal_value(&expected["d"]),
            ];
            for (actual, expected) in actual.into_iter().zip(expected_long_run_gradient) {
                maximum_error = maximum_error.max((actual - expected).abs());
            }
            assert!(actual.iter().all(|value| value.is_finite()), "{probe_name}");
        }
        eprintln!("FIEGARCH Decimal nonstationary-gradient max error={maximum_error:e}");
        // Measured 1.066e-14 on 2026-09-02; tolerance is approximately 4x.
        assert!(maximum_error <= 4.27e-14);
    }

    #[test]
    fn fiegarch_qmle_matches_decimal_faces_and_paths() {
        let payload = fiegarch_golden();
        let cases = payload["cases"].as_array().expect("cases");
        let mut maximum_coefficient_error = 0.0_f64;
        let mut maximum_mu_error = 0.0_f64;
        let mut maximum_objective_error = 0.0_f64;
        let mut maximum_log_variance_error = 0.0_f64;
        let mut maximum_variance_relative_error = 0.0_f64;
        let mut maximum_forecast_relative_error = 0.0_f64;

        for case in cases.iter().filter(|case| case["outcome"] == "success") {
            let name = case["name"].as_str().expect("case name");
            let observations = golden_observations(case);
            let truncation = case["truncation"].as_u64().expect("truncation") as usize;
            let mut model = Fiegarch::new();
            model.truncation = truncation;
            let fitted = model
                .fit_series(
                    &Vector::from_slice(&observations),
                    &Session::new("fiegarch-decimal", name),
                )
                .expect("oracle expects an attained FIEGARCH fit");
            let expected = &case["fit"];
            maximum_coefficient_error = maximum_coefficient_error
                .max((fitted.value.alpha - decimal_value(&expected["alpha"])).abs())
                .max((fitted.value.gamma - decimal_value(&expected["gamma"])).abs())
                .max((fitted.value.beta - decimal_value(&expected["beta"])).abs())
                .max((fitted.value.d - decimal_value(&expected["d"])).abs());
            maximum_mu_error = maximum_mu_error
                .max((fitted.value.omega - decimal_value(&expected["mu_physical"])).abs());

            let data = normalized_log_squares(fitted.value.resid.as_slice())
                .expect("fitted residual normalization");
            let parameters = FiegarchQmlParameters {
                mu_normalized: fitted.value.omega - 2.0 * data.scale.ln(),
                alpha: fitted.value.alpha,
                gamma: fitted.value.gamma,
                beta: fitted.value.beta,
                d: fitted.value.d,
            };
            let path =
                fiegarch_qml_path(&data, parameters, truncation).expect("fitted FIEGARCH path");
            maximum_objective_error = maximum_objective_error.max(
                (fiegarch_qml_objective(&data, parameters, truncation)
                    - decimal_value(&expected["objective"]))
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
            let forecast = one_step_normalized_log_variance(&path, parameters);
            let physical_forecast = (2.0 * data.scale.ln() + forecast).exp();
            maximum_forecast_relative_error = maximum_forecast_relative_error.max(
                (physical_forecast
                    / decimal_value(&case["one_step_forecast"]["physical_variance"])
                    - 1.0)
                    .abs(),
            );

            match expected["selection"].as_str().expect("selection") {
                "d_zero" => {
                    assert_eq!(fitted.value.d, 0.0, "{name}");
                    assert_eq!(fitted.report.n_parameters, Some(5), "{name}");
                    assert!(
                        fitted.report.contains(IssueCode::ParameterAtBoundary),
                        "{name}"
                    );
                    assert!(
                        !fitted.report.contains(IssueCode::InfiniteFilterTruncated),
                        "{name}"
                    );
                }
                "interior" => {
                    assert!(fitted.value.d > 0.0 && fitted.value.d < 0.5, "{name}");
                    assert_eq!(fitted.report.n_parameters, Some(6), "{name}");
                    assert!(
                        !fitted.report.contains(IssueCode::ParameterAtBoundary),
                        "{name}"
                    );
                    assert!(
                        fitted.report.contains(IssueCode::InfiniteFilterTruncated),
                        "{name}"
                    );
                }
                selection => panic!("unknown FIEGARCH selection {selection}"),
            }
            assert_eq!(fitted.value.truncation, truncation, "{name}");
        }

        let constant = cases
            .iter()
            .find(|case| case["outcome"] == "unidentified_constant_series")
            .expect("constant case");
        let constant_failure = Fiegarch::new()
            .fit_series(
                &Vector::from_slice(&golden_observations(constant)),
                &Session::new("fiegarch-decimal", "constant"),
            )
            .unwrap_err();
        assert_eq!(constant_failure.primary.code, IssueCode::UnidentifiedModel);

        eprintln!(
            "FIEGARCH Decimal fit max errors: coefficient={maximum_coefficient_error:e}, mu={maximum_mu_error:e}, objective={maximum_objective_error:e}, log-variance={maximum_log_variance_error:e}, variance-relative={maximum_variance_relative_error:e}, forecast-relative={maximum_forecast_relative_error:e}"
        );
        // Measured on 2026-09-02: coefficient 9.505e-8, physical long-run
        // level 1.121e-8, objective 2.132e-14, physical log path 8.696e-8,
        // variance relative 8.696e-8, forecast relative 6.003e-8.
        // Tolerances below are approximately four times those maxima.
        assert!(maximum_coefficient_error <= 3.81e-7);
        assert!(maximum_mu_error <= 4.49e-8);
        assert!(maximum_objective_error <= 8.53e-14);
        assert!(maximum_log_variance_error <= 3.48e-7);
        assert!(maximum_variance_relative_error <= 3.48e-7);
        assert!(maximum_forecast_relative_error <= 2.41e-7);
    }

    #[test]
    fn reflection_scale_and_finite_history_invariants_hold() {
        let residuals = fixture_residuals();
        let reflected = residuals.map(|value| -value);
        let scaled = residuals.map(|value| value * 16.0);
        let data = normalized_log_squares(&residuals).expect("normalized residuals");
        let reflected_data = normalized_log_squares(&reflected).expect("reflected residuals");
        let scaled_data = normalized_log_squares(&scaled).expect("scaled residuals");
        let parameters = fixture_parameters();
        let reflected_parameters = FiegarchQmlParameters {
            gamma: -parameters.gamma,
            ..parameters
        };
        let original = fiegarch_qml_path(&data, parameters, residuals.len()).expect("path");
        let reflected_path =
            fiegarch_qml_path(&reflected_data, reflected_parameters, reflected.len())
                .expect("reflected path");
        let scaled_path =
            fiegarch_qml_path(&scaled_data, parameters, scaled.len()).expect("scaled path");
        let reflection_error =
            maximum_absolute_difference(&original.log_variances, &reflected_path.log_variances);
        let scale_error =
            maximum_absolute_difference(&original.log_variances, &scaled_path.log_variances);
        eprintln!(
            "FIEGARCH invariant errors: reflection={reflection_error:e}, scale={scale_error:e}"
        );
        assert_eq!(reflection_error, 0.0);
        // Measured 1.111e-16 on 2026-09-02; tolerance is approximately 4x.
        assert!(scale_error <= 4.45e-16);

        let mut future_reflected = residuals;
        for value in &mut future_reflected[7..] {
            *value = -*value;
        }
        let future_data =
            normalized_log_squares(&future_reflected).expect("future-reflected residuals");
        let future_path =
            fiegarch_qml_path(&future_data, parameters, residuals.len()).expect("future path");
        assert_eq!(
            &original.log_variances[..=7],
            &future_path.log_variances[..=7]
        );

        let short = fiegarch_qml_path(&data, parameters, 3).expect("short truncation");
        let long = fiegarch_qml_path(&data, parameters, 9).expect("long truncation");
        assert_eq!(&short.log_variances[..=3], &long.log_variances[..=3]);
    }

    #[test]
    fn configuration_data_and_sample_budget_fail_before_optimization() {
        let session = Session::new("fiegarch-validation", "fit");
        let mut model = Fiegarch::new();
        assert!(model.fit_series(&Vector::zeros(0), &session).is_err());
        assert!(model
            .fit_series(&Vector::from_slice(&[0.0; 7]), &session)
            .is_err());
        assert!(model
            .fit_series(
                &Vector::from_slice(&[0.0, 1.0, f64::NAN, 2.0, 3.0, 4.0, 5.0]),
                &session,
            )
            .is_err());
        let short = Vector::from_iter((0..29).map(|index| {
            let x = index as f64;
            (0.31 * x).sin() + 0.2 * (0.17 * x).cos()
        }));
        assert!(model.fit_series(&short, &session).is_err());

        model.truncation = 1;
        let enough = Vector::from_iter((0..30).map(|index| {
            let x = index as f64;
            (0.31 * x).sin() + 0.2 * (0.17 * x).cos()
        }));
        assert!(model.fit_series(&enough, &session).is_err());
        model.truncation = FIEGARCH_DEFAULT_TRUNCATION;
        model.simplex_step = f64::NAN;
        assert!(model.fit_series(&enough, &session).is_err());
        model.simplex_step = FIEGARCH_DEFAULT_SIMPLEX_STEP;
        model.optimizer.policy.model_parameter_tol = FIEGARCH_D_UPPER;
        assert!(model.fit_series(&enough, &session).is_err());
        model.optimizer.policy.model_parameter_tol = f64::EPSILON.sqrt();
        model.truncation = usize::MAX;
        assert!(model.fit_series(&enough, &session).is_err());
    }

    #[test]
    fn exact_d_zero_avoids_a_false_truncation_warning_during_partial_search() {
        let observations = Vector::from_iter((0..48).map(|index| {
            let x = index as f64;
            0.8 * (0.37 * x).sin() + 0.35 * (0.11 * x).cos() + 0.07 * ((index % 5) as f64 - 2.0)
        }));
        let mut model = Fiegarch::new();
        model.truncation = 5;
        model.optimizer.max_iterations = 1;
        let fitted = model
            .fit_series(&observations, &Session::new("fiegarch-truncated", "fit"))
            .expect("finite seed fit");
        assert_eq!(fitted.value.d, 0.0);
        assert!(!fitted.report.contains(IssueCode::InfiniteFilterTruncated));
        assert!(fitted.report.contains(IssueCode::MaxIterReached));
        assert!(matches!(fitted.report.n_parameters, Some(5 | 6)));
        assert_eq!(fitted.value.truncation, 5);
        assert_eq!(fitted.value.sigma2.len(), observations.len());
        assert_eq!(fitted.value.log_sigma2.len(), observations.len());
        assert!(fitted
            .value
            .log_sigma2
            .as_slice()
            .iter()
            .all(|value| value.is_finite()));
        assert!(fitted
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }
}
