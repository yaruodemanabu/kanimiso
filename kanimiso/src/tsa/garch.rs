//! Fixed-order GARCH(1,1) Gaussian QMLE.

use super::common::{
    gaussian_qml_profile, inspect_scale_invariant_univariate, normalized_log_squares,
    select_ranked_objective_candidate, NormalizedLogSquares,
};
use crate::context::FitCtx;
use crate::data::Vector;
use crate::optimize::NelderMead;
use crate::special::logsumexp;
use crate::traits::FitSeries;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, NumericalCompromise, Qualified, Result, Severity};

const GARCH_PERSISTENCE_SEEDS: [f64; 7] = [0.05, 0.25, 0.5, 0.75, 0.9, 0.97, 0.99];
const GARCH_ARCH_SHARE_SEEDS: [f64; 5] = [0.1, 0.25, 0.5, 0.75, 0.9];
const GARCH_OMEGA_MULTIPLIER_SEEDS: [f64; 7] = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0];

#[derive(Clone, Copy, Debug)]
struct GarchQmlParameters {
    log_omega_normalized: f64,
    alpha: f64,
    beta: f64,
    slack: f64,
    log_alpha: f64,
    log_beta: f64,
}

fn log_expm1_nonnegative(value: f64) -> Option<f64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    if value == 0.0 {
        return Some(f64::NEG_INFINITY);
    }
    if value <= std::f64::consts::LN_2 {
        Some(value.exp_m1().ln())
    } else {
        Some(value + (-(-value).exp()).ln_1p())
    }
}

fn decode_garch_qml_point(point: &[f64]) -> Option<GarchQmlParameters> {
    if point.len() != 3 || !point[0].is_finite() {
        return None;
    }
    let log_arch_weight = log_expm1_nonnegative(point[1])?;
    let log_garch_weight = log_expm1_nonnegative(point[2])?;
    let log_denominator = logsumexp(&[0.0, log_arch_weight, log_garch_weight]);
    if !log_denominator.is_finite() {
        return None;
    }
    let log_alpha = log_arch_weight - log_denominator;
    let log_beta = log_garch_weight - log_denominator;
    let alpha = log_alpha.exp();
    let beta = log_beta.exp();
    let coefficient_sum = alpha + beta;
    if !alpha.is_finite()
        || !beta.is_finite()
        || !(alpha >= 0.0 && beta >= 0.0 && coefficient_sum < 1.0)
    {
        return None;
    }
    let slack = 1.0 - coefficient_sum;
    if slack <= 0.0 || !slack.is_finite() {
        return None;
    }
    Some(GarchQmlParameters {
        log_omega_normalized: point[0],
        alpha,
        beta,
        slack,
        log_alpha,
        log_beta,
    })
}

fn encode_garch_coefficients(alpha: f64, beta: f64) -> Option<[f64; 2]> {
    let slack = 1.0 - alpha - beta;
    if !alpha.is_finite() || !beta.is_finite() || !(alpha >= 0.0 && beta >= 0.0 && slack > 0.0) {
        return None;
    }
    let arch_coordinate = (alpha / slack).ln_1p();
    let garch_coordinate = (beta / slack).ln_1p();
    (arch_coordinate.is_finite() && garch_coordinate.is_finite())
        .then_some([arch_coordinate, garch_coordinate])
}

fn garch_qml_log_variances(
    data: &NormalizedLogSquares,
    parameters: GarchQmlParameters,
) -> Option<Vec<f64>> {
    if data.scale == 0.0 || data.values.is_empty() || !data.log_mean_square.is_finite() {
        return None;
    }
    let mut log_variances = vec![data.log_mean_square; data.values.len()];
    for time in 1..data.values.len() {
        log_variances[time] = logsumexp(&[
            parameters.log_omega_normalized,
            parameters.log_alpha + data.values[time - 1],
            parameters.log_beta + log_variances[time - 1],
        ]);
        if !log_variances[time].is_finite() {
            return None;
        }
    }
    Some(log_variances)
}

fn garch_qml_objective(data: &NormalizedLogSquares, parameters: GarchQmlParameters) -> f64 {
    let Some(log_variances) = garch_qml_log_variances(data, parameters) else {
        return f64::NAN;
    };
    gaussian_qml_profile(&data.values, &log_variances)
}

#[cfg(test)]
fn garch_qml_gradient(data: &NormalizedLogSquares, parameters: GarchQmlParameters) -> [f64; 3] {
    let Some(log_variances) = garch_qml_log_variances(data, parameters) else {
        return [f64::NAN; 3];
    };
    let mut relative_derivative = [0.0; 3];
    let mut gradient = [0.0; 3];
    for time in 0..log_variances.len() {
        let standardized_square = if data.values[time] == f64::NEG_INFINITY {
            0.0
        } else {
            (data.values[time] - log_variances[time]).exp()
        };
        let contribution = 0.5 * (1.0 - standardized_square);
        for coordinate in 0..3 {
            gradient[coordinate] += contribution * relative_derivative[coordinate];
        }
        if time + 1 < log_variances.len() {
            let next_log_variance = log_variances[time + 1];
            let previous_variance_weight =
                (parameters.log_beta + log_variances[time] - next_log_variance).exp();
            relative_derivative = [
                (-next_log_variance).exp() + previous_variance_weight * relative_derivative[0],
                if data.values[time] == f64::NEG_INFINITY {
                    0.0
                } else {
                    (data.values[time] - next_log_variance).exp()
                } + previous_variance_weight * relative_derivative[1],
                (log_variances[time] - next_log_variance).exp()
                    + previous_variance_weight * relative_derivative[2],
            ];
        }
    }
    gradient
}

fn tracked_garch_qml_objective(
    data: &NormalizedLogSquares,
    parameters: GarchQmlParameters,
    overflowed: &mut usize,
) -> f64 {
    let objective = garch_qml_objective(data, parameters);
    if objective == f64::INFINITY {
        *overflowed += 1;
    }
    objective
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GarchQmlFace {
    Interior,
    AlphaZero,
    BetaZero,
    Corner,
}

impl GarchQmlFace {
    const ALL: [Self; 4] = [
        Self::Interior,
        Self::AlphaZero,
        Self::BetaZero,
        Self::Corner,
    ];

    fn for_coefficients(alpha: f64, beta: f64) -> Self {
        match (alpha == 0.0, beta == 0.0) {
            (false, false) => Self::Interior,
            (true, false) => Self::AlphaZero,
            (false, true) => Self::BetaZero,
            (true, true) => Self::Corner,
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Interior => 0,
            Self::AlphaZero => 1,
            Self::BetaZero => 2,
            Self::Corner => 3,
        }
    }

    fn dimension(self) -> usize {
        match self {
            Self::Interior => 3,
            Self::AlphaZero | Self::BetaZero => 2,
            Self::Corner => 1,
        }
    }

    fn selection_rank(self) -> usize {
        match self {
            Self::Corner => 0,
            Self::AlphaZero => 1,
            Self::BetaZero => 2,
            Self::Interior => 3,
        }
    }

    fn session_name(self) -> &'static str {
        match self {
            Self::Interior => "qml-interior",
            Self::AlphaZero => "qml-alpha-zero",
            Self::BetaZero => "qml-beta-zero",
            Self::Corner => "qml-corner",
        }
    }

    fn project(self, full_point: &[f64]) -> Vector {
        debug_assert_eq!(full_point.len(), 3);
        match self {
            Self::Interior => Vector::from_slice(full_point),
            Self::AlphaZero => Vector::from_slice(&[full_point[0], full_point[2]]),
            Self::BetaZero => Vector::from_slice(&[full_point[0], full_point[1]]),
            Self::Corner => Vector::from_slice(&[full_point[0]]),
        }
    }

    fn expand(self, face_point: &[f64]) -> Option<[f64; 3]> {
        match (self, face_point) {
            (Self::Interior, [log_omega, arch, garch]) => Some([*log_omega, *arch, *garch]),
            (Self::AlphaZero, [log_omega, garch]) => Some([*log_omega, 0.0, *garch]),
            (Self::BetaZero, [log_omega, arch]) => Some([*log_omega, *arch, 0.0]),
            (Self::Corner, [log_omega]) => Some([*log_omega, 0.0, 0.0]),
            _ => None,
        }
    }
}

struct GarchQmlCandidate {
    face: GarchQmlFace,
    point: Vector,
    objective: f64,
}

fn select_garch_candidate(
    candidates: Vec<GarchQmlCandidate>,
    objective_tie_ulps: usize,
) -> Option<GarchQmlCandidate> {
    select_ranked_objective_candidate(
        candidates,
        objective_tie_ulps,
        |candidate| candidate.objective,
        |candidate| candidate.face.selection_rank(),
    )
}

/// GARCH(1,1) specification (Gaussian QMLE on a demeaned series).
///
/// Optimization is performed on max-absolute normalized residuals in the log
/// domain. Runtime coordinates guarantee `ω > 0`, `α >= 0`, `β >= 0`, and
/// strict covariance stationarity `α + β < 1` without floors or clamps.
/// A deterministic persistence/share/scale basin grid seeds separate
/// Nelder–Mead searches over the interior, the two zero-coefficient faces, and
/// their corner. This is a reproducible initialization heuristic, not a global
/// optimality certificate.
#[derive(Clone, Debug)]
pub struct Garch11 {
    /// Shared derivative-free solver and numerical-quality policy.
    pub optimizer: NelderMead,
    /// Dimensionless edge length of the deterministic initial simplex.
    pub simplex_step: f64,
}

impl Default for Garch11 {
    fn default() -> Self {
        Self {
            optimizer: NelderMead::default(),
            simplex_step: 0.25,
        }
    }
}

impl Garch11 {
    /// Default QMLE settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted GARCH(1,1) variance recursion.
#[derive(Clone, Debug)]
pub struct FittedGarch11 {
    /// ω.
    pub omega: f64,
    /// ARCH coefficient.
    pub alpha: f64,
    /// GARCH coefficient.
    pub beta: f64,
    /// In-sample conditional variances.
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

impl FittedGarch11 {
    /// Iterate the variance recursion `h` steps (using `E[ε²]=σ²`).
    pub fn forecast_variance(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let last = self.sigma2.as_slice().last().copied().unwrap_or(self.omega);
        let last_e2 = self.resid.as_slice().last().copied().unwrap_or(0.0).powi(2);
        let mut s = self.omega + self.alpha * last_e2 + self.beta * last;
        let mut out = Vector::zeros(h);
        for i in 0..h {
            out[i] = s;
            s = self.omega + (self.alpha + self.beta) * s;
        }
        ctx.finish(out)
    }
}

impl FitSeries for Garch11 {
    type Fitted = FittedGarch11;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedGarch11>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.policy = self.optimizer.policy.clone();
        inspect_scale_invariant_univariate(&mut ctx, y);
        if y.is_empty() || y.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(ctx.finish_failure());
        }
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("GARCH(1,1) QMLE needs a longer series")
                    .metric("n", y.len() as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if !self.simplex_step.is_finite() || self.simplex_step <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("GARCH simplex_step must be finite and positive")
                    .metric("simplex_step", self.simplex_step)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let boundary_tolerance = self.optimizer.policy.model_parameter_tol;
        if !boundary_tolerance.is_finite() || boundary_tolerance <= 0.0 || boundary_tolerance >= 1.0
        {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message(
                        "Policy::model_parameter_tol must be finite, positive, and smaller than one for GARCH",
                    )
                    .metric("model_parameter_tol", boundary_tolerance)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        if !mean.is_finite() || e.as_slice().iter().any(|value| !value.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("GARCH demeaning produced a non-finite residual")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let Some(data) = normalized_log_squares(e.as_slice()) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("GARCH residual normalization failed")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        if data.scale == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message(
                        "GARCH parameters are unidentified because every demeaned residual is zero",
                    )
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let mut overflowed_objectives = 0usize;
        let mut coefficient_seeds = Vec::with_capacity(
            GARCH_PERSISTENCE_SEEDS.len() * (GARCH_ARCH_SHARE_SEEDS.len() + 2) + 1,
        );
        for persistence in GARCH_PERSISTENCE_SEEDS {
            for arch_share in GARCH_ARCH_SHARE_SEEDS {
                coefficient_seeds
                    .push((persistence * arch_share, persistence * (1.0 - arch_share)));
            }
            coefficient_seeds.push((0.0, persistence));
            coefficient_seeds.push((persistence, 0.0));
        }
        coefficient_seeds.push((0.0, 0.0));

        let mut best_seeds: [Option<(Vector, f64)>; 4] = std::array::from_fn(|_| None);
        for (alpha, beta) in coefficient_seeds {
            let face = GarchQmlFace::for_coefficients(alpha, beta);
            let [arch_coordinate, garch_coordinate] =
                encode_garch_coefficients(alpha, beta).expect("valid GARCH seed coefficients");
            let log_target_omega = data.log_mean_square + (1.0 - alpha - beta).ln();
            for omega_multiplier in GARCH_OMEGA_MULTIPLIER_SEEDS {
                let point = Vector::from_slice(&[
                    log_target_omega + omega_multiplier.ln(),
                    arch_coordinate,
                    garch_coordinate,
                ]);
                let parameters = decode_garch_qml_point(point.as_slice())
                    .expect("encoded GARCH seed must decode");
                let objective =
                    tracked_garch_qml_objective(&data, parameters, &mut overflowed_objectives);
                if objective.is_nan() || objective == f64::NEG_INFINITY {
                    ctx.push(
                        Issue::builder(IssueCode::LossIsNan)
                            .message(
                                "a deterministic GARCH QMLE seed produced an invalid objective",
                            )
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                }
                if objective.is_finite()
                    && best_seeds[face.index()]
                        .as_ref()
                        .map(|(_, best)| objective < *best)
                        .unwrap_or(true)
                {
                    best_seeds[face.index()] = Some((point, objective));
                }
            }
        }
        if best_seeds.iter().all(Option::is_none) {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .severity(Severity::Fatal)
                    .message("every deterministic GARCH QMLE seed overflowed")
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

        let mut candidates = Vec::with_capacity(GarchQmlFace::ALL.len());
        let missing_seed_faces = best_seeds.iter().filter(|seed| seed.is_none()).count();
        let mut max_iteration_faces = 0usize;
        let mut collapsed_faces = 0usize;
        for face in GarchQmlFace::ALL {
            let Some((full_seed_point, seed_objective)) = best_seeds[face.index()].take() else {
                continue;
            };
            let face_seed_point = face.project(full_seed_point.as_slice());
            let mut simplex = Vec::with_capacity(face_seed_point.len() + 1);
            simplex.push(face_seed_point.clone());
            for coordinate in 0..face_seed_point.len() {
                let mut vertex = face_seed_point.as_slice().to_vec();
                vertex[coordinate] += self.simplex_step;
                simplex.push(Vector::from_iter(vertex));
            }
            let optimized = match self.optimizer.minimize_nested(
                &simplex,
                |face_point| {
                    let Some(full_point) = face.expand(face_point) else {
                        return f64::INFINITY;
                    };
                    let Some(parameters) = decode_garch_qml_point(&full_point) else {
                        return f64::INFINITY;
                    };
                    tracked_garch_qml_objective(&data, parameters, &mut overflowed_objectives)
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
                (full_seed_point, seed_objective)
            } else {
                let Some(full_point) = face.expand(optimization.point.as_slice()) else {
                    ctx.push(
                        Issue::builder(IssueCode::NonFiniteOutput)
                            .message("GARCH QMLE returned a point with the wrong face dimension")
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                };
                (Vector::from_slice(&full_point), optimization.value)
            };
            candidates.push(GarchQmlCandidate {
                face,
                point,
                objective,
            });
        }
        let Some(selected) = select_garch_candidate(
            candidates,
            self.optimizer.policy.optimizer_objective_tie_ulps,
        ) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("GARCH QMLE did not produce a candidate from any parameter face")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        ctx.report.set_n_parameters(selected.face.dimension());
        if missing_seed_faces > 0 || max_iteration_faces > 0 || collapsed_faces > 0 {
            let code = match (missing_seed_faces, max_iteration_faces, collapsed_faces) {
                (0, positive, 0) if positive > 0 => IssueCode::MaxIterReached,
                (0, 0, positive) if positive > 0 => IssueCode::StepSizeCollapsed,
                _ => IssueCode::DidNotConverge,
            };
            ctx.push(
                Issue::builder(code)
                    .message(
                        "one or more GARCH parameter-face searches were incomplete; the reported fit is the best completed candidate",
                    )
                    .metric("missing_seed_faces", missing_seed_faces as f64)
                    .metric("max_iteration_faces", max_iteration_faces as f64)
                    .metric("collapsed_faces", collapsed_faces as f64)
                    .metric(
                        "searched_faces",
                        (GarchQmlFace::ALL.len() - missing_seed_faces) as f64,
                    )
                    .metric("total_faces", GarchQmlFace::ALL.len() as f64)
                    .build(),
            );
        }
        let selected_point = selected.point;
        let selected_objective = selected.objective;
        let Some(parameters) = decode_garch_qml_point(selected_point.as_slice()) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("GARCH QMLE could not decode the selected optimizer point")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        if !selected_objective.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("GARCH QMLE selected a non-finite objective")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if overflowed_objectives > 0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .severity(Severity::Advisory)
                    .message(format!(
                        "{overflowed_objectives} dominated GARCH QMLE candidate objectives overflowed to positive infinity"
                    ))
                    .metric("overflowed_objectives", overflowed_objectives as f64)
                    .build(),
            );
        }
        let relative_omega = (parameters.log_omega_normalized - data.log_mean_square).exp();
        if relative_omega <= boundary_tolerance
            || parameters.alpha <= boundary_tolerance
            || parameters.beta <= boundary_tolerance
            || parameters.slack <= boundary_tolerance
        {
            ctx.push(
                Issue::builder(IssueCode::ParameterAtBoundary)
                    .message(
                        "GARCH QMLE approached zero omega, selected a zero coefficient, or approached the stationarity boundary",
                    )
                    .metric("omega_over_initial_variance", relative_omega)
                    .metric("alpha", parameters.alpha)
                    .metric("beta", parameters.beta)
                    .metric("stationarity_slack", parameters.slack)
                    .metric("parameter_tolerance", boundary_tolerance)
                    .build(),
            );
        }

        let log_physical_scale_square = 2.0 * data.scale.ln();
        let omega = (parameters.log_omega_normalized + log_physical_scale_square).exp();
        if !omega.is_finite() || omega <= 0.0 {
            let code = if omega == 0.0 {
                IssueCode::NumericalUnderflow
            } else {
                IssueCode::NumericalOverflow
            };
            ctx.push(
                Issue::builder(code)
                    .severity(Severity::Fatal)
                    .message("the fitted physical-scale GARCH omega is not representable")
                    .metric("log_omega_normalized", parameters.log_omega_normalized)
                    .metric("scale", data.scale)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let Some(log_variances) = garch_qml_log_variances(&data, parameters) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("GARCH selected parameters produced an invalid log-variance path")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        let sigma2 = log_variances
            .iter()
            .map(|log_variance| (log_physical_scale_square + log_variance).exp())
            .collect::<Vec<_>>();
        if sigma2.iter().any(|variance| !variance.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("GARCH physical conditional variance overflowed")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let underflowed = sigma2.iter().filter(|variance| **variance == 0.0).count();
        if underflowed > 0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalUnderflow)
                    .message(format!(
                        "{underflowed} GARCH conditional variances underflowed to zero"
                    ))
                    .metric("underflowed_variances", underflowed as f64)
                    .compromise(NumericalCompromise::new(
                        "return every positive GARCH conditional variance in binary64",
                        "return zero where the physical-scale variance is below binary64 range",
                        "the normalized log recurrence is positive, but rescaling underflowed",
                        "zero entries are representational underflow, not zero process variance",
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedGarch11 {
            omega,
            alpha: parameters.alpha,
            beta: parameters.beta,
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn garch_golden_case(name: &str) -> serde_json::Value {
        let payload: serde_json::Value =
            serde_json::from_str(include_str!("../../../golden/garch_qml.json"))
                .expect("golden/garch_qml.json");
        payload["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("missing GARCH golden case {name}"))
            .clone()
    }

    fn decimal_observations(case: &serde_json::Value) -> Vec<f64> {
        case["input"]["observations"]
            .as_array()
            .expect("observations")
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .expect("decimal string")
                    .parse::<f64>()
                    .expect("binary64 observation")
            })
            .collect()
    }

    #[test]
    fn garch_parameter_transform_round_trips_and_enforces_stationarity() {
        for (alpha, beta) in [
            (0.0, 0.0),
            (0.2, 0.0),
            (0.0, 0.8),
            (0.05, 0.8),
            (0.1, 0.6),
            (0.1, 0.899),
        ] {
            let encoded = encode_garch_coefficients(alpha, beta).expect("valid coefficients");
            let decoded = decode_garch_qml_point(&[-2.0, encoded[0], encoded[1]])
                .expect("encoded coefficients decode");
            assert!(decoded.alpha >= 0.0);
            assert!(decoded.beta >= 0.0);
            assert!(decoded.alpha + decoded.beta < 1.0);
            assert!(decoded.slack > 0.0);
            assert!((decoded.alpha - alpha).abs() <= 32.0 * f64::EPSILON);
            assert!((decoded.beta - beta).abs() <= 32.0 * f64::EPSILON);
        }
        assert!(decode_garch_qml_point(&[-2.0, -f64::EPSILON, 0.0]).is_none());
        assert!(decode_garch_qml_point(&[-2.0, 0.0, -f64::EPSILON]).is_none());
        assert!(decode_garch_qml_point(&[f64::INFINITY, 0.0, 0.0]).is_none());
        assert!(encode_garch_coefficients(0.5, 0.5).is_none());
    }

    #[test]
    fn garch_log_recurrence_matches_direct_closed_form() {
        let residuals = [1.0, -2.0, 0.5, -0.25];
        let data = normalized_log_squares(&residuals).expect("finite residuals");
        let alpha = 0.1;
        let beta = 0.8;
        let omega_normalized: f64 = 0.05;
        let coordinates = encode_garch_coefficients(alpha, beta).expect("stationary coefficients");
        let parameters =
            decode_garch_qml_point(&[omega_normalized.ln(), coordinates[0], coordinates[1]])
                .expect("valid point");
        let actual = garch_qml_log_variances(&data, parameters).expect("log recurrence");
        let normalized = residuals.map(|value| value / data.scale);
        let mut expected = vec![
            normalized.iter().map(|value| value * value).sum::<f64>()
                / normalized.len() as f64;
            normalized.len()
        ];
        for time in 1..expected.len() {
            expected[time] = omega_normalized
                + alpha * normalized[time - 1] * normalized[time - 1]
                + beta * expected[time - 1];
        }
        for (log_actual, expected) in actual.iter().zip(&expected) {
            let relative_error = (log_actual.exp() / expected - 1.0).abs();
            assert!(relative_error <= 16.0 * f64::EPSILON);
        }
        let direct_objective = expected
            .iter()
            .zip(normalized)
            .map(|(variance, residual)| 0.5 * (variance.ln() + residual * residual / variance))
            .sum::<f64>();
        assert!(
            (garch_qml_objective(&data, parameters) - direct_objective).abs()
                <= 32.0 * f64::EPSILON
        );
    }

    #[test]
    fn garch_analytic_gradient_matches_centered_differences() {
        let residuals = [1.0, -2.0, 0.5, -0.25, 1.5, -0.75, 0.1, -0.1];
        let data = normalized_log_squares(&residuals).expect("finite residuals");
        let base = [0.05_f64, 0.1, 0.8];
        let parameters = |omega: f64, alpha: f64, beta: f64| {
            let coordinates =
                encode_garch_coefficients(alpha, beta).expect("stationary coefficients");
            decode_garch_qml_point(&[omega.ln(), coordinates[0], coordinates[1]])
                .expect("valid GARCH point")
        };
        let analytic = garch_qml_gradient(&data, parameters(base[0], base[1], base[2]));
        let root_epsilon = f64::EPSILON.cbrt();
        let steps = [base[0] * root_epsilon, root_epsilon, root_epsilon];
        let mut maximum_error = 0.0_f64;
        for coordinate in 0..3 {
            let mut plus = base;
            let mut minus = base;
            plus[coordinate] += steps[coordinate];
            minus[coordinate] -= steps[coordinate];
            let numerical = (garch_qml_objective(&data, parameters(plus[0], plus[1], plus[2]))
                - garch_qml_objective(&data, parameters(minus[0], minus[1], minus[2])))
                / (2.0 * steps[coordinate]);
            maximum_error = maximum_error.max((analytic[coordinate] - numerical).abs());
        }
        eprintln!("GARCH analytic gradient max centered-difference error={maximum_error:e}");
        // Measured 3.199e-10 on 2026-09-02; tolerance is about 4x.
        assert!(maximum_error <= 1.27e-9);
    }

    #[test]
    fn garch_face_selection_anchors_ulp_ties_to_the_global_minimum() {
        let objective = -100.0_f64;
        let one_ulp_worse = f64::from_bits(objective.to_bits() - 1);
        let sixteen_ulps_worse = f64::from_bits(objective.to_bits() - 16);
        let beyond_tie = f64::from_bits(objective.to_bits() - 17);
        let thirty_two_ulps_worse = f64::from_bits(objective.to_bits() - 32);
        let candidate = |face, value| GarchQmlCandidate {
            face,
            point: Vector::zeros(face.dimension()),
            objective: value,
        };
        let tie_ulps = signlred::Policy::default().optimizer_objective_tie_ulps;
        assert_eq!(tie_ulps, 16);
        let tied = select_garch_candidate(
            vec![
                candidate(GarchQmlFace::Interior, objective),
                candidate(GarchQmlFace::Corner, one_ulp_worse),
            ],
            tie_ulps,
        )
        .expect("candidate");
        assert_eq!(tied.face, GarchQmlFace::Corner);

        let outside = select_garch_candidate(
            vec![
                candidate(GarchQmlFace::Interior, objective),
                candidate(GarchQmlFace::Corner, beyond_tie),
            ],
            tie_ulps,
        )
        .expect("candidate");
        assert_eq!(outside.face, GarchQmlFace::Interior);

        let non_transitive_chain = select_garch_candidate(
            vec![
                candidate(GarchQmlFace::Interior, objective),
                candidate(GarchQmlFace::AlphaZero, sixteen_ulps_worse),
                candidate(GarchQmlFace::Corner, thirty_two_ulps_worse),
            ],
            tie_ulps,
        )
        .expect("candidate");
        assert_eq!(non_transitive_chain.face, GarchQmlFace::AlphaZero);
    }

    #[test]
    fn garch_qmle_returns_a_finite_strictly_stationary_recurrence() {
        let case = garch_golden_case("interior");
        let observations = decimal_observations(&case);
        let fitted = Garch11::new()
            .fit_series(
                &Vector::from_slice(&observations),
                &Session::new("garch", "interior-smoke"),
            )
            .expect("GARCH QMLE");
        eprintln!(
            "GARCH smoke omega={:e}, alpha={:e}, beta={:e}, persistence={:e}",
            fitted.value.omega,
            fitted.value.alpha,
            fitted.value.beta,
            fitted.value.alpha + fitted.value.beta
        );
        assert!(fitted.value.omega.is_finite() && fitted.value.omega > 0.0);
        assert!(fitted.value.alpha >= 0.0);
        assert!(fitted.value.beta >= 0.0);
        assert!(fitted.value.alpha + fitted.value.beta < 1.0);
        assert_eq!(fitted.value.sigma2.len(), observations.len());
        assert!(fitted
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|variance| variance.is_finite() && *variance > 0.0));
        let initial = fitted
            .value
            .resid
            .as_slice()
            .iter()
            .map(|residual| residual * residual)
            .sum::<f64>()
            / fitted.value.resid.len() as f64;
        let initial_relative_error = (fitted.value.sigma2[0] / initial - 1.0).abs();
        assert!(initial_relative_error <= 64.0 * f64::EPSILON);
        for time in 1..fitted.value.sigma2.len() {
            let expected = fitted.value.omega
                + fitted.value.alpha * fitted.value.resid[time - 1] * fitted.value.resid[time - 1]
                + fitted.value.beta * fitted.value.sigma2[time - 1];
            let relative_error = (fitted.value.sigma2[time] / expected - 1.0).abs();
            assert!(relative_error <= 128.0 * f64::EPSILON);
        }
        assert!(!fitted.report.contains(IssueCode::NonStationary));
    }

    #[test]
    fn garch_qmle_is_scale_equivariant() {
        let observations = decimal_observations(&garch_golden_case("interior"));
        let base = Garch11::new()
            .fit_series(
                &Vector::from_slice(&observations),
                &Session::new("garch", "base-scale"),
            )
            .expect("base GARCH");
        let mut maximum_coefficient_error = 0.0_f64;
        let mut maximum_omega_relative_error = 0.0_f64;
        let mut maximum_variance_relative_error = 0.0_f64;
        let mut maximum_residual_relative_error = 0.0_f64;
        for factor in [1e-100, 1e100] {
            let scaled_observations = observations
                .iter()
                .map(|value| value * factor)
                .collect::<Vec<_>>();
            let scaled = Garch11::new()
                .fit_series(
                    &Vector::from_slice(&scaled_observations),
                    &Session::new("garch", "scaled"),
                )
                .expect("scaled GARCH");
            maximum_coefficient_error = maximum_coefficient_error
                .max((scaled.value.alpha - base.value.alpha).abs())
                .max((scaled.value.beta - base.value.beta).abs());
            let scale_square = factor * factor;
            maximum_omega_relative_error = maximum_omega_relative_error
                .max((scaled.value.omega / (base.value.omega * scale_square) - 1.0).abs());
            for (scaled_variance, base_variance) in scaled
                .value
                .sigma2
                .as_slice()
                .iter()
                .zip(base.value.sigma2.as_slice())
            {
                maximum_variance_relative_error = maximum_variance_relative_error
                    .max((scaled_variance / (base_variance * scale_square) - 1.0).abs());
            }
            for (scaled_residual, base_residual) in scaled
                .value
                .resid
                .as_slice()
                .iter()
                .zip(base.value.resid.as_slice())
            {
                if *base_residual != 0.0 {
                    maximum_residual_relative_error = maximum_residual_relative_error
                        .max((scaled_residual / (base_residual * factor) - 1.0).abs());
                }
            }
        }
        eprintln!(
            "GARCH scale equivariance: coefficients={maximum_coefficient_error:e}, omega={maximum_omega_relative_error:e}, variance={maximum_variance_relative_error:e}, residual={maximum_residual_relative_error:e}"
        );
        // Measured on 2026-09-02: coefficient 6.431e-8, omega 2.349e-7,
        // variance 1.417e-7, and residual 5.108e-15; tolerances are about 4x.
        assert!(maximum_coefficient_error <= 2.57e-7);
        assert!(maximum_omega_relative_error <= 9.39e-7);
        assert!(maximum_variance_relative_error <= 5.66e-7);
        assert!(maximum_residual_relative_error <= 2.04e-14);
    }

    #[test]
    fn garch_qmle_rejects_empty_short_nonfinite_constant_and_invalid_config() {
        let empty = Garch11::new()
            .fit_series(&Vector::zeros(0), &Session::new("garch", "empty"))
            .unwrap_err();
        assert_eq!(empty.primary.code, IssueCode::EmptyMatrix);

        let short = Garch11::new()
            .fit_series(
                &Vector::from_slice(&[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0]),
                &Session::new("garch", "short"),
            )
            .unwrap_err();
        assert_eq!(short.primary.code, IssueCode::InsufficientSample);

        let nonfinite = Garch11::new()
            .fit_series(
                &Vector::from_slice(&[1.0, -1.0, 0.5, -0.5, f64::NAN, 0.25, -0.25, 0.0]),
                &Session::new("garch", "nonfinite"),
            )
            .unwrap_err();
        assert_eq!(nonfinite.primary.code, IssueCode::NonFiniteInput);

        let constant = Garch11::new()
            .fit_series(
                &Vector::from_slice(&[7.0; 8]),
                &Session::new("garch", "constant"),
            )
            .unwrap_err();
        assert_eq!(constant.primary.code, IssueCode::UnidentifiedModel);

        let mut invalid = Garch11::new();
        invalid.simplex_step = 0.0;
        let invalid = invalid
            .fit_series(
                &Vector::from_slice(&[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1]),
                &Session::new("garch", "invalid-config"),
            )
            .unwrap_err();
        assert_eq!(invalid.primary.code, IssueCode::InvalidParameter);

        let mut invalid_boundary_policy = Garch11::new();
        invalid_boundary_policy.optimizer.policy.model_parameter_tol = 1.0;
        let invalid_boundary_policy = invalid_boundary_policy
            .fit_series(
                &Vector::from_slice(&[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1]),
                &Session::new("garch", "invalid-boundary-policy"),
            )
            .unwrap_err();
        assert_eq!(
            invalid_boundary_policy.primary.code,
            IssueCode::InvalidParameter
        );
    }

    #[test]
    fn garch_preserves_nested_optimizer_reports_and_strict_failures() {
        let observations = decimal_observations(&garch_golden_case("interior"));
        let y = Vector::from_slice(&observations);
        let mut capped = Garch11::new();
        capped.optimizer.max_iterations = 1;
        let capped_session = Session::new("garch", "capped");
        let capped = capped
            .fit_series(&y, &capped_session)
            .expect("iteration cap returns a qualified GARCH fit");
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
            "the nested optimizer issue must be ingested exactly once"
        );
        assert_eq!(
            capped_session
                .ledger()
                .of_kind(ojizou_san::EventKind::FitFinished)
                .len(),
            1,
            "only the enclosing GARCH fit records a terminal event"
        );

        let mut strict = Garch11::new();
        strict.optimizer.max_iterations = 1;
        strict.optimizer.policy.abort_at = Severity::Warning;
        let strict_failure = strict
            .fit_series(&y, &Session::new("garch", "strict-capped"))
            .unwrap_err();
        assert_eq!(strict_failure.primary.code, IssueCode::MaxIterReached);
        assert!(strict_failure.report.contains(IssueCode::MaxIterReached));
    }

    #[test]
    fn fitted_garch_forecast_uses_the_exact_variance_recursion() {
        let fitted = FittedGarch11 {
            omega: 0.2,
            alpha: 0.1,
            beta: 0.7,
            sigma2: Vector::from_slice(&[1.5, 2.0]),
            resid: Vector::from_slice(&[-0.5, 1.0]),
        };
        let forecast = fitted
            .forecast_variance(3, &Session::new("garch", "forecast"))
            .expect("finite forecast")
            .value;
        let first = 0.2 + 0.1 * 1.0_f64.powi(2) + 0.7 * 2.0;
        let second = 0.2 + (0.1 + 0.7) * first;
        let third = 0.2 + (0.1 + 0.7) * second;
        assert_eq!(forecast.as_slice(), &[first, second, third]);
    }

    #[test]
    fn garch_rejects_unrepresentable_physical_scale_parameters() {
        let observations = decimal_observations(&garch_golden_case("interior"));
        let tiny = Vector::from_iter(observations.iter().map(|value| value * 1e-162));
        let underflow = Garch11::new()
            .fit_series(&tiny, &Session::new("garch", "physical-underflow"))
            .unwrap_err();
        assert_eq!(underflow.primary.code, IssueCode::NumericalUnderflow);

        let huge = Vector::from_iter(observations.iter().map(|value| value * 1e155));
        let overflow = Garch11::new()
            .fit_series(&huge, &Session::new("garch", "physical-overflow"))
            .unwrap_err();
        assert_eq!(overflow.primary.code, IssueCode::NumericalOverflow);
    }

    #[test]
    fn garch_decimal_golden_replay() {
        let payload: serde_json::Value =
            serde_json::from_str(include_str!("../../../golden/garch_qml.json"))
                .expect("golden/garch_qml.json");
        let cases = payload["cases"].as_array().expect("cases");
        assert_eq!(cases.len(), 6, "oracle generator documents six cases");
        let parse = |value: &serde_json::Value| {
            value
                .as_str()
                .expect("decimal string")
                .parse::<f64>()
                .expect("binary64 value")
        };
        let mut maximum_coefficient_error = 0.0_f64;
        let mut maximum_omega_relative_error = 0.0_f64;
        let mut maximum_objective_error = 0.0_f64;
        let mut maximum_variance_relative_error = 0.0_f64;

        for case in cases {
            let name = case["name"].as_str().expect("case name");
            let outcome = case["expected"]["outcome"].as_str().expect("outcome");
            if outcome == "failure" {
                let observations = decimal_observations(case);
                let failure = Garch11::new()
                    .fit_series(
                        &Vector::from_slice(&observations),
                        &Session::new("garch", name),
                    )
                    .expect_err("oracle expects an unidentified fit");
                assert_eq!(failure.primary.code, IssueCode::UnidentifiedModel);
                continue;
            }
            if outcome == "extended_real_probe" {
                let input = &case["input"];
                let mut residuals = input["prefix"]
                    .as_array()
                    .expect("prefix")
                    .iter()
                    .map(&parse)
                    .collect::<Vec<_>>();
                residuals.extend(std::iter::repeat_n(
                    parse(&input["repeat_value"]),
                    input["repeat_count"].as_u64().expect("repeat count") as usize,
                ));
                residuals.extend(
                    input["suffix"]
                        .as_array()
                        .expect("suffix")
                        .iter()
                        .map(&parse),
                );
                let data = normalized_log_squares(&residuals).expect("finite probe residuals");
                let expected = &case["expected"]["parameters"];
                let alpha = parse(&expected["alpha"]);
                let beta = parse(&expected["beta"]);
                let omega_normalized = parse(&expected["omega_normalized"]);
                let coordinates = encode_garch_coefficients(alpha, beta).expect("stationary probe");
                let parameters = decode_garch_qml_point(&[
                    omega_normalized.ln(),
                    coordinates[0],
                    coordinates[1],
                ])
                .expect("valid extended-real probe point");
                assert_eq!(garch_qml_objective(&data, parameters), f64::INFINITY);
                assert!(case["expected"]["decimal_objective_is_finite"]
                    .as_bool()
                    .expect("Decimal finiteness"));
                assert_eq!(case["binary64_replay"]["outcome"], "positive_infinity");
                continue;
            }

            let observations = decimal_observations(case);
            let fitted = Garch11::new()
                .fit_series(
                    &Vector::from_slice(&observations),
                    &Session::new("garch", name),
                )
                .expect("oracle expects a fitted GARCH");
            let expected = &case["expected"];
            let expected_omega = parse(&expected["omega_physical"]);
            let expected_alpha = parse(&expected["alpha"]);
            let expected_beta = parse(&expected["beta"]);
            maximum_coefficient_error = maximum_coefficient_error
                .max((fitted.value.alpha - expected_alpha).abs())
                .max((fitted.value.beta - expected_beta).abs());
            maximum_omega_relative_error =
                maximum_omega_relative_error.max((fitted.value.omega / expected_omega - 1.0).abs());

            let data = normalized_log_squares(fitted.value.resid.as_slice())
                .expect("fitted residual normalization");
            let coordinates = encode_garch_coefficients(fitted.value.alpha, fitted.value.beta)
                .expect("fitted coefficients are stationary");
            let parameters = decode_garch_qml_point(&[
                fitted.value.omega.ln() - 2.0 * data.scale.ln(),
                coordinates[0],
                coordinates[1],
            ])
            .expect("fitted normalized parameters");
            let actual_objective = garch_qml_objective(&data, parameters);
            let expected_objective = parse(&expected["objective_without_constants"]);
            maximum_objective_error =
                maximum_objective_error.max((actual_objective - expected_objective).abs());

            let expected_sigma2 = expected["sigma2"].as_array().expect("sigma2");
            assert_eq!(fitted.value.sigma2.len(), expected_sigma2.len());
            for (actual, expected) in fitted.value.sigma2.as_slice().iter().zip(expected_sigma2) {
                maximum_variance_relative_error =
                    maximum_variance_relative_error.max((*actual / parse(expected) - 1.0).abs());
            }
            match expected["selection"].as_str().expect("selection") {
                "alpha_zero" => {
                    assert_eq!(fitted.value.alpha, 0.0, "{name}");
                    assert_eq!(fitted.report.n_parameters, Some(2), "{name}");
                    assert!(fitted.report.contains(IssueCode::ParameterAtBoundary));
                }
                "beta_zero" => {
                    assert_eq!(fitted.value.beta, 0.0, "{name}");
                    assert_eq!(fitted.report.n_parameters, Some(2), "{name}");
                    assert!(fitted.report.contains(IssueCode::ParameterAtBoundary));
                }
                "interior" => {
                    assert!(
                        fitted.value.alpha > 0.0 && fitted.value.beta > 0.0,
                        "{name}"
                    );
                    assert_eq!(fitted.report.n_parameters, Some(3), "{name}");
                    assert!(!fitted.report.contains(IssueCode::ParameterAtBoundary));
                }
                selection => panic!("unknown oracle selection {selection}"),
            }
        }
        eprintln!(
            "GARCH Decimal golden max errors: coefficient={maximum_coefficient_error:e}, omega-relative={maximum_omega_relative_error:e}, objective={maximum_objective_error:e}, variance-relative={maximum_variance_relative_error:e}"
        );
        // Measured on 2026-09-02: coefficient 2.158e-8, omega-relative
        // 4.914e-8, objective 8.527e-14, and variance-relative 9.195e-8.
        // Tolerances are about 4x those binary64-versus-Decimal maxima.
        assert!(maximum_coefficient_error <= 8.63e-8);
        assert!(maximum_omega_relative_error <= 1.965e-7);
        assert!(maximum_objective_error <= 3.41e-13);
        assert!(maximum_variance_relative_error <= 3.677e-7);
    }
}
