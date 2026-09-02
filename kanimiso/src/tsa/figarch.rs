//! Fixed-order standard FIGARCH(1,d,1) Gaussian QMLE.

use super::common::{
    gaussian_qml_profile, inspect_scale_invariant_univariate, normalized_log_squares,
    select_ranked_objective_candidate, NormalizedLogSquares,
};
use crate::context::FitCtx;
use crate::data::Vector;
use crate::optimize::{decode_open_interval, encode_open_interval, NelderMead};
use crate::special::logsumexp;
use crate::traits::FitSeries;
use ojizou_san::Session;
use signlred::{
    insufficient_sample, Issue, IssueCode, NumericalCompromise, Qualified, Result, Severity,
};

const FIGARCH_DEFAULT_SIMPLEX_STEP: f64 = 0.25;
const FIGARCH_DEFAULT_TRUNCATION: usize = 1_000;
const FIGARCH_MINIMUM_TRUNCATION: usize = 3;
const FIGARCH_INTERIOR_PARAMETER_COUNT: usize = 5;
const FIGARCH_D_LOWER: f64 = 0.0;
const FIGARCH_D_UPPER: f64 = 1.0;
const FIGARCH_OMEGA_OFFSETS: [f64; 5] = [-6.0, -4.0, -2.0, 0.0, 2.0];
const FIGARCH_D_SEEDS: [f64; 5] = [0.08, 0.25, 0.45, 0.70, 0.90];
const FIGARCH_PHI_SHARE_SEEDS: [f64; 3] = [0.15, 0.45, 0.75];
const FIGARCH_BETA_SHARE_SEEDS: [f64; 3] = [0.15, 0.50, 0.85];
const FIGARCH_FIXED_AXIS_SEED: [f64; 1] = [0.5];

#[derive(Clone, Copy, Debug)]
struct FigarchQmlParameters {
    log_intercept_normalized: f64,
    phi: f64,
    d: f64,
    beta: f64,
    phi_share: f64,
    beta_share: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FacePosition {
    Lower,
    Interior,
    Upper,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FigarchQmlFace {
    d: FacePosition,
    phi: FacePosition,
    beta: FacePosition,
}

impl FigarchQmlFace {
    // The two d=0/beta=upper faces are omitted: all ARCH weights vanish and
    // omega/(1-beta) makes their phi/beta coordinates unidentified.  At d=1,
    // phi is forced to zero and beta=1 is outside the model domain.
    const ALL: [Self; 16] = [
        Self::new(
            FacePosition::Lower,
            FacePosition::Lower,
            FacePosition::Lower,
        ),
        Self::new(
            FacePosition::Lower,
            FacePosition::Interior,
            FacePosition::Lower,
        ),
        Self::new(
            FacePosition::Lower,
            FacePosition::Interior,
            FacePosition::Interior,
        ),
        Self::new(
            FacePosition::Lower,
            FacePosition::Upper,
            FacePosition::Lower,
        ),
        Self::new(
            FacePosition::Lower,
            FacePosition::Upper,
            FacePosition::Interior,
        ),
        Self::new(
            FacePosition::Interior,
            FacePosition::Lower,
            FacePosition::Lower,
        ),
        Self::new(
            FacePosition::Interior,
            FacePosition::Lower,
            FacePosition::Interior,
        ),
        Self::new(
            FacePosition::Interior,
            FacePosition::Lower,
            FacePosition::Upper,
        ),
        Self::new(
            FacePosition::Interior,
            FacePosition::Interior,
            FacePosition::Lower,
        ),
        Self::new(
            FacePosition::Interior,
            FacePosition::Interior,
            FacePosition::Interior,
        ),
        Self::new(
            FacePosition::Interior,
            FacePosition::Interior,
            FacePosition::Upper,
        ),
        Self::new(
            FacePosition::Interior,
            FacePosition::Upper,
            FacePosition::Lower,
        ),
        Self::new(
            FacePosition::Interior,
            FacePosition::Upper,
            FacePosition::Interior,
        ),
        Self::new(
            FacePosition::Interior,
            FacePosition::Upper,
            FacePosition::Upper,
        ),
        Self::new(
            FacePosition::Upper,
            FacePosition::Lower,
            FacePosition::Lower,
        ),
        Self::new(
            FacePosition::Upper,
            FacePosition::Lower,
            FacePosition::Interior,
        ),
    ];

    const fn new(d: FacePosition, phi: FacePosition, beta: FacePosition) -> Self {
        Self { d, phi, beta }
    }

    fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|candidate| *candidate == self)
            .expect("FIGARCH face belongs to the exhaustive face table")
    }

    fn dimension(self) -> usize {
        1 + usize::from(self.d == FacePosition::Interior)
            + usize::from(self.phi == FacePosition::Interior)
            + usize::from(self.beta == FacePosition::Interior)
    }

    fn selection_rank(self) -> usize {
        self.dimension() * Self::ALL.len() + self.index()
    }

    fn session_name(self) -> String {
        format!(
            "qml-d-{}-phi-{}-beta-{}",
            position_name(self.d),
            position_name(self.phi),
            position_name(self.beta)
        )
    }

    fn has_boundary(self) -> bool {
        self.d != FacePosition::Interior
            || self.phi != FacePosition::Interior
            || self.beta != FacePosition::Interior
    }
}

fn position_name(position: FacePosition) -> &'static str {
    match position {
        FacePosition::Lower => "lower",
        FacePosition::Interior => "interior",
        FacePosition::Upper => "upper",
    }
}

struct FigarchQmlCandidate {
    face: FigarchQmlFace,
    point: Vector,
    objective: f64,
}

struct FigarchQmlWeights {
    log_lambdas: Vec<f64>,
    log_suffix_sums: Vec<f64>,
    last_log_delta: f64,
    #[cfg(test)]
    log_deltas: Vec<f64>,
    #[cfg(test)]
    log_direct_arch_weights: Vec<f64>,
}

struct FigarchQmlPath {
    log_variances: Vec<f64>,
    weights: FigarchQmlWeights,
}

fn decode_face_coordinate(
    position: FacePosition,
    point: &[f64],
    cursor: &mut usize,
) -> Option<f64> {
    match position {
        FacePosition::Lower => Some(0.0),
        FacePosition::Upper => Some(1.0),
        FacePosition::Interior => {
            let coordinate = *point.get(*cursor)?;
            *cursor += 1;
            decode_open_interval(coordinate, 0.0, 1.0)
        }
    }
}

fn decode_figarch_qml_point(face: FigarchQmlFace, point: &[f64]) -> Option<FigarchQmlParameters> {
    if point.len() != face.dimension() || !point.first()?.is_finite() {
        return None;
    }
    let mut cursor = 1;
    let d = decode_face_coordinate(face.d, point, &mut cursor)?;
    let phi_share = decode_face_coordinate(face.phi, point, &mut cursor)?;
    let phi_upper = (1.0 - d) * 0.5;
    if face.phi != FacePosition::Lower && phi_upper == 0.0 {
        return None;
    }
    let phi = phi_upper * phi_share;
    let beta_share = decode_face_coordinate(face.beta, point, &mut cursor)?;
    let beta_upper = d + phi;
    if face.beta != FacePosition::Lower && beta_upper == 0.0 {
        return None;
    }
    let beta = beta_upper * beta_share;
    if cursor != point.len()
        || !d.is_finite()
        || !phi.is_finite()
        || !beta.is_finite()
        || !(FIGARCH_D_LOWER..=FIGARCH_D_UPPER).contains(&d)
        || phi < 0.0
        || phi > phi_upper
        || beta < 0.0
        || beta > beta_upper
        || beta >= 1.0
        || (face.d == FacePosition::Interior && !(d > FIGARCH_D_LOWER && d < FIGARCH_D_UPPER))
        || (face.phi == FacePosition::Interior && !(phi > 0.0 && phi < phi_upper))
        || (face.beta == FacePosition::Interior && !(beta > 0.0 && beta < beta_upper))
    {
        return None;
    }
    Some(FigarchQmlParameters {
        log_intercept_normalized: point[0],
        phi,
        d,
        beta,
        phi_share,
        beta_share,
    })
}

fn figarch_parameters_from_omega(
    omega: f64,
    phi: f64,
    d: f64,
    beta: f64,
) -> Option<FigarchQmlParameters> {
    let phi_upper = (1.0 - d) * 0.5;
    let beta_upper = d + phi;
    let one_minus_beta = 1.0 - beta;
    if !omega.is_finite()
        || omega <= 0.0
        || !phi.is_finite()
        || !d.is_finite()
        || !beta.is_finite()
        || !(FIGARCH_D_LOWER..=FIGARCH_D_UPPER).contains(&d)
        || phi < 0.0
        || phi > phi_upper
        || beta < 0.0
        || beta > beta_upper
        || one_minus_beta <= 0.0
    {
        return None;
    }
    let log_intercept = omega.ln() - (-beta).ln_1p();
    log_intercept.is_finite().then_some(FigarchQmlParameters {
        log_intercept_normalized: log_intercept,
        phi,
        d,
        beta,
        phi_share: if phi_upper == 0.0 {
            0.0
        } else {
            phi / phi_upper
        },
        beta_share: if beta_upper == 0.0 {
            0.0
        } else {
            beta / beta_upper
        },
    })
}

fn log_product(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY || right == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else {
        left + right
    }
}

fn fallible_filled(length: usize, value: f64) -> Option<Vec<f64>> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).ok()?;
    values.resize(length, value);
    Some(values)
}

fn figarch_qml_weights(
    parameters: FigarchQmlParameters,
    truncation: usize,
) -> Option<FigarchQmlWeights> {
    if truncation == 0
        || !parameters.log_intercept_normalized.is_finite()
        || !parameters.phi.is_finite()
        || !parameters.d.is_finite()
        || !parameters.beta.is_finite()
        || !(FIGARCH_D_LOWER..=FIGARCH_D_UPPER).contains(&parameters.d)
        || parameters.phi < 0.0
        || parameters.phi > (1.0 - parameters.d) * 0.5
        || parameters.beta < 0.0
        || parameters.beta > parameters.d + parameters.phi
        || parameters.beta >= 1.0
    {
        return None;
    }

    let mut log_lambdas = fallible_filled(truncation, f64::NEG_INFINITY)?;
    let mut log_deltas = fallible_filled(truncation, f64::NEG_INFINITY)?;
    let mut log_direct_arch_weights = fallible_filled(truncation, f64::NEG_INFINITY)?;
    let log_beta = if parameters.beta == 0.0 {
        f64::NEG_INFINITY
    } else {
        parameters.beta.ln()
    };

    if parameters.d == FIGARCH_D_LOWER {
        let alpha = parameters.phi * (1.0 - parameters.beta_share);
        if alpha < 0.0 {
            return None;
        }
        if alpha > 0.0 {
            let log_alpha = alpha.ln();
            for lag in 0..truncation {
                log_lambdas[lag] = if lag == 0 {
                    log_alpha
                } else if parameters.beta == 0.0 {
                    f64::NEG_INFINITY
                } else {
                    log_alpha + lag as f64 * log_beta
                };
            }
            log_direct_arch_weights[0] = log_alpha;
        }
    } else if parameters.d == FIGARCH_D_UPPER {
        if parameters.phi != 0.0 {
            return None;
        }
        let one_minus_beta = 1.0 - parameters.beta;
        if one_minus_beta <= 0.0 {
            return None;
        }
        let log_first = one_minus_beta.ln();
        log_deltas[0] = 0.0;
        log_direct_arch_weights[0] = log_first;
        for lag in 0..truncation {
            log_lambdas[lag] = if lag == 0 {
                log_first
            } else if parameters.beta == 0.0 {
                f64::NEG_INFINITY
            } else {
                log_first + lag as f64 * log_beta
            };
        }
    } else {
        if !(parameters.d > FIGARCH_D_LOWER && parameters.d < FIGARCH_D_UPPER) {
            return None;
        }
        let beta_upper = parameters.d + parameters.phi;
        let first_direct = beta_upper * (1.0 - parameters.beta_share);
        if first_direct < 0.0 {
            return None;
        }
        log_deltas[0] = parameters.d.ln();
        if first_direct > 0.0 {
            log_direct_arch_weights[0] = first_direct.ln();
            log_lambdas[0] = log_direct_arch_weights[0];
        }
        let phi_upper = (1.0 - parameters.d) * 0.5;
        for index in 1..truncation {
            let lag = index + 1;
            let ratio = (lag as f64 - 1.0 - parameters.d) / lag as f64;
            if ratio <= 0.0 || !ratio.is_finite() {
                return None;
            }
            log_deltas[index] = log_deltas[index - 1] + ratio.ln();
            let stable_gap = (lag as f64 - 2.0) * (1.0 + parameters.d) / (2.0 * lag as f64)
                + phi_upper * (1.0 - parameters.phi_share);
            if stable_gap < 0.0 || !stable_gap.is_finite() {
                return None;
            }
            if stable_gap > 0.0 {
                log_direct_arch_weights[index] = log_deltas[index - 1] + stable_gap.ln();
            }
            log_lambdas[index] = logsumexp(&[
                log_product(log_beta, log_lambdas[index - 1]),
                log_direct_arch_weights[index],
            ]);
            if !log_lambdas[index].is_finite() && log_lambdas[index] != f64::NEG_INFINITY {
                return None;
            }
        }
    }

    let suffix_length = truncation.checked_add(1)?;
    let mut log_suffix_sums = fallible_filled(suffix_length, f64::NEG_INFINITY)?;
    for index in (0..truncation).rev() {
        log_suffix_sums[index] = logsumexp(&[log_lambdas[index], log_suffix_sums[index + 1]]);
        if !log_suffix_sums[index].is_finite() && log_suffix_sums[index] != f64::NEG_INFINITY {
            return None;
        }
    }
    Some(FigarchQmlWeights {
        log_lambdas,
        log_suffix_sums,
        last_log_delta: *log_deltas.last()?,
        #[cfg(test)]
        log_deltas,
        #[cfg(test)]
        log_direct_arch_weights,
    })
}

fn figarch_qml_path(
    data: &NormalizedLogSquares,
    parameters: FigarchQmlParameters,
    truncation: usize,
) -> Option<FigarchQmlPath> {
    figarch_qml_path_with_backcast(data, parameters, truncation, data.log_mean_square)
}

fn figarch_log_variance_at(
    time: usize,
    historical_log_squares: &[f64],
    future_log_squares: &[f64],
    log_backcast: f64,
    log_intercept: f64,
    weights: &FigarchQmlWeights,
    terms: &mut Vec<f64>,
) -> Option<f64> {
    let available = historical_log_squares
        .len()
        .checked_add(future_log_squares.len())?;
    if time > available || !log_backcast.is_finite() || !log_intercept.is_finite() {
        return None;
    }
    terms.clear();
    terms.push(log_intercept);
    let observed = time.min(weights.log_lambdas.len());
    for index in 0..observed {
        let source = time.checked_sub(index.checked_add(1)?)?;
        let log_square = if source < historical_log_squares.len() {
            historical_log_squares[source]
        } else {
            *future_log_squares.get(source - historical_log_squares.len())?
        };
        if !log_square.is_finite() && log_square != f64::NEG_INFINITY {
            return None;
        }
        let log_term = log_product(weights.log_lambdas[index], log_square);
        if log_term != f64::NEG_INFINITY {
            terms.push(log_term);
        }
    }
    let log_presample_weight = *weights.log_suffix_sums.get(observed)?;
    if log_presample_weight != f64::NEG_INFINITY {
        terms.push(log_presample_weight + log_backcast);
    }
    let log_variance = logsumexp(terms);
    log_variance.is_finite().then_some(log_variance)
}

fn figarch_qml_path_with_backcast(
    data: &NormalizedLogSquares,
    parameters: FigarchQmlParameters,
    truncation: usize,
    log_backcast: f64,
) -> Option<FigarchQmlPath> {
    if data.scale == 0.0
        || data.values.is_empty()
        || data.normalized_values.len() != data.values.len()
        || !data.log_mean_square.is_finite()
        || !log_backcast.is_finite()
    {
        return None;
    }
    let weights = figarch_qml_weights(parameters, truncation)?;
    let mut log_variances = Vec::new();
    log_variances.try_reserve_exact(data.values.len()).ok()?;
    let mut terms = Vec::new();
    terms.try_reserve_exact(truncation.checked_add(2)?).ok()?;
    for time in 0..data.values.len() {
        let log_variance = figarch_log_variance_at(
            time,
            &data.values,
            &[],
            log_backcast,
            parameters.log_intercept_normalized,
            &weights,
            &mut terms,
        )?;
        log_variances.push(log_variance);
    }
    Some(FigarchQmlPath {
        log_variances,
        weights,
    })
}

fn figarch_qml_objective(
    data: &NormalizedLogSquares,
    parameters: FigarchQmlParameters,
    truncation: usize,
) -> f64 {
    let Some(path) = figarch_qml_path(data, parameters, truncation) else {
        return f64::INFINITY;
    };
    gaussian_qml_profile(&data.values, &path.log_variances)
}

fn tracked_figarch_qml_objective(
    data: &NormalizedLogSquares,
    parameters: FigarchQmlParameters,
    truncation: usize,
    overflowed: &mut usize,
) -> f64 {
    let objective = figarch_qml_objective(data, parameters, truncation);
    if objective == f64::INFINITY {
        *overflowed += 1;
    }
    objective
}

fn select_figarch_candidate(
    candidates: Vec<FigarchQmlCandidate>,
    objective_tie_ulps: usize,
) -> Option<FigarchQmlCandidate> {
    select_ranked_objective_candidate(
        candidates,
        objective_tie_ulps,
        |candidate| candidate.objective,
        |candidate| candidate.face.selection_rank(),
    )
}

fn admissible_figarch_candidate(face: FigarchQmlFace, point: &[f64], boundary_guard: f64) -> bool {
    let Some(parameters) = decode_figarch_qml_point(face, point) else {
        return false;
    };
    let away_from_boundary = |position: FacePosition, share: f64| {
        position != FacePosition::Interior
            || (share > boundary_guard && 1.0 - share > boundary_guard)
    };
    away_from_boundary(face.d, parameters.d)
        && away_from_boundary(face.phi, parameters.phi_share)
        && away_from_boundary(face.beta, parameters.beta_share)
}

fn seed_axis(position: FacePosition, interior: &'static [f64]) -> &'static [f64] {
    if position == FacePosition::Interior {
        interior
    } else {
        &FIGARCH_FIXED_AXIS_SEED
    }
}

fn encode_seed_point(
    face: FigarchQmlFace,
    d: f64,
    phi_share: f64,
    beta_share: f64,
) -> Option<Vector> {
    let mut point = Vec::with_capacity(face.dimension());
    point.push(0.0);
    if face.d == FacePosition::Interior {
        point.push(encode_open_interval(d, 0.0, 1.0)?);
    }
    if face.phi == FacePosition::Interior {
        point.push(encode_open_interval(phi_share, 0.0, 1.0)?);
    }
    if face.beta == FacePosition::Interior {
        point.push(encode_open_interval(beta_share, 0.0, 1.0)?);
    }
    Some(Vector::from_iter(point))
}

fn best_figarch_seed(
    face: FigarchQmlFace,
    data: &NormalizedLogSquares,
    truncation: usize,
    overflowed: &mut usize,
) -> Option<(Vector, f64)> {
    let mut best: Option<(Vector, f64)> = None;
    for omega_offset in FIGARCH_OMEGA_OFFSETS {
        for &d in seed_axis(face.d, &FIGARCH_D_SEEDS) {
            for &phi_share in seed_axis(face.phi, &FIGARCH_PHI_SHARE_SEEDS) {
                for &beta_share in seed_axis(face.beta, &FIGARCH_BETA_SHARE_SEEDS) {
                    let mut point = encode_seed_point(face, d, phi_share, beta_share)?;
                    let preliminary = decode_figarch_qml_point(face, point.as_slice())?;
                    point[0] = data.log_mean_square + omega_offset;
                    let parameters = FigarchQmlParameters {
                        log_intercept_normalized: point[0],
                        ..preliminary
                    };
                    let objective =
                        tracked_figarch_qml_objective(data, parameters, truncation, overflowed);
                    if objective.is_finite()
                        && best
                            .as_ref()
                            .map(|(_, current)| objective < *current)
                            .unwrap_or(true)
                    {
                        best = Some((point, objective));
                    }
                }
            }
        }
    }
    best
}

fn log_omitted_arch_weight(
    parameters: FigarchQmlParameters,
    weights: &FigarchQmlWeights,
) -> Option<f64> {
    let truncation = weights.log_lambdas.len();
    if parameters.d == FIGARCH_D_LOWER {
        let alpha = parameters.phi * (1.0 - parameters.beta_share);
        if alpha == 0.0 || parameters.beta == 0.0 {
            return Some(f64::NEG_INFINITY);
        }
        let log_tail =
            alpha.ln() + truncation as f64 * parameters.beta.ln() - (-parameters.beta).ln_1p();
        return (log_tail.is_finite() || log_tail == f64::NEG_INFINITY).then_some(log_tail);
    }
    if parameters.d == FIGARCH_D_UPPER {
        if parameters.beta == 0.0 {
            return Some(f64::NEG_INFINITY);
        }
        let log_tail = truncation as f64 * parameters.beta.ln();
        return (log_tail.is_finite() || log_tail == f64::NEG_INFINITY).then_some(log_tail);
    }
    let last_log_lambda = *weights.log_lambdas.last()?;
    let tail_factor_numerator = truncation as f64 * (1.0 - parameters.phi) - parameters.d;
    if !weights.last_log_delta.is_finite()
        || !tail_factor_numerator.is_finite()
        || tail_factor_numerator <= 0.0
    {
        return None;
    }
    let log_beta = if parameters.beta == 0.0 {
        f64::NEG_INFINITY
    } else {
        parameters.beta.ln()
    };
    let log_numerator = logsumexp(&[
        log_product(log_beta, last_log_lambda),
        weights.last_log_delta + tail_factor_numerator.ln() - parameters.d.ln(),
    ]);
    let log_tail = log_numerator - (-parameters.beta).ln_1p();
    (log_tail.is_finite() || log_tail == f64::NEG_INFINITY).then_some(log_tail)
}

fn has_nonzero_omitted_arch_tail(parameters: FigarchQmlParameters) -> bool {
    if parameters.d == FIGARCH_D_LOWER {
        parameters.phi * (1.0 - parameters.beta_share) > 0.0 && parameters.beta > 0.0
    } else if parameters.d == FIGARCH_D_UPPER {
        parameters.beta > 0.0
    } else {
        true
    }
}

/// Standard power-two FIGARCH(1,d,1) with a finite ARCH-infinity expansion.
///
/// The model is
/// `(1-beta*L) h[t] = omega + {1-beta*L-(1-phi*L)(1-L)^d} e[t]^2`.
/// It uses the Baillie--Bollerslev--Mikkelsen sample-mean-square presample
/// value, keeps the first `truncation` ARCH-infinity coefficients without
/// renormalizing them, and includes the initial observation in Gaussian QML.
/// The audited sufficient positivity wedge is `0 <= d <= 1`,
/// `0 <= phi <= (1-d)/2`, `0 <= beta <= d+phi`, and `beta < 1`.
#[derive(Clone, Debug)]
pub struct Figarch {
    /// Shared derivative-free solver and numerical-quality policy.
    pub optimizer: NelderMead,
    /// Dimensionless edge length of every deterministic initial simplex.
    pub simplex_step: f64,
    /// Number of retained ARCH-infinity coefficients; must be at least three.
    pub truncation: usize,
}

impl Default for Figarch {
    fn default() -> Self {
        Self {
            optimizer: NelderMead::default(),
            simplex_step: FIGARCH_DEFAULT_SIMPLEX_STEP,
            truncation: FIGARCH_DEFAULT_TRUNCATION,
        }
    }
}

/// Fitted standard FIGARCH(1,d,1) finite-filter state.
#[derive(Clone, Debug)]
pub struct FittedFigarch {
    /// Physical-scale equation intercept `omega` (not `omega/(1-beta)`).
    pub omega: f64,
    /// Short-memory ARCH-polynomial coefficient.
    pub phi: f64,
    /// Fractional differencing order in `[0,1]`.
    pub d: f64,
    /// Short-memory variance coefficient, constrained below one.
    pub beta: f64,
    /// Configured number of retained ARCH-infinity coefficients.
    pub truncation: usize,
    /// Frozen physical-scale BBM presample mean square.
    pub backcast: f64,
    /// Logarithm of the frozen physical-scale BBM presample mean square.
    pub log_backcast: f64,
    /// In-sample conditional variances on the observation scale.
    pub sigma2: Vector,
    /// In-sample log conditional variances on the observation scale.
    pub log_sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

impl FittedFigarch {
    /// Forecast conditional variances with the fitted finite ARCH-infinity filter.
    ///
    /// The first step conditions on every observed residual. At later steps,
    /// the recursion substitutes `E[e[t]^2 | F] = h[t]` for future squared
    /// innovations. Presample lags continue to use the frozen fitted backcast.
    pub fn forecast_variance(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast-variance"));
        if h == 0 {
            return ctx.finish(Vector::zeros(0));
        }
        if self.resid.is_empty()
            || self
                .resid
                .as_slice()
                .iter()
                .any(|residual| !residual.is_finite())
            || !self.log_backcast.is_finite()
        {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("FIGARCH cannot forecast from an invalid or empty fitted state")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if self.truncation < FIGARCH_MINIMUM_TRUNCATION {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("fitted FIGARCH truncation must retain at least three coefficients")
                    .metric("truncation", self.truncation as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if self.resid.len().checked_add(h).is_none() {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("FIGARCH forecast horizon exceeds addressable index space")
                    .metric("h", h as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let Some(parameters) =
            figarch_parameters_from_omega(self.omega, self.phi, self.d, self.beta)
        else {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("fitted FIGARCH coefficients are outside the documented domain")
                    .metric("omega", self.omega)
                    .metric("phi", self.phi)
                    .metric("d", self.d)
                    .metric("beta", self.beta)
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        let Some(weights) = figarch_qml_weights(parameters, self.truncation) else {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message("FIGARCH could not construct its forecast filter workspace")
                    .metric("truncation", self.truncation as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        };

        let mut historical_log_squares = Vec::new();
        let mut future_log_squares = Vec::new();
        let mut forecast = Vec::new();
        let mut terms = Vec::new();
        if historical_log_squares
            .try_reserve_exact(self.resid.len())
            .is_err()
            || future_log_squares.try_reserve_exact(h).is_err()
            || forecast.try_reserve_exact(h).is_err()
            || self
                .truncation
                .checked_add(2)
                .is_none_or(|capacity| terms.try_reserve_exact(capacity).is_err())
        {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("FIGARCH forecast horizon or truncation is too large to allocate")
                    .metric("h", h as f64)
                    .metric("truncation", self.truncation as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        historical_log_squares.extend(self.resid.as_slice().iter().map(|residual| {
            if *residual == 0.0 {
                f64::NEG_INFINITY
            } else {
                2.0 * residual.abs().ln()
            }
        }));

        let mut underflowed = 0usize;
        for step in 0..h {
            let Some(time) = self.resid.len().checked_add(step) else {
                unreachable!("forecast end was checked before allocating workspaces");
            };
            let Some(log_variance) = figarch_log_variance_at(
                time,
                &historical_log_squares,
                &future_log_squares,
                self.log_backcast,
                parameters.log_intercept_normalized,
                &weights,
                &mut terms,
            ) else {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message("FIGARCH conditional log-variance forecast became non-finite")
                        .metric("forecast_step", (step + 1) as f64)
                        .build(),
                );
                return Err(ctx.finish_failure());
            };
            let variance = log_variance.exp();
            if !variance.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NumericalOverflow)
                        .message("FIGARCH physical variance forecast overflowed")
                        .metric("forecast_step", (step + 1) as f64)
                        .metric("log_variance", log_variance)
                        .build(),
                );
                return Err(ctx.finish_failure());
            }
            underflowed += usize::from(variance == 0.0);
            forecast.push(variance);
            future_log_squares.push(log_variance);
        }
        if underflowed > 0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalUnderflow)
                    .message("one or more positive FIGARCH variance forecasts underflowed to zero")
                    .metric("underflowed_forecasts", underflowed as f64)
                    .compromise(NumericalCompromise::new(
                        "return every positive FIGARCH forecast variance in binary64",
                        "return zero after completing the recursion in the log domain",
                        "the physical-scale exponential is below binary64 range",
                        "zero forecast entries are representational underflow, not zero process variance",
                    ))
                    .build(),
            );
        }
        ctx.finish(Vector::from_iter(forecast))
    }
}

impl FitSeries for Figarch {
    type Fitted = FittedFigarch;

    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedFigarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.policy = self.optimizer.policy.clone();
        inspect_scale_invariant_univariate(&mut ctx, y);
        if y.is_empty() || y.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(ctx.finish_failure());
        }
        if !self.simplex_step.is_finite() || self.simplex_step <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("FIGARCH simplex_step must be finite and positive")
                    .metric("simplex_step", self.simplex_step)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if self.truncation < FIGARCH_MINIMUM_TRUNCATION {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("FIGARCH truncation must retain at least three coefficients")
                    .metric("truncation", self.truncation as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if self.truncation > self.optimizer.policy.max_infinite_filter_terms {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("FIGARCH truncation exceeds Policy::max_infinite_filter_terms")
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
        if !boundary_tolerance.is_finite() || boundary_tolerance <= 0.0 || boundary_tolerance >= 0.5
        {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message(
                        "Policy::model_parameter_tol must be finite, positive, and smaller than one half for FIGARCH",
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
                    .message("FIGARCH demeaning produced a non-finite residual")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let Some(data) = normalized_log_squares(residuals.as_slice()) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("FIGARCH residual normalization failed")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        if data.scale == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message(
                        "FIGARCH parameters are unidentified because every demeaned residual is zero",
                    )
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if y.len() <= FIGARCH_INTERIOR_PARAMETER_COUNT {
            ctx.push(
                Issue::builder(IssueCode::SampleSmallerThanFeatures)
                    .message("FIGARCH Gaussian QMLE needs more observations than fitted parameters")
                    .metric("n", y.len() as f64)
                    .metric("p", FIGARCH_INTERIOR_PARAMETER_COUNT as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if let Some(issue) =
            insufficient_sample(y.len(), FIGARCH_INTERIOR_PARAMETER_COUNT, &ctx.policy)
        {
            ctx.push(issue);
            return Err(ctx.finish_failure());
        }

        let mut overflowed_objectives = 0usize;
        let mut seeds = Vec::with_capacity(FigarchQmlFace::ALL.len());
        for face in FigarchQmlFace::ALL {
            seeds.push(best_figarch_seed(
                face,
                &data,
                self.truncation,
                &mut overflowed_objectives,
            ));
        }
        if seeds.iter().all(Option::is_none) {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .severity(Severity::Fatal)
                    .message("every deterministic FIGARCH Gaussian-QML seed overflowed")
                    .metric("overflowed_objectives", overflowed_objectives as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let initial_objective = seeds
            .iter()
            .filter_map(|seed| seed.as_ref().map(|(_, objective)| *objective))
            .fold(f64::INFINITY, f64::min);
        ctx.session.step(0, initial_objective, None);

        let missing_seed_faces = seeds.iter().filter(|seed| seed.is_none()).count();
        let mut max_iteration_faces = 0usize;
        let mut collapsed_faces = 0usize;
        let mut saturated_interior_faces = 0usize;
        let mut candidates = Vec::with_capacity(FigarchQmlFace::ALL.len());
        for (face, seed) in FigarchQmlFace::ALL.into_iter().zip(seeds) {
            let Some((seed_point, seed_objective)) = seed else {
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
                    let Some(parameters) = decode_figarch_qml_point(face, point) else {
                        return f64::INFINITY;
                    };
                    tracked_figarch_qml_objective(
                        &data,
                        parameters,
                        self.truncation,
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
            let optimized_is_admissible = optimization.value.is_finite()
                && admissible_figarch_candidate(
                    face,
                    optimization.point.as_slice(),
                    boundary_tolerance,
                );
            saturated_interior_faces +=
                usize::from(optimization.value.is_finite() && !optimized_is_admissible);
            let (point, objective) =
                if optimized_is_admissible && optimization.value < seed_objective {
                    (optimization.point, optimization.value)
                } else {
                    (seed_point, seed_objective)
                };
            candidates.push(FigarchQmlCandidate {
                face,
                point,
                objective,
            });
        }
        let Some(selected) = select_figarch_candidate(
            candidates,
            self.optimizer.policy.optimizer_objective_tie_ulps,
        ) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("FIGARCH Gaussian QMLE did not produce a finite face candidate")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        ctx.report.set_n_parameters(selected.face.dimension() + 1);
        if missing_seed_faces > 0
            || max_iteration_faces > 0
            || collapsed_faces > 0
            || saturated_interior_faces > 0
        {
            let code = match (
                missing_seed_faces,
                max_iteration_faces,
                collapsed_faces,
                saturated_interior_faces,
            ) {
                (0, positive, 0, 0) if positive > 0 => IssueCode::MaxIterReached,
                (0, 0, positive, 0) if positive > 0 => IssueCode::StepSizeCollapsed,
                _ => IssueCode::DidNotConverge,
            };
            ctx.push(
                Issue::builder(code)
                    .message(
                        "one or more FIGARCH parameter-face searches were incomplete; the reported fit is the best finite candidate and may come from a partial search",
                    )
                    .metric("missing_seed_faces", missing_seed_faces as f64)
                    .metric("max_iteration_faces", max_iteration_faces as f64)
                    .metric("collapsed_faces", collapsed_faces as f64)
                    .metric("saturated_interior_faces", saturated_interior_faces as f64)
                    .metric(
                        "searched_faces",
                        (FigarchQmlFace::ALL.len() - missing_seed_faces) as f64,
                    )
                    .metric("total_faces", FigarchQmlFace::ALL.len() as f64)
                    .build(),
            );
        }

        let selected_face = selected.face;
        let selected_objective = selected.objective;
        let Some(parameters) = decode_figarch_qml_point(selected_face, selected.point.as_slice())
        else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("FIGARCH could not decode the selected optimizer point")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        if !selected_objective.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("FIGARCH selected a non-finite Gaussian-QML objective")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if overflowed_objectives > 0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .severity(Severity::Advisory)
                    .message(format!(
                        "{overflowed_objectives} dominated FIGARCH candidate objectives overflowed to positive infinity"
                    ))
                    .metric("overflowed_objectives", overflowed_objectives as f64)
                    .build(),
            );
        }
        if selected_face.has_boundary() {
            ctx.push(
                Issue::builder(IssueCode::ParameterAtBoundary)
                    .message(format!(
                        "FIGARCH selected the d={}, phi={}, beta={} parameter face",
                        position_name(selected_face.d),
                        position_name(selected_face.phi),
                        position_name(selected_face.beta)
                    ))
                    .metric("d", parameters.d)
                    .metric("phi", parameters.phi)
                    .metric("beta", parameters.beta)
                    .build(),
            );
        }
        let log_relative_omega =
            parameters.log_intercept_normalized + (-parameters.beta).ln_1p() - data.log_mean_square;
        if log_relative_omega <= boundary_tolerance.ln() {
            ctx.push(
                Issue::builder(IssueCode::ParameterAtBoundary)
                    .message("FIGARCH omega approached its excluded zero boundary")
                    .metric("omega_over_backcast", log_relative_omega.exp())
                    .metric("parameter_tolerance", boundary_tolerance)
                    .build(),
            );
        }

        let Some(path) = figarch_qml_path(&data, parameters, self.truncation) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("FIGARCH selected parameters produced an invalid log-variance path")
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        let Some(log_omitted_weight) = log_omitted_arch_weight(parameters, &path.weights) else {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message(
                        "FIGARCH could not evaluate its omitted ARCH-infinity coefficient mass",
                    )
                    .build(),
            );
            return Err(ctx.finish_failure());
        };
        if has_nonzero_omitted_arch_tail(parameters) {
            ctx.push(
                Issue::builder(IssueCode::InfiniteFilterTruncated)
                    .message("FIGARCH used the configured finite ARCH-infinity expansion")
                    .metric("truncation", self.truncation as f64)
                    .metric("omitted_weight", log_omitted_weight.exp())
                    .metric("log_omitted_weight", log_omitted_weight)
                    .compromise(
                        NumericalCompromise::new(
                            "evaluate every coefficient of the infinite FIGARCH ARCH representation",
                            format!(
                                "retain the first {} coefficients without renormalization",
                                self.truncation
                            ),
                            "an infinite distributed lag cannot be evaluated in finite time",
                            "innovations older than the configured lag count have zero weight in the fitted approximation",
                        )
                        .violate("the FIGARCH ARCH representation has an omitted nonzero tail"),
                    )
                    .build(),
            );
        }

        let log_physical_scale_square = 2.0 * data.scale.ln();
        let log_omega_normalized = parameters.log_intercept_normalized + (-parameters.beta).ln_1p();
        let log_omega = log_physical_scale_square + log_omega_normalized;
        let omega = log_omega.exp();
        if !log_omega.is_finite() || !omega.is_finite() || omega <= 0.0 {
            let code = if omega == 0.0 {
                IssueCode::NumericalUnderflow
            } else {
                IssueCode::NumericalOverflow
            };
            ctx.push(
                Issue::builder(code)
                    .severity(Severity::Fatal)
                    .message("the fitted physical-scale FIGARCH omega is not representable")
                    .metric("log_omega", log_omega)
                    .metric("scale", data.scale)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let log_backcast = log_physical_scale_square + data.log_mean_square;
        let backcast = log_backcast.exp();
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
                    .message("FIGARCH physical log-variance path is not representable")
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
                    .message("FIGARCH physical conditional variance overflowed")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let underflowed = sigma2.iter().filter(|variance| **variance == 0.0).count();
        if underflowed > 0 || backcast == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalUnderflow)
                    .message(
                        "one or more positive FIGARCH variance quantities underflowed on the physical scale",
                    )
                    .metric("underflowed_variances", underflowed as f64)
                    .metric("backcast_underflowed", f64::from(backcast == 0.0))
                    .compromise(NumericalCompromise::new(
                        "return every positive FIGARCH variance quantity in binary64",
                        "return zero while preserving its finite logarithm",
                        "one or more physical-scale exponentials are below binary64 range",
                        "zero entries are representational underflow, not zero process variance",
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedFigarch {
            omega,
            phi: parameters.phi,
            d: parameters.d,
            beta: parameters.beta,
            truncation: self.truncation,
            backcast,
            log_backcast,
            sigma2: Vector::from_iter(sigma2),
            log_sigma2: Vector::from_iter(log_sigma2),
            resid: residuals,
        })
    }
}

#[cfg(test)]
fn figarch_qml_gradient(
    data: &NormalizedLogSquares,
    parameters: FigarchQmlParameters,
    truncation: usize,
) -> [f64; 4] {
    let Some(path) = figarch_qml_path(data, parameters, truncation) else {
        return [f64::NAN; 4];
    };
    let mut delta = parameters.d;
    let mut delta_d = 1.0;
    let mut lambda = parameters.d + parameters.phi - parameters.beta;
    let mut lambda_derivatives = [0.0, 1.0, 1.0, -1.0];
    let mut lambdas = Vec::with_capacity(truncation);
    let mut derivatives = Vec::with_capacity(truncation);
    lambdas.push(lambda);
    derivatives.push(lambda_derivatives);
    for index in 1..truncation {
        let lag = index + 1;
        let ratio = (lag as f64 - 1.0 - parameters.d) / lag as f64;
        let next_delta = delta * ratio;
        let next_delta_d = delta_d * ratio - delta / lag as f64;
        let next_lambda = parameters.beta * lambda + next_delta - parameters.phi * delta;
        let next_derivatives = [
            0.0,
            parameters.beta * lambda_derivatives[1] - delta,
            parameters.beta * lambda_derivatives[2] + next_delta_d - parameters.phi * delta_d,
            lambda + parameters.beta * lambda_derivatives[3],
        ];
        delta = next_delta;
        delta_d = next_delta_d;
        lambda = next_lambda;
        lambda_derivatives = next_derivatives;
        lambdas.push(lambda);
        derivatives.push(lambda_derivatives);
    }
    let intercept = parameters.log_intercept_normalized.exp();
    if !intercept.is_finite() {
        return [f64::NAN; 4];
    }
    let backcast = data.log_mean_square.exp();
    let mut gradient = [0.0; 4];
    for time in 0..data.values.len() {
        let variance = path.log_variances[time].exp();
        if !variance.is_finite() || variance <= 0.0 {
            return [f64::NAN; 4];
        }
        let mut variance_derivative = [intercept, 0.0, 0.0, 0.0];
        let observed = time.min(truncation);
        for index in 0..truncation {
            let square = if index < observed {
                data.normalized_values[time - index - 1].powi(2)
            } else {
                backcast
            };
            for coordinate in 1..4 {
                variance_derivative[coordinate] += derivatives[index][coordinate] * square;
            }
        }
        let standardized_square = if data.values[time] == f64::NEG_INFINITY {
            0.0
        } else {
            (data.values[time] - path.log_variances[time]).exp()
        };
        let contribution = 0.5 * (1.0 - standardized_square) / variance;
        for coordinate in 0..4 {
            gradient[coordinate] += contribution * variance_derivative[coordinate];
        }
    }
    gradient
}

#[cfg(test)]
fn one_step_log_variance(
    data: &NormalizedLogSquares,
    parameters: FigarchQmlParameters,
    truncation: usize,
) -> Option<f64> {
    let weights = figarch_qml_weights(parameters, truncation)?;
    let mut terms = Vec::new();
    terms.try_reserve_exact(truncation.checked_add(2)?).ok()?;
    figarch_log_variance_at(
        data.values.len(),
        &data.values,
        &[],
        data.log_mean_square,
        parameters.log_intercept_normalized,
        &weights,
        &mut terms,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(
        log_intercept_normalized: f64,
        phi: f64,
        d: f64,
        beta: f64,
    ) -> FigarchQmlParameters {
        let phi_upper = (1.0 - d) * 0.5;
        let beta_upper = d + phi;
        FigarchQmlParameters {
            log_intercept_normalized,
            phi,
            d,
            beta,
            phi_share: if phi_upper == 0.0 {
                0.0
            } else {
                phi / phi_upper
            },
            beta_share: if beta_upper == 0.0 {
                0.0
            } else {
                beta / beta_upper
            },
        }
    }

    fn fixture_residuals() -> [f64; 12] {
        [
            1.0, -2.0, 0.5, -0.25, 1.5, -0.75, 0.1, -0.4, 0.9, -1.2, 0.35, -0.15,
        ]
    }

    fn maximum_absolute_difference(left: &[f64], right: &[f64]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max)
    }

    fn figarch_golden() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../golden/figarch_qml.json"))
            .expect("golden/figarch_qml.json")
    }

    fn decimal_value(value: &serde_json::Value) -> f64 {
        match value.as_str().expect("decimal string") {
            "+inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            text => text.parse::<f64>().expect("binary64 value"),
        }
    }

    fn decimal_array(value: &serde_json::Value) -> Vec<f64> {
        value
            .as_array()
            .expect("decimal array")
            .iter()
            .map(decimal_value)
            .collect()
    }

    fn relative_error(actual: f64, expected: f64) -> f64 {
        if expected == 0.0 {
            actual.abs()
        } else {
            (actual / expected - 1.0).abs()
        }
    }

    fn golden_qml_parameters(fit: &serde_json::Value) -> FigarchQmlParameters {
        let d = match fit["d_status"].as_str().expect("d status") {
            "lower" => FIGARCH_D_LOWER,
            "upper" => FIGARCH_D_UPPER,
            "interior" => decimal_value(&fit["d"]),
            status => panic!("unknown d status {status}"),
        };
        let phi_upper = (1.0 - d) * 0.5;
        let phi = match fit["phi_status"].as_str().expect("phi status") {
            "lower" => 0.0,
            "upper" => phi_upper,
            "interior" => decimal_value(&fit["phi"]),
            status => panic!("unknown phi status {status}"),
        };
        let beta_upper = d + phi;
        let beta = match fit["beta_status"].as_str().expect("beta status") {
            "lower" => 0.0,
            "upper" => beta_upper,
            "interior" => decimal_value(&fit["beta"]),
            status => panic!("unknown beta status {status}"),
        };
        figarch_parameters_from_omega(decimal_value(&fit["omega_normalized"]), phi, d, beta)
            .expect("valid canonical golden QML parameters")
    }

    fn assert_golden_face_position(
        value: f64,
        status: &str,
        lower: f64,
        upper: f64,
        parameter: &str,
        case: &str,
    ) {
        match status {
            "lower" => assert_eq!(value, lower, "{case} {parameter}"),
            "upper" => assert_eq!(value, upper, "{case} {parameter}"),
            "interior" => assert!(
                value > lower && value < upper,
                "{case} {parameter}={value:e} is not interior to [{lower:e}, {upper:e}]"
            ),
            other => panic!("unknown {parameter} status {other}"),
        }
    }

    #[test]
    fn hand_computable_weights_path_objective_and_forecast_match() {
        let residuals = [0.5, -1.0, 0.25, 1.5, -0.75];
        let data = normalized_log_squares(&residuals).expect("normalized residuals");
        let physical_omega: f64 = 0.08;
        let beta: f64 = 0.20;
        let scale_square = data.scale * data.scale;
        let p = parameters(
            (physical_omega / scale_square / (1.0 - beta)).ln(),
            0.12,
            0.35,
            beta,
        );
        let path = figarch_qml_path(&data, p, 4).expect("FIGARCH path");
        let lambdas = path
            .weights
            .log_lambdas
            .iter()
            .map(|value| value.exp())
            .collect::<Vec<_>>();
        let expected_lambdas = [0.27, 0.12575, 0.0740625, 0.04875265625];
        let deltas = path
            .weights
            .log_deltas
            .iter()
            .map(|value| value.exp())
            .collect::<Vec<_>>();
        let expected_deltas = [0.35, 0.11375, 0.0625625, 0.04144765625];
        let physical_variances = path
            .log_variances
            .iter()
            .map(|value| (value + scale_square.ln()).exp())
            .collect::<Vec<_>>();
        let expected_variances = [
            0.52781625390625,
            0.37256625390625,
            0.50276000390625,
            0.30136156640625,
            0.8016100390625,
        ];
        let measured_lambda_error = maximum_absolute_difference(&lambdas, &expected_lambdas);
        let measured_delta_error = maximum_absolute_difference(&deltas, &expected_deltas);
        let measured_variance_error =
            maximum_absolute_difference(&physical_variances, &expected_variances);
        // Measured on 2026-09-03: lambda 5.552e-17, delta 5.552e-17,
        // and variance 2.221e-16. Nonzero tolerances are approximately 4x.
        assert!(measured_lambda_error <= 2.23e-16);
        assert!(measured_delta_error <= 2.23e-16);
        assert!(measured_variance_error <= 8.89e-16);
        let objective =
            figarch_qml_objective(&data, p, 4) + residuals.len() as f64 * data.scale.ln();
        let measured_objective_error = (objective - 3.8576549384568533).abs();
        // Measured on 2026-09-03: 8.882e-16; tolerance is approximately 4x.
        assert!(measured_objective_error <= 3.56e-15);
        let forecast =
            (one_step_log_variance(&data, p, 4).expect("forecast") + scale_square.ln()).exp();
        let measured_forecast_error = (forecast - 0.5881940625).abs();
        assert_eq!(measured_forecast_error, 0.0);
        let log_sigma2 = path
            .log_variances
            .iter()
            .map(|value| value + scale_square.ln())
            .collect::<Vec<_>>();
        let fitted = FittedFigarch {
            omega: physical_omega,
            phi: p.phi,
            d: p.d,
            beta: p.beta,
            truncation: 4,
            backcast: (data.log_mean_square + scale_square.ln()).exp(),
            log_backcast: data.log_mean_square + scale_square.ln(),
            sigma2: Vector::from_iter(log_sigma2.iter().map(|value| value.exp())),
            log_sigma2: Vector::from_iter(log_sigma2),
            resid: Vector::from_slice(&residuals),
        };
        let public_forecast = fitted
            .forecast_variance(1, &Session::new("figarch", "public-forecast"))
            .expect("public forecast");
        let measured_public_forecast_error = (public_forecast.value[0] - forecast).abs();
        assert_eq!(measured_public_forecast_error, 0.0);
        eprintln!(
            "FIGARCH hand-computable errors: lambda={measured_lambda_error:e}, delta={measured_delta_error:e}, variance={measured_variance_error:e}, objective={measured_objective_error:e}, forecast={measured_forecast_error:e}, public-forecast={measured_public_forecast_error:e}"
        );
    }

    #[test]
    fn figarch_decimal_golden_replays_fixed_kernels_and_invalid_domains() {
        let payload = figarch_golden();
        let fixed_cases = payload["fixed_cases"].as_array().expect("fixed cases");
        assert_eq!(
            fixed_cases.len(),
            payload["fixed_case_count"].as_u64().expect("fixed count") as usize
        );
        let mut maximum_scale_error = 0.0_f64;
        let mut maximum_backcast_error = 0.0_f64;
        let mut maximum_delta_error = 0.0_f64;
        let mut maximum_lambda_error = 0.0_f64;
        let mut maximum_lambda_sum_error = 0.0_f64;
        let mut maximum_normalized_variance_relative_error = 0.0_f64;
        let mut maximum_physical_variance_relative_error = 0.0_f64;
        let mut maximum_objective_error = 0.0_f64;
        let mut maximum_forecast_relative_error = 0.0_f64;

        for case in fixed_cases {
            let name = case["name"].as_str().expect("case name");
            let residuals = decimal_array(&case["residuals"]);
            let data = normalized_log_squares(&residuals).expect("golden residuals");
            let scale = decimal_value(&case["scale"]);
            maximum_scale_error = maximum_scale_error.max((data.scale - scale).abs());
            maximum_backcast_error = maximum_backcast_error.max(
                (data.log_mean_square.exp() - decimal_value(&case["backcast_normalized"])).abs(),
            );
            let source = &case["parameters"];
            let omega_normalized = decimal_value(&source["omega_normalized"]);
            let d = decimal_value(&source["d"]);
            let mut phi = decimal_value(&source["phi"]);
            let mut beta = decimal_value(&source["beta"]);
            match name {
                "phi_upper_face" => phi = (1.0 - d) * 0.5,
                "beta_upper_face" => beta = d + phi,
                _ => {}
            }
            let parameters = figarch_parameters_from_omega(omega_normalized, phi, d, beta)
                .unwrap_or_else(|| panic!("valid golden parameters for {name}"));
            let truncation = case["truncation"].as_u64().expect("truncation") as usize;
            let path = figarch_qml_path(&data, parameters, truncation)
                .unwrap_or_else(|| panic!("representable golden path for {name}"));
            assert_eq!(path.weights.log_lambdas.len(), truncation, "{name}");

            for checkpoint in case["delta_checkpoints"]
                .as_array()
                .expect("delta checkpoints")
            {
                let lag = checkpoint["lag"].as_u64().expect("delta lag") as usize;
                maximum_delta_error = maximum_delta_error.max(
                    (path.weights.log_deltas[lag - 1].exp() - decimal_value(&checkpoint["value"]))
                        .abs(),
                );
            }
            for checkpoint in case["lambda_checkpoints"]
                .as_array()
                .expect("lambda checkpoints")
            {
                let lag = checkpoint["lag"].as_u64().expect("lambda lag") as usize;
                maximum_lambda_error = maximum_lambda_error.max(
                    (path.weights.log_lambdas[lag - 1].exp() - decimal_value(&checkpoint["value"]))
                        .abs(),
                );
            }
            maximum_lambda_sum_error = maximum_lambda_sum_error.max(
                (path.weights.log_suffix_sums[0].exp()
                    - decimal_value(&case["lambda_sum_retained"]))
                .abs(),
            );

            let expected_normalized = case["normalized_variances"]
                .as_array()
                .expect("normalized variances");
            let expected_physical = case["physical_variances"]
                .as_array()
                .expect("physical variances");
            assert_eq!(
                path.log_variances.len(),
                expected_normalized.len(),
                "{name}"
            );
            assert_eq!(path.log_variances.len(), expected_physical.len(), "{name}");
            let log_scale_square = 2.0 * data.scale.ln();
            for ((actual_log, expected_normalized), expected_physical) in path
                .log_variances
                .iter()
                .zip(expected_normalized)
                .zip(expected_physical)
            {
                maximum_normalized_variance_relative_error =
                    maximum_normalized_variance_relative_error.max(relative_error(
                        actual_log.exp(),
                        decimal_value(expected_normalized),
                    ));
                maximum_physical_variance_relative_error = maximum_physical_variance_relative_error
                    .max(relative_error(
                        (log_scale_square + actual_log).exp(),
                        decimal_value(expected_physical),
                    ));
            }
            let objective = figarch_qml_objective(&data, parameters, truncation)
                + residuals.len() as f64 * data.scale.ln();
            maximum_objective_error =
                maximum_objective_error.max((objective - decimal_value(&case["objective"])).abs());
            let forecast_log = one_step_log_variance(&data, parameters, truncation)
                .unwrap_or_else(|| panic!("one-step forecast for {name}"));
            maximum_forecast_relative_error = maximum_forecast_relative_error.max(
                relative_error(
                    forecast_log.exp(),
                    decimal_value(&case["one_step_forecast"]["normalized_variance"]),
                )
                .max(relative_error(
                    (log_scale_square + forecast_log).exp(),
                    decimal_value(&case["one_step_forecast"]["physical_variance"]),
                )),
            );

            let physical_log_variances = path
                .log_variances
                .iter()
                .map(|value| log_scale_square + value)
                .collect::<Vec<_>>();
            let fitted = FittedFigarch {
                omega: omega_normalized * data.scale * data.scale,
                phi: parameters.phi,
                d: parameters.d,
                beta: parameters.beta,
                truncation,
                backcast: (log_scale_square + data.log_mean_square).exp(),
                log_backcast: log_scale_square + data.log_mean_square,
                sigma2: Vector::from_iter(physical_log_variances.iter().map(|value| value.exp())),
                log_sigma2: Vector::from_iter(physical_log_variances),
                resid: Vector::from_slice(&residuals),
            };
            let public_forecast = fitted
                .forecast_variance(1, &Session::new("figarch-decimal", name))
                .unwrap_or_else(|failure| panic!("public forecast for {name}: {failure}"));
            maximum_forecast_relative_error = maximum_forecast_relative_error.max(relative_error(
                public_forecast.value[0],
                decimal_value(&case["one_step_forecast"]["physical_variance"]),
            ));
        }

        let invalid = payload["invalid_candidates"]
            .as_array()
            .expect("invalid candidates");
        assert_eq!(
            invalid.len(),
            payload["invalid_candidate_count"]
                .as_u64()
                .expect("invalid count") as usize
        );
        let data = normalized_log_squares(&fixture_residuals()).expect("invalid probe data");
        for candidate in invalid {
            let name = candidate["name"].as_str().expect("invalid name");
            let objective = figarch_parameters_from_omega(
                decimal_value(&candidate["omega_normalized"]),
                decimal_value(&candidate["phi"]),
                decimal_value(&candidate["d"]),
                decimal_value(&candidate["beta"]),
            )
            .map_or(f64::INFINITY, |parameters| {
                figarch_qml_objective(
                    &data,
                    parameters,
                    candidate["truncation"].as_u64().expect("truncation") as usize,
                )
            });
            assert_eq!(objective, f64::INFINITY, "{name}");
            assert!(!objective.is_nan(), "{name}");
        }

        eprintln!(
            "FIGARCH Decimal fixed-kernel max errors: scale={maximum_scale_error:e}, backcast={maximum_backcast_error:e}, delta={maximum_delta_error:e}, lambda={maximum_lambda_error:e}, lambda-sum={maximum_lambda_sum_error:e}, normalized-variance-relative={maximum_normalized_variance_relative_error:e}, physical-variance-relative={maximum_physical_variance_relative_error:e}, objective={maximum_objective_error:e}, forecast-relative={maximum_forecast_relative_error:e}"
        );
        // Measured on 2026-09-02: scale 0, backcast 1.666e-16, delta
        // 5.552e-17, lambda 1.701e-16, retained sum 1.477e-13,
        // normalized variance relative 1.668e-13, physical variance relative
        // 1.666e-13, objective 5.996e-14, and forecast relative 1.664e-13.
        // Nonzero tolerances below are approximately four times those maxima.
        assert_eq!(maximum_scale_error, 0.0);
        assert!(maximum_backcast_error <= 6.67e-16);
        assert!(maximum_delta_error <= 2.23e-16);
        assert!(maximum_lambda_error <= 6.81e-16);
        assert!(maximum_lambda_sum_error <= 5.91e-13);
        assert!(maximum_normalized_variance_relative_error <= 6.68e-13);
        assert!(maximum_physical_variance_relative_error <= 6.67e-13);
        assert!(maximum_objective_error <= 2.40e-13);
        assert!(maximum_forecast_relative_error <= 6.66e-13);
    }

    #[test]
    fn figarch_decimal_golden_replays_selected_qml_kernels() {
        let payload = figarch_golden();
        let cases = payload["qml_cases"].as_array().expect("QML cases");
        assert_eq!(
            cases.len(),
            payload["qml_case_count"].as_u64().expect("QML count") as usize
        );
        let mut maximum_mean_error = 0.0_f64;
        let mut maximum_residual_error = 0.0_f64;
        let mut maximum_scale_error = 0.0_f64;
        let mut maximum_backcast_error = 0.0_f64;
        let mut maximum_lambda_error = 0.0_f64;
        let mut maximum_lambda_sum_error = 0.0_f64;
        let mut maximum_variance_relative_error = 0.0_f64;
        let mut maximum_objective_error = 0.0_f64;
        let mut maximum_forecast_relative_error = 0.0_f64;

        for case in cases {
            let name = case["name"].as_str().expect("case name");
            let observations = decimal_array(&case["observations"]);
            let expected_residuals = decimal_array(&case["demeaned_residuals"]);
            let mean = Vector::from_slice(&observations).mean();
            maximum_mean_error =
                maximum_mean_error.max((mean - decimal_value(&case["mean"])).abs());
            assert_eq!(observations.len(), expected_residuals.len(), "{name}");
            for (observation, expected_residual) in observations.iter().zip(&expected_residuals) {
                maximum_residual_error =
                    maximum_residual_error.max((observation - mean - expected_residual).abs());
            }

            let data = normalized_log_squares(&expected_residuals)
                .unwrap_or_else(|| panic!("normalized golden residuals for {name}"));
            maximum_scale_error =
                maximum_scale_error.max((data.scale - decimal_value(&case["scale"])).abs());
            maximum_backcast_error = maximum_backcast_error.max(
                (data.log_mean_square.exp() - decimal_value(&case["backcast_normalized"])).abs(),
            );
            let parameters = golden_qml_parameters(&case["fit"]);
            let truncation = case["truncation"].as_u64().expect("truncation") as usize;
            let path = figarch_qml_path(&data, parameters, truncation)
                .unwrap_or_else(|| panic!("selected golden QML path for {name}"));
            for checkpoint in case["lambda_checkpoints"]
                .as_array()
                .expect("lambda checkpoints")
            {
                let lag = checkpoint["lag"].as_u64().expect("lambda lag") as usize;
                maximum_lambda_error = maximum_lambda_error.max(
                    (path.weights.log_lambdas[lag - 1].exp() - decimal_value(&checkpoint["value"]))
                        .abs(),
                );
            }
            maximum_lambda_sum_error = maximum_lambda_sum_error.max(
                (path.weights.log_suffix_sums[0].exp()
                    - decimal_value(&case["lambda_sum_retained"]))
                .abs(),
            );
            let expected_normalized = case["normalized_variances"]
                .as_array()
                .expect("normalized variances");
            let expected_physical = case["physical_variances"]
                .as_array()
                .expect("physical variances");
            assert_eq!(
                path.log_variances.len(),
                expected_normalized.len(),
                "{name}"
            );
            assert_eq!(path.log_variances.len(), expected_physical.len(), "{name}");
            let log_scale_square = 2.0 * data.scale.ln();
            for ((actual_log, expected_normalized), expected_physical) in path
                .log_variances
                .iter()
                .zip(expected_normalized)
                .zip(expected_physical)
            {
                maximum_variance_relative_error = maximum_variance_relative_error
                    .max(relative_error(
                        actual_log.exp(),
                        decimal_value(expected_normalized),
                    ))
                    .max(relative_error(
                        (log_scale_square + actual_log).exp(),
                        decimal_value(expected_physical),
                    ));
            }
            let objective = figarch_qml_objective(&data, parameters, truncation)
                + expected_residuals.len() as f64 * data.scale.ln();
            maximum_objective_error = maximum_objective_error
                .max((objective - decimal_value(&case["fit"]["objective"])).abs());
            let forecast_log = one_step_log_variance(&data, parameters, truncation)
                .unwrap_or_else(|| panic!("selected one-step forecast for {name}"));
            maximum_forecast_relative_error = maximum_forecast_relative_error.max(
                relative_error(
                    forecast_log.exp(),
                    decimal_value(&case["one_step_forecast"]["normalized_variance"]),
                )
                .max(relative_error(
                    (log_scale_square + forecast_log).exp(),
                    decimal_value(&case["one_step_forecast"]["physical_variance"]),
                )),
            );

            let physical_log_variances = path
                .log_variances
                .iter()
                .map(|value| log_scale_square + value)
                .collect::<Vec<_>>();
            let fitted = FittedFigarch {
                omega: decimal_value(&case["fit"]["omega_physical"]),
                phi: parameters.phi,
                d: parameters.d,
                beta: parameters.beta,
                truncation,
                backcast: (log_scale_square + data.log_mean_square).exp(),
                log_backcast: log_scale_square + data.log_mean_square,
                sigma2: Vector::from_iter(physical_log_variances.iter().map(|value| value.exp())),
                log_sigma2: Vector::from_iter(physical_log_variances),
                resid: Vector::from_slice(&expected_residuals),
            };
            let public_forecast = fitted
                .forecast_variance(1, &Session::new("figarch-decimal-qml", name))
                .unwrap_or_else(|failure| panic!("public forecast for {name}: {failure}"));
            maximum_forecast_relative_error = maximum_forecast_relative_error.max(relative_error(
                public_forecast.value[0],
                decimal_value(&case["one_step_forecast"]["physical_variance"]),
            ));
        }

        eprintln!(
            "FIGARCH Decimal selected-kernel max errors: mean={maximum_mean_error:e}, residual={maximum_residual_error:e}, scale={maximum_scale_error:e}, backcast={maximum_backcast_error:e}, lambda={maximum_lambda_error:e}, lambda-sum={maximum_lambda_sum_error:e}, variance-relative={maximum_variance_relative_error:e}, objective={maximum_objective_error:e}, forecast-relative={maximum_forecast_relative_error:e}"
        );
        // Measured on 2026-09-03: mean 2.221e-16, residual 4.441e-16,
        // scale 0, backcast 1.111e-16, lambda 1.111e-16, retained sum 1.111e-16,
        // variance relative 6.662e-16, objective 7.106e-15, and forecast
        // relative 2.221e-16. Nonzero tolerances are approximately 4x.
        assert!(maximum_mean_error <= 8.89e-16);
        assert!(maximum_residual_error <= 1.78e-15);
        assert_eq!(maximum_scale_error, 0.0);
        assert!(maximum_backcast_error <= 4.45e-16);
        assert!(maximum_lambda_error <= 4.45e-16);
        assert!(maximum_lambda_sum_error <= 4.45e-16);
        assert!(maximum_variance_relative_error <= 2.67e-15);
        assert!(maximum_objective_error <= 2.85e-14);
        assert!(maximum_forecast_relative_error <= 8.89e-16);
    }

    #[test]
    fn figarch_public_qmle_matches_decimal_face_oracle() {
        let payload = figarch_golden();
        let cases = payload["qml_cases"].as_array().expect("QML cases");
        let mut maximum_residual_error = 0.0_f64;
        let mut maximum_coefficient_error = 0.0_f64;
        let mut maximum_omega_relative_error = 0.0_f64;
        let mut maximum_objective_error = 0.0_f64;
        let mut maximum_log_variance_error = 0.0_f64;
        let mut maximum_variance_relative_error = 0.0_f64;
        let mut maximum_forecast_relative_error = 0.0_f64;
        let mut all_interior_cases = 0usize;

        for case in cases {
            let name = case["name"].as_str().expect("case name");
            let observations = decimal_array(&case["observations"]);
            let expected_residuals = decimal_array(&case["demeaned_residuals"]);
            let truncation = case["truncation"].as_u64().expect("truncation") as usize;
            let mut model = Figarch::default();
            model.truncation = truncation;
            let fitted = model
                .fit_series(
                    &Vector::from_slice(&observations),
                    &Session::new("figarch-decimal-fit", name),
                )
                .unwrap_or_else(|failure| panic!("oracle fit for {name}: {failure}"));
            let expected = &case["fit"];
            let expected_parameters = golden_qml_parameters(expected);
            eprintln!(
                "FIGARCH {name}: actual omega={:.17e} phi={:.17e} d={:.17e} beta={:.17e}; oracle omega={:.17e} phi={:.17e} d={:.17e} beta={:.17e}",
                fitted.value.omega,
                fitted.value.phi,
                fitted.value.d,
                fitted.value.beta,
                decimal_value(&expected["omega_physical"]),
                expected_parameters.phi,
                expected_parameters.d,
                expected_parameters.beta,
            );
            assert_eq!(fitted.value.truncation, truncation, "{name}");
            assert_eq!(fitted.value.resid.len(), observations.len(), "{name}");
            assert_eq!(fitted.value.sigma2.len(), observations.len(), "{name}");
            assert_eq!(fitted.value.log_sigma2.len(), observations.len(), "{name}");
            for (actual, expected_residual) in fitted
                .value
                .resid
                .as_slice()
                .iter()
                .zip(&expected_residuals)
            {
                maximum_residual_error =
                    maximum_residual_error.max((actual - expected_residual).abs());
            }
            maximum_coefficient_error = maximum_coefficient_error
                .max((fitted.value.phi - expected_parameters.phi).abs())
                .max((fitted.value.d - expected_parameters.d).abs())
                .max((fitted.value.beta - expected_parameters.beta).abs());
            maximum_omega_relative_error = maximum_omega_relative_error.max(relative_error(
                fitted.value.omega,
                decimal_value(&expected["omega_physical"]),
            ));

            let phi_upper = (1.0 - fitted.value.d) * 0.5;
            assert_golden_face_position(
                fitted.value.d,
                expected["d_status"].as_str().expect("d status"),
                FIGARCH_D_LOWER,
                FIGARCH_D_UPPER,
                "d",
                name,
            );
            assert_golden_face_position(
                fitted.value.phi,
                expected["phi_status"].as_str().expect("phi status"),
                0.0,
                phi_upper,
                "phi",
                name,
            );
            assert_golden_face_position(
                fitted.value.beta,
                expected["beta_status"].as_str().expect("beta status"),
                0.0,
                fitted.value.d + fitted.value.phi,
                "beta",
                name,
            );
            let expected_n_parameters = 2 + ["d_status", "phi_status", "beta_status"]
                .iter()
                .filter(|field| expected[*field] == "interior")
                .count();
            assert_eq!(
                fitted.report.n_parameters,
                Some(expected_n_parameters),
                "{name}"
            );
            let expected_has_boundary = ["d_status", "phi_status", "beta_status"]
                .iter()
                .any(|field| expected[*field] != "interior");
            if !expected_has_boundary {
                all_interior_cases += 1;
                assert_eq!(name, "qml_all_interior_simulated");
            }
            assert_eq!(
                fitted.report.contains(IssueCode::ParameterAtBoundary),
                expected_has_boundary,
                "{name} boundary diagnostic"
            );

            let data = normalized_log_squares(fitted.value.resid.as_slice())
                .unwrap_or_else(|| panic!("fitted normalization for {name}"));
            let parameters = figarch_parameters_from_omega(
                fitted.value.omega / (data.scale * data.scale),
                fitted.value.phi,
                fitted.value.d,
                fitted.value.beta,
            )
            .unwrap_or_else(|| panic!("fitted parameters for {name}"));
            let objective = figarch_qml_objective(&data, parameters, truncation)
                + observations.len() as f64 * data.scale.ln();
            maximum_objective_error = maximum_objective_error
                .max((objective - decimal_value(&expected["objective"])).abs());
            let expected_variances = case["physical_variances"]
                .as_array()
                .expect("physical variances");
            assert_eq!(
                fitted.value.sigma2.len(),
                expected_variances.len(),
                "{name}"
            );
            for ((actual_log, actual_variance), expected_variance) in fitted
                .value
                .log_sigma2
                .as_slice()
                .iter()
                .zip(fitted.value.sigma2.as_slice())
                .zip(expected_variances)
            {
                let expected_variance = decimal_value(expected_variance);
                maximum_log_variance_error =
                    maximum_log_variance_error.max((actual_log - expected_variance.ln()).abs());
                maximum_variance_relative_error = maximum_variance_relative_error
                    .max(relative_error(*actual_variance, expected_variance));
            }
            let forecast = fitted
                .value
                .forecast_variance(1, &Session::new("figarch-decimal-fit", "forecast"))
                .unwrap_or_else(|failure| panic!("fitted forecast for {name}: {failure}"));
            maximum_forecast_relative_error = maximum_forecast_relative_error.max(relative_error(
                forecast.value[0],
                decimal_value(&case["one_step_forecast"]["physical_variance"]),
            ));
            let should_have_omitted_tail = has_nonzero_omitted_arch_tail(expected_parameters);
            assert_eq!(
                fitted.report.contains(IssueCode::InfiniteFilterTruncated),
                should_have_omitted_tail,
                "{name}"
            );
        }

        assert_eq!(
            all_interior_cases, 1,
            "the public QMLE oracle must retain one all-interior selected solution"
        );
        eprintln!(
            "FIGARCH Decimal public-fit max errors: residual={maximum_residual_error:e}, coefficient={maximum_coefficient_error:e}, omega-relative={maximum_omega_relative_error:e}, objective={maximum_objective_error:e}, log-variance={maximum_log_variance_error:e}, variance-relative={maximum_variance_relative_error:e}, forecast-relative={maximum_forecast_relative_error:e}"
        );
        // Measured on 2026-09-03: residual 4.441e-16, coefficient
        // 2.294e-8, omega relative 4.663e-8, objective 7.106e-15,
        // physical log variance 6.036e-8, variance relative 6.036e-8,
        // and forecast relative 1.967e-8. Tolerances are approximately 4x.
        assert!(maximum_residual_error <= 1.78e-15);
        assert!(maximum_coefficient_error <= 9.18e-8);
        assert!(maximum_omega_relative_error <= 1.87e-7);
        assert!(maximum_objective_error <= 2.85e-14);
        assert!(maximum_log_variance_error <= 2.42e-7);
        assert!(maximum_variance_relative_error <= 2.42e-7);
        assert!(maximum_forecast_relative_error <= 7.87e-8);
    }

    #[test]
    fn face_selection_prefers_lower_dimension_only_within_the_ulp_tie_band() {
        let simplest = FigarchQmlFace::ALL[0];
        let interior = FigarchQmlFace::ALL
            .iter()
            .copied()
            .find(|face| face.dimension() == 4)
            .expect("interior face");
        let tied = select_figarch_candidate(
            vec![
                FigarchQmlCandidate {
                    face: interior,
                    point: Vector::zeros(interior.dimension()),
                    objective: f64::from_bits(1.0_f64.to_bits() - 16),
                },
                FigarchQmlCandidate {
                    face: simplest,
                    point: Vector::zeros(simplest.dimension()),
                    objective: 1.0,
                },
            ],
            16,
        )
        .expect("tied selection");
        assert_eq!(tied.face, simplest);

        let separated = select_figarch_candidate(
            vec![
                FigarchQmlCandidate {
                    face: simplest,
                    point: Vector::zeros(simplest.dimension()),
                    objective: 1.0,
                },
                FigarchQmlCandidate {
                    face: interior,
                    point: Vector::zeros(interior.dimension()),
                    objective: f64::from_bits(1.0_f64.to_bits() - 17),
                },
            ],
            16,
        )
        .expect("separated selection");
        assert_eq!(separated.face, interior);
    }

    #[test]
    fn incomplete_face_search_is_aggregated_and_respects_strict_policy() {
        let payload = figarch_golden();
        let case = &payload["qml_cases"][0];
        let observations = Vector::from_iter(decimal_array(&case["observations"]));
        let mut model = Figarch::default();
        model.truncation = 12;
        model.optimizer.max_iterations = 1;
        let warning_session = Session::new("figarch-incomplete", "warning");
        let fitted = model
            .fit_series(&observations, &warning_session)
            .expect("warning policy returns the best finite candidate");
        let aggregate_issues = fitted
            .report
            .issues()
            .iter()
            .filter(|issue| issue.metrics.iter().any(|(name, _)| name == "total_faces"))
            .collect::<Vec<_>>();
        assert_eq!(
            aggregate_issues.len(),
            1,
            "child optimizer reports must collapse into one face-search issue"
        );
        let aggregate = aggregate_issues[0];
        assert_eq!(
            warning_session
                .ledger()
                .of_kind(ojizou_san::EventKind::FitFinished)
                .len(),
            1,
            "a qualified fit has exactly one terminal success event"
        );
        let metric = |name: &str| {
            aggregate
                .metrics
                .iter()
                .find_map(|(candidate, value)| (candidate == name).then_some(*value))
                .unwrap_or_else(|| panic!("missing aggregate metric {name}"))
        };
        assert_eq!(metric("total_faces"), FigarchQmlFace::ALL.len() as f64);
        assert_eq!(
            metric("searched_faces") + metric("missing_seed_faces"),
            FigarchQmlFace::ALL.len() as f64
        );
        assert!(
            metric("max_iteration_faces")
                + metric("collapsed_faces")
                + metric("saturated_interior_faces")
                + metric("missing_seed_faces")
                > 0.0
        );
        assert!(matches!(
            aggregate.code,
            IssueCode::MaxIterReached | IssueCode::StepSizeCollapsed | IssueCode::DidNotConverge
        ));

        let mut strict = model;
        strict.optimizer.policy.abort_at = Severity::Warning;
        let strict_session = Session::new("figarch-incomplete", "strict");
        let failure = strict
            .fit_series(&observations, &strict_session)
            .expect_err("strict policy must reject an incomplete face search");
        assert!(matches!(
            failure.primary.code,
            IssueCode::MaxIterReached | IssueCode::StepSizeCollapsed | IssueCode::DidNotConverge
        ));
        assert_eq!(
            strict_session
                .ledger()
                .of_kind(ojizou_san::EventKind::FitFailed)
                .len(),
            1,
            "a strict rejection has exactly one terminal failure event"
        );
        assert!(strict_session
            .ledger()
            .of_kind(ojizou_san::EventKind::FitFinished)
            .is_empty());
    }

    #[test]
    fn every_runtime_face_decodes_inside_the_audited_wedge() {
        assert_eq!(FigarchQmlFace::ALL.len(), 16);
        for face in FigarchQmlFace::ALL {
            let point = encode_seed_point(face, 0.4, 0.4, 0.4).expect("face seed");
            let p = decode_figarch_qml_point(face, point.as_slice()).expect("decoded face");
            assert!((0.0..=1.0).contains(&p.d));
            assert!(p.phi >= 0.0 && p.phi <= (1.0 - p.d) * 0.5);
            assert!(p.beta >= 0.0 && p.beta <= p.d + p.phi && p.beta < 1.0);
            assert_eq!(point.len(), face.dimension());
        }
    }

    #[test]
    fn boundary_geometric_weights_match_garch_and_igarch_faces() {
        let garch = parameters(-0.4, 0.42, 0.0, 0.31);
        let garch_weights = figarch_qml_weights(garch, 16).expect("d=0 weights");
        let mut maximum_garch_error = 0.0_f64;
        for (index, log_weight) in garch_weights.log_lambdas.iter().enumerate() {
            let expected = (garch.phi - garch.beta) * garch.beta.powi(index as i32);
            let error = (log_weight.exp() - expected).abs();
            maximum_garch_error = maximum_garch_error.max(error);
            assert!(error <= 5.56e-17);
        }
        let igarch = parameters(-0.4, 0.0, 1.0, 0.83);
        let igarch_weights = figarch_qml_weights(igarch, 16).expect("d=1 weights");
        let mut maximum_igarch_error = 0.0_f64;
        for (index, log_weight) in igarch_weights.log_lambdas.iter().enumerate() {
            let expected = (1.0 - igarch.beta) * igarch.beta.powi(index as i32);
            let error = (log_weight.exp() - expected).abs();
            maximum_igarch_error = maximum_igarch_error.max(error);
            assert!(error <= 5.56e-17);
        }
        // Measured on 2026-09-03: both maxima 1.388e-17; tolerances are 4x.
        eprintln!(
            "FIGARCH boundary geometric max errors: garch={maximum_garch_error:e}, igarch={maximum_igarch_error:e}"
        );
    }

    #[test]
    fn upper_faces_have_structural_zero_prefix_without_floors() {
        let p = parameters(-0.3, 0.3, 0.4, 0.7);
        let weights = figarch_qml_weights(p, 8).expect("upper-face weights");
        assert_eq!(weights.log_lambdas[0], f64::NEG_INFINITY);
        assert_eq!(weights.log_lambdas[1], f64::NEG_INFINITY);
        assert!(weights.log_lambdas[2].is_finite());
        assert!(weights
            .log_lambdas
            .iter()
            .all(|value| value.is_finite() || *value == f64::NEG_INFINITY));
    }

    #[test]
    fn direct_and_recursive_arch_weights_agree() {
        let mut maximum_error = 0.0_f64;
        for &(phi, d, beta) in &[(0.12, 0.35, 0.20), (0.0, 0.62, 0.4), (0.1, 0.2, 0.3)] {
            let p = parameters(-0.7, phi, d, beta);
            let weights = figarch_qml_weights(p, 64).expect("weights");
            let mut recursive = 0.0;
            for index in 0..64 {
                let direct = weights.log_direct_arch_weights[index].exp();
                recursive = if index == 0 {
                    direct
                } else {
                    beta * recursive + direct
                };
                let error = (weights.log_lambdas[index].exp() - recursive).abs();
                maximum_error = maximum_error.max(error);
                assert!(error <= 1.12e-16);
            }
        }
        // Measured on 2026-09-03: 2.776e-17; tolerance is approximately 4x.
        eprintln!("FIGARCH direct/recursive weight max error: {maximum_error:e}");
    }

    #[test]
    fn coefficient_prefix_does_not_depend_on_truncation() {
        let p = parameters(-0.5, 0.08, 0.37, 0.26);
        let short = figarch_qml_weights(p, 12).expect("short weights");
        let long = figarch_qml_weights(p, 128).expect("long weights");
        assert_eq!(short.log_lambdas, long.log_lambdas[..12]);
    }

    #[test]
    fn omitted_tail_matches_boundary_closed_forms() {
        let d_zero = parameters(-0.4, 0.42, 0.0, 0.31);
        let weights = figarch_qml_weights(d_zero, 25).expect("d=0 weights");
        let got = log_omitted_arch_weight(d_zero, &weights)
            .expect("tail")
            .exp();
        let expected = (d_zero.phi - d_zero.beta) * d_zero.beta.powi(25) / (1.0 - d_zero.beta);
        let d_zero_relative_error = (got - expected).abs() / expected;
        assert!(d_zero_relative_error <= 9.06e-15);

        let d_one = parameters(-0.4, 0.0, 1.0, 0.83);
        let weights = figarch_qml_weights(d_one, 25).expect("d=1 weights");
        let got = log_omitted_arch_weight(d_one, &weights)
            .expect("tail")
            .exp();
        let d_one_error = (got - d_one.beta.powi(25)).abs();
        assert!(d_one_error <= 2.09e-17);
        // Measured on 2026-09-03: d=0 relative 2.265e-15 and d=1 absolute
        // 5.205e-18. Nonzero tolerances are approximately 4x.
        eprintln!(
            "FIGARCH boundary tail errors: d-zero-relative={d_zero_relative_error:e}, d-one-absolute={d_one_error:e}"
        );
    }

    #[test]
    fn sign_reflection_and_scale_shift_preserve_the_normalized_kernel() {
        let residuals = fixture_residuals();
        let reflected = residuals.map(|value| -value);
        let scaled = residuals.map(|value| value * 1.0e120);
        let data = normalized_log_squares(&residuals).expect("base data");
        let reflected_data = normalized_log_squares(&reflected).expect("reflected data");
        let scaled_data = normalized_log_squares(&scaled).expect("scaled data");
        let p = parameters(-0.8, 0.11, 0.39, 0.28);
        let base = figarch_qml_path(&data, p, 20).expect("base path");
        let reflected_path = figarch_qml_path(&reflected_data, p, 20).expect("reflected path");
        let scaled_path = figarch_qml_path(&scaled_data, p, 20).expect("scaled path");
        assert_eq!(base.log_variances, reflected_path.log_variances);
        let measured_scale_error =
            maximum_absolute_difference(&base.log_variances, &scaled_path.log_variances);
        // Measured on 2026-09-03: 2.443e-15; tolerance is approximately 4x.
        assert!(measured_scale_error <= 9.78e-15);
        eprintln!("FIGARCH normalized scale-shift max error: {measured_scale_error:e}");
    }

    #[test]
    fn log_domain_qml_matches_the_ordinary_formula_at_moderate_scale() {
        let data = normalized_log_squares(&fixture_residuals()).expect("data");
        let p = parameters(-0.8, 0.11, 0.39, 0.28);
        let path = figarch_qml_path(&data, p, 20).expect("path");
        let ordinary = data
            .normalized_values
            .iter()
            .zip(&path.log_variances)
            .map(|(residual, log_variance)| {
                let variance = log_variance.exp();
                0.5 * (variance.ln() + residual * residual / variance)
            })
            .sum::<f64>();
        let logged = figarch_qml_objective(&data, p, 20);
        let measured = (ordinary - logged).abs();
        // Measured on 2026-09-03: 1.111e-16; tolerance is approximately 4x.
        assert!(measured <= 4.45e-16);
        eprintln!("FIGARCH ordinary/log-domain objective error: {measured:e}");
    }

    #[test]
    fn one_step_uses_the_frozen_training_backcast() {
        let residuals = fixture_residuals();
        let data = normalized_log_squares(&residuals).expect("training data");
        let p = parameters(-0.8, 0.11, 0.39, 0.28);
        let forecast = one_step_log_variance(&data, p, 20).expect("forecast");
        let mut extended_residuals = residuals.to_vec();
        extended_residuals.push(0.0);
        let extended = normalized_log_squares(&extended_residuals).expect("extended data");
        let extended_path = figarch_qml_path_with_backcast(&extended, p, 20, data.log_mean_square)
            .expect("extended path with frozen backcast");
        let measured = (forecast - extended_path.log_variances[residuals.len()]).abs();
        assert_eq!(measured, 0.0);
        eprintln!("FIGARCH frozen-backcast forecast error: {measured:e}");
    }

    #[test]
    fn high_truncation_and_beta_near_one_remain_in_log_domain() {
        let data = normalized_log_squares(&fixture_residuals()).expect("data");
        let beta = f64::from_bits(1.0_f64.to_bits() - 1);
        let p = parameters(-720.0, 0.0, 1.0, beta);
        let path = figarch_qml_path(&data, p, 4_000).expect("near-unit path");
        assert!(path.log_variances.iter().all(|value| value.is_finite()));
        let tail = log_omitted_arch_weight(p, &path.weights).expect("tail");
        assert!(tail.is_finite());
        assert!(figarch_qml_objective(&data, p, 4_000).is_finite());
    }

    #[test]
    fn interior_omitted_mass_is_positive_and_consistent_with_the_prefix() {
        for &(phi, d, beta) in &[(0.12, 0.35, 0.20), (0.0, 0.62, 0.4), (0.1, 0.2, 0.3)] {
            let p = parameters(-0.7, phi, d, beta);
            let weights = figarch_qml_weights(p, 1_000).expect("weights");
            let log_tail = log_omitted_arch_weight(p, &weights).expect("tail");
            assert!(log_tail.is_finite() && log_tail < 0.0);
            let included = weights.log_suffix_sums[0].exp();
            let tail = log_tail.exp();
            // Measured worst error was 4.219e-15 on 2026-09-02; tolerance is 4x.
            assert!((included + tail - 1.0).abs() <= 1.7e-14);
        }
    }

    #[test]
    fn analytic_gradient_matches_centered_finite_differences() {
        let data = normalized_log_squares(&fixture_residuals()).expect("data");
        let p = parameters(-0.7, 0.11, 0.39, 0.28);
        let analytic = figarch_qml_gradient(&data, p, 20);
        let physical = [p.log_intercept_normalized, p.phi, p.d, p.beta];
        let mut numerical = [0.0; 4];
        for coordinate in 0..4 {
            // Chosen in the centered-difference roundoff/truncation plateau.
            let step = 2.0e-6;
            let mut left = physical;
            let mut right = physical;
            left[coordinate] -= step;
            right[coordinate] += step;
            let left_p = parameters(left[0], left[1], left[2], left[3]);
            let right_p = parameters(right[0], right[1], right[2], right[3]);
            numerical[coordinate] = (figarch_qml_objective(&data, right_p, 20)
                - figarch_qml_objective(&data, left_p, 20))
                / (2.0 * step);
        }
        let measured = maximum_absolute_difference(&analytic, &numerical);
        // Measured on 2026-09-03: 1.219e-10; tolerance is approximately 4x.
        eprintln!("FIGARCH analytic/centered-gradient max error: {measured:e}");
        assert!(measured <= 4.88e-10, "measured gradient error {measured:e}");
    }

    #[test]
    fn configuration_and_constant_series_fail_explicitly() {
        let series = Vector::from_slice(&[1.0, -0.5, 0.2, -0.1, 0.7, -0.9, 0.4, -0.2]);
        let mut short_filter = Figarch::default();
        short_filter.truncation = 2;
        let failure = short_filter
            .fit_series(&series, &Session::new("figarch", "short-filter"))
            .expect_err("K<3 must fail");
        assert!(failure.report.contains(IssueCode::InvalidParameter));

        let mut invalid_policy = Figarch::default();
        invalid_policy.optimizer.policy.model_parameter_tol = 0.5;
        let failure = invalid_policy
            .fit_series(&series, &Session::new("figarch", "invalid-policy"))
            .expect_err("invalid boundary policy must fail");
        assert!(failure.report.contains(IssueCode::InvalidParameter));

        let mut impossible_filter = Figarch::default();
        impossible_filter.truncation = usize::MAX;
        let failure = impossible_filter
            .fit_series(&series, &Session::new("figarch", "impossible-filter"))
            .expect_err("unallocatable K must fail before allocation");
        assert!(failure.report.contains(IssueCode::InvalidParameter));

        let constant = Vector::from_slice(&[3.0; 12]);
        let failure = Figarch::default()
            .fit_series(&constant, &Session::new("figarch", "constant"))
            .expect_err("constant series must fail");
        assert!(failure.report.contains(IssueCode::UnidentifiedModel));
    }
}
