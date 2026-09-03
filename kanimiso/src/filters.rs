//! Linear filters: Baxter–King band-pass and a local-linear-trend Kalman smoother.
//!
//! A series shorter than the BK lead/lag window cannot identify the cycle.
//! Degenerate local-linear-trend covariance choices are rejected by the shared
//! state-space validator instead of being repaired or treated as a fitted line.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::state_space::LinearGaussianStateSpace;
use crate::traits::FitSeries;
use crate::validate::inspect_xy;
use ojizou_san::Session;
use signlred::{
    Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result, Severity,
};

fn inspect_univariate(ctx: &mut FitCtx, y: &Vector) {
    let x = Matrix::from_vector(y);
    inspect_xy(&mut ctx.report, &x, Some(y), &ctx.policy);
}

fn bandpass_frequencies(
    ctx: &mut FitCtx,
    filter: &str,
    low: f64,
    high: f64,
) -> Option<(f64, f64, f64)> {
    const MINIMUM_PERIOD: f64 = 2.0;
    if !(low.is_finite() && high.is_finite()) || low < MINIMUM_PERIOD || high <= low {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message(format!(
                    "{filter} periods must satisfy 2 <= low < high (got {low}, {high})"
                ))
                .build(),
        );
        return None;
    }

    let lower = 2.0 * std::f64::consts::PI / high;
    let upper = 2.0 * std::f64::consts::PI / low;
    if !(lower.is_finite() && upper.is_finite() && lower < upper) {
        ctx.push(
            Issue::builder(IssueCode::NumericalUnderflow)
                .message(format!(
                    "{filter} cut-off frequencies are indistinguishable at f64 precision"
                ))
                .build(),
        );
        return None;
    }
    let center = (upper - lower) / std::f64::consts::PI;
    if center == 0.0 || !center.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::NumericalUnderflow)
                .message(format!(
                    "{filter} pass-band width is not representable at f64 precision"
                ))
                .build(),
        );
        return None;
    }
    Some((lower, upper, center))
}

fn checked_filter_sum(
    ctx: &mut FitCtx,
    filter: &str,
    time: usize,
    terms: impl IntoIterator<Item = (f64, f64)>,
) -> Option<f64> {
    let mut accumulator = 0.0;
    for (term, (coefficient, observation)) in terms.into_iter().enumerate() {
        let contribution = coefficient * observation;
        if coefficient != 0.0 && observation != 0.0 && contribution == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalUnderflow)
                    .message(format!(
                        "{filter} contribution {term} underflowed at output {time}"
                    ))
                    .build(),
            );
            return None;
        }
        let updated = accumulator + contribution;
        if !contribution.is_finite() || !updated.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message(format!(
                        "{filter} weighted sum overflowed at output {time}, term {term}"
                    ))
                    .build(),
            );
            return None;
        }
        accumulator = updated;
    }
    Some(accumulator)
}

/// Baxter–King band-pass filter (statsmodels `bkfilter`).
///
/// `low` / `high` are periods in samples (e.g. 6 and 32 for quarterly
/// business-cycle bounds). `k` is the lead/lag truncation. The returned cycle
/// has length `y.len() - 2 * k` because the symmetric window has no endpoint
/// estimates.
pub fn bk_filter(
    y: &Vector,
    low: f64,
    high: f64,
    k: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let frequencies = bandpass_frequencies(&mut ctx, "BK", low, high);
    if k == 0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message("Baxter–King lead/lag truncation K must be positive")
                .build(),
        );
    }
    let kernel_len = k.checked_mul(2).and_then(|twice| twice.checked_add(1));
    if kernel_len.is_none() {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message(format!(
                    "Baxter–King K={k} is too large to form a 2K+1 kernel"
                ))
                .build(),
        );
    }
    if y.is_empty() || !y.as_slice().iter().all(|value| value.is_finite()) {
        return Err(ctx.finish_failure());
    }
    let ((w1, w2, center), kernel_len) = match (frequencies, kernel_len) {
        (Some(frequencies), Some(kernel_len)) if k > 0 => (frequencies, kernel_len),
        _ => return Err(ctx.finish_failure()),
    };
    let n = y.len();
    if n < kernel_len {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!(
                    "Baxter–King K={k} needs n≥{kernel_len}; series has {n}"
                ))
                .meaninglessness(Meaninglessness::vacuous(
                    "band-pass cycle",
                    "the symmetric MA is undefined at the ends when n < 2K+1",
                    "shorten K or lengthen the series",
                ))
                .build(),
        );
        return Err(ctx.finish_failure());
    }

    let mut kernel = vec![0.0; kernel_len];
    kernel[k] = center;
    for lag in 1..=k {
        let lag_f64 = lag as f64;
        let weight =
            ((lag_f64 * w2).sin() - (lag_f64 * w1).sin()) / (lag_f64 * std::f64::consts::PI);
        if !weight.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message(format!("Baxter–King weight is non-finite at lag {lag}"))
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        kernel[k - lag] = weight;
        kernel[k + lag] = weight;
    }
    let kernel_sum = kernel.iter().sum::<f64>();
    let adjustment = kernel_sum / kernel_len as f64;
    if !kernel_sum.is_finite() || !adjustment.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteOutput)
                .message("Baxter–King weight normalization produced a non-finite value")
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    for weight in &mut kernel {
        *weight -= adjustment;
    }
    let qualified = match lfilter(
        &Vector::from_slice(&kernel),
        &Vector::from_slice(&[1.0]),
        y,
        &session.child("shared-valid-convolution"),
    ) {
        Ok(qualified) => qualified,
        Err(failure) => return Err(ctx.merge_failure(failure)),
    };
    let (filtered, report) = qualified.into_parts();
    ctx.report.merge(report);
    let first_valid = kernel_len - 1;
    let cycle = Vector::from_slice(&filtered.as_slice()[first_valid..]);
    ctx.push(
        Issue::builder(IssueCode::SpectralLeakage)
            .severity(Severity::Advisory)
            .message(format!(
                "Baxter–King truncates the ideal band-pass at K={k}; K observations are omitted at each end"
            ))
            .compromise(NumericalCompromise::new(
                "two-sided ideal band-pass",
                "symmetric MA of length 2K+1 with zero-sum weights",
                "finite K leaks power near the cut-off frequencies",
                "align the returned observations with input indices K..n-K",
            ))
            .build(),
    );
    ctx.finish(cycle)
}

/// Christiano–Fitzgerald asymmetric band-pass (statsmodels `cffilter`).
///
/// The endpoint-to-endpoint drift is removed before applying the random-walk
/// optimal asymmetric endpoint weights used by statsmodels with `drift=true`.
pub fn cf_filter(y: &Vector, low: f64, high: f64, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let frequencies = bandpass_frequencies(&mut ctx, "CF", low, high);
    let n = y.len();
    if n < 3 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("Christiano–Fitzgerald needs n≥3; series has {n}"))
                .meaninglessness(Meaninglessness::vacuous(
                    "asymmetric band-pass cycle",
                    "a series shorter than 3 cannot identify a cycle after drift removal",
                    "lengthen the series",
                ))
                .build(),
        );
    }
    if y.is_empty()
        || !y.as_slice().iter().all(|value| value.is_finite())
        || frequencies.is_none()
        || n < 3
    {
        return Err(ctx.finish_failure());
    }
    let (w1, w2, center) = match frequencies {
        Some(frequencies) => frequencies,
        None => return Err(ctx.finish_failure()),
    };

    let endpoint_change = y[n - 1] - y[0];
    if !endpoint_change.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::NumericalOverflow)
                .message("CF endpoint difference overflowed while estimating deterministic drift")
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    let drift = endpoint_change / (n - 1) as f64;
    if endpoint_change != 0.0 && drift == 0.0 {
        ctx.push(
            Issue::builder(IssueCode::NumericalUnderflow)
                .message("CF endpoint drift underflowed to zero")
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    let mut adjusted = Vec::with_capacity(n);
    for (time, observation) in y.as_slice().iter().copied().enumerate() {
        let trend = time as f64 * drift;
        let value = observation - trend;
        if !trend.is_finite() || !value.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message(format!(
                        "CF drift adjustment overflowed at observation {time}"
                    ))
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        adjusted.push(value);
    }
    ctx.push(
        Issue::builder(IssueCode::SpectralLeakage)
            .severity(Severity::Advisory)
            .message("CF removes the endpoint-to-endpoint deterministic drift before filtering")
            .compromise(NumericalCompromise::new(
                "CF random-walk filter without deterministic drift",
                "subtract t * (y[n-1] - y[0]) / (n-1)",
                "the endpoint slope is treated as deterministic drift",
                "interpret the returned cycle relative to that drift assumption",
            ))
            .build(),
    );

    let mut coefficients = vec![0.0; n - 1];
    coefficients[0] = center;
    let mut prefix = vec![0.0; n - 1];
    for lag in 1..n - 1 {
        let lag_f64 = lag as f64;
        coefficients[lag] =
            ((lag_f64 * w2).sin() - (lag_f64 * w1).sin()) / (lag_f64 * std::f64::consts::PI);
        prefix[lag] = prefix[lag - 1] + coefficients[lag];
        if !coefficients[lag].is_finite() || !prefix[lag].is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message(format!("CF coefficient accumulation failed at lag {lag}"))
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
    }

    let mut cycle = Vector::zeros(n);
    for time in 0..n {
        let right_lags = (n - time).saturating_sub(2);
        let left_lags = time.saturating_sub(1);
        let right_sum = prefix[right_lags];
        let left_sum = prefix[left_lags];
        let last_weight = -0.5 * coefficients[0] - right_sum;
        let first_weight = -coefficients[0] - right_sum - left_sum - last_weight;

        let terms = std::iter::once((coefficients[0], adjusted[time]))
            .chain((1..=right_lags).map(|lag| (coefficients[lag], adjusted[time + lag])))
            .chain(std::iter::once((last_weight, adjusted[n - 1])))
            .chain((1..=left_lags).map(|lag| (coefficients[lag], adjusted[time - lag])))
            .chain(std::iter::once((first_weight, adjusted[0])));
        cycle[time] = match checked_filter_sum(&mut ctx, "CF", time, terms) {
            Some(value) => value,
            None => return Err(ctx.finish_failure()),
        };
    }
    ctx.finish(cycle)
}

/// Local linear trend (unobserved-components level + slope).
#[derive(Clone, Debug)]
pub struct LocalLinearTrend {
    /// Observation variance \(\sigma_\varepsilon^2\).
    pub obs_var: f64,
    /// Level innovation \(\sigma_\eta^2\).
    pub level_var: f64,
    /// Slope innovation \(\sigma_\zeta^2\).
    pub slope_var: f64,
    /// Innovation covariance between level and slope.
    pub level_slope_cov: f64,
    /// Prior mean of the level at the first observation.
    pub initial_level: f64,
    /// Prior mean of the slope at the first observation.
    pub initial_slope: f64,
    /// Prior variance of the initial level.
    pub initial_level_var: f64,
    /// Prior variance of the initial slope.
    pub initial_slope_var: f64,
    /// Prior covariance between the initial level and slope.
    pub initial_level_slope_cov: f64,
}

impl Default for LocalLinearTrend {
    fn default() -> Self {
        Self {
            obs_var: 1.0,
            level_var: 0.1,
            slope_var: 0.01,
            level_slope_cov: 0.0,
            initial_level: 0.0,
            initial_slope: 0.0,
            initial_level_var: 100.0,
            initial_slope_var: 100.0,
            initial_level_slope_cov: 0.0,
        }
    }
}

impl LocalLinearTrend {
    /// Default local linear trend.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Kalman-smoothed local linear trend.
#[derive(Clone, Debug)]
pub struct FittedLocalLinearTrend {
    /// Smoothed level.
    pub level: Vector,
    /// Smoothed slope.
    pub slope: Vector,
    /// Irregular \(y - \mathrm{level}\).
    pub irregular: Vector,
    /// Smoothed covariance of `[level, slope]` at each time.
    pub state_covariance: Vec<Matrix>,
    /// Exact Gaussian log-likelihood under the supplied parameters.
    pub log_likelihood: f64,
}

impl FitSeries for LocalLinearTrend {
    type Fitted = FittedLocalLinearTrend;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLocalLinearTrend>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        if n == 0 {
            return ctx.finish(FittedLocalLinearTrend {
                level: Vector::zeros(0),
                slope: Vector::zeros(0),
                irregular: Vector::zeros(0),
                state_covariance: Vec::new(),
                log_likelihood: f64::NAN,
            });
        }
        if self.obs_var <= 0.0 && self.level_var <= 0.0 && self.slope_var <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("all local-linear-trend variances are 0; the state is a perfect line")
                    .meaninglessness(Meaninglessness::vacuous(
                        "level and slope",
                        "zero process and measurement noise interpolates the first two points",
                        "set at least one positive variance",
                    ))
                    .build(),
            );
        }
        if n < 3 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("local linear trend needs n≥3 to separate level, slope, and irregular")
                    .build(),
            );
        }
        let model = match LinearGaussianStateSpace::new(
            Matrix::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) | (0, 1) | (1, 1) => 1.0,
                _ => 0.0,
            }),
            Matrix::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 0.0 }),
            Matrix::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) => self.level_var,
                (1, 1) => self.slope_var,
                _ => self.level_slope_cov,
            }),
            Matrix::from_fn(1, 1, |_, _| self.obs_var),
            Vector::zeros(2),
            Vector::zeros(1),
            Vector::from_slice(&[self.initial_level, self.initial_slope]),
            Matrix::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) => self.initial_level_var,
                (1, 1) => self.initial_slope_var,
                _ => self.initial_level_slope_cov,
            }),
            ctx.policy.clone(),
        ) {
            Ok(model) => model,
            Err(failure) => return Err(ctx.merge_failure(failure)),
        };
        let qualified = match model.smooth(
            &Matrix::from_vector(y),
            None,
            &session.child("local-linear-trend"),
        ) {
            Ok(qualified) => qualified,
            Err(failure) => return Err(ctx.merge_failure(failure)),
        };
        let (smoothed, nested_report) = qualified.into_parts();
        ctx.report.merge(nested_report);
        let level = Vector::from_iter((0..n).map(|time| smoothed.smoothed_mean.get(time, 0)));
        let slope = Vector::from_iter((0..n).map(|time| smoothed.smoothed_mean.get(time, 1)));
        let irregular = y.sub(&level);
        ctx.finish(FittedLocalLinearTrend {
            level,
            slope,
            irregular,
            state_covariance: smoothed.smoothed_covariance,
            log_likelihood: smoothed.filter.log_likelihood,
        })
    }
}

/// FIR convolution (statsmodels `tsa.filters.filtertools.convolution_filter`).
///
/// Kernel length is not identification `p`.
pub fn convolution_filter(
    y: &Vector,
    kernel: &Vector,
    session: &Session,
) -> Result<Qualified<Vector>> {
    lfilter(
        kernel,
        &Vector::from_slice(&[1.0]),
        y,
        &session.child("shared-lfilter"),
    )
}

/// IIR recursion \(y_t = x_t + \sum_j a_j y_{t-j}\) (statsmodels `recursive_filter`).
///
/// AR coefficient count is not identification `p`.
pub fn recursive_filter(x: &Vector, ar: &Vector, session: &Session) -> Result<Qualified<Vector>> {
    let denominator = Vector::from_iter(
        std::iter::once(1.0).chain(ar.as_slice().iter().map(|coefficient| -coefficient)),
    );
    lfilter(
        &Vector::from_slice(&[1.0]),
        &denominator,
        x,
        &session.child("shared-lfilter"),
    )
}

/// Multi-input single-output FIR (statsmodels `miso_lfilter`).
///
/// Each column of `x` is convolved with `kernel` and the channels are summed.
/// Filter length is not identification `p`.
pub fn miso_lfilter(x: &Matrix, kernel: &Vector, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    ctx.report.set_sample_shape(x.nrows(), x.ncols());
    if x.nrows() == 0 || x.ncols() == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message(format!(
                    "miso_lfilter requires a non-empty time-by-channel matrix; got {}×{}",
                    x.nrows(),
                    x.ncols()
                ))
                .build(),
        );
        return Err(ctx.finish_failure());
    }

    let denominator = Vector::from_slice(&[1.0]);
    let mut output = Vector::zeros(x.nrows());
    for column in 0..x.ncols() {
        let channel = Vector::from_iter((0..x.nrows()).map(|row| x.get(row, column)));
        let qualified = match lfilter(
            kernel,
            &denominator,
            &channel,
            &session.child(format!("shared-lfilter-channel-{column}")),
        ) {
            Ok(qualified) => qualified,
            Err(failure) => return Err(ctx.merge_failure(failure)),
        };
        let (filtered, report) = qualified.into_parts();
        ctx.report.merge(report);
        for time in 0..x.nrows() {
            let combined = output[time] + filtered[time];
            if !combined.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NumericalOverflow)
                        .message(format!(
                            "miso_lfilter channel accumulation overflowed at time {time}, channel {column}"
                        ))
                        .build(),
                );
                return Err(ctx.finish_failure());
            }
            output[time] = combined;
        }
    }
    ctx.finish(output)
}

/// Single-input single-output linear filter (SciPy / statsmodels `lfilter`).
///
/// \(a_0 y_t = \sum_i b_i x_{t-i} - \sum_{j\ge 1} a_j y_{t-j}\). Filter length
/// is not identification `p`.
pub fn lfilter(b: &Vector, a: &Vector, x: &Vector, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let mut invalid_input = false;
    if x.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("lfilter requires at least one input sample")
                .build(),
        );
        invalid_input = true;
    }
    if b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("lfilter received an empty numerator")
                .build(),
        );
        invalid_input = true;
    }
    if !b.as_slice().iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("lfilter numerator is non-finite")
                .build(),
        );
        invalid_input = true;
    }
    if !a.is_empty() && !a.as_slice().iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("lfilter denominator is non-finite")
                .build(),
        );
        invalid_input = true;
    }
    if !x.as_slice().iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("lfilter input is non-finite")
                .build(),
        );
        invalid_input = true;
    }
    let a0 = a.as_slice().first().copied().unwrap_or(1.0);
    if a0 == 0.0 {
        ctx.push(
            Issue::builder(IssueCode::ScaleFactorZero)
                .message("lfilter denominator leading coefficient a[0] is zero")
                .build(),
        );
        invalid_input = true;
    }
    if invalid_input {
        return Err(ctx.finish_failure());
    }

    let mut normalized_b = Vec::with_capacity(b.len());
    for (lag, coefficient) in b.as_slice().iter().copied().enumerate() {
        let normalized = coefficient / a0;
        if !normalized.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message(format!(
                        "lfilter numerator coefficient normalization overflowed at lag {lag}"
                    ))
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if coefficient != 0.0 && normalized == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalUnderflow)
                    .message(format!(
                        "lfilter numerator coefficient normalization underflowed at lag {lag}"
                    ))
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        normalized_b.push(normalized);
    }
    let mut normalized_a = Vec::with_capacity(a.len().saturating_sub(1));
    for (lag, coefficient) in a.as_slice().iter().copied().enumerate().skip(1) {
        let normalized = coefficient / a0;
        if !normalized.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::NumericalOverflow)
                    .message(format!(
                        "lfilter denominator coefficient normalization overflowed at lag {lag}"
                    ))
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        if coefficient != 0.0 && normalized == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::NumericalUnderflow)
                    .message(format!(
                        "lfilter denominator coefficient normalization underflowed at lag {lag}"
                    ))
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        normalized_a.push(normalized);
    }

    let mut y = Vector::zeros(x.len());
    for t in 0..x.len() {
        let mut s = 0.0_f64;
        for i in 0..normalized_b.len() {
            if t >= i {
                let contribution = normalized_b[i] * x[t - i];
                let updated = s + contribution;
                if !contribution.is_finite() || !updated.is_finite() {
                    ctx.push(
                        Issue::builder(IssueCode::NumericalOverflow)
                            .message(format!(
                                "lfilter feed-forward recurrence overflowed at output {t}, numerator lag {i}"
                            ))
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                }
                s = updated;
            }
        }
        for j in 1..=normalized_a.len() {
            if t >= j {
                let contribution = normalized_a[j - 1] * y[t - j];
                let updated = s - contribution;
                if !contribution.is_finite() || !updated.is_finite() {
                    ctx.push(
                        Issue::builder(IssueCode::NumericalOverflow)
                            .message(format!(
                                "lfilter feedback recurrence overflowed at output {t}, denominator lag {j}"
                            ))
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                }
                s = updated;
            }
        }
        y[t] = s;
    }
    ctx.finish(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bk_annihilates_a_linear_trend() {
        let y = Vector::from_iter((0..48).map(|i| i as f64));
        let c = bk_filter(&y, 6.0, 32.0, 8, &Session::new("bk", "fit"))
            .expect("bk")
            .value;
        assert_eq!(c.len(), 32);
        let maximum_error = c.max_abs();
        // Measured 2.720e-15 on 2026-09-03; tolerance is approximately 3.97x.
        assert!(maximum_error <= 1.08e-14);
        let ker = Vector::from_slice(&[0.25, 0.5, 0.25]);
        let conv = convolution_filter(&y, &ker, &Session::new("conv", "fit"))
            .expect("conv")
            .value;
        assert_eq!(conv.len(), 48);
        assert!(conv.as_slice().iter().all(|v| v.is_finite()));
        let ar = Vector::from_slice(&[0.1]);
        let rec = recursive_filter(&y, &ar, &Session::new("rec", "fit"))
            .expect("rec")
            .value;
        assert!(rec.as_slice().iter().all(|v| v.is_finite()));
        let x2 = Matrix::from_fn(
            48,
            2,
            |i, j| {
                if j == 0 {
                    i as f64
                } else {
                    (i as f64).sin()
                }
            },
        );
        let miso = miso_lfilter(&x2, &ker, &Session::new("miso", "fit"))
            .expect("miso")
            .value;
        assert_eq!(miso.len(), 48);
        assert!(miso.as_slice().iter().all(|v| v.is_finite()));
        let one = Vector::from_slice(&[1.0]);
        let lf = lfilter(&ker, &one, &y, &Session::new("lf", "fit"))
            .expect("lfilter")
            .value;
        assert_eq!(lf.len(), 48);
        assert!(lf.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cf_annihilates_a_linear_trend() {
        let y = Vector::from_iter((0..48).map(|i| i as f64));
        let c = cf_filter(&y, 6.0, 32.0, &Session::new("cf", "fit"))
            .expect("cf")
            .value;
        let maximum_error = c.max_abs();
        assert_eq!(maximum_error, 0.0);
    }

    #[test]
    fn bandpass_filters_are_equivariant_to_scale_and_affine_trend() {
        let input = Vector::from_iter((0..40).map(|time| {
            let time = time as f64;
            (0.37 * time).sin() + 0.4 * (0.11 * time).cos() + time.rem_euclid(5.0) - 2.0
        }));
        let scale = -1.75;
        let transformed = Vector::from_iter(
            input
                .as_slice()
                .iter()
                .copied()
                .enumerate()
                .map(|(time, value)| scale * value + 2.25 - 0.125 * time as f64),
        );

        let bk = bk_filter(&input, 4.0, 15.0, 6, &Session::new("bk", "affine"))
            .expect("BK base series")
            .value;
        let bk_transformed = bk_filter(
            &transformed,
            4.0,
            15.0,
            6,
            &Session::new("bk", "affine-transformed"),
        )
        .expect("BK affine transform")
        .value;
        let bk_maximum_error = bk
            .as_slice()
            .iter()
            .zip(bk_transformed.as_slice())
            .map(|(base, actual)| (scale * base - actual).abs())
            .fold(0.0_f64, f64::max);

        let cf = cf_filter(&input, 4.0, 15.0, &Session::new("cf", "affine"))
            .expect("CF base series")
            .value;
        let cf_transformed = cf_filter(
            &transformed,
            4.0,
            15.0,
            &Session::new("cf", "affine-transformed"),
        )
        .expect("CF affine transform")
        .value;
        let cf_maximum_error = cf
            .as_slice()
            .iter()
            .zip(cf_transformed.as_slice())
            .map(|(base, actual)| (scale * base - actual).abs())
            .fold(0.0_f64, f64::max);

        // Measured 8.882e-16 on 2026-09-03; tolerance is approximately 3.94x.
        assert!(bk_maximum_error <= 3.5e-15);
        // Measured 3.109e-15 on 2026-09-03; tolerance is approximately 3.99x.
        assert!(cf_maximum_error <= 1.24e-14);
    }

    #[test]
    fn bandpass_filters_match_statsmodels_source_oracles() {
        // Expected values independently evaluated from the exact formulas in
        // statsmodels 0.15.1 `bk_filter.bkfilter` and `cf_filter.cffilter`.
        let input = Vector::from_slice(&[3.0, -1.0, 4.0, 1.0, 5.0, -9.0, 2.0, 6.0, 5.0, 8.0]);
        let bk_expected = [
            -0.5265020371886062,
            3.8418066541846763,
            -0.057590459792815074,
            -5.398675658045285,
            -2.151247968362353,
            3.333813379372154,
        ];
        let bk = bk_filter(
            &input,
            3.0,
            8.0,
            2,
            &Session::new("bk", "statsmodels-source-oracle"),
        )
        .expect("BK source oracle")
        .value;
        let bk_maximum_error = bk
            .as_slice()
            .iter()
            .zip(bk_expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        // Measured 8.882e-16 on 2026-09-03; tolerance is approximately 3.94x.
        assert!(bk_maximum_error <= 3.5e-15);

        let cf_expected = [
            0.00259779537522431,
            -1.9947494483884909,
            0.2931742442543208,
            4.485698410785202,
            0.5343465896633119,
            -5.157127472640472,
            -1.49256568837228,
            3.1990081028426953,
            1.0221898803365932,
            -0.596224483959243,
        ];
        let cf = cf_filter(
            &input,
            3.0,
            8.0,
            &Session::new("cf", "statsmodels-source-oracle"),
        )
        .expect("CF source oracle")
        .value;
        let cf_maximum_error = cf
            .as_slice()
            .iter()
            .zip(cf_expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        // Measured 1.776e-15 on 2026-09-03; tolerance is approximately 4.00x.
        assert!(cf_maximum_error <= 7.1e-15);
    }

    #[test]
    fn bandpass_filters_reject_invalid_domains_without_substitution() {
        let input = Vector::from_slice(&[1.0, -2.0, 3.0, -4.0, 5.0]);
        let zero_k = bk_filter(&input, 2.0, 8.0, 0, &Session::new("bk", "zero-k"))
            .expect_err("K=0 cannot be silently replaced by K=1");
        assert_eq!(zero_k.primary.code, IssueCode::InvalidParameter);

        let length_overflow = bk_filter(
            &input,
            2.0,
            8.0,
            usize::MAX,
            &Session::new("bk", "length-overflow"),
        )
        .expect_err("2K+1 overflow must fail before allocation");
        assert_eq!(length_overflow.primary.code, IssueCode::InvalidParameter);

        let short = bk_filter(
            &Vector::from_slice(&[1.0, 2.0, 4.0, 8.0]),
            2.0,
            8.0,
            2,
            &Session::new("bk", "short"),
        )
        .expect_err("a short series cannot return zero-filled output");
        assert_eq!(short.primary.code, IssueCode::WindowTooShort);

        let invalid_period = cf_filter(&input, 8.0, 8.0, &Session::new("cf", "invalid-period"))
            .expect_err("equal cut-off periods are invalid");
        assert_eq!(invalid_period.primary.code, IssueCode::InvalidParameter);

        let _bk_result = bk_filter(&input, 2.0, 8.0, 1, &Session::new("bk", "nyquist-boundary"))
            .expect("low=2 is the valid Nyquist boundary");
        let _cf_result = cf_filter(&input, 2.0, 8.0, &Session::new("cf", "nyquist-boundary"))
            .expect("low=2 is the valid Nyquist boundary");

        let short_cf = cf_filter(
            &Vector::from_slice(&[1.0, 2.0]),
            2.0,
            8.0,
            &Session::new("cf", "short"),
        )
        .expect_err("CF needs enough observations for endpoint filtering");
        assert_eq!(short_cf.primary.code, IssueCode::WindowTooShort);

        let overflow = cf_filter(
            &Vector::from_slice(&[f64::MAX, 0.0, -f64::MAX]),
            2.0,
            8.0,
            &Session::new("cf", "drift-overflow"),
        )
        .expect_err("CF cannot fall back when endpoint detrending overflows");
        assert!(overflow.report.contains(IssueCode::NumericalOverflow));
    }

    #[test]
    fn local_linear_tracks_a_ramp() {
        let y = Vector::from_iter((0..20).map(|i| 3.0 + 0.5 * i as f64));
        let q = LocalLinearTrend {
            obs_var: 0.05,
            level_var: 0.01,
            slope_var: 1e-4,
            ..LocalLinearTrend::default()
        }
        .fit_series(&y, &Session::new("llt", "fit"))
        .expect("llt");
        assert!((q.value.level[10] - y[10]).abs() < 0.5);
        assert!(q.value.slope.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn local_linear_adapter_matches_the_decimal_state_space_oracle() {
        let y = Vector::from_slice(&[0.0, 2.0, -1.0, 3.0, 0.0]);
        let result = LocalLinearTrend {
            obs_var: 0.4,
            level_var: 0.1,
            slope_var: 0.05,
            level_slope_cov: 0.02,
            initial_level: 0.0,
            initial_slope: 0.0,
            initial_level_var: 1.0,
            initial_slope_var: 1.0,
            initial_level_slope_cov: 0.2,
        }
        .fit_series(&y, &Session::new("llt", "oracle"))
        .expect("local linear trend")
        .value;

        let expected_level = [
            0.42621045176067277,
            0.7321260470446674,
            0.7161871541175819,
            1.0601545980283152,
            0.9032470769286324,
        ];
        let maximum_error = result
            .level
            .as_slice()
            .iter()
            .zip(expected_level)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        // Measured 2.220e-16 on 2026-09-03; tolerance is approximately 4x.
        assert!(
            maximum_error <= 9.0e-16,
            "maximum level error={maximum_error:e}"
        );
        let likelihood_error = (result.log_likelihood + 18.15112930519994).abs();
        // Measured 3.553e-15 on 2026-09-03; tolerance is approximately 4x.
        assert!(likelihood_error <= 1.5e-14);
        assert!(
            result.level[0] > 0.0,
            "regression for the former RTS sign/orientation bug"
        );
        assert_eq!(result.state_covariance.len(), y.len());
    }

    #[test]
    fn lfilter_matches_closed_form_impulse_and_recurrence_identity() {
        let impulse = Vector::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let numerator = Vector::from_slice(&[1.0]);
        let denominator = Vector::from_slice(&[2.0, -1.0]);
        let response = lfilter(
            &numerator,
            &denominator,
            &impulse,
            &Session::new("lfilter", "closed-form"),
        )
        .expect("stable first-order impulse response")
        .value;
        let expected = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625];
        assert_eq!(response.as_slice(), expected);

        let x = Vector::from_slice(&[0.25, -1.0, 2.5, 0.75, -0.5, 1.25, 3.0]);
        let b = Vector::from_slice(&[0.75, -0.2, 0.1]);
        let a = Vector::from_slice(&[1.5, -0.4, 0.15]);
        let actual = lfilter(&b, &a, &x, &Session::new("lfilter", "recurrence"))
            .expect("finite stable recurrence")
            .value;
        let mut maximum_residual = 0.0_f64;
        for t in 0..x.len() {
            let left = a[0] * actual[t]
                + (1..a.len())
                    .filter(|lag| t >= *lag)
                    .map(|lag| a[lag] * actual[t - lag])
                    .sum::<f64>();
            let right = (0..b.len())
                .filter(|lag| t >= *lag)
                .map(|lag| b[lag] * x[t - lag])
                .sum::<f64>();
            maximum_residual = maximum_residual.max((left - right).abs());
        }
        assert!(maximum_residual.is_finite());
        // Measured 2.220e-16 on 2026-09-03; tolerance is approximately 4x.
        assert!(maximum_residual <= 9.0e-16);

        let scaled_b = Vector::from_iter(b.as_slice().iter().map(|value| -8.0 * value));
        let scaled_a = Vector::from_iter(a.as_slice().iter().map(|value| -8.0 * value));
        let scaled = lfilter(
            &scaled_b,
            &scaled_a,
            &x,
            &Session::new("lfilter", "scale-invariance"),
        )
        .expect("common coefficient scaling")
        .value;
        assert_eq!(actual.as_slice(), scaled.as_slice());

        let tiny = f64::MIN_POSITIVE;
        let tiny_scale = lfilter(
            &Vector::from_slice(&[tiny]),
            &Vector::from_slice(&[tiny]),
            &x,
            &Session::new("lfilter", "tiny-common-scale"),
        )
        .expect("representable coefficient ratio must survive a tiny common scale")
        .value;
        assert_eq!(tiny_scale.as_slice(), x.as_slice());
    }

    #[test]
    fn lfilter_rejects_invalid_inputs_and_overflow_without_substitution() {
        let one = Vector::from_slice(&[1.0]);
        let invalid_cases = [
            (
                Vector::zeros(0),
                one.clone(),
                one.clone(),
                IssueCode::EmptyMatrix,
            ),
            (
                one.clone(),
                Vector::from_slice(&[0.0]),
                one.clone(),
                IssueCode::ScaleFactorZero,
            ),
            (
                Vector::from_slice(&[f64::NAN]),
                one.clone(),
                one.clone(),
                IssueCode::NonFiniteInput,
            ),
            (
                one.clone(),
                one.clone(),
                Vector::from_slice(&[f64::INFINITY]),
                IssueCode::NonFiniteInput,
            ),
        ];
        for (b, a, x, expected) in invalid_cases {
            let failure = lfilter(&b, &a, &x, &Session::new("lfilter", "invalid"))
                .expect_err("invalid filter input must fail");
            assert_eq!(failure.primary.code, expected);
        }

        let overflow = lfilter(
            &Vector::from_slice(&[f64::MAX]),
            &one,
            &Vector::from_slice(&[2.0]),
            &Session::new("lfilter", "overflow"),
        )
        .expect_err("overflow cannot be replaced by zero");
        assert_eq!(overflow.primary.code, IssueCode::NumericalOverflow);

        let normalization_overflow = lfilter(
            &Vector::from_slice(&[f64::MAX]),
            &Vector::from_slice(&[f64::MIN_POSITIVE]),
            &one,
            &Session::new("lfilter", "normalization-overflow"),
        )
        .expect_err("unrepresentable normalized coefficient must fail");
        assert_eq!(
            normalization_overflow.primary.code,
            IssueCode::NumericalOverflow
        );
        let normalization_underflow = lfilter(
            &Vector::from_slice(&[f64::from_bits(1)]),
            &Vector::from_slice(&[f64::MAX]),
            &one,
            &Session::new("lfilter", "normalization-underflow"),
        )
        .expect_err("a nonzero coefficient cannot silently normalize to zero");
        assert_eq!(
            normalization_underflow.primary.code,
            IssueCode::NumericalUnderflow
        );

        let constant = lfilter(
            &one,
            &one,
            &Vector::from_slice(&[3.0, 3.0, 3.0]),
            &Session::new("lfilter", "constant"),
        )
        .expect("constant signals are valid filter inputs");
        assert!(constant.report.issues().is_empty());
        assert_eq!(constant.value.as_slice(), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn fir_and_recursive_adapters_share_the_lfilter_kernel() {
        let session = Session::new("filter-adapters", "closed-form");
        let impulse = Vector::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        let kernel = Vector::from_slice(&[0.25, -0.5, 0.75]);
        let fir = convolution_filter(&impulse, &kernel, &session)
            .expect("finite FIR")
            .value;
        assert_eq!(fir.as_slice(), &[0.25, -0.5, 0.75, 0.0, 0.0]);
        let direct_fir = lfilter(&kernel, &Vector::from_slice(&[1.0]), &impulse, &session)
            .expect("direct FIR")
            .value;
        assert_eq!(fir.as_slice(), direct_fir.as_slice());

        let ar = Vector::from_slice(&[0.5]);
        let recursive = recursive_filter(&impulse, &ar, &session)
            .expect("stable first-order recursion")
            .value;
        assert_eq!(recursive.as_slice(), &[1.0, 0.5, 0.25, 0.125, 0.0625]);
        let direct_recursive = lfilter(
            &Vector::from_slice(&[1.0]),
            &Vector::from_slice(&[1.0, -0.5]),
            &impulse,
            &session,
        )
        .expect("direct recursion")
        .value;
        assert_eq!(recursive.as_slice(), direct_recursive.as_slice());

        let overflow = recursive_filter(
            &Vector::from_slice(&[f64::MAX, f64::MAX]),
            &Vector::from_slice(&[1.0]),
            &session,
        )
        .expect_err("recursive overflow must not be replaced by zero");
        assert_eq!(overflow.primary.code, IssueCode::NumericalOverflow);
    }

    #[test]
    fn multi_input_filter_is_the_sum_of_shared_lfilter_channels() {
        let session = Session::new("miso-lfilter", "linearity");
        let x = Matrix::from_row_major(4, 2, &[1.0, 4.0, 2.0, -1.0, 0.0, 3.0, -2.0, 1.0]);
        let kernel = Vector::from_slice(&[0.5, -0.25]);
        let actual = miso_lfilter(&x, &kernel, &session)
            .expect("finite MISO filter")
            .value;
        let first = lfilter(
            &kernel,
            &Vector::from_slice(&[1.0]),
            &Vector::from_iter((0..x.nrows()).map(|row| x.get(row, 0))),
            &session,
        )
        .expect("first channel")
        .value;
        let second = lfilter(
            &kernel,
            &Vector::from_slice(&[1.0]),
            &Vector::from_iter((0..x.nrows()).map(|row| x.get(row, 1))),
            &session,
        )
        .expect("second channel")
        .value;
        let expected = first.add(&second);
        assert_eq!(actual.as_slice(), expected.as_slice());
        assert_eq!(actual.as_slice(), &[2.5, -0.75, 1.25, -1.25]);

        let overflow = miso_lfilter(
            &Matrix::from_row_major(1, 2, &[f64::MAX, f64::MAX]),
            &Vector::from_slice(&[1.0]),
            &session,
        )
        .expect_err("cross-channel overflow must not be replaced by zero");
        assert_eq!(overflow.primary.code, IssueCode::NumericalOverflow);
        let nonfinite = miso_lfilter(
            &Matrix::from_row_major(1, 1, &[f64::NAN]),
            &Vector::from_slice(&[1.0]),
            &session,
        )
        .expect_err("a non-finite channel must fail through lfilter");
        assert_eq!(nonfinite.primary.code, IssueCode::NonFiniteInput);
        let empty = miso_lfilter(&Matrix::zeros(0, 1), &kernel, &session)
            .expect_err("empty MISO input must fail");
        assert_eq!(empty.primary.code, IssueCode::EmptyMatrix);
    }
}
