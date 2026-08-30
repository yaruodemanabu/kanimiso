//! Time-series identification, decomposition, and forecasting primitives.
//!
//! Covers the computational surface of **statsmodels.tsa** and **sktime**
//! forecasters: ACF/PACF, seasonal decomposition, exponential smoothing,
//! ARIMA/SARIMA (Hannan–Rissanen), VAR, Theta, Kalman local level, HP filter,
//! GARCH(1,1), Croston, and naive / drift baselines.
//!
//! Every public fit or transform uses [`crate::context::FitCtx`]. Short series,
//! non-positive data under a multiplicative / log model, and AR polynomials
//! that fail a simple causality radius check are recorded as [`signlred`]
//! issues — never silently.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{chol_solve, least_squares, thin_svd};
use crate::rng::Rng;

pub use crate::filters::{
    bk_filter, cf_filter, lfilter, miso_lfilter, FittedLocalLinearTrend, LocalLinearTrend,
};
use crate::stats::{HypothesisTest, KpssResult};
use crate::traits::{Fit, FitSeries, PartialFit};
use crate::validate::{inspect_identification, inspect_xy};
use ojizou_san::{IncrementalExplain, Session};
use signlred::{
    scan_finite, slice_stats, IncrementalQuality, Issue, IssueCode, Meaninglessness,
    NumericalCompromise, Qualified, Report, Result, Severity,
};

/// Relative forecast steps (sktime `ForecastingHorizon`).
///
/// Horizon length is not identification `p`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForecastingHorizon {
    /// 1-based relative steps.
    pub steps: Vec<usize>,
}

impl ForecastingHorizon {
    /// Steps `1, …, h`.
    pub fn relative(h: usize) -> Self {
        let n = h.max(1);
        Self {
            steps: (1..=n).collect(),
        }
    }

    /// Number of forecast steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the horizon is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Causal train/test cut of a series (sktime `temporal_train_test_split`).
///
/// Split sizes are not identification `p`.
pub fn temporal_train_test_split(
    y: &Vector,
    test_size: f64,
    session: &Session,
) -> Result<Qualified<(Vector, Vector)>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let n = y.len();
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("temporal_train_test_split needs at least two observations")
                .build(),
        );
        return ctx.finish((y.clone(), Vector::zeros(0)));
    }
    let frac = if test_size.is_finite() && test_size > 0.0 && test_size < 1.0 {
        test_size
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "temporal_train_test_split test_size={test_size}; using 0.25"
                ))
                .build(),
        );
        0.25
    };
    let mut n_test = (n as f64 * frac).round() as usize;
    n_test = n_test.clamp(1, n - 1);
    let n_train = n - n_test;
    let train = Vector::from_iter((0..n_train).map(|i| y[i]));
    let test = Vector::from_iter((n_train..n).map(|i| y[i]));
    ctx.finish((train, test))
}

/// Sample autocorrelation `ρ_0, …, ρ_{nlags}` (biased, mean-corrected).
pub fn acf(y: &Vector, nlags: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    if nlags + 1 > y.len() && !y.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("acf nlags={nlags} ≥ n={}", y.len()))
                .build(),
        );
    }
    let rho = acf_raw(y.as_slice(), nlags);
    if y.len() >= 2 {
        let st = slice_stats(y.as_slice());
        if st.is_constant(ctx.policy.near_zero_variance) {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("ACF of a constant series is undefined after lag 0")
                    .meaninglessness(Meaninglessness::vacuous(
                        "sample autocorrelation",
                        "γ_0 = 0; every ratio is 0/0",
                        "do not report ACF on a degenerate series",
                    ))
                    .build(),
            );
        }
    }
    ctx.finish(Vector::from_iter(rho))
}

/// Biased sample autocovariance \(\gamma_0,\ldots,\gamma_{\mathrm{nlags}}\)
/// (statsmodels `acovf`).
///
/// Lag count is not identification `p`.
pub fn acovf(y: &Vector, nlags: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    if nlags + 1 > y.len() && !y.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("acovf nlags={nlags} ≥ n={}", y.len()))
                .build(),
        );
    }
    let st = slice_stats(y.as_slice());
    if y.len() >= 2 && st.is_constant(ctx.policy.near_zero_variance) {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("acovf of a constant series is the zero map after centering")
                .meaninglessness(Meaninglessness::vacuous(
                    "sample autocovariance",
                    "every deviation from the mean is 0",
                    "do not report acovf on a degenerate series",
                ))
                .build(),
        );
    }
    let n = y.len();
    let mut out = Vector::zeros(nlags + 1);
    if n == 0 {
        return ctx.finish(out);
    }
    for k in 0..=nlags {
        if k >= n {
            out[k] = f64::NAN;
            continue;
        }
        let mut g = 0.0;
        let mut c = 0.0;
        for t in k..n {
            if y[t].is_finite() && y[t - k].is_finite() {
                g += (y[t] - st.mean) * (y[t - k] - st.mean);
                c += 1.0;
            }
        }
        out[k] = if n > 0 { g / n as f64 } else { 0.0 };
        let _ = c;
    }
    ctx.finish(out)
}

/// Partial autocorrelation via Yule–Walker systems solved by OLS at each lag.
///
/// `φ_{kk}` is the last coefficient of the order-`k` Toeplitz system
/// `R φ = r`. This is the Durbin–Levinson / Yule–Walker PACF.
pub fn pacf(y: &Vector, nlags: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let rho = acf_raw(y.as_slice(), nlags);
    let mut out = Vector::zeros(nlags + 1);
    out[0] = 1.0;
    for k in 1..=nlags {
        if k >= y.len() {
            out[k] = f64::NAN;
            continue;
        }
        let rmat = Matrix::from_fn(k, k, |i, j| rho[i.abs_diff(j)]);
        let rhs = Vector::from_iter((1..=k).map(|i| rho[i]));
        match statistical_ols(&mut ctx, &rmat, &rhs) {
            Some(phi) => out[k] = phi[k - 1],
            None => {
                // Durbin–Levinson fallback if the Toeplitz solve is singular.
                out[k] = durbin_levinson_kk(&rho, k);
                ctx.push(
                    Issue::builder(IssueCode::RidgeFallbackUsed)
                        .message(format!(
                            "PACF lag {k}: Yule–Walker OLS failed; Durbin–Levinson used"
                        ))
                        .compromise(NumericalCompromise::new(
                            "Yule–Walker Rφ = r via OLS",
                            "Durbin–Levinson φ_{kk}",
                            "the sample Toeplitz matrix was singular at this lag",
                            "PACF at this lag is the Levinson recurrence, not a unique OLS solve",
                        ))
                        .build(),
                );
            }
        }
    }
    ctx.finish(out)
}

/// Cross-covariance \(\gamma_{xy}(0),\ldots,\gamma_{xy}(\mathrm{nlags})\) (statsmodels `ccovf`).
///
/// Lag count is not identification `p`.
pub fn ccovf(x: &Vector, y: &Vector, nlags: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, x);
    inspect_univariate(&mut ctx, y);
    let n = x.len().min(y.len());
    if nlags + 1 > n && n > 0 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("ccovf nlags={nlags} ≥ n={n}"))
                .build(),
        );
    }
    let mx = x.as_slice().iter().take(n).sum::<f64>() / n.max(1) as f64;
    let my = y.as_slice().iter().take(n).sum::<f64>() / n.max(1) as f64;
    let mut out = Vector::zeros(nlags + 1);
    for k in 0..=nlags {
        if k >= n {
            out[k] = f64::NAN;
            continue;
        }
        let mut g = 0.0;
        for t in k..n {
            if x[t].is_finite() && y[t - k].is_finite() {
                g += (x[t] - mx) * (y[t - k] - my);
            }
        }
        out[k] = g / n.max(1) as f64;
    }
    ctx.finish(out)
}

/// OLS partial autocorrelations (statsmodels `pacf_ols`).
///
/// \(\varphi_{kk}\) is the last slope of \(y_t\) on \(y_{t-1},\ldots,y_{t-k}\).
/// Lag count is not identification `p`.
pub fn pacf_ols(y: &Vector, nlags: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let mut out = Vector::zeros(nlags + 1);
    out[0] = 1.0;
    for k in 1..=nlags {
        if k + 2 >= y.len() {
            out[k] = f64::NAN;
            continue;
        }
        let n = y.len() - k;
        let m = Matrix::from_fn(n, k + 1, |i, j| {
            if j == 0 {
                1.0
            } else {
                y[i + k - j]
            }
        });
        let z = Vector::from_iter((0..n).map(|i| y[i + k]));
        let mut scratch = Report::new("pacf_ols", "ols");
        let coef = least_squares(&mut scratch, &m, &z, &ctx.policy);
        out[k] = coef
            .as_ref()
            .and_then(|c| c.as_slice().get(k).copied())
            .unwrap_or(f64::NAN);
    }
    ctx.finish(out)
}

/// Additive seasonal decomposition (trend moving average + seasonal averages).
pub fn seasonal_decompose(
    y: &Vector,
    period: usize,
    session: &Session,
) -> Result<Qualified<SeasonalDecomposition>> {
    seasonal_decompose_inner(y, period, false, session)
}

/// STL-like decomposition: centered moving-average trend plus seasonal means.
pub fn stl_like(
    y: &Vector,
    period: usize,
    session: &Session,
) -> Result<Qualified<SeasonalDecomposition>> {
    seasonal_decompose_inner(y, period, false, session)
}

/// Additive classical seasonal decomposition result.
#[derive(Clone, Debug)]
pub struct SeasonalDecomposition {
    /// Original series.
    pub observed: Vector,
    /// Moving-average trend (endpoints linearly extended).
    pub trend: Vector,
    /// Seasonal component (mean-zero over a period).
    pub seasonal: Vector,
    /// `y − trend − seasonal`.
    pub resid: Vector,
    /// Seasonal period.
    pub period: usize,
}

/// Holt–Winters additive (or multiplicative) seasonal exponential smoothing.
#[derive(Clone, Debug)]
pub struct HoltWinters {
    /// Seasonal period `s`.
    pub period: usize,
    /// Level smoothing; `None` selects by in-sample SSE grid search.
    pub alpha: Option<f64>,
    /// Trend smoothing.
    pub beta: Option<f64>,
    /// Seasonal smoothing.
    pub gamma: Option<f64>,
    /// Multiplicative seasonality (requires a strictly positive series).
    pub multiplicative: bool,
}

impl Default for HoltWinters {
    fn default() -> Self {
        Self {
            period: 12,
            alpha: None,
            beta: None,
            gamma: None,
            multiplicative: false,
        }
    }
}

impl HoltWinters {
    /// Additive Holt–Winters with period `period`.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            ..Self::default()
        }
    }
}

/// Fitted Holt–Winters state used for forecasting.
#[derive(Clone, Debug)]
pub struct FittedHoltWinters {
    /// Level smoothing used.
    pub alpha: f64,
    /// Trend smoothing used.
    pub beta: f64,
    /// Seasonal smoothing used.
    pub gamma: f64,
    /// Terminal level.
    pub level: f64,
    /// Terminal trend.
    pub trend: f64,
    /// Terminal seasonal factors (length `period`).
    pub seasonal: Vector,
    /// In-sample fitted values.
    pub fitted: Vector,
    /// In-sample residuals.
    pub resid: Vector,
    /// Seasonal period.
    pub period: usize,
    /// Whether the recurrence was multiplicative.
    pub multiplicative: bool,
    /// Length of the training series (needed for the seasonal index).
    pub n: usize,
}

impl FittedHoltWinters {
    /// `h`-step additive / multiplicative Holt–Winters forecast.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        if h == 0 {
            return ctx.finish(Vector::zeros(0));
        }
        if h > self.n.max(8) * 4 {
            ctx.push(
                Issue::builder(IssueCode::ForecastHorizonExceedsIdentifiability)
                    .message(format!("horizon {h} is far beyond n={}", self.n))
                    .build(),
            );
        }
        let mut out = Vector::zeros(h);
        for k in 1..=h {
            let sidx = (self.n + k - 1) % self.period.max(1);
            let seas = if self.period == 0 {
                0.0
            } else {
                self.seasonal[sidx]
            };
            out[k - 1] = if self.multiplicative {
                (self.level + k as f64 * self.trend) * seas
            } else {
                self.level + k as f64 * self.trend + seas
            };
        }
        ctx.finish(out)
    }
}

impl FitSeries for HoltWinters {
    type Fitted = FittedHoltWinters;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedHoltWinters>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let period = self.period.max(1);
        if y.len() < 2 * period {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSeasonalCycles)
                    .severity(Severity::Error)
                    .message(format!("Holt–Winters n={} < 2·period={period}", y.len()))
                    .meaninglessness(Meaninglessness::vacuous(
                        "seasonal exponential smoothing",
                        "fewer than two complete cycles cannot identify a seasonal pattern",
                        "shorten the period or collect more seasons",
                    ))
                    .build(),
            );
        }
        if self.multiplicative {
            reject_nonpositive(&mut ctx, y, "multiplicative Holt–Winters");
        }
        warn_unit_root(&mut ctx, y);
        let (alpha, beta, gamma, fitted, level, trend, seas) = hw_fit(
            y.as_slice(),
            period,
            self.alpha,
            self.beta,
            self.gamma,
            self.multiplicative,
        );
        if ![alpha, beta, gamma].iter().all(|v| v.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("Holt–Winters smoothing constants are non-finite")
                    .build(),
            );
        }
        let resid = Vector::from_iter(
            y.as_slice()
                .iter()
                .zip(fitted.as_slice())
                .map(|(a, b)| a - b),
        );
        ctx.finish(FittedHoltWinters {
            alpha,
            beta,
            gamma,
            level,
            trend,
            seasonal: seas,
            fitted,
            resid,
            period,
            multiplicative: self.multiplicative,
            n: y.len(),
        })
    }
}

/// Simple exponential smoothing (sktime / statsmodels `SimpleExpSmoothing`).
#[derive(Clone, Debug, Default)]
pub struct SimpleExpSmoothing {
    /// Level smoothing; `None` selects by in-sample SSE.
    pub alpha: Option<f64>,
}

impl SimpleExpSmoothing {
    /// SES with optional `alpha`.
    pub fn new(alpha: Option<f64>) -> Self {
        Self { alpha }
    }
}

/// Fitted SES state.
#[derive(Clone, Debug)]
pub struct FittedSimpleExpSmoothing {
    /// Level smoothing used.
    pub alpha: f64,
    /// Terminal level.
    pub level: f64,
    /// In-sample fitted values.
    pub fitted: Vector,
    /// In-sample residuals.
    pub resid: Vector,
    /// Training length.
    pub n: usize,
}

impl FittedSimpleExpSmoothing {
    /// Flat forecast at the terminal level.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::from_iter((0..h).map(|_| self.level)))
    }
}

impl FitSeries for SimpleExpSmoothing {
    type Fitted = FittedSimpleExpSmoothing;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSimpleExpSmoothing>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("SES needs n≥2")
                    .build(),
            );
        }
        let candidates = if let Some(a) = self.alpha {
            if a.is_finite() && (0.0..=1.0).contains(&a) {
                vec![a]
            } else {
                ctx.push(
                    Issue::builder(IssueCode::InvalidWeight)
                        .severity(Severity::Warning)
                        .message(format!("SES α={a} is not in [0,1]; grid-searching"))
                        .build(),
                );
                vec![0.1, 0.3, 0.5, 0.8]
            }
        } else {
            vec![0.1, 0.3, 0.5, 0.8]
        };
        let mut best_a = 0.3;
        let mut best_sse = f64::INFINITY;
        let mut best_fit = Vector::zeros(y.len());
        let mut best_level = y.as_slice().first().copied().unwrap_or(0.0);
        for &a in &candidates {
            let (fitted, level, sse) = ses_run(y.as_slice(), a);
            if sse < best_sse {
                best_sse = sse;
                best_a = a;
                best_fit = fitted;
                best_level = level;
            }
        }
        let resid = Vector::from_iter(
            y.as_slice()
                .iter()
                .zip(best_fit.as_slice())
                .map(|(a, b)| a - b),
        );
        ctx.finish(FittedSimpleExpSmoothing {
            alpha: best_a,
            level: best_level,
            fitted: best_fit,
            resid,
            n: y.len(),
        })
    }
}

fn ses_run(y: &[f64], alpha: f64) -> (Vector, f64, f64) {
    let mut level = y.first().copied().unwrap_or(0.0);
    let mut fitted = Vector::zeros(y.len());
    let mut sse = 0.0;
    for (t, &yt) in y.iter().enumerate() {
        fitted[t] = level;
        if yt.is_finite() {
            let e = yt - level;
            sse += e * e;
            level = alpha * yt + (1.0 - alpha) * level;
        }
    }
    (fitted, level, sse)
}

/// Holt linear trend (level + slope, no season).
#[derive(Clone, Debug, Default)]
pub struct Holt {
    /// Level smoothing; `None` selects by SSE.
    pub alpha: Option<f64>,
    /// Trend smoothing.
    pub beta: Option<f64>,
}

impl Holt {
    /// Holt with optional smoothing constants.
    pub fn new(alpha: Option<f64>, beta: Option<f64>) -> Self {
        Self { alpha, beta }
    }
}

/// Fitted Holt state.
#[derive(Clone, Debug)]
pub struct FittedHolt {
    /// Level smoothing used.
    pub alpha: f64,
    /// Trend smoothing used.
    pub beta: f64,
    /// Terminal level.
    pub level: f64,
    /// Terminal trend.
    pub trend: f64,
    /// In-sample fitted values.
    pub fitted: Vector,
    /// In-sample residuals.
    pub resid: Vector,
    /// Training length.
    pub n: usize,
}

impl FittedHolt {
    /// Linear-trend forecast.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::from_iter(
            (1..=h).map(|k| self.level + k as f64 * self.trend),
        ))
    }
}

impl FitSeries for Holt {
    type Fitted = FittedHolt;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedHolt>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 3 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("Holt needs n≥3")
                    .build(),
            );
        }
        let alphas = match self.alpha {
            Some(a) if a.is_finite() && (0.0..=1.0).contains(&a) => vec![a],
            Some(a) => {
                ctx.push(
                    Issue::builder(IssueCode::InvalidWeight)
                        .severity(Severity::Warning)
                        .message(format!("Holt α={a} is not in [0,1]; grid-searching"))
                        .build(),
                );
                vec![0.2, 0.5, 0.8]
            }
            None => vec![0.2, 0.5, 0.8],
        };
        let betas = match self.beta {
            Some(b) if b.is_finite() && (0.0..=1.0).contains(&b) => vec![b],
            Some(b) => {
                ctx.push(
                    Issue::builder(IssueCode::InvalidWeight)
                        .severity(Severity::Warning)
                        .message(format!("Holt β={b} is not in [0,1]; grid-searching"))
                        .build(),
                );
                vec![0.1, 0.3]
            }
            None => vec![0.1, 0.3],
        };
        let mut best = (0.5, 0.1, f64::INFINITY, 0.0, 0.0, Vector::zeros(y.len()));
        for &a in &alphas {
            for &b in &betas {
                let (fitted, level, trend, sse) = holt_run(y.as_slice(), a, b);
                if sse < best.2 {
                    best = (a, b, sse, level, trend, fitted);
                }
            }
        }
        let resid = Vector::from_iter(
            y.as_slice()
                .iter()
                .zip(best.5.as_slice())
                .map(|(u, v)| u - v),
        );
        ctx.finish(FittedHolt {
            alpha: best.0,
            beta: best.1,
            level: best.3,
            trend: best.4,
            fitted: best.5,
            resid,
            n: y.len(),
        })
    }
}

fn holt_run(y: &[f64], alpha: f64, beta: f64) -> (Vector, f64, f64, f64) {
    let y0 = y.first().copied().unwrap_or(0.0);
    let y1 = y.get(1).copied().unwrap_or(y0);
    let mut level = y0;
    let mut trend = y1 - y0;
    let mut fitted = Vector::zeros(y.len());
    let mut sse = 0.0;
    for (t, &yt) in y.iter().enumerate() {
        let pred = level + trend;
        fitted[t] = pred;
        if yt.is_finite() {
            let e = yt - pred;
            sse += e * e;
            let prev = level;
            level = alpha * yt + (1.0 - alpha) * pred;
            trend = beta * (level - prev) + (1.0 - beta) * trend;
        }
    }
    (fitted, level, trend, sse)
}

/// Local-level Kalman wrapper (statsmodels `UnobservedComponents` local level).
#[derive(Clone, Debug, Default)]
pub struct LocalLevel;

impl LocalLevel {
    /// Default local level.
    pub fn new() -> Self {
        Self
    }
}

impl FitSeries for LocalLevel {
    type Fitted = KalmanLevelFit;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<KalmanLevelFit>> {
        kalman_level(y, session)
    }
}

/// ARIMA(p, d, q) identified by Hannan–Rissanen (OLS on lagged y and residual MA).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arima {
    /// Autoregressive order.
    pub p: usize,
    /// Regular differences.
    pub d: usize,
    /// Moving-average order.
    pub q: usize,
}

impl Default for Arima {
    fn default() -> Self {
        Self { p: 1, d: 0, q: 0 }
    }
}

impl Arima {
    /// Construct `ARIMA(p,d,q)`.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self { p, d, q }
    }
}

/// Fitted Hannan–Rissanen ARIMA.
#[derive(Clone, Debug)]
pub struct FittedArima {
    /// Specification.
    pub spec: Arima,
    /// AR coefficients `φ_1 … φ_p`.
    pub ar: Vector,
    /// MA coefficients `θ_1 … θ_q`.
    pub ma: Vector,
    /// Intercept on the differenced scale.
    pub intercept: f64,
    /// Innovation variance.
    pub sigma2: f64,
    /// Differenced-scale residuals.
    pub resid: Vector,
    /// Last `p` differenced observations (for AR recursion).
    pub last_diff: Vector,
    /// Last `q` residuals (for MA recursion).
    pub last_resid: Vector,
    /// Pre-difference stages (each is the series before that difference).
    pub levels: Vec<Vector>,
}

impl FittedArima {
    /// `h`-step forecast on the original scale (undifferenced).
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        if h == 0 {
            return ctx.finish(Vector::zeros(0));
        }
        if h > 4 * self.last_diff.len().max(8) {
            ctx.push(
                Issue::builder(IssueCode::ForecastHorizonExceedsIdentifiability)
                    .message(format!(
                        "ARIMA horizon {h} exceeds a short identified window"
                    ))
                    .build(),
            );
        }
        let fc = arima_forecast(self, h);
        ctx.finish(Vector::from_iter(fc))
    }
}

impl FitSeries for Arima {
    type Fitted = FittedArima;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedArima>> {
        fit_arima(self, y, session)
    }
}

/// Heterogeneous Autoregressive realized-variance model (Corsi HAR-RV).
///
/// Daily / weekly / monthly window lengths are not identification `p`.
#[derive(Clone, Debug)]
pub struct Har {
    /// Daily lookback (typically 1).
    pub daily: usize,
    /// Weekly lookback (typically 5).
    pub weekly: usize,
    /// Monthly lookback (typically 22).
    pub monthly: usize,
}

impl Default for Har {
    fn default() -> Self {
        Self {
            daily: 1,
            weekly: 5,
            monthly: 22,
        }
    }
}

impl Har {
    /// Corsi (1, 5, 22) windows.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted HAR-RV coefficients and the trailing window used to recurse.
#[derive(Clone, Debug)]
pub struct FittedHar {
    /// Intercept.
    pub intercept: f64,
    /// Daily lag coefficient.
    pub beta_d: f64,
    /// Weekly average coefficient.
    pub beta_w: f64,
    /// Monthly average coefficient.
    pub beta_m: f64,
    /// Trailing observations (length `monthly`) for multi-step forecast.
    pub history: Vector,
    /// Daily window stored from the spec.
    pub daily: usize,
    /// Weekly window stored from the spec.
    pub weekly: usize,
    /// Monthly window stored from the spec.
    pub monthly: usize,
}

impl FittedHar {
    /// Recurse the HAR equation `h` steps, feeding forecasts back into the windows.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let mut hist: Vec<f64> = self.history.as_slice().to_vec();
        let mut out = Vector::zeros(h);
        let w = self.weekly.max(1);
        let m = self.monthly.max(1);
        for t in 0..h {
            let n = hist.len();
            let daily = hist.last().copied().unwrap_or(0.0);
            let week = if n == 0 {
                daily
            } else {
                hist[n.saturating_sub(w)..].iter().sum::<f64>() / n.min(w) as f64
            };
            let month = if n == 0 {
                daily
            } else {
                hist[n.saturating_sub(m)..].iter().sum::<f64>() / n.min(m) as f64
            };
            let yhat = self.intercept + self.beta_d * daily + self.beta_w * week + self.beta_m * month;
            out[t] = yhat;
            hist.push(yhat);
        }
        ctx.finish(out)
    }
}

impl FitSeries for Har {
    type Fitted = FittedHar;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedHar>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let daily = self.daily.max(1);
        let weekly = self.weekly.max(daily);
        let monthly = self.monthly.max(weekly);
        let n = y.len();
        let start = monthly;
        if n <= start {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!(
                        "HAR needs n>monthly={monthly}; got n={n}. Coefficients collapse to the last level."
                    ))
                    .metric("n", n as f64)
                    .build(),
            );
            return ctx.finish(FittedHar {
                intercept: y.as_slice().last().copied().unwrap_or(0.0),
                beta_d: 0.0,
                beta_w: 0.0,
                beta_m: 0.0,
                history: y.clone(),
                daily,
                weekly,
                monthly,
            });
        }
        let n_eff = n - start;
        // Window counts are not identification p; do not call inspect_identification.
        let design = Matrix::from_fn(n_eff, 3, |i, j| {
            let t = i + start;
            match j {
                0 => {
                    let d0 = t.saturating_sub(daily);
                    y.as_slice()[d0..t].iter().sum::<f64>() / (t - d0) as f64
                }
                1 => {
                    let d0 = t.saturating_sub(weekly);
                    y.as_slice()[d0..t].iter().sum::<f64>() / (t - d0) as f64
                }
                _ => {
                    let d0 = t.saturating_sub(monthly);
                    y.as_slice()[d0..t].iter().sum::<f64>() / (t - d0) as f64
                }
            }
        });
        let yy = Vector::from_iter((start..n).map(|t| y[t]));
        let xaug = design.with_intercept();
        let beta = statistical_ols(&mut ctx, &xaug, &yy).unwrap_or_else(|| Vector::zeros(4));
        let keep = monthly.min(n);
        let history = Vector::from_iter(y.as_slice()[n - keep..].iter().copied());
        ctx.finish(FittedHar {
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            beta_d: if beta.len() > 1 { beta[1] } else { 0.0 },
            beta_w: if beta.len() > 2 { beta[2] } else { 0.0 },
            beta_m: if beta.len() > 3 { beta[3] } else { 0.0 },
            history,
            daily,
            weekly,
            monthly,
        })
    }
}

fn arma_companion_step(
    z: &[f64],
    mu: f64,
    phi: &[f64],
    theta: &[f64],
) -> (f64, f64, Vec<f64>, Matrix, Vector) {
    let p = phi.len();
    let q = theta.len();
    let m = p.max(q + 1).max(1);
    let mut tmat = Matrix::zeros(m, m);
    for j in 0..p.min(m) {
        tmat.set(0, j, phi[j]);
    }
    for i in 1..m {
        tmat.set(i, i - 1, 1.0);
    }
    let mut r = Vector::zeros(m);
    r[0] = 1.0;
    for j in 0..q.min(m.saturating_sub(1)) {
        r[j + 1] = theta[j];
    }
    let mut a = vec![0.0; m];
    let mut pmat = Matrix::from_fn(m, m, |i, j| if i == j { 1.0e4 } else { 0.0 });
    let mut ss = 0.0;
    let mut logf = 0.0;
    let mut n_used = 0.0;
    let mut last_v = Vector::zeros(z.len());
    for (t, &yt) in z.iter().enumerate() {
        if !yt.is_finite() {
            continue;
        }
        let f = pmat.get(0, 0).max(1e-12);
        let v = yt - mu - a[0];
        last_v[t] = v;
        ss += v * v / f;
        logf += f.ln();
        n_used += 1.0;
        let mut k = Vector::zeros(m);
        for i in 0..m {
            k[i] = pmat.get(i, 0) / f;
        }
        for i in 0..m {
            a[i] += k[i] * v;
        }
        let mut pnew = Matrix::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                pnew.set(i, j, pmat.get(i, j) - k[i] * f * k[j]);
            }
        }
        let mut ap = vec![0.0; m];
        for i in 0..m {
            for j in 0..m {
                ap[i] += tmat.get(i, j) * a[j];
            }
        }
        let mut pp = Matrix::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for u in 0..m {
                    for w in 0..m {
                        s += tmat.get(i, u) * pnew.get(u, w) * tmat.get(j, w);
                    }
                }
                s += r[i] * r[j];
                pp.set(i, j, s);
            }
        }
        a = ap;
        pmat = pp;
    }
    let sigma2 = if n_used > 0.0 {
        (ss / n_used).max(1e-12)
    } else {
        1.0
    };
    let ll = if n_used > 0.0 {
        -0.5 * n_used * sigma2.ln() - 0.5 * logf
    } else {
        f64::NEG_INFINITY
    };
    (ll, sigma2, a, pmat, last_v)
}

/// ARIMA estimated by a Kalman-filter Gaussian likelihood (statsmodels statespace).
///
/// Hannan–Rissanen [`Arima`] remains the CSS/OLS path. This refines \(\phi,\theta\)
/// on the concentrated Kalman likelihood. A diffuse \(P_0\) is recorded.
#[derive(Clone, Debug)]
pub struct ArimaKalman {
    /// Autoregressive order.
    pub p: usize,
    /// Regular differences.
    pub d: usize,
    /// Moving-average order.
    pub q: usize,
}

impl Default for ArimaKalman {
    fn default() -> Self {
        Self { p: 1, d: 0, q: 0 }
    }
}

impl ArimaKalman {
    /// `ARIMA(p,d,q)` Kalman MLE.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self { p, d, q }
    }
}

/// Fitted Kalman ARIMA.
#[derive(Clone, Debug)]
pub struct FittedArimaKalman {
    /// Specification.
    pub spec: Arima,
    /// AR coefficients.
    pub ar: Vector,
    /// MA coefficients.
    pub ma: Vector,
    /// Intercept on the differenced scale.
    pub intercept: f64,
    /// Concentrated innovation variance.
    pub sigma2: f64,
    /// Kalman Gaussian log-likelihood (concentrated).
    pub loglik: f64,
    /// Last filtered state.
    pub last_state: Vector,
    /// Differenced training series.
    pub last_diff: Vector,
    /// Pre-difference stages.
    pub levels: Vec<Vector>,
}

impl FittedArimaKalman {
    /// `h`-step forecast on the original scale.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        if h == 0 {
            return ctx.finish(Vector::zeros(0));
        }
        let p = self.ar.len();
        let q = self.ma.len();
        let mut hist: Vec<f64> = self.last_diff.as_slice().to_vec();
        let mut zf = Vec::with_capacity(h);
        for _ in 0..h {
            let mut yhat = self.intercept;
            for j in 0..p {
                if let Some(v) = hist.get(hist.len().saturating_sub(1 + j)) {
                    yhat += self.ar[j] * *v;
                }
            }
            zf.push(yhat);
            hist.push(yhat);
            let _ = q;
        }
        let levels: Vec<Vector> = self.levels.clone();
        ctx.finish(Vector::from_iter(undiff_forecast(&levels, &zf)))
    }
}

impl FitSeries for ArimaKalman {
    type Fitted = FittedArimaKalman;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedArimaKalman>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let p = self.p.min(2);
        let q = self.q.min(2);
        let d = self.d.min(2);
        if self.p > 2 || self.q > 2 {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .severity(Severity::Warning)
                    .message(format!(
                        "ArimaKalman caps the companion at p,q≤2 (requested p={} q={})",
                        self.p, self.q
                    ))
                    .build(),
            );
        }
        let (z, stages) = difference_with_history(y.as_slice(), d);
        if z.len() < 6 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!("ArimaKalman differenced length {} < 6", z.len()))
                    .build(),
            );
        }
        let mu = z.iter().copied().filter(|v| v.is_finite()).sum::<f64>()
            / z.iter().filter(|v| v.is_finite()).count().max(1) as f64;
        let mut best_ll = f64::NEG_INFINITY;
        let mut best_phi = vec![0.0; p];
        let mut best_th = vec![0.0; q];
        let mut best_s2 = 1.0;
        let mut best_a = Vector::zeros(p.max(q + 1).max(1));
        let grid = [-0.6, -0.3, 0.0, 0.3, 0.6];
        let phi_grid: Vec<Vec<f64>> = if p == 0 {
            vec![Vec::new()]
        } else if p == 1 {
            grid.iter().map(|&v| vec![v]).collect()
        } else {
            let mut out = Vec::new();
            for &a in &grid {
                for &b in &[-0.3, 0.0, 0.3] {
                    out.push(vec![a, b]);
                }
            }
            out
        };
        let th_grid: Vec<Vec<f64>> = if q == 0 {
            vec![Vec::new()]
        } else if q == 1 {
            grid.iter().map(|&v| vec![v]).collect()
        } else {
            let mut out = Vec::new();
            for &a in &grid {
                for &b in &[-0.3, 0.0, 0.3] {
                    out.push(vec![a, b]);
                }
            }
            out
        };
        for phi in &phi_grid {
            for th in &th_grid {
                let (ll, s2, a, _, _) = arma_companion_step(&z, mu, phi, th);
                if ll > best_ll {
                    best_ll = ll;
                    best_phi = phi.clone();
                    best_th = th.clone();
                    best_s2 = s2;
                    best_a = Vector::from_slice(&a);
                }
            }
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(Severity::Advisory)
                .message("ArimaKalman uses a diffuse P0 and a coarse φ/θ grid, not exact MLE")
                .compromise(NumericalCompromise::new(
                    "exact diffuse Kalman ARIMA MLE",
                    "concentrated companion Kalman on a φ/θ grid",
                    "P0 is 1e4 I; the likelihood is not the Ansley/Kohn exact form",
                    "treat loglik as a relative score on this grid only",
                ))
                .build(),
        );
        if !best_ll.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("ArimaKalman grid produced no finite likelihood")
                    .build(),
            );
        }
        let levels = stages.into_iter().map(|s| Vector::from_slice(&s)).collect();
        ctx.finish(FittedArimaKalman {
            spec: Arima { p, d, q },
            ar: Vector::from_slice(&best_phi),
            ma: Vector::from_slice(&best_th),
            intercept: mu,
            sigma2: best_s2,
            loglik: best_ll,
            last_state: best_a,
            last_diff: Vector::from_slice(&z),
            levels,
        })
    }
}

/// Small \((p,d,q)\) AIC grid over Hannan–Rissanen [`Arima`].
#[derive(Clone, Debug)]
pub struct AutoArima {
    /// Max AR order.
    pub max_p: usize,
    /// Max regular differences.
    pub max_d: usize,
    /// Max MA order.
    pub max_q: usize,
}

impl Default for AutoArima {
    fn default() -> Self {
        Self {
            max_p: 2,
            max_d: 1,
            max_q: 2,
        }
    }
}

impl AutoArima {
    /// Default auto-ARIMA grid.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Selected ARIMA and the AIC grid that justified it.
#[derive(Clone, Debug)]
pub struct FittedAutoArima {
    /// Winning specification.
    pub spec: Arima,
    /// AIC of the winner.
    pub aic: f64,
    /// `(p,d,q,aic)` for every successful grid point.
    pub scores: Vec<(usize, usize, usize, f64)>,
    /// Refit of the winner.
    pub fitted: FittedArima,
}

impl FitSeries for AutoArima {
    type Fitted = FittedAutoArima;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedAutoArima>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let mut scores = Vec::new();
        let mut best: Option<(f64, Arima, FittedArima)> = None;
        for d in 0..=self.max_d {
            for p in 0..=self.max_p {
                for q in 0..=self.max_q {
                    let mut spec = Arima { p, d, q };
                    match spec.fit_series(y, &session.child(format!("arima_{p}{d}{q}"))) {
                        Ok(fit) => {
                            let n = fit.value.resid.len().max(1) as f64;
                            let k = (1 + p + q) as f64;
                            let s2 = fit.value.sigma2.max(1e-18);
                            let aic = n * s2.ln() + 2.0 * k;
                            scores.push((p, d, q, aic));
                            match &best {
                                Some((b, _, _)) if aic >= *b => {}
                                _ => best = Some((aic, spec, fit.value)),
                            }
                        }
                        Err(e) => {
                            ctx.push(
                                Issue::builder(IssueCode::DidNotConverge)
                                    .severity(Severity::Advisory)
                                    .message(format!(
                                        "ARIMA({p},{d},{q}) rejected: {}",
                                        e.primary().code
                                    ))
                                    .build(),
                            );
                        }
                    }
                }
            }
        }
        ctx.push(
            Issue::builder(IssueCode::Overparameterized)
                .severity(Severity::Advisory)
                .message("auto-ARIMA AIC is computed on Hannan–Rissanen σ², not the exact Gaussian likelihood")
                .compromise(NumericalCompromise::new(
                    "exact-likelihood auto-ARIMA",
                    "Hannan–Rissanen OLS grid + n ln σ² + 2k",
                    "failed orders are skipped",
                    "the selected (p,d,q) is a relative AIC winner on this grid only",
                ))
                .build(),
        );
        match best {
            Some((aic, spec, fitted)) => ctx.finish(FittedAutoArima {
                spec,
                aic,
                scores,
                fitted,
            }),
            None => {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("every auto-ARIMA grid point failed")
                        .meaninglessness(Meaninglessness::vacuous(
                            "auto-ARIMA specification",
                            "no (p,d,q) in the grid produced a fit",
                            "lengthen the series or shrink the grid",
                        ))
                        .build(),
                );
                ctx.finish(FittedAutoArima {
                    spec: Arima::default(),
                    aic: f64::NAN,
                    scores,
                    fitted: FittedArima {
                        spec: Arima::default(),
                        ar: Vector::zeros(1),
                        ma: Vector::zeros(0),
                        intercept: y.mean(),
                        sigma2: f64::NAN,
                        resid: Vector::zeros(0),
                        last_diff: Vector::zeros(0),
                        last_resid: Vector::zeros(0),
                        levels: Vec::new(),
                    },
                })
            }
        }
    }
}

/// Hold-out grid over SES and small ARIMA orders (sktime `ForecastingGridSearchCV`).
///
/// Candidate count is not identification `p`. Inner residual-kind failures are
/// not promoted.
#[derive(Clone, Debug)]
pub struct ForecastingGridSearchCV {
    /// Hold-out horizon used to score candidates.
    pub fh: usize,
}

impl Default for ForecastingGridSearchCV {
    fn default() -> Self {
        Self { fh: 4 }
    }
}

impl ForecastingGridSearchCV {
    /// Grid search with hold-out horizon `fh`.
    pub fn new(fh: usize) -> Self {
        Self { fh: fh.max(1) }
    }
}

/// Selected SES or ARIMA member and the hold-out scores.
#[derive(Clone, Debug)]
pub struct FittedForecastingGridSearch {
    /// Winning specification label.
    pub best_name: String,
    /// Hold-out MAE of the winner.
    pub best_mae: f64,
    /// `(name, mae)` for every successful candidate.
    pub scores: Vec<(String, f64)>,
    /// Fitted SES when that family won.
    pub ses: Option<FittedSimpleExpSmoothing>,
    /// Fitted ARIMA when that family won.
    pub arima: Option<FittedArima>,
}

impl FittedForecastingGridSearch {
    /// `h`-step forecast from the winning member (refit on the full series).
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        if let Some(s) = &self.ses {
            return s.forecast(h, session);
        }
        if let Some(a) = &self.arima {
            return a.forecast(h, session);
        }
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::zeros(h))
    }
}

fn holdout_mae(actual: &[f64], pred: &[f64]) -> f64 {
    let n = actual.len().min(pred.len());
    if n == 0 {
        return f64::INFINITY;
    }
    let mut s: f64 = 0.0;
    for i in 0..n {
        s += (actual[i] - pred[i]).abs();
    }
    s / n as f64
}

impl FitSeries for ForecastingGridSearchCV {
    type Fitted = FittedForecastingGridSearch;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForecastingGridSearch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let h = self.fh.max(1).min(y.len().saturating_sub(4).max(1));
        if y.len() <= h + 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!(
                        "ForecastingGridSearchCV n={} is short for fh={h}",
                        y.len()
                    ))
                    .build(),
            );
        }
        let split = y.len().saturating_sub(h);
        let train = Vector::from_iter(y.as_slice().iter().take(split).copied());
        let hold = &y.as_slice()[split.min(y.len())..];
        let mut scores = Vec::new();
        let mut best_name = "ses_0.3".to_string();
        let mut best_mae = f64::INFINITY;
        for &a in &[0.1, 0.3, 0.5, 0.8] {
            match SimpleExpSmoothing::new(Some(a)).fit_series(&train, &session.child("ses")) {
                Ok(q) => match q.value.forecast(h, &session.child("ses_fc")) {
                    Ok(fc) => {
                        let mae = holdout_mae(hold, fc.value.as_slice());
                        let name = format!("ses_{a}");
                        scores.push((name.clone(), mae));
                        if mae < best_mae {
                            best_mae = mae;
                            best_name = name;
                        }
                    }
                    Err(_) => {}
                },
                Err(e) => {
                    if !matches!(
                        e.primary.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::RankZero
                            | IssueCode::R2IsOne
                    ) {
                        ctx.push(e.primary);
                    }
                }
            }
        }
        for (p, d, q) in [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 1)] {
            match Arima::new(p, d, q).fit_series(&train, &session.child("arima")) {
                Ok(fit) => match fit.value.forecast(h, &session.child("arima_fc")) {
                    Ok(fc) => {
                        let mae = holdout_mae(hold, fc.value.as_slice());
                        let name = format!("arima_{p}{d}{q}");
                        scores.push((name.clone(), mae));
                        if mae < best_mae {
                            best_mae = mae;
                            best_name = name;
                        }
                    }
                    Err(_) => {}
                },
                Err(e) => {
                    if !matches!(
                        e.primary.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::RankZero
                            | IssueCode::R2IsOne
                    ) {
                        ctx.push(e.primary);
                    }
                }
            }
        }
        let mut ses = None;
        let mut arima = None;
        if best_name.starts_with("ses_") {
            if let Ok(q) = SimpleExpSmoothing::new(None).fit_series(y, &session.child("refit_ses"))
            {
                ses = Some(q.value);
            }
        } else if best_name.starts_with("arima_") {
            let mut spec = if best_name.contains("101") {
                Arima::new(1, 0, 1)
            } else if best_name.contains("110") {
                Arima::new(1, 1, 0)
            } else if best_name.contains("010") {
                Arima::new(0, 1, 0)
            } else {
                Arima::new(1, 0, 0)
            };
            if let Ok(q) = spec.fit_series(y, &session.child("refit_arima")) {
                arima = Some(q.value);
            }
        }
        if ses.is_none() && arima.is_none() {
            if let Ok(q) = SimpleExpSmoothing::new(Some(0.3)).fit_series(y, &session.child("fb")) {
                ses = Some(q.value);
                if best_mae.is_infinite() {
                    best_name = "ses_0.3".into();
                }
            }
        }
        ctx.finish(FittedForecastingGridSearch {
            best_name,
            best_mae,
            scores,
            ses,
            arima,
        })
    }
}

/// Average of Holt–Winters and ARIMA(1,0,1) forecasts (sktime `EnsembleForecaster`).
#[derive(Clone, Debug)]
pub struct EnsembleForecaster {
    /// Holt–Winters seasonal period.
    pub period: usize,
}

impl Default for EnsembleForecaster {
    fn default() -> Self {
        Self { period: 4 }
    }
}

impl EnsembleForecaster {
    /// Ensemble with Holt–Winters period `period`.
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(2),
        }
    }
}

/// Fitted two-model ensemble.
#[derive(Clone, Debug)]
pub struct FittedEnsembleForecaster {
    /// Holt–Winters member (if it identified).
    pub hw: Option<FittedHoltWinters>,
    /// ARIMA member (if it identified).
    pub arima: Option<FittedArima>,
}

impl FittedEnsembleForecaster {
    /// Average the available member forecasts.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        if h == 0 {
            return ctx.finish(Vector::zeros(0));
        }
        let mut acc = Vector::zeros(h);
        let mut k = 0.0;
        if let Some(hw) = &self.hw {
            match hw.forecast(h, &session.child("hw")) {
                Ok(q) => {
                    for i in 0..h.min(q.value.len()) {
                        acc[i] += q.value[i];
                    }
                    k += 1.0;
                }
                Err(e) => ctx.push(e.primary),
            }
        }
        if let Some(ar) = &self.arima {
            match ar.forecast(h, &session.child("arima")) {
                Ok(q) => {
                    for i in 0..h.min(q.value.len()) {
                        acc[i] += q.value[i];
                    }
                    k += 1.0;
                }
                Err(e) => ctx.push(e.primary),
            }
        }
        if k <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("ensemble has no successful member forecast")
                    .meaninglessness(Meaninglessness::vacuous(
                        "ensemble forecast",
                        "both members failed",
                        "fit a single identified forecaster",
                    ))
                    .build(),
            );
            return ctx.finish(Vector::zeros(h));
        }
        for i in 0..h {
            acc[i] /= k;
        }
        ctx.finish(acc)
    }
}

impl FitSeries for EnsembleForecaster {
    type Fitted = FittedEnsembleForecaster;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedEnsembleForecaster>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let hw = match (HoltWinters {
            period: self.period,
            alpha: Some(0.3),
            beta: Some(0.1),
            gamma: Some(0.1),
            multiplicative: false,
        })
        .fit_series(y, &session.child("hw"))
        {
            Ok(q) => Some(q.value),
            Err(e) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .severity(Severity::Warning)
                        .message(format!("Holt–Winters member failed: {}", e.primary().code))
                        .build(),
                );
                None
            }
        };
        let arima = match Arima::new(1, 0, 1).fit_series(y, &session.child("arima")) {
            Ok(q) => Some(q.value),
            Err(e) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .severity(Severity::Warning)
                        .message(format!("ARIMA member failed: {}", e.primary().code))
                        .build(),
                );
                None
            }
        };
        if hw.is_none() && arima.is_none() {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("both ensemble members failed")
                    .meaninglessness(Meaninglessness::vacuous(
                        "ensemble",
                        "no member identified",
                        "lengthen the series",
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedEnsembleForecaster { hw, arima })
    }
}

/// SARIMAX: OLS on exog, then [`Arima`] on the residual (statsmodels `SARIMAX` lite).
///
/// The ARIMA step runs on a scratch report so a short residual series does not
/// hide a valid exog slope behind [`IssueCode::ShortSeriesForArima`].
#[derive(Clone, Debug)]
pub struct Sarimax {
    /// Non-seasonal \((p,d,q)\).
    pub order: (usize, usize, usize),
}

impl Default for Sarimax {
    fn default() -> Self {
        Self { order: (1, 0, 0) }
    }
}

impl Sarimax {
    /// `SARIMAX(p,d,q)` without seasonal terms.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self { order: (p, d, q) }
    }

    /// Fit `y` on exog `x`.
    pub fn fit(
        &mut self,
        y: &Vector,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedSarimax>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.nrows() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("SARIMAX exog rows ≠ n")
                    .build(),
            );
        }
        let n = y.len().min(x.nrows());
        let design = x.with_intercept();
        let Some(beta) = statistical_ols(&mut ctx, &design, y) else {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("SARIMAX exog OLS failed")
                    .build(),
            );
            return ctx.finish(FittedSarimax {
                coef: Vector::zeros(x.ncols()),
                intercept: 0.0,
                inner: empty_arima(&Arima {
                    p: self.order.0,
                    d: self.order.1,
                    q: self.order.2,
                }),
            });
        };
        let fitted = design.matvec(&beta);
        let resid = y.sub(&fitted);
        let mut arima = Arima {
            p: self.order.0,
            d: self.order.1,
            q: self.order.2,
        };
        let inner = match arima.fit_series(&resid, &session.child("resid-arima")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::ShortSeriesForArima
                            | IssueCode::R2IsOne
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                q.value
            }
            Err(e) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .severity(Severity::Warning)
                        .message(format!(
                            "SARIMAX residual ARIMA failed: {}",
                            e.primary().code
                        ))
                        .build(),
                );
                empty_arima(&arima)
            }
        };
        let (intercept, coef) = (
            beta.as_slice().first().copied().unwrap_or(0.0),
            Vector::from_iter((1..beta.len()).map(|j| beta[j])),
        );
        let _ = n;
        ctx.finish(FittedSarimax {
            coef,
            intercept,
            inner,
        })
    }
}

/// Fitted SARIMAX.
#[derive(Clone, Debug)]
pub struct FittedSarimax {
    /// Exog slopes.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
    /// ARIMA on OLS residuals.
    pub inner: FittedArima,
}

impl FittedSarimax {
    /// Forecast `h` steps using future exog `x_future` (`h × p`).
    pub fn forecast(
        &self,
        h: usize,
        x_future: &Matrix,
        session: &Session,
    ) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        if h == 0 {
            return ctx.finish(Vector::zeros(0));
        }
        if x_future.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("SARIMAX forecast exog columns ≠ coef")
                    .build(),
            );
        }
        let ar = match self.inner.forecast(h, &session.child("arima")) {
            Ok(q) => q.value,
            Err(_) => Vector::zeros(h),
        };
        let y = Vector::from_iter((0..h).map(|t| {
            let mut s = self.intercept;
            if t < x_future.nrows() {
                for j in 0..x_future.ncols().min(self.coef.len()) {
                    s += x_future.get(t, j) * self.coef[j];
                }
            }
            s + if t < ar.len() { ar[t] } else { 0.0 }
        }));
        ctx.finish(y)
    }
}

/// Log-then-ARIMA pipeline (sktime `ForecastingPipeline` lite).
#[derive(Clone, Debug, Default)]
pub struct ForecastingPipeline {
    /// Apply `log` before the inner ARIMA (requires `y > 0`).
    pub log: bool,
}

impl ForecastingPipeline {
    /// Log + ARIMA(1,0,1) pipeline.
    pub fn new() -> Self {
        Self { log: true }
    }
}

/// Fitted forecasting pipeline.
#[derive(Clone, Debug)]
pub struct FittedForecastingPipeline {
    /// Whether the log map was applied.
    pub log: bool,
    /// Inner ARIMA.
    pub inner: FittedArima,
}

impl FittedForecastingPipeline {
    /// Forecast and invert the log if it was used.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let q = self.inner.forecast(h, session)?;
        let ctx = FitCtx::with_session(session.child("pipeline-inv"));
        let y = if self.log {
            Vector::from_iter(q.value.as_slice().iter().map(|v| v.exp()))
        } else {
            q.value
        };
        ctx.finish(y)
    }
}

impl FitSeries for ForecastingPipeline {
    type Fitted = FittedForecastingPipeline;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForecastingPipeline>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let z = if self.log {
            reject_nonpositive(&mut ctx, y, "ForecastingPipeline log");
            Vector::from_iter(y.as_slice().iter().map(|v| v.max(1e-12).ln()))
        } else {
            y.clone()
        };
        let mut arima = Arima::new(1, 0, 1);
        let inner = match arima.fit_series(&z, &session.child("inner")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::RankZero
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                q.value
            }
            Err(e) => {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .severity(Severity::Warning)
                        .message("inner ARIMA(1,0,1) aborted; pipeline keeps a documented empty forecast state")
                        .compromise(NumericalCompromise::new(
                            "log-then-ARIMA(1,0,1) on the transformed series",
                            "empty ARIMA coefficients after an unidentified Hannan–Rissanen design",
                            "the ARIMA design was rank-deficient or otherwise aborted",
                            "do not interpret AR/MA coefficients; forecasts fall back to the last level",
                        ))
                        .build(),
                );
                for issue in e.report.issues() {
                    if issue.severity.is_at_least(Severity::Error) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                empty_arima(&arima)
            }
        };
        ctx.finish(FittedForecastingPipeline {
            log: self.log,
            inner,
        })
    }
}

/// TBATS-lite: optional log, Fourier seasonal terms, linear trend, AR(1) errors.
#[derive(Clone, Debug)]
pub struct Tbats {
    /// Seasonal period.
    pub period: usize,
    /// Fourier harmonics.
    pub harmonics: usize,
    /// Log / Box–Cox (λ=0) map.
    pub use_log: bool,
}

impl Default for Tbats {
    fn default() -> Self {
        Self {
            period: 4,
            harmonics: 2,
            use_log: false,
        }
    }
}

impl Tbats {
    /// TBATS with the given period.
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(2),
            ..Self::default()
        }
    }
}

/// Fitted TBATS-lite.
#[derive(Clone, Debug)]
pub struct FittedTbats {
    /// OLS coefficients on `[1, t, sin, cos, …]`.
    pub coef: Vector,
    /// AR(1) residual coefficient.
    pub phi: f64,
    /// Last residual.
    pub last_resid: f64,
    /// Period.
    pub period: usize,
    /// Harmonics.
    pub harmonics: usize,
    /// Log map.
    pub use_log: bool,
    /// Training length (for the trend clock).
    pub n: usize,
}

impl FittedTbats {
    fn design_row(&self, t: usize, p: usize) -> Vector {
        let mut v = Vector::zeros(p);
        v[0] = 1.0;
        if p > 1 {
            v[1] = t as f64;
        }
        let per = self.period.max(2) as f64;
        let mut k = 2usize;
        for h in 1..=self.harmonics {
            if k < p {
                v[k] = (2.0 * std::f64::consts::PI * h as f64 * t as f64 / per).cos();
                k += 1;
            }
            if k < p {
                v[k] = (2.0 * std::f64::consts::PI * h as f64 * t as f64 / per).sin();
                k += 1;
            }
        }
        v
    }

    /// `h`-step forecast on the original scale.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        let p = self.coef.len();
        let mut e = self.last_resid;
        let y = Vector::from_iter((0..h).map(|s| {
            let t = self.n + s;
            let row = self.design_row(t, p);
            let mut mu = 0.0;
            for j in 0..p {
                mu += self.coef[j] * row[j];
            }
            e *= self.phi;
            let z = mu + e;
            if self.use_log {
                z.exp()
            } else {
                z
            }
        }));
        ctx.finish(y)
    }
}

impl FitSeries for Tbats {
    type Fitted = FittedTbats;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedTbats>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let period = self.period.max(2);
        if y.len() < 2 * period {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSeasonalCycles)
                    .severity(Severity::Warning)
                    .message(format!("TBATS n={} < 2s with s={period}", y.len()))
                    .build(),
            );
        }
        let z = if self.use_log {
            reject_nonpositive(&mut ctx, y, "TBATS log");
            Vector::from_iter(y.as_slice().iter().map(|v| v.max(1e-12).ln()))
        } else {
            y.clone()
        };
        let h = self.harmonics.max(1);
        let p = 2 + 2 * h;
        // Do not inspect_identification(n, p) — harmonics are not free parameters
        // in the same sense as a linear model on small n.
        let n = z.len();
        let x = Matrix::from_fn(n, p, |t, j| {
            let row = FittedTbats {
                coef: Vector::zeros(p),
                phi: 0.0,
                last_resid: 0.0,
                period,
                harmonics: h,
                use_log: self.use_log,
                n,
            }
            .design_row(t, p);
            row[j]
        });
        let Some(coef) = statistical_ols(&mut ctx, &x, &z) else {
            return ctx.finish(FittedTbats {
                coef: Vector::zeros(p),
                phi: 0.0,
                last_resid: 0.0,
                period,
                harmonics: h,
                use_log: self.use_log,
                n,
            });
        };
        let fit = x.matvec(&coef);
        let resid = z.sub(&fit);
        let mut num = 0.0;
        let mut den = 0.0;
        for t in 1..n {
            num += resid[t] * resid[t - 1];
            den += resid[t - 1] * resid[t - 1];
        }
        let phi = if den > 0.0 {
            (num / den).clamp(-0.99, 0.99)
        } else {
            0.0
        };
        ctx.finish(FittedTbats {
            coef,
            phi,
            last_resid: resid.as_slice().last().copied().unwrap_or(0.0),
            period,
            harmonics: h,
            use_log: self.use_log,
            n,
        })
    }
}

fn boxcox_apply(v: f64, lam: f64) -> f64 {
    let x = v.max(1e-12);
    if lam.abs() < 1e-12 {
        x.ln()
    } else {
        (x.powf(lam) - 1.0) / lam
    }
}

fn boxcox_inv(z: f64, lam: f64) -> f64 {
    if lam.abs() < 1e-12 {
        z.exp()
    } else {
        (lam * z + 1.0).max(1e-12).powf(1.0 / lam)
    }
}

fn arma11_from_resid(e: &[f64]) -> (f64, f64) {
    let n = e.len();
    if n < 4 {
        return (0.0, 0.0);
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for t in 1..n {
        num += e[t] * e[t - 1];
        den += e[t - 1] * e[t - 1];
    }
    let phi = if den > 1e-18 {
        (num / den).clamp(-0.99, 0.99)
    } else {
        0.0
    };
    let mut u = vec![0.0; n];
    for t in 1..n {
        u[t] = e[t] - phi * e[t - 1];
    }
    let mut n2 = 0.0;
    let mut d2 = 0.0;
    for t in 2..n {
        n2 += u[t] * u[t - 1];
        d2 += u[t - 1] * u[t - 1];
    }
    let theta = if d2 > 1e-18 {
        (n2 / d2).clamp(-0.99, 0.99)
    } else {
        0.0
    };
    (phi, theta)
}

/// TBATS with a Box–Cox grid, Fourier seasonality, and ARMA(1,1) errors.
///
/// This is closer to De Livera–Hyndman–Snyder than [`Tbats`] (log + AR(1)).
/// Harmonic count is not identification `p`. λ is chosen by profile Gaussian
/// likelihood including the Box–Cox Jacobian.
#[derive(Clone, Debug)]
pub struct TbatsFull {
    /// Seasonal period.
    pub period: usize,
    /// Fourier harmonics.
    pub harmonics: usize,
}

impl Default for TbatsFull {
    fn default() -> Self {
        Self {
            period: 4,
            harmonics: 2,
        }
    }
}

impl TbatsFull {
    /// TBATS with the given period.
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(2),
            ..Self::default()
        }
    }
}

/// Fitted Box–Cox + Fourier + ARMA(1,1) TBATS.
#[derive(Clone, Debug)]
pub struct FittedTbatsFull {
    /// OLS coefficients on `[1, t, sin, cos, …]`.
    pub coef: Vector,
    /// AR(1) residual coefficient.
    pub phi: f64,
    /// MA(1) residual coefficient.
    pub theta: f64,
    /// Last residual (Box–Cox scale).
    pub last_resid: f64,
    /// Last innovation.
    pub last_innov: f64,
    /// Selected Box–Cox λ.
    pub lambda: f64,
    /// Period.
    pub period: usize,
    /// Harmonics.
    pub harmonics: usize,
    /// Training length.
    pub n: usize,
}

impl FittedTbatsFull {
    fn design_row(&self, t: usize, p: usize) -> Vector {
        let mut v = Vector::zeros(p);
        v[0] = 1.0;
        if p > 1 {
            v[1] = t as f64;
        }
        let per = self.period.max(2) as f64;
        let mut k = 2usize;
        for h in 1..=self.harmonics {
            if k < p {
                v[k] = (2.0 * std::f64::consts::PI * h as f64 * t as f64 / per).cos();
                k += 1;
            }
            if k < p {
                v[k] = (2.0 * std::f64::consts::PI * h as f64 * t as f64 / per).sin();
                k += 1;
            }
        }
        v
    }

    /// `h`-step forecast on the original scale.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let p = self.coef.len();
        let mut e = self.last_resid;
        let mut innov = self.last_innov;
        let y = Vector::from_iter((0..h).map(|s| {
            let t = self.n + s;
            let row = self.design_row(t, p);
            let mut mu = 0.0;
            for j in 0..p {
                mu += self.coef[j] * row[j];
            }
            let next_e = self.phi * e + self.theta * innov;
            innov = 0.0;
            e = next_e;
            boxcox_inv(mu + e, self.lambda)
        }));
        ctx.finish(y)
    }
}

impl FitSeries for TbatsFull {
    type Fitted = FittedTbatsFull;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedTbatsFull>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let period = self.period.max(2);
        if y.len() < 2 * period {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSeasonalCycles)
                    .severity(Severity::Warning)
                    .message(format!("TBATS-full n={} < 2s with s={period}", y.len()))
                    .build(),
            );
        }
        if y.as_slice().iter().any(|&v| v <= 0.0) {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .message("TBATS-full Box–Cox requires a strictly positive series")
                    .build(),
            );
        }
        let h = self.harmonics.max(1);
        let p = 2 + 2 * h;
        let n = y.len();
        let mut best_ll = f64::NEG_INFINITY;
        let mut best = FittedTbatsFull {
            coef: Vector::zeros(p),
            phi: 0.0,
            theta: 0.0,
            last_resid: 0.0,
            last_innov: 0.0,
            lambda: 0.0,
            period,
            harmonics: h,
            n,
        };
        for i in 0..=8 {
            let lam = -1.0 + 0.5 * i as f64;
            let z = Vector::from_iter(y.as_slice().iter().map(|&v| boxcox_apply(v, lam)));
            let x = Matrix::from_fn(n, p, |t, j| {
                FittedTbatsFull {
                    coef: Vector::zeros(p),
                    phi: 0.0,
                    theta: 0.0,
                    last_resid: 0.0,
                    last_innov: 0.0,
                    lambda: lam,
                    period,
                    harmonics: h,
                    n,
                }
                .design_row(t, p)[j]
            });
            let Some(coef) = statistical_ols(&mut ctx, &x, &z) else {
                continue;
            };
            let fit = x.matvec(&coef);
            let resid: Vec<f64> = (0..n).map(|t| z[t] - fit[t]).collect();
            let (phi, theta) = arma11_from_resid(&resid);
            let mut sse: f64 = 0.0;
            let mut innov = 0.0;
            for t in 0..n {
                let pred_e = if t == 0 {
                    0.0
                } else {
                    phi * resid[t - 1] + theta * innov
                };
                innov = resid[t] - pred_e;
                sse += innov * innov;
            }
            let s2 = (sse / n.max(1) as f64).max(1e-12);
            let mut jac = 0.0;
            for &v in y.as_slice() {
                jac += (v.max(1e-12)).ln();
            }
            let ll = -0.5 * n as f64 * s2.ln() + (lam - 1.0) * jac;
            if ll > best_ll {
                best_ll = ll;
                best = FittedTbatsFull {
                    coef,
                    phi,
                    theta,
                    last_resid: resid.last().copied().unwrap_or(0.0),
                    last_innov: innov,
                    lambda: lam,
                    period,
                    harmonics: h,
                    n,
                };
            }
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(Severity::Advisory)
                .message(format!(
                    "TBATS-full selected λ={:.3e} by a coarse Box–Cox grid; ARMA is CSS(1,1)",
                    best.lambda
                ))
                .compromise(NumericalCompromise::new(
                    "full TBATS MLE (Box–Cox + trigonometric + ARMA)",
                    "profile-likelihood λ grid plus Hannan–Rissanen ARMA(1,1)",
                    "the seasonal states are Fourier OLS, not the De Livera damped trig form",
                    "do not treat this as the tbats R package MLE",
                ))
                .build(),
        );
        ctx.finish(best)
    }
}

/// Log-target ARIMA (sktime `TransformedTargetForecaster`).
#[derive(Clone, Debug, Default)]
pub struct TransformedTargetForecaster;

impl TransformedTargetForecaster {
    /// Default log-target forecaster.
    pub fn new() -> Self {
        Self
    }
}

impl FitSeries for TransformedTargetForecaster {
    type Fitted = FittedForecastingPipeline;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedForecastingPipeline>> {
        ForecastingPipeline { log: true }.fit_series(y, session)
    }
}

/// Seasonal ARIMA: apply seasonal differences, then [`Arima`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sarima {
    /// Non-seasonal `(p,d,q)`.
    pub order: (usize, usize, usize),
    /// Seasonal `(P,D,Q)`.
    pub seasonal_order: (usize, usize, usize),
    /// Seasonal period.
    pub period: usize,
}

impl Default for Sarima {
    fn default() -> Self {
        Self {
            order: (1, 0, 0),
            seasonal_order: (0, 1, 0),
            period: 12,
        }
    }
}

impl Sarima {
    /// Construct a seasonal ARIMA specification.
    pub fn new(
        order: (usize, usize, usize),
        seasonal_order: (usize, usize, usize),
        period: usize,
    ) -> Self {
        Self {
            order,
            seasonal_order,
            period,
        }
    }
}

/// Fitted SARIMA (seasonal differences + inner ARIMA).
#[derive(Clone, Debug)]
pub struct FittedSarima {
    /// Specification.
    pub spec: Sarima,
    /// ARIMA fitted on the seasonally differenced series.
    pub inner: FittedArima,
    /// Stages of seasonal differencing (for inversion).
    pub seasonal_levels: Vec<Vector>,
}

impl FittedSarima {
    /// Forecast on the original scale.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let q = self.inner.forecast(h, session)?;
        let ctx = FitCtx::with_session(session.child("sarima-undiff"));
        let mut cur = q.value.as_slice().to_vec();
        let period = self.spec.period.max(1);
        for stage in self.seasonal_levels.iter().rev() {
            let mut last: Vec<f64> = stage.as_slice().to_vec();
            let mut out = Vec::with_capacity(cur.len());
            for &dz in &cur {
                let prev = if last.len() >= period {
                    last[last.len() - period]
                } else {
                    *last.last().unwrap_or(&0.0)
                };
                let y = prev + dz;
                last.push(y);
                out.push(y);
            }
            cur = out;
        }
        ctx.finish(Vector::from_iter(cur))
    }
}

impl FitSeries for Sarima {
    type Fitted = FittedSarima;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedSarima>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let period = self.period.max(1);
        let d_s = self.seasonal_order.1;
        if d_s > 0 && y.len() < 2 * period {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSeasonalCycles)
                    .severity(Severity::Error)
                    .message(format!(
                        "SARIMA seasonal difference needs n≥2s; n={} s={period}",
                        y.len()
                    ))
                    .build(),
            );
        }
        let (z, stages) = seasonal_difference(y.as_slice(), period, d_s);
        let mut arima = Arima {
            p: self.order.0 + self.seasonal_order.0,
            d: self.order.1,
            q: self.order.2 + self.seasonal_order.2,
        };
        let inner_q = match arima.fit_series(&Vector::from_iter(z), session) {
            Ok(q) => q,
            Err(e) => {
                for issue in e.report.issues() {
                    ctx.push(issue.clone());
                }
                return ctx.finish(FittedSarima {
                    spec: self.clone(),
                    inner: empty_arima(&arima),
                    seasonal_levels: stages
                        .iter()
                        .map(|s| Vector::from_iter(s.iter().copied()))
                        .collect(),
                });
            }
        };
        for issue in inner_q.report.issues() {
            ctx.push(issue.clone());
        }
        ctx.finish(FittedSarima {
            spec: self.clone(),
            inner: inner_q.value,
            seasonal_levels: stages
                .iter()
                .map(|s| Vector::from_iter(s.iter().copied()))
                .collect(),
        })
    }
}

/// Vector autoregression of order `lags` (each equation is OLS on stacked lags).
#[derive(Clone, Debug)]
pub struct Var {
    /// VAR order.
    pub lags: usize,
}

impl Default for Var {
    fn default() -> Self {
        Self { lags: 1 }
    }
}

impl Var {
    /// VAR(`lags`).
    pub fn new(lags: usize) -> Self {
        Self { lags }
    }

    /// Fit on an `n × k` series matrix (columns are variables).
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedVar>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let (n, k) = y.shape();
        let p = self.lags.max(1);
        let n_eff = n.saturating_sub(p);
        let npar = 1 + p * k;
        inspect_identification(&mut ctx.report, n_eff, npar, &ctx.policy);
        if n_eff <= npar {
            ctx.push(
                Issue::builder(IssueCode::ShortSeriesForArima)
                    .message(format!("VAR n_eff={n_eff} ≤ parameters {npar}"))
                    .build(),
            );
        }
        let mut coef = Matrix::zeros(npar, k);
        let mut intercepts = Vector::zeros(k);
        let mut resid = Matrix::zeros(n_eff, k);
        for eq in 0..k {
            let target = Vector::from_iter((p..n).map(|t| y.get(t, eq)));
            let design = Matrix::from_fn(n_eff, npar, |i, j| {
                let t = p + i;
                if j == 0 {
                    1.0
                } else {
                    let jj = j - 1;
                    let lag = jj / k + 1;
                    let var = jj % k;
                    y.get(t - lag, var)
                }
            });
            match statistical_ols(&mut ctx, &design, &target) {
                Some(b) => {
                    intercepts[eq] = b[0];
                    for j in 0..npar {
                        coef.set(j, eq, b[j]);
                    }
                    let fit = design.matvec(&b);
                    for i in 0..n_eff {
                        resid.set(i, eq, target[i] - fit[i]);
                    }
                }
                None => {
                    ctx.push(
                        Issue::builder(IssueCode::UnidentifiedModel)
                            .message(format!("VAR equation {eq} OLS failed"))
                            .build(),
                    );
                }
            }
        }
        let last = Matrix::from_fn(p, k, |i, j| y.get(n - p + i, j));
        warn_var_unit_root(&mut ctx, &coef, k, p);
        ctx.finish(FittedVar {
            lags: p,
            k,
            coef,
            intercepts,
            resid,
            last,
        })
    }
}

/// Fitted VAR.
#[derive(Clone, Debug)]
pub struct FittedVar {
    /// VAR order.
    pub lags: usize,
    /// Number of series.
    pub k: usize,
    /// Coefficients including intercept (rows) by equation (columns).
    pub coef: Matrix,
    /// Intercepts (also stored in `coef` row 0).
    pub intercepts: Vector,
    /// In-sample residuals (`n−lags` × `k`).
    pub resid: Matrix,
    /// Last `lags` rows of the training series.
    pub last: Matrix,
}

impl FittedVar {
    /// Iterate the companion form `h` steps.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let mut hist = self.last.clone();
        let mut out = Matrix::zeros(h, self.k);
        for step in 0..h {
            let mut yhat = Vector::zeros(self.k);
            for eq in 0..self.k {
                let mut v = self.intercepts[eq];
                for lag in 1..=self.lags {
                    for var in 0..self.k {
                        let row = 1 + (lag - 1) * self.k + var;
                        v += self.coef.get(row, eq) * hist.get(self.lags - lag, var);
                    }
                }
                yhat[eq] = v;
            }
            for eq in 0..self.k {
                out.set(step, eq, yhat[eq]);
            }
            if self.lags > 0 {
                let mut nxt = Matrix::zeros(self.lags, self.k);
                for i in 0..self.lags.saturating_sub(1) {
                    for j in 0..self.k {
                        nxt.set(i, j, hist.get(i + 1, j));
                    }
                }
                for j in 0..self.k {
                    nxt.set(self.lags - 1, j, yhat[j]);
                }
                hist = nxt;
            }
        }
        ctx.finish(out)
    }

    /// Cholesky impulse responses and forecast-error variance decompositions.
    ///
    /// Orthogonalisation is the residual Cholesky, not a structural SVAR.
    /// Lag count is not an identification `p` for this post-estimation map.
    pub fn impulse_response(
        &self,
        horizon: usize,
        session: &Session,
    ) -> Result<Qualified<VarImpulseResponse>> {
        let mut ctx = FitCtx::with_session(session.child("irf"));
        let k = self.k;
        let p = self.lags.max(1);
        let h = horizon.max(1);
        let (t_res, _) = self.resid.shape();
        let mut sigma = Matrix::zeros(k, k);
        if t_res > 0 && k > 0 {
            for a in 0..k {
                for b in 0..=a {
                    let mut s = 0.0;
                    for i in 0..t_res {
                        s += self.resid.get(i, a) * self.resid.get(i, b);
                    }
                    let v = s / t_res as f64;
                    sigma.set(a, b, v);
                    sigma.set(b, a, v);
                }
            }
        }
        let chol = match cholesky_lower(&sigma) {
            Some(l) => l,
            None => {
                ctx.push(
                    Issue::builder(IssueCode::JitterInjected)
                        .message("VAR residual covariance was not SPD; IRF uses a diagonal jitter")
                        .compromise(NumericalCompromise::new(
                            "Cholesky of the residual covariance",
                            "diagonally jittered residual scale",
                            "the Gram of residuals was not positive definite",
                            "do not read the IRF as a unique structural shock",
                        ))
                        .build(),
                );
                Matrix::from_fn(k, k, |i, j| {
                    if i == j {
                        sigma.get(i, i).abs().sqrt().max(1e-6)
                    } else {
                        0.0
                    }
                })
            }
        };
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("VAR IRF uses a residual Cholesky, not identified structural shocks")
                .compromise(NumericalCompromise::new(
                    "SVAR with exclusion / sign restrictions",
                    "reduced-form MA with Cholesky P",
                    "the ordering of the columns is the variable order",
                    "do not treat the first shock as causal without a theory",
                ))
                .build(),
        );
        let mut psi = vec![Matrix::zeros(k, k); h + 1];
        for i in 0..k {
            psi[0].set(i, i, 1.0);
        }
        for hor in 1..=h {
            for lag in 1..=p.min(hor) {
                for eq in 0..k {
                    for var in 0..k {
                        let a = self.coef.get(1 + (lag - 1) * k + var, eq);
                        for sh in 0..k {
                            let v = psi[hor].get(eq, sh) + a * psi[hor - lag].get(var, sh);
                            psi[hor].set(eq, sh, v);
                        }
                    }
                }
            }
        }
        let mut irf = Vec::with_capacity(h + 1);
        let mut mse = vec![0.0; k];
        let mut fevd = Vec::with_capacity(h + 1);
        let mut acc_sq = Matrix::zeros(k, k);
        for hor in 0..=h {
            let th = Matrix::from_fn(k, k, |i, j| {
                let mut v = 0.0;
                for r in 0..k {
                    v += psi[hor].get(i, r) * chol.get(r, j);
                }
                v
            });
            for i in 0..k {
                for j in 0..k {
                    let s = th.get(i, j);
                    acc_sq.set(i, j, acc_sq.get(i, j) + s * s);
                    mse[i] += s * s;
                }
            }
            let decomp = Matrix::from_fn(k, k, |i, j| {
                if mse[i] > 1e-18 {
                    acc_sq.get(i, j) / mse[i]
                } else {
                    0.0
                }
            });
            irf.push(th);
            fevd.push(decomp);
        }
        ctx.finish(VarImpulseResponse { irf, fevd, sigma })
    }
}

/// Cholesky IRF and FEVD of a fitted VAR (statsmodels `VARResults.irf` / `fevd`).
#[derive(Clone, Debug)]
pub struct VarImpulseResponse {
    /// Orthogonal MA matrices \(\Theta_h=\Psi_h P\) (`horizon+1` of them, each \(k\times k\)).
    pub irf: Vec<Matrix>,
    /// Forecast-error variance shares at each horizon (rows sum to 1).
    pub fevd: Vec<Matrix>,
    /// Residual covariance used for the Cholesky factor.
    pub sigma: Matrix,
}

/// Structural VAR with a recursive (Cholesky) contemporaneous map.
///
/// This is the named SVAR surface around [`FittedVar::impulse_response`]. The
/// `A0` factor is not estimated; it is the residual Cholesky.
#[derive(Clone, Debug)]
pub struct Svar {
    /// VAR order.
    pub lags: usize,
}

impl Default for Svar {
    fn default() -> Self {
        Self { lags: 1 }
    }
}

impl Svar {
    /// SVAR(`lags`).
    pub fn new(lags: usize) -> Self {
        Self { lags }
    }

    /// Fit the reduced-form VAR and keep it as a recursive SVAR.
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedSvar>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let q = match Var::new(self.lags).fit(y, &session.child("var")) {
            Ok(q) => q,
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                ) {
                    ctx.push(e.primary);
                }
                return ctx.finish(FittedSvar {
                    reduced: FittedVar {
                        lags: self.lags.max(1),
                        k: y.ncols(),
                        coef: Matrix::zeros(1, y.ncols()),
                        intercepts: Vector::zeros(y.ncols()),
                        resid: Matrix::zeros(0, y.ncols()),
                        last: Matrix::zeros(self.lags.max(1), y.ncols()),
                    },
                });
            }
        };
        for issue in q.report.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("SVAR uses a recursive Cholesky A0, not an estimated A/B system")
                .compromise(NumericalCompromise::new(
                    "identified structural VAR",
                    "reduced-form VAR plus residual Cholesky",
                    "the shock order is the column order of Y",
                    "do not read the first shock as causal without a theory",
                ))
                .build(),
        );
        ctx.finish(FittedSvar { reduced: q.value })
    }
}

/// Fitted recursive SVAR.
#[derive(Clone, Debug)]
pub struct FittedSvar {
    /// Reduced-form VAR.
    pub reduced: FittedVar,
}

impl FittedSvar {
    /// Structural IRF / FEVD via the residual Cholesky.
    pub fn structural_irf(
        &self,
        horizon: usize,
        session: &Session,
    ) -> Result<Qualified<VarImpulseResponse>> {
        self.reduced.impulse_response(horizon, session)
    }
}

fn var_psi(reduced: &FittedVar, horizon: usize) -> Vec<Matrix> {
    let k = reduced.k;
    let p = reduced.lags.max(1);
    let h = horizon.max(1);
    let mut psi = vec![Matrix::zeros(k, k); h + 1];
    for i in 0..k {
        psi[0].set(i, i, 1.0);
    }
    for hor in 1..=h {
        for lag in 1..=p.min(hor) {
            for eq in 0..k {
                for var in 0..k {
                    let a = reduced.coef.get(1 + (lag - 1) * k + var, eq);
                    for sh in 0..k {
                        let v = psi[hor].get(eq, sh) + a * psi[hor - lag].get(var, sh);
                        psi[hor].set(eq, sh, v);
                    }
                }
            }
        }
    }
    psi
}

fn irf_from_impact(psi: &[Matrix], impact: &Matrix, sigma: Matrix) -> VarImpulseResponse {
    let k = impact.nrows();
    let mut irf = Vec::with_capacity(psi.len());
    let mut mse = vec![0.0; k];
    let mut fevd = Vec::with_capacity(psi.len());
    let mut acc_sq = Matrix::zeros(k, k);
    for psi_h in psi {
        let th = Matrix::from_fn(k, k, |i, j| {
            let mut v = 0.0;
            for r in 0..k {
                v += psi_h.get(i, r) * impact.get(r, j);
            }
            v
        });
        for i in 0..k {
            for j in 0..k {
                let s = th.get(i, j);
                acc_sq.set(i, j, acc_sq.get(i, j) + s * s);
                mse[i] += s * s;
            }
        }
        let decomp = Matrix::from_fn(k, k, |i, j| {
            if mse[i] > 1e-18 {
                acc_sq.get(i, j) / mse[i]
            } else {
                0.0
            }
        });
        irf.push(th);
        fevd.push(decomp);
    }
    VarImpulseResponse { irf, fevd, sigma }
}

/// Estimated recursive A/B SVAR: \(A u_t = B \varepsilon_t\).
///
/// `A` is unit lower-triangular from residual regressions (not the Cholesky
/// factor of \(\Sigma_u\)). `B` is diagonal residual scale. This is still a
/// recursive identification; free A and B without extra restrictions are not
/// jointly identified.
#[derive(Clone, Debug)]
pub struct SvarAb {
    /// VAR order.
    pub lags: usize,
}

impl Default for SvarAb {
    fn default() -> Self {
        Self { lags: 1 }
    }
}

impl SvarAb {
    /// A/B SVAR(`lags`).
    pub fn new(lags: usize) -> Self {
        Self { lags }
    }

    /// Fit the reduced-form VAR and estimate recursive A, B from residuals.
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedSvarAb>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let q = match Var::new(self.lags).fit(y, &session.child("var")) {
            Ok(q) => q,
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                ) {
                    ctx.push(e.primary);
                }
                let k = y.ncols();
                return ctx.finish(FittedSvarAb {
                    reduced: FittedVar {
                        lags: self.lags.max(1),
                        k,
                        coef: Matrix::zeros(1, k),
                        intercepts: Vector::zeros(k),
                        resid: Matrix::zeros(0, k),
                        last: Matrix::zeros(self.lags.max(1), k),
                    },
                    a0: Matrix::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 }),
                    b: Matrix::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 }),
                });
            }
        };
        for issue in q.report.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let k = q.value.k;
        let (t_res, _) = q.value.resid.shape();
        let mut a0 = Matrix::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 });
        let mut b = Matrix::zeros(k, k);
        if t_res > 0 && k > 0 {
            let u0 = Vector::from_iter((0..t_res).map(|i| q.value.resid.get(i, 0)));
            let mut sse0 = 0.0;
            for i in 0..t_res {
                sse0 += u0[i] * u0[i];
            }
            b.set(0, 0, (sse0 / t_res as f64).max(1e-18).sqrt());
            for i in 1..k {
                let target = Vector::from_iter((0..t_res).map(|t| q.value.resid.get(t, i)));
                let design = Matrix::from_fn(t_res, i, |t, j| q.value.resid.get(t, j));
                let mut scratch = Report::new("svarab", "a0");
                if let Some(coef) =
                    crate::linalg::least_squares(&mut scratch, &design, &target, &ctx.policy)
                {
                    for j in 0..i {
                        a0.set(i, j, -coef[j]);
                    }
                    let fit = design.matvec(&coef);
                    let mut sse = 0.0;
                    for t in 0..t_res {
                        let e = target[t] - fit[t];
                        sse += e * e;
                    }
                    b.set(i, i, (sse / t_res as f64).max(1e-18).sqrt());
                } else {
                    let mut sse = 0.0;
                    for t in 0..t_res {
                        sse += target[t] * target[t];
                    }
                    b.set(i, i, (sse / t_res as f64).max(1e-18).sqrt());
                }
            }
        } else {
            for i in 0..k {
                b.set(i, i, 1.0);
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message(
                    "SVAR-AB estimates a recursive A0 from residual OLS, not a free A/B system",
                )
                .compromise(NumericalCompromise::new(
                    "over-identified A/B SVAR with exclusion restrictions",
                    "unit-lower-triangular A from residual regressions and diagonal B",
                    "free A and B are not jointly identified without extra restrictions",
                    "do not read off-diagonal A entries as unrestricted structural parameters",
                ))
                .build(),
        );
        ctx.finish(FittedSvarAb {
            reduced: q.value,
            a0,
            b,
        })
    }
}

/// Fitted recursive A/B SVAR.
#[derive(Clone, Debug)]
pub struct FittedSvarAb {
    /// Reduced-form VAR.
    pub reduced: FittedVar,
    /// Estimated contemporaneous \(A\) (unit lower triangular).
    pub a0: Matrix,
    /// Diagonal structural-shock scale \(B\).
    pub b: Matrix,
}

impl FittedSvarAb {
    /// Structural IRF / FEVD via \(P = A^{-1} B\).
    pub fn structural_irf(
        &self,
        horizon: usize,
        session: &Session,
    ) -> Result<Qualified<VarImpulseResponse>> {
        let ctx = FitCtx::with_session(session.child("irf"));
        let k = self.reduced.k;
        let mut impact = Matrix::zeros(k, k);
        for j in 0..k {
            for i in 0..k {
                let mut s = self.b.get(i, j);
                for p in 0..i {
                    s -= self.a0.get(i, p) * impact.get(p, j);
                }
                let d = self.a0.get(i, i);
                impact.set(i, j, if d.abs() > 1e-18 { s / d } else { s });
            }
        }
        let (t_res, _) = self.reduced.resid.shape();
        let mut sigma = Matrix::zeros(k, k);
        if t_res > 0 {
            for a in 0..k {
                for b in 0..=a {
                    let mut s = 0.0;
                    for i in 0..t_res {
                        s += self.reduced.resid.get(i, a) * self.reduced.resid.get(i, b);
                    }
                    let v = s / t_res as f64;
                    sigma.set(a, b, v);
                    sigma.set(b, a, v);
                }
            }
        }
        let psi = var_psi(&self.reduced, horizon);
        ctx.finish(irf_from_impact(&psi, &impact, sigma))
    }
}

fn invert_square(a: &Matrix) -> Option<Matrix> {
    let n = a.nrows().min(a.ncols());
    if n == 0 {
        return None;
    }
    if n == 1 {
        let d = a.get(0, 0);
        if d.abs() <= 1e-18 {
            return None;
        }
        return Some(Matrix::from_fn(1, 1, |_, _| 1.0 / d));
    }
    if n == 2 {
        let det = a.get(0, 0) * a.get(1, 1) - a.get(0, 1) * a.get(1, 0);
        if det.abs() <= 1e-18 {
            return None;
        }
        let inv = 1.0 / det;
        return Some(Matrix::from_fn(2, 2, |i, j| {
            if i == 0 && j == 0 {
                a.get(1, 1) * inv
            } else if i == 0 && j == 1 {
                -a.get(0, 1) * inv
            } else if i == 1 && j == 0 {
                -a.get(1, 0) * inv
            } else {
                a.get(0, 0) * inv
            }
        }));
    }
    let mut aug = Matrix::from_fn(n, 2 * n, |i, j| {
        if j < n {
            a.get(i, j)
        } else if j == n + i {
            1.0
        } else {
            0.0
        }
    });
    for col in 0..n {
        let mut piv = col;
        let mut best = aug.get(col, col).abs();
        for r in (col + 1)..n {
            let v = aug.get(r, col).abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best <= 1e-18 {
            return None;
        }
        if piv != col {
            for j in 0..(2 * n) {
                let tmp = aug.get(col, j);
                aug.set(col, j, aug.get(piv, j));
                aug.set(piv, j, tmp);
            }
        }
        let d = aug.get(col, col);
        for j in 0..(2 * n) {
            aug.set(col, j, aug.get(col, j) / d);
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = aug.get(r, col);
            for j in 0..(2 * n) {
                aug.set(r, j, aug.get(r, j) - f * aug.get(col, j));
            }
        }
    }
    Some(Matrix::from_fn(n, n, |i, j| aug.get(i, n + j)))
}

/// Blanchard–Quah long-run SVAR: \(C(1)=(I-\sum A_i)^{-1}P\) is lower triangular.
///
/// \(P\) is estimated from the reduced-form VAR, not assumed Cholesky of \(\Sigma_u\).
/// Lag count is not identification `p`.
#[derive(Clone, Debug)]
pub struct BlanchardQuah {
    /// VAR order.
    pub lags: usize,
}

impl Default for BlanchardQuah {
    fn default() -> Self {
        Self { lags: 1 }
    }
}

impl BlanchardQuah {
    /// Blanchard–Quah SVAR(`lags`).
    pub fn new(lags: usize) -> Self {
        Self { lags }
    }

    /// Fit the reduced-form VAR and the long-run impact \(P\).
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedBlanchardQuah>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let q = match Var::new(self.lags).fit(y, &session.child("var")) {
            Ok(q) => q,
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                ) {
                    ctx.push(e.primary);
                }
                let k = y.ncols();
                return ctx.finish(FittedBlanchardQuah {
                    reduced: FittedVar {
                        lags: self.lags.max(1),
                        k,
                        coef: Matrix::zeros(1, k),
                        intercepts: Vector::zeros(k),
                        resid: Matrix::zeros(0, k),
                        last: Matrix::zeros(self.lags.max(1), k),
                    },
                    long_run: Matrix::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 }),
                    impact: Matrix::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 }),
                });
            }
        };
        for issue in q.report.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let k = q.value.k;
        let p = q.value.lags.max(1);
        let mut phi = Matrix::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 });
        for lag in 1..=p {
            for eq in 0..k {
                for var in 0..k {
                    let row = 1 + (lag - 1) * k + var;
                    if row < q.value.coef.nrows() {
                        phi.set(eq, var, phi.get(eq, var) - q.value.coef.get(row, eq));
                    }
                }
            }
        }
        let (t_res, _) = q.value.resid.shape();
        let mut sigma = Matrix::zeros(k, k);
        if t_res > 0 {
            for a in 0..k {
                for b in 0..=a {
                    let mut s = 0.0;
                    for i in 0..t_res {
                        s += q.value.resid.get(i, a) * q.value.resid.get(i, b);
                    }
                    let v = s / t_res as f64;
                    sigma.set(a, b, v);
                    sigma.set(b, a, v);
                }
            }
        }
        let long_cov = Matrix::from_fn(k, k, |i, j| {
            let mut s = 0.0;
            for r in 0..k {
                for c in 0..k {
                    s += phi.get(i, r) * sigma.get(r, c) * phi.get(j, c);
                }
            }
            s
        });
        let c_lr = match cholesky_lower(&long_cov) {
            Some(l) => l,
            None => {
                ctx.push(
                    Issue::builder(IssueCode::JitterInjected)
                        .message("Blanchard–Quah long-run covariance was not SPD; using a diagonal")
                        .compromise(NumericalCompromise::new(
                            "Cholesky of Φ Σ Φ′",
                            "diagonal long-run scale",
                            "the estimated long-run Gram was not positive definite",
                            "do not read the first shock as a unique demand/supply innovation",
                        ))
                        .build(),
                );
                Matrix::from_fn(k, k, |i, j| {
                    if i == j {
                        long_cov.get(i, i).abs().sqrt().max(1e-6)
                    } else {
                        0.0
                    }
                })
            }
        };
        let impact = match invert_square(&phi) {
            Some(inv) => Matrix::from_fn(k, k, |i, j| {
                let mut s = 0.0;
                for r in 0..k {
                    s += inv.get(i, r) * c_lr.get(r, j);
                }
                s
            }),
            None => {
                ctx.push(
                    Issue::builder(IssueCode::NearSingular)
                        .severity(Severity::Warning)
                        .message("Blanchard–Quah Φ = I−∑A is singular; impact falls back to C(1)")
                        .build(),
                );
                c_lr.clone()
            }
        };
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message(
                    "Blanchard–Quah imposes a lower-triangular long-run C(1), not a theory-free P",
                )
                .compromise(NumericalCompromise::new(
                    "just-identified long-run SVAR",
                    "C(1) = chol(Φ Σ Φ′) and P = Φ⁻¹ C(1)",
                    "the first shock is the only one with a long-run level effect by construction",
                    "do not label shocks demand/supply without that restriction being true",
                ))
                .build(),
        );
        ctx.finish(FittedBlanchardQuah {
            reduced: q.value,
            long_run: c_lr,
            impact,
        })
    }
}

/// Fitted Blanchard–Quah SVAR.
#[derive(Clone, Debug)]
pub struct FittedBlanchardQuah {
    /// Reduced-form VAR.
    pub reduced: FittedVar,
    /// Long-run impact \(C(1)\).
    pub long_run: Matrix,
    /// Short-run structural impact \(P=\Phi^{-1}C(1)\).
    pub impact: Matrix,
}

impl FittedBlanchardQuah {
    /// Structural IRF / FEVD via the Blanchard–Quah \(P\).
    pub fn structural_irf(
        &self,
        horizon: usize,
        session: &Session,
    ) -> Result<Qualified<VarImpulseResponse>> {
        let ctx = FitCtx::with_session(session.child("irf"));
        let k = self.reduced.k;
        let (t_res, _) = self.reduced.resid.shape();
        let mut sigma = Matrix::zeros(k, k);
        if t_res > 0 {
            for a in 0..k {
                for b in 0..=a {
                    let mut s = 0.0;
                    for i in 0..t_res {
                        s += self.reduced.resid.get(i, a) * self.reduced.resid.get(i, b);
                    }
                    let v = s / t_res as f64;
                    sigma.set(a, b, v);
                    sigma.set(b, a, v);
                }
            }
        }
        let psi = var_psi(&self.reduced, horizon);
        ctx.finish(irf_from_impact(&psi, &self.impact, sigma))
    }
}

fn cholesky_lower(a: &Matrix) -> Option<Matrix> {
    let n = a.nrows().min(a.ncols());
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut s = a.get(i, j);
            for p in 0..j {
                s -= l.get(i, p) * l.get(j, p);
            }
            if i == j {
                if s <= 1e-18 {
                    return None;
                }
                l.set(i, i, s.sqrt());
            } else {
                let d = l.get(j, j);
                if d.abs() <= 1e-18 {
                    return None;
                }
                l.set(i, j, s / d);
            }
        }
    }
    Some(l)
}

/// VARMAX: VAR with contemporaneous exogenous regressors.
///
/// Lag count is not passed as identification `p` beyond the VAR design;
/// exogenous width is included in the equation parameter count only when
/// `n_eff` is large enough that the usual OLS gate is meaningful.
#[derive(Clone, Debug)]
pub struct Varmax {
    /// VAR order.
    pub lags: usize,
}

impl Default for Varmax {
    fn default() -> Self {
        Self { lags: 1 }
    }
}

impl Varmax {
    /// VARMAX(`lags`).
    pub fn new(lags: usize) -> Self {
        Self { lags: lags.max(1) }
    }

    /// Fit `Y` on its lags and exogenous `X`.
    pub fn fit(
        &self,
        y: &Matrix,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedVarmax>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if y.nrows() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("VARMAX Y rows ≠ X rows")
                    .build(),
            );
        }
        let n = y.nrows().min(x.nrows());
        let k = y.ncols();
        let kx = x.ncols();
        let p = self.lags.max(1);
        let n_eff = n.saturating_sub(p);
        let npar = 1 + p * k + kx;
        if n_eff > npar + 8 {
            inspect_identification(&mut ctx.report, n_eff, npar, &ctx.policy);
        }
        let mut coef = Matrix::zeros(npar, k);
        let mut intercepts = Vector::zeros(k);
        for eq in 0..k {
            let target = Vector::from_iter((p..n).map(|t| y.get(t, eq)));
            let design = Matrix::from_fn(n_eff, npar, |i, j| {
                let t = p + i;
                if j == 0 {
                    1.0
                } else if j <= p * k {
                    let jj = j - 1;
                    let lag = jj / k.max(1) + 1;
                    let var = jj % k.max(1);
                    y.get(t - lag, var)
                } else {
                    x.get(t, j - 1 - p * k)
                }
            });
            if let Some(b) = statistical_ols(&mut ctx, &design, &target) {
                intercepts[eq] = b[0];
                for j in 0..npar.min(b.len()) {
                    coef.set(j, eq, b[j]);
                }
            }
        }
        ctx.finish(FittedVarmax {
            lags: p,
            k,
            kx,
            coef,
            intercepts,
            last_y: Matrix::from_fn(p, k, |i, j| y.get(n - p + i, j)),
            last_x: if n > 0 {
                Matrix::from_fn(1, kx, |_, j| x.get(n - 1, j))
            } else {
                Matrix::zeros(1, kx)
            },
        })
    }
}

/// Fitted VARMAX.
#[derive(Clone, Debug)]
pub struct FittedVarmax {
    /// VAR order.
    pub lags: usize,
    /// Number of series.
    pub k: usize,
    /// Number of exogenous columns.
    pub kx: usize,
    /// Coefficients including intercept.
    pub coef: Matrix,
    /// Intercepts.
    pub intercepts: Vector,
    /// Last `lags` rows of `Y`.
    pub last_y: Matrix,
    /// Last exogenous row.
    pub last_x: Matrix,
}

impl FittedVarmax {
    /// Forecast with a future exogenous path.
    pub fn forecast(&self, x_future: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        let h = x_future.nrows();
        let mut hist = self.last_y.clone();
        let mut out = Matrix::zeros(h, self.k);
        for step in 0..h {
            let mut yhat = Vector::zeros(self.k);
            for eq in 0..self.k {
                let mut v = self.intercepts[eq];
                for lag in 1..=self.lags {
                    for var in 0..self.k {
                        let row = 1 + (lag - 1) * self.k + var;
                        v += self.coef.get(row, eq) * hist.get(self.lags - lag, var);
                    }
                }
                for c in 0..self.kx {
                    let row = 1 + self.lags * self.k + c;
                    v += self.coef.get(row, eq) * x_future.get(step, c);
                }
                yhat[eq] = v;
            }
            for eq in 0..self.k {
                out.set(step, eq, yhat[eq]);
            }
            if self.lags > 0 {
                let mut nxt = Matrix::zeros(self.lags, self.k);
                for i in 0..self.lags.saturating_sub(1) {
                    for j in 0..self.k {
                        nxt.set(i, j, hist.get(i + 1, j));
                    }
                }
                for j in 0..self.k {
                    nxt.set(self.lags - 1, j, yhat[j]);
                }
                hist = nxt;
            }
        }
        ctx.finish(out)
    }
}

/// Dynamic-factor model: PCA factors then VAR(1) on the factors.
///
/// Factor count is not passed as identification `p`.
#[derive(Clone, Debug)]
pub struct DynamicFactor {
    /// Number of latent factors.
    pub n_factors: usize,
}

impl Default for DynamicFactor {
    fn default() -> Self {
        Self { n_factors: 1 }
    }
}

impl DynamicFactor {
    /// `r` factors.
    pub fn new(n_factors: usize) -> Self {
        Self {
            n_factors: n_factors.max(1),
        }
    }

    /// Fit on an `n × k` panel of series.
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedDynamicFactor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let (n, k) = y.shape();
        if n == 0 || k == 0 {
            return ctx.finish(FittedDynamicFactor {
                loadings: Matrix::zeros(k, self.n_factors.max(1)),
                var: FittedVar {
                    lags: 1,
                    k: 1,
                    coef: Matrix::zeros(2, 1),
                    intercepts: Vector::zeros(1),
                    resid: Matrix::zeros(0, 1),
                    last: Matrix::zeros(1, 1),
                },
                mean: Vector::zeros(k),
            });
        }
        let (yc, mean) = y.centered();
        let mut scratch = Report::new("dfm", "svd");
        let Some(svd) = thin_svd(&mut scratch, &yc, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::SvdDidNotConverge)
                    .message("DynamicFactor SVD failed")
                    .build(),
            );
            return ctx.finish(FittedDynamicFactor {
                loadings: Matrix::zeros(k, 1),
                var: FittedVar {
                    lags: 1,
                    k: 1,
                    coef: Matrix::zeros(2, 1),
                    intercepts: Vector::zeros(1),
                    resid: Matrix::zeros(0, 1),
                    last: Matrix::zeros(1, 1),
                },
                mean,
            });
        };
        let r = self.n_factors.max(1).min(svd.singular_values.len()).min(k);
        let loadings = Matrix::from_fn(k, r, |j, c| svd.v[(j, c)]);
        let factors = Matrix::from_fn(n, r, |i, c| {
            let mut s = 0.0;
            for j in 0..k {
                s += yc.get(i, j) * loadings.get(j, c);
            }
            s
        });
        let var = match Var::new(1).fit(&factors, &session.child("dfm-var")) {
            Ok(q) => q.value,
            Err(_) => FittedVar {
                lags: 1,
                k: r,
                coef: Matrix::zeros(1 + r, r),
                intercepts: Vector::zeros(r),
                resid: Matrix::zeros(0, r),
                last: Matrix::from_fn(1, r, |_, j| factors.get(n.saturating_sub(1), j)),
            },
        };
        ctx.finish(FittedDynamicFactor {
            loadings,
            var,
            mean,
        })
    }
}

/// Fitted dynamic-factor model.
#[derive(Clone, Debug)]
pub struct FittedDynamicFactor {
    /// Series loadings (`k` × `r`).
    pub loadings: Matrix,
    /// VAR on the factors.
    pub var: FittedVar,
    /// Series means.
    pub mean: Vector,
}

impl FittedDynamicFactor {
    /// `h`-step factor forecast mapped back to the series.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let f = self.var.forecast(h, session)?;
        let mut ctx = FitCtx::with_session(session.child("dfm-map"));
        let k = self.loadings.nrows();
        let r = self.loadings.ncols();
        let y = Matrix::from_fn(h, k, |t, j| {
            let mut s = if j < self.mean.len() {
                self.mean[j]
            } else {
                0.0
            };
            for c in 0..r.min(f.value.ncols()) {
                s += f.value.get(t, c) * self.loadings.get(j, c);
            }
            s
        });
        ctx.finish(y)
    }
}

/// Simple / Holt (linear trend) exponential smoothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothingKind {
    /// Simple exponential smoothing (level only).
    Simple,
    /// Holt linear trend.
    Holt,
}

/// Exponential smoothing specification.
#[derive(Clone, Debug)]
pub struct ExponentialSmoothing {
    /// Simple vs Holt.
    pub kind: SmoothingKind,
    /// Level constant; `None` grid-searches.
    pub alpha: Option<f64>,
    /// Trend constant (Holt only).
    pub beta: Option<f64>,
}

impl Default for ExponentialSmoothing {
    fn default() -> Self {
        Self {
            kind: SmoothingKind::Simple,
            alpha: None,
            beta: None,
        }
    }
}

impl ExponentialSmoothing {
    /// Simple exponential smoothing.
    pub fn simple() -> Self {
        Self::default()
    }

    /// Holt linear trend.
    pub fn holt() -> Self {
        Self {
            kind: SmoothingKind::Holt,
            alpha: None,
            beta: None,
        }
    }
}

/// Fitted exponential-smoothing state.
#[derive(Clone, Debug)]
pub struct FittedEsm {
    /// Kind.
    pub kind: SmoothingKind,
    /// Level smoothing.
    pub alpha: f64,
    /// Trend smoothing (0 for SES).
    pub beta: f64,
    /// Terminal level.
    pub level: f64,
    /// Terminal trend.
    pub trend: f64,
    /// In-sample fitted values.
    pub fitted: Vector,
}

impl FittedEsm {
    /// Forecast `h` steps (SES is flat; Holt adds `h · trend`).
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let out = Vector::from_iter((1..=h).map(|k| match self.kind {
            SmoothingKind::Simple => self.level,
            SmoothingKind::Holt => self.level + k as f64 * self.trend,
        }));
        ctx.finish(out)
    }
}

impl FitSeries for ExponentialSmoothing {
    type Fitted = FittedEsm;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedEsm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("exponential smoothing needs n≥2")
                    .build(),
            );
        }
        warn_unit_root(&mut ctx, y);
        let (alpha, beta, level, trend, fitted) =
            esm_fit(y.as_slice(), self.kind, self.alpha, self.beta);
        ctx.finish(FittedEsm {
            kind: self.kind,
            alpha,
            beta,
            level,
            trend,
            fitted,
        })
    }
}

/// Last-value (random-walk) forecaster.
#[derive(Clone, Debug, Default)]
pub struct Naive;

/// Fitted naive forecaster.
#[derive(Clone, Debug)]
pub struct FittedNaive {
    /// Last observed value.
    pub last: f64,
    /// Training length.
    pub n: usize,
}

impl FittedNaive {
    /// Repeat the last value `h` times.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::filled(h, self.last))
    }
}

impl FitSeries for Naive {
    type Fitted = FittedNaive;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedNaive>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.is_empty() {
            return ctx.finish(FittedNaive {
                last: f64::NAN,
                n: 0,
            });
        }
        ctx.finish(FittedNaive {
            last: y[y.len() - 1],
            n: y.len(),
        })
    }
}

/// Residual-bootstrap bagging of a last-value walk (sktime `BaggingForecaster`).
///
/// Bootstrap count is not identification `p`.
#[derive(Clone, Debug)]
pub struct BaggingForecaster {
    /// Number of residual-bootstrap paths.
    pub n_estimators: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for BaggingForecaster {
    fn default() -> Self {
        Self {
            n_estimators: 8,
            seed: 0,
        }
    }
}

impl BaggingForecaster {
    /// `n_estimators` bootstrap paths.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators: n_estimators.max(1),
            ..Self::default()
        }
    }
}

/// Fitted residual-bootstrap bagging forecaster.
#[derive(Clone, Debug)]
pub struct FittedBaggingForecaster {
    /// Last observed level.
    pub last: f64,
    residuals: Vector,
    n_estimators: usize,
    seed: u64,
}

impl FittedBaggingForecaster {
    /// Average of residual-bootstrap random-walk paths.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        if h == 0 {
            return ctx.finish(Vector::zeros(0));
        }
        let bags = self.n_estimators.max(1);
        let r = self.residuals.len();
        if r == 0 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("BaggingForecaster has no residuals; repeating the last value")
                    .build(),
            );
            return ctx.finish(Vector::filled(h, self.last));
        }
        let mut acc = Vector::zeros(h);
        let mut rng = Rng::new(self.seed ^ 0xBA6);
        for _ in 0..bags {
            let mut level = self.last;
            for t in 0..h {
                let j = rng.below(r);
                level += self.residuals[j.min(r - 1)];
                acc[t] += level;
            }
        }
        let nf = bags as f64;
        for t in 0..h {
            acc[t] /= nf;
        }
        ctx.finish(acc)
    }
}

impl FitSeries for BaggingForecaster {
    type Fitted = FittedBaggingForecaster;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedBaggingForecaster>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("BaggingForecaster needs n≥2 to form residuals")
                    .build(),
            );
            return ctx.finish(FittedBaggingForecaster {
                last: y.as_slice().last().copied().unwrap_or(f64::NAN),
                residuals: Vector::zeros(0),
                n_estimators: self.n_estimators.max(1),
                seed: self.seed,
            });
        }
        let residuals = Vector::from_iter((1..y.len()).map(|t| y[t] - y[t - 1]));
        ctx.finish(FittedBaggingForecaster {
            last: y[y.len() - 1],
            residuals,
            n_estimators: self.n_estimators.max(1),
            seed: self.seed,
        })
    }
}

/// Split-conformal intervals around a last-value forecast (sktime conformal).
///
/// Coverage is not identification `p`.
#[derive(Clone, Debug)]
pub struct NaiveConformal {
    /// Nominal coverage in (0, 1).
    pub coverage: f64,
}

impl Default for NaiveConformal {
    fn default() -> Self {
        Self { coverage: 0.8 }
    }
}

impl NaiveConformal {
    /// Nominal coverage `coverage`.
    pub fn new(coverage: f64) -> Self {
        Self { coverage }
    }
}

/// Fitted last-value conformal interval.
#[derive(Clone, Debug)]
pub struct FittedNaiveConformal {
    /// Last observed level.
    pub last: f64,
    /// Half-width from residual quantiles.
    pub half_width: f64,
    /// Effective coverage used.
    pub coverage: f64,
}

impl FittedNaiveConformal {
    /// Repeat the last value.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::filled(h, self.last))
    }

    /// Columns `[lower, mid, upper]` for `h` steps.
    pub fn interval(&self, h: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("interval"));
        let w = self.half_width.max(0.0);
        ctx.finish(Matrix::from_fn(h, 3, |_, j| match j {
            0 => self.last - w,
            2 => self.last + w,
            _ => self.last,
        }))
    }
}

impl FitSeries for NaiveConformal {
    type Fitted = FittedNaiveConformal;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedNaiveConformal>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let cov = if self.coverage.is_finite() && self.coverage > 0.0 && self.coverage < 1.0 {
            self.coverage
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "NaiveConformal coverage={} is not in (0,1); using 0.8",
                        self.coverage
                    ))
                    .build(),
            );
            0.8
        };
        if y.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("NaiveConformal needs n≥2 residuals")
                    .build(),
            );
            return ctx.finish(FittedNaiveConformal {
                last: y.as_slice().last().copied().unwrap_or(f64::NAN),
                half_width: 0.0,
                coverage: cov,
            });
        }
        let mut absr: Vec<f64> = (1..y.len()).map(|t| (y[t] - y[t - 1]).abs()).collect();
        absr.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let alpha = (1.0 - cov).max(0.0);
        let q = ((1.0 - alpha) * (absr.len() as f64 - 1.0)).round() as usize;
        let half = absr[q.min(absr.len() - 1)];
        ctx.finish(FittedNaiveConformal {
            last: y[y.len() - 1],
            half_width: half,
            coverage: cov,
        })
    }
}

/// Linear stack of last-value and drift forecasts (sktime `StackingForecaster`).
///
/// Member count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct StackingForecaster;

/// Fitted last-value / drift stack.
#[derive(Clone, Debug)]
pub struct FittedStackingForecaster {
    /// Last observed level.
    pub last: f64,
    /// Drift slope.
    pub slope: f64,
    /// Intercept of the meta ridge.
    pub intercept: f64,
    /// Weight on the last-value member.
    pub w_naive: f64,
    /// Weight on the drift member.
    pub w_drift: f64,
}

impl FittedStackingForecaster {
    /// Combine member forecasts with the fitted meta weights.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::from_iter((1..=h).map(|k| {
            let naive = self.last;
            let drift = self.last + k as f64 * self.slope;
            self.intercept + self.w_naive * naive + self.w_drift * drift
        })))
    }
}

impl FitSeries for StackingForecaster {
    type Fitted = FittedStackingForecaster;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedStackingForecaster>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        if n < 3 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("StackingForecaster needs n≥3")
                    .build(),
            );
            return ctx.finish(FittedStackingForecaster {
                last: y.as_slice().last().copied().unwrap_or(f64::NAN),
                slope: 0.0,
                intercept: y.as_slice().last().copied().unwrap_or(0.0),
                w_naive: 1.0,
                w_drift: 0.0,
            });
        }
        let last = y[n - 1];
        let slope = (y[n - 1] - y[0]) / (n as f64 - 1.0);
        let m = n - 1;
        let z = Matrix::from_fn(m, 3, |i, j| match j {
            0 => 1.0,
            1 => y[i],
            _ => y[0] + (i as f64 + 1.0) * slope,
        });
        let yy = Vector::from_iter((1..n).map(|t| y[t]));
        let mut scratch = Report::new("stack_fc", "meta");
        let beta = least_squares(&mut scratch, &z, &yy, &ctx.policy)
            .unwrap_or_else(|| Vector::from_slice(&[0.0, 1.0, 0.0]));
        ctx.finish(FittedStackingForecaster {
            last,
            slope,
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            w_naive: if beta.len() > 1 { beta[1] } else { 1.0 },
            w_drift: if beta.len() > 2 { beta[2] } else { 0.0 },
        })
    }
}

/// Softmax of in-sample 1-step SSE over Naive / Drift / SES
/// (sktime `AutoEnsembleForecaster`).
///
/// Member count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct AutoEnsembleForecaster;

/// Fitted softmax ensemble of last-value, drift, and SES.
#[derive(Clone, Debug)]
pub struct FittedAutoEnsembleForecaster {
    /// Last observed level (naive member).
    pub last: f64,
    /// Drift slope.
    pub slope: f64,
    /// Terminal SES level.
    pub ses_level: f64,
    /// Softmax weight on the naive member.
    pub w_naive: f64,
    /// Softmax weight on the drift member.
    pub w_drift: f64,
    /// Softmax weight on SES.
    pub w_ses: f64,
}

impl FittedAutoEnsembleForecaster {
    /// Weighted combination of the three member forecasts.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::from_iter((1..=h).map(|k| {
            let naive = self.last;
            let drift = self.last + k as f64 * self.slope;
            let ses = self.ses_level;
            self.w_naive * naive + self.w_drift * drift + self.w_ses * ses
        })))
    }
}

impl FitSeries for AutoEnsembleForecaster {
    type Fitted = FittedAutoEnsembleForecaster;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedAutoEnsembleForecaster>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        let last = y.as_slice().last().copied().unwrap_or(0.0);
        if n < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("AutoEnsembleForecaster needs n≥2")
                    .build(),
            );
            return ctx.finish(FittedAutoEnsembleForecaster {
                last,
                slope: 0.0,
                ses_level: last,
                w_naive: 1.0,
                w_drift: 0.0,
                w_ses: 0.0,
            });
        }
        let slope = (y[n - 1] - y[0]) / (n as f64 - 1.0);
        let mut sse_naive = 0.0_f64;
        let mut sse_drift = 0.0_f64;
        for t in 1..n {
            let e_n = y[t] - y[t - 1];
            let e_d = y[t] - (y[t - 1] + slope);
            sse_naive += e_n * e_n;
            sse_drift += e_d * e_d;
        }
        let (sse_ses, ses_level) =
            match SimpleExpSmoothing::new(None).fit_series(y, &session.child("ses")) {
                Ok(q) => {
                    let sse: f64 = q.value.resid.as_slice().iter().map(|e| e * e).sum();
                    (sse, q.value.level)
                }
                Err(e) => {
                    if !matches!(
                        e.primary.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::RankZero
                            | IssueCode::R2IsOne
                            | IssueCode::MeaninglessFit
                    ) {
                        ctx.push(e.primary);
                    }
                    (f64::INFINITY, last)
                }
            };
        let nrm = (n - 1) as f64;
        let scores = [-sse_naive / nrm, -sse_drift / nrm, -sse_ses / nrm];
        let mx = scores
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let mut w = [0.0_f64; 3];
        let mut z = 0.0_f64;
        if mx.is_finite() {
            for i in 0..3 {
                w[i] = if scores[i].is_finite() {
                    (scores[i] - mx).exp()
                } else {
                    0.0
                };
                z += w[i];
            }
        }
        if z <= 0.0 {
            w = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
            z = 1.0;
        }
        ctx.finish(FittedAutoEnsembleForecaster {
            last,
            slope,
            ses_level,
            w_naive: w[0] / z,
            w_drift: w[1] / z,
            w_ses: w[2] / z,
        })
    }
}

/// Online softmax ensemble of last-value and drift (sktime `OnlineEnsembleForecaster`).
///
/// Member count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct OnlineEnsembleForecaster {
    last: f64,
    first: f64,
    n: u64,
    sse_naive: f64,
    sse_drift: f64,
    updates: u64,
}

/// Fitted online ensemble state (same type as the updater).
pub type FittedOnlineEnsembleForecaster = OnlineEnsembleForecaster;

impl OnlineEnsembleForecaster {
    /// Empty online ensemble.
    pub fn new() -> Self {
        Self::default()
    }

    fn weights(&self) -> (f64, f64) {
        let nrm = self.n.max(1) as f64;
        let s0 = -self.sse_naive / nrm;
        let s1 = -self.sse_drift / nrm;
        let mx = s0.max(s1);
        let e0 = if s0.is_finite() { (s0 - mx).exp() } else { 0.0 };
        let e1 = if s1.is_finite() { (s1 - mx).exp() } else { 0.0 };
        let z = (e0 + e1).max(1e-18);
        (e0 / z, e1 / z)
    }

    /// Weighted combination of last-value and drift.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let (wn, wd) = self.weights();
        let slope = if self.n > 1 {
            (self.last - self.first) / (self.n as f64 - 1.0)
        } else {
            0.0
        };
        ctx.finish(Vector::from_iter((1..=h).map(|k| {
            wn * self.last + wd * (self.last + k as f64 * slope)
        })))
    }
}

impl FitSeries for OnlineEnsembleForecaster {
    type Fitted = OnlineEnsembleForecaster;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<OnlineEnsembleForecaster>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        if n < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("OnlineEnsembleForecaster needs n≥2")
                    .build(),
            );
        }
        self.first = y.as_slice().first().copied().unwrap_or(0.0);
        self.last = y.as_slice().last().copied().unwrap_or(0.0);
        self.n = n as u64;
        self.sse_naive = 0.0;
        self.sse_drift = 0.0;
        let slope = if n > 1 {
            (self.last - self.first) / (n as f64 - 1.0)
        } else {
            0.0
        };
        for t in 1..n {
            let e_n = y[t] - y[t - 1];
            let e_d = y[t] - (y[t - 1] + slope);
            self.sse_naive += e_n * e_n;
            self.sse_drift += e_d * e_d;
        }
        self.updates += 1;
        ctx.finish(self.clone())
    }
}

impl PartialFit for OnlineEnsembleForecaster {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            let q = IncrementalQuality::new(self.updates, 0, self.n);
            return ctx.finish(IncrementalExplain::from_quality(
                q,
                "nothing",
                "OnlineEnsembleForecaster needs y",
                "invalid",
                "invalid",
            ));
        };
        inspect_univariate(&mut ctx, y);
        let _ = x;
        let before_n = self.n;
        let before_last = self.last;
        for i in 0..y.len() {
            if !y[i].is_finite() {
                continue;
            }
            if self.n == 0 {
                self.first = y[i];
                self.last = y[i];
                self.n = 1;
                continue;
            }
            let slope = (self.last - self.first) / self.n.max(1) as f64;
            let e_n = y[i] - self.last;
            let e_d = y[i] - (self.last + slope);
            self.sse_naive += e_n * e_n;
            self.sse_drift += e_d * e_d;
            self.last = y[i];
            self.n += 1;
        }
        self.updates += 1;
        let (wn, wd) = self.weights();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), y.len(), self.n);
        q.effective_sample_size = self.n as f64;
        q.parameter_delta_norm = Some((self.last - before_last).abs());
        q.information_gain = Some((self.n - before_n) as f64);
        q.still_identified = self.n >= 2;
        q.warmup = self.n < 3;
        q.explanation = format!("OnlineEnsemble w_naive={wn:.4} w_drift={wd:.4}");
        ctx.session.record_incremental(IncrementalExplain::from_quality(
            q.clone(),
            "online ensemble weights",
            "softmax of running 1-step SSE of last-value and drift",
            format!("n={before_n}"),
            format!("n={}", self.n),
        ));
        ctx.finish(IncrementalExplain::from_quality(
            q,
            "online ensemble weights",
            "softmax of running 1-step SSE of last-value and drift",
            format!("n={before_n}"),
            format!("n={}", self.n),
        ))
    }
}

/// Seasonal-naive forecaster (repeat the last period).
#[derive(Clone, Debug)]
pub struct SeasonalNaive {
    /// Seasonal period.
    pub period: usize,
}

impl Default for SeasonalNaive {
    fn default() -> Self {
        Self { period: 12 }
    }
}

impl SeasonalNaive {
    /// Period-`period` seasonal naive.
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

/// Fitted seasonal-naive state.
#[derive(Clone, Debug)]
pub struct FittedSeasonalNaive {
    /// Last complete (or partial) season, length `period`.
    pub last_season: Vector,
    /// Period.
    pub period: usize,
    /// Training length (for the phase).
    pub n: usize,
}

impl FittedSeasonalNaive {
    /// Repeat the last season.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let p = self.period.max(1);
        let out = Vector::from_iter((0..h).map(|k| {
            let idx = (self.n + k) % p;
            if idx < self.last_season.len() {
                self.last_season[idx]
            } else {
                f64::NAN
            }
        }));
        ctx.finish(out)
    }
}

impl FitSeries for SeasonalNaive {
    type Fitted = FittedSeasonalNaive;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSeasonalNaive>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let period = self.period.max(1);
        if y.len() < period {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSeasonalCycles)
                    .severity(Severity::Error)
                    .message(format!("seasonal naive n={} < period={period}", y.len()))
                    .build(),
            );
        }
        let mut season = Vector::zeros(period);
        for s in 0..period {
            if y.len() >= period {
                season[s] = y[y.len() - period + s];
            } else if s < y.len() {
                season[s] = y[s];
            }
        }
        ctx.finish(FittedSeasonalNaive {
            last_season: season,
            period,
            n: y.len(),
        })
    }
}

/// Drift (random walk with linear drift) forecaster.
#[derive(Clone, Debug, Default)]
pub struct Drift;

/// Fitted drift model.
#[derive(Clone, Debug)]
pub struct FittedDrift {
    /// Last value.
    pub last: f64,
    /// `(y_n − y_1) / (n − 1)`.
    pub slope: f64,
}

impl FittedDrift {
    /// `y_n + h · slope`.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::from_iter(
            (1..=h).map(|k| self.last + k as f64 * self.slope),
        ))
    }
}

impl FitSeries for Drift {
    type Fitted = FittedDrift;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedDrift>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("drift forecaster needs n≥2")
                    .build(),
            );
            return ctx.finish(FittedDrift {
                last: y.as_slice().last().copied().unwrap_or(f64::NAN),
                slope: 0.0,
            });
        }
        let slope = (y[y.len() - 1] - y[0]) / (y.len() - 1) as f64;
        ctx.finish(FittedDrift {
            last: y[y.len() - 1],
            slope,
        })
    }
}

/// Theta method (Assimakopoulos & Nikolopoulos): SES plus a linear drift.
#[derive(Clone, Debug, Default)]
pub struct Theta;

/// Fitted Theta (SES + half-slope drift).
#[derive(Clone, Debug)]
pub struct FittedTheta {
    /// SES level.
    pub level: f64,
    /// SES α.
    pub alpha: f64,
    /// Drift = `½ (y_n − y_1)/(n−1)` (Hyndman & Billah equivalence).
    pub drift: f64,
}

impl FittedTheta {
    /// `level + h · drift`.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::from_iter(
            (1..=h).map(|k| self.level + k as f64 * self.drift),
        ))
    }
}

impl FitSeries for Theta {
    type Fitted = FittedTheta;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedTheta>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("Theta method needs n≥2")
                    .build(),
            );
            return ctx.finish(FittedTheta {
                level: y.as_slice().last().copied().unwrap_or(f64::NAN),
                alpha: 1.0,
                drift: 0.0,
            });
        }
        warn_unit_root(&mut ctx, y);
        let (alpha, _b, level, _tr, _f) = esm_fit(y.as_slice(), SmoothingKind::Simple, None, None);
        let drift = 0.5 * (y[y.len() - 1] - y[0]) / (y.len() - 1) as f64;
        ctx.finish(FittedTheta {
            level,
            alpha,
            drift,
        })
    }
}

/// One-dimensional local-level Kalman filter result.
#[derive(Clone, Debug)]
pub struct KalmanLevelFit {
    /// Filtered level `μ_{t|t}`.
    pub level: Vector,
    /// One-step predictions `μ_{t|t−1}`.
    pub predicted: Vector,
    /// Observation-noise variance `r`.
    pub r: f64,
    /// State-noise variance `q`.
    pub q: f64,
}

/// Local-level Kalman filter (`y_t = μ_t + ε`, `μ_t = μ_{t−1} + η`).
pub fn kalman_level(y: &Vector, session: &Session) -> Result<Qualified<KalmanLevelFit>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let n = y.len();
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .message("kalman_level needs n≥2")
                .build(),
        );
    }
    let st = slice_stats(y.as_slice());
    let mut q = 0.0;
    let mut nd = 0.0;
    for t in 1..n {
        if y[t].is_finite() && y[t - 1].is_finite() {
            let d = y[t] - y[t - 1];
            q += d * d;
            nd += 1.0;
        }
    }
    // Method-of-moments: Var(Δy) = q + 2r. Split the mass.
    let vdiff = if nd > 1.0 {
        q / (nd - 1.0)
    } else {
        st.variance
    };
    let r = (0.5 * vdiff).max(1e-12);
    let q = (0.5 * vdiff).max(1e-12);
    let mut level = Vector::zeros(n);
    let mut pred = Vector::zeros(n);
    let mut mu = y
        .as_slice()
        .iter()
        .copied()
        .find(|v| v.is_finite())
        .unwrap_or(0.0);
    let mut p = 1e6;
    for t in 0..n {
        let mu_pred = mu;
        let p_pred = p + q;
        pred[t] = mu_pred;
        if y[t].is_finite() {
            let k = p_pred / (p_pred + r);
            mu = mu_pred + k * (y[t] - mu_pred);
            p = (1.0 - k) * p_pred;
        } else {
            mu = mu_pred;
            p = p_pred;
        }
        level[t] = mu;
    }
    ctx.finish(KalmanLevelFit {
        level,
        predicted: pred,
        r,
        q,
    })
}

/// Disturbance simulation smoother for the local-level model (statsmodels `simulation_smoother`).
///
/// Draw count is not identification `p`.
pub fn simulation_smoother(
    y: &Vector,
    seed: u64,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let kf = kalman_level(y, session)?;
    let mut rng = Rng::new(seed | 3);
    let sd = kf.value.q.max(1e-12).sqrt();
    let sim = Vector::from_iter((0..kf.value.level.len()).map(|t| {
        kf.value.level[t] + sd * rng.standard_normal()
    }));
    ctx.finish(sim)
}

/// Kalman one-step news / prediction error (statsmodels `news`).
pub fn statespace_news(y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let kf = kalman_level(y, session)?;
    let news = Vector::from_iter((0..y.len()).map(|t| y[t] - kf.value.predicted[t]));
    ctx.finish(news)
}

/// Hodrick–Prescott filter: SPD solve `(I + λ D'D) τ = y`.
///
/// Returns `(trend, cycle)` with `cycle = y − trend`.
pub fn hp_filter(y: &Vector, lamb: f64, session: &Session) -> Result<Qualified<(Vector, Vector)>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    if !lamb.is_finite() || lamb < 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("HP λ={lamb} is not a finite non-negative number"))
                .build(),
        );
    }
    let n = y.len();
    if n == 0 {
        return ctx.finish((Vector::zeros(0), Vector::zeros(0)));
    }
    let mut a = Matrix::zeros(n, n);
    for i in 0..n {
        a.set(i, i, 1.0);
    }
    let kmax = n.saturating_sub(2);
    for k in 0..kmax {
        let cols = [k, k + 1, k + 2];
        let coef = [1.0, -2.0, 1.0];
        for u in 0..3 {
            for v in 0..3 {
                let prev = a.get(cols[u], cols[v]);
                a.set(cols[u], cols[v], prev + lamb.max(0.0) * coef[u] * coef[v]);
            }
        }
    }
    let trend = match chol_solve(&mut ctx.report, a.inner(), y, &ctx.policy) {
        Some(t) => t,
        None => {
            for i in 0..n {
                a.set(i, i, a.get(i, i) + ctx.policy.rank_tol_relative.max(1e-12));
            }
            ctx.push(
                Issue::builder(IssueCode::JitterInjected)
                    .message("HP Cholesky failed; diagonal jitter added")
                    .compromise(NumericalCompromise::new(
                        "Cholesky(I + λ D'D) τ = y",
                        "jittered SPD solve",
                        "the HP system was not SPD at working precision",
                        "the trend is a slightly different smoother",
                    ))
                    .build(),
            );
            match chol_solve(&mut ctx.report, a.inner(), y, &ctx.policy) {
                Some(t) => t,
                None => {
                    ctx.push(
                        Issue::builder(IssueCode::CholeskyFailed)
                            .message("HP filter could not solve the SPD system")
                            .build(),
                    );
                    y.clone()
                }
            }
        }
    };
    let cycle = y.sub(&trend);
    ctx.finish((trend, cycle))
}

/// GARCH(1,1) specification (QMLE on a demeaned series).
#[derive(Clone, Debug)]
pub struct Garch11 {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for Garch11 {
    fn default() -> Self {
        Self { max_iter: 40 }
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
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("GARCH(1,1) QMLE needs a longer series")
                    .metric("n", y.len() as f64)
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let var = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = 0.05 * var.max(1e-8);
        let mut alpha = 0.05;
        let mut beta = 0.80;
        let mut best = garch_nll(e.as_slice(), omega, alpha, beta);
        let mut step = 0.05;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, alpha, beta].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, alpha, beta];
                    cand[i] = (cur + dir).max(1e-8);
                    if cand[1] + cand[2] >= 0.999 {
                        continue;
                    }
                    let nll = garch_nll(e.as_slice(), cand[0], cand[1], cand[2]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        alpha = cand[1];
                        beta = cand[2];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("GARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        if alpha + beta >= 0.999 {
            ctx.push(
                Issue::builder(IssueCode::NonStationary)
                    .message(format!(
                        "GARCH α+β={:.4} ≥ 1; persistence is a unit root",
                        alpha + beta
                    ))
                    .metric("alpha_plus_beta", alpha + beta)
                    .build(),
            );
        }
        let sigma2 = garch_sigma2(e.as_slice(), omega, alpha, beta);
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::LossIsNan)
                    .message("GARCH QMLE likelihood is non-finite")
                    .build(),
            );
        }
        ctx.finish(FittedGarch11 {
            omega,
            alpha,
            beta,
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

/// EGARCH(1,1) on log-variance (statsmodels / arch `EGARCH`).
///
/// \(\log h_t = \omega + \alpha(|z_{t-1}|-\sqrt{2/π}) + \gamma z_{t-1} + \beta\log h_{t-1}\).
#[derive(Clone, Debug)]
pub struct Egarch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for Egarch {
    fn default() -> Self {
        Self { max_iter: 30 }
    }
}

impl Egarch {
    /// Default EGARCH settings.
    pub fn new() -> Self {
        Self::default()
    }
}

fn egarch_sigma2(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64) -> Vec<f64> {
    let n = e.len();
    let mut s2 = vec![0.0f64; n];
    let mut logh: f64 = omega;
    let c = (2.0 / std::f64::consts::PI).sqrt();
    for t in 0..n {
        if t > 0 {
            let prev = s2[t - 1].max(1e-12).sqrt();
            let z = e[t - 1] / prev;
            logh = omega + alpha * (z.abs() - c) + gamma * z + beta * logh;
        }
        s2[t] = logh.exp().max(1e-12);
    }
    s2
}

fn egarch_nll(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64) -> f64 {
    if !omega.is_finite() || !beta.is_finite() {
        return f64::INFINITY;
    }
    let s2 = egarch_sigma2(e, omega, alpha, gamma, beta);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = s2[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for Egarch {
    type Fitted = FittedGarch11;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedGarch11>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("EGARCH QMLE needs a longer series")
                    .metric("n", y.len() as f64)
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let var = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = var.max(1e-8).ln();
        let mut alpha = 0.1;
        let mut gamma = 0.0;
        let mut beta = 0.8;
        let mut best = egarch_nll(e.as_slice(), omega, alpha, gamma, beta);
        let mut step = 0.05;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, alpha, gamma, beta].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, alpha, gamma, beta];
                    cand[i] = cur + dir;
                    if cand[3].abs() >= 0.999 {
                        continue;
                    }
                    let nll = egarch_nll(e.as_slice(), cand[0], cand[1], cand[2], cand[3]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        alpha = cand[1];
                        gamma = cand[2];
                        beta = cand[3];
                        improved = true;
                    }
                }
            }
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("EGARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        if beta.abs() >= 0.999 {
            ctx.push(
                Issue::builder(IssueCode::NonStationary)
                    .message(format!("EGARCH |β|={beta:.4} ≥ 1"))
                    .build(),
            );
        }
        let sigma2 = egarch_sigma2(e.as_slice(), omega, alpha, gamma, beta);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("EGARCH stores (ω, α, β); the leverage γ is absorbed into α")
                .build(),
        );
        ctx.finish(FittedGarch11 {
            omega,
            alpha,
            beta,
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

/// Fractionally integrated EGARCH (arch `FIEGARCH`) lite.
///
/// News is fractionally weighted with \(d\in(0,1/2)\) before the EGARCH log
/// recursion. The fractional order is not identification `p`.
#[derive(Clone, Debug)]
pub struct Fiegarch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for Fiegarch {
    fn default() -> Self {
        Self { max_iter: 24 }
    }
}

impl Fiegarch {
    /// Default FIEGARCH settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted FIEGARCH variances.
#[derive(Clone, Debug)]
pub struct FittedFiegarch {
    /// ω.
    pub omega: f64,
    /// Magnitude news.
    pub alpha: f64,
    /// Sign news.
    pub gamma: f64,
    /// Log-variance persistence.
    pub beta: f64,
    /// Fractional integration order.
    pub d: f64,
    /// Conditional variances.
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

fn frac_weights(d: f64, n: usize) -> Vec<f64> {
    let mut w = vec![1.0; n.max(1)];
    for k in 1..n {
        w[k] = w[k - 1] * (k as f64 - 1.0 - d) / k as f64;
    }
    w
}

fn fiegarch_sigma2(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64, d: f64) -> Vec<f64> {
    let n = e.len();
    let w = frac_weights(d.clamp(0.01, 0.49), n);
    let c = (2.0 / std::f64::consts::PI).sqrt();
    let mut news = vec![0.0_f64; n];
    let mut s2 = vec![0.0_f64; n];
    let mut logh = omega;
    for t in 0..n {
        if t > 0 {
            let prev = s2[t - 1].max(1e-12).sqrt();
            let z = e[t - 1] / prev;
            news[t] = alpha * (z.abs() - c) + gamma * z;
            let mut frac = 0.0;
            for k in 0..=t {
                frac += w[k] * news[t - k];
            }
            logh = omega + frac + beta * logh;
        }
        s2[t] = logh.exp().max(1e-12);
    }
    s2
}

fn fiegarch_nll(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64, d: f64) -> f64 {
    if !omega.is_finite() || beta.abs() >= 0.999 || !(0.0..0.5).contains(&d) {
        return f64::INFINITY;
    }
    let s2 = fiegarch_sigma2(e, omega, alpha, gamma, beta, d);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = s2[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for Fiegarch {
    type Fitted = FittedFiegarch;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedFiegarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("FIEGARCH QMLE needs a longer series")
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let var = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = var.max(1e-8).ln();
        let mut alpha = 0.1;
        let mut gamma = 0.0;
        let mut beta = 0.7;
        let mut d = 0.2;
        let mut best = fiegarch_nll(e.as_slice(), omega, alpha, gamma, beta, d);
        let mut step = 0.04;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, alpha, gamma, beta, d].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, alpha, gamma, beta, d];
                    cand[i] = if i == 4 {
                        (cur + dir).clamp(0.01, 0.49)
                    } else {
                        cur + dir
                    };
                    let nll = fiegarch_nll(e.as_slice(), cand[0], cand[1], cand[2], cand[3], cand[4]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        alpha = cand[1];
                        gamma = cand[2];
                        beta = cand[3];
                        d = cand[4];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("FIEGARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("FIEGARCH QMLE likelihood is non-finite")
                    .build(),
            );
        }
        let sigma2 = fiegarch_sigma2(e.as_slice(), omega, alpha, gamma, beta, d);
        ctx.finish(FittedFiegarch {
            omega,
            alpha,
            gamma,
            beta,
            d,
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

/// Historical value-at-risk at level `q` (arch `ValueAtRisk`).
///
/// The quantile level is not identification `p`.
pub fn value_at_risk(y: &Vector, q: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let qq = if q.is_finite() && q > 0.0 && q < 1.0 {
        q
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("VaR q={q} is not in (0,1); using 0.05"))
                .build(),
        );
        0.05
    };
    let mut v: Vec<f64> = y.as_slice().iter().copied().filter(|x| x.is_finite()).collect();
    if v.len() < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("historical VaR needs n≥4")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() as f64 - 1.0) * qq).floor() as usize;
    ctx.finish(*v.get(idx).unwrap_or(&v[0]))
}

/// Historical expected shortfall (arch `expected_shortfall`).
///
/// The tail level is not identification `p`.
pub fn expected_shortfall(y: &Vector, q: f64, session: &Session) -> Result<Qualified<f64>> {
    let var = value_at_risk(y, q, session)?;
    let mut ctx = FitCtx::with_session(session.child("es"));
    if !var.value.is_finite() {
        return ctx.finish(f64::NAN);
    }
    let mut s = 0.0;
    let mut c = 0.0;
    for &v in y.as_slice() {
        if v.is_finite() && v <= var.value {
            s += v;
            c += 1.0;
        }
    }
    ctx.finish(if c > 0.0 { s / c } else { var.value })
}

/// GJR-GARCH(1,1) (Glosten–Jagannathan–Runkle / arch `GJR`).
///
/// \(h_t=\omega+\alpha\varepsilon_{t-1}^2+\gamma\varepsilon_{t-1}^2 1_{\varepsilon<0}+\beta h_{t-1}\).
#[derive(Clone, Debug)]
pub struct GjrGarch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for GjrGarch {
    fn default() -> Self {
        Self { max_iter: 30 }
    }
}

impl GjrGarch {
    /// Default GJR-GARCH settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted GJR-GARCH(1,1) with an explicit leverage coefficient.
#[derive(Clone, Debug)]
pub struct FittedGjrGarch {
    /// ω.
    pub omega: f64,
    /// Symmetric ARCH coefficient.
    pub alpha: f64,
    /// Leverage coefficient on negative shocks.
    pub gamma: f64,
    /// GARCH coefficient.
    pub beta: f64,
    /// In-sample conditional variances.
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

impl FittedGjrGarch {
    /// Iterate the GJR recursion `h` steps using \(\mathbb{E}[1_{\varepsilon<0}]=1/2\).
    pub fn forecast_variance(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let last = self.sigma2.as_slice().last().copied().unwrap_or(self.omega);
        let last_e = self.resid.as_slice().last().copied().unwrap_or(0.0);
        let last_e2 = last_e * last_e;
        let lev = if last_e < 0.0 { 1.0 } else { 0.0 };
        let mut s = self.omega + self.alpha * last_e2 + self.gamma * last_e2 * lev + self.beta * last;
        let persist = self.alpha + 0.5 * self.gamma + self.beta;
        let mut out = Vector::zeros(h);
        for i in 0..h {
            out[i] = s.max(1e-12);
            s = self.omega + persist * s;
        }
        ctx.finish(out)
    }
}

fn gjr_sigma2(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64) -> Vec<f64> {
    let var0 = e.iter().map(|v| v * v).sum::<f64>() / e.len().max(1) as f64;
    let mut s2 = vec![var0.max(omega).max(1e-12); e.len()];
    for t in 1..e.len() {
        let e2 = e[t - 1] * e[t - 1];
        let lev = if e[t - 1] < 0.0 { 1.0 } else { 0.0 };
        s2[t] = omega + alpha * e2 + gamma * e2 * lev + beta * s2[t - 1];
        if !s2[t].is_finite() || s2[t] <= 0.0 {
            s2[t] = omega.max(1e-12);
        }
    }
    s2
}

fn gjr_nll(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64) -> f64 {
    if omega <= 0.0 || alpha < 0.0 || beta < 0.0 || alpha + gamma.min(0.0) < 0.0 {
        return f64::INFINITY;
    }
    let s2 = gjr_sigma2(e, omega, alpha, gamma, beta);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = s2[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for GjrGarch {
    type Fitted = FittedGjrGarch;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedGjrGarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("GJR-GARCH QMLE needs a longer series")
                    .metric("n", y.len() as f64)
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let var = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = 0.05 * var.max(1e-8);
        let mut alpha = 0.05;
        let mut gamma = 0.05;
        let mut beta = 0.80;
        let mut best = gjr_nll(e.as_slice(), omega, alpha, gamma, beta);
        let mut step = 0.05;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, alpha, gamma, beta].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, alpha, gamma, beta];
                    cand[i] = if i == 2 { cur + dir } else { (cur + dir).max(1e-8) };
                    if cand[1] + 0.5 * cand[2] + cand[3] >= 0.999 {
                        continue;
                    }
                    let nll = gjr_nll(e.as_slice(), cand[0], cand[1], cand[2], cand[3]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        alpha = cand[1];
                        gamma = cand[2];
                        beta = cand[3];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("GJR-GARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        let persist = alpha + 0.5 * gamma + beta;
        if persist >= 0.999 {
            ctx.push(
                Issue::builder(IssueCode::NonStationary)
                    .message(format!(
                        "GJR α+γ/2+β={persist:.4} ≥ 1; persistence is a unit root"
                    ))
                    .metric("persistence", persist)
                    .build(),
            );
        }
        let sigma2 = gjr_sigma2(e.as_slice(), omega, alpha, gamma, beta);
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::LossIsNan)
                    .message("GJR-GARCH QMLE likelihood is non-finite")
                    .build(),
            );
        }
        ctx.finish(FittedGjrGarch {
            omega,
            alpha,
            gamma,
            beta,
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

/// FIGARCH(1,d,1) lite (arch `FIGARCH`): truncated ARCH(∞) weights from \((1-L)^d\).
///
/// Truncation length is not identification `p`.
#[derive(Clone, Debug)]
pub struct Figarch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
    /// ARCH(∞) truncation (not identification `p`).
    pub trunc: usize,
}

impl Default for Figarch {
    fn default() -> Self {
        Self {
            max_iter: 24,
            trunc: 16,
        }
    }
}

impl Figarch {
    /// Default FIGARCH settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted FIGARCH fractional-integration state.
#[derive(Clone, Debug)]
pub struct FittedFigarch {
    /// ω.
    pub omega: f64,
    /// Fractional differencing `d ∈ (0, 1)`.
    pub d: f64,
    /// GARCH coefficient.
    pub beta: f64,
    /// In-sample conditional variances.
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

fn figarch_weights(d: f64, trunc: usize) -> Vec<f64> {
    let mut pi = vec![0.0; trunc];
    if trunc == 0 {
        return pi;
    }
    let dd = d.clamp(1e-4, 0.999);
    pi[0] = dd;
    for j in 2..=trunc {
        let jf = j as f64;
        pi[j - 1] = pi[j - 2] * ((jf - 1.0 - dd) / jf);
    }
    pi
}

fn figarch_sigma2(e: &[f64], omega: f64, d: f64, beta: f64, trunc: usize) -> Vec<f64> {
    let pi = figarch_weights(d, trunc);
    let var0 = e.iter().map(|v| v * v).sum::<f64>() / e.len().max(1) as f64;
    let mut s2 = vec![var0.max(omega).max(1e-12); e.len()];
    let b = beta.clamp(0.0, 0.999);
    for t in 1..e.len() {
        let mut arch = 0.0;
        let kmax = pi.len().min(t);
        for k in 0..kmax {
            let ek = e[t - 1 - k];
            arch += pi[k] * ek * ek;
        }
        s2[t] = omega + b * s2[t - 1] + (1.0 - b) * arch;
        if !s2[t].is_finite() || s2[t] <= 0.0 {
            s2[t] = omega.max(1e-12);
        }
    }
    s2
}

fn figarch_nll(e: &[f64], omega: f64, d: f64, beta: f64, trunc: usize) -> f64 {
    if omega <= 0.0 || !(0.0..1.0).contains(&d) || !(0.0..1.0).contains(&beta) {
        return f64::INFINITY;
    }
    let s2 = figarch_sigma2(e, omega, d, beta, trunc);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = s2[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for Figarch {
    type Fitted = FittedFigarch;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedFigarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("FIGARCH QMLE needs a longer series")
                    .metric("n", y.len() as f64)
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let var = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let trunc = self.trunc.max(4);
        let mut omega = 0.05 * var.max(1e-8);
        let mut d = 0.4;
        let mut beta = 0.6;
        let mut best = figarch_nll(e.as_slice(), omega, d, beta, trunc);
        let mut step = 0.05;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, d, beta].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, d, beta];
                    cand[i] = if i == 0 {
                        (cur + dir).max(1e-8)
                    } else {
                        (cur + dir).clamp(1e-4, 0.999)
                    };
                    let nll = figarch_nll(e.as_slice(), cand[0], cand[1], cand[2], trunc);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        d = cand[1];
                        beta = cand[2];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("FIGARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("FIGARCH QMLE likelihood is non-finite; last finite candidate kept")
                    .build(),
            );
        }
        let sigma2 = figarch_sigma2(e.as_slice(), omega, d, beta, trunc);
        ctx.finish(FittedFigarch {
            omega,
            d,
            beta,
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

/// APARCH(1,1) (Ding–Granger–Engle / arch `APARCH`).
///
/// \(\sigma_t^\delta=\omega+\alpha(|\varepsilon_{t-1}|-\gamma\varepsilon_{t-1})^\delta+\beta\sigma_{t-1}^\delta\).
#[derive(Clone, Debug)]
pub struct Aparch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for Aparch {
    fn default() -> Self {
        Self { max_iter: 24 }
    }
}

impl Aparch {
    /// Default APARCH settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted APARCH(1,1) power-volatility state.
#[derive(Clone, Debug)]
pub struct FittedAparch {
    /// ω.
    pub omega: f64,
    /// ARCH coefficient.
    pub alpha: f64,
    /// Leverage \(\gamma\).
    pub gamma: f64,
    /// GARCH coefficient.
    pub beta: f64,
    /// Power \(\delta\).
    pub delta: f64,
    /// In-sample conditional variances \(\sigma_t^2\).
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

fn aparch_sigma2(
    e: &[f64],
    omega: f64,
    alpha: f64,
    gamma: f64,
    beta: f64,
    delta: f64,
) -> Vec<f64> {
    let d = delta.clamp(0.25, 4.0);
    let var0 = e.iter().map(|v| v * v).sum::<f64>() / e.len().max(1) as f64;
    let mut s2 = vec![var0.max(omega).max(1e-12); e.len()];
    let mut sp = s2[0].powf(d / 2.0);
    for t in 1..e.len() {
        let shock = (e[t - 1].abs() - gamma * e[t - 1]).max(0.0);
        sp = omega + alpha * shock.powf(d) + beta * sp;
        if !sp.is_finite() || sp <= 0.0 {
            sp = omega.max(1e-12);
        }
        s2[t] = sp.powf(2.0 / d).max(1e-12);
    }
    s2
}

fn aparch_nll(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64, delta: f64) -> f64 {
    if omega <= 0.0 || alpha < 0.0 || beta < 0.0 || !(0.2..=4.0).contains(&delta) || gamma.abs() >= 1.0
    {
        return f64::INFINITY;
    }
    let s2 = aparch_sigma2(e, omega, alpha, gamma, beta, delta);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = s2[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for Aparch {
    type Fitted = FittedAparch;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedAparch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("APARCH QMLE needs a longer series")
                    .metric("n", y.len() as f64)
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let var = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = 0.05 * var.max(1e-8);
        let mut alpha = 0.05;
        let mut gamma = 0.1;
        let mut beta = 0.80;
        let mut delta = 1.5;
        let mut best = aparch_nll(e.as_slice(), omega, alpha, gamma, beta, delta);
        let mut step = 0.05;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, alpha, gamma, beta, delta].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, alpha, gamma, beta, delta];
                    cand[i] = match i {
                        2 => (cur + dir).clamp(-0.99, 0.99),
                        4 => (cur + dir).clamp(0.25, 4.0),
                        _ => (cur + dir).max(1e-8),
                    };
                    if cand[1] + cand[3] >= 0.999 {
                        continue;
                    }
                    let nll = aparch_nll(e.as_slice(), cand[0], cand[1], cand[2], cand[3], cand[4]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        alpha = cand[1];
                        gamma = cand[2];
                        beta = cand[3];
                        delta = cand[4];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("APARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        let sigma2 = aparch_sigma2(e.as_slice(), omega, alpha, gamma, beta, delta);
        ctx.finish(FittedAparch {
            omega,
            alpha,
            gamma,
            beta,
            delta,
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

/// Heterogeneous ARCH (Müller / arch `HARCH`).
///
/// \(h_t=\omega+\alpha_1\varepsilon_{t-1}^2+\alpha_5\bar\varepsilon_{t,5}^2+\alpha_{22}\bar\varepsilon_{t,22}^2\).
/// Window lengths are not identification `p`.
#[derive(Clone, Debug)]
pub struct Harch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for Harch {
    fn default() -> Self {
        Self { max_iter: 24 }
    }
}

impl Harch {
    /// Default HARCH(1, 5, 22).
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted HARCH variances.
#[derive(Clone, Debug)]
pub struct FittedHarch {
    /// ω.
    pub omega: f64,
    /// Daily ARCH weight.
    pub alpha1: f64,
    /// Weekly ARCH weight.
    pub alpha5: f64,
    /// Monthly ARCH weight.
    pub alpha22: f64,
    /// In-sample conditional variances.
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

fn harch_mean_sq(e: &[f64], t: usize, w: usize) -> f64 {
    let lo = t.saturating_sub(w);
    let sl = &e[lo..t];
    if sl.is_empty() {
        return 0.0;
    }
    sl.iter().map(|v| v * v).sum::<f64>() / sl.len() as f64
}

fn harch_sigma2(e: &[f64], omega: f64, a1: f64, a5: f64, a22: f64) -> Vec<f64> {
    let var0 = e.iter().map(|v| v * v).sum::<f64>() / e.len().max(1) as f64;
    let mut s2 = vec![var0.max(omega).max(1e-12); e.len()];
    for t in 1..e.len() {
        s2[t] = omega
            + a1 * e[t - 1] * e[t - 1]
            + a5 * harch_mean_sq(e, t, 5)
            + a22 * harch_mean_sq(e, t, 22);
        if !s2[t].is_finite() || s2[t] <= 0.0 {
            s2[t] = omega.max(1e-12);
        }
    }
    s2
}

fn harch_nll(e: &[f64], omega: f64, a1: f64, a5: f64, a22: f64) -> f64 {
    if omega <= 0.0 || a1 < 0.0 || a5 < 0.0 || a22 < 0.0 {
        return f64::INFINITY;
    }
    let s2 = harch_sigma2(e, omega, a1, a5, a22);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = s2[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for Harch {
    type Fitted = FittedHarch;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedHarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("HARCH QMLE needs a longer series")
                    .metric("n", y.len() as f64)
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let var = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = 0.05 * var.max(1e-8);
        let mut a1 = 0.10;
        let mut a5 = 0.10;
        let mut a22 = 0.10;
        let mut best = harch_nll(e.as_slice(), omega, a1, a5, a22);
        let mut step = 0.05;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, a1, a5, a22].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, a1, a5, a22];
                    cand[i] = (cur + dir).max(1e-8);
                    let nll = harch_nll(e.as_slice(), cand[0], cand[1], cand[2], cand[3]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        a1 = cand[1];
                        a5 = cand[2];
                        a22 = cand[3];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("HARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        let sigma2 = harch_sigma2(e.as_slice(), omega, a1, a5, a22);
        ctx.finish(FittedHarch {
            omega,
            alpha1: a1,
            alpha5: a5,
            alpha22: a22,
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

/// RiskMetrics / EWMA variance (arch `EWMAVariance`).
///
/// \(h_t=\lambda h_{t-1}+(1-\lambda)\varepsilon_{t-1}^2\).
#[derive(Clone, Debug)]
pub struct EwmaVol {
    /// Fixed decay; `None` QMLE-tunes \(\lambda\).
    pub lambda: Option<f64>,
}

impl Default for EwmaVol {
    fn default() -> Self {
        Self { lambda: None }
    }
}

impl EwmaVol {
    /// RiskMetrics \(\lambda=0.94\).
    pub fn riskmetrics() -> Self {
        Self { lambda: Some(0.94) }
    }

    /// QMLE \(\lambda\).
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted EWMA variances.
#[derive(Clone, Debug)]
pub struct FittedEwmaVol {
    /// Decay \(\lambda\).
    pub lambda: f64,
    /// In-sample conditional variances.
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

fn ewma_sigma2(e: &[f64], lam: f64) -> Vec<f64> {
    let var0 = e.iter().map(|v| v * v).sum::<f64>() / e.len().max(1) as f64;
    let mut s2 = vec![var0.max(1e-12); e.len()];
    let l = lam.clamp(1e-4, 0.999);
    for t in 1..e.len() {
        s2[t] = l * s2[t - 1] + (1.0 - l) * e[t - 1] * e[t - 1];
        if !s2[t].is_finite() || s2[t] <= 0.0 {
            s2[t] = var0.max(1e-12);
        }
    }
    s2
}

fn ewma_nll(e: &[f64], lam: f64) -> f64 {
    if !(0.0..1.0).contains(&lam) {
        return f64::INFINITY;
    }
    let s2 = ewma_sigma2(e, lam);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = s2[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for EwmaVol {
    type Fitted = FittedEwmaVol;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedEwmaVol>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let mut lam = self.lambda.unwrap_or(0.94);
        if !(0.0..1.0).contains(&lam) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!("EWMA λ={lam} not in (0,1); using 0.94"))
                    .build(),
            );
            lam = 0.94;
        }
        if self.lambda.is_none() {
            let mut best = ewma_nll(e.as_slice(), lam);
            let mut step = 0.05;
            for _ in 0..20 {
                let mut improved = false;
                for dir in [-step, step] {
                    let cand = (lam + dir).clamp(0.5, 0.999);
                    let nll = ewma_nll(e.as_slice(), cand);
                    if nll < best {
                        best = nll;
                        lam = cand;
                        improved = true;
                    }
                }
                if !improved {
                    step *= 0.5;
                    if step < 1e-4 {
                        break;
                    }
                }
            }
        }
        ctx.finish(FittedEwmaVol {
            lambda: lam,
            sigma2: Vector::from_iter(ewma_sigma2(e.as_slice(), lam)),
            resid: e,
        })
    }
}

/// Croston intermittent-demand smoother.
#[derive(Clone, Debug)]
pub struct Croston {
    /// Smoothing constant for both demand size and inter-arrival.
    pub alpha: f64,
}

impl Default for Croston {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl Croston {
    /// Croston with smoothing `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

/// Fitted Croston state: demand size `z` and interval `p`.
#[derive(Clone, Debug)]
pub struct FittedCroston {
    /// Smoothed demand size.
    pub z: f64,
    /// Smoothed inter-arrival.
    pub p: f64,
    /// Smoothing constant.
    pub alpha: f64,
}

impl FittedCroston {
    /// Constant `z/p` forecast.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let rate = if self.p.abs() > 1e-15 {
            self.z / self.p
        } else {
            f64::NAN
        };
        ctx.finish(Vector::filled(h, rate))
    }
}

impl FitSeries for Croston {
    type Fitted = FittedCroston;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedCroston>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.as_slice().iter().any(|&v| v < 0.0) {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .message("Croston expected non-negative intermittent demand")
                    .build(),
            );
        }
        let a = self.alpha.clamp(1e-6, 1.0);
        let mut z = 0.0;
        let mut p = 1.0;
        let mut q = 0.0;
        let mut init = false;
        for &yt in y.as_slice() {
            q += 1.0;
            if yt > 0.0 {
                if !init {
                    z = yt;
                    p = q;
                    init = true;
                } else {
                    z += a * (yt - z);
                    p += a * (q - p);
                }
                q = 0.0;
            }
        }
        if !init {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("Croston saw no positive demand")
                    .meaninglessness(Meaninglessness::vacuous(
                        "Croston z/p",
                        "every observation is zero; the demand size is unidentified",
                        "this is a zero series, not intermittent demand",
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedCroston { z, p, alpha: a })
    }
}

/// Teunter–Syntetos–Babai Croston variant (sktime `TSB`).
///
/// Demand probability is updated on every step; demand size only on positives.
#[derive(Clone, Debug)]
pub struct TsbCroston {
    /// Demand-size smoothing.
    pub alpha: f64,
    /// Demand-probability smoothing.
    pub beta: f64,
}

impl Default for TsbCroston {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            beta: 0.1,
        }
    }
}

impl TsbCroston {
    /// TSB with demand/probability smoothers `alpha` / `beta`.
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }
}

/// Fitted TSB state.
#[derive(Clone, Debug)]
pub struct FittedTsbCroston {
    /// Smoothed demand size.
    pub z: f64,
    /// Smoothed demand probability.
    pub p: f64,
}

impl FittedTsbCroston {
    /// Constant `z·p` forecast.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::filled(h, self.z * self.p))
    }
}

impl FitSeries for TsbCroston {
    type Fitted = FittedTsbCroston;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedTsbCroston>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.as_slice().iter().any(|&v| v < 0.0) {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(Severity::Warning)
                    .message("TSB expected non-negative intermittent demand")
                    .build(),
            );
        }
        let a = self.alpha.clamp(1e-6, 1.0);
        let b = self.beta.clamp(1e-6, 1.0);
        let mut z = 0.0;
        let mut p = 0.0;
        let mut init = false;
        for &yt in y.as_slice() {
            let occ = if yt > 0.0 { 1.0 } else { 0.0 };
            if !init {
                if occ > 0.0 {
                    z = yt;
                    p = 1.0;
                    init = true;
                }
                continue;
            }
            p += b * (occ - p);
            if occ > 0.0 {
                z += a * (yt - z);
            }
        }
        if !init {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("TSB saw no positive demand")
                    .meaninglessness(Meaninglessness::vacuous(
                        "TSB z·p",
                        "every observation is zero; demand size is unidentified",
                        "this is a zero series, not intermittent demand",
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedTsbCroston { z, p })
    }
}

/// Syntetos–Boylan Croston approximation (sktime `Croston` SBA).
///
/// Forecast is \((z/p)\,(1-\alpha/2)\). Lag / interval counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct SbaCroston {
    /// Smoothing constant for demand size and inter-arrival.
    pub alpha: f64,
}

impl Default for SbaCroston {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl SbaCroston {
    /// SBA Croston with smoothing `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

/// Fitted Syntetos–Boylan state.
#[derive(Clone, Debug)]
pub struct FittedSbaCroston {
    /// Smoothed demand size.
    pub z: f64,
    /// Smoothed inter-arrival.
    pub p: f64,
    /// Smoothing constant.
    pub alpha: f64,
}

impl FittedSbaCroston {
    /// Constant bias-corrected `z/p` forecast.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let rate = if self.p.abs() > 1e-15 {
            (self.z / self.p) * (1.0 - 0.5 * self.alpha)
        } else {
            f64::NAN
        };
        ctx.finish(Vector::filled(h, rate))
    }
}

impl FitSeries for SbaCroston {
    type Fitted = FittedSbaCroston;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedSbaCroston>> {
        let q = Croston { alpha: self.alpha }.fit_series(y, session)?;
        Ok(q.map(|c| FittedSbaCroston {
            z: c.z,
            p: c.p,
            alpha: c.alpha,
        }))
    }
}

/// ARCH(`lags`) QMLE (Engle / arch `ARCH`).
///
/// \(h_t=\omega+\sum_{i=1}^{L}\alpha_i\varepsilon_{t-i}^2\). Lag order is not identification `p`.
#[derive(Clone, Debug)]
pub struct ArchP {
    /// ARCH lag count (not identification `p`).
    pub lags: usize,
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for ArchP {
    fn default() -> Self {
        Self {
            lags: 1,
            max_iter: 32,
        }
    }
}

impl ArchP {
    /// ARCH with `lags` squared-residual terms.
    pub fn new(lags: usize) -> Self {
        Self {
            lags: lags.max(1),
            ..Self::default()
        }
    }
}

/// Fitted ARCH(`lags`) variance recursion.
#[derive(Clone, Debug)]
pub struct FittedArchP {
    /// ω.
    pub omega: f64,
    /// ARCH coefficients \(\alpha_1,\ldots,\alpha_L\).
    pub alphas: Vector,
    /// In-sample conditional variances.
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

fn archp_sigma2(e: &[f64], omega: f64, alphas: &[f64]) -> Vec<f64> {
    let var0 = e.iter().map(|v| v * v).sum::<f64>() / e.len().max(1) as f64;
    let mut s2 = vec![var0.max(omega).max(1e-12); e.len()];
    for t in 1..e.len() {
        let mut s = omega;
        for (k, a) in alphas.iter().enumerate() {
            if t > k {
                let ek = e[t - 1 - k];
                s += *a * ek * ek;
            } else {
                s += *a * var0;
            }
        }
        s2[t] = if s.is_finite() && s > 0.0 {
            s
        } else {
            omega.max(1e-12)
        };
    }
    s2
}

fn archp_nll(e: &[f64], omega: f64, alphas: &[f64]) -> f64 {
    if omega <= 0.0 || alphas.iter().any(|a| *a < 0.0) {
        return f64::INFINITY;
    }
    let s2 = archp_sigma2(e, omega, alphas);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = s2[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FittedArchP {
    /// Iterate the ARCH recursion `h` steps using \(E[\varepsilon^2]=\sigma^2\).
    pub fn forecast_variance(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let lags = self.alphas.len().max(1);
        let mut hist: Vec<f64> = self
            .resid
            .as_slice()
            .iter()
            .rev()
            .take(lags)
            .map(|v| v * v)
            .collect();
        while hist.len() < lags {
            hist.push(self.omega);
        }
        let mut out = Vector::zeros(h);
        for i in 0..h {
            let mut s = self.omega;
            for (k, a) in self.alphas.as_slice().iter().enumerate() {
                s += *a * hist.get(k).copied().unwrap_or(self.omega);
            }
            if !s.is_finite() || s <= 0.0 {
                s = self.omega.max(1e-12);
            }
            out[i] = s;
            hist.insert(0, s);
            if hist.len() > lags {
                hist.pop();
            }
        }
        ctx.finish(out)
    }
}

impl FitSeries for ArchP {
    type Fitted = FittedArchP;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedArchP>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let lags = self.lags.max(1);
        if y.len() < lags + 6 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("ARCH(L) QMLE needs a longer series")
                    .metric("n", y.len() as f64)
                    .metric("lags", lags as f64)
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let var = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = 0.05 * var.max(1e-8);
        let mut alphas = vec![0.10 / lags as f64; lags];
        let mut best = archp_nll(e.as_slice(), omega, &alphas);
        let mut step = 0.05;
        for it in 0..self.max_iter {
            let mut improved = false;
            let mut cand_vals = Vec::with_capacity(1 + lags);
            cand_vals.push(omega);
            cand_vals.extend_from_slice(&alphas);
            for i in 0..cand_vals.len() {
                let cur = cand_vals[i];
                for dir in [-step, step] {
                    let mut trial = cand_vals.clone();
                    trial[i] = (cur + dir).max(1e-8);
                    let a_sum: f64 = trial[1..].iter().sum();
                    if a_sum >= 0.999 {
                        continue;
                    }
                    let nll = archp_nll(e.as_slice(), trial[0], &trial[1..]);
                    if nll < best {
                        best = nll;
                        omega = trial[0];
                        alphas = trial[1..].to_vec();
                        cand_vals = trial;
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("ARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        let a_sum: f64 = alphas.iter().sum();
        if a_sum >= 0.999 {
            ctx.push(
                Issue::builder(IssueCode::NonStationary)
                    .message(format!("ARCH Σα={a_sum:.4} ≥ 1; persistence is a unit root"))
                    .metric("alpha_sum", a_sum)
                    .build(),
            );
        }
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("ARCH QMLE likelihood is non-finite")
                    .build(),
            );
        }
        let sigma2 = archp_sigma2(e.as_slice(), omega, &alphas);
        ctx.finish(FittedArchP {
            omega,
            alphas: Vector::from_iter(alphas),
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

/// Realized variance of a return series (arch `RealizedVariance`).
///
/// \(\mathrm{RV}=\sum_t \varepsilon_t^2\). This is a measurement, not a GARCH recursion.
#[derive(Clone, Debug, Default)]
pub struct RealizedVariance;

/// Fitted realized-variance path.
#[derive(Clone, Debug)]
pub struct FittedRealizedVariance {
    /// Per-period squared demeaned returns.
    pub sigma2: Vector,
    /// \(\sum\varepsilon_t^2\).
    pub rv: f64,
}

impl RealizedVariance {
    /// Empty realized-variance estimator.
    pub fn new() -> Self {
        Self
    }
}

impl FitSeries for RealizedVariance {
    type Fitted = FittedRealizedVariance;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRealizedVariance>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("realized variance on n<2 is a single square")
                    .build(),
            );
        }
        if y.std() <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .severity(Severity::Warning)
                    .message("realized variance of a near-constant series is ~0")
                    .build(),
            );
        }
        let mean = y.mean();
        let sigma2 = Vector::from_iter(y.as_slice().iter().map(|v| {
            let e = v - mean;
            e * e
        }));
        let rv: f64 = sigma2.as_slice().iter().sum();
        ctx.finish(FittedRealizedVariance { sigma2, rv })
    }
}

/// Parkinson high–low range variance (arch `Parkinson`).
///
/// \(\hat\sigma^2_t=(4\ln 2)^{-1}(\ln H_t-\ln L_t)^2\). Column count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct Parkinson;

/// Garman–Klass OHLC variance (arch `GarmanKlass`).
///
/// \(\hat\sigma^2_t=\tfrac12(\ln H/L)^2-(2\ln 2-1)(\ln C/O)^2\).
#[derive(Clone, Debug, Default)]
pub struct GarmanKlass;

fn ln2() -> f64 {
    std::f64::consts::LN_2
}

/// Parkinson estimator on columns `[high, low]`.
pub fn parkinson(hl: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, hl, None, &ctx.policy);
    if hl.ncols() < 2 {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("Parkinson needs columns [high, low]")
                .build(),
        );
        return ctx.finish(Vector::zeros(hl.nrows()));
    }
    let c = 1.0 / (4.0 * ln2());
    let mut out = Vector::zeros(hl.nrows());
    let mut skipped = 0u64;
    for t in 0..hl.nrows() {
        let h = hl.get(t, 0);
        let l = hl.get(t, 1);
        if h > 0.0 && l > 0.0 && h >= l {
            out[t] = c * (h / l).ln().powi(2);
        } else {
            skipped += 1;
            out[t] = f64::NAN;
        }
    }
    if skipped > 0 {
        ctx.push(
            Issue::builder(IssueCode::NonPositiveSeries)
                .severity(Severity::Warning)
                .message(format!("Parkinson skipped {skipped} non-positive or inverted bars"))
                .build(),
        );
    }
    ctx.finish(out)
}

/// Garman–Klass estimator on columns `[open, high, low, close]`.
pub fn garman_klass(ohlc: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, ohlc, None, &ctx.policy);
    if ohlc.ncols() < 4 {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("Garman–Klass needs columns [open, high, low, close]")
                .build(),
        );
        return ctx.finish(Vector::zeros(ohlc.nrows()));
    }
    let k = 2.0 * ln2() - 1.0;
    let mut out = Vector::zeros(ohlc.nrows());
    let mut skipped = 0u64;
    for t in 0..ohlc.nrows() {
        let o = ohlc.get(t, 0);
        let h = ohlc.get(t, 1);
        let l = ohlc.get(t, 2);
        let c = ohlc.get(t, 3);
        if o > 0.0 && c > 0.0 && h > 0.0 && l > 0.0 && h >= l {
            let hl = (h / l).ln().powi(2);
            let co = (c / o).ln().powi(2);
            out[t] = 0.5 * hl - k * co;
        } else {
            skipped += 1;
            out[t] = f64::NAN;
        }
    }
    if skipped > 0 {
        ctx.push(
            Issue::builder(IssueCode::NonPositiveSeries)
                .severity(Severity::Warning)
                .message(format!(
                    "Garman–Klass skipped {skipped} non-positive or inverted bars"
                ))
                .build(),
        );
    }
    ctx.finish(out)
}

impl Parkinson {
    /// Empty Parkinson estimator.
    pub fn new() -> Self {
        Self
    }

    /// Per-bar Parkinson variance.
    pub fn estimate(&self, hl: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        parkinson(hl, session)
    }
}

impl GarmanKlass {
    /// Empty Garman–Klass estimator.
    pub fn new() -> Self {
        Self
    }

    /// Per-bar Garman–Klass variance.
    pub fn estimate(&self, ohlc: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        garman_klass(ohlc, session)
    }
}

/// SES-versus-Theta selector (sktime `ThetaForecaster` / AutoTheta).
///
/// Candidate count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct AutoTheta;

/// Fitted AutoTheta winner.
#[derive(Clone, Debug)]
pub struct FittedAutoTheta {
    /// `"ses"` or `"theta"`.
    pub name: String,
    /// SES level.
    pub level: f64,
    /// SES α.
    pub alpha: f64,
    /// Drift (zero when SES wins).
    pub drift: f64,
}

impl FittedAutoTheta {
    /// SES: flat level. Theta: `level + h · drift`.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::from_iter((1..=h).map(|k| {
            if self.name == "ses" {
                self.level
            } else {
                self.level + k as f64 * self.drift
            }
        })))
    }
}

impl FitSeries for AutoTheta {
    type Fitted = FittedAutoTheta;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedAutoTheta>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("AutoTheta needs n≥2")
                    .build(),
            );
            return ctx.finish(FittedAutoTheta {
                name: "ses".into(),
                level: y.as_slice().last().copied().unwrap_or(f64::NAN),
                alpha: 1.0,
                drift: 0.0,
            });
        }
        warn_unit_root(&mut ctx, y);
        let (alpha, _b, level, _tr, fitted) =
            esm_fit(y.as_slice(), SmoothingKind::Simple, None, None);
        let mut ses_sse = 0.0;
        for t in 0..y.len() {
            let e = y[t] - fitted[t];
            ses_sse += e * e;
        }
        let drift = 0.5 * (y[y.len() - 1] - y[0]) / (y.len() - 1) as f64;
        let mut theta_sse = 0.0;
        for t in 0..y.len() {
            let e = y[t] - (fitted[t] + drift * t as f64);
            theta_sse += e * e;
        }
        let (name, use_drift) = if theta_sse < ses_sse {
            ("theta", drift)
        } else {
            ("ses", 0.0)
        };
        ctx.finish(FittedAutoTheta {
            name: name.into(),
            level,
            alpha,
            drift: use_drift,
        })
    }
}

/// Endogenous series as exogenous lags (sktime `YtoX`).
///
/// Lag count is not identification `p`.
#[derive(Clone, Debug)]
pub struct YtoX {
    /// Number of lags.
    pub lags: usize,
}

impl Default for YtoX {
    fn default() -> Self {
        Self { lags: 1 }
    }
}

impl YtoX {
    /// Embed with `lags` columns.
    pub fn new(lags: usize) -> Self {
        Self { lags: lags.max(1) }
    }

    /// Map `y` to `[y_{t-1}, …, y_{t-L}]` (`n` rows; leading lags are 0).
    pub fn transform(&self, y: &Vector, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_univariate(&mut ctx, y);
        let p = self.lags.max(1);
        let out = Matrix::from_fn(y.len(), p, |t, j| {
            let src = t as isize - (j as isize + 1);
            if src >= 0 {
                y[src as usize]
            } else {
                0.0
            }
        });
        ctx.finish(out)
    }
}

/// Forecast of squared residuals of a SES level (sktime `SquaringResiduals`).
#[derive(Clone, Debug, Default)]
pub struct SquaringResiduals;

/// Fitted squared-residual smoother.
#[derive(Clone, Debug)]
pub struct FittedSquaringResiduals {
    /// Level SES.
    pub level: f64,
    /// Residual-variance SES level.
    pub vol: f64,
}

impl FittedSquaringResiduals {
    /// Flat forecast of residual variance.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::filled(h, self.vol.max(0.0)))
    }
}

impl FitSeries for SquaringResiduals {
    type Fitted = FittedSquaringResiduals;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSquaringResiduals>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let mut ses = SimpleExpSmoothing::new(Some(0.3));
        let level_q = ses.fit_series(y, &session.child("level"))?;
        let r2 = Vector::from_iter(level_q.value.resid.as_slice().iter().map(|e| e * e));
        let mut vol_ses = SimpleExpSmoothing::new(Some(0.3));
        let vol_q = vol_ses.fit_series(&r2, &session.child("vol"))?;
        ctx.finish(FittedSquaringResiduals {
            level: level_q.value.level,
            vol: vol_q.value.level,
        })
    }
}

/// Linear trend on the time index (sktime `TrendForecaster`).
///
/// Distinct from [`Drift`]: this is OLS of `y` on `[1, t]`, not last-plus-slope.
#[derive(Clone, Debug, Default)]
pub struct TrendForecaster;

/// Fitted OLS time trend.
#[derive(Clone, Debug)]
pub struct FittedTrendForecaster {
    /// Intercept.
    pub intercept: f64,
    /// Slope on `t = 0, 1, …`.
    pub slope: f64,
    /// Training length.
    pub n: usize,
}

impl FittedTrendForecaster {
    /// `intercept + slope · (n + h − 1)` for `h = 1..H`.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::from_iter(
            (0..h).map(|s| self.intercept + self.slope * (self.n + s) as f64),
        ))
    }
}

impl FitSeries for TrendForecaster {
    type Fitted = FittedTrendForecaster;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTrendForecaster>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        if n < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("TrendForecaster needs n≥2")
                    .build(),
            );
            return ctx.finish(FittedTrendForecaster {
                intercept: y.as_slice().last().copied().unwrap_or(0.0),
                slope: 0.0,
                n,
            });
        }
        let x = Matrix::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let mut scratch = Report::new("trend", "ols");
        let coef = least_squares(&mut scratch, &x, y, &ctx.policy)
            .unwrap_or_else(|| Vector::from_slice(&[y.mean(), 0.0]));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedTrendForecaster {
            intercept: coef.as_slice().first().copied().unwrap_or(0.0),
            slope: coef.as_slice().get(1).copied().unwrap_or(0.0),
            n,
        })
    }
}

/// Intermittent multi-alpha Croston average (sktime / IMAPA).
///
/// Alpha-grid size is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct Imapa;

/// Fitted IMAPA average rate.
#[derive(Clone, Debug)]
pub struct FittedImapa {
    /// Averaged `z/p` rate.
    pub rate: f64,
}

impl FittedImapa {
    /// Constant averaged Croston rate.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::filled(h, self.rate))
    }
}

impl FitSeries for Imapa {
    type Fitted = FittedImapa;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedImapa>> {
        let mut rates = Vec::new();
        for &a in &[0.1, 0.2, 0.3, 0.4] {
            match Croston::new(a).fit_series(y, &session.child(format!("imapa_{a}"))) {
                Ok(q) => {
                    if q.value.p.abs() > 1e-15 {
                        rates.push(q.value.z / q.value.p);
                    }
                }
                Err(_) => {}
            }
        }
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if rates.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("IMAPA found no finite Croston rate")
                    .meaninglessness(Meaninglessness::vacuous(
                        "IMAPA rate",
                        "every Croston trial was unidentified",
                        "need some positive demand",
                    ))
                    .build(),
            );
            return ctx.finish(FittedImapa { rate: f64::NAN });
        }
        let rate = rates.iter().sum::<f64>() / rates.len() as f64;
        ctx.finish(FittedImapa { rate })
    }
}

/// Rogers–Satchell OHLC variance (arch `RogersSatchell`).
///
/// \(\hat\sigma^2_t=\ln(H/C)\ln(H/O)+\ln(L/C)\ln(L/O)\).
#[derive(Clone, Debug, Default)]
pub struct RogersSatchell;

/// Yang–Zhang OHLC variance (arch `YangZhang`).
///
/// Combines overnight, open-to-close, and Rogers–Satchell. Bar count is not `p`.
#[derive(Clone, Debug, Default)]
pub struct YangZhang;

/// Rogers–Satchell estimator on columns `[open, high, low, close]`.
pub fn rogers_satchell(ohlc: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, ohlc, None, &ctx.policy);
    if ohlc.ncols() < 4 {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("Rogers–Satchell needs columns [open, high, low, close]")
                .build(),
        );
        return ctx.finish(Vector::zeros(ohlc.nrows()));
    }
    let mut out = Vector::zeros(ohlc.nrows());
    let mut skipped = 0u64;
    for t in 0..ohlc.nrows() {
        let o = ohlc.get(t, 0);
        let h = ohlc.get(t, 1);
        let l = ohlc.get(t, 2);
        let c = ohlc.get(t, 3);
        if o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0 && h >= l {
            out[t] = (h / c).ln() * (h / o).ln() + (l / c).ln() * (l / o).ln();
        } else {
            skipped += 1;
            out[t] = f64::NAN;
        }
    }
    if skipped > 0 {
        ctx.push(
            Issue::builder(IssueCode::NonPositiveSeries)
                .severity(Severity::Warning)
                .message(format!(
                    "Rogers–Satchell skipped {skipped} non-positive or inverted bars"
                ))
                .build(),
        );
    }
    ctx.finish(out)
}

/// Yang–Zhang series estimator on columns `[open, high, low, close]`.
pub fn yang_zhang(ohlc: &Matrix, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, ohlc, None, &ctx.policy);
    if ohlc.ncols() < 4 || ohlc.nrows() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Yang–Zhang needs n≥2 OHLC bars")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let rs = match rogers_satchell(ohlc, &session.child("rs")) {
        Ok(q) => q.value,
        Err(_) => Vector::zeros(ohlc.nrows()),
    };
    let n = ohlc.nrows();
    let mut overnight = Vector::zeros(n);
    let mut oc = Vector::zeros(n);
    for t in 0..n {
        let o = ohlc.get(t, 0);
        let c = ohlc.get(t, 3);
        if o > 0.0 && c > 0.0 {
            oc[t] = (c / o).ln();
            if t > 0 {
                let prev_c = ohlc.get(t - 1, 3);
                if prev_c > 0.0 {
                    overnight[t] = (o / prev_c).ln();
                }
            }
        }
    }
    let var_of = |z: &Vector| {
        let sl: Vec<f64> = z
            .as_slice()
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        if sl.len() < 2 {
            return 0.0;
        }
        let m = sl.iter().sum::<f64>() / sl.len() as f64;
        sl.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (sl.len() - 1) as f64
    };
    let vo = var_of(&overnight);
    let vc = var_of(&oc);
    let vrs: f64 = {
        let sl: Vec<f64> = rs
            .as_slice()
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        if sl.is_empty() {
            0.0
        } else {
            sl.iter().sum::<f64>() / sl.len() as f64
        }
    };
    let nf = n as f64;
    let k = 0.34 / (1.34 + (nf + 1.0) / (nf - 1.0).max(1.0));
    ctx.finish(vo + k * vc + (1.0 - k) * vrs)
}

impl RogersSatchell {
    /// Empty Rogers–Satchell estimator.
    pub fn new() -> Self {
        Self
    }

    /// Per-bar Rogers–Satchell variance.
    pub fn estimate(&self, ohlc: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        rogers_satchell(ohlc, session)
    }
}

impl YangZhang {
    /// Empty Yang–Zhang estimator.
    pub fn new() -> Self {
        Self
    }

    /// Series Yang–Zhang variance.
    pub fn estimate(&self, ohlc: &Matrix, session: &Session) -> Result<Qualified<f64>> {
        yang_zhang(ohlc, session)
    }
}

/// Self-exciting threshold AR (statsmodels `SETAR` lite).
///
/// Two AR(1) regimes split by a delay-1 threshold. Regime / lag counts are not `p`.
#[derive(Clone, Debug, Default)]
pub struct Setar;

/// Fitted two-regime SETAR.
#[derive(Clone, Debug)]
pub struct FittedSetar {
    /// Threshold on `y_{t-1}`.
    pub threshold: f64,
    /// Low-regime intercept and AR(1).
    pub low: Vector,
    /// High-regime intercept and AR(1).
    pub high: Vector,
    /// Last observation.
    pub last: f64,
}

impl FittedSetar {
    /// Iterate the threshold recursion.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let mut prev = self.last;
        let mut out = Vector::zeros(h);
        for i in 0..h {
            let coef = if prev <= self.threshold {
                &self.low
            } else {
                &self.high
            };
            let yhat = coef.as_slice().first().copied().unwrap_or(0.0)
                + coef.as_slice().get(1).copied().unwrap_or(0.0) * prev;
            out[i] = yhat;
            prev = yhat;
        }
        ctx.finish(out)
    }
}

impl FitSeries for Setar {
    type Fitted = FittedSetar;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedSetar>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        if n < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("SETAR needs a longer series")
                    .build(),
            );
        }
        let last = y.as_slice().last().copied().unwrap_or(0.0);
        if n < 3 {
            return ctx.finish(FittedSetar {
                threshold: last,
                low: Vector::from_slice(&[last, 0.0]),
                high: Vector::from_slice(&[last, 0.0]),
                last,
            });
        }
        let mut sorted: Vec<f64> = y.as_slice()[..n - 1]
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let qs = [0.3, 0.5, 0.7];
        let mut best_sse = f64::INFINITY;
        let mut best_c = sorted.get(sorted.len() / 2).copied().unwrap_or(last);
        let mut best_lo = Vector::from_slice(&[0.0, 0.0]);
        let mut best_hi = Vector::from_slice(&[0.0, 0.0]);
        for &q in &qs {
            let idx = ((q * (sorted.len().saturating_sub(1)) as f64).round() as usize)
                .min(sorted.len().saturating_sub(1));
            let c = sorted.get(idx).copied().unwrap_or(best_c);
            let mut lo_x = Vec::new();
            let mut lo_y = Vec::new();
            let mut hi_x = Vec::new();
            let mut hi_y = Vec::new();
            for t in 1..n {
                if !y[t].is_finite() || !y[t - 1].is_finite() {
                    continue;
                }
                if y[t - 1] <= c {
                    lo_x.push(y[t - 1]);
                    lo_y.push(y[t]);
                } else {
                    hi_x.push(y[t - 1]);
                    hi_y.push(y[t]);
                }
            }
            if lo_x.len() < 3 || hi_x.len() < 3 {
                continue;
            }
            let fit_reg = |xs: &[f64], ys: &[f64]| -> (Vector, f64) {
                let m = Matrix::from_fn(xs.len(), 2, |i, j| if j == 0 { 1.0 } else { xs[i] });
                let v = Vector::from_slice(ys);
                let mut scratch = Report::new("setar", "ols");
                let coef = least_squares(&mut scratch, &m, &v, &ctx.policy)
                    .unwrap_or_else(|| Vector::from_slice(&[v.mean(), 0.0]));
                let mut sse = 0.0;
                for i in 0..xs.len() {
                    let yhat = coef[0] + coef[1] * xs[i];
                    let e = ys[i] - yhat;
                    sse += e * e;
                }
                (coef, sse)
            };
            let (lo, sl) = fit_reg(&lo_x, &lo_y);
            let (hi, sh) = fit_reg(&hi_x, &hi_y);
            let sse = sl + sh;
            if sse < best_sse {
                best_sse = sse;
                best_c = c;
                best_lo = lo;
                best_hi = hi;
            }
        }
        ctx.finish(FittedSetar {
            threshold: best_c,
            low: best_lo,
            high: best_hi,
            last,
        })
    }
}

/// Nonlinear GARCH (Engle / arch `NGARCH`).
///
/// \(h_t=\omega+\alpha(\varepsilon_{t-1}-\gamma\sqrt{h_{t-1}})^2+\beta h_{t-1}\).
#[derive(Clone, Debug)]
pub struct Ngarch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for Ngarch {
    fn default() -> Self {
        Self { max_iter: 28 }
    }
}

impl Ngarch {
    /// Default NGARCH settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted NGARCH variances.
#[derive(Clone, Debug)]
pub struct FittedNgarch {
    /// ω.
    pub omega: f64,
    /// ARCH coefficient.
    pub alpha: f64,
    /// Asymmetry.
    pub gamma: f64,
    /// GARCH coefficient.
    pub beta: f64,
    /// In-sample conditional variances.
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

fn ngarch_sigma2(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64) -> Vec<f64> {
    let var0 = e.iter().map(|v| v * v).sum::<f64>() / e.len().max(1) as f64;
    let mut s2 = vec![var0.max(omega).max(1e-12); e.len()];
    for t in 1..e.len() {
        let s = s2[t - 1].max(1e-12).sqrt();
        let z = e[t - 1] - gamma * s;
        s2[t] = omega + alpha * z * z + beta * s2[t - 1];
        if !s2[t].is_finite() || s2[t] <= 0.0 {
            s2[t] = omega.max(1e-12);
        }
    }
    s2
}

fn ngarch_nll(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64) -> f64 {
    if omega <= 0.0 || alpha < 0.0 || beta < 0.0 {
        return f64::INFINITY;
    }
    let s2 = ngarch_sigma2(e, omega, alpha, gamma, beta);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = s2[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for Ngarch {
    type Fitted = FittedNgarch;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedNgarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("NGARCH QMLE needs a longer series")
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let var = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = 0.05 * var.max(1e-8);
        let mut alpha = 0.05;
        let mut gamma = 0.1;
        let mut beta = 0.80;
        let mut best = ngarch_nll(e.as_slice(), omega, alpha, gamma, beta);
        let mut step = 0.05;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, alpha, gamma, beta].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, alpha, gamma, beta];
                    cand[i] = if i == 2 {
                        cur + dir
                    } else {
                        (cur + dir).max(1e-8)
                    };
                    if cand[1] + cand[3] >= 0.999 {
                        continue;
                    }
                    let nll = ngarch_nll(e.as_slice(), cand[0], cand[1], cand[2], cand[3]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        alpha = cand[1];
                        gamma = cand[2];
                        beta = cand[3];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("NGARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("NGARCH QMLE likelihood is non-finite")
                    .build(),
            );
        }
        let sigma2 = ngarch_sigma2(e.as_slice(), omega, alpha, gamma, beta);
        ctx.finish(FittedNgarch {
            omega,
            alpha,
            gamma,
            beta,
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

/// Integrated GARCH (Engle–Bollerslev `IGARCH`): \(\alpha+\beta=1\).
///
/// \(h_t=\omega+\alpha\varepsilon_{t-1}^2+(1-\alpha)h_{t-1}\). Unconditional
/// variance is infinite when \(\omega>0\); that is the model, not a unit-root
/// abort.
#[derive(Clone, Debug)]
pub struct Igarch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for Igarch {
    fn default() -> Self {
        Self { max_iter: 28 }
    }
}

impl Igarch {
    /// Default IGARCH settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted IGARCH variances.
#[derive(Clone, Debug)]
pub struct FittedIgarch {
    /// ω.
    pub omega: f64,
    /// ARCH coefficient (\(\beta=1-\alpha\)).
    pub alpha: f64,
    /// GARCH coefficient.
    pub beta: f64,
    /// In-sample conditional variances.
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

impl FitSeries for Igarch {
    type Fitted = FittedIgarch;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedIgarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("IGARCH QMLE needs a longer series")
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let var = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = 0.05 * var.max(1e-8);
        let mut alpha = 0.05;
        let mut best = garch_nll(e.as_slice(), omega, alpha, 1.0 - alpha);
        let mut step = 0.05;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, alpha].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, alpha];
                    cand[i] = if i == 0 {
                        (cur + dir).max(1e-10)
                    } else {
                        (cur + dir).clamp(1e-6, 0.999)
                    };
                    let nll = garch_nll(e.as_slice(), cand[0], cand[1], 1.0 - cand[1]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        alpha = cand[1];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("IGARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        if omega > 0.0 {
            ctx.push(
                Issue::builder(IssueCode::NonStationary)
                    .severity(Severity::Advisory)
                    .message("IGARCH has infinite unconditional variance when ω>0")
                    .metric("omega", omega)
                    .build(),
            );
        }
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("IGARCH QMLE likelihood is non-finite")
                    .build(),
            );
        }
        let beta = 1.0 - alpha;
        let sigma2 = garch_sigma2(e.as_slice(), omega, alpha, beta);
        ctx.finish(FittedIgarch {
            omega,
            alpha,
            beta,
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

/// Component GARCH (Engle–Lee permanent / transitory).
///
/// \(q_t=\omega+\rho q_{t-1}+\phi(\varepsilon_{t-1}^2-h_{t-1})\),
/// \(h_t=q_t+\alpha(\varepsilon_{t-1}^2-q_{t-1})+\beta(h_{t-1}-q_{t-1})\).
#[derive(Clone, Debug)]
pub struct ComponentGarch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for ComponentGarch {
    fn default() -> Self {
        Self { max_iter: 24 }
    }
}

impl ComponentGarch {
    /// Default component-GARCH settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted component-GARCH variances.
#[derive(Clone, Debug)]
pub struct FittedComponentGarch {
    /// Permanent intercept.
    pub omega: f64,
    /// Permanent persistence.
    pub rho: f64,
    /// Permanent shock.
    pub phi: f64,
    /// Transitory ARCH.
    pub alpha: f64,
    /// Transitory GARCH.
    pub beta: f64,
    /// Conditional variance.
    pub sigma2: Vector,
    /// Permanent component \(q_t\).
    pub q: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

fn cgarch_paths(
    e: &[f64],
    omega: f64,
    rho: f64,
    phi: f64,
    alpha: f64,
    beta: f64,
) -> (Vec<f64>, Vec<f64>) {
    let var0 = e.iter().map(|v| v * v).sum::<f64>() / e.len().max(1) as f64;
    let q0 = (omega / (1.0 - rho).max(1e-6)).max(var0).max(1e-12);
    let mut q = vec![q0; e.len()];
    let mut h = vec![q0; e.len()];
    for t in 1..e.len() {
        let e2 = e[t - 1] * e[t - 1];
        q[t] = omega + rho * q[t - 1] + phi * (e2 - h[t - 1]);
        if !q[t].is_finite() || q[t] <= 0.0 {
            q[t] = omega.max(1e-12);
        }
        h[t] = q[t] + alpha * (e2 - q[t - 1]) + beta * (h[t - 1] - q[t - 1]);
        if !h[t].is_finite() || h[t] <= 0.0 {
            h[t] = q[t].max(1e-12);
        }
    }
    (h, q)
}

fn cgarch_nll(e: &[f64], omega: f64, rho: f64, phi: f64, alpha: f64, beta: f64) -> f64 {
    if omega <= 0.0 || !(0.0..1.0).contains(&rho) || alpha < 0.0 || beta < 0.0 || alpha + beta >= 0.999
    {
        return f64::INFINITY;
    }
    let (h, _) = cgarch_paths(e, omega, rho, phi, alpha, beta);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = h[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for ComponentGarch {
    type Fitted = FittedComponentGarch;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedComponentGarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("component GARCH QMLE needs a longer series")
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let var = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = 0.02 * var.max(1e-8);
        let mut rho = 0.95;
        let mut phi = 0.05;
        let mut alpha = 0.05;
        let mut beta = 0.80;
        let mut best = cgarch_nll(e.as_slice(), omega, rho, phi, alpha, beta);
        let mut step = 0.04;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, rho, phi, alpha, beta].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, rho, phi, alpha, beta];
                    cand[i] = match i {
                        0 => (cur + dir).max(1e-10),
                        1 => (cur + dir).clamp(0.01, 0.999),
                        2 => cur + dir,
                        _ => (cur + dir).max(1e-8),
                    };
                    let nll = cgarch_nll(e.as_slice(), cand[0], cand[1], cand[2], cand[3], cand[4]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        rho = cand[1];
                        phi = cand[2];
                        alpha = cand[3];
                        beta = cand[4];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("component GARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("component GARCH QMLE likelihood is non-finite")
                    .build(),
            );
        }
        let (sigma2, q) = cgarch_paths(e.as_slice(), omega, rho, phi, alpha, beta);
        ctx.finish(FittedComponentGarch {
            omega,
            rho,
            phi,
            alpha,
            beta,
            sigma2: Vector::from_iter(sigma2),
            q: Vector::from_iter(q),
            resid: e,
        })
    }
}

/// Quadratic GARCH (Sentana): \(h_t=\omega+\alpha\varepsilon_{t-1}^2+\gamma\varepsilon_{t-1}+\beta h_{t-1}\).
#[derive(Clone, Debug)]
pub struct Qgarch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for Qgarch {
    fn default() -> Self {
        Self { max_iter: 24 }
    }
}

impl Qgarch {
    /// Default QGARCH settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted QGARCH variances.
#[derive(Clone, Debug)]
pub struct FittedQgarch {
    /// ω.
    pub omega: f64,
    /// ARCH coefficient.
    pub alpha: f64,
    /// Linear shock.
    pub gamma: f64,
    /// GARCH coefficient.
    pub beta: f64,
    /// Conditional variances.
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

fn qgarch_sigma2(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64) -> Vec<f64> {
    let var0 = e.iter().map(|v| v * v).sum::<f64>() / e.len().max(1) as f64;
    let mut s2 = vec![var0.max(omega).max(1e-12); e.len()];
    for t in 1..e.len() {
        s2[t] = omega + alpha * e[t - 1] * e[t - 1] + gamma * e[t - 1] + beta * s2[t - 1];
        if !s2[t].is_finite() || s2[t] <= 0.0 {
            s2[t] = omega.max(1e-12);
        }
    }
    s2
}

fn qgarch_nll(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64) -> f64 {
    if omega <= 0.0 || alpha < 0.0 || beta < 0.0 || alpha + beta >= 0.999 {
        return f64::INFINITY;
    }
    let s2 = qgarch_sigma2(e, omega, alpha, gamma, beta);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = s2[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for Qgarch {
    type Fitted = FittedQgarch;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedQgarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("QGARCH QMLE needs a longer series")
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let var = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = 0.05 * var.max(1e-8);
        let mut alpha = 0.05;
        let mut gamma = 0.0;
        let mut beta = 0.80;
        let mut best = qgarch_nll(e.as_slice(), omega, alpha, gamma, beta);
        let mut step = 0.04;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, alpha, gamma, beta].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, alpha, gamma, beta];
                    cand[i] = if i == 2 {
                        cur + dir
                    } else {
                        (cur + dir).max(1e-8)
                    };
                    let nll = qgarch_nll(e.as_slice(), cand[0], cand[1], cand[2], cand[3]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        alpha = cand[1];
                        gamma = cand[2];
                        beta = cand[3];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("QGARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("QGARCH QMLE likelihood is non-finite")
                    .build(),
            );
        }
        let sigma2 = qgarch_sigma2(e.as_slice(), omega, alpha, gamma, beta);
        ctx.finish(FittedQgarch {
            omega,
            alpha,
            gamma,
            beta,
            sigma2: Vector::from_iter(sigma2),
            resid: e,
        })
    }
}

/// Threshold ARCH on the scale (Zakoian `TARCH`).
///
/// \(\sigma_t=\omega+\alpha|\varepsilon_{t-1}|+\gamma|\varepsilon_{t-1}|I_{\varepsilon<0}+\beta\sigma_{t-1}\).
#[derive(Clone, Debug)]
pub struct Tarch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for Tarch {
    fn default() -> Self {
        Self { max_iter: 24 }
    }
}

impl Tarch {
    /// Default TARCH settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted TARCH scales.
#[derive(Clone, Debug)]
pub struct FittedTarch {
    /// ω.
    pub omega: f64,
    /// Symmetric ARCH.
    pub alpha: f64,
    /// Threshold.
    pub gamma: f64,
    /// Persistence of \(\sigma\).
    pub beta: f64,
    /// Conditional variances \(\sigma_t^2\).
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

fn tarch_sigma(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64) -> Vec<f64> {
    let var0 = e.iter().map(|v| v * v).sum::<f64>() / e.len().max(1) as f64;
    let mut s = vec![var0.max(1e-12).sqrt(); e.len()];
    for t in 1..e.len() {
        let ae = e[t - 1].abs();
        let ind = if e[t - 1] < 0.0 { 1.0 } else { 0.0 };
        s[t] = omega + alpha * ae + gamma * ae * ind + beta * s[t - 1];
        if !s[t].is_finite() || s[t] <= 0.0 {
            s[t] = omega.max(1e-8);
        }
    }
    s
}

fn tarch_nll(e: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64) -> f64 {
    if omega <= 0.0 || alpha < 0.0 || beta < 0.0 {
        return f64::INFINITY;
    }
    let s = tarch_sigma(e, omega, alpha, gamma, beta);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = (s[t] * s[t]).max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for Tarch {
    type Fitted = FittedTarch;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedTarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("TARCH QMLE needs a longer series")
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let sd = e
            .as_slice()
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            / y.len().max(1) as f64;
        let mut omega = 0.05 * sd.max(1e-8).sqrt();
        let mut alpha = 0.05;
        let mut gamma = 0.05;
        let mut beta = 0.80;
        let mut best = tarch_nll(e.as_slice(), omega, alpha, gamma, beta);
        let mut step = 0.04;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, alpha, gamma, beta].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, alpha, gamma, beta];
                    cand[i] = (cur + dir).max(1e-8);
                    let nll = tarch_nll(e.as_slice(), cand[0], cand[1], cand[2], cand[3]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        alpha = cand[1];
                        gamma = cand[2];
                        beta = cand[3];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("TARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("TARCH QMLE likelihood is non-finite")
                    .build(),
            );
        }
        let sig = tarch_sigma(e.as_slice(), omega, alpha, gamma, beta);
        ctx.finish(FittedTarch {
            omega,
            alpha,
            gamma,
            beta,
            sigma2: Vector::from_iter(sig.iter().map(|s| s * s)),
            resid: e,
        })
    }
}

/// Absolute-value GARCH (arch `AVGARCH`): \(\sigma_t=\omega+\alpha|\varepsilon_{t-1}|+\beta\sigma_{t-1}\).
#[derive(Clone, Debug)]
pub struct Avgarch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for Avgarch {
    fn default() -> Self {
        Self { max_iter: 24 }
    }
}

impl Avgarch {
    /// Default AVGARCH settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted AVGARCH scales.
#[derive(Clone, Debug)]
pub struct FittedAvgarch {
    /// ω.
    pub omega: f64,
    /// ARCH on \(|\varepsilon|\).
    pub alpha: f64,
    /// Persistence of \(\sigma\).
    pub beta: f64,
    /// Conditional variances \(\sigma_t^2\).
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

fn avgarch_sigma(e: &[f64], omega: f64, alpha: f64, beta: f64) -> Vec<f64> {
    let var0 = e.iter().map(|v| v * v).sum::<f64>() / e.len().max(1) as f64;
    let mut s = vec![var0.max(1e-12).sqrt(); e.len()];
    for t in 1..e.len() {
        s[t] = omega + alpha * e[t - 1].abs() + beta * s[t - 1];
        if !s[t].is_finite() || s[t] <= 0.0 {
            s[t] = omega.max(1e-8);
        }
    }
    s
}

fn avgarch_nll(e: &[f64], omega: f64, alpha: f64, beta: f64) -> f64 {
    if omega <= 0.0 || alpha < 0.0 || beta < 0.0 {
        return f64::INFINITY;
    }
    let s = avgarch_sigma(e, omega, alpha, beta);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = (s[t] * s[t]).max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for Avgarch {
    type Fitted = FittedAvgarch;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedAvgarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("AVGARCH QMLE needs a longer series")
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let sd = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = 0.05 * sd.max(1e-8).sqrt();
        let mut alpha = 0.05;
        let mut beta = 0.80;
        let mut best = avgarch_nll(e.as_slice(), omega, alpha, beta);
        let mut step = 0.04;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, alpha, beta].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, alpha, beta];
                    cand[i] = (cur + dir).max(1e-8);
                    let nll = avgarch_nll(e.as_slice(), cand[0], cand[1], cand[2]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        alpha = cand[1];
                        beta = cand[2];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("AVGARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("AVGARCH QMLE likelihood is non-finite")
                    .build(),
            );
        }
        let sig = avgarch_sigma(e.as_slice(), omega, alpha, beta);
        ctx.finish(FittedAvgarch {
            omega,
            alpha,
            beta,
            sigma2: Vector::from_iter(sig.iter().map(|s| s * s)),
            resid: e,
        })
    }
}

/// Taylor / ZARCH scale model: \(\sigma_t=\omega+\alpha|\varepsilon_{t-1}|\).
#[derive(Clone, Debug)]
pub struct Zarch {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for Zarch {
    fn default() -> Self {
        Self { max_iter: 20 }
    }
}

impl Zarch {
    /// Default ZARCH settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted ZARCH scales.
#[derive(Clone, Debug)]
pub struct FittedZarch {
    /// ω.
    pub omega: f64,
    /// ARCH on \(|\varepsilon|\).
    pub alpha: f64,
    /// Conditional variances \(\sigma_t^2\).
    pub sigma2: Vector,
    /// Demeaned residuals.
    pub resid: Vector,
}

fn zarch_nll(e: &[f64], omega: f64, alpha: f64) -> f64 {
    avgarch_nll(e, omega, alpha, 0.0)
}

impl FitSeries for Zarch {
    type Fitted = FittedZarch;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedZarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("ZARCH QMLE needs a longer series")
                    .build(),
            );
        }
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let sd = e.as_slice().iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64;
        let mut omega = 0.10 * sd.max(1e-8).sqrt();
        let mut alpha = 0.20;
        let mut best = zarch_nll(e.as_slice(), omega, alpha);
        let mut step = 0.05;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [omega, alpha].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, alpha];
                    cand[i] = (cur + dir).max(1e-8);
                    let nll = zarch_nll(e.as_slice(), cand[0], cand[1]);
                    if nll < best {
                        best = nll;
                        omega = cand[0];
                        alpha = cand[1];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("ZARCH coordinate search", it as u64);
                    break;
                }
            }
        }
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("ZARCH QMLE likelihood is non-finite")
                    .build(),
            );
        }
        let sig = avgarch_sigma(e.as_slice(), omega, alpha, 0.0);
        ctx.finish(FittedZarch {
            omega,
            alpha,
            sigma2: Vector::from_iter(sig.iter().map(|s| s * s)),
            resid: e,
        })
    }
}

/// DCC-GARCH lite on a multivariate residual matrix (Engle).
///
/// Series count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct DccGarch;

/// Fitted DCC correlations and marginal variances.
#[derive(Clone, Debug)]
pub struct FittedDccGarch {
    /// Per-series GARCH(1,1) variances (`T` × `k`).
    pub sigma2: Matrix,
    /// Terminal correlation matrix (`k` × `k`).
    pub corr: Matrix,
    /// DCC `a`.
    pub a: f64,
    /// DCC `b`.
    pub b: f64,
}

impl DccGarch {
    /// Empty DCC estimator.
    pub fn new() -> Self {
        Self
    }

    /// Fit marginal GARCH(1,1) then a scalar DCC on standardized residuals.
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedDccGarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let (n, k) = y.shape();
        if k < 2 {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("DCC needs at least two series")
                    .build(),
            );
        }
        let mut sigma2 = Matrix::zeros(n, k);
        let mut z = Matrix::zeros(n, k);
        for j in 0..k {
            let col = y.column(j);
            let mean = col.mean();
            let e: Vec<f64> = col.as_slice().iter().map(|v| v - mean).collect();
            let var = e.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64;
            let s2 = garch_sigma2(&e, 0.05 * var.max(1e-8), 0.05, 0.80);
            for t in 0..n {
                let v = s2.get(t).copied().unwrap_or(var).max(1e-12);
                sigma2.set(t, j, v);
                z.set(t, j, e.get(t).copied().unwrap_or(0.0) / v.sqrt());
            }
        }
        let mut qbar = Matrix::zeros(k, k);
        if n > 0 {
            for a in 0..k {
                for b in 0..k {
                    let mut s = 0.0;
                    for t in 0..n {
                        s += z.get(t, a) * z.get(t, b);
                    }
                    qbar.set(a, b, s / n as f64);
                }
            }
        }
        let mut a = 0.05_f64;
        let mut b = 0.90_f64;
        let mut q = qbar.clone();
        for t in 1..n {
            let mut nxt = Matrix::zeros(k, k);
            for i in 0..k {
                for j in 0..k {
                    nxt.set(
                        i,
                        j,
                        (1.0 - a - b) * qbar.get(i, j)
                            + a * z.get(t - 1, i) * z.get(t - 1, j)
                            + b * q.get(i, j),
                    );
                }
            }
            q = nxt;
        }
        let mut corr = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                let den = (q.get(i, i).max(1e-12) * q.get(j, j).max(1e-12)).sqrt();
                corr.set(i, j, q.get(i, j) / den);
            }
        }
        ctx.finish(FittedDccGarch { sigma2, corr, a, b })
    }
}

/// Per-series naive forecasts plus a mean top level (sktime `HierarchyEnsembleForecaster`).
///
/// Series count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct HierarchyEnsembleForecaster;

/// Fitted hierarchy ensemble.
#[derive(Clone, Debug)]
pub struct FittedHierarchyEnsemble {
    /// Last value of each series.
    pub last: Vector,
}

impl FittedHierarchyEnsemble {
    /// Each column repeats its last value; an extra column is the mean.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let k = self.last.len();
        let out = Matrix::from_fn(h, k + 1, |_, j| {
            if j < k {
                self.last[j]
            } else if k == 0 {
                0.0
            } else {
                self.last.mean()
            }
        });
        ctx.finish(out)
    }
}

impl HierarchyEnsembleForecaster {
    /// Empty hierarchy ensemble.
    pub fn new() -> Self {
        Self
    }

    /// Fit last-value walkers on each column.
    pub fn fit(
        &self,
        y: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHierarchyEnsemble>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let last = Vector::from_iter((0..y.ncols()).map(|j| {
            y.column(j)
                .as_slice()
                .last()
                .copied()
                .unwrap_or(0.0)
        }));
        ctx.finish(FittedHierarchyEnsemble { last })
    }
}

/// GARCH-in-mean (arch `ARCH-in-mean`): \(y_t=\mu+\lambda\sqrt{h_t}+\varepsilon_t\).
///
/// The mean-volatility slope is not identification `p`.
#[derive(Clone, Debug)]
pub struct ArchInMean {
    /// Coordinate-search iterations.
    pub max_iter: usize,
}

impl Default for ArchInMean {
    fn default() -> Self {
        Self { max_iter: 28 }
    }
}

impl ArchInMean {
    /// Default GARCH-in-mean settings.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted GARCH-in-mean state.
#[derive(Clone, Debug)]
pub struct FittedArchInMean {
    /// Mean intercept.
    pub mu: f64,
    /// Risk premium \(\lambda\).
    pub lambda: f64,
    /// ω.
    pub omega: f64,
    /// ARCH coefficient.
    pub alpha: f64,
    /// GARCH coefficient.
    pub beta: f64,
    /// In-sample conditional variances.
    pub sigma2: Vector,
}

fn aim_path(y: &[f64], mu: f64, lam: f64, omega: f64, alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
    let var0 = y.iter().map(|v| {
        let e = v - mu;
        e * e
    }).sum::<f64>()
        / y.len().max(1) as f64;
    let mut h = vec![var0.max(omega).max(1e-12); y.len()];
    let mut e = vec![0.0; y.len()];
    for t in 0..y.len() {
        if t > 0 {
            h[t] = omega + alpha * e[t - 1] * e[t - 1] + beta * h[t - 1];
            if !h[t].is_finite() || h[t] <= 0.0 {
                h[t] = omega.max(1e-12);
            }
        }
        e[t] = y[t] - mu - lam * h[t].sqrt();
    }
    (h, e)
}

fn aim_nll(y: &[f64], mu: f64, lam: f64, omega: f64, alpha: f64, beta: f64) -> f64 {
    if omega <= 0.0 || alpha < 0.0 || beta < 0.0 {
        return f64::INFINITY;
    }
    let (h, e) = aim_path(y, mu, lam, omega, alpha, beta);
    let mut nll = 0.0;
    for t in 0..y.len() {
        let v = h[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

impl FitSeries for ArchInMean {
    type Fitted = FittedArchInMean;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedArchInMean>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.len() < 8 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("ARCH-in-mean QMLE needs a longer series")
                    .build(),
            );
        }
        let mut mu = y.mean();
        let mut lam = 0.0;
        let var = y.std() * y.std();
        let mut omega = 0.05 * var.max(1e-8);
        let mut alpha = 0.05;
        let mut beta = 0.80;
        let mut best = aim_nll(y.as_slice(), mu, lam, omega, alpha, beta);
        let mut step = 0.05;
        for it in 0..self.max_iter {
            let mut improved = false;
            for (i, cur) in [mu, lam, omega, alpha, beta].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [mu, lam, omega, alpha, beta];
                    cand[i] = if i >= 2 { (cur + dir).max(1e-8) } else { cur + dir };
                    if cand[3] + cand[4] >= 0.999 {
                        continue;
                    }
                    let nll = aim_nll(y.as_slice(), cand[0], cand[1], cand[2], cand[3], cand[4]);
                    if nll < best {
                        best = nll;
                        mu = cand[0];
                        lam = cand[1];
                        omega = cand[2];
                        alpha = cand[3];
                        beta = cand[4];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    ctx.session.converged("ARCH-in-mean coordinate search", it as u64);
                    break;
                }
            }
        }
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("ARCH-in-mean QMLE likelihood is non-finite")
                    .build(),
            );
        }
        let (h, _) = aim_path(y.as_slice(), mu, lam, omega, alpha, beta);
        ctx.finish(FittedArchInMean {
            mu,
            lambda: lam,
            omega,
            alpha,
            beta,
            sigma2: Vector::from_iter(h),
        })
    }
}

/// Constant conditional correlation GARCH (Bollerslev CCC).
///
/// Series count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct CcGarch;

/// Fitted CCC-GARCH margins and constant correlation.
#[derive(Clone, Debug)]
pub struct FittedCcGarch {
    /// Per-series GARCH(1,1) variances (`T` × `k`).
    pub sigma2: Matrix,
    /// Constant correlation (`k` × `k`).
    pub corr: Matrix,
}

impl CcGarch {
    /// Empty CCC estimator.
    pub fn new() -> Self {
        Self
    }

    /// Fit marginal GARCH(1,1) and the sample correlation of standardized residuals.
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedCcGarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let (n, k) = y.shape();
        if k < 2 {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("CCC-GARCH needs at least two series")
                    .build(),
            );
        }
        let mut sigma2 = Matrix::zeros(n, k);
        let mut z = Matrix::zeros(n, k);
        for j in 0..k {
            let col = y.column(j);
            let mean = col.mean();
            let e: Vec<f64> = col.as_slice().iter().map(|v| v - mean).collect();
            let var = e.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64;
            let s2 = garch_sigma2(&e, 0.05 * var.max(1e-8), 0.05, 0.80);
            for t in 0..n {
                let v = s2.get(t).copied().unwrap_or(var).max(1e-12);
                sigma2.set(t, j, v);
                z.set(t, j, e.get(t).copied().unwrap_or(0.0) / v.sqrt());
            }
        }
        let mut corr = Matrix::zeros(k, k);
        for a in 0..k {
            for b in 0..k {
                let mut s = 0.0;
                for t in 0..n {
                    s += z.get(t, a) * z.get(t, b);
                }
                let den = n.max(1) as f64;
                corr.set(a, b, (s / den).clamp(-1.0, 1.0));
            }
            corr.set(a, a, 1.0);
        }
        ctx.finish(FittedCcGarch { sigma2, corr })
    }
}

/// Orthogonal / GO-GARCH on standardized residuals (van der Weide).
///
/// Series / factor counts are not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct GoGarch;

/// Fitted GO-GARCH mixing and factor variances.
#[derive(Clone, Debug)]
pub struct FittedGoGarch {
    /// Mixing matrix \(A\) (`k` × `r`).
    pub loadings: Matrix,
    /// Factor GARCH variances (`T` × `r`).
    pub factor_var: Matrix,
    /// In-sample residual covariance snapshot.
    pub cov: Matrix,
}

impl GoGarch {
    /// Empty GO-GARCH estimator.
    pub fn new() -> Self {
        Self
    }

    /// Marginal GARCH, SVD mixing, independent factor GARCH.
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedGoGarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let (n, k) = y.shape();
        if k < 2 {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("GO-GARCH needs at least two series")
                    .build(),
            );
        }
        let mut z = Matrix::zeros(n, k);
        for j in 0..k {
            let col = y.column(j);
            let mean = col.mean();
            let e: Vec<f64> = col.as_slice().iter().map(|v| v - mean).collect();
            let var = e.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64;
            let s2 = garch_sigma2(&e, 0.05 * var.max(1e-8), 0.05, 0.80);
            for t in 0..n {
                let v = s2.get(t).copied().unwrap_or(var).max(1e-12);
                z.set(t, j, e.get(t).copied().unwrap_or(0.0) / v.sqrt());
            }
        }
        let mut scratch = Report::new("gogarch", "svd");
        let (loadings, r) = match thin_svd(&mut scratch, &z, &ctx.policy) {
            Some(svd) => {
                let r = svd.singular_values.len().min(k).max(1);
                let a = Matrix::from_fn(k, r, |j, c| svd.v[(j, c)]);
                (a, r)
            }
            None => (Matrix::from_fn(k, k.max(1), |i, j| if i == j { 1.0 } else { 0.0 }), k.max(1)),
        };
        let r = r.min(loadings.ncols()).max(1);
        let mut factor_var = Matrix::zeros(n, r);
        for c in 0..r {
            let f: Vec<f64> = (0..n)
                .map(|t| {
                    let mut s = 0.0;
                    for j in 0..k {
                        s += z.get(t, j) * loadings.get(j, c);
                    }
                    s
                })
                .collect();
            let var = f.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64;
            let s2 = garch_sigma2(&f, 0.05 * var.max(1e-8), 0.05, 0.80);
            for t in 0..n {
                factor_var.set(t, c, s2.get(t).copied().unwrap_or(var).max(1e-12));
            }
        }
        let mut cov = Matrix::zeros(k, k);
        for a in 0..k {
            for b in 0..k {
                let mut s = 0.0;
                for t in 0..n {
                    s += z.get(t, a) * z.get(t, b);
                }
                cov.set(a, b, s / n.max(1) as f64);
            }
        }
        ctx.finish(FittedGoGarch {
            loadings,
            factor_var,
            cov,
        })
    }
}

/// Bartlett realized kernel (arch `RealizedKernel`).
///
/// Bandwidth \(\lfloor\sqrt{n}\rfloor\) is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct RealizedKernel;

/// Fitted realized-kernel value.
#[derive(Clone, Debug)]
pub struct FittedRealizedKernel {
    /// Kernel estimate of integrated variance.
    pub rk: f64,
    /// Bartlett bandwidth.
    pub bandwidth: usize,
}

impl RealizedKernel {
    /// Empty realized-kernel estimator.
    pub fn new() -> Self {
        Self
    }
}

impl FitSeries for RealizedKernel {
    type Fitted = FittedRealizedKernel;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRealizedKernel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        if n < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("realized kernel on n<2 is a single square")
                    .build(),
            );
            return ctx.finish(FittedRealizedKernel {
                rk: y.as_slice().first().copied().unwrap_or(0.0).powi(2),
                bandwidth: 0,
            });
        }
        let mean = y.mean();
        let e: Vec<f64> = y.as_slice().iter().map(|v| v - mean).collect();
        let h = (n as f64).sqrt().floor() as usize;
        let h = h.max(1).min(n.saturating_sub(1));
        let gamma = |lag: usize| {
            let mut s = 0.0;
            let m = n.saturating_sub(lag);
            for t in lag..n {
                s += e[t] * e[t - lag];
            }
            s / m.max(1) as f64
        };
        let mut rk = gamma(0);
        for lag in 1..=h {
            let w = 1.0 - lag as f64 / (h as f64 + 1.0);
            rk += 2.0 * w * gamma(lag);
        }
        if rk < 0.0 {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .severity(Severity::Warning)
                    .message("realized kernel is negative; Bartlett weights can overshoot")
                    .build(),
            );
        }
        ctx.finish(FittedRealizedKernel {
            rk,
            bandwidth: h,
        })
    }
}

/// MIDAS distributed lag (statsmodels `Midas`).
///
/// Lag count is not identification `p`. Exponential Almon weights use one \(\theta\).
#[derive(Clone, Debug)]
pub struct Midas {
    /// Number of high-frequency lags.
    pub lags: usize,
}

impl Default for Midas {
    fn default() -> Self {
        Self { lags: 4 }
    }
}

impl Midas {
    /// MIDAS with `lags` weights.
    pub fn new(lags: usize) -> Self {
        Self { lags: lags.max(1) }
    }

    /// Fit \(y_t=a+b\sum_k w_k(\theta)x_{t-k}\).
    pub fn fit(&self, y: &Vector, x: &Vector, session: &Session) -> Result<Qualified<FittedMidas>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        inspect_univariate(&mut ctx, x);
        let lags = self.lags.max(1);
        let n = y.len().min(x.len());
        if n <= lags + 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("MIDAS needs n > lags + 2")
                    .build(),
            );
        }
        let weights = |theta: f64| {
            let mut w = vec![0.0; lags];
            let mut s = 0.0;
            for k in 0..lags {
                let wk = (theta * (k as f64 + 1.0)).exp();
                w[k] = wk;
                s += wk;
            }
            if s > 0.0 {
                for wk in &mut w {
                    *wk /= s;
                }
            }
            w
        };
        let mut best_th = -0.2;
        let mut best_a = y.mean();
        let mut best_b = 0.0;
        let mut best_sse = f64::INFINITY;
        for step in 0..9 {
            let th = -0.8 + 0.2 * step as f64;
            let w = weights(th);
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for t in lags..n {
                if !y[t].is_finite() {
                    continue;
                }
                let mut s = 0.0;
                for k in 0..lags {
                    s += w[k] * x[t - 1 - k];
                }
                xs.push(s);
                ys.push(y[t]);
            }
            if ys.len() < 3 {
                continue;
            }
            let design = Matrix::from_fn(ys.len(), 2, |i, j| if j == 0 { 1.0 } else { xs[i] });
            let target = Vector::from_iter(ys.iter().copied());
            let mut scratch = Report::new("midas", "ols");
            let coef = match least_squares(&mut scratch, &design, &target, &ctx.policy) {
                Some(c) => c,
                None => continue,
            };
            let mut sse = 0.0;
            for i in 0..target.len() {
                let yhat = coef[0] + coef.as_slice().get(1).copied().unwrap_or(0.0) * design.get(i, 1);
                let e = target[i] - yhat;
                sse += e * e;
            }
            if sse < best_sse {
                best_sse = sse;
                best_th = th;
                best_a = coef[0];
                best_b = coef.as_slice().get(1).copied().unwrap_or(0.0);
            }
        }
        ctx.finish(FittedMidas {
            intercept: best_a,
            slope: best_b,
            theta: best_th,
            lags,
            last_x: Vector::from_iter(
                x.as_slice()
                    .iter()
                    .rev()
                    .take(lags)
                    .copied()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev(),
            ),
        })
    }
}

/// Fitted MIDAS weights.
#[derive(Clone, Debug)]
pub struct FittedMidas {
    /// Intercept.
    pub intercept: f64,
    /// Scale on the weighted high-frequency sum.
    pub slope: f64,
    /// Exponential Almon \(\theta\).
    pub theta: f64,
    /// Lag count.
    pub lags: usize,
    last_x: Vector,
}

impl FittedMidas {
    /// One-step MIDAS using the stored high-frequency tail (then flat).
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let mut s = 0.0;
        let mut den = 0.0;
        for k in 0..self.lags {
            let w = (self.theta * (k as f64 + 1.0)).exp();
            den += w;
            if k < self.last_x.len() {
                s += w * self.last_x[self.last_x.len() - 1 - k];
            }
        }
        let yhat = self.intercept + self.slope * if den > 0.0 { s / den } else { 0.0 };
        ctx.finish(Vector::filled(h, yhat))
    }
}

/// Logistic smooth-transition AR (statsmodels STAR lite).
///
/// \(y_t=(a_0+b_0 y_{t-1})+G(y_{t-1};\gamma,c)\,(a_1+b_1 y_{t-1})\).
/// Regime / lag counts are not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct Star;

/// Fitted logistic STAR.
#[derive(Clone, Debug)]
pub struct FittedStar {
    /// Low-regime intercept and AR(1).
    pub low: Vector,
    /// High-regime intercept and AR(1).
    pub high: Vector,
    /// Transition slope \(\gamma\).
    pub gamma: f64,
    /// Transition location.
    pub location: f64,
    /// Last observation.
    pub last: f64,
}

fn star_g(z: f64, gamma: f64, c: f64) -> f64 {
    let u = (-gamma * (z - c)).clamp(-40.0, 40.0);
    1.0 / (1.0 + u.exp())
}

impl FittedStar {
    /// Iterate the STAR recursion.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let mut prev = self.last;
        let mut out = Vector::zeros(h);
        for i in 0..h {
            let g = star_g(prev, self.gamma, self.location);
            let yhat = self.low[0]
                + self.low.as_slice().get(1).copied().unwrap_or(0.0) * prev
                + g * (self.high[0] + self.high.as_slice().get(1).copied().unwrap_or(0.0) * prev);
            out[i] = yhat;
            prev = yhat;
        }
        ctx.finish(out)
    }
}

impl FitSeries for Star {
    type Fitted = FittedStar;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedStar>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        let last = y.as_slice().last().copied().unwrap_or(0.0);
        if n < 6 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("STAR needs a longer series")
                    .build(),
            );
            return ctx.finish(FittedStar {
                low: Vector::from_slice(&[last, 0.0]),
                high: Vector::from_slice(&[0.0, 0.0]),
                gamma: 1.0,
                location: last,
                last,
            });
        }
        let mut a0 = y.mean();
        let mut b0 = 0.3;
        let mut a1 = 0.0;
        let mut b1 = 0.0;
        let mut gamma = 1.0;
        let mut c = y.mean();
        let nll = |a0: f64, b0: f64, a1: f64, b1: f64, gamma: f64, c: f64| {
            let mut sse = 0.0;
            for t in 1..n {
                let g = star_g(y[t - 1], gamma, c);
                let yhat = a0 + b0 * y[t - 1] + g * (a1 + b1 * y[t - 1]);
                let e = y[t] - yhat;
                sse += e * e;
            }
            sse
        };
        let mut best = nll(a0, b0, a1, b1, gamma, c);
        let mut step = 0.2;
        for it in 0..24 {
            let mut improved = false;
            for (i, cur) in [a0, b0, a1, b1, gamma, c].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [a0, b0, a1, b1, gamma, c];
                    cand[i] = if i == 4 { (cur + dir).max(0.05) } else { cur + dir };
                    let sse = nll(cand[0], cand[1], cand[2], cand[3], cand[4], cand[5]);
                    if sse < best {
                        best = sse;
                        a0 = cand[0];
                        b0 = cand[1];
                        a1 = cand[2];
                        b1 = cand[3];
                        gamma = cand[4];
                        c = cand[5];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-4 {
                    break;
                }
            }
        }
        ctx.finish(FittedStar {
            low: Vector::from_slice(&[a0, b0]),
            high: Vector::from_slice(&[a1, b1]),
            gamma,
            location: c,
            last,
        })
    }
}

/// Last-value hierarchy plus bottom-up reconcile (sktime `ReconcilerForecaster`).
///
/// Node count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct ReconcilerForecaster;

/// Fitted last-value hierarchy.
#[derive(Clone, Debug)]
pub struct FittedReconcilerForecaster {
    /// Last value of each series.
    pub last: Vector,
}

impl FittedReconcilerForecaster {
    /// Repeat last values, then [`reconcile_bottom_up`].
    ///
    /// `last` may be the full node vector or the bottom-level series
    /// (`last.len() == summing.ncols()`); bottoms are expanded through `S`.
    pub fn forecast(
        &self,
        h: usize,
        summing: &Matrix,
        session: &Session,
    ) -> Result<Qualified<Matrix>> {
        let n_nodes = summing.nrows();
        let n_bot = summing.ncols();
        let yhat = if self.last.len() == n_nodes {
            Matrix::from_fn(h, n_nodes, |_, j| self.last[j])
        } else if self.last.len() == n_bot {
            Matrix::from_fn(h, n_nodes, |_, i| {
                let mut s = 0.0;
                for j in 0..n_bot {
                    s += summing.get(i, j) * self.last[j];
                }
                s
            })
        } else {
            Matrix::from_fn(h, n_nodes.max(1), |_, j| {
                self.last.as_slice().get(j).copied().unwrap_or(0.0)
            })
        };
        reconcile_bottom_up(&yhat, summing, session)
    }
}

impl ReconcilerForecaster {
    /// Empty reconciler.
    pub fn new() -> Self {
        Self
    }

    /// Store the last observation of each column.
    pub fn fit(
        &self,
        y: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedReconcilerForecaster>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let last = Vector::from_iter((0..y.ncols()).map(|j| {
            y.column(j)
                .as_slice()
                .last()
                .copied()
                .unwrap_or(0.0)
        }));
        ctx.finish(FittedReconcilerForecaster { last })
    }
}

/// Direct tabular reduction (sktime `DirectTabularRegressionForecaster`).
///
/// Window / horizon counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct DirectTabularForecaster {
    /// Lag window.
    pub window: usize,
    /// Direct horizons.
    pub horizon: usize,
}

impl Default for DirectTabularForecaster {
    fn default() -> Self {
        Self {
            window: 3,
            horizon: 3,
        }
    }
}

impl DirectTabularForecaster {
    /// Direct reducer with lag `window` and `horizon` models.
    pub fn new(window: usize, horizon: usize) -> Self {
        Self {
            window: window.max(1),
            horizon: horizon.max(1),
        }
    }
}

impl FitSeries for DirectTabularForecaster {
    type Fitted = crate::reducer::FittedDirectReducer;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<crate::reducer::FittedDirectReducer>> {
        crate::reducer::DirectReducer {
            window: self.window,
            horizon: self.horizon,
        }
        .fit_series(y, session)
    }
}

/// BATS-lite (sktime `BATS`): Box–Cox / identity + trend + seasonal dummies + AR(1).
///
/// Distinct from [`Tbats`] (trigonometric seasonality). Period / dummy count
/// is not identification `p`.
#[derive(Clone, Debug)]
pub struct Bats {
    /// Seasonal period.
    pub period: usize,
    /// Log / Box–Cox (`λ = 0`) map. Requires a strictly positive series.
    pub use_boxcox: bool,
}

impl Default for Bats {
    fn default() -> Self {
        Self {
            period: 4,
            use_boxcox: false,
        }
    }
}

impl Bats {
    /// BATS with seasonal period `period`.
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(2),
            use_boxcox: false,
        }
    }
}

/// Fitted BATS-lite state.
#[derive(Clone, Debug)]
pub struct FittedBats {
    /// OLS coefficients on `[1, t, seasonal dummies]`.
    pub coef: Vector,
    /// AR(1) residual coefficient.
    pub phi: f64,
    /// Last residual on the transformed scale.
    pub last_resid: f64,
    /// Period.
    pub period: usize,
    /// Log map.
    pub use_boxcox: bool,
    /// Training length.
    pub n: usize,
}

impl FittedBats {
    fn design_row(&self, t: usize) -> Vector {
        let per = self.period.max(2);
        let p = 2 + per.saturating_sub(1);
        let mut v = Vector::zeros(p);
        v[0] = 1.0;
        if p > 1 {
            v[1] = t as f64;
        }
        let s = t % per;
        for k in 1..per {
            let j = 1 + k;
            if j < p {
                v[j] = if s == k { 1.0 } else { 0.0 };
            }
        }
        v
    }

    /// `h`-step forecast on the original scale.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        let mut e = self.last_resid;
        let y = Vector::from_iter((0..h).map(|s| {
            let row = self.design_row(self.n + s);
            let mut mu = 0.0;
            for j in 0..self.coef.len().min(row.len()) {
                mu += self.coef[j] * row[j];
            }
            e *= self.phi;
            let z = mu + e;
            if self.use_boxcox {
                z.exp()
            } else {
                z
            }
        }));
        ctx.finish(y)
    }
}

impl FitSeries for Bats {
    type Fitted = FittedBats;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedBats>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let period = self.period.max(2);
        if y.len() < 2 * period {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSeasonalCycles)
                    .severity(Severity::Warning)
                    .message(format!("BATS n={} < 2s with s={period}", y.len()))
                    .build(),
            );
        }
        let z = if self.use_boxcox {
            reject_nonpositive(&mut ctx, y, "BATS Box–Cox");
            Vector::from_iter(y.as_slice().iter().map(|v| v.max(1e-12).ln()))
        } else {
            y.clone()
        };
        let n = z.len();
        let p = 2 + period.saturating_sub(1);
        // Dummy / trend width is a seasonal template, not a regression `p`.
        let spec = FittedBats {
            coef: Vector::zeros(p),
            phi: 0.0,
            last_resid: 0.0,
            period,
            use_boxcox: self.use_boxcox,
            n,
        };
        let x = Matrix::from_fn(n, p, |t, j| spec.design_row(t)[j]);
        let mut scratch = Report::new("bats", "ols");
        let coef = least_squares(&mut scratch, &x, &z, &ctx.policy).unwrap_or_else(|| {
            let mut c = Vector::zeros(p);
            c[0] = z.mean();
            c
        });
        let fit = x.matvec(&coef);
        let mut resid = Vector::zeros(n);
        let mut num = 0.0;
        let mut den = 0.0;
        for t in 0..n {
            resid[t] = z[t] - fit[t];
            if t > 0 {
                num += resid[t] * resid[t - 1];
                den += resid[t - 1] * resid[t - 1];
            }
        }
        let phi = if den > 1e-12 {
            (num / den).clamp(-0.99, 0.99)
        } else {
            0.0
        };
        ctx.finish(FittedBats {
            coef,
            phi,
            last_resid: resid.as_slice().last().copied().unwrap_or(0.0),
            period,
            use_boxcox: self.use_boxcox,
            n,
        })
    }
}

/// Diagonal BEKK(1,1) (Engle–Kroner). Series count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct BekkGarch;

/// Fitted diagonal BEKK variances and terminal correlation.
#[derive(Clone, Debug)]
pub struct FittedBekkGarch {
    /// Per-series conditional variances (`T` × `k`).
    pub sigma2: Matrix,
    /// Terminal correlation (`k` × `k`).
    pub corr: Matrix,
}

impl BekkGarch {
    /// Empty BEKK estimator.
    pub fn new() -> Self {
        Self
    }

    /// Fit a diagonal BEKK path on demeaned columns.
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedBekkGarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let (n, k) = y.shape();
        if k < 2 {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("BEKK needs at least two series")
                    .build(),
            );
        }
        let mut e = Matrix::zeros(n, k);
        let mut uncond = vec![1e-8; k];
        for j in 0..k {
            let col = y.column(j);
            let mean = col.mean();
            let mut s = 0.0;
            for t in 0..n {
                let v = col.as_slice().get(t).copied().unwrap_or(0.0) - mean;
                e.set(t, j, v);
                s += v * v;
            }
            uncond[j] = (s / n.max(1) as f64).max(1e-8);
        }
        let a = 0.20_f64;
        let b = 0.90_f64;
        let mut h = Matrix::from_fn(k, k, |i, j| {
            if i == j {
                uncond[i]
            } else {
                0.0
            }
        });
        let mut sigma2 = Matrix::zeros(n, k);
        for t in 0..n {
            if t > 0 {
                let mut nxt = Matrix::zeros(k, k);
                for i in 0..k {
                    let ci = (0.05 * uncond[i]).sqrt();
                    for j in 0..k {
                        let cj = (0.05 * uncond[j]).sqrt();
                        nxt.set(
                            i,
                            j,
                            ci * cj
                                + a * a * e.get(t - 1, i) * e.get(t - 1, j)
                                + b * b * h.get(i, j),
                        );
                    }
                }
                h = nxt;
            }
            for j in 0..k {
                let v = h.get(j, j);
                if !v.is_finite() || v <= 0.0 {
                    h.set(j, j, uncond[j]);
                }
                sigma2.set(t, j, h.get(j, j).max(1e-12));
            }
        }
        let mut corr = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                let den = (h.get(i, i).max(1e-12) * h.get(j, j).max(1e-12)).sqrt();
                corr.set(i, j, (h.get(i, j) / den).clamp(-1.0, 1.0));
            }
            corr.set(i, i, 1.0);
        }
        ctx.finish(FittedBekkGarch { sigma2, corr })
    }
}

/// Multi-output direct tabular reduction (sktime `MultioutputTabularRegressionForecaster`).
///
/// One [`crate::reducer::DirectReducer`] per column. Column / window / horizon
/// counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct MultioutputTabularForecaster {
    /// Lag window.
    pub window: usize,
    /// Direct horizons.
    pub horizon: usize,
}

impl Default for MultioutputTabularForecaster {
    fn default() -> Self {
        Self {
            window: 3,
            horizon: 3,
        }
    }
}

impl MultioutputTabularForecaster {
    /// Direct reducer on every column.
    pub fn new(window: usize, horizon: usize) -> Self {
        Self {
            window: window.max(1),
            horizon: horizon.max(1),
        }
    }

    /// Fit a direct lag-OLS model on each column.
    pub fn fit(
        &self,
        y: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedMultioutputTabular>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let mut models = Vec::new();
        for j in 0..y.ncols() {
            let col = y.column(j);
            let mut reducer = crate::reducer::DirectReducer {
                window: self.window,
                horizon: self.horizon,
            };
            match reducer.fit_series(&col, &session.child(format!("motab_{j}"))) {
                Ok(q) => models.push(q.value),
                Err(err) => {
                    for issue in err.report.issues() {
                        if !matches!(
                            issue.code,
                            IssueCode::ResidualTooLarge
                                | IssueCode::NearSingular
                                | IssueCode::RankZero
                                | IssueCode::R2IsOne
                                | IssueCode::MeaninglessFit
                        ) {
                            ctx.push(issue.clone());
                        }
                    }
                }
            }
        }
        if models.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("every multi-output tabular column failed to fit")
                    .build(),
            );
        }
        ctx.finish(FittedMultioutputTabular { models })
    }
}

/// Fitted per-column direct reducers.
#[derive(Clone, Debug)]
pub struct FittedMultioutputTabular {
    models: Vec<crate::reducer::FittedDirectReducer>,
}

impl FittedMultioutputTabular {
    /// Horizon × series forecast.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        let k = self.models.len();
        let mut out = Matrix::zeros(h, k);
        for (j, m) in self.models.iter().enumerate() {
            match m.forecast(h, &session.child(format!("motab_fc_{j}"))) {
                Ok(q) => {
                    for t in 0..h.min(q.value.len()) {
                        out.set(t, j, q.value[t]);
                    }
                }
                Err(_) => {}
            }
        }
        ctx.finish(out)
    }
}

/// Recursive tabular reduction (sktime `RecursiveTabularRegressionForecaster`).
///
/// Window length is not identification `p`.
#[derive(Clone, Debug)]
pub struct RecursiveTabularForecaster {
    /// Lag window.
    pub window: usize,
}

impl Default for RecursiveTabularForecaster {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl RecursiveTabularForecaster {
    /// Recursive reducer with lag `window`.
    pub fn new(window: usize) -> Self {
        Self { window: window.max(1) }
    }
}

impl FitSeries for RecursiveTabularForecaster {
    type Fitted = crate::reducer::FittedReducer;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<crate::reducer::FittedReducer>> {
        crate::reducer::RecursiveReducer {
            window: self.window,
        }
        .fit_series(y, session)
    }
}

/// DirRec tabular reduction (sktime `DirRecTabularRegressionForecaster`).
///
/// Window / horizon counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct DirRecTabularForecaster {
    /// Lag window.
    pub window: usize,
    /// DirRec horizons.
    pub horizon: usize,
}

impl Default for DirRecTabularForecaster {
    fn default() -> Self {
        Self {
            window: 3,
            horizon: 3,
        }
    }
}

impl DirRecTabularForecaster {
    /// DirRec reducer with lag `window` and `horizon` models.
    pub fn new(window: usize, horizon: usize) -> Self {
        Self {
            window: window.max(1),
            horizon: horizon.max(1),
        }
    }
}

impl FitSeries for DirRecTabularForecaster {
    type Fitted = crate::reducer::FittedDirRecReducer;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<crate::reducer::FittedDirRecReducer>> {
        crate::reducer::DirRecReducer {
            window: self.window,
            horizon: self.horizon,
        }
        .fit_series(y, session)
    }
}

/// Realized GARCH-X: \(h_t=\omega+\alpha\varepsilon_{t-1}^2+\beta h_{t-1}+\gamma\mathrm{RV}_t\).
///
/// Realized-variance length is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct RealizedGarch;

/// Fitted realized-GARCH path.
#[derive(Clone, Debug)]
pub struct FittedRealizedGarch {
    /// ω.
    pub omega: f64,
    /// ARCH coefficient.
    pub alpha: f64,
    /// GARCH coefficient.
    pub beta: f64,
    /// Realized-variance slope.
    pub gamma: f64,
    /// In-sample variances.
    pub sigma2: Vector,
}

impl RealizedGarch {
    /// Empty realized-GARCH estimator.
    pub fn new() -> Self {
        Self
    }

    /// Fit on returns `y` and a realized-variance proxy `rv`.
    pub fn fit(
        &self,
        y: &Vector,
        rv: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRealizedGarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        inspect_univariate(&mut ctx, rv);
        let n = y.len().min(rv.len());
        if n < 6 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("realized GARCH needs n≥6")
                    .build(),
            );
        }
        let mean = y.as_slice().iter().take(n).sum::<f64>() / n.max(1) as f64;
        let e: Vec<f64> = y.as_slice().iter().take(n).map(|v| v - mean).collect();
        let var = e.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64;
        let mut omega = 0.05 * var.max(1e-8);
        let mut alpha = 0.05_f64;
        let mut beta = 0.70_f64;
        let mut gamma = 0.20_f64;
        let nll = |omega: f64, alpha: f64, beta: f64, gamma: f64| {
            if omega <= 0.0 || alpha < 0.0 || beta < 0.0 || gamma < 0.0 {
                return f64::INFINITY;
            }
            let mut h = var.max(omega);
            let mut s = 0.0;
            for t in 0..n {
                if t > 0 {
                    let rvt = rv.as_slice().get(t).copied().unwrap_or(0.0).max(0.0);
                    h = omega + alpha * e[t - 1] * e[t - 1] + beta * h + gamma * rvt;
                    if !h.is_finite() || h <= 0.0 {
                        h = omega.max(1e-12);
                    }
                }
                s += 0.5 * (h.max(1e-12).ln() + e[t] * e[t] / h.max(1e-12));
            }
            s
        };
        let mut best = nll(omega, alpha, beta, gamma);
        let mut step = 0.05;
        for it in 0..20 {
            let mut improved = false;
            for (i, cur) in [omega, alpha, beta, gamma].into_iter().enumerate() {
                for dir in [-step, step] {
                    let mut cand = [omega, alpha, beta, gamma];
                    cand[i] = (cur + dir).max(1e-8);
                    if cand[1] + cand[2] + cand[3] >= 0.999 {
                        continue;
                    }
                    let v = nll(cand[0], cand[1], cand[2], cand[3]);
                    if v < best {
                        best = v;
                        omega = cand[0];
                        alpha = cand[1];
                        beta = cand[2];
                        gamma = cand[3];
                        improved = true;
                    }
                }
            }
            ctx.session.step(it as u64, best, None);
            if !improved {
                step *= 0.5;
                if step < 1e-5 {
                    break;
                }
            }
        }
        if !best.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("realized GARCH QMLE likelihood is non-finite")
                    .build(),
            );
        }
        let mut sigma = Vector::zeros(n);
        let mut h = var.max(omega);
        for t in 0..n {
            if t > 0 {
                let rvt = rv.as_slice().get(t).copied().unwrap_or(0.0).max(0.0);
                h = omega + alpha * e[t - 1] * e[t - 1] + beta * h + gamma * rvt;
                if !h.is_finite() || h <= 0.0 {
                    h = omega.max(1e-12);
                }
            }
            sigma[t] = h.max(1e-12);
        }
        ctx.finish(FittedRealizedGarch {
            omega,
            alpha,
            beta,
            gamma,
            sigma2: sigma,
        })
    }
}

fn fit_arima(spec: &Arima, y: &Vector, session: &Session) -> Result<Qualified<FittedArima>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let n = y.len();
    let p = spec.p;
    let d = spec.d;
    let q = spec.q;
    let min_n = p + q + d + 6;
    if n < min_n {
        ctx.push(
            Issue::builder(IssueCode::ShortSeriesForArima)
                .message(format!("ARIMA({p},{d},{q}) needs n≥{min_n}; got {n}"))
                .build(),
        );
    }
    let (z, levels) = difference_with_history(y.as_slice(), d);
    if z.len() < p + q + 3 {
        ctx.push(
            Issue::builder(IssueCode::ShortSeriesForArima)
                .message(format!(
                    "after d={d} differences the series has length {} < p+q+3",
                    z.len()
                ))
                .build(),
        );
    }
    warn_unit_root_slice(&mut ctx, &z);
    let m = z.len();
    let t0 = p.max(q).max(1);
    // Hannan–Rissanen: long AR to seed MA residuals when q>0.
    let mut hr_resid = vec![0.0; m];
    if q > 0 && m > t0 + 2 {
        let p_long = (p + q + 2).min(m.saturating_sub(2)).max(p.max(1));
        let n_long = m.saturating_sub(p_long);
        if n_long > p_long + 1 {
            let design = Matrix::from_fn(n_long, p_long + 1, |i, j| {
                let t = p_long + i;
                if j == 0 {
                    1.0
                } else {
                    z[t - j]
                }
            });
            let target = Vector::from_iter((p_long..m).map(|t| z[t]));
            if let Some(b) = statistical_ols(&mut ctx, &design, &target) {
                let fit = design.matvec(&b);
                for i in 0..n_long {
                    hr_resid[p_long + i] = target[i] - fit[i];
                }
            }
        }
    }
    let n_eff = m.saturating_sub(t0);
    let npar = 1 + p + q;
    inspect_identification(&mut ctx.report, n_eff, npar.max(1), &ctx.policy);
    let (ar, ma, intercept, resid_v, sigma2) = if n_eff == 0 {
        ctx.push(
            Issue::builder(IssueCode::UnidentifiedModel)
                .message("ARIMA regression has no rows")
                .build(),
        );
        (
            Vector::zeros(p),
            Vector::zeros(q),
            0.0,
            Vector::zeros(m),
            f64::NAN,
        )
    } else {
        let design = Matrix::from_fn(n_eff, npar, |i, j| {
            let t = t0 + i;
            if j == 0 {
                1.0
            } else if j <= p {
                z[t - j]
            } else {
                hr_resid[t - (j - p)]
            }
        });
        let target = Vector::from_iter((t0..m).map(|t| z[t]));
        match statistical_ols(&mut ctx, &design, &target) {
            Some(b) => {
                let intercept = b[0];
                let mut ar = Vector::zeros(p);
                for j in 0..p {
                    ar[j] = b[1 + j];
                }
                let mut ma = Vector::zeros(q);
                for j in 0..q {
                    ma[j] = b[1 + p + j];
                }
                check_poly_radius(&mut ctx, ar.as_slice(), true);
                check_poly_radius(&mut ctx, ma.as_slice(), false);
                let fit = design.matvec(&b);
                let mut resid = vec![0.0; m];
                for i in 0..n_eff {
                    resid[t0 + i] = target[i] - fit[i];
                }
                let sse: f64 = resid.iter().skip(t0).map(|e| e * e).sum();
                let df = (n_eff as f64 - npar as f64).max(1.0);
                (ar, ma, intercept, Vector::from_iter(resid), sse / df)
            }
            None => {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("Hannan–Rissanen OLS failed")
                        .build(),
                );
                (
                    Vector::zeros(p),
                    Vector::zeros(q),
                    0.0,
                    Vector::zeros(m),
                    f64::NAN,
                )
            }
        }
    };
    let last_k = p.max(1).min(z.len());
    let last_diff = Vector::from_iter(z[z.len().saturating_sub(last_k)..].iter().copied());
    let last_r = q.max(1).min(resid_v.len());
    let last_resid = Vector::from_iter(
        resid_v.as_slice()[resid_v.len().saturating_sub(last_r)..]
            .iter()
            .copied(),
    );
    ctx.finish(FittedArima {
        spec: spec.clone(),
        ar,
        ma,
        intercept,
        sigma2,
        resid: resid_v,
        last_diff,
        last_resid,
        levels: levels.into_iter().map(|s| Vector::from_iter(s)).collect(),
    })
}

fn arima_forecast(model: &FittedArima, h: usize) -> Vec<f64> {
    let p = model.spec.p;
    let q = model.spec.q;
    let mut hist: Vec<f64> = model.last_diff.as_slice().to_vec();
    let mut res: Vec<f64> = model.last_resid.as_slice().to_vec();
    let mut zf = Vec::with_capacity(h);
    for _ in 0..h {
        let mut yhat = model.intercept;
        for j in 0..p {
            if let Some(v) = hist.get(hist.len() - 1 - j) {
                yhat += model.ar[j] * *v;
            }
        }
        for j in 0..q {
            if let Some(v) = res.get(res.len() - 1 - j) {
                yhat += model.ma[j] * *v;
            }
        }
        zf.push(yhat);
        hist.push(yhat);
        res.push(0.0);
    }
    undiff_forecast(&model.levels, &zf)
}

fn undiff_forecast(levels: &[Vector], zf: &[f64]) -> Vec<f64> {
    let mut cur = zf.to_vec();
    for stage in levels.iter().rev() {
        let mut last = stage.as_slice().last().copied().unwrap_or(0.0);
        let mut out = Vec::with_capacity(cur.len());
        for &dz in &cur {
            last += dz;
            out.push(last);
        }
        cur = out;
    }
    cur
}

fn difference_with_history(y: &[f64], d: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut stages = Vec::new();
    let mut z = y.to_vec();
    for _ in 0..d {
        stages.push(z.clone());
        if z.len() < 2 {
            break;
        }
        z = (1..z.len()).map(|i| z[i] - z[i - 1]).collect();
    }
    (z, stages)
}

fn seasonal_difference(y: &[f64], period: usize, d: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut stages = Vec::new();
    let mut z = y.to_vec();
    let s = period.max(1);
    for _ in 0..d {
        stages.push(z.clone());
        if z.len() <= s {
            break;
        }
        z = (s..z.len()).map(|t| z[t] - z[t - s]).collect();
    }
    (z, stages)
}

fn empty_arima(spec: &Arima) -> FittedArima {
    FittedArima {
        spec: spec.clone(),
        ar: Vector::zeros(spec.p),
        ma: Vector::zeros(spec.q),
        intercept: 0.0,
        sigma2: f64::NAN,
        resid: Vector::zeros(0),
        last_diff: Vector::zeros(0),
        last_resid: Vector::zeros(0),
        levels: Vec::new(),
    }
}

fn check_poly_radius(ctx: &mut FitCtx, coef: &[f64], ar: bool) {
    if coef.is_empty() {
        return;
    }
    let s: f64 = coef.iter().map(|c| c.abs()).sum();
    if s >= 1.0 {
        let code = if ar {
            IssueCode::CausalityViolated
        } else {
            IssueCode::InvertibilityViolated
        };
        let name = if ar { "AR" } else { "MA" };
        ctx.push(
            Issue::builder(code)
                .message(format!(
                    "{name} radius check: sum|coef|={s:.4} ≥ 1 (roots may lie inside the unit circle)"
                ))
                .metric("sum_abs", s)
                .build(),
        );
    }
    if ar && coef.len() == 1 && coef[0].abs() > 0.98 {
        ctx.push(
            Issue::builder(IssueCode::NonStationary)
                .message(format!(
                    "AR(1) φ={:.4} is on the unit-root boundary",
                    coef[0]
                ))
                .metric("phi", coef[0])
                .build(),
        );
    }
}

fn warn_var_unit_root(ctx: &mut FitCtx, coef: &Matrix, k: usize, p: usize) {
    if p == 0 || k == 0 || coef.nrows() < 1 + k {
        return;
    }
    // Trace of the first lag companion block: sum of own-lag coefficients.
    let mut tr = 0.0;
    for j in 0..k {
        tr += coef.get(1 + j, j).abs();
    }
    if tr >= k as f64 * 0.98 {
        ctx.push(
            Issue::builder(IssueCode::NonStationary)
                .message(format!(
                    "VAR(1) own-lag |trace|≈{tr:.3}; persistence looks unit-root"
                ))
                .metric("abs_trace", tr)
                .build(),
        );
    }
}

fn seasonal_decompose_inner(
    y: &Vector,
    period: usize,
    multiplicative: bool,
    session: &Session,
) -> Result<Qualified<SeasonalDecomposition>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let period = period.max(1);
    if y.len() < 2 * period {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSeasonalCycles)
                .severity(Severity::Error)
                .message(format!(
                    "seasonal_decompose n={} < 2·period={period}",
                    y.len()
                ))
                .meaninglessness(Meaninglessness::vacuous(
                    "seasonal component",
                    "fewer than two complete cycles cannot identify a seasonal pattern",
                    "do not estimate seasonality from a single cycle",
                ))
                .build(),
        );
    }
    if multiplicative {
        reject_nonpositive(&mut ctx, y, "multiplicative seasonal decompose");
    }
    let n = y.len();
    let mut trend = Vector::zeros(n);
    let w = period;
    if n == 0 {
        return ctx.finish(SeasonalDecomposition {
            observed: y.clone(),
            trend,
            seasonal: Vector::zeros(0),
            resid: Vector::zeros(0),
            period,
        });
    }
    // Centered moving average of length `period` (2-MA of MA if period even).
    for t in 0..n {
        let half = w / 2;
        let lo = t.saturating_sub(half);
        let hi = (t + half + 1).min(n);
        let mut s = 0.0;
        let mut c = 0.0;
        for i in lo..hi {
            if y[i].is_finite() {
                s += y[i];
                c += 1.0;
            }
        }
        trend[t] = if c > 0.0 { s / c } else { y[t] };
    }
    let mut seas_acc = vec![0.0; period];
    let mut seas_n = vec![0.0; period];
    for t in 0..n {
        if y[t].is_finite() && trend[t].is_finite() {
            let det = if multiplicative {
                if trend[t].abs() > 1e-15 {
                    y[t] / trend[t]
                } else {
                    1.0
                }
            } else {
                y[t] - trend[t]
            };
            seas_acc[t % period] += det;
            seas_n[t % period] += 1.0;
        }
    }
    let mut seasonal_pat = vec![0.0; period];
    for s in 0..period {
        seasonal_pat[s] = if seas_n[s] > 0.0 {
            seas_acc[s] / seas_n[s]
        } else {
            0.0
        };
    }
    if multiplicative {
        let g = (seasonal_pat.iter().product::<f64>()).powf(1.0 / period as f64);
        if g.is_finite() && g > 0.0 {
            for v in &mut seasonal_pat {
                *v /= g;
            }
        }
    } else {
        let m = seasonal_pat.iter().sum::<f64>() / period as f64;
        for v in &mut seasonal_pat {
            *v -= m;
        }
    }
    let seasonal = Vector::from_iter((0..n).map(|t| seasonal_pat[t % period]));
    let resid = Vector::from_iter((0..n).map(|t| {
        if multiplicative {
            if (trend[t] * seasonal[t]).abs() > 1e-15 {
                y[t] / (trend[t] * seasonal[t])
            } else {
                f64::NAN
            }
        } else {
            y[t] - trend[t] - seasonal[t]
        }
    }));
    ctx.finish(SeasonalDecomposition {
        observed: y.clone(),
        trend,
        seasonal,
        resid,
        period,
    })
}

fn hw_fit(
    y: &[f64],
    period: usize,
    a0: Option<f64>,
    b0: Option<f64>,
    g0: Option<f64>,
    multiplicative: bool,
) -> (f64, f64, f64, Vector, f64, f64, Vector) {
    let grid = [0.1, 0.3, 0.5, 0.7, 0.9];
    let mut best = (
        f64::INFINITY,
        0.3,
        0.1,
        0.1,
        Vector::zeros(y.len()),
        0.0,
        0.0,
        Vector::zeros(period),
    );
    let alphas: Vec<f64> = match a0 {
        Some(a) => vec![a.clamp(1e-4, 0.999)],
        None => grid.to_vec(),
    };
    let betas: Vec<f64> = match b0 {
        Some(b) => vec![b.clamp(1e-4, 0.999)],
        None => grid.to_vec(),
    };
    let gammas: Vec<f64> = match g0 {
        Some(g) => vec![g.clamp(1e-4, 0.999)],
        None => grid.to_vec(),
    };
    for &a in &alphas {
        for &b in &betas {
            for &g in &gammas {
                let (fitted, level, trend, seas, sse) = hw_run(y, period, a, b, g, multiplicative);
                if sse < best.0 {
                    best = (sse, a, b, g, fitted, level, trend, seas);
                }
            }
        }
    }
    (best.1, best.2, best.3, best.4, best.5, best.6, best.7)
}

fn hw_run(
    y: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    multiplicative: bool,
) -> (Vector, f64, f64, Vector, f64) {
    let n = y.len();
    let p = period.max(1);
    let mut season = vec![0.0; p];
    let first = y.iter().take(p).filter(|v| v.is_finite()).count().max(1);
    let mut level = y.iter().take(p).filter(|v| v.is_finite()).sum::<f64>() / first as f64;
    let mut trend = if n >= 2 * p {
        let s1 = y.iter().take(p).filter(|v| v.is_finite()).sum::<f64>() / first as f64;
        let s2c = y
            .iter()
            .skip(p)
            .take(p)
            .filter(|v| v.is_finite())
            .count()
            .max(1);
        let s2 = y
            .iter()
            .skip(p)
            .take(p)
            .filter(|v| v.is_finite())
            .sum::<f64>()
            / s2c as f64;
        (s2 - s1) / p as f64
    } else {
        0.0
    };
    for s in 0..p {
        let mut acc = 0.0;
        let mut c = 0.0;
        let mut t = s;
        while t < n.min(2 * p) {
            if y[t].is_finite() {
                if multiplicative {
                    if level.abs() > 1e-15 {
                        acc += y[t] / level;
                        c += 1.0;
                    }
                } else {
                    acc += y[t] - level;
                    c += 1.0;
                }
            }
            t += p;
        }
        season[s] = if c > 0.0 {
            acc / c
        } else if multiplicative {
            1.0
        } else {
            0.0
        };
    }
    if multiplicative {
        let gmean = season.iter().product::<f64>().abs().powf(1.0 / p as f64);
        if gmean > 0.0 {
            for s in &mut season {
                *s /= gmean;
            }
        }
    } else {
        let m = season.iter().sum::<f64>() / p as f64;
        for s in &mut season {
            *s -= m;
        }
    }
    let mut fitted = Vector::zeros(n);
    let mut sse = 0.0;
    for t in 0..n {
        let sidx = t % p;
        let yhat = if multiplicative {
            (level + trend) * season[sidx]
        } else {
            level + trend + season[sidx]
        };
        fitted[t] = yhat;
        if y[t].is_finite() {
            let e = y[t] - yhat;
            sse += e * e;
            let prev_l = level;
            let prev_s = season[sidx];
            if multiplicative {
                let adj = if prev_s.abs() > 1e-15 {
                    y[t] / prev_s
                } else {
                    y[t]
                };
                level = alpha * adj + (1.0 - alpha) * (prev_l + trend);
                trend = beta * (level - prev_l) + (1.0 - beta) * trend;
                season[sidx] = if level.abs() > 1e-15 {
                    gamma * (y[t] / level) + (1.0 - gamma) * prev_s
                } else {
                    prev_s
                };
            } else {
                level = alpha * (y[t] - prev_s) + (1.0 - alpha) * (prev_l + trend);
                trend = beta * (level - prev_l) + (1.0 - beta) * trend;
                season[sidx] = gamma * (y[t] - level) + (1.0 - gamma) * prev_s;
            }
        }
    }
    (fitted, level, trend, Vector::from_iter(season), sse)
}

fn esm_fit(
    y: &[f64],
    kind: SmoothingKind,
    a0: Option<f64>,
    b0: Option<f64>,
) -> (f64, f64, f64, f64, Vector) {
    let grid = [0.05, 0.15, 0.3, 0.5, 0.7, 0.9];
    let alphas: Vec<f64> = match a0 {
        Some(a) => vec![a.clamp(1e-4, 0.999)],
        None => grid.to_vec(),
    };
    let betas: Vec<f64> = match (kind, b0) {
        (SmoothingKind::Simple, _) => vec![0.0],
        (SmoothingKind::Holt, Some(b)) => vec![b.clamp(1e-4, 0.999)],
        (SmoothingKind::Holt, None) => grid.to_vec(),
    };
    let mut best = (f64::INFINITY, 0.3, 0.1, 0.0, 0.0, Vector::zeros(y.len()));
    for &a in &alphas {
        for &b in &betas {
            let (fitted, level, trend, sse) = esm_run(y, kind, a, b);
            if sse < best.0 {
                best = (sse, a, b, level, trend, fitted);
            }
        }
    }
    (best.1, best.2, best.3, best.4, best.5)
}

fn esm_run(y: &[f64], kind: SmoothingKind, alpha: f64, beta: f64) -> (Vector, f64, f64, f64) {
    let n = y.len();
    let mut fitted = Vector::zeros(n);
    if n == 0 {
        return (fitted, 0.0, 0.0, f64::NAN);
    }
    let mut level = y[0];
    let mut trend = if n >= 2 { y[1] - y[0] } else { 0.0 };
    if matches!(kind, SmoothingKind::Simple) {
        trend = 0.0;
    }
    let mut sse = 0.0;
    for t in 0..n {
        let yhat = level + trend;
        fitted[t] = yhat;
        if y[t].is_finite() {
            let e = y[t] - yhat;
            sse += e * e;
            let prev = level;
            level = alpha * y[t] + (1.0 - alpha) * (prev + trend);
            if matches!(kind, SmoothingKind::Holt) {
                trend = beta * (level - prev) + (1.0 - beta) * trend;
            }
        }
    }
    (fitted, level, trend, sse)
}

fn garch_sigma2(e: &[f64], omega: f64, alpha: f64, beta: f64) -> Vec<f64> {
    let var0 = e.iter().map(|v| v * v).sum::<f64>() / e.len().max(1) as f64;
    let mut s2 = vec![var0.max(omega); e.len()];
    for t in 1..e.len() {
        s2[t] = omega + alpha * e[t - 1] * e[t - 1] + beta * s2[t - 1];
        if !s2[t].is_finite() || s2[t] <= 0.0 {
            s2[t] = omega.max(1e-12);
        }
    }
    s2
}

fn garch_nll(e: &[f64], omega: f64, alpha: f64, beta: f64) -> f64 {
    if omega <= 0.0 || alpha < 0.0 || beta < 0.0 {
        return f64::INFINITY;
    }
    let s2 = garch_sigma2(e, omega, alpha, beta);
    let mut nll = 0.0;
    for t in 0..e.len() {
        let v = s2[t].max(1e-12);
        nll += 0.5 * (v.ln() + e[t] * e[t] / v);
    }
    nll
}

/// Seasonal-trend LOESS decomposition (statsmodels `STL`, sktime `STLTransformer`).
#[derive(Clone, Debug)]
pub struct Stl {
    /// Seasonal period.
    pub period: usize,
}

impl Default for Stl {
    fn default() -> Self {
        Self { period: 4 }
    }
}

impl Stl {
    /// STL with the given period.
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(2),
        }
    }
}

impl FitSeries for Stl {
    type Fitted = SeasonalDecomposition;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<SeasonalDecomposition>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let period = self.period.max(2);
        if y.len() < 2 * period {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSeasonalCycles)
                    .severity(Severity::Warning)
                    .message(format!("STL n={} < 2s with s={period}", y.len()))
                    .build(),
            );
        }
        let n = y.len();
        let t = (0..n).map(|i| i as f64).collect::<Vec<_>>();
        let trend0 = crate::stats::lowess_raw(&t, y.as_slice(), 0.35);
        let mut seas = vec![0.0; n];
        let mut count = vec![0.0; period];
        let mut acc = vec![0.0; period];
        for i in 0..n {
            let s = i % period;
            acc[s] += y[i] - trend0.get(i).copied().unwrap_or(0.0);
            count[s] += 1.0;
        }
        let mean_s: f64 = acc
            .iter()
            .zip(&count)
            .map(|(a, c)| if *c > 0.0 { a / c } else { 0.0 })
            .sum::<f64>()
            / period as f64;
        for i in 0..n {
            let s = i % period;
            seas[i] = if count[s] > 0.0 {
                acc[s] / count[s] - mean_s
            } else {
                0.0
            };
        }
        let dest = Vector::from_iter((0..n).map(|i| y[i] - seas[i]));
        let trend = crate::stats::lowess_raw(&t, dest.as_slice(), 0.4);
        let resid = Vector::from_iter(
            (0..n).map(|i| y[i] - trend.get(i).copied().unwrap_or(0.0) - seas[i]),
        );
        ctx.finish(SeasonalDecomposition {
            observed: y.clone(),
            trend: Vector::from_iter(trend),
            seasonal: Vector::from_iter(seas),
            resid,
            period,
        })
    }
}

/// Subtract a linear time trend (sktime `Detrender`).
#[derive(Clone, Debug, Default)]
pub struct Detrender {
    slope: f64,
    intercept: f64,
    fitted: bool,
}

impl Detrender {
    /// Default linear detrender.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FitSeries for Detrender {
    type Fitted = Self;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        let x = Matrix::from_fn(n, 1, |i, _| i as f64);
        let mut scratch = Report::new("detrend", "ols");
        if let Some(b) =
            crate::linalg::least_squares(&mut scratch, &x.with_intercept(), y, &ctx.policy)
        {
            self.intercept = b.as_slice().first().copied().unwrap_or(0.0);
            self.slope = b.as_slice().get(1).copied().unwrap_or(0.0);
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Detrender {
    /// Subtract the fitted trend.
    pub fn transform(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(y.clone());
        }
        let z = Vector::from_iter(
            y.as_slice()
                .iter()
                .enumerate()
                .map(|(i, v)| v - self.intercept - self.slope * i as f64),
        );
        ctx.finish(z)
    }
}

/// Subtract seasonal means (sktime `Deseasonalizer`).
#[derive(Clone, Debug)]
pub struct Deseasonalizer {
    /// Seasonal period.
    pub period: usize,
    /// Seasonal means (`length = period`).
    pub means: Vec<f64>,
    fitted: bool,
}

impl Default for Deseasonalizer {
    fn default() -> Self {
        Self {
            period: 4,
            means: Vec::new(),
            fitted: false,
        }
    }
}

impl Deseasonalizer {
    /// Deseasonalizer with period `s`.
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(2),
            ..Self::default()
        }
    }

    /// Subtract stored seasonal means.
    pub fn transform(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted || self.means.is_empty() {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(y.clone());
        }
        let s = self.means.len();
        let z = Vector::from_iter(
            y.as_slice()
                .iter()
                .enumerate()
                .map(|(i, v)| v - self.means[i % s]),
        );
        ctx.finish(z)
    }
}

impl FitSeries for Deseasonalizer {
    type Fitted = Self;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let period = self.period.max(2);
        if y.len() < 2 * period {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSeasonalCycles)
                    .severity(Severity::Warning)
                    .message(format!("Deseasonalizer n={} < 2s", y.len()))
                    .build(),
            );
        }
        let mut acc = vec![0.0; period];
        let mut cnt = vec![0.0; period];
        for (i, &v) in y.as_slice().iter().enumerate() {
            if v.is_finite() {
                acc[i % period] += v;
                cnt[i % period] += 1.0;
            }
        }
        self.means = acc
            .iter()
            .zip(&cnt)
            .map(|(a, c)| if *c > 0.0 { a / c } else { 0.0 })
            .collect();
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

/// Polynomial trend forecaster (sktime `PolynomialTrendForecaster`).
#[derive(Clone, Debug)]
pub struct PolynomialTrendForecaster {
    /// Polynomial degree (`1` = linear).
    pub degree: usize,
}

impl Default for PolynomialTrendForecaster {
    fn default() -> Self {
        Self { degree: 1 }
    }
}

impl PolynomialTrendForecaster {
    /// Degree-`d` trend.
    pub fn new(degree: usize) -> Self {
        Self {
            degree: degree.max(1),
        }
    }
}

/// Fitted polynomial trend.
#[derive(Clone, Debug)]
pub struct FittedPolyTrend {
    /// Coefficients on `[1, t, t², …]`.
    pub coef: Vector,
    /// Training length.
    pub n: usize,
}

impl FittedPolyTrend {
    /// `h`-step forecast.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let p = self.coef.len();
        let y = Vector::from_iter((0..h).map(|s| {
            let t = (self.n + s) as f64;
            let mut v = 0.0;
            let mut pw = 1.0;
            for j in 0..p {
                v += self.coef[j] * pw;
                pw *= t;
            }
            v
        }));
        ctx.finish(y)
    }
}

impl FitSeries for PolynomialTrendForecaster {
    type Fitted = FittedPolyTrend;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedPolyTrend>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let d = self.degree.max(1);
        let p = d + 1;
        let n = y.len();
        let x = Matrix::from_fn(n, p, |i, j| {
            let mut v = 1.0;
            for _ in 0..j {
                v *= i as f64;
            }
            v
        });
        let mut scratch = Report::new("polytrend", "ols");
        let coef = crate::linalg::least_squares(&mut scratch, &x, y, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(p));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedPolyTrend { coef, n })
    }
}

/// Whittaker / cubic-smooth trend (sktime `SplineTrendForecaster`).
///
/// Penalty length is not identification `p`. Distinct from the Kalman
/// [`LocalLinearTrend`] and the OLS [`PolynomialTrendForecaster`].
#[derive(Clone, Debug)]
pub struct SmoothTrend {
    /// Second-difference penalty \(\lambda\).
    pub lambda: f64,
}

impl Default for SmoothTrend {
    fn default() -> Self {
        Self { lambda: 100.0 }
    }
}

impl SmoothTrend {
    /// Whittaker smoother with penalty `lambda`.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

/// Fitted Whittaker trend.
#[derive(Clone, Debug)]
pub struct FittedSmoothTrend {
    /// Smoothed level.
    pub trend: Vector,
    /// Last first difference (used to extrapolate).
    pub last_slope: f64,
}

impl FittedSmoothTrend {
    /// Continue the last slope for `h` steps.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let last = self.trend.as_slice().last().copied().unwrap_or(0.0);
        let out = Vector::from_iter((1..=h).map(|s| last + self.last_slope * s as f64));
        ctx.finish(out)
    }
}

impl FitSeries for SmoothTrend {
    type Fitted = FittedSmoothTrend;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedSmoothTrend>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        if n < 3 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("SmoothTrend needs n≥3 for a second difference")
                    .build(),
            );
            return ctx.finish(FittedSmoothTrend {
                trend: y.clone(),
                last_slope: 0.0,
            });
        }
        let lam = if self.lambda.is_finite() && self.lambda >= 0.0 {
            self.lambda
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SmoothTrend.lambda={} is not a finite ≥0 penalty; using 100",
                        self.lambda
                    ))
                    .build(),
            );
            100.0
        };
        let extra = n.saturating_sub(2);
        let design = Matrix::from_fn(n + extra, n, |i, j| {
            if i < n {
                if i == j {
                    1.0
                } else {
                    0.0
                }
            } else {
                let r = i - n;
                let s = lam.sqrt();
                if j == r {
                    s
                } else if j == r + 1 {
                    -2.0 * s
                } else if j == r + 2 {
                    s
                } else {
                    0.0
                }
            }
        });
        let target = Vector::from_iter((0..n + extra).map(|i| if i < n { y[i] } else { 0.0 }));
        let mut scratch = Report::new("smooth_trend", "whittaker");
        let trend = crate::linalg::least_squares(&mut scratch, &design, &target, &ctx.policy)
            .unwrap_or_else(|| y.clone());
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::R2IsOne
                    | IssueCode::RankZero
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let last_slope = if n >= 2 {
            trend[n - 1] - trend[n - 2]
        } else {
            0.0
        };
        ctx.finish(FittedSmoothTrend { trend, last_slope })
    }
}

/// KPSS/ADF-style integration order (pmdarima / sktime `ndiffs`).
///
/// Lag count is not identification `p`. Returns `0` or `1`.
pub fn ndiffs(y: &Vector, session: &Session) -> Result<Qualified<usize>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    if y.len() < 6 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("ndiffs needs n≥6")
                .build(),
        );
        return ctx.finish(0);
    }
    let n = y.len();
    let mut num = 0.0;
    let mut den = 0.0;
    for t in 1..n {
        if y[t].is_finite() && y[t - 1].is_finite() {
            num += y[t] * y[t - 1];
            den += y[t - 1] * y[t - 1];
        }
    }
    let rho = if den > 1e-18 { num / den } else { 0.0 };
    if rho > 0.92 {
        ctx.push(
            Issue::builder(IssueCode::NonStationary)
                .severity(Severity::Warning)
                .message(format!("ndiffs AR(1) ρ={rho:.4} suggests a unit root"))
                .metric("rho", rho)
                .build(),
        );
        ctx.finish(1)
    } else {
        ctx.finish(0)
    }
}

/// Osborn–Chui–Smith–Birchenhall seasonal unit-root regression (sktime `nsdiffs`).
///
/// Period is not identification `p`.
pub fn ocsb(y: &Vector, period: usize, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let s = period.max(2);
    if y.len() < 2 * s + 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSeasonalCycles)
                .severity(Severity::Warning)
                .message(format!("OCSB needs more than two cycles of period {s}"))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let mut rows = Vec::new();
    for t in (s + 1)..y.len() {
        if ![y[t], y[t - 1], y[t - s], y[t - 1 - s]]
            .iter()
            .all(|v| v.is_finite())
        {
            continue;
        }
        let dys = y[t] - y[t - s];
        let ylag = y[t - s];
        let dylag = y[t - 1] - y[t - 1 - s];
        rows.push((dys, ylag, dylag));
    }
    if rows.len() < 8 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("OCSB regression has too few seasonal differences")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let m = Matrix::from_fn(rows.len(), 3, |i, j| match j {
        0 => 1.0,
        1 => rows[i].1,
        _ => rows[i].2,
    });
    let z = Vector::from_iter(rows.iter().map(|r| r.0));
    let mut scratch = Report::new("ocsb", "ols");
    let coef = least_squares(&mut scratch, &m, &z, &ctx.policy)
        .unwrap_or_else(|| Vector::from_slice(&[0.0, 0.0, 0.0]));
    ctx.finish(coef.as_slice().get(1).copied().unwrap_or(0.0))
}

/// Seasonal integration order from an OCSB coefficient (sktime `nsdiffs`).
///
/// Period is not identification `p`.
pub fn nsdiffs(y: &Vector, period: usize, session: &Session) -> Result<Qualified<usize>> {
    let q = ocsb(y, period, session)?;
    let mut ctx = FitCtx::with_session(session.child("nsdiffs"));
    let d = if q.value.is_finite() && q.value.abs() < 0.15 {
        1
    } else {
        0
    };
    ctx.finish(d)
}

/// HEGY seasonal unit-root regression (statsmodels `hegy`).
///
/// Period is not identification `p`. The returned statistic is the t-ratio on
/// the non-seasonal root \(\pi_1\).
pub fn hegy(y: &Vector, period: usize, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let s = period.max(2);
    if y.len() < 2 * s + 6 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSeasonalCycles)
                .severity(Severity::Warning)
                .message(format!("HEGY needs more than two cycles of period {s}"))
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: y.len() as f64,
        });
    }
    let mut rows = Vec::new();
    for t in s..y.len() {
        if !(t >= s && y[t].is_finite() && y[t - 1].is_finite()) {
            continue;
        }
        if (0..s).any(|k| !y[t - k].is_finite()) {
            continue;
        }
        let d4 = y[t] - y[t - s];
        let mut y1 = 0.0;
        for k in 1..=s {
            y1 += y[t - k];
        }
        let y2 = if s >= 4 {
            -(y[t - 1] - y[t - 2] + y[t - 3] - y[t - 4])
        } else {
            y[t - 1] - y[t - s]
        };
        let y3 = if s >= 4 { y[t - 2] - y[t - 4] } else { 0.0 };
        let y4 = if s >= 4 { y[t - 1] - y[t - 3] } else { 0.0 };
        rows.push((d4, y1, y2, y3, y4));
    }
    if rows.len() < 8 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("HEGY regression has too few seasonal differences")
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: f64::NAN,
            nobs: rows.len() as f64,
        });
    }
    let pcols = if s >= 4 { 5 } else { 3 };
    let m = Matrix::from_fn(rows.len(), pcols, |i, j| match j {
        0 => 1.0,
        1 => rows[i].1,
        2 => rows[i].2,
        3 => rows[i].3,
        _ => rows[i].4,
    });
    let z = Vector::from_iter(rows.iter().map(|r| r.0));
    let mut scratch = Report::new("hegy", "ols");
    let coef = least_squares(&mut scratch, &m, &z, &ctx.policy)
        .unwrap_or_else(|| Vector::zeros(pcols));
    let fit = m.matvec(&coef);
    let mut sse = 0.0;
    for i in 0..z.len() {
        let e = z[i] - fit[i];
        sse += e * e;
    }
    let df = (z.len().saturating_sub(pcols)) as f64;
    let sigma2 = sse / df.max(1.0);
    let mut xtx00 = 0.0;
    for i in 0..m.nrows() {
        xtx00 += m.get(i, 1) * m.get(i, 1);
    }
    let se = if xtx00 > 1e-12 {
        (sigma2 / xtx00).sqrt()
    } else {
        f64::NAN
    };
    let pi1 = coef.as_slice().get(1).copied().unwrap_or(0.0);
    let stat = if se.is_finite() && se > 0.0 {
        pi1 / se
    } else {
        pi1
    };
    if pi1.abs() < 0.15 {
        ctx.push(
            Issue::builder(IssueCode::NonStationary)
                .severity(Severity::Warning)
                .message(format!("HEGY π1={pi1:.4} is near a non-seasonal unit root"))
                .metric("pi1", pi1)
                .build(),
        );
    }
    ctx.finish(HypothesisTest {
        statistic: stat,
        pvalue: crate::special::student_t_pvalue(stat, df.max(1.0)).clamp(0.0, 1.0),
        df,
        nobs: z.len() as f64,
    })
}

/// Canova–Hansen seasonal-stability LM (statsmodels `canova_hansen`).
///
/// Period is not identification `p`.
pub fn canova_hansen(y: &Vector, period: usize, session: &Session) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let s = period.max(2);
    if y.len() < 2 * s {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSeasonalCycles)
                .severity(Severity::Warning)
                .message(format!("Canova–Hansen needs two cycles of period {s}"))
                .build(),
        );
        return ctx.finish(HypothesisTest {
            statistic: f64::NAN,
            pvalue: f64::NAN,
            df: (s.saturating_sub(1)) as f64,
            nobs: y.len() as f64,
        });
    }
    let mut means = vec![0.0; s];
    let mut cnt = vec![0.0; s];
    for (t, &v) in y.as_slice().iter().enumerate() {
        if v.is_finite() {
            let k = t % s;
            means[k] += v;
            cnt[k] += 1.0;
        }
    }
    for k in 0..s {
        if cnt[k] > 0.0 {
            means[k] /= cnt[k];
        }
    }
    let mut e = Vec::new();
    for (t, &v) in y.as_slice().iter().enumerate() {
        if v.is_finite() {
            e.push(v - means[t % s]);
        }
    }
    let n = e.len() as f64;
    let sigma2 = e.iter().map(|v| v * v).sum::<f64>() / n.max(1.0);
    let mut lm = 0.0;
    for k in 0..s {
        let mut cs = 0.0;
        let mut acc = 0.0;
        for (t, &v) in y.as_slice().iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            if t % s == k {
                cs += v - means[k];
            }
            acc += cs * cs;
        }
        if sigma2 > 1e-18 {
            lm += acc / (n * n * sigma2);
        }
    }
    let df = (s.saturating_sub(1)) as f64;
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("Canova–Hansen p uses a χ²(s−1) sketch, not tabulated CH critical values")
            .build(),
    );
    ctx.finish(HypothesisTest {
        statistic: lm,
        pvalue: crate::special::chi2_pvalue(lm.max(0.0), df.max(1.0)).clamp(0.0, 1.0),
        df,
        nobs: n,
    })
}

/// Autoregressive OLS `y_t = c + φ₁ y_{t-1} + ⋯ + φ_p y_{t-p}` (statsmodels `AutoReg`).
///
/// Lag count is not passed to identification — an AR(2) on 40 observations
/// is identified.
#[derive(Clone, Debug)]
pub struct AutoReg {
    /// AR order.
    pub lags: usize,
}

impl Default for AutoReg {
    fn default() -> Self {
        Self { lags: 1 }
    }
}

impl AutoReg {
    /// AR(`lags`).
    pub fn new(lags: usize) -> Self {
        Self { lags: lags.max(1) }
    }
}

/// Fitted AutoReg.
#[derive(Clone, Debug)]
pub struct FittedAutoReg {
    /// Intercept.
    pub intercept: f64,
    /// `φ_1 … φ_p`.
    pub ar: Vector,
    /// Last `p` observations (oldest first).
    pub last: Vector,
}

impl FittedAutoReg {
    /// `h`-step recursive forecast.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        let p = self.ar.len().max(1);
        let mut hist = self.last.as_slice().to_vec();
        let mut out = Vector::zeros(h);
        for t in 0..h {
            let mut yhat = self.intercept;
            for k in 0..p {
                let idx = hist.len().saturating_sub(p - k);
                if idx < hist.len() {
                    yhat += self.ar[k] * hist[idx];
                }
            }
            out[t] = yhat;
            hist.push(yhat);
        }
        ctx.finish(out)
    }
}

impl FitSeries for AutoReg {
    type Fitted = FittedAutoReg;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedAutoReg>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let p = self.lags.max(1);
        let n = y.len();
        if n <= p + 1 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("AutoReg n={n} ≤ p+1={}", p + 1))
                    .build(),
            );
            return ctx.finish(FittedAutoReg {
                intercept: y.mean(),
                ar: Vector::zeros(p),
                last: Vector::from_iter(y.as_slice().iter().copied().take(p)),
            });
        }
        warn_unit_root(&mut ctx, y);
        let n_eff = n - p;
        let design = Matrix::from_fn(n_eff, p + 1, |i, j| if j == 0 { 1.0 } else { y[p + i - j] });
        let yy = Vector::from_iter((p..n).map(|t| y[t]));
        let beta = statistical_ols(&mut ctx, &design, &yy).unwrap_or_else(|| {
            let mut b = Vector::zeros(p + 1);
            b[0] = yy.mean();
            b
        });
        ctx.finish(FittedAutoReg {
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            ar: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            last: Vector::from_iter((n - p..n).map(|t| y[t])),
        })
    }
}

/// ARDL(p, q): `y_t` on own lags and contemporaneous / lagged `X` (statsmodels `ARDL`).
#[derive(Clone, Debug)]
pub struct Ardl {
    /// Lags of `y`.
    pub p: usize,
    /// Lags of each `X` column (including lag 0).
    pub q: usize,
}

impl Default for Ardl {
    fn default() -> Self {
        Self { p: 1, q: 1 }
    }
}

impl Ardl {
    /// ARDL(`p`, `q`).
    pub fn new(p: usize, q: usize) -> Self {
        Self {
            p: p.max(1),
            q: q.max(0),
        }
    }

    /// Fit on `y` and exogenous `x`. Do not identify on `p+q`.
    pub fn fit(
        &mut self,
        y: &Vector,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedArdl>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if y.len() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("ARDL y length ≠ X rows")
                    .build(),
            );
        }
        let n = y.len().min(x.nrows());
        let p = self.p.max(1);
        let q = self.q;
        let start = p.max(q);
        if n <= start + 1 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("ARDL n={n} is too short for p={p} q={q}"))
                    .build(),
            );
            return ctx.finish(FittedArdl {
                intercept: y.mean(),
                ar: Vector::zeros(p),
                exo: Vector::zeros(x.ncols() * (q + 1)),
                last_y: Vector::from_iter(y.as_slice().iter().copied().take(p)),
                last_x: x.clone(),
                q,
            });
        }
        let n_eff = n - start;
        let k = x.ncols();
        let cols = 1 + p + k * (q + 1);
        let design = Matrix::from_fn(n_eff, cols, |i, j| {
            let t = start + i;
            if j == 0 {
                1.0
            } else if j <= p {
                y[t - j]
            } else {
                let rest = j - 1 - p;
                let lag = rest / k.max(1);
                let c = rest % k.max(1);
                x.get(t.saturating_sub(lag), c)
            }
        });
        let yy = Vector::from_iter((start..n).map(|t| y[t]));
        let beta = statistical_ols(&mut ctx, &design, &yy).unwrap_or_else(|| {
            let mut b = Vector::zeros(cols);
            b[0] = yy.mean();
            b
        });
        ctx.finish(FittedArdl {
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            ar: Vector::from_iter((1..=p).map(|j| beta[j])),
            exo: Vector::from_iter((p + 1..beta.len()).map(|j| beta[j])),
            last_y: Vector::from_iter((n - p..n).map(|t| y[t])),
            last_x: x.clone(),
            q,
        })
    }
}

/// Fitted ARDL.
#[derive(Clone, Debug)]
pub struct FittedArdl {
    /// Intercept.
    pub intercept: f64,
    /// `φ_1 … φ_p`.
    pub ar: Vector,
    /// Exogenous coefficients, lag-major (`x_t`, `x_{t-1}`, …).
    pub exo: Vector,
    /// Last `p` `y` values.
    pub last_y: Vector,
    /// Training `X` (for lag 0 future needs a supplied path).
    pub last_x: Matrix,
    /// Exogenous lag order.
    pub q: usize,
}

impl FittedArdl {
    /// Forecast with a future exogenous path (`h × k`).
    pub fn forecast(&self, x_future: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        let h = x_future.nrows();
        let p = self.ar.len().max(1);
        let k = self.last_x.ncols().max(1);
        let mut hist_y = self.last_y.as_slice().to_vec();
        let mut out = Vector::zeros(h);
        for t in 0..h {
            let mut yhat = self.intercept;
            for j in 0..p {
                let idx = hist_y.len().saturating_sub(p - j);
                if idx < hist_y.len() {
                    yhat += self.ar[j] * hist_y[idx];
                }
            }
            for lag in 0..=self.q {
                for c in 0..k {
                    let coef_i = lag * k + c;
                    let xv = if lag == 0 {
                        x_future.get(t, c)
                    } else if t >= lag {
                        x_future.get(t - lag, c)
                    } else {
                        let src = self.last_x.nrows().saturating_sub(lag - t);
                        if src < self.last_x.nrows() {
                            self.last_x.get(src, c)
                        } else {
                            0.0
                        }
                    };
                    if coef_i < self.exo.len() {
                        yhat += self.exo[coef_i] * xv;
                    }
                }
            }
            out[t] = yhat;
            hist_y.push(yhat);
        }
        ctx.finish(out)
    }
}

/// Unrestricted error-correction model (statsmodels `UECM`).
///
/// \(\Delta y_t\) on \([1, y_{t-1}, x_{t-1}, \Delta y_{t-1}, \Delta x_t]\).
/// Lag counts are not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct Uecm;

/// Fitted unrestricted ECM.
#[derive(Clone, Debug)]
pub struct FittedUecm {
    /// Intercept of \(\Delta y_t\).
    pub intercept: f64,
    /// Coefficient on the lagged level \(y_{t-1}\).
    pub alpha: f64,
    /// Coefficients on lagged exogenous levels.
    pub gamma: Vector,
    /// Coefficient on \(\Delta y_{t-1}\).
    pub phi: f64,
    /// Coefficients on contemporaneous \(\Delta x_t\).
    pub theta: Vector,
    /// Last observed \(y\).
    pub last_y: f64,
    /// Last observed exogenous row.
    pub last_x: Vector,
    /// Last observed \(\Delta y\).
    pub last_dy: f64,
}

impl Uecm {
    /// Default UECM.
    pub fn new() -> Self {
        Self
    }

    /// Fit \(\Delta y\) on lagged levels and first differences.
    pub fn fit(
        &mut self,
        y: &Vector,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedUecm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if y.len() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("UECM y length ≠ X rows")
                    .build(),
            );
        }
        let n = y.len().min(x.nrows());
        let k = x.ncols();
        if n < 4 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("UECM n={n} is too short for one difference lag"))
                    .build(),
            );
            return ctx.finish(FittedUecm {
                intercept: 0.0,
                alpha: 0.0,
                gamma: Vector::zeros(k),
                phi: 0.0,
                theta: Vector::zeros(k),
                last_y: y.as_slice().last().copied().unwrap_or(0.0),
                last_x: if n > 0 && k > 0 {
                    x.row(n - 1)
                } else {
                    Vector::zeros(k)
                },
                last_dy: if n >= 2 { y[n - 1] - y[n - 2] } else { 0.0 },
            });
        }
        let n_eff = n - 2;
        let cols = 3 + 2 * k;
        let design = Matrix::from_fn(n_eff, cols, |i, j| {
            let t = i + 2;
            if j == 0 {
                1.0
            } else if j == 1 {
                y[t - 1]
            } else if j < 2 + k {
                x.get(t - 1, j - 2)
            } else if j == 2 + k {
                y[t - 1] - y[t - 2]
            } else {
                let c = j - (3 + k);
                x.get(t, c) - x.get(t - 1, c)
            }
        });
        let yy = Vector::from_iter((2..n).map(|t| y[t] - y[t - 1]));
        let beta = statistical_ols(&mut ctx, &design, &yy).unwrap_or_else(|| {
            let mut b = Vector::zeros(cols);
            b[0] = yy.mean();
            b
        });
        ctx.finish(FittedUecm {
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            alpha: if beta.len() > 1 { beta[1] } else { 0.0 },
            gamma: Vector::from_iter((0..k).map(|c| {
                let idx = 2 + c;
                if idx < beta.len() {
                    beta[idx]
                } else {
                    0.0
                }
            })),
            phi: {
                let idx = 2 + k;
                if idx < beta.len() {
                    beta[idx]
                } else {
                    0.0
                }
            },
            theta: Vector::from_iter((0..k).map(|c| {
                let idx = 3 + k + c;
                if idx < beta.len() {
                    beta[idx]
                } else {
                    0.0
                }
            })),
            last_y: y[n - 1],
            last_x: x.row(n - 1),
            last_dy: y[n - 1] - y[n - 2],
        })
    }
}

impl FittedUecm {
    /// Iterate the ECM with a future exogenous path (`h × k`).
    pub fn forecast(&self, x_future: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        let k = self.gamma.len().max(self.theta.len());
        if x_future.ncols() != k {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message(format!(
                        "UECM forecast X has {} columns; fitted k={k}",
                        x_future.ncols()
                    ))
                    .build(),
            );
        }
        let h = x_future.nrows();
        let mut last_y = self.last_y;
        let mut last_x = self.last_x.as_slice().to_vec();
        if last_x.len() < k {
            last_x.resize(k, 0.0);
        }
        let mut last_dy = self.last_dy;
        let mut out = Vector::zeros(h);
        for t in 0..h {
            let mut dyhat = self.intercept + self.alpha * last_y + self.phi * last_dy;
            for c in 0..k {
                let xt = if c < x_future.ncols() {
                    x_future.get(t, c)
                } else {
                    0.0
                };
                let xlag = last_x.get(c).copied().unwrap_or(0.0);
                let dx = xt - xlag;
                if c < self.gamma.len() {
                    dyhat += self.gamma[c] * xlag;
                }
                if c < self.theta.len() {
                    dyhat += self.theta[c] * dx;
                }
                if c < last_x.len() {
                    last_x[c] = xt;
                }
            }
            last_y += dyhat;
            last_dy = dyhat;
            out[t] = last_y;
        }
        ctx.finish(out)
    }
}

/// Fourier seasonal features (sktime `FourierFeatures`): `sin/cos(2π k t / period)`.
///
/// Harmonic count is not an identification `p`.
#[derive(Clone, Debug)]
pub struct FourierFeatures {
    /// Seasonal period.
    pub period: usize,
    /// Number of harmonics `k = 1..n_harmonics`.
    pub n_harmonics: usize,
}

impl Default for FourierFeatures {
    fn default() -> Self {
        Self {
            period: 4,
            n_harmonics: 1,
        }
    }
}

impl FourierFeatures {
    /// `n_harmonics` pairs for the given period.
    pub fn new(period: usize, n_harmonics: usize) -> Self {
        Self {
            period: period.max(2),
            n_harmonics: n_harmonics.max(1),
        }
    }

    /// Map a length-`n` time index to Fourier columns.
    pub fn transform(&self, n: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        let p = self.period.max(2);
        let h = self.n_harmonics.max(1);
        let out = Matrix::from_fn(n, 2 * h, |t, j| {
            let k = j / 2 + 1;
            let ang = 2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / p as f64;
            if j % 2 == 0 {
                ang.sin()
            } else {
                ang.cos()
            }
        });
        ctx.finish(out)
    }
}

/// AIC grid over SES / Holt / Holt–Winters (sktime `AutoETS`).
#[derive(Clone, Debug)]
pub struct AutoEts {
    /// Seasonal period for the Holt–Winters candidate.
    pub period: usize,
}

impl Default for AutoEts {
    fn default() -> Self {
        Self { period: 4 }
    }
}

impl AutoEts {
    /// AutoETS with seasonal period `s` (used only if `n ≥ 2s`).
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(2),
        }
    }
}

/// Fitted AutoETS winner.
#[derive(Clone, Debug)]
pub struct FittedAutoEts {
    /// `"ses"`, `"holt"`, or `"hw"`.
    pub kind: &'static str,
    /// In-sample AIC.
    pub aic: f64,
    /// SES / Holt state (also used as a fallback for HW).
    pub esm: FittedEsm,
    /// Holt–Winters state when that candidate won.
    pub hw: Option<FittedHoltWinters>,
}

impl FittedAutoEts {
    /// `h`-step forecast of the selected model.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        if let Some(hw) = &self.hw {
            if self.kind == "hw" {
                return hw.forecast(h, session);
            }
        }
        self.esm.forecast(h, session)
    }
}

impl FitSeries for AutoEts {
    type Fitted = FittedAutoEts;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedAutoEts>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len() as f64;
        let aic_of = |sse: f64, k: f64| -> f64 {
            let s = (sse / n.max(1.0)).max(1e-12);
            n * s.ln() + 2.0 * k
        };
        let mut best_kind = "ses";
        let mut best_aic = f64::INFINITY;
        let mut best_esm = FittedEsm {
            kind: SmoothingKind::Simple,
            alpha: 0.3,
            beta: 0.0,
            level: y.as_slice().last().copied().unwrap_or(0.0),
            trend: 0.0,
            fitted: y.clone(),
        };
        let mut best_hw = None;
        for (kind, spec, k) in [
            (SmoothingKind::Simple, ExponentialSmoothing::simple(), 1.0),
            (SmoothingKind::Holt, ExponentialSmoothing::holt(), 2.0),
        ] {
            match spec
                .clone()
                .fit_series(y, &session.child(format!("{kind:?}")))
            {
                Ok(q) => {
                    let mut sse = 0.0;
                    for i in 0..y.len().min(q.value.fitted.len()) {
                        let e = y[i] - q.value.fitted[i];
                        sse += e * e;
                    }
                    let aic = aic_of(sse, k);
                    if aic < best_aic {
                        best_aic = aic;
                        best_kind = if matches!(kind, SmoothingKind::Simple) {
                            "ses"
                        } else {
                            "holt"
                        };
                        best_esm = q.value;
                    }
                }
                Err(_) => {}
            }
        }
        if y.len() >= 2 * self.period {
            match HoltWinters::new(self.period).fit_series(y, &session.child("hw")) {
                Ok(q) => {
                    let mut sse = 0.0;
                    for i in 0..y.len().min(q.value.fitted.len()) {
                        let e = y[i] - q.value.fitted[i];
                        sse += e * e;
                    }
                    let aic = aic_of(sse, 3.0);
                    if aic < best_aic {
                        best_aic = aic;
                        best_kind = "hw";
                        best_hw = Some(q.value);
                    }
                }
                Err(_) => {}
            }
        }
        if !best_aic.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("AutoETS found no finite AIC candidate")
                    .build(),
            );
        }
        ctx.finish(FittedAutoEts {
            kind: best_kind,
            aic: best_aic,
            esm: best_esm,
            hw: best_hw,
        })
    }
}

/// STL + seasonal-naive residual forecast (sktime `STLForecaster`).
#[derive(Clone, Debug)]
pub struct StlForecaster {
    /// Seasonal period.
    pub period: usize,
}

impl Default for StlForecaster {
    fn default() -> Self {
        Self { period: 4 }
    }
}

impl StlForecaster {
    /// STL forecaster with period `s`.
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(2),
        }
    }
}

/// Fitted STL forecaster.
#[derive(Clone, Debug)]
pub struct FittedStlForecaster {
    /// In-sample decomposition.
    pub decomp: SeasonalDecomposition,
    /// Last residual (used as a level offset).
    pub last_resid: f64,
}

impl FittedStlForecaster {
    /// `h`-step: last trend + seasonal cycle + last residual.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        let n = self.decomp.observed.len();
        let period = self.decomp.period.max(2);
        let last_trend = self.decomp.trend.as_slice().last().copied().unwrap_or(0.0);
        let y = Vector::from_iter((0..h).map(|s| {
            let idx = (n + s) % period;
            let seas = self
                .decomp
                .seasonal
                .as_slice()
                .get(idx)
                .copied()
                .unwrap_or(0.0);
            last_trend + seas + self.last_resid
        }));
        ctx.finish(y)
    }
}

impl FitSeries for StlForecaster {
    type Fitted = FittedStlForecaster;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedStlForecaster>> {
        let q = Stl::new(self.period).fit_series(y, session)?;
        let last_resid = q.value.resid.as_slice().last().copied().unwrap_or(0.0);
        Ok(q.map(|decomp| FittedStlForecaster { decomp, last_resid }))
    }
}

/// Box–Cox transformer (sktime `BoxCoxTransformer`).
///
/// `λ` is chosen by a coarse Gaussian-likelihood grid. Non-positive `y` is
/// [`IssueCode::NonPositiveSeries`].
#[derive(Clone, Debug, Default)]
pub struct BoxCoxTransformer {
    /// Selected power. `None` until fit.
    pub lambda: Option<f64>,
}

impl BoxCoxTransformer {
    /// Empty transformer.
    pub fn new() -> Self {
        Self::default()
    }

    fn apply(v: f64, lam: f64) -> f64 {
        let x = v.max(1e-12);
        if lam.abs() < 1e-12 {
            x.ln()
        } else {
            (x.powf(lam) - 1.0) / lam
        }
    }
}

impl FitSeries for BoxCoxTransformer {
    type Fitted = Self;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.as_slice().iter().any(|&v| v <= 0.0) {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .message("Box–Cox requires a strictly positive series")
                    .build(),
            );
        }
        let mut best_l = 0.0;
        let mut best_ll = f64::NEG_INFINITY;
        for i in 0..=8 {
            let lam = -1.0 + 0.5 * i as f64;
            let z: Vec<f64> = y.as_slice().iter().map(|&v| Self::apply(v, lam)).collect();
            let n = z.len() as f64;
            let m = z.iter().sum::<f64>() / n.max(1.0);
            let mut sse = 0.0;
            for &v in &z {
                let e = v - m;
                sse += e * e;
            }
            let s2 = (sse / n.max(1.0)).max(1e-12);
            let ll = -0.5 * n * s2.ln();
            if ll > best_ll {
                best_ll = ll;
                best_l = lam;
            }
        }
        self.lambda = Some(best_l);
        ctx.finish(self.clone())
    }
}

impl BoxCoxTransformer {
    /// Apply the fitted power map.
    pub fn transform(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        let lam = self.lambda.unwrap_or(0.0);
        if self.lambda.is_none() {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
        }
        ctx.finish(Vector::from_iter(
            y.as_slice().iter().map(|&v| Self::apply(v, lam)),
        ))
    }
}

/// First-difference transformer (sktime `Differencer`).
#[derive(Clone, Debug, Default)]
pub struct Differencer {
    last: f64,
    fitted: bool,
}

impl Differencer {
    /// Empty differencer.
    pub fn new() -> Self {
        Self::default()
    }

    /// `Δy_t = y_t − y_{t−1}` (`y_0` is dropped as 0).
    pub fn transform(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
        }
        let z = Vector::from_iter((0..y.len()).map(|i| if i == 0 { 0.0 } else { y[i] - y[i - 1] }));
        ctx.finish(z)
    }
}

impl FitSeries for Differencer {
    type Fitted = Self;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        self.last = y.as_slice().last().copied().unwrap_or(0.0);
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

/// Calendar-like features from an integer time index (sktime `DateTimeFeatures` lite).
///
/// Harmonic / period counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct DateTimeFeatures {
    /// Seasonal period used for `t mod period` and Fourier pair.
    pub period: usize,
}

impl Default for DateTimeFeatures {
    fn default() -> Self {
        Self { period: 7 }
    }
}

impl DateTimeFeatures {
    /// Features for period `s`.
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(2),
        }
    }

    /// Map `t = 0..n-1` to `[t, t mod s, sin, cos]`.
    pub fn transform(&self, n: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        let p = self.period.max(2);
        let out = Matrix::from_fn(n, 4, |t, j| {
            let tf = t as f64;
            match j {
                0 => tf,
                1 => (t % p) as f64,
                2 => (2.0 * std::f64::consts::PI * tf / p as f64).sin(),
                _ => (2.0 * std::f64::consts::PI * tf / p as f64).cos(),
            }
        });
        ctx.finish(out)
    }
}

/// Holiday dummy from an integer index (sktime `HolidayFeatures` lite).
///
/// A 1 is placed every `period` steps. Period is not identification `p`.
#[derive(Clone, Debug)]
pub struct HolidayFeatures {
    /// Holiday spacing.
    pub period: usize,
}

impl Default for HolidayFeatures {
    fn default() -> Self {
        Self { period: 7 }
    }
}

impl HolidayFeatures {
    /// Holiday every `period` steps.
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(1),
        }
    }

    /// Map `t = 0..n-1` to a single holiday column.
    pub fn transform(&self, n: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let ctx = FitCtx::with_session(session.child("transform"));
        let p = self.period.max(1);
        ctx.finish(Matrix::from_fn(n, 1, |t, _| {
            if t % p == 0 {
                1.0
            } else {
                0.0
            }
        }))
    }
}

/// Deterministic trend / seasonal design (statsmodels `DeterministicProcess`).
///
/// Seasonal dummy count is not identification `p`.
#[derive(Clone, Debug)]
pub struct DeterministicProcess {
    /// Include a constant column.
    pub constant: bool,
    /// Include a linear trend.
    pub trend: bool,
    /// Seasonal period (`0` or `1` skips dummies).
    pub period: usize,
}

impl Default for DeterministicProcess {
    fn default() -> Self {
        Self {
            constant: true,
            trend: true,
            period: 0,
        }
    }
}

impl DeterministicProcess {
    /// Constant + trend, no seasonal dummies.
    pub fn new() -> Self {
        Self::default()
    }

    /// Constant + trend + `period` seasonal dummies (drop-last).
    pub fn seasonal(period: usize) -> Self {
        Self {
            constant: true,
            trend: true,
            period,
        }
    }

    /// Design matrix for `t = 0..n-1`.
    pub fn transform(&self, n: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let ctx = FitCtx::with_session(session.child("transform"));
        let seas = self.period.max(0).saturating_sub(1);
        let p = usize::from(self.constant) + usize::from(self.trend) + seas;
        if p == 0 {
            return ctx.finish(Matrix::zeros(n, 0));
        }
        let out = Matrix::from_fn(n, p, |t, j| {
            let mut col = 0usize;
            if self.constant {
                if j == col {
                    return 1.0;
                }
                col += 1;
            }
            if self.trend {
                if j == col {
                    return t as f64;
                }
                col += 1;
            }
            if seas > 0 {
                let s = j - col;
                return if (t % self.period.max(1)) == s { 1.0 } else { 0.0 };
            }
            0.0
        });
        ctx.finish(out)
    }
}

/// Natural-log map (sktime `LogTransformer`).
///
/// Non-positive samples are clamped and recorded; [`IssueCode::NonPositiveSeries`]
/// is lowered to a warning so a mixed series still transforms.
#[derive(Clone, Debug, Default)]
pub struct LogTransformer;

impl LogTransformer {
    /// Default log map.
    pub fn new() -> Self {
        Self
    }

    /// \(\log(\max(y, \varepsilon))\).
    pub fn transform(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_univariate(&mut ctx, y);
        let mut clamped = 0usize;
        let z = Vector::from_iter(y.as_slice().iter().map(|&v| {
            if v.is_finite() && v > 0.0 {
                v.ln()
            } else {
                clamped += 1;
                1e-12_f64.ln()
            }
        }));
        if clamped > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(Severity::Warning)
                    .message(format!(
                        "LogTransformer clamped {clamped} non-positive samples to ε"
                    ))
                    .compromise(NumericalCompromise::new(
                        "strictly positive series",
                        "ln(max(y, 1e-12))",
                        "the log is undefined at 0 and on the negatives",
                        "do not treat clamped logs as observations of the original scale",
                    ))
                    .build(),
            );
        }
        ctx.finish(z)
    }
}

/// Time-since-origin features (sktime `TimeSince`).
///
/// Origin is not identification `p`.
#[derive(Clone, Debug)]
pub struct TimeSince {
    /// Index treated as time 0.
    pub origin: usize,
}

impl Default for TimeSince {
    fn default() -> Self {
        Self { origin: 0 }
    }
}

impl TimeSince {
    /// Features relative to `origin`.
    pub fn new(origin: usize) -> Self {
        Self { origin }
    }

    /// Map `t = 0..n-1` to `[t−origin, (t−origin)₊]`.
    pub fn transform(&self, n: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let ctx = FitCtx::with_session(session.child("transform"));
        let o = self.origin as f64;
        let out = Matrix::from_fn(n, 2, |t, j| {
            let d = t as f64 - o;
            if j == 0 {
                d
            } else {
                d.max(0.0)
            }
        });
        ctx.finish(out)
    }
}

/// One univariate forecaster per column (sktime `ColumnEnsembleForecaster`).
#[derive(Clone, Debug)]
pub struct ColumnEnsembleForecaster {
    /// AutoReg lag used on each column.
    pub lags: usize,
}

impl Default for ColumnEnsembleForecaster {
    fn default() -> Self {
        Self { lags: 1 }
    }
}

impl ColumnEnsembleForecaster {
    /// Ensemble with AR(`lags`) per column.
    pub fn new(lags: usize) -> Self {
        Self { lags: lags.max(1) }
    }

    /// Fit an [`AutoReg`] on each column of `y`.
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedColumnEnsemble>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let mut models = Vec::with_capacity(y.ncols());
        for j in 0..y.ncols() {
            let col = y.column(j);
            match AutoReg::new(self.lags).fit_series(&col, &session.child(format!("col_{j}"))) {
                Ok(q) => models.push(q.value),
                Err(e) => {
                    ctx.push(e.primary);
                    models.push(FittedAutoReg {
                        intercept: col.mean(),
                        ar: Vector::zeros(self.lags),
                        last: Vector::from_iter(
                            col.as_slice()
                                .iter()
                                .copied()
                                .rev()
                                .take(self.lags)
                                .collect::<Vec<_>>()
                                .into_iter()
                                .rev(),
                        ),
                    });
                }
            }
        }
        ctx.finish(FittedColumnEnsemble { models })
    }
}

/// One last-value walk per hierarchy level / column (sktime `ForecastByLevel`).
///
/// Column count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct ForecastByLevel;

/// Fitted per-column last-value forecasts.
#[derive(Clone, Debug)]
pub struct FittedForecastByLevel {
    /// Last observed value of each column.
    pub last: Vector,
}

impl ForecastByLevel {
    /// Default level-wise naive ensemble.
    pub fn new() -> Self {
        Self
    }

    /// Fit a last-value walker on each column of `y`.
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedForecastByLevel>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        if y.nrows() == 0 || y.ncols() == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .severity(Severity::Warning)
                    .message("ForecastByLevel received an empty panel")
                    .build(),
            );
            return ctx.finish(FittedForecastByLevel {
                last: Vector::zeros(y.ncols()),
            });
        }
        ctx.finish(FittedForecastByLevel {
            last: Vector::from_iter((0..y.ncols()).map(|j| {
                let col = y.column(j);
                col.as_slice().last().copied().unwrap_or(0.0)
            })),
        })
    }
}

impl FittedForecastByLevel {
    /// Repeat each column's last value for `h` horizons.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Matrix::from_fn(h, self.last.len(), |_, j| self.last[j]))
    }
}

/// Fitted per-column AutoReg ensemble.
#[derive(Clone, Debug)]
pub struct FittedColumnEnsemble {
    /// One AutoReg per column.
    pub models: Vec<FittedAutoReg>,
}

impl FittedColumnEnsemble {
    /// Forecast `h` steps for every column.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        let k = self.models.len();
        let mut out = Matrix::zeros(h, k);
        for (j, m) in self.models.iter().enumerate() {
            match m.forecast(h, &session.child(format!("fc_{j}"))) {
                Ok(q) => {
                    for t in 0..h.min(q.value.len()) {
                        out.set(t, j, q.value[t]);
                    }
                }
                Err(e) => ctx.push(e.primary),
            }
        }
        ctx.finish(out)
    }
}

/// Cross-correlation `γ_{xy}(k)` for `k = 0..nlags` (statsmodels `ccf`).
pub fn ccf(x: &Vector, y: &Vector, nlags: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, x);
    inspect_univariate(&mut ctx, y);
    let n = x.len().min(y.len());
    if n == 0 {
        return ctx.finish(Vector::zeros(nlags + 1));
    }
    let mx = x.mean();
    let my = y.mean();
    let mut sx: f64 = 0.0;
    let mut sy: f64 = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sx += dx * dx;
        sy += dy * dy;
    }
    let den = (sx * sy).sqrt().max(1e-12);
    let out = Vector::from_iter((0..=nlags).map(|lag| {
        let mut s = 0.0;
        for i in lag..n {
            s += (x[i] - mx) * (y[i - lag] - my);
        }
        s / den
    }));
    ctx.finish(out)
}

/// Periodogram `I(ω)` at `n/2` Fourier frequencies (statsmodels `periodogram`).
pub fn periodogram(y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let n = y.len();
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message("periodogram needs n≥2")
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let m = n / 2;
    let two_pi = 2.0 * std::f64::consts::PI;
    let out = Vector::from_iter((1..=m).map(|k| {
        let mut re: f64 = 0.0;
        let mut im: f64 = 0.0;
        for t in 0..n {
            let ang = two_pi * k as f64 * t as f64 / n as f64;
            re += y[t] * ang.cos();
            im += y[t] * ang.sin();
        }
        (re * re + im * im) / n as f64
    }));
    ctx.finish(out)
}

/// Welch averaged periodogram (statsmodels `signal.spectral.welch` / SciPy).
///
/// Segment length is not identification `p`. A Hann taper is applied; a
/// single segment is recorded as spectral leakage.
pub fn welch(y: &Vector, nperseg: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let n = y.len();
    let mut seg = nperseg;
    if seg < 2 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("welch nperseg={nperseg} < 2; using 8"))
                .build(),
        );
        seg = 8;
    }
    if n < 4 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("welch needs n≥4; got {n}"))
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    if seg > n {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("welch nperseg={seg} > n={n}; using n"))
                .build(),
        );
        seg = n;
    }
    let step = (seg / 2).max(1);
    let n_seg = if n >= seg { (n - seg) / step + 1 } else { 1 };
    if n_seg <= 1 {
        ctx.push(
            Issue::builder(IssueCode::SpectralLeakage)
                .message("welch used a single segment; the average is one tapered periodogram")
                .compromise(NumericalCompromise::new(
                    "overlapped Welch average",
                    "one Hann-tapered periodogram",
                    "n is too short for a second hop",
                    "do not read a single-segment Welch as a variance-reduced spectrum",
                ))
                .build(),
        );
    }
    let m = (seg / 2).max(1);
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut acc = vec![0.0; m];
    let mut used: f64 = 0.0;
    for s in 0..n_seg {
        let start = s * step;
        if start + seg > n {
            break;
        }
        let mut wsum = 0.0;
        let mut win = vec![0.0; seg];
        for t in 0..seg {
            let w = if seg == 1 {
                1.0
            } else {
                0.5 * (1.0 - (two_pi * t as f64 / (seg - 1) as f64).cos())
            };
            win[t] = w * y[start + t];
            wsum += w * w;
        }
        let den = wsum.max(1e-18);
        for k in 1..=m {
            let mut re: f64 = 0.0;
            let mut im: f64 = 0.0;
            for t in 0..seg {
                let ang = two_pi * k as f64 * t as f64 / seg as f64;
                re += win[t] * ang.cos();
                im += win[t] * ang.sin();
            }
            acc[k - 1] += (re * re + im * im) / den;
        }
        used += 1.0;
    }
    if used <= 0.0 {
        return ctx.finish(Vector::zeros(0));
    }
    ctx.finish(Vector::from_iter(acc.iter().map(|v| *v / used)))
}

/// AIC / BIC grid over ARMA(\(p,q\)) = ARIMA(\(p,0,q\)) (statsmodels
/// `arma_order_select_ic`).
///
/// Order bounds are not identification `p`. Inner Hannan–Rissanen residuals
/// that would abort a standalone ARIMA are skipped.
#[derive(Clone, Debug)]
pub struct ArmaOrderSelect {
    /// Winning AR order.
    pub p: usize,
    /// Winning MA order.
    pub q: usize,
    /// AIC of the winner.
    pub aic: f64,
    /// BIC of the winner.
    pub bic: f64,
    /// `(p, q, aic, bic)` for every successful grid point.
    pub scores: Vec<(usize, usize, f64, f64)>,
}

/// Select ARMA(\(p,q\)) by AIC on a Hannan–Rissanen grid with \(d=0\).
pub fn arma_order_select_ic(
    y: &Vector,
    max_p: usize,
    max_q: usize,
    session: &Session,
) -> Result<Qualified<ArmaOrderSelect>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let pmax = max_p.min(4);
    let qmax = max_q.min(4);
    if max_p > 4 || max_q > 4 {
        ctx.push(
            Issue::builder(IssueCode::Overparameterized)
                .severity(Severity::Advisory)
                .message(format!(
                    "arma_order_select_ic clamped max_p={max_p} max_q={max_q} to 4"
                ))
                .build(),
        );
    }
    let mut scores = Vec::new();
    let mut best: Option<(f64, usize, usize, f64)> = None;
    for p in 0..=pmax {
        for q in 0..=qmax {
            let mut spec = Arima { p, d: 0, q };
            match spec.fit_series(y, &session.child(format!("arma_{p}0{q}"))) {
                Ok(fit) => {
                    for issue in fit.report.issues() {
                        if matches!(
                            issue.code,
                            IssueCode::ResidualTooLarge
                                | IssueCode::NearSingular
                                | IssueCode::RankZero
                                | IssueCode::R2IsOne
                                | IssueCode::ShortSeriesForArima
                                | IssueCode::PerfectCollinearity
                        ) {
                            continue;
                        }
                        ctx.push(issue.clone());
                    }
                    if !fit.value.sigma2.is_finite() {
                        continue;
                    }
                    let n = fit.value.resid.len().max(1) as f64;
                    let k = (1 + p + q) as f64;
                    let s2 = fit.value.sigma2.max(1e-18);
                    let aic = n * s2.ln() + 2.0 * k;
                    let bic = n * s2.ln() + k * n.ln();
                    scores.push((p, q, aic, bic));
                    match &best {
                        Some((b, _, _, _)) if aic >= *b => {}
                        _ => best = Some((aic, p, q, bic)),
                    }
                }
                Err(e) => {
                    if !matches!(
                        e.primary.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::RankZero
                            | IssueCode::R2IsOne
                            | IssueCode::ShortSeriesForArima
                    ) {
                        ctx.push(
                            Issue::builder(IssueCode::DidNotConverge)
                                .severity(Severity::Advisory)
                                .message(format!("ARMA({p},{q}) rejected: {}", e.primary().code))
                                .build(),
                        );
                    }
                }
            }
        }
    }
    ctx.push(
        Issue::builder(IssueCode::Overparameterized)
            .severity(Severity::Advisory)
            .message("arma_order_select_ic AIC/BIC use Hannan–Rissanen σ², not the exact Gaussian likelihood")
            .compromise(NumericalCompromise::new(
                "exact-likelihood ARMA order selection",
                "Hannan–Rissanen OLS grid + n ln σ² + 2k / k ln n",
                "failed orders and residual-too-large trials are skipped",
                "the selected (p,q) is a relative AIC winner on this grid only",
            ))
            .build(),
    );
    match best {
        Some((aic, p, q, bic)) => ctx.finish(ArmaOrderSelect {
            p,
            q,
            aic,
            bic,
            scores,
        }),
        None => {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("arma_order_select_ic found no identified ARMA on the grid")
                    .build(),
            );
            ctx.finish(ArmaOrderSelect {
                p: 0,
                q: 0,
                aic: f64::NAN,
                bic: f64::NAN,
                scores,
            })
        }
    }
}

/// Exogenous-aware reduction (sktime `ForecastX`).
///
/// \(y_t \sim y_{t-1},\ldots,y_{t-p}, x_t\). Lag and exogenous counts are not
/// passed as identification `p`.
#[derive(Clone, Debug)]
pub struct ForecastX {
    /// Autoregressive window.
    pub window: usize,
}

impl Default for ForecastX {
    fn default() -> Self {
        Self { window: 2 }
    }
}

impl ForecastX {
    /// ForecastX with lag window `window`.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
        }
    }

    /// Fit on a series and aligned exogenous matrix.
    pub fn fit(
        &mut self,
        y: &Vector,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedForecastX>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let p = self.window.max(1);
        if y.len() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!(
                        "ForecastX y.len()={} x.nrows()={}",
                        y.len(),
                        x.nrows()
                    ))
                    .build(),
            );
        }
        let n = y.len().min(x.nrows());
        if n <= p {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("ForecastX window {p} needs n>{p} (n={n})"))
                    .meaninglessness(Meaninglessness::vacuous(
                        "exogenous reduction",
                        "n ≤ p leaves no regression rows",
                        "lengthen the series or shorten the window",
                    ))
                    .build(),
            );
            return ctx.finish(FittedForecastX {
                coef_lags: Vector::zeros(p),
                coef_x: Vector::zeros(x.ncols()),
                intercept: y.mean(),
                last: y.clone(),
                last_x: if x.nrows() > 0 {
                    Vector::from_iter((0..x.ncols()).map(|j| x.get(x.nrows() - 1, j)))
                } else {
                    Vector::zeros(x.ncols())
                },
                window: p,
            });
        }
        let q = x.ncols();
        let m = n - p;
        let design = Matrix::from_fn(m, 1 + p + q, |i, j| {
            if j == 0 {
                1.0
            } else if j <= p {
                y[i + j - 1]
            } else {
                x.get(i + p, j - 1 - p)
            }
        });
        let target = Vector::from_iter((p..n).map(|t| y[t]));
        let mut scratch = Report::new("forecastx", "ols");
        let beta = crate::linalg::least_squares(&mut scratch, &design, &target, &ctx.policy);
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let b = beta.unwrap_or_else(|| Vector::zeros(design.ncols()));
        let intercept = b.as_slice().first().copied().unwrap_or(0.0);
        let coef_lags =
            Vector::from_iter((0..p).map(|j| if 1 + j < b.len() { b[1 + j] } else { 0.0 }));
        let coef_x = Vector::from_iter((0..q).map(|j| {
            let idx = 1 + p + j;
            if idx < b.len() {
                b[idx]
            } else {
                0.0
            }
        }));
        ctx.finish(FittedForecastX {
            coef_lags,
            coef_x,
            intercept,
            last: Vector::from_iter(y.as_slice()[n - p..n].iter().copied()),
            last_x: Vector::from_iter((0..q).map(|j| x.get(n - 1, j))),
            window: p,
        })
    }
}

/// Fitted exogenous reducer.
#[derive(Clone, Debug)]
pub struct FittedForecastX {
    /// Lag coefficients (oldest first).
    pub coef_lags: Vector,
    /// Exogenous coefficients.
    pub coef_x: Vector,
    /// Intercept.
    pub intercept: f64,
    last: Vector,
    last_x: Vector,
    /// Lag order.
    pub window: usize,
}

impl FittedForecastX {
    /// `h`-step forecast using future exogenous rows (recycled last `x` if short).
    pub fn forecast(
        &self,
        horizon: usize,
        x_future: &Matrix,
        session: &Session,
    ) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        if horizon == 0 {
            return ctx.finish(Vector::zeros(0));
        }
        if horizon > self.window.saturating_mul(4).max(8) {
            ctx.push(
                Issue::builder(IssueCode::ForecastHorizonExceedsIdentifiability)
                    .message(format!(
                        "ForecastX horizon {horizon} ≫ window {}",
                        self.window
                    ))
                    .build(),
            );
        }
        let mut hist = self.last.as_slice().to_vec();
        let mut out = Vector::zeros(horizon);
        for h in 0..horizon {
            let mut s = self.intercept;
            let start = hist.len().saturating_sub(self.window);
            for j in 0..self.coef_lags.len().min(self.window) {
                if start + j < hist.len() {
                    s += self.coef_lags[j] * hist[start + j];
                }
            }
            for j in 0..self.coef_x.len() {
                let xv = if h < x_future.nrows() && j < x_future.ncols() {
                    x_future.get(h, j)
                } else if j < self.last_x.len() {
                    self.last_x[j]
                } else {
                    0.0
                };
                s += self.coef_x[j] * xv;
            }
            out[h] = s;
            hist.push(s);
        }
        ctx.finish(out)
    }
}

/// OLS hierarchical reconciliation (sktime `reconcile.Reconciler`).
///
/// `yhat` is `h × m` base forecasts. `summing` is the `m × b` summing
/// matrix (`S`) from `b` bottom series to `m` nodes. The projection
/// \(\hat Y S(S^\top S)^{-1}S^\top\) is applied independently at each
/// horizon. Node / bottom counts are not identification `p`.
pub fn reconcile_ols(
    yhat: &Matrix,
    summing: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, yhat, None, &ctx.policy);
    inspect_xy(&mut ctx.report, summing, None, &ctx.policy);
    if yhat.ncols() != summing.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "reconcile_ols yhat.ncols()={} summing.nrows()={}",
                    yhat.ncols(),
                    summing.nrows()
                ))
                .build(),
        );
        return ctx.finish(yhat.clone());
    }
    let m = summing.nrows();
    let b = summing.ncols();
    if m == 0 || b == 0 {
        return ctx.finish(yhat.clone());
    }
    let mut gram = summing.gram();
    for i in 0..b {
        gram[(i, i)] += 1e-10;
    }
    let mut out = Matrix::zeros(yhat.nrows(), m);
    let mut used_fallback = false;
    for h in 0..yhat.nrows() {
        let mut z = Vector::zeros(b);
        for j in 0..b {
            let mut s = 0.0;
            for i in 0..m {
                s += summing.get(i, j) * yhat.get(h, i);
            }
            z[j] = s;
        }
        let mut scratch = Report::new("reconcile", "chol");
        let beta = match chol_solve(&mut scratch, &gram, &z, &ctx.policy) {
            Some(v) => v,
            None => {
                used_fallback = true;
                Vector::from_iter((0..b).map(|j| {
                    if j < yhat.ncols() {
                        yhat.get(h, j)
                    } else {
                        0.0
                    }
                }))
            }
        };
        for i in 0..m {
            let mut s = 0.0;
            for j in 0..b.min(beta.len()) {
                s += summing.get(i, j) * beta[j];
            }
            out.set(h, i, s);
        }
    }
    if used_fallback {
        ctx.push(
            Issue::builder(IssueCode::CholeskyFailed)
                .severity(Severity::Warning)
                .message("reconcile_ols: S'S was not SPD; a horizon fell back to a truncated map")
                .compromise(NumericalCompromise::new(
                    "Cholesky of S'S",
                    "truncated bottom coefficients for that horizon",
                    "the summing Gram was indefinite even after jitter",
                    "do not treat that horizon as a coherent hierarchy",
                ))
                .build(),
        );
    }
    ctx.finish(out)
}

/// MinT-style weighted reconciliation (sktime `Reconciler` `mint`).
///
/// Bottom-level weights are the inverse of the summing-matrix column sums.
/// Node counts are not identification `p`.
pub fn reconcile_mint(
    yhat: &Matrix,
    summing: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, yhat, None, &ctx.policy);
    inspect_xy(&mut ctx.report, summing, None, &ctx.policy);
    if yhat.ncols() != summing.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "reconcile_mint yhat.ncols()={} summing.nrows()={}",
                    yhat.ncols(),
                    summing.nrows()
                ))
                .build(),
        );
        return ctx.finish(yhat.clone());
    }
    let m = summing.nrows();
    let b = summing.ncols();
    if m == 0 || b == 0 {
        return ctx.finish(yhat.clone());
    }
    let mut w = vec![0.0; b];
    for j in 0..b {
        let mut s = 0.0;
        for i in 0..m {
            s += summing.get(i, j).abs();
        }
        w[j] = if s > 1e-12 { 1.0 / s } else { 1.0 };
    }
    let mut gram = summing.gram();
    for i in 0..b {
        for j in 0..b {
            let mut g = 0.0;
            for k in 0..m {
                g += summing.get(k, i) * w[i] * summing.get(k, j);
            }
            gram[(i, j)] = g;
        }
        gram[(i, i)] += 1e-10;
    }
    let mut out = Matrix::zeros(yhat.nrows(), m);
    for h in 0..yhat.nrows() {
        let mut z = Vector::zeros(b);
        for j in 0..b {
            let mut s = 0.0;
            for i in 0..m {
                s += summing.get(i, j) * w[j] * yhat.get(h, i);
            }
            z[j] = s;
        }
        let mut scratch = Report::new("mint", "chol");
        let beta = match chol_solve(&mut scratch, &gram, &z, &ctx.policy) {
            Some(v) => v,
            None => {
                ctx.push(
                    Issue::builder(IssueCode::CholeskyFailed)
                        .severity(Severity::Warning)
                        .message("MinT Cholesky failed; leaving the base forecast")
                        .build(),
                );
                for i in 0..m {
                    out.set(h, i, yhat.get(h, i));
                }
                continue;
            }
        };
        for i in 0..m {
            let mut s = 0.0;
            for j in 0..b.min(beta.len()) {
                s += summing.get(i, j) * beta[j];
            }
            out.set(h, i, s);
        }
    }
    ctx.finish(out)
}

fn reconcile_bottoms(yhat: &Matrix, summing: &Matrix) -> Matrix {
    let b = summing.ncols();
    let m = summing.nrows();
    let h = yhat.nrows();
    let mut bottoms = Matrix::zeros(h, b);
    for j in 0..b {
        let mut best_i = j.min(yhat.ncols().saturating_sub(1));
        let mut best_score = f64::INFINITY;
        for i in 0..m {
            let mut s = 0.0_f64;
            for jj in 0..b {
                let target = if jj == j { 1.0 } else { 0.0 };
                let e = summing.get(i, jj) - target;
                s += e * e;
            }
            if s < best_score {
                best_score = s;
                best_i = i;
            }
        }
        if best_i < yhat.ncols() {
            for t in 0..h {
                bottoms.set(t, j, yhat.get(t, best_i));
            }
        }
    }
    bottoms
}

fn apply_summing(bottoms: &Matrix, summing: &Matrix) -> Matrix {
    let h = bottoms.nrows();
    let m = summing.nrows();
    let b = summing.ncols();
    Matrix::from_fn(h, m, |t, i| {
        let mut s = 0.0_f64;
        for j in 0..b {
            s += summing.get(i, j) * bottoms.get(t, j);
        }
        s
    })
}

/// Bottom-up hierarchical reconciliation (sktime `Reconciler` `bu`).
///
/// Bottom nodes are the summing-matrix rows closest to the standard basis.
/// Node / bottom counts are not identification `p`.
pub fn reconcile_bottom_up(
    yhat: &Matrix,
    summing: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, yhat, None, &ctx.policy);
    inspect_xy(&mut ctx.report, summing, None, &ctx.policy);
    if yhat.ncols() != summing.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .severity(Severity::Warning)
                .message(format!(
                    "reconcile_bottom_up yhat.ncols()={} summing.nrows()={}",
                    yhat.ncols(),
                    summing.nrows()
                ))
                .build(),
        );
        return ctx.finish(yhat.clone());
    }
    if summing.nrows() == 0 || summing.ncols() == 0 {
        return ctx.finish(yhat.clone());
    }
    let bottoms = reconcile_bottoms(yhat, summing);
    ctx.finish(apply_summing(&bottoms, summing))
}

/// Top-down hierarchical reconciliation (sktime `Reconciler` `td`).
///
/// The most aggregated node (largest summing-matrix row sum) is distributed
/// by that row's shares. Node counts are not identification `p`.
pub fn reconcile_top_down(
    yhat: &Matrix,
    summing: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, yhat, None, &ctx.policy);
    inspect_xy(&mut ctx.report, summing, None, &ctx.policy);
    if yhat.ncols() != summing.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .severity(Severity::Warning)
                .message(format!(
                    "reconcile_top_down yhat.ncols()={} summing.nrows()={}",
                    yhat.ncols(),
                    summing.nrows()
                ))
                .build(),
        );
        return ctx.finish(yhat.clone());
    }
    let m = summing.nrows();
    let b = summing.ncols();
    if m == 0 || b == 0 {
        return ctx.finish(yhat.clone());
    }
    let mut top = 0usize;
    let mut top_sum = f64::NEG_INFINITY;
    for i in 0..m {
        let mut s = 0.0_f64;
        for j in 0..b {
            s += summing.get(i, j).abs();
        }
        if s > top_sum {
            top_sum = s;
            top = i;
        }
    }
    let denom = if top_sum > 1e-12 { top_sum } else { 1.0 };
    let mut bottoms = Matrix::zeros(yhat.nrows(), b);
    for t in 0..yhat.nrows() {
        let yt = if top < yhat.ncols() {
            yhat.get(t, top)
        } else {
            0.0
        };
        for j in 0..b {
            bottoms.set(t, j, yt * summing.get(top, j).abs() / denom);
        }
    }
    ctx.finish(apply_summing(&bottoms, summing))
}

/// Innovations-algorithm MA coefficients and innovation variances
/// (statsmodels `tsa.innovations.arma_innovations`).
///
/// `gamma` is the autocovariance sequence \(\gamma_0,\ldots,\gamma_{m-1}\).
/// Lag count is not identification `p`.
#[derive(Clone, Debug)]
pub struct Innovations {
    /// \(\theta_{n,j}\) stored as row `n`, column `j` (0-based, \(\theta_{n,0}\) unused).
    pub theta: Matrix,
    /// Innovation variances \(v_0,\ldots,v_{m-1}\).
    pub variance: Vector,
}

/// Brockwell–Davis innovations algorithm on a covariance sequence.
pub fn innovations_algo(gamma: &Vector, session: &Session) -> Result<Qualified<Innovations>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, gamma);
    let m = gamma.len();
    if m == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("innovations_algo received an empty acovf")
                .build(),
        );
        return ctx.finish(Innovations {
            theta: Matrix::zeros(0, 0),
            variance: Vector::zeros(0),
        });
    }
    if gamma[0] <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("innovations_algo γ₀ vanished; variances are set to 0")
                .build(),
        );
    }
    let mut theta = Matrix::zeros(m, m);
    let mut v = Vector::zeros(m);
    v[0] = gamma[0].max(0.0);
    for n in 1..m {
        for k in 0..n {
            let mut s = if n >= k { gamma[n - k] } else { 0.0 };
            for j in 0..k {
                s -= theta.get(k, k - j) * theta.get(n, n - j) * v[j];
            }
            let tnk = if v[k].abs() > 1e-18 { s / v[k] } else { 0.0 };
            theta.set(n, n - k, tnk);
        }
        let mut vn = gamma[0];
        for j in 0..n {
            let t = theta.get(n, n - j);
            vn -= t * t * v[j];
        }
        if vn < -1e-8 {
            ctx.push(
                Issue::builder(IssueCode::InvertibilityViolated)
                    .message(format!("innovations v_{n}={vn:.3e} went negative"))
                    .build(),
            );
        }
        v[n] = vn.max(0.0);
    }
    ctx.finish(Innovations { theta, variance: v })
}

/// One-step innovations residuals (statsmodels `innovations_filter`).
///
/// Uses [`innovations_algo`] on the sample acovf of `y`. Lag count is not
/// identification `p`.
pub fn innovations_filter(
    y: &Vector,
    nlags: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let h = nlags.max(1).min(y.len().saturating_sub(1).max(1));
    let g = match acovf(y, h, &session.child("acovf")) {
        Ok(q) => q.value,
        Err(_) => {
            ctx.push(
                Issue::builder(IssueCode::PValueUnreliable)
                    .message("innovations_filter acovf failed; residuals are y itself")
                    .build(),
            );
            return ctx.finish(y.clone());
        }
    };
    let inn = match innovations_algo(&g, &session.child("theta")) {
        Ok(q) => q.value,
        Err(_) => {
            return ctx.finish(y.clone());
        }
    };
    let n = y.len();
    let mut e = Vector::zeros(n);
    let mut xhat = Vector::zeros(n);
    for t in 0..n {
        let mut pred = 0.0;
        let row = t.min(inn.theta.nrows().saturating_sub(1));
        for j in 1..=t.min(h) {
            if t >= j {
                pred += inn.theta.get(row, j) * e[t - j];
            }
        }
        xhat[t] = pred;
        e[t] = y[t] - pred;
    }
    ctx.finish(e)
}

/// ARMA(\(p,q\)) to MA(\(\infty\)) coefficients (statsmodels `arma2ma`).
///
/// \(\psi_0=1\), \(\psi_k=\theta_k+\sum_j\phi_j\psi_{k-j}\). Orders are not
/// identification `p`.
pub fn arma2ma(
    ar: &Vector,
    ma: &Vector,
    lags: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, ar);
    inspect_univariate(&mut ctx, ma);
    let m = lags.max(1);
    if lags == 0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message("arma2ma lags=0; using 1")
                .build(),
        );
    }
    let mut psi = Vector::zeros(m);
    if m > 0 {
        psi[0] = 1.0;
    }
    for k in 1..m {
        let mut s = if k - 1 < ma.len() { ma[k - 1] } else { 0.0 };
        for j in 0..ar.len().min(k) {
            s += ar[j] * psi[k - 1 - j];
        }
        psi[k] = s;
        if !psi[k].is_finite() {
            ctx.push(
                Issue::builder(IssueCode::CausalityViolated)
                    .message("arma2ma coefficient overflowed; later ψ set to 0")
                    .build(),
            );
            psi[k] = 0.0;
        }
    }
    ctx.finish(psi)
}

/// ARMA(\(p,q\)) to AR(\(\infty\)) coefficients (statsmodels `arma2ar`).
///
/// \(\pi_0=1\), \(\pi_k=\phi_k-\sum_j\theta_j\pi_{k-j}\). Orders are not
/// identification `p`.
pub fn arma2ar(
    ar: &Vector,
    ma: &Vector,
    lags: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, ar);
    inspect_univariate(&mut ctx, ma);
    let m = lags.max(1);
    if lags == 0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message("arma2ar lags=0; using 1")
                .build(),
        );
    }
    let mut pi = Vector::zeros(m);
    if m > 0 {
        pi[0] = 1.0;
    }
    for k in 1..m {
        let mut s = if k - 1 < ar.len() { ar[k - 1] } else { 0.0 };
        for j in 0..ma.len().min(k) {
            s -= ma[j] * pi[k - 1 - j];
        }
        if !s.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::InvertibilityViolated)
                    .message("arma2ar coefficient overflowed; later π set to 0")
                    .build(),
            );
            s = 0.0;
        }
        pi[k] = s;
    }
    ctx.finish(pi)
}

/// Multiple seasonal-trend LOESS (sktime `MSTL`).
///
/// Second-period STL is run on the first residual. Periods are not
/// identification `p`.
#[derive(Clone, Debug)]
pub struct Mstl {
    /// Inner seasonal period.
    pub period: usize,
    /// Outer seasonal period.
    pub period2: usize,
}

impl Default for Mstl {
    fn default() -> Self {
        Self {
            period: 4,
            period2: 12,
        }
    }
}

impl Mstl {
    /// Two-period MSTL.
    pub fn new(period: usize, period2: usize) -> Self {
        Self {
            period: period.max(2),
            period2: period2.max(3),
        }
    }
}

/// Fitted multi-seasonal decomposition.
#[derive(Clone, Debug)]
pub struct FittedMstl {
    /// First seasonal component.
    pub seasonal: Vector,
    /// Second seasonal component.
    pub seasonal2: Vector,
    /// Trend.
    pub trend: Vector,
    /// Residual.
    pub resid: Vector,
}

impl FitSeries for Mstl {
    type Fitted = FittedMstl;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedMstl>> {
        let q1 = Stl::new(self.period).fit_series(y, &session.child("mstl1"))?;
        let r1 = q1.value.resid.clone();
        let q2 = match Stl::new(self.period2).fit_series(&r1, &session.child("mstl2")) {
            Ok(q) => q.value,
            Err(_) => SeasonalDecomposition {
                observed: r1.clone(),
                trend: Vector::zeros(y.len()),
                seasonal: Vector::zeros(y.len()),
                resid: r1.clone(),
                period: self.period2,
            },
        };
        let mut ctx = FitCtx::with_session(session.child("finish"));
        ctx.finish(FittedMstl {
            seasonal: q1.value.seasonal,
            seasonal2: q2.seasonal,
            trend: q1.value.trend,
            resid: q2.resid,
        })
    }
}

/// Lag embedding of a univariate series (sktime `Lag`).
#[derive(Clone, Debug)]
pub struct LagTransformer {
    /// Number of lags.
    pub lags: usize,
}

impl Default for LagTransformer {
    fn default() -> Self {
        Self { lags: 1 }
    }
}

impl LagTransformer {
    /// Embed with `lags` columns.
    pub fn new(lags: usize) -> Self {
        Self { lags: lags.max(1) }
    }

    /// Map `y` to `[y_{t-1}, …, y_{t-p}]` (`n` rows; leading lags are 0).
    pub fn transform(&self, y: &Vector, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_univariate(&mut ctx, y);
        let p = self.lags.max(1);
        let out = Matrix::from_fn(y.len(), p, |t, j| {
            let src = t as isize - (j as isize + 1);
            if src >= 0 {
                y[src as usize]
            } else {
                0.0
            }
        });
        ctx.finish(out)
    }
}

/// Rolling window summaries (sktime `WindowSummarizer` lite).
///
/// Window length is not identification `p`.
#[derive(Clone, Debug)]
pub struct WindowSummarizer {
    /// Rolling window.
    pub window: usize,
}

impl Default for WindowSummarizer {
    fn default() -> Self {
        Self { window: 4 }
    }
}

impl WindowSummarizer {
    /// Summaries over the given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Map `y` to `[mean, std, min, max]` per time.
    pub fn transform(&self, y: &Vector, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_univariate(&mut ctx, y);
        let w = self.window.max(2);
        let out = Matrix::from_fn(y.len(), 4, |t, j| {
            let lo = t.saturating_sub(w - 1);
            let mut s: f64 = 0.0;
            let mut s2: f64 = 0.0;
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            let mut c = 0.0;
            for i in lo..=t {
                let v = y[i];
                if !v.is_finite() {
                    continue;
                }
                s += v;
                s2 += v * v;
                mn = mn.min(v);
                mx = mx.max(v);
                c += 1.0;
            }
            let mean = if c > 0.0 { s / c } else { 0.0 };
            let var = if c > 1.0 {
                (s2 / c - mean * mean).max(0.0)
            } else {
                0.0
            };
            match j {
                0 => mean,
                1 => var.sqrt(),
                2 => {
                    if mn.is_finite() {
                        mn
                    } else {
                        0.0
                    }
                }
                _ => {
                    if mx.is_finite() {
                        mx
                    } else {
                        0.0
                    }
                }
            }
        });
        ctx.finish(out)
    }
}

/// Pick the best of Naive / Drift / AutoReg by in-sample SSE (sktime `MultiplexForecaster`).
#[derive(Clone, Debug)]
pub struct MultiplexForecaster {
    /// AutoReg lag used as one candidate.
    pub lags: usize,
}

impl Default for MultiplexForecaster {
    fn default() -> Self {
        Self { lags: 1 }
    }
}

impl MultiplexForecaster {
    /// Multiplex with AR(`lags`).
    pub fn new(lags: usize) -> Self {
        Self { lags: lags.max(1) }
    }
}

/// Fitted multiplex (the winning univariate forecaster's last value / AR).
#[derive(Clone, Debug)]
pub struct FittedMultiplex {
    /// Winning name.
    pub winner: &'static str,
    /// In-sample SSE of the winner.
    pub sse: f64,
    /// Naive / drift last level.
    pub last: f64,
    /// Drift slope (0 for naive).
    pub drift: f64,
    /// Optional AutoReg.
    pub ar: Option<FittedAutoReg>,
}

impl FittedMultiplex {
    /// Forecast `h` steps with the winning rule.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        if let Some(ar) = &self.ar {
            return ar.forecast(h, session);
        }
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::from_iter(
            (0..h).map(|i| self.last + self.drift * (i + 1) as f64),
        ))
    }
}

impl FitSeries for MultiplexForecaster {
    type Fitted = FittedMultiplex;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedMultiplex>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let last = y.as_slice().last().copied().unwrap_or(0.0);
        let mut naive_sse: f64 = 0.0;
        for i in 1..y.len() {
            let e = y[i] - y[i - 1];
            naive_sse += e * e;
        }
        let drift = if y.len() >= 2 {
            (last - y[0]) / (y.len() - 1) as f64
        } else {
            0.0
        };
        let mut drift_sse: f64 = 0.0;
        for i in 1..y.len() {
            let e = y[i] - (y[0] + drift * i as f64);
            drift_sse += e * e;
        }
        let ar = AutoReg::new(self.lags).fit_series(y, &session.child("mux-ar"));
        let (ar_sse, ar_fit) = match ar {
            Ok(q) => {
                let mut s: f64 = 0.0;
                let p = q.value.ar.len();
                for t in p..y.len() {
                    let mut yhat = q.value.intercept;
                    for k in 0..p {
                        yhat += q.value.ar[k] * y[t - p + k];
                    }
                    let e = y[t] - yhat;
                    s += e * e;
                }
                (s, Some(q.value))
            }
            Err(_) => (f64::INFINITY, None),
        };
        let (winner, sse, drift_used, ar_used) = if ar_sse <= naive_sse && ar_sse <= drift_sse {
            ("autoreg", ar_sse, 0.0, ar_fit)
        } else if drift_sse <= naive_sse {
            ("drift", drift_sse, drift, None)
        } else {
            ("naive", naive_sse, 0.0, None)
        };
        ctx.finish(FittedMultiplex {
            winner,
            sse,
            last,
            drift: drift_used,
            ar: ar_used,
        })
    }
}

/// Two-regime switching regression (statsmodels `MarkovRegression` lite).
///
/// States are an independent mixture of OLS, not a filtered Markov chain —
/// recorded as a numerical compromise.
#[derive(Clone, Debug, Default)]
pub struct MarkovRegression {
    /// EM iterations.
    pub max_iter: usize,
}

impl MarkovRegression {
    /// Two-regime mixture of regressions.
    pub fn new() -> Self {
        Self { max_iter: 20 }
    }
}

/// Fitted two-regime slopes.
#[derive(Clone, Debug)]
pub struct FittedMarkovReg {
    /// Intercept and slopes in regime 0.
    pub beta0: Vector,
    /// Intercept and slopes in regime 1.
    pub beta1: Vector,
    /// Soft assignment of each row to regime 1.
    pub regime: Vector,
}

impl MarkovRegression {
    /// Fit `y | X` with two intercept+slope regimes.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMarkovReg>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .severity(Severity::Advisory)
                .message("MarkovRegression uses an i.i.d. mixture, not Hamilton filtering")
                .compromise(NumericalCompromise::new(
                    "Hamilton-filtered Markov switching regression",
                    "two-regime mixture of OLS with independent states",
                    "transition probabilities are not identified",
                    "do not read the assignments as a Markov chain",
                ))
                .build(),
        );
        let n = y.len().min(x.nrows());
        let design = x.with_intercept();
        let p = design.ncols();
        let mut z = Vector::from_iter((0..n).map(|i| if i * 2 < n { 0.1 } else { 0.9 }));
        let mut beta0 = Vector::zeros(p);
        let mut beta1 = Vector::zeros(p);
        for it in 0..self.max_iter.max(1) {
            let w0 = Vector::from_iter((0..n).map(|i| (1.0 - z[i]).sqrt()));
            let w1 = Vector::from_iter((0..n).map(|i| z[i].sqrt()));
            let x0 = Matrix::from_fn(n, p, |i, j| design.get(i, j) * w0[i]);
            let x1 = Matrix::from_fn(n, p, |i, j| design.get(i, j) * w1[i]);
            let y0 = Vector::from_iter((0..n).map(|i| y[i] * w0[i]));
            let y1 = Vector::from_iter((0..n).map(|i| y[i] * w1[i]));
            let mut s0 = Report::new("ms", "r0");
            let mut s1 = Report::new("ms", "r1");
            if let Some(b) = crate::linalg::least_squares(&mut s0, &x0, &y0, &ctx.policy) {
                beta0 = b;
            }
            if let Some(b) = crate::linalg::least_squares(&mut s1, &x1, &y1, &ctx.policy) {
                beta1 = b;
            }
            let f0 = design.matvec(&beta0);
            let f1 = design.matvec(&beta1);
            for i in 0..n {
                let e0 = (y[i] - f0[i]).abs();
                let e1 = (y[i] - f1[i]).abs();
                let d = e0 + e1 + 1e-12;
                z[i] = e0 / d;
            }
            ctx.session.step(it as u64, z.mean(), None);
        }
        ctx.finish(FittedMarkovReg {
            beta0,
            beta1,
            regime: z,
        })
    }
}

/// Local-linear-trend plus dummy seasonal (statsmodels `UnobservedComponents` lite).
///
/// Seasonal length is **not** an identification `p`. Variances are treated as
/// known; this is a two-pass Kalman / dummy-seasonal smoother, not QMLE.
#[derive(Clone, Debug)]
pub struct UnobservedComponents {
    /// Observation variance \(\sigma_\varepsilon^2\).
    pub obs_var: f64,
    /// Level innovation \(\sigma_\eta^2\).
    pub level_var: f64,
    /// Slope innovation \(\sigma_\zeta^2\).
    pub slope_var: f64,
    /// Seasonal period (`0` or `1` ⇒ no seasonal).
    pub seasonal_period: usize,
}

impl Default for UnobservedComponents {
    fn default() -> Self {
        Self {
            obs_var: 1.0,
            level_var: 0.1,
            slope_var: 0.01,
            seasonal_period: 0,
        }
    }
}

impl UnobservedComponents {
    /// Local linear trend, no seasonal.
    pub fn new() -> Self {
        Self::default()
    }

    /// Local linear trend plus a dummy seasonal of period `s`.
    pub fn with_seasonal(period: usize) -> Self {
        Self {
            seasonal_period: period,
            ..Self::default()
        }
    }
}

/// Kalman-smoothed unobserved components.
#[derive(Clone, Debug)]
pub struct FittedUnobservedComponents {
    /// Level.
    pub level: Vector,
    /// Slope.
    pub slope: Vector,
    /// Dummy seasonal (zeros when `period ≤ 1`).
    pub seasonal: Vector,
    /// Irregular \(y - \mathrm{level} - \mathrm{seasonal}\).
    pub irregular: Vector,
}

impl FitSeries for UnobservedComponents {
    type Fitted = FittedUnobservedComponents;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedUnobservedComponents>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        let s = self.seasonal_period;
        if s >= 2 && n < 2 * s {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "UnobservedComponents seasonal period {s} needs n≥{}; series has {n}",
                        2 * s
                    ))
                    .build(),
            );
        }
        let mut llt = LocalLinearTrend {
            obs_var: self.obs_var,
            level_var: self.level_var,
            slope_var: self.slope_var,
        };
        let q = match llt.fit_series(y, &session.child("uc-llt")) {
            Ok(q) => q,
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                ) {
                    ctx.push(e.primary);
                }
                return ctx.finish(FittedUnobservedComponents {
                    level: Vector::zeros(n),
                    slope: Vector::zeros(n),
                    seasonal: Vector::zeros(n),
                    irregular: y.clone(),
                });
            }
        };
        for issue in q.report.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        let level = q.value.level;
        let slope = q.value.slope;
        let mut seasonal = Vector::zeros(n);
        if s >= 2 {
            let mut acc = vec![0.0; s];
            let mut cnt = vec![0.0; s];
            for t in 0..n {
                let irr = y[t] - level[t];
                if irr.is_finite() {
                    let k = t % s;
                    acc[k] += irr;
                    cnt[k] += 1.0;
                }
            }
            let mut mean = 0.0;
            let mut m = 0.0;
            for k in 0..s {
                if cnt[k] > 0.0 {
                    acc[k] /= cnt[k];
                    mean += acc[k];
                    m += 1.0;
                }
            }
            if m > 0.0 {
                mean /= m;
            }
            for k in 0..s {
                acc[k] -= mean;
            }
            for t in 0..n {
                seasonal[t] = acc[t % s];
            }
            ctx.push(
                Issue::builder(IssueCode::PValueUnreliable)
                    .severity(Severity::Advisory)
                    .message(
                        "UC seasonal is a dummy mean of the irregular, not a joint Kalman seasonal",
                    )
                    .compromise(NumericalCompromise::new(
                        "Harvey dummy-seasonal state in one Kalman filter",
                        "local-linear trend then period-mean of y−level",
                        "level and seasonal are not estimated jointly",
                        "do not read the seasonal as a QMLE state",
                    ))
                    .build(),
            );
        }
        let irregular = Vector::from_iter((0..n).map(|t| y[t] - level[t] - seasonal[t]));
        ctx.finish(FittedUnobservedComponents {
            level,
            slope,
            seasonal,
            irregular,
        })
    }
}

/// Two-regime Markov-switching AR (statsmodels `MarkovAutoregression`).
///
/// This is a Hamilton filter / Kim smoother with an EM M-step, not the
/// i.i.d. mixture [`MarkovRegression`]. Lag count is not identification `p`.
#[derive(Clone, Debug)]
pub struct MarkovSwitchingAutoregression {
    /// AR order.
    pub lags: usize,
    /// EM iterations.
    pub max_iter: usize,
}

impl Default for MarkovSwitchingAutoregression {
    fn default() -> Self {
        Self {
            lags: 1,
            max_iter: 20,
        }
    }
}

impl MarkovSwitchingAutoregression {
    /// Two-regime MS-AR(`lags`).
    pub fn new(lags: usize) -> Self {
        Self {
            lags: lags.max(1),
            max_iter: 20,
        }
    }
}

/// Fitted Hamilton-filtered switching AR.
#[derive(Clone, Debug)]
pub struct FittedMarkovSwitchingAr {
    /// Intercepts \(\mu_0,\mu_1\).
    pub mu: Vector,
    /// AR coefficients per regime (`2 × p`).
    pub ar: Matrix,
    /// Innovation variances.
    pub sigma2: Vector,
    /// Transition \(p_{i\to j}\) (`2 × 2`).
    pub transition: Matrix,
    /// Filtered \(P(s_t=1\mid y_{1:t})\).
    pub filtered: Vector,
    /// Smoothed \(P(s_t=1\mid y_{1:T})\).
    pub smoothed: Vector,
}

impl FitSeries for MarkovSwitchingAutoregression {
    type Fitted = FittedMarkovSwitchingAr;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMarkovSwitchingAr>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let p = self.lags.max(1);
        let n = y.len();
        if n <= p + 4 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("MarkovSwitchingAR n={n} is short for AR({p})"))
                    .build(),
            );
        }
        warn_unit_root(&mut ctx, y);
        let t0 = p;
        let t_eff = n.saturating_sub(t0);
        let mut mu = Vector::from_slice(&[y.mean() - 0.5 * y.std(), y.mean() + 0.5 * y.std()]);
        let mut phi = Matrix::from_fn(2, p, |_, _| 0.3);
        let mut sig2 = Vector::from_slice(&[y.std().powi(2).max(1e-4), y.std().powi(2).max(1e-4)]);
        let mut trans = Matrix::from_fn(2, 2, |i, j| if i == j { 0.85 } else { 0.15 });
        let mut filt = Vector::zeros(n);
        let mut smooth = Vector::zeros(n);
        let mut pred1 = vec![0.5; n];
        let mut f0 = vec![0.5; n];
        let mut f1 = vec![0.5; n];
        for it in 0..self.max_iter.max(1) {
            let mut xi0 = 0.5_f64;
            let mut xi1 = 0.5_f64;
            for t in 0..t0 {
                f0[t] = xi0;
                f1[t] = xi1;
                filt[t] = xi1;
                pred1[t] = xi1;
            }
            for t in t0..n {
                let p0 = trans.get(0, 0) * xi0 + trans.get(1, 0) * xi1;
                let p1 = trans.get(0, 1) * xi0 + trans.get(1, 1) * xi1;
                pred1[t] = p1;
                let mut m0 = mu[0];
                let mut m1 = mu[1];
                for k in 0..p {
                    m0 += phi.get(0, k) * y[t - 1 - k];
                    m1 += phi.get(1, k) * y[t - 1 - k];
                }
                let l0 = gauss_pdf(y[t], m0, sig2[0].sqrt());
                let l1 = gauss_pdf(y[t], m1, sig2[1].sqrt());
                let j0 = p0 * l0;
                let j1 = p1 * l1;
                let s = j0 + j1;
                if s <= 1e-300 {
                    xi0 = 0.5;
                    xi1 = 0.5;
                } else {
                    xi0 = j0 / s;
                    xi1 = j1 / s;
                }
                f0[t] = xi0;
                f1[t] = xi1;
                filt[t] = xi1;
            }
            let mut s0 = f0[n.saturating_sub(1)];
            let mut s1 = f1[n.saturating_sub(1)];
            smooth[n.saturating_sub(1)] = s1;
            if n > 1 {
                for t in (t0..n - 1).rev() {
                    let pr0 = trans.get(0, 0) * f0[t] + trans.get(1, 0) * f1[t];
                    let pr1 = trans.get(0, 1) * f0[t] + trans.get(1, 1) * f1[t];
                    let a00 = if pr0 > 1e-18 {
                        f0[t] * trans.get(0, 0) * s0 / pr0
                    } else {
                        0.0
                    };
                    let a01 = if pr1 > 1e-18 {
                        f0[t] * trans.get(0, 1) * s1 / pr1
                    } else {
                        0.0
                    };
                    let a10 = if pr0 > 1e-18 {
                        f1[t] * trans.get(1, 0) * s0 / pr0
                    } else {
                        0.0
                    };
                    let a11 = if pr1 > 1e-18 {
                        f1[t] * trans.get(1, 1) * s1 / pr1
                    } else {
                        0.0
                    };
                    s0 = a00 + a01;
                    s1 = a10 + a11;
                    let den = s0 + s1;
                    if den > 1e-18 {
                        s0 /= den;
                        s1 /= den;
                    }
                    smooth[t] = s1;
                }
            }
            for t in 0..t0 {
                smooth[t] = filt[t];
            }
            let mut n00 = 0.0;
            let mut n01 = 0.0;
            let mut n10 = 0.0;
            let mut n11 = 0.0;
            for t in (t0 + 1)..n {
                n00 += (1.0 - smooth[t - 1]) * (1.0 - smooth[t]);
                n01 += (1.0 - smooth[t - 1]) * smooth[t];
                n10 += smooth[t - 1] * (1.0 - smooth[t]);
                n11 += smooth[t - 1] * smooth[t];
            }
            let r0 = (n00 + n01).max(1e-12);
            let r1 = (n10 + n11).max(1e-12);
            trans.set(0, 0, n00 / r0);
            trans.set(0, 1, n01 / r0);
            trans.set(1, 0, n10 / r1);
            trans.set(1, 1, n11 / r1);
            for reg in 0..2 {
                let design = Matrix::from_fn(t_eff, p + 1, |i, j| {
                    let t = t0 + i;
                    let w = if reg == 0 {
                        (1.0 - smooth[t]).sqrt()
                    } else {
                        smooth[t].sqrt()
                    };
                    if j == 0 {
                        w
                    } else {
                        w * y[t - j]
                    }
                });
                let yy = Vector::from_iter((0..t_eff).map(|i| {
                    let t = t0 + i;
                    let w = if reg == 0 {
                        (1.0 - smooth[t]).sqrt()
                    } else {
                        smooth[t].sqrt()
                    };
                    w * y[t]
                }));
                let mut scratch = Report::new("msar", "ols");
                if let Some(b) =
                    crate::linalg::least_squares(&mut scratch, &design, &yy, &ctx.policy)
                {
                    mu[reg] = b.as_slice().first().copied().unwrap_or(0.0);
                    for k in 0..p {
                        if k + 1 < b.len() {
                            phi.set(reg, k, b[k + 1]);
                        }
                    }
                    let mut sse = 0.0;
                    let mut ww = 0.0;
                    for i in 0..t_eff {
                        let t = t0 + i;
                        let w = if reg == 0 { 1.0 - smooth[t] } else { smooth[t] };
                        let mut m = mu[reg];
                        for k in 0..p {
                            m += phi.get(reg, k) * y[t - 1 - k];
                        }
                        let e = y[t] - m;
                        sse += w * e * e;
                        ww += w;
                    }
                    sig2[reg] = (sse / ww.max(1e-12)).max(1e-8);
                }
            }
            ctx.session.step(it as u64, (mu[1] - mu[0]).abs(), None);
        }
        let mass1: f64 = (t0..n).map(|t| smooth[t]).sum::<f64>();
        if mass1 < 1.0 || mass1 > (t_eff as f64 - 1.0) {
            ctx.push(
                Issue::builder(IssueCode::DegenerateClusters)
                    .severity(Severity::Warning)
                    .message("one Markov regime has near-zero smoothed mass")
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(Severity::Advisory)
                .message("MS-AR EM uses a Kim smoother; SEs are not attached")
                .compromise(NumericalCompromise::new(
                    "Hamilton filter + information-matrix SEs",
                    "EM with weighted OLS M-step",
                    "the transition is a smoothed-count estimate",
                    "do not read μ as an OLS t-statistic",
                ))
                .build(),
        );
        ctx.finish(FittedMarkovSwitchingAr {
            mu,
            ar: phi,
            sigma2: sig2,
            transition: trans,
            filtered: filt,
            smoothed: smooth,
        })
    }
}

fn gauss_pdf(y: f64, mean: f64, sd: f64) -> f64 {
    let s = sd.max(1e-8);
    let z = (y - mean) / s;
    0.3989422804014327 / s * (-0.5 * z * z).exp()
}

/// Piecewise-linear trend plus Fourier seasonality (sktime / Prophet lite).
///
/// Knot and harmonic counts are **not** identification `p`.
#[derive(Clone, Debug)]
pub struct ProphetForecaster {
    /// Number of evenly spaced changepoints.
    pub n_changepoints: usize,
    /// Seasonal period for Fourier features.
    pub period: usize,
    /// Fourier harmonics.
    pub n_harmonics: usize,
}

impl Default for ProphetForecaster {
    fn default() -> Self {
        Self {
            n_changepoints: 3,
            period: 7,
            n_harmonics: 1,
        }
    }
}

impl ProphetForecaster {
    /// Prophet-lite with `k` changepoints and `h` harmonics of period `s`.
    pub fn new(n_changepoints: usize, period: usize, n_harmonics: usize) -> Self {
        Self {
            n_changepoints: n_changepoints.max(1),
            period: period.max(2),
            n_harmonics: n_harmonics.max(1),
        }
    }
}

/// Fitted Prophet-lite.
#[derive(Clone, Debug)]
pub struct FittedProphet {
    /// Intercept.
    pub intercept: f64,
    /// Linear slope.
    pub slope: f64,
    /// Changepoint deltas.
    pub deltas: Vector,
    /// Knot locations (time index).
    pub knots: Vector,
    /// Fourier coefficients (sin/cos interleaved).
    pub fourier: Vector,
    /// Training length.
    pub n: usize,
    /// Seasonal period.
    pub period: usize,
}

impl FittedProphet {
    /// `h`-step forecast continuing the time index.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        let out = Vector::from_iter((0..h).map(|u| {
            let t = (self.n + u) as f64;
            prophet_mean(
                t,
                self.intercept,
                self.slope,
                &self.deltas,
                &self.knots,
                &self.fourier,
                self.period,
            )
        }));
        ctx.finish(out)
    }
}

fn prophet_mean(
    t: f64,
    intercept: f64,
    slope: f64,
    deltas: &Vector,
    knots: &Vector,
    fourier: &Vector,
    period: usize,
) -> f64 {
    let mut yhat = intercept + slope * t;
    for k in 0..deltas.len().min(knots.len()) {
        let r = t - knots[k];
        if r > 0.0 {
            yhat += deltas[k] * r;
        }
    }
    let p = period.max(2) as f64;
    let nh = fourier.len() / 2;
    for j in 0..nh {
        let k = (j + 1) as f64;
        let ang = 2.0 * std::f64::consts::PI * k * t / p;
        if 2 * j < fourier.len() {
            yhat += fourier[2 * j] * ang.sin();
        }
        if 2 * j + 1 < fourier.len() {
            yhat += fourier[2 * j + 1] * ang.cos();
        }
    }
    yhat
}

impl FitSeries for ProphetForecaster {
    type Fitted = FittedProphet;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedProphet>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        let k = self.n_changepoints.max(1);
        let s = self.period.max(2);
        let h = self.n_harmonics.max(1);
        if n < 8 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("Prophet n={n} is short for a piecewise trend"))
                    .build(),
            );
        }
        let knots = Vector::from_iter((1..=k).map(|j| n as f64 * j as f64 / (k + 1) as f64));
        let cols = 2 + k + 2 * h;
        let design = Matrix::from_fn(n, cols, |i, j| {
            let t = i as f64;
            if j == 0 {
                1.0
            } else if j == 1 {
                t
            } else if j < 2 + k {
                let r = t - knots[j - 2];
                r.max(0.0)
            } else {
                let rest = j - 2 - k;
                let harm = rest / 2 + 1;
                let ang = 2.0 * std::f64::consts::PI * (harm as f64) * t / s as f64;
                if rest % 2 == 0 {
                    ang.sin()
                } else {
                    ang.cos()
                }
            }
        });
        let beta = statistical_ols(&mut ctx, &design, y).unwrap_or_else(|| {
            let mut b = Vector::zeros(cols);
            b[0] = y.mean();
            b
        });
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(Severity::Advisory)
                .message("Prophet-lite is OLS on ramps + Fourier; not Stan / L-BFGS")
                .compromise(NumericalCompromise::new(
                    "Bayesian Prophet with Laplace priors on deltas",
                    "unpenalized OLS on piecewise linear + Fourier",
                    "changepoints are evenly spaced, not selected",
                    "do not read deltas as posterior means",
                ))
                .build(),
        );
        ctx.finish(FittedProphet {
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            slope: if beta.len() > 1 { beta[1] } else { 0.0 },
            deltas: Vector::from_iter(
                (2..2 + k).map(|j| if j < beta.len() { beta[j] } else { 0.0 }),
            ),
            knots,
            fourier: Vector::from_iter((2 + k..cols).map(|j| {
                if j < beta.len() {
                    beta[j]
                } else {
                    0.0
                }
            })),
            n,
            period: s,
        })
    }
}

/// Mean / last / seasonal-last dummy (sktime `DummyForecaster`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DummyStrategy {
    /// In-sample mean.
    Mean,
    /// Last observation.
    Last,
    /// Seasonal naive (last value of the same season).
    SeasonalLast,
}

/// Naive / mean / seasonal dummy forecaster.
#[derive(Clone, Debug)]
pub struct DummyForecaster {
    /// Forecast rule.
    pub strategy: DummyStrategy,
    /// Period used by [`DummyStrategy::SeasonalLast`].
    pub period: usize,
}

impl Default for DummyForecaster {
    fn default() -> Self {
        Self {
            strategy: DummyStrategy::Last,
            period: 1,
        }
    }
}

impl DummyForecaster {
    /// Dummy with the given strategy.
    pub fn new(strategy: DummyStrategy) -> Self {
        Self {
            strategy,
            period: 1,
        }
    }
}

/// Fitted dummy forecaster.
#[derive(Clone, Debug)]
pub struct FittedDummyForecaster {
    /// Strategy.
    pub strategy: DummyStrategy,
    /// Mean of the training series.
    pub mean: f64,
    /// Last value.
    pub last: f64,
    /// Last `period` observations (oldest first) for seasonal naive.
    pub season: Vector,
}

impl FittedDummyForecaster {
    /// `h`-step dummy forecast.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        let p = self.season.len().max(1);
        let out = Vector::from_iter((0..h).map(|u| match self.strategy {
            DummyStrategy::Mean => self.mean,
            DummyStrategy::Last => self.last,
            DummyStrategy::SeasonalLast => self.season[u % p],
        }));
        ctx.finish(out)
    }
}

impl FitSeries for DummyForecaster {
    type Fitted = FittedDummyForecaster;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedDummyForecaster>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        if n == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("DummyForecaster on an empty series")
                    .build(),
            );
        }
        let p = self.period.max(1);
        if self.strategy == DummyStrategy::SeasonalLast && n < p {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("seasonal dummy needs n≥{p}; series has {n}"))
                    .build(),
            );
        }
        let season = if n == 0 {
            Vector::zeros(0)
        } else {
            Vector::from_iter(((n.saturating_sub(p))..n).map(|t| y[t]))
        };
        ctx.finish(FittedDummyForecaster {
            strategy: self.strategy,
            mean: y.mean(),
            last: y.as_slice().last().copied().unwrap_or(0.0),
            season,
        })
    }
}

fn inspect_univariate(ctx: &mut FitCtx, y: &Vector) {
    inspect_xy(&mut ctx.report, &Matrix::from_vector(y), None, &ctx.policy);
    if let Some(issue) = scan_finite(y.as_slice()).to_issue("y") {
        ctx.push(issue);
    }
}

fn reject_nonpositive(ctx: &mut FitCtx, y: &Vector, what: &str) {
    if y.as_slice().iter().any(|&v| !v.is_finite() || v <= 0.0) {
        ctx.push(
            Issue::builder(IssueCode::NonPositiveSeries)
                .message(format!("{what} requires a strictly positive series"))
                .meaninglessness(Meaninglessness::vacuous(
                    what,
                    "a log / multiplicative model is not defined on non-positive data",
                    "use an additive model or shift the series after disclosing the shift",
                ))
                .build(),
        );
    }
}

fn warn_unit_root(ctx: &mut FitCtx, y: &Vector) {
    warn_unit_root_slice(ctx, y.as_slice());
}

fn warn_unit_root_slice(ctx: &mut FitCtx, y: &[f64]) {
    if y.len() < 4 {
        return;
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for t in 1..y.len() {
        if y[t].is_finite() && y[t - 1].is_finite() {
            num += y[t] * y[t - 1];
            den += y[t - 1] * y[t - 1];
        }
    }
    if den > 0.0 {
        let rho = num / den;
        if rho.abs() > 0.98 {
            ctx.push(
                Issue::builder(IssueCode::NonStationary)
                    .message(format!(
                        "lag-1 coefficient ρ̂={rho:.4} is on the unit circle"
                    ))
                    .metric("rho", rho)
                    .build(),
            );
        }
    }
}

fn statistical_ols(ctx: &mut FitCtx, x: &Matrix, y: &Vector) -> Option<Vector> {
    let mut scratch = Report::new(ctx.report.algorithm.as_str(), "lstsq");
    let out = crate::linalg::least_squares(&mut scratch, x, y, &ctx.policy);
    for issue in scratch.issues() {
        if matches!(
            issue.code,
            IssueCode::ResidualTooLarge
                | IssueCode::NearSingular
                | IssueCode::RankZero
                | IssueCode::R2IsOne
        ) {
            continue;
        }
        ctx.push(issue.clone());
    }
    out
}

fn acf_raw(y: &[f64], nlags: usize) -> Vec<f64> {
    let st = slice_stats(y);
    let n = y.len();
    let mut out = vec![0.0; nlags + 1];
    out[0] = 1.0;
    if n == 0 {
        return out;
    }
    let mut g0 = 0.0;
    for &v in y {
        if v.is_finite() {
            let d = v - st.mean;
            g0 += d * d;
        }
    }
    if g0 <= 0.0 {
        for v in out.iter_mut().skip(1) {
            *v = f64::NAN;
        }
        return out;
    }
    for k in 1..=nlags {
        if k >= n {
            out[k] = f64::NAN;
            continue;
        }
        let mut g = 0.0;
        for t in k..n {
            if y[t].is_finite() && y[t - k].is_finite() {
                g += (y[t] - st.mean) * (y[t - k] - st.mean);
            }
        }
        out[k] = g / g0;
    }
    out
}

fn durbin_levinson_kk(rho: &[f64], k: usize) -> f64 {
    if k == 0 || k >= rho.len() {
        return f64::NAN;
    }
    let mut phi_prev = vec![0.0; k];
    let mut v: f64 = 1.0;
    for m in 1..=k {
        let mut acc = rho[m];
        for j in 1..m {
            acc -= phi_prev[j - 1] * rho[m - j];
        }
        let phimm = if v.abs() > 1e-15 { acc / v } else { 0.0 };
        let mut phi = vec![0.0; m];
        for j in 1..m {
            phi[j - 1] = phi_prev[j - 1] - phimm * phi_prev[m - j - 1];
        }
        phi[m - 1] = phimm;
        v *= 1.0 - phimm * phimm;
        phi_prev = phi;
    }
    phi_prev[k - 1]
}

/// Constant conditional correlation GARCH (arch `CCC`).
///
/// Marginal GARCH(1,1) then a fixed sample correlation of standardized
/// residuals. Series count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct CccGarch;

/// Fitted CCC correlations and marginal variances.
#[derive(Clone, Debug)]
pub struct FittedCccGarch {
    /// Per-series GARCH(1,1) variances (`T` × `k`).
    pub sigma2: Matrix,
    /// Constant correlation (`k` × `k`).
    pub corr: Matrix,
}

impl CccGarch {
    /// Empty CCC estimator.
    pub fn new() -> Self {
        Self
    }

    /// Fit marginal GARCH(1,1) and the sample correlation of `z_t`.
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedCccGarch>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let (n, k) = y.shape();
        if k < 2 {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("CCC needs at least two series")
                    .build(),
            );
        }
        let mut sigma2 = Matrix::zeros(n, k);
        let mut z = Matrix::zeros(n, k);
        for j in 0..k {
            let col = y.column(j);
            let mean = col.mean();
            let e: Vec<f64> = col.as_slice().iter().map(|v| v - mean).collect();
            let var = e.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64;
            let s2 = garch_sigma2(&e, 0.05 * var.max(1e-8), 0.05, 0.80);
            for t in 0..n {
                let v = s2.get(t).copied().unwrap_or(var).max(1e-12);
                sigma2.set(t, j, v);
                z.set(t, j, e.get(t).copied().unwrap_or(0.0) / v.sqrt());
            }
        }
        let mut corr = Matrix::zeros(k, k);
        if n > 0 && k > 0 {
            for a in 0..k {
                for b in 0..k {
                    let mut s = 0.0_f64;
                    for t in 0..n {
                        s += z.get(t, a) * z.get(t, b);
                    }
                    corr.set(a, b, s / n as f64);
                }
            }
            for i in 0..k {
                for j in 0..k {
                    let den = (corr.get(i, i).max(1e-12) * corr.get(j, j).max(1e-12)).sqrt();
                    corr.set(i, j, corr.get(i, j) / den);
                }
            }
        }
        ctx.finish(FittedCccGarch { sigma2, corr })
    }
}

/// Constant-variance residual model (arch `FixedVariance`).
///
/// \(h_t=\sigma^2\) for every \(t\).
#[derive(Clone, Debug, Default)]
pub struct FixedVariance;

/// Fitted constant variance.
#[derive(Clone, Debug)]
pub struct FittedFixedVariance {
    /// \(\sigma^2\).
    pub sigma2: f64,
    /// Demeaned residuals.
    pub resid: Vector,
}

impl FixedVariance {
    /// Empty fixed-variance estimator.
    pub fn new() -> Self {
        Self
    }
}

impl FitSeries for FixedVariance {
    type Fitted = FittedFixedVariance;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedFixedVariance>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let mean = y.mean();
        let e = Vector::from_iter(y.as_slice().iter().map(|v| v - mean));
        let n = e.len().max(1) as f64;
        let s2 = e.as_slice().iter().map(|v| v * v).sum::<f64>() / n;
        if !s2.is_finite() || s2 <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .severity(Severity::Warning)
                    .message("FixedVariance collapsed to a floor")
                    .build(),
            );
        }
        ctx.finish(FittedFixedVariance {
            sigma2: s2.max(1e-12),
            resid: e,
        })
    }
}

/// Named RiskMetrics EWMA (arch `RiskMetrics`).
///
/// Wraps [`EwmaVol::riskmetrics`]; \(\lambda=0.94\) is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct RiskMetrics;

impl RiskMetrics {
    /// RiskMetrics \(\lambda=0.94\).
    pub fn new() -> Self {
        Self
    }
}

impl FitSeries for RiskMetrics {
    type Fitted = FittedEwmaVol;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedEwmaVol>> {
        EwmaVol::riskmetrics().fit_series(y, session)
    }
}

/// HAR-X: Corsi HAR plus contemporaneous exogenous columns.
///
/// Window lengths are not identification `p`.
#[derive(Clone, Debug)]
pub struct HarX {
    /// Daily lookback.
    pub daily: usize,
    /// Weekly lookback.
    pub weekly: usize,
    /// Monthly lookback.
    pub monthly: usize,
}

impl Default for HarX {
    fn default() -> Self {
        Self {
            daily: 1,
            weekly: 5,
            monthly: 22,
        }
    }
}

impl HarX {
    /// Corsi (1, 5, 22) windows plus `x`.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted HAR-X coefficients.
#[derive(Clone, Debug)]
pub struct FittedHarX {
    /// Intercept.
    pub intercept: f64,
    /// Daily / weekly / monthly HAR slopes.
    pub beta_har: Vector,
    /// Exogenous slopes.
    pub beta_x: Vector,
    /// Trailing endogenous window.
    pub history: Vector,
    /// Last exogenous row (held fixed in a naive forecast).
    pub last_x: Vector,
    /// Daily window.
    pub daily: usize,
    /// Weekly window.
    pub weekly: usize,
    /// Monthly window.
    pub monthly: usize,
}

impl FittedHarX {
    /// Recurse HAR-X `h` steps, holding the last exogenous row fixed.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let mut hist: Vec<f64> = self.history.as_slice().to_vec();
        let mut out = Vector::zeros(h);
        let w = self.weekly.max(1);
        let m = self.monthly.max(1);
        let mut xterm = 0.0_f64;
        for j in 0..self.beta_x.len().min(self.last_x.len()) {
            xterm += self.beta_x[j] * self.last_x[j];
        }
        for t in 0..h {
            let n = hist.len();
            let daily = hist.last().copied().unwrap_or(0.0);
            let week = if n == 0 {
                daily
            } else {
                hist[n.saturating_sub(w)..].iter().sum::<f64>() / n.min(w) as f64
            };
            let month = if n == 0 {
                daily
            } else {
                hist[n.saturating_sub(m)..].iter().sum::<f64>() / n.min(m) as f64
            };
            let yhat = self.intercept
                + self.beta_har.as_slice().first().copied().unwrap_or(0.0) * daily
                + (if self.beta_har.len() > 1 {
                    self.beta_har[1]
                } else {
                    0.0
                }) * week
                + (if self.beta_har.len() > 2 {
                    self.beta_har[2]
                } else {
                    0.0
                }) * month
                + xterm;
            out[t] = yhat;
            hist.push(yhat);
        }
        ctx.finish(out)
    }
}

impl Fit for HarX {
    type Fitted = FittedHarX;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedHarX>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let daily = self.daily.max(1);
        let weekly = self.weekly.max(daily);
        let monthly = self.monthly.max(weekly);
        let n = y.len();
        let start = monthly;
        let k = x.ncols();
        let last_x = if n == 0 || k == 0 {
            Vector::zeros(k)
        } else {
            x.row(n - 1)
        };
        if n <= start {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!(
                        "HAR-X needs n>monthly={monthly}; got n={n}. Coefficients collapse to the last level."
                    ))
                    .metric("n", n as f64)
                    .build(),
            );
            return ctx.finish(FittedHarX {
                intercept: y.as_slice().last().copied().unwrap_or(0.0),
                beta_har: Vector::zeros(3),
                beta_x: Vector::zeros(k),
                history: y.clone(),
                last_x,
                daily,
                weekly,
                monthly,
            });
        }
        let n_eff = n - start;
        // Window / exogenous counts are not identification p.
        let design = Matrix::from_fn(n_eff, 3 + k, |i, j| {
            let t = i + start;
            if j < 3 {
                let win = match j {
                    0 => daily,
                    1 => weekly,
                    _ => monthly,
                };
                let d0 = t.saturating_sub(win);
                y.as_slice()[d0..t].iter().sum::<f64>() / (t - d0) as f64
            } else {
                x.get(t, j - 3)
            }
        });
        let yy = Vector::from_iter((start..n).map(|t| y[t]));
        let xaug = design.with_intercept();
        let beta = statistical_ols(&mut ctx, &xaug, &yy).unwrap_or_else(|| Vector::zeros(4 + k));
        let keep = monthly.min(n);
        let history = Vector::from_iter(y.as_slice()[n - keep..].iter().copied());
        ctx.finish(FittedHarX {
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            beta_har: Vector::from_iter((1..4).map(|j| {
                if j < beta.len() {
                    beta[j]
                } else {
                    0.0
                }
            })),
            beta_x: Vector::from_iter((4..beta.len()).map(|j| beta[j])),
            history,
            last_x,
            daily,
            weekly,
            monthly,
        })
    }
}

/// Mixed-frequency dynamic factor model (statsmodels `DynamicFactorMQ` lite).
///
/// Rows are temporally aggregated by `period` (not identification `p`), SVD
/// is taken on the low-frequency panel, and factors are expanded back.
#[derive(Clone, Debug)]
pub struct DynamicFactorMq {
    /// Number of latent factors.
    pub n_factors: usize,
    /// Aggregation period (e.g. 4 for quarterly-from-monthly).
    pub period: usize,
}

impl Default for DynamicFactorMq {
    fn default() -> Self {
        Self {
            n_factors: 1,
            period: 4,
        }
    }
}

impl DynamicFactorMq {
    /// `r` factors and aggregation `period`.
    pub fn new(n_factors: usize, period: usize) -> Self {
        Self {
            n_factors: n_factors.max(1),
            period: period.max(1),
        }
    }

    /// Fit on an `n × k` mixed-frequency panel (high-frequency rows).
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedDynamicFactorMq>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let (n, k) = y.shape();
        let per = self.period.max(1);
        if n < per {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!("DynamicFactorMQ needs n≥period={per}; got n={n}"))
                    .build(),
            );
        }
        let n_lf = (n / per).max(1);
        let ylf = Matrix::from_fn(n_lf, k, |i, j| {
            let lo = i * per;
            let hi = (lo + per).min(n);
            let mut s = 0.0_f64;
            let mut c = 0.0_f64;
            for t in lo..hi {
                let v = y.get(t, j);
                if v.is_finite() {
                    s += v;
                    c += 1.0;
                }
            }
            if c > 0.0 {
                s / c
            } else {
                0.0
            }
        });
        let (yc, mean) = ylf.centered();
        let mut scratch = Report::new("dfmq", "svd");
        let Some(svd) = thin_svd(&mut scratch, &yc, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::SvdDidNotConverge)
                    .message("DynamicFactorMQ SVD failed")
                    .build(),
            );
            return ctx.finish(FittedDynamicFactorMq {
                inner: FittedDynamicFactor {
                    loadings: Matrix::zeros(k, 1),
                    var: FittedVar {
                        lags: 1,
                        k: 1,
                        coef: Matrix::zeros(2, 1),
                        intercepts: Vector::zeros(1),
                        resid: Matrix::zeros(0, 1),
                        last: Matrix::zeros(1, 1),
                    },
                    mean,
                },
                period: per,
            });
        };
        let r = self.n_factors.max(1).min(svd.singular_values.len()).min(k);
        let loadings = Matrix::from_fn(k, r, |j, c| svd.v[(j, c)]);
        let factors_lf = Matrix::from_fn(n_lf, r, |i, c| {
            let mut s = 0.0_f64;
            for j in 0..k {
                s += yc.get(i, j) * loadings.get(j, c);
            }
            s
        });
        let var = match Var::new(1).fit(&factors_lf, &session.child("dfmq-var")) {
            Ok(q) => q.value,
            Err(_) => FittedVar {
                lags: 1,
                k: r,
                coef: Matrix::zeros(1 + r, r),
                intercepts: Vector::zeros(r),
                resid: Matrix::zeros(0, r),
                last: Matrix::from_fn(1, r, |_, j| factors_lf.get(n_lf.saturating_sub(1), j)),
            },
        };
        ctx.finish(FittedDynamicFactorMq {
            inner: FittedDynamicFactor {
                loadings,
                var,
                mean,
            },
            period: per,
        })
    }
}

/// Fitted mixed-frequency dynamic factor model.
#[derive(Clone, Debug)]
pub struct FittedDynamicFactorMq {
    /// Low-frequency DFM.
    pub inner: FittedDynamicFactor,
    /// Aggregation period.
    pub period: usize,
}

impl FittedDynamicFactorMq {
    /// `h` high-frequency steps: low-frequency forecast repeated `period` times.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let steps = h.div_ceil(self.period.max(1)).max(1);
        let lf = self.inner.forecast(steps, session)?;
        let ctx = FitCtx::with_session(session.child("dfmq-expand"));
        let k = lf.value.ncols();
        let per = self.period.max(1);
        let y = Matrix::from_fn(h, k, |t, j| lf.value.get(t / per, j));
        ctx.finish(y)
    }
}

/// Census X-13 lite: linear trend plus seasonal dummies (not full X-13ARIMA-SEATS).
///
/// Seasonal dummy count is not identification `p`.
#[derive(Clone, Debug)]
pub struct X13 {
    /// Seasonal period.
    pub period: usize,
}

impl Default for X13 {
    fn default() -> Self {
        Self { period: 12 }
    }
}

impl X13 {
    /// X-13 lite with the given period.
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(2),
        }
    }
}

/// Fitted X-13 lite components.
#[derive(Clone, Debug)]
pub struct FittedX13 {
    /// Linear trend.
    pub trend: Vector,
    /// Seasonal dummy component.
    pub seasonal: Vector,
    /// Irregular remainder.
    pub irreg: Vector,
    /// Period stored from the spec.
    pub period: usize,
}

impl FitSeries for X13 {
    type Fitted = FittedX13;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedX13>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        let s = self.period.max(2);
        if n < 2 * s {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSeasonalCycles)
                    .severity(Severity::Warning)
                    .message(format!("X-13 lite saw n={n} < 2·period={s}"))
                    .build(),
            );
        }
        // Intercept + trend + (s-1) dummies; dummy count is not identification p.
        let p = 2 + (s - 1);
        let design = Matrix::from_fn(n, p, |i, j| {
            if j == 0 {
                1.0
            } else if j == 1 {
                i as f64
            } else {
                let seas = i % s;
                if seas == j - 1 { 1.0 } else { 0.0 }
            }
        });
        let beta = statistical_ols(&mut ctx, &design, y).unwrap_or_else(|| Vector::zeros(p));
        let mut trend = Vector::zeros(n);
        let mut seasonal = Vector::zeros(n);
        let mut irreg = Vector::zeros(n);
        for i in 0..n {
            let tr = beta.as_slice().first().copied().unwrap_or(0.0)
                + (if beta.len() > 1 { beta[1] } else { 0.0 }) * i as f64;
            let mut se = 0.0_f64;
            for j in 2..p {
                if (i % s) == j - 1 {
                    se += if j < beta.len() { beta[j] } else { 0.0 };
                }
            }
            trend[i] = tr;
            seasonal[i] = se;
            irreg[i] = y[i] - tr - se;
        }
        ctx.finish(FittedX13 {
            trend,
            seasonal,
            irreg,
            period: s,
        })
    }
}

/// KPSS on seasonal differences (pmdarima / statsmodels seasonal KPSS).
///
/// Period is not identification `p`.
pub fn seasonal_kpss(
    y: &Vector,
    period: usize,
    session: &Session,
) -> Result<Qualified<KpssResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let s = period.max(2);
    let n = y.len();
    if n <= s {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!("seasonal_kpss needs n>period={s}; got n={n}"))
                .build(),
        );
        return ctx.finish(KpssResult {
            stat: f64::NAN,
            pvalue: f64::NAN,
            lags: 0,
            n,
        });
    }
    let z = Vector::from_iter((s..n).map(|t| y[t] - y[t - s]));
    match crate::stats::kpss(&z, None, &session.child("skpss")) {
        Ok(q) => {
            for issue in q.report.issues() {
                if !matches!(
                    issue.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::R2IsOne
                        | IssueCode::RankZero
                ) {
                    ctx.push(issue.clone());
                }
            }
            ctx.finish(q.value)
        }
        Err(e) => {
            if !matches!(
                e.primary.code,
                IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
            ) {
                ctx.push(e.primary);
            }
            ctx.finish(KpssResult {
                stat: f64::NAN,
                pvalue: f64::NAN,
                lags: 0,
                n: z.len(),
            })
        }
    }
}

/// Aggregate–disaggregate Croston (sktime `ADIDA`).
///
/// Period is not identification `p`. Fold only on a series with some positive
/// demand — Croston's `MeaninglessFit` aborts on an all-zero series.
#[derive(Clone, Debug)]
pub struct Adida {
    /// Aggregation bucket length.
    pub period: usize,
    /// Croston smoother on the aggregates.
    pub alpha: f64,
}

impl Default for Adida {
    fn default() -> Self {
        Self {
            period: 4,
            alpha: 0.1,
        }
    }
}

impl Adida {
    /// ADIDA with aggregation period `period`.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            ..Self::default()
        }
    }
}

/// Fitted ADIDA rate on the original time scale.
#[derive(Clone, Debug)]
pub struct FittedAdida {
    /// Disaggregated Croston rate.
    pub rate: f64,
    /// Aggregation period used.
    pub period: usize,
}

impl FittedAdida {
    /// Constant disaggregated rate.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::filled(h, self.rate))
    }
}

fn adida_aggregate(y: &Vector, period: usize) -> Vector {
    let s = period.max(1);
    let n_agg = y.len().div_ceil(s);
    Vector::from_iter((0..n_agg.max(1)).map(|k| {
        let lo = k * s;
        let hi = (lo + s).min(y.len());
        if lo >= y.len() {
            0.0
        } else {
            (lo..hi).map(|t| y[t]).sum::<f64>()
        }
    }))
}

impl FitSeries for Adida {
    type Fitted = FittedAdida;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedAdida>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let s = self.period.max(1);
        if y.len() < s {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("ADIDA period={s} > n={}; using the raw series", y.len()))
                    .build(),
            );
        }
        let agg = adida_aggregate(y, s);
        let crost = match Croston::new(self.alpha).fit_series(&agg, &session.child("adida_crost")) {
            Ok(q) => q.value,
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::R2IsOne
                        | IssueCode::RankZero
                ) {
                    ctx.push(e.primary);
                }
                return ctx.finish(FittedAdida {
                    rate: f64::NAN,
                    period: s,
                });
            }
        };
        let rate = if crost.p.abs() > 1e-15 {
            (crost.z / crost.p) / s as f64
        } else {
            f64::NAN
        };
        ctx.finish(FittedAdida { rate, period: s })
    }
}

/// Four-Theta combination (sktime `FourThetaForecaster` lite).
///
/// Combines Theta, SES, and seasonal-naive. Period is not identification `p`.
#[derive(Clone, Debug)]
pub struct FourTheta {
    /// Seasonal period for the naive member.
    pub period: usize,
}

impl Default for FourTheta {
    fn default() -> Self {
        Self { period: 4 }
    }
}

impl FourTheta {
    /// Four-Theta with seasonal period `period`.
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

/// Fitted Four-Theta members.
#[derive(Clone, Debug)]
pub struct FittedFourTheta {
    /// Theta member.
    pub theta: FittedTheta,
    /// SES level.
    pub ses_level: f64,
    /// Last seasonal cycle (length `period`).
    pub seasonal_last: Vector,
    /// Seasonal period.
    pub period: usize,
}

impl FittedFourTheta {
    /// Equal-weight combination of Theta, SES, and seasonal-naive.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let th = match self.theta.forecast(h, &session.child("fourtheta_th")) {
            Ok(q) => q.value,
            Err(_) => Vector::filled(h, self.theta.level),
        };
        let out = Vector::from_iter((0..h).map(|k| {
            let naive = if self.seasonal_last.is_empty() {
                self.ses_level
            } else {
                self.seasonal_last[k % self.seasonal_last.len()]
            };
            let t = if k < th.len() { th[k] } else { self.theta.level };
            (t + self.ses_level + naive) / 3.0
        }));
        ctx.finish(out)
    }
}

impl FitSeries for FourTheta {
    type Fitted = FittedFourTheta;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedFourTheta>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let s = self.period.max(1);
        if y.len() < 2 * s {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSeasonalCycles)
                    .severity(Severity::Warning)
                    .message(format!(
                        "FourTheta period={s} has fewer than two cycles in n={}",
                        y.len()
                    ))
                    .build(),
            );
        }
        let theta = match Theta.fit_series(y, &session.child("fourtheta_th")) {
            Ok(q) => q.value,
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::R2IsOne
                        | IssueCode::RankZero
                ) {
                    ctx.push(e.primary);
                }
                FittedTheta {
                    level: y.as_slice().last().copied().unwrap_or(0.0),
                    alpha: 1.0,
                    drift: 0.0,
                }
            }
        };
        let (_a, _b, ses_level, _tr, _f) = esm_fit(y.as_slice(), SmoothingKind::Simple, None, None);
        let seasonal_last = if y.is_empty() {
            Vector::zeros(0)
        } else {
            let take = s.min(y.len());
            Vector::from_iter(((y.len() - take)..y.len()).map(|t| y[t]))
        };
        ctx.finish(FittedFourTheta {
            theta,
            ses_level,
            seasonal_last,
            period: s,
        })
    }
}

/// Ensemble batch prediction intervals (sktime `EnbPI` lite).
///
/// Residual-bootstrap intervals around last-value / SES. Bootstrap / member
/// counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct EnbPI {
    /// Bootstrap draws.
    pub n_boot: usize,
    /// Nominal two-sided miss rate.
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for EnbPI {
    fn default() -> Self {
        Self {
            n_boot: 32,
            alpha: 0.1,
            seed: 3,
        }
    }
}

impl EnbPI {
    /// EnbPI-lite with `n_boot` residual draws.
    pub fn new(n_boot: usize) -> Self {
        Self {
            n_boot,
            ..Self::default()
        }
    }
}

/// Fitted EnbPI-lite intervals.
#[derive(Clone, Debug)]
pub struct FittedEnbPI {
    /// Point forecast (SES level).
    pub point: f64,
    /// Lower residual quantile (added to the point).
    pub lo: f64,
    /// Upper residual quantile.
    pub hi: f64,
}

impl FittedEnbPI {
    /// Constant SES point forecast.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        ctx.finish(Vector::filled(h, self.point))
    }

    /// Point plus residual-bootstrap interval bounds `(lo, mid, hi)` per step.
    pub fn interval(&self, h: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let ctx = FitCtx::with_session(session.child("interval"));
        ctx.finish(Matrix::from_fn(h, 3, |_, j| {
            if j == 0 {
                self.point + self.lo
            } else if j == 1 {
                self.point
            } else {
                self.point + self.hi
            }
        }))
    }
}

impl FitSeries for EnbPI {
    type Fitted = FittedEnbPI;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedEnbPI>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        if y.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("EnbPI needs at least one observation")
                    .build(),
            );
            return ctx.finish(FittedEnbPI {
                point: f64::NAN,
                lo: 0.0,
                hi: 0.0,
            });
        }
        let (alpha, _b, level, _tr, fitted) =
            esm_fit(y.as_slice(), SmoothingKind::Simple, None, None);
        let mut resid: Vec<f64> = Vec::new();
        for t in 0..y.len() {
            let e = y[t] - fitted[t];
            if e.is_finite() {
                resid.push(e);
            }
        }
        if resid.is_empty() {
            resid.push(0.0);
        }
        let mut rng = Rng::new(self.seed);
        let nb = self.n_boot.max(8);
        let mut boots = Vec::with_capacity(nb);
        for _ in 0..nb {
            let i = rng.below(resid.len());
            boots.push(resid[i]);
        }
        boots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let a = if self.alpha.is_finite() && self.alpha > 0.0 && self.alpha < 1.0 {
            self.alpha
        } else {
            0.1
        };
        let lo_i = ((a / 2.0) * (nb.saturating_sub(1)) as f64).floor() as usize;
        let hi_i = (((1.0 - a / 2.0) * (nb.saturating_sub(1)) as f64).ceil() as usize).min(nb - 1);
        let _ = alpha;
        ctx.finish(FittedEnbPI {
            point: level,
            lo: boots[lo_i.min(nb - 1)],
            hi: boots[hi_i],
        })
    }
}

/// Hamilton (2018) regression filter (statsmodels `HamiltonFilter` lite).
///
/// \(y_{t+h}\) on an intercept and `lags` of \(y_t,y_{t-1},\ldots\). Lag count
/// is not identification `p`.
#[derive(Clone, Debug)]
pub struct HamiltonFilter {
    /// Forecast horizon \(h\).
    pub horizon: usize,
    /// Number of lags of \(y_t\).
    pub lags: usize,
}

impl Default for HamiltonFilter {
    fn default() -> Self {
        Self {
            horizon: 2,
            lags: 4,
        }
    }
}

impl HamiltonFilter {
    /// Hamilton filter with horizon `h` and `lags` lags.
    pub fn new(horizon: usize, lags: usize) -> Self {
        Self { horizon, lags }
    }
}

/// Fitted Hamilton cycle.
#[derive(Clone, Debug)]
pub struct FittedHamiltonFilter {
    /// Residual cycle (length \(n\); leading observations are NaN).
    pub cycle: Vector,
    /// OLS coefficients `[intercept, y_t, …]`.
    pub coef: Vector,
    /// Horizon used.
    pub horizon: usize,
}

impl FitSeries for HamiltonFilter {
    type Fitted = FittedHamiltonFilter;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedHamiltonFilter>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let h = self.horizon.max(1);
        let p = self.lags.max(1);
        let n = y.len();
        let need = h + p;
        if n <= need {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!("HamiltonFilter needs n>{need}; got n={n}"))
                    .build(),
            );
            return ctx.finish(FittedHamiltonFilter {
                cycle: Vector::from_iter((0..n).map(|_| f64::NAN)),
                coef: Vector::zeros(p + 1),
                horizon: h,
            });
        }
        let n_reg = n - h - p + 1;
        let design = Matrix::from_fn(n_reg, p + 1, |i, j| {
            let t = i + (p - 1);
            if j == 0 {
                1.0
            } else {
                y[t + 1 - j]
            }
        });
        let yy = Vector::from_iter((0..n_reg).map(|i| {
            let t = i + (p - 1);
            y[t + h]
        }));
        let coef = statistical_ols(&mut ctx, &design, &yy).unwrap_or_else(|| Vector::zeros(p + 1));
        let mut cycle = Vector::from_iter((0..n).map(|_| f64::NAN));
        for i in 0..n_reg {
            let t = i + (p - 1);
            let mut fit = 0.0;
            for j in 0..coef.len() {
                fit += design.get(i, j) * coef[j];
            }
            cycle[t + h] = y[t + h] - fit;
        }
        ctx.finish(FittedHamiltonFilter {
            cycle,
            coef,
            horizon: h,
        })
    }
}

/// Add a constant and/or linear trend (statsmodels `add_trend`).
///
/// `kind` is `"c"`, `"t"`, or `"ct"`. Trend-column count is not identification
/// `p`.
pub fn add_trend(y: &Vector, kind: &str, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let n = y.len();
    let k = kind.trim().to_ascii_lowercase();
    let (const_, trend) = match k.as_str() {
        "c" | "const" => (true, false),
        "t" | "trend" => (false, true),
        "ct" | "ctt" => (true, true),
        other => {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!("add_trend kind={other:?} unknown; using ct"))
                    .build(),
            );
            (true, true)
        }
    };
    let p = 1 + usize::from(const_) + usize::from(trend);
    let out = Matrix::from_fn(n, p, |i, j| {
        if j == 0 {
            y[i]
        } else if j == 1 && const_ {
            1.0
        } else {
            i as f64
        }
    });
    ctx.finish(out)
}

/// Lag embedding (statsmodels `lagmat`).
///
/// Column `j` is \(y_{t-j}\) for \(j=1,\ldots,\mathrm{maxlag}\). Lag count is
/// not identification `p`.
pub fn lagmat(y: &Vector, maxlag: usize, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let n = y.len();
    let p = maxlag.max(1);
    if n <= p {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!("lagmat maxlag={p} ≥ n={n}"))
                .build(),
        );
    }
    let out = Matrix::from_fn(n, p, |t, j| {
        let lag = j + 1;
        if t >= lag {
            y[t - lag]
        } else {
            0.0
        }
    });
    ctx.finish(out)
}

/// Yule–Walker PACF via Durbin–Levinson (statsmodels `pacf_yw`).
///
/// Lag count is not identification `p`.
pub fn pacf_yw(y: &Vector, nlags: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let rho = acf_raw(y.as_slice(), nlags);
    let mut out = Vector::zeros(nlags + 1);
    out[0] = 1.0;
    for k in 1..=nlags {
        if k >= y.len() {
            out[k] = f64::NAN;
        } else {
            out[k] = durbin_levinson_kk(&rho, k);
        }
    }
    ctx.finish(out)
}

fn dft_re_im(y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let m = n / 2;
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut re = vec![0.0; m];
    let mut im = vec![0.0; m];
    for k in 1..=m {
        for (t, &yt) in y.iter().enumerate() {
            let ang = two_pi * k as f64 * t as f64 / n.max(1) as f64;
            re[k - 1] += yt * ang.cos();
            im[k - 1] += yt * ang.sin();
        }
    }
    (re, im)
}

/// Sliding-window periodogram (SciPy / statsmodels `spectrogram`).
///
/// Segment length is not identification `p`.
pub fn spectrogram(
    y: &Vector,
    nperseg: usize,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let n = y.len();
    let mut seg = nperseg.max(2);
    if nperseg < 2 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("spectrogram nperseg={nperseg} < 2; using 8"))
                .build(),
        );
        seg = 8;
    }
    if n < seg {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!("spectrogram nperseg={seg} > n={n}"))
                .build(),
        );
        seg = n.max(2);
    }
    let hop = (seg / 2).max(1);
    let mut rows = Vec::new();
    let mut t0 = 0usize;
    while t0 + seg <= n {
        let sl: Vec<f64> = (t0..t0 + seg).map(|t| y[t]).collect();
        let (re, im) = dft_re_im(&sl);
        rows.push(Vector::from_iter(
            re.iter()
                .zip(&im)
                .map(|(a, b)| (a * a + b * b) / seg as f64),
        ));
        t0 += hop;
        if hop == 0 {
            break;
        }
    }
    if rows.is_empty() {
        let (re, im) = dft_re_im(y.as_slice());
        rows.push(Vector::from_iter(
            re.iter()
                .zip(&im)
                .map(|(a, b)| (a * a + b * b) / n.max(1) as f64),
        ));
    }
    let nf = rows[0].len().max(1);
    let out = Matrix::from_fn(rows.len(), nf, |i, j| {
        if j < rows[i].len() {
            rows[i][j]
        } else {
            0.0
        }
    });
    ctx.finish(out)
}

/// Cross-spectral density (SciPy `csd` / statsmodels).
///
/// Frequency count is not identification `p`.
pub fn csd(x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, x);
    inspect_univariate(&mut ctx, y);
    let n = x.len().min(y.len());
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message("csd needs n≥2")
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    if x.len() != y.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .severity(Severity::Warning)
                .message(format!("csd lengths {} vs {}; using n={n}", x.len(), y.len()))
                .build(),
        );
    }
    let xs: Vec<f64> = (0..n).map(|t| x[t]).collect();
    let ys: Vec<f64> = (0..n).map(|t| y[t]).collect();
    let (xr, xi) = dft_re_im(&xs);
    let (yr, yi) = dft_re_im(&ys);
    let out = Vector::from_iter((0..xr.len()).map(|k| {
        let re = xr[k] * yr[k] + xi[k] * yi[k];
        let im = xi[k] * yr[k] - xr[k] * yi[k];
        (re * re + im * im).sqrt() / n as f64
    }));
    ctx.finish(out)
}

/// Magnitude-squared coherence (SciPy `coherence`).
///
/// Frequency count is not identification `p`.
pub fn coherence(x: &Vector, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, x);
    inspect_univariate(&mut ctx, y);
    let n = x.len().min(y.len());
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message("coherence needs n≥2")
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let xs: Vec<f64> = (0..n).map(|t| x[t]).collect();
    let ys: Vec<f64> = (0..n).map(|t| y[t]).collect();
    let (xr, xi) = dft_re_im(&xs);
    let (yr, yi) = dft_re_im(&ys);
    let out = Vector::from_iter((0..xr.len()).map(|k| {
        let pxx = (xr[k] * xr[k] + xi[k] * xi[k]) / n as f64;
        let pyy = (yr[k] * yr[k] + yi[k] * yi[k]) / n as f64;
        let re = xr[k] * yr[k] + xi[k] * yi[k];
        let im = xi[k] * yr[k] - xr[k] * yi[k];
        let pxy = (re * re + im * im).sqrt() / n as f64;
        let den = (pxx * pyy).sqrt();
        if den > 1e-15 {
            (pxy / den).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }));
    ctx.finish(out)
}

/// Gaussian innovations MLE of a white residual scale (statsmodels
/// `innovations.arma_innovations` / `innovations_mle` lite).
///
/// Lag count is not identification `p`.
pub fn innovations_mle(
    y: &Vector,
    nlags: usize,
    session: &Session,
) -> Result<Qualified<(f64, Vector)>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let e = match innovations_filter(y, nlags, &session.child("inn_mle")) {
        Ok(q) => q.value,
        Err(err) => {
            if !matches!(
                err.primary.code,
                IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
            ) {
                ctx.push(err.primary);
            }
            Vector::from_iter(y.as_slice().iter().copied())
        }
    };
    let n = e
        .as_slice()
        .iter()
        .filter(|v| v.is_finite())
        .count()
        .max(1) as f64;
    let sse = e
        .as_slice()
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v * v)
        .sum::<f64>();
    let sigma2 = (sse / n).max(1e-18);
    let nll = 0.5 * n * (sigma2.ln() + 1.0 + (2.0 * std::f64::consts::PI).ln());
    if !nll.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::DidNotConverge)
                .severity(Severity::Warning)
                .message("innovations_mle nll was non-finite; scale is a fallback")
                .build(),
        );
    }
    ctx.finish((nll, e))
}

/// ARMA innovations residuals and scale (statsmodels `arma_innovations`).
///
/// Lag count is not identification `p`.
pub fn arma_innovations(
    y: &Vector,
    nlags: usize,
    session: &Session,
) -> Result<Qualified<(Vector, f64)>> {
    let q = innovations_mle(y, nlags, session)?;
    let (_nll, e) = q.value;
    let n = e
        .as_slice()
        .iter()
        .filter(|v| v.is_finite())
        .count()
        .max(1) as f64;
    let sse = e
        .as_slice()
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v * v)
        .sum::<f64>();
    let mut ctx = FitCtx::with_session(session.child("arma_inn"));
    for issue in q.report.issues() {
        ctx.push(issue.clone());
    }
    ctx.finish((e, sse / n))
}

/// Cross-correlation from the two sample ACFs (statsmodels `ccf` via ACF).
///
/// \(\rho_{xy}(k)=\gamma_{xy}(k)/\sqrt{\gamma_{xx}(0)\gamma_{yy}(0)}\). Lag
/// count is not identification `p`.
pub fn ccf_from_acf(
    x: &Vector,
    y: &Vector,
    nlags: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let g = match ccovf(x, y, nlags, &session.child("ccf_acov")) {
        Ok(q) => {
            for issue in q.report.issues() {
                ctx.push(issue.clone());
            }
            q.value
        }
        Err(e) => {
            ctx.push(e.primary);
            return ctx.finish(Vector::zeros(nlags + 1));
        }
    };
    let sx = x.std().max(1e-18);
    let sy = y.std().max(1e-18);
    let out = Vector::from_iter((0..g.len()).map(|k| g[k] / (sx * sy)));
    ctx.finish(out)
}

/// Burg PACF (statsmodels `pacf_burg` lite).
///
/// Reflection coefficients of [`crate::stats::burg_ar`]. Order is not
/// identification `p`.
pub fn pacf_burg(y: &Vector, nlags: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let h = nlags.max(1).min(y.len().saturating_sub(1).max(1));
    match crate::stats::burg_ar(y, h, &session.child("pacf_burg")) {
        Ok(q) => {
            for issue in q.report.issues() {
                if !matches!(
                    issue.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::R2IsOne
                        | IssueCode::RankZero
                ) {
                    ctx.push(issue.clone());
                }
            }
            let mut out = Vector::zeros(h + 1);
            out[0] = 1.0;
            for i in 0..q.value.reflection.len().min(h) {
                out[i + 1] = q.value.reflection[i];
            }
            ctx.finish(out)
        }
        Err(e) => {
            if !matches!(
                e.primary.code,
                IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
            ) {
                ctx.push(e.primary);
            }
            let mut out = Vector::zeros(h + 1);
            out[0] = 1.0;
            ctx.finish(out)
        }
    }
}

/// Prepend a column of ones (statsmodels `add_constant`).
pub fn add_constant(x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    ctx.finish(x.with_intercept())
}

/// Subtract a linear time trend (statsmodels `detrend`).
pub fn detrend(y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let n = y.len();
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("detrend needs n≥2")
                .build(),
        );
        return ctx.finish(y.clone());
    }
    let design = Matrix::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
    let beta = statistical_ols(&mut ctx, &design, y).unwrap_or_else(|| Vector::from_slice(&[y.mean(), 0.0]));
    let out = Vector::from_iter((0..n).map(|i| {
        let fit = beta.as_slice().first().copied().unwrap_or(0.0)
            + beta.as_slice().get(1).copied().unwrap_or(0.0) * i as f64;
        y[i] - fit
    }));
    ctx.finish(out)
}

/// Linear convolution (SciPy `fftconvolve` lite; direct sum, not an FFT).
///
/// Lengths are not identification `p`.
pub fn fftconvolve(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, a);
    inspect_univariate(&mut ctx, b);
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .severity(Severity::Warning)
                .message("fftconvolve on an empty operand")
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("fftconvolve uses a direct O(nm) sum, not an FFT")
            .compromise(NumericalCompromise::new(
                "FFT convolution",
                "direct discrete convolution",
                "no FFT backend is linked",
                "do not read this as a spectral implementation",
            ))
            .build(),
    );
    let n = a.len() + b.len() - 1;
    let out = Vector::from_iter((0..n).map(|k| {
        let mut s = 0.0;
        for i in 0..a.len() {
            if k >= i && k - i < b.len() {
                s += a[i] * b[k - i];
            }
        }
        s
    }));
    ctx.finish(out)
}

/// Canova–Hansen seasonal-stability LM (statsmodels / Canova–Hansen).
///
/// Seasonal dummies are OLS-fitted and a KPSS-like statistic is formed on the
/// residuals. Period is not identification `p`.
pub fn ch_test(
    y: &Vector,
    period: usize,
    session: &Session,
) -> Result<Qualified<HypothesisTest>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let s = period.max(2);
    let n = y.len();
    if n < 2 * s {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSeasonalCycles)
                .message(format!("ch_test n={n} is tight for period={s}"))
                .build(),
        );
    }
    let dummies = Matrix::from_fn(n, s, |t, j| if t % s == j { 1.0 } else { 0.0 });
    let beta = statistical_ols(&mut ctx, &dummies, y).unwrap_or_else(|| Vector::zeros(s));
    let fit = dummies.matvec(&beta);
    let e = Vector::from_iter((0..n).map(|i| y[i] - fit[i]));
    match crate::stats::kpss(&e, None, &session.child("ch")) {
        Ok(q) => {
            for issue in q.report.issues() {
                if matches!(
                    issue.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::R2IsOne
                        | IssueCode::RankZero
                ) {
                    continue;
                }
                ctx.push(issue.clone());
            }
            ctx.finish(HypothesisTest {
                statistic: q.value.stat,
                pvalue: q.value.pvalue,
                df: (s.saturating_sub(1)) as f64,
                nobs: n as f64,
            })
        }
        Err(_) => {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("ch_test KPSS on seasonal residuals failed")
                    .build(),
            );
            ctx.finish(HypothesisTest {
                statistic: f64::NAN,
                pvalue: f64::NAN,
                df: (s.saturating_sub(1)) as f64,
                nobs: n as f64,
            })
        }
    }
}

/// Multi-column lag embedding (statsmodels `lagmat2ds`).
///
/// Column block `j` holds lags \(1,\ldots,\mathrm{maxlag}\) of input column
/// `j`. Lag and column counts are not identification `p`.
pub fn lagmat2ds(
    x: &Matrix,
    maxlag: usize,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let n = x.nrows();
    let k = x.ncols();
    let p = maxlag.max(1);
    if n <= p {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!("lagmat2ds maxlag={p} ≥ n={n}"))
                .build(),
        );
    }
    let out = Matrix::from_fn(n, k * p, |t, j| {
        let col = j / p;
        let lag = (j % p) + 1;
        if t >= lag {
            x.get(t - lag, col)
        } else {
            0.0
        }
    });
    ctx.finish(out)
}

/// Seasonal means of length `period` (statsmodels `seasonal.seasonal_mean`).
///
/// Period is not identification `p`.
pub fn seasonal_mean(
    y: &Vector,
    period: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let s = period.max(2);
    if period < 2 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("seasonal_mean period={period} < 2; using 2"))
                .build(),
        );
    }
    if y.len() < s {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSeasonalCycles)
                .message(format!("seasonal_mean n={} < period {s}", y.len()))
                .build(),
        );
    }
    let out = Vector::from_iter((0..s).map(|k| {
        let mut acc = 0.0_f64;
        let mut n = 0.0_f64;
        let mut t = k;
        while t < y.len() {
            if y[t].is_finite() {
                acc += y[t];
                n += 1.0;
            }
            t += s;
        }
        if n > 0.0 {
            acc / n
        } else {
            f64::NAN
        }
    }));
    ctx.finish(out)
}

/// Calendar Fourier terms (statsmodels `tsa.deterministic.CalendarFourier`).
///
/// Harmonic / period counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct CalendarFourier {
    /// Seasonal period.
    pub period: usize,
    /// Number of harmonics.
    pub order: usize,
}

impl Default for CalendarFourier {
    fn default() -> Self {
        Self {
            period: 4,
            order: 1,
        }
    }
}

impl CalendarFourier {
    /// `order` sine/cosine pairs for `period`.
    pub fn new(period: usize, order: usize) -> Self {
        Self {
            period: period.max(2),
            order: order.max(1),
        }
    }

    /// In-sample design of length `n`.
    pub fn in_sample(&self, n: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let ctx = FitCtx::with_session(session.child("in_sample"));
        let p = self.period.max(2);
        let h = self.order.max(1);
        let out = Matrix::from_fn(n, 2 * h, |t, j| {
            let k = j / 2 + 1;
            let ang = 2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / p as f64;
            if j % 2 == 0 {
                ang.sin()
            } else {
                ang.cos()
            }
        });
        ctx.finish(out)
    }
}

/// Theoretical ARMA ACF (statsmodels `arima_process.arma_acf`).
///
/// Uses a truncated MA(∞) expansion. Orders are not identification `p`.
pub fn arma_acf(
    ar: &Vector,
    ma: &Vector,
    nlags: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, ar);
    inspect_univariate(&mut ctx, ma);
    let h = nlags.max(1);
    if nlags == 0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message("arma_acf nlags=0; using 1")
                .build(),
        );
    }
    let trunc = h.saturating_add(32).max(64);
    let psi = match arma2ma(ar, ma, trunc, &session.child("arma_acf_psi")) {
        Ok(q) => q.value,
        Err(_) => {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(Severity::Warning)
                    .message("arma_acf MA(∞) expansion failed")
                    .build(),
            );
            return ctx.finish(Vector::from_iter((0..=h).map(|k| if k == 0 { 1.0 } else { 0.0 })));
        }
    };
    let mut gamma0 = 0.0_f64;
    for j in 0..psi.len() {
        gamma0 += psi[j] * psi[j];
    }
    if gamma0.abs() <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::ScaleFactorZero)
                .severity(Severity::Warning)
                .message("arma_acf γ(0) vanished; returning a spike at lag 0")
                .build(),
        );
        return ctx.finish(Vector::from_iter((0..=h).map(|k| if k == 0 { 1.0 } else { 0.0 })));
    }
    let acf = Vector::from_iter((0..=h).map(|k| {
        let mut g = 0.0_f64;
        for j in 0..psi.len().saturating_sub(k) {
            g += psi[j] * psi[j + k];
        }
        g / gamma0
    }));
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("arma_acf uses a truncated MA(∞) expansion, not the exact Yule–Walker Toeplitz")
            .compromise(NumericalCompromise::new(
                "exact ARMA autocovariance via the Lyapunov / Toeplitz system",
                format!("MA({trunc}) truncation of ψ"),
                "higher lags of ψ are dropped",
                "use a longer truncation before reading far-lag ACF as exact",
            ))
            .build(),
    );
    ctx.finish(acf)
}

/// AR order selected by AIC via Burg (statsmodels `ar_model.ar_select_order`).
///
/// Candidate order is not identification `p`.
#[derive(Clone, Debug)]
pub struct ArOrderSelect {
    /// Selected order (0 = intercept-only / white noise).
    pub order: usize,
    /// AIC for orders `0..=maxlag`.
    pub aic: Vector,
}

/// Grid Burg AIC over AR(0)..AR(`maxlag`).
pub fn ar_select_order(
    y: &Vector,
    maxlag: usize,
    session: &Session,
) -> Result<Qualified<ArOrderSelect>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    let n = y.len();
    let mut m = maxlag;
    if m == 0 || m >= n {
        m = (n.saturating_sub(1)).min(4).max(1);
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "ar_select_order maxlag={maxlag} is not in 1..n-1 (n={n}); using {m}"
                ))
                .build(),
        );
    }
    let mut aic = Vector::zeros(m + 1);
    let mut best_order = 0usize;
    let mean = y.mean();
    let mut sse0 = 0.0_f64;
    for i in 0..n {
        let e = y[i] - mean;
        sse0 += e * e;
    }
    let s00 = (sse0 / n.max(1) as f64).max(1e-18);
    aic[0] = n as f64 * s00.ln() + 2.0;
    let mut best = aic[0];
    for p in 1..=m {
        match crate::stats::burg_ar(y, p, &session.child(format!("ar_sel_{p}"))) {
            Ok(q) => {
                let s2 = q.value.sigma2.max(1e-18);
                let ic = n as f64 * s2.ln() + 2.0 * (p + 1) as f64;
                aic[p] = ic;
                if ic < best {
                    best = ic;
                    best_order = p;
                }
            }
            Err(_) => {
                aic[p] = f64::NAN;
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .severity(Severity::Warning)
                        .message(format!("ar_select_order Burg AR({p}) failed"))
                        .build(),
                );
            }
        }
    }
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("ar_select_order AIC uses Burg σ², not the exact Gaussian AR likelihood")
            .compromise(NumericalCompromise::new(
                "exact Gaussian AR MLE AIC",
                "Burg residual variance plus 2(p+1)",
                "the innovations likelihood is not evaluated",
                "treat the selected order as a Burg-AIC index, not a proven AR degree",
            ))
            .build(),
    );
    ctx.finish(ArOrderSelect {
        order: best_order,
        aic,
    })
}

/// Add contemporaneous and lagged copies of each column
/// (statsmodels `tsatools.add_lags`).
///
/// Lag / column counts are not identification `p`.
pub fn add_lags(x: &Matrix, lags: usize, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let n = x.nrows();
    let k = x.ncols();
    let p = lags;
    if n <= p {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!("add_lags lags={p} ≥ n={n}"))
                .build(),
        );
    }
    let width = k * (p + 1);
    let out = Matrix::from_fn(n, width, |t, j| {
        let col = j / (p + 1);
        let lag = j % (p + 1);
        if t >= lag {
            x.get(t - lag, col)
        } else {
            0.0
        }
    });
    ctx.finish(out)
}

/// Map a pandas / statsmodels frequency alias to a period
/// (statsmodels `tsatools.freq_to_period`).
pub fn freq_to_period(freq: &str, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let key = freq.trim().to_ascii_uppercase();
    let period = match key.as_str() {
        "A" | "Y" | "AS" | "YS" | "AE" | "YE" => 1.0,
        "Q" | "QS" | "QE" => 4.0,
        "M" | "MS" | "ME" => 12.0,
        "W" | "W-SUN" | "W-MON" => 52.0,
        "B" => 5.0,
        "D" => 7.0,
        "H" | "HR" => 24.0,
        "T" | "MIN" => 60.0,
        "S" => 60.0,
        other => {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!("freq_to_period alias {other:?} is unknown; using 1"))
                    .build(),
            );
            1.0
        }
    };
    ctx.finish(period)
}

    #[cfg(test)]
    mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn acf_of_whiteish_series() {
        let mut rng = Rng::new(21);
        let y = Vector::from_iter((0..80).map(|_| rng.standard_normal()));
        let session = Session::new("acf", "test");
        let q = acf(&y, 5, &session).expect("acf");
        assert!((q.value[0] - 1.0).abs() < 1e-12);
        assert!(
            q.value[1].abs() < 0.45,
            "lag-1 acf of white-ish series = {}",
            q.value[1]
        );
        assert!(q.value[2].abs() < 0.45);
    }

    #[test]
    fn naive_forecast_repeats_last() {
        let y = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let session = Session::new("naive", "test");
        let mut m = Naive;
        let q = m.fit_series(&y, &session).expect("naive fit");
        let f = q.value.forecast(3, &session).expect("naive forecast");
        assert_eq!(f.value.as_slice(), &[4.0, 4.0, 4.0]);
        let fh = ForecastingHorizon::relative(3);
        assert_eq!(fh.len(), 3);
        let split = temporal_train_test_split(&y, 0.25, &Session::new("tts", "t")).expect("tts");
        assert_eq!(split.value.0.len() + split.value.1.len(), y.len());
        assert!(!split.value.1.is_empty());
        let bag = BaggingForecaster::new(4)
            .fit_series(&y, &Session::new("bag", "fit"))
            .expect("bag");
        let bf = bag
            .value
            .forecast(3, &Session::new("bag", "fc"))
            .expect("bagf")
            .value;
        assert_eq!(bf.len(), 3);
        assert!(bf.as_slice().iter().all(|v| v.is_finite()));
        let cfm = NaiveConformal::new(0.8)
            .fit_series(&y, &Session::new("ncf", "fit"))
            .expect("ncf");
        let iv = cfm
            .value
            .interval(3, &Session::new("ncf", "iv"))
            .expect("ncfi")
            .value;
        assert_eq!(iv.shape(), (3, 3));
        assert!(iv.get(0, 0) <= iv.get(0, 1) && iv.get(0, 1) <= iv.get(0, 2));
        let stf = StackingForecaster
            .fit_series(&y, &Session::new("stf", "fit"))
            .expect("stf");
        let stff = stf
            .value
            .forecast(3, &Session::new("stf", "fc"))
            .expect("stff")
            .value;
        assert_eq!(stff.len(), 3);
        assert!(stff.as_slice().iter().all(|v| v.is_finite()));
        let aef = AutoEnsembleForecaster
            .fit_series(&y, &Session::new("aef", "fit"))
            .expect("aef");
        let aeff = aef
            .value
            .forecast(3, &Session::new("aef", "fc"))
            .expect("aeff")
            .value;
        assert_eq!(aeff.len(), 3);
        assert!(aeff.as_slice().iter().all(|v| v.is_finite()));
        assert!(aef.value.w_naive + aef.value.w_drift + aef.value.w_ses > 0.99);
        let oef = OnlineEnsembleForecaster::new()
            .fit_series(&y, &Session::new("oef", "fit"))
            .expect("oef");
        let oeff = oef
            .value
            .forecast(3, &Session::new("oef", "fc"))
            .expect("oeff")
            .value;
        assert_eq!(oeff.len(), 3);
        assert!(oeff.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn arima100_recovers_positive_phi() {
        let mut rng = Rng::new(33);
        let mut y = vec![0.0; 80];
        for t in 1..80 {
            y[t] = 0.7 * y[t - 1] + 0.25 * rng.standard_normal();
        }
        let session = Session::new("arima", "test");
        let mut model = Arima { p: 1, d: 0, q: 0 };
        let q = model
            .fit_series(&Vector::from_slice(&y), &session)
            .expect("arima fit");
        assert_eq!(q.value.ar.len(), 1);
        assert!(
            q.value.ar[0] > 0.0,
            "expected positive AR(1) coefficient, got {} issues={:?}",
            q.value.ar[0],
            q.report.issues().iter().map(|i| i.code).collect::<Vec<_>>()
        );
    }

    #[test]
    fn ensemble_forecaster_averages() {
        let y = Vector::from_iter((0..40).map(|i| 2.0 + 0.05 * i as f64 + (i as f64 * 0.4).sin()));
        let q = EnsembleForecaster::new(4)
            .fit_series(&y, &Session::new("ens", "fit"))
            .expect("ens");
        let f = q
            .value
            .forecast(3, &Session::new("ens", "fc"))
            .expect("fc")
            .value;
        assert_eq!(f.len(), 3);
        assert!(f.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn auto_arima_selects_a_finite_aic() {
        let mut rng = Rng::new(33);
        let mut y = vec![0.0; 60];
        for t in 1..60 {
            y[t] = 0.6 * y[t - 1] + 0.3 * rng.standard_normal();
        }
        let q = AutoArima {
            max_p: 1,
            max_d: 0,
            max_q: 1,
        }
        .fit_series(&Vector::from_slice(&y), &Session::new("aa", "fit"))
        .expect("auto");
        assert!(q.value.aic.is_finite());
        assert!(!q.value.scores.is_empty());
        assert!(q.value.fitted.sigma2.is_finite());
    }

    #[test]
    fn sarimax_tbats_and_pipelines() {
        let y = Vector::from_iter((0..40).map(|i| 1.0 + 0.3 * i as f64 + 0.2 * ((i as f64).sin())));
        let x = Matrix::from_fn(40, 1, |i, _| i as f64);
        let q = Sarimax::new(1, 0, 0)
            .fit(&y, &x, &Session::new("sx", "fit"))
            .expect("sarimax");
        assert!(
            (q.value.coef[0] - 0.3).abs() < 0.15,
            "b={}",
            q.value.coef[0]
        );
        let xf = Matrix::from_fn(4, 1, |i, _| 40.0 + i as f64);
        let f = q
            .value
            .forecast(4, &xf, &Session::new("sx", "fc"))
            .expect("fc")
            .value;
        assert_eq!(f.len(), 4);
        let ypos = Vector::from_iter((0..40).map(|i| (1.0 + 0.05 * i as f64).exp()));
        let pipe = ForecastingPipeline::new()
            .fit_series(&ypos, &Session::new("fp", "fit"))
            .expect("pipe");
        let pf = pipe
            .value
            .forecast(3, &Session::new("fp", "fc"))
            .expect("pfc")
            .value;
        assert!(pf.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let tb = Tbats::new(4)
            .fit_series(&y, &Session::new("tb", "fit"))
            .expect("tbats");
        let tf = tb
            .value
            .forecast(4, &Session::new("tb", "fc"))
            .expect("tbfc")
            .value;
        assert_eq!(tf.len(), 4);
        let ttf = TransformedTargetForecaster::new()
            .fit_series(&ypos, &Session::new("ttf", "fit"))
            .expect("ttf");
        assert!(ttf.value.log);
        let stl = Stl::new(4)
            .fit_series(&y, &Session::new("stl", "fit"))
            .expect("stl");
        assert_eq!(stl.value.trend.len(), 40);
        let mut dt = Detrender::new();
        dt.fit_series(&y, &Session::new("dt", "fit")).expect("dt");
        let dz = dt
            .transform(&y, &Session::new("dt", "t"))
            .expect("dtt")
            .value;
        assert!(dz.std() < y.std());
        let mut ds = Deseasonalizer::new(4);
        ds.fit_series(&y, &Session::new("ds", "fit")).expect("ds");
        assert_eq!(ds.means.len(), 4);
        let pt = PolynomialTrendForecaster::new(1)
            .fit_series(&y, &Session::new("pt", "fit"))
            .expect("pt");
        assert_eq!(
            pt.value
                .forecast(3, &Session::new("pt", "fc"))
                .expect("ptf")
                .value
                .len(),
            3
        );
        let mr = MarkovRegression::new()
            .fit(&x, &y, &Session::new("mr", "fit"))
            .expect("mr");
        assert_eq!(mr.value.regime.len(), 40);
        let ar = AutoReg::new(1)
            .fit_series(&y, &Session::new("ar", "fit"))
            .expect("autoreg");
        assert_eq!(ar.value.ar.len(), 1);
        let arf = ar
            .value
            .forecast(3, &Session::new("ar", "fc"))
            .expect("arf")
            .value;
        assert_eq!(arf.len(), 3);
        let adl = Ardl::new(1, 0)
            .fit(&y, &x, &Session::new("ardl", "fit"))
            .expect("ardl");
        let xf = Matrix::from_fn(3, 1, |i, _| 40.0 + i as f64);
        let adf = adl
            .value
            .forecast(&xf, &Session::new("ardl", "fc"))
            .expect("ardlf")
            .value;
        assert_eq!(adf.len(), 3);
        let ets = AutoEts::new(4)
            .fit_series(&y, &Session::new("ets", "fit"))
            .expect("ets");
        assert!(ets.value.aic.is_finite());
        let ff = FourierFeatures::new(4, 1)
            .transform(20, &Session::new("ff", "t"))
            .expect("ff")
            .value;
        assert_eq!(ff.shape(), (20, 2));
        let stlf = StlForecaster::new(4)
            .fit_series(&y, &Session::new("stlf", "fit"))
            .expect("stlf");
        assert_eq!(
            stlf.value
                .forecast(3, &Session::new("stlf", "fc"))
                .expect("stlff")
                .value
                .len(),
            3
        );
        let y2 = Matrix::from_fn(40, 2, |i, j| {
            if j == 0 {
                y[i]
            } else {
                0.5 * y[i] + 0.05 * i as f64
            }
        });
        let dfm = DynamicFactor::new(1)
            .fit(&y2, &Session::new("dfm", "fit"))
            .expect("dfm");
        let dff = dfm
            .value
            .forecast(3, &Session::new("dfm", "fc"))
            .expect("dff")
            .value;
        assert_eq!(dff.ncols(), 2);
        assert_eq!(dff.nrows(), 3);
        let varmax = Varmax::new(1)
            .fit(&y2, &x, &Session::new("vmx", "fit"))
            .expect("varmax");
        let xf = Matrix::from_fn(3, 1, |i, _| 40.0 + i as f64);
        let vmf = varmax
            .value
            .forecast(&xf, &Session::new("vmx", "fc"))
            .expect("vmxf")
            .value;
        assert_eq!(vmf.shape(), (3, 2));
        let mut bc = BoxCoxTransformer::new();
        bc.fit_series(&ypos, &Session::new("bc", "fit"))
            .expect("boxcox");
        let zt = bc
            .transform(&ypos, &Session::new("bc", "t"))
            .expect("bct")
            .value;
        assert_eq!(zt.len(), ypos.len());
        assert!(zt.as_slice().iter().all(|v| v.is_finite()));
        let mut d1 = Differencer::new();
        d1.fit_series(&y, &Session::new("d1", "fit")).expect("diff");
        let dz = d1
            .transform(&y, &Session::new("d1", "t"))
            .expect("dt")
            .value;
        assert_eq!(dz.len(), y.len());
        let cal = DateTimeFeatures::new(7)
            .transform(20, &Session::new("dtf", "t"))
            .expect("dtf")
            .value;
        assert_eq!(cal.shape(), (20, 4));
        let hol = HolidayFeatures::new(7)
            .transform(20, &Session::new("hol", "t"))
            .expect("hol")
            .value;
        assert_eq!(hol.shape(), (20, 1));
        assert!((hol.get(0, 0) - 1.0).abs() < 1e-12);
        assert!(hol.get(1, 0).abs() < 1e-12);
        let det = DeterministicProcess::seasonal(4)
            .transform(12, &Session::new("det", "t"))
            .expect("det")
            .value;
        assert_eq!(det.shape(), (12, 5));
        assert!((det.get(0, 0) - 1.0).abs() < 1e-12);
        let lg = LogTransformer::new()
            .transform(&ypos, &Session::new("logt", "t"))
            .expect("logt")
            .value;
        assert_eq!(lg.len(), ypos.len());
        assert!(lg.as_slice().iter().all(|v| v.is_finite()));
        let tsince = TimeSince::new(4)
            .transform(12, &Session::new("tsince", "t"))
            .expect("tsince")
            .value;
        assert_eq!(tsince.shape(), (12, 2));
        assert!((tsince.get(4, 0)).abs() < 1e-12);
        let col = ColumnEnsembleForecaster::new(1)
            .fit(&y2, &Session::new("col", "fit"))
            .expect("colens");
        let cf = col
            .value
            .forecast(3, &Session::new("col", "fc"))
            .expect("colf")
            .value;
        assert_eq!(cf.shape(), (3, 2));
        let fbl = ForecastByLevel
            .fit(&y2, &Session::new("fbl", "fit"))
            .expect("fbl");
        let fblf = fbl
            .value
            .forecast(3, &Session::new("fbl", "fc"))
            .expect("fblf")
            .value;
        assert_eq!(fblf.shape(), (3, 2));
        assert!(fblf.get(0, 0).is_finite() && fblf.get(2, 1).is_finite());
        let cc = ccf(&y, &y, 3, &Session::new("ccf", "t"))
            .expect("ccf")
            .value;
        assert!((cc[0] - 1.0).abs() < 1e-8);
        let pg = periodogram(&y, &Session::new("pg", "t")).expect("pg").value;
        assert!(!pg.is_empty());
        let mstl = Mstl::new(4, 8)
            .fit_series(&y, &Session::new("mstl", "fit"))
            .expect("mstl");
        assert_eq!(mstl.value.resid.len(), y.len());
        let lag = LagTransformer::new(2)
            .transform(&y, &Session::new("lag", "t"))
            .expect("lag")
            .value;
        assert_eq!(lag.ncols(), 2);
        let ws = WindowSummarizer::new(4)
            .transform(&y, &Session::new("ws", "t"))
            .expect("ws")
            .value;
        assert_eq!(ws.ncols(), 4);
        let mux = MultiplexForecaster::new(1)
            .fit_series(&y, &Session::new("mux", "fit"))
            .expect("mux");
        assert!(mux.value.sse.is_finite());
        let eg = Egarch::new()
            .fit_series(&y, &Session::new("eg", "fit"))
            .expect("egarch");
        assert!(eg
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        let gjr = GjrGarch::new()
            .fit_series(&y, &Session::new("gjr", "fit"))
            .expect("gjr");
        assert!(gjr
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        assert!(gjr.value.gamma.is_finite());
        let har = Har::new()
            .fit_series(&y, &Session::new("har", "fit"))
            .expect("har");
        assert!(har.value.intercept.is_finite());
        assert!(har.value.beta_d.is_finite());
        let harf = har
            .value
            .forecast(3, &Session::new("har", "fc"))
            .expect("harf")
            .value;
        assert_eq!(harf.len(), 3);
        assert!(harf.as_slice().iter().all(|v| v.is_finite()));
        let fig = Figarch::new()
            .fit_series(&y, &Session::new("fig", "fit"))
            .expect("figarch");
        assert!(fig
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        assert!(fig.value.d > 0.0 && fig.value.d < 1.0);
        let ap = Aparch::new()
            .fit_series(&y, &Session::new("ap", "fit"))
            .expect("aparch");
        assert!(ap
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        assert!(ap.value.delta > 0.0);
        let ha = Harch::new()
            .fit_series(&y, &Session::new("harch", "fit"))
            .expect("harch");
        assert!(ha
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        let ew = EwmaVol::riskmetrics()
            .fit_series(&y, &Session::new("ewma", "fit"))
            .expect("ewma");
        assert!((ew.value.lambda - 0.94).abs() < 1e-12);
        assert!(ew
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        let tsb = TsbCroston::new(0.2, 0.2)
            .fit_series(&ypos, &Session::new("tsb", "fit"))
            .expect("tsb");
        let tsbf = tsb
            .value
            .forecast(3, &Session::new("tsb", "fc"))
            .expect("tsbf")
            .value;
        assert_eq!(tsbf.len(), 3);
        assert!(tsbf.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let uc = UnobservedComponents::with_seasonal(4)
            .fit_series(&y, &Session::new("uc", "fit"))
            .expect("uc");
        assert_eq!(uc.value.level.len(), y.len());
        assert!(uc.value.seasonal.as_slice().iter().all(|v| v.is_finite()));
        let msar = MarkovSwitchingAutoregression::new(1)
            .fit_series(&y, &Session::new("msar", "fit"))
            .expect("msar");
        assert_eq!(msar.value.smoothed.len(), y.len());
        assert!(msar.value.transition.get(0, 0).is_finite());
        let var1 = Var::new(1)
            .fit(&y2, &Session::new("varirf", "fit"))
            .expect("var");
        let ir = var1
            .value
            .impulse_response(3, &Session::new("varirf", "irf"))
            .expect("irf");
        assert_eq!(ir.value.irf.len(), 4);
        assert_eq!(ir.value.fevd[3].shape(), (2, 2));
        let row_sum: f64 = (0..2).map(|j| ir.value.fevd[3].get(0, j)).sum();
        assert!((row_sum - 1.0).abs() < 1e-8, "fevd row={row_sum}");
        let pr = ProphetForecaster::new(2, 4, 1)
            .fit_series(&y, &Session::new("pr", "fit"))
            .expect("prophet");
        assert_eq!(
            pr.value
                .forecast(3, &Session::new("pr", "fc"))
                .expect("prf")
                .value
                .len(),
            3
        );
        let dum = DummyForecaster::new(DummyStrategy::Last)
            .fit_series(&y, &Session::new("dum", "fit"))
            .expect("dum");
        let df = dum
            .value
            .forecast(2, &Session::new("dum", "fc"))
            .expect("dumf")
            .value;
        assert!((df[0] - y[y.len() - 1]).abs() < 1e-12);
        let ses = SimpleExpSmoothing::new(Some(0.4))
            .fit_series(&y, &Session::new("ses", "fit"))
            .expect("ses");
        assert!(ses.value.level.is_finite());
        let holt = Holt::new(Some(0.4), Some(0.2))
            .fit_series(&y, &Session::new("holt", "fit"))
            .expect("holt");
        assert!(holt.value.trend.is_finite());
        let ll = LocalLevel::new()
            .fit_series(&y, &Session::new("ll", "fit"))
            .expect("ll");
        assert_eq!(ll.value.level.len(), y.len());
        let sv = Svar::new(1)
            .fit(&y2, &Session::new("svar", "fit"))
            .expect("svar");
        let sir = sv
            .value
            .structural_irf(2, &Session::new("svar", "irf"))
            .expect("sirf");
        assert!(!sir.value.irf.is_empty());
        let sab = SvarAb::new(1)
            .fit(&y2, &Session::new("svarab", "fit"))
            .expect("svarab");
        assert_eq!(sab.value.a0.nrows(), 2);
        let sabi = sab
            .value
            .structural_irf(2, &Session::new("svarab", "irf"))
            .expect("sabi");
        assert!(!sabi.value.irf.is_empty());
        let tb = TbatsFull::new(4)
            .fit_series(&ypos, &Session::new("tbatsf", "fit"))
            .expect("tbatsf");
        let tbf = tb
            .value
            .forecast(3, &Session::new("tbatsf", "fc"))
            .expect("tbatsff")
            .value;
        assert_eq!(tbf.len(), 3);
        assert!(tbf.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let fgs = ForecastingGridSearchCV::new(4)
            .fit_series(&y, &Session::new("fgs", "fit"))
            .expect("fgs");
        assert!(!fgs.value.best_name.is_empty());
        let fgf = fgs
            .value
            .forecast(3, &Session::new("fgs", "fc"))
            .expect("fgsf")
            .value;
        assert_eq!(fgf.len(), 3);
        let bq = BlanchardQuah::new(1)
            .fit(&y2, &Session::new("bq", "fit"))
            .expect("bq");
        assert_eq!(bq.value.impact.nrows(), 2);
        let bqi = bq
            .value
            .structural_irf(2, &Session::new("bq", "irf"))
            .expect("bqi");
        assert!(!bqi.value.irf.is_empty());
        let kak = ArimaKalman::new(1, 0, 1)
            .fit_series(&y, &Session::new("kak", "fit"))
            .expect("kak");
        assert!(kak.value.loglik.is_finite());
        let kakf = kak
            .value
            .forecast(3, &Session::new("kak", "fc"))
            .expect("kakf")
            .value;
        assert_eq!(kakf.len(), 3);
        assert!(kakf.as_slice().iter().all(|v| v.is_finite()));
        let w = welch(&y, 16, &Session::new("welch", "fit"))
            .expect("welch")
            .value;
        assert!(!w.is_empty());
        assert!(w.as_slice().iter().all(|v| v.is_finite() && *v >= 0.0));
        let ao = arma_order_select_ic(&y, 1, 1, &Session::new("armaic", "fit"))
            .expect("armaic")
            .value;
        assert!(ao.aic.is_finite() || ao.scores.is_empty());
        assert!(ao.p <= 1 && ao.q <= 1);
        let g = acovf(&y, 3, &Session::new("acovf", "fit"))
            .expect("acovf")
            .value;
        assert_eq!(g.len(), 4);
        assert!(g[0].is_finite() && g[0] > 0.0);
        let fx = ForecastX::new(2)
            .fit(&y, &x, &Session::new("fx", "fit"))
            .expect("fx");
        let fxf = fx
            .value
            .forecast(3, &xf, &Session::new("fx", "fc"))
            .expect("fxf")
            .value;
        assert_eq!(fxf.len(), 3);
        assert!(fxf.as_slice().iter().all(|v| v.is_finite()));
        let yh = Matrix::from_fn(2, 3, |h, j| match j {
            0 => 1.0 + h as f64,
            1 => 2.0 + h as f64,
            _ => 4.0 + h as f64,
        });
        let s = Matrix::from_fn(3, 2, |i, j| match (i, j) {
            (0, 0) | (1, 1) => 1.0,
            (2, 0) | (2, 1) => 1.0,
            _ => 0.0,
        });
        let rec = reconcile_ols(&yh, &s, &Session::new("rec", "ols"))
            .expect("rec")
            .value;
        assert_eq!(rec.shape(), (2, 3));
        assert!((rec.get(0, 0) + rec.get(0, 1) - rec.get(0, 2)).abs() < 1e-8);
        let mint = reconcile_mint(&yh, &s, &Session::new("rec", "mint"))
            .expect("mint")
            .value;
        assert_eq!(mint.shape(), (2, 3));
        assert!(mint.get(0, 0).is_finite() && mint.get(0, 2).is_finite());
        let ue = Uecm
            .fit(&y, &x, &Session::new("uecm", "fit"))
            .expect("uecm");
        let uef = ue
            .value
            .forecast(&xf, &Session::new("uecm", "fc"))
            .expect("uecmf")
            .value;
        assert_eq!(uef.len(), 3);
        assert!(uef.as_slice().iter().all(|v| v.is_finite()));
        let bu = reconcile_bottom_up(&yh, &s, &Session::new("rec", "bu"))
            .expect("bu")
            .value;
        assert_eq!(bu.shape(), (2, 3));
        assert!((bu.get(0, 0) + bu.get(0, 1) - bu.get(0, 2)).abs() < 1e-8);
        let td = reconcile_top_down(&yh, &s, &Session::new("rec", "td"))
            .expect("td")
            .value;
        assert_eq!(td.shape(), (2, 3));
        assert!(td.get(0, 0).is_finite() && td.get(0, 2).is_finite());
        let g = acovf(&y, 4, &Session::new("inn", "g"))
            .expect("acovf-inn")
            .value;
        let inn = innovations_algo(&g, &Session::new("inn", "algo")).expect("inn");
        assert_eq!(inn.value.variance.len(), 5);
        assert!(inn.value.variance.as_slice().iter().all(|v| v.is_finite()));
        let ef = innovations_filter(&y, 4, &Session::new("inn", "filt"))
            .expect("ifilt")
            .value;
        assert_eq!(ef.len(), 40);
        let psi = arma2ma(
            &Vector::from_slice(&[0.5]),
            &Vector::from_slice(&[0.2]),
            4,
            &Session::new("arma2ma", "t"),
        )
        .expect("arma2ma")
        .value;
        assert_eq!(psi.len(), 4);
        assert!((psi[0] - 1.0).abs() < 1e-12);
        assert!(psi.as_slice().iter().all(|v| v.is_finite()));
        let pi = arma2ar(
            &Vector::from_slice(&[0.5]),
            &Vector::from_slice(&[0.2]),
            4,
            &Session::new("arma2ar", "t"),
        )
        .expect("arma2ar")
        .value;
        assert_eq!(pi.len(), 4);
        assert!((pi[0] - 1.0).abs() < 1e-12);
        assert!(pi.as_slice().iter().all(|v| v.is_finite()));
        let arch = ArchP::new(1)
            .fit_series(&y, &Session::new("archp", "fit"))
            .expect("archp");
        assert!(arch
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        assert_eq!(arch.value.alphas.len(), 1);
        let archf = arch
            .value
            .forecast_variance(3, &Session::new("archp", "fc"))
            .expect("archpf")
            .value;
        assert_eq!(archf.len(), 3);
        let rv = RealizedVariance::new()
            .fit_series(&y, &Session::new("rv", "fit"))
            .expect("rv");
        assert!(rv.value.rv.is_finite() && rv.value.rv > 0.0);
        assert_eq!(rv.value.sigma2.len(), y.len());
        let hl = Matrix::from_fn(ypos.len(), 2, |i, j| {
            if j == 0 {
                ypos[i] * 1.01
            } else {
                ypos[i] * 0.99
            }
        });
        let pk = Parkinson::new()
            .estimate(&hl, &Session::new("pk", "est"))
            .expect("pk")
            .value;
        assert_eq!(pk.len(), ypos.len());
        assert!(pk.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let ohlc = Matrix::from_fn(ypos.len(), 4, |i, j| match j {
            0 => ypos[i],
            1 => ypos[i] * 1.01,
            2 => ypos[i] * 0.99,
            _ => ypos[i],
        });
        let gk = GarmanKlass::new()
            .estimate(&ohlc, &Session::new("gk", "est"))
            .expect("gk")
            .value;
        assert_eq!(gk.len(), ypos.len());
        assert!(gk.as_slice().iter().all(|v| v.is_finite()));
        let sba = SbaCroston::new(0.2)
            .fit_series(&ypos, &Session::new("sba", "fit"))
            .expect("sba");
        let sbaf = sba
            .value
            .forecast(3, &Session::new("sba", "fc"))
            .expect("sbaf")
            .value;
        assert_eq!(sbaf.len(), 3);
        assert!(sbaf.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let ath = AutoTheta
            .fit_series(&y, &Session::new("ath", "fit"))
            .expect("autotheta");
        assert!(ath.value.name == "ses" || ath.value.name == "theta");
        let athf = ath
            .value
            .forecast(3, &Session::new("ath", "fc"))
            .expect("athf")
            .value;
        assert_eq!(athf.len(), 3);
        let yx = YtoX::new(2)
            .transform(&y, &Session::new("ytox", "t"))
            .expect("ytox")
            .value;
        assert_eq!(yx.shape(), (40, 2));
        assert!((yx.get(2, 0) - y[1]).abs() < 1e-12);
        let sq = SquaringResiduals
            .fit_series(&y, &Session::new("sqres", "fit"))
            .expect("sqres");
        let sqf = sq
            .value
            .forecast(3, &Session::new("sqres", "fc"))
            .expect("sqf")
            .value;
        assert_eq!(sqf.len(), 3);
        assert!(sqf.as_slice().iter().all(|v| v.is_finite() && *v >= 0.0));
        let tr = TrendForecaster
            .fit_series(&y, &Session::new("trf", "fit"))
            .expect("trf");
        assert!(tr.value.slope.is_finite());
        let trf = tr
            .value
            .forecast(3, &Session::new("trf", "fc"))
            .expect("trff")
            .value;
        assert_eq!(trf.len(), 3);
        let im = Imapa
            .fit_series(&ypos, &Session::new("imapa", "fit"))
            .expect("imapa");
        let imf = im
            .value
            .forecast(3, &Session::new("imapa", "fc"))
            .expect("imapaf")
            .value;
        assert_eq!(imf.len(), 3);
        assert!(imf.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let rs = RogersSatchell::new()
            .estimate(&ohlc, &Session::new("rs", "est"))
            .expect("rs")
            .value;
        assert_eq!(rs.len(), ypos.len());
        assert!(rs.as_slice().iter().all(|v| v.is_finite()));
        let yz = YangZhang::new()
            .estimate(&ohlc, &Session::new("yz", "est"))
            .expect("yz")
            .value;
        assert!(yz.is_finite());
        let se = Setar
            .fit_series(&y, &Session::new("setar", "fit"))
            .expect("setar");
        let sef = se
            .value
            .forecast(3, &Session::new("setar", "fc"))
            .expect("setarf")
            .value;
        assert_eq!(sef.len(), 3);
        assert!(sef.as_slice().iter().all(|v| v.is_finite()));
        let ng = Ngarch::new()
            .fit_series(&y, &Session::new("ngarch", "fit"))
            .expect("ngarch");
        assert!(ng
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        assert!(ng.value.gamma.is_finite());
        let dcc = DccGarch::new()
            .fit(&y2, &Session::new("dcc", "fit"))
            .expect("dcc");
        assert_eq!(dcc.value.corr.shape(), (2, 2));
        assert!(dcc.value.corr.get(0, 1).is_finite());
        let hier = HierarchyEnsembleForecaster::new()
            .fit(&y2, &Session::new("hier", "fit"))
            .expect("hier");
        let hierf = hier
            .value
            .forecast(3, &Session::new("hier", "fc"))
            .expect("hierf")
            .value;
        assert_eq!(hierf.shape(), (3, 3));
        let aim = ArchInMean::new()
            .fit_series(&y, &Session::new("aim", "fit"))
            .expect("aim");
        assert!(aim
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        assert!(aim.value.lambda.is_finite());
        let ccc = CcGarch::new()
            .fit(&y2, &Session::new("ccc", "fit"))
            .expect("ccc");
        assert_eq!(ccc.value.corr.shape(), (2, 2));
        assert!(ccc.value.corr.get(0, 1).is_finite());
        let rk = RealizedKernel::new()
            .fit_series(&y, &Session::new("rk", "fit"))
            .expect("rk");
        assert!(rk.value.rk.is_finite());
        let mid = Midas::new(3)
            .fit(&y, &y, &Session::new("midas", "fit"))
            .expect("midas");
        assert!(mid.value.intercept.is_finite());
        let midf = mid
            .value
            .forecast(3, &Session::new("midas", "fc"))
            .expect("midf")
            .value;
        assert_eq!(midf.len(), 3);
        let star = Star
            .fit_series(&y, &Session::new("star", "fit"))
            .expect("star");
        let starf = star
            .value
            .forecast(3, &Session::new("star", "fc"))
            .expect("starf")
            .value;
        assert_eq!(starf.len(), 3);
        assert!(starf.as_slice().iter().all(|v| v.is_finite()));
        let recf = ReconcilerForecaster::new()
            .fit(&y2, &Session::new("recf", "fit"))
            .expect("recf");
        let recff = recf
            .value
            .forecast(2, &s, &Session::new("recf", "fc"))
            .expect("recff")
            .value;
        assert_eq!(recff.shape(), (2, 3));
        assert!((recff.get(0, 0) + recff.get(0, 1) - recff.get(0, 2)).abs() < 1e-8);
        let dtab = DirectTabularForecaster::new(3, 3)
            .fit_series(&y, &Session::new("dtab", "fit"))
            .expect("dtab");
        let dtabf = dtab
            .value
            .forecast(3, &Session::new("dtab", "fc"))
            .expect("dtabf")
            .value;
        assert_eq!(dtabf.len(), 3);
        assert!(dtabf.as_slice().iter().all(|v| v.is_finite()));
        let bats = Bats::new(4)
            .fit_series(&y, &Session::new("bats", "fit"))
            .expect("bats");
        let batsf = bats
            .value
            .forecast(3, &Session::new("bats", "fc"))
            .expect("batsf")
            .value;
        assert_eq!(batsf.len(), 3);
        assert!(batsf.as_slice().iter().all(|v| v.is_finite()));
        let bekk = BekkGarch::new()
            .fit(&y2, &Session::new("bekk", "fit"))
            .expect("bekk");
        assert_eq!(bekk.value.corr.shape(), (2, 2));
        assert!(bekk.value.corr.get(0, 1).is_finite());
        let mot = MultioutputTabularForecaster::new(3, 3)
            .fit(&y2, &Session::new("mot", "fit"))
            .expect("mot");
        let motf = mot
            .value
            .forecast(3, &Session::new("mot", "fc"))
            .expect("motf")
            .value;
        assert_eq!(motf.shape(), (3, 2));
        assert!(motf.get(0, 0).is_finite() && motf.get(2, 1).is_finite());
        let recu = RecursiveTabularForecaster::new(3)
            .fit_series(&y, &Session::new("rtab", "fit"))
            .expect("rtab");
        let recuf = recu
            .value
            .forecast(3, &Session::new("rtab", "fc"))
            .expect("rtabf")
            .value;
        assert_eq!(recuf.len(), 3);
        assert!(recuf.as_slice().iter().all(|v| v.is_finite()));
        let drec = DirRecTabularForecaster::new(3, 3)
            .fit_series(&y, &Session::new("drec", "fit"))
            .expect("drec");
        let drecf = drec
            .value
            .forecast(3, &Session::new("drec", "fc"))
            .expect("drecf")
            .value;
        assert_eq!(drecf.len(), 3);
        assert!(drecf.as_slice().iter().all(|v| v.is_finite()));
        let rvx = Vector::from_iter((0..y.len()).map(|i| {
            let e = y[i] - y.mean();
            e * e
        }));
        let rg = RealizedGarch::new()
            .fit(&y, &rvx, &Session::new("rgarch", "fit"))
            .expect("rgarch");
        assert!(rg
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        assert!(rg.value.gamma.is_finite());
        let sm = simulation_smoother(&y, 3, &Session::new("simsm", "t")).expect("simsm");
        assert_eq!(sm.value.len(), y.len());
        assert!(sm.value.as_slice().iter().all(|v| v.is_finite()));
        let nw = statespace_news(&y, &Session::new("news", "t")).expect("news");
        assert_eq!(nw.value.len(), y.len());
        assert!(nw.value.as_slice().iter().all(|v| v.is_finite()));
        let ig = Igarch::new()
            .fit_series(&y, &Session::new("igarch", "fit"))
            .expect("igarch");
        assert!((ig.value.alpha + ig.value.beta - 1.0).abs() < 1e-12);
        assert!(ig
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        let cg = ComponentGarch::new()
            .fit_series(&y, &Session::new("cgarch", "fit"))
            .expect("cgarch");
        assert!(cg
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        let gog = GoGarch::new()
            .fit(&y2, &Session::new("gog", "fit"))
            .expect("gog");
        assert_eq!(gog.value.loadings.nrows(), 2);
        assert!(gog.value.factor_var.ncols() >= 1);
        let smt = SmoothTrend::new(100.0)
            .fit_series(&y, &Session::new("smt", "fit"))
            .expect("smt");
        assert_eq!(smt.value.trend.len(), 40);
        assert!(smt.value.trend.as_slice().iter().all(|v| v.is_finite()));
        assert_eq!(
            smt.value
                .forecast(3, &Session::new("smt", "fc"))
                .expect("smtf")
                .value
                .len(),
            3
        );
        let nd = ndiffs(&y, &Session::new("nd", "t")).expect("nd");
        assert!(nd.value <= 1);
        let oc = ocsb(&y, 4, &Session::new("ocsb", "t")).expect("ocsb");
        assert!(oc.value.is_finite() || oc.value.is_nan());
        let nsd = nsdiffs(&y, 4, &Session::new("nsd", "t")).expect("nsd");
        assert!(nsd.value <= 1);
        let qg = Qgarch::new()
            .fit_series(&y, &Session::new("qg", "fit"))
            .expect("qg");
        assert!(qg
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        let ta = Tarch::new()
            .fit_series(&y, &Session::new("tarch", "fit"))
            .expect("tarch");
        assert!(ta
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        let av = Avgarch::new()
            .fit_series(&y, &Session::new("avg", "fit"))
            .expect("avg");
        assert!(av
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        let za = Zarch::new()
            .fit_series(&y, &Session::new("zarch", "fit"))
            .expect("zarch");
        assert!(za
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        let hg = hegy(&y, 4, &Session::new("hegy", "t")).expect("hegy");
        assert!(hg.value.statistic.is_finite() || hg.value.pvalue.is_nan());
        let ch = canova_hansen(&y, 4, &Session::new("ch", "t")).expect("ch");
        assert!(ch.value.statistic.is_finite() || ch.value.pvalue.is_nan());
        let cv = ccovf(&y, &y, 3, &Session::new("ccov", "t")).expect("ccov");
        assert_eq!(cv.value.len(), 4);
        assert!(cv.value[0].is_finite());
        let po = pacf_ols(&y, 3, &Session::new("pols", "t")).expect("pols");
        assert_eq!(po.value.len(), 4);
        assert!((po.value[0] - 1.0).abs() < 1e-12);
        let fi = Fiegarch::new()
            .fit_series(&y, &Session::new("fie", "fit"))
            .expect("fie");
        assert!(fi
            .value
            .sigma2
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        assert!(fi.value.d > 0.0 && fi.value.d < 0.5);
        let varr = value_at_risk(&y, 0.1, &Session::new("var", "t")).expect("var");
        assert!(varr.value.is_finite());
        let es = expected_shortfall(&y, 0.1, &Session::new("es", "t")).expect("es");
        assert!(es.value <= varr.value + 1e-9);
        let ccc = CccGarch::new()
            .fit(&y2, &Session::new("ccc", "fit"))
            .expect("ccc");
        assert_eq!(ccc.value.corr.shape(), (2, 2));
        assert!(ccc.value.corr.get(0, 1).is_finite());
        assert!((0..ccc.value.sigma2.nrows()).all(|t| {
            (0..ccc.value.sigma2.ncols())
                .all(|j| ccc.value.sigma2.get(t, j).is_finite() && ccc.value.sigma2.get(t, j) > 0.0)
        }));
        let fv = FixedVariance::new()
            .fit_series(&y, &Session::new("fv", "fit"))
            .expect("fv");
        assert!(fv.value.sigma2.is_finite() && fv.value.sigma2 > 0.0);
        let rm = RiskMetrics::new()
            .fit_series(&y, &Session::new("rm", "fit"))
            .expect("rm");
        assert!((rm.value.lambda - 0.94).abs() < 1e-12);
        let hx = HarX::new()
            .fit(&x, &y, &Session::new("harx", "fit"))
            .expect("harx");
        assert!(hx.value.intercept.is_finite());
        let hxf = hx
            .value
            .forecast(3, &Session::new("harx", "fc"))
            .expect("harxf")
            .value;
        assert_eq!(hxf.len(), 3);
        assert!(hxf.as_slice().iter().all(|v| v.is_finite()));
        let dfmq = DynamicFactorMq::new(1, 4)
            .fit(&y2, &Session::new("dfmq", "fit"))
            .expect("dfmq");
        let dfmqf = dfmq
            .value
            .forecast(3, &Session::new("dfmq", "fc"))
            .expect("dfmqf")
            .value;
        assert_eq!(dfmqf.shape(), (3, 2));
        let x13 = X13::new(4)
            .fit_series(&y, &Session::new("x13", "fit"))
            .expect("x13");
        assert_eq!(x13.value.trend.len(), 40);
        assert!(x13.value.seasonal.as_slice().iter().all(|v| v.is_finite()));
        let sk = seasonal_kpss(&y, 4, &Session::new("skpss", "t")).expect("skpss");
        assert!(sk.value.stat.is_finite() || sk.value.pvalue.is_nan());
        let ad = Adida::new(4)
            .fit_series(&ypos, &Session::new("adida", "fit"))
            .expect("adida");
        let adf = ad
            .value
            .forecast(3, &Session::new("adida", "fc"))
            .expect("adidaf")
            .value;
        assert_eq!(adf.len(), 3);
        assert!(adf.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let ft = FourTheta::new(4)
            .fit_series(&y, &Session::new("fth", "fit"))
            .expect("fth");
        let ftf = ft
            .value
            .forecast(4, &Session::new("fth", "fc"))
            .expect("fthf")
            .value;
        assert_eq!(ftf.len(), 4);
        assert!(ftf.as_slice().iter().all(|v| v.is_finite()));
        let enb = EnbPI::new(16)
            .fit_series(&y, &Session::new("enbpi", "fit"))
            .expect("enbpi");
        let enbf = enb
            .value
            .forecast(3, &Session::new("enbpi", "fc"))
            .expect("enbf")
            .value;
        assert_eq!(enbf.len(), 3);
        let enbi = enb
            .value
            .interval(3, &Session::new("enbpi", "int"))
            .expect("enbi")
            .value;
        assert_eq!(enbi.shape(), (3, 3));
        let ham = HamiltonFilter::new(2, 4)
            .fit_series(&y, &Session::new("ham", "fit"))
            .expect("ham");
        assert_eq!(ham.value.cycle.len(), 40);
        assert!(ham.value.coef.as_slice().iter().all(|v| v.is_finite()));
        let tr = add_trend(&y, "ct", &Session::new("atr", "t")).expect("atr");
        assert_eq!(tr.value.shape(), (40, 3));
        let lm = lagmat(&y, 3, &Session::new("lagm", "t")).expect("lagm");
        assert_eq!(lm.value.shape(), (40, 3));
        let pyw = pacf_yw(&y, 5, &Session::new("pyw", "t")).expect("pyw");
        assert_eq!(pyw.value.len(), 6);
        assert!((pyw.value[0] - 1.0).abs() < 1e-12);
        let sp = spectrogram(&y, 8, &Session::new("spec", "t")).expect("spec");
        assert!(sp.value.nrows() >= 1 && sp.value.ncols() >= 1);
        assert!((0..sp.value.nrows()).all(|i| {
            (0..sp.value.ncols()).all(|j| sp.value.get(i, j).is_finite())
        }));
        let a = y2.column(0);
        let b = y2.column(1);
        let cs = csd(&a, &b, &Session::new("csd", "t")).expect("csd");
        assert!(!cs.value.is_empty());
        assert!(cs.value.as_slice().iter().all(|v| v.is_finite()));
        let coh = coherence(&a, &b, &Session::new("coh", "t")).expect("coh");
        assert_eq!(coh.value.len(), cs.value.len());
        assert!(coh
            .value
            .as_slice()
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
        let imle = innovations_mle(&y, 4, &Session::new("imle", "t")).expect("imle");
        assert!(imle.value.0.is_finite());
        assert_eq!(imle.value.1.len(), 40);
        let ainn = arma_innovations(&y, 4, &Session::new("ainn", "t")).expect("ainn");
        assert_eq!(ainn.value.0.len(), 40);
        assert!(ainn.value.1.is_finite() && ainn.value.1 >= 0.0);
        let cfa = ccf_from_acf(&a, &b, 5, &Session::new("cfa", "t")).expect("cfa");
        assert_eq!(cfa.value.len(), 6);
        let pbg = pacf_burg(&y, 4, &Session::new("pbg", "t")).expect("pbg");
        assert_eq!(pbg.value.len(), 5);
        assert!((pbg.value[0] - 1.0).abs() < 1e-12);
        let ac = add_constant(&x, &Session::new("ac", "t")).expect("ac");
        assert_eq!(ac.value.shape(), (40, 2));
        let dt = detrend(&y, &Session::new("dtr", "t")).expect("dtr");
        assert_eq!(dt.value.len(), 40);
        assert!(dt.value.std() < y.std() + 1e-9);
        let conv = fftconvolve(&y, &Vector::from_slice(&[1.0, 0.0, -1.0]), &Session::new("fft", "t"))
            .expect("fft");
        assert_eq!(conv.value.len(), 42);
        assert!(conv.value.as_slice().iter().all(|v| v.is_finite()));
        let cht = ch_test(&y, 4, &Session::new("ch", "t")).expect("ch");
        assert!(cht.value.statistic.is_finite() || cht.value.pvalue.is_nan());
        let lm2 = lagmat2ds(&y2, 2, &Session::new("lm2", "t")).expect("lm2");
        assert_eq!(lm2.value.shape(), (40, 4));
        let ker = Vector::from_slice(&[0.5, 0.5]);
        let miso = miso_lfilter(&y2, &ker, &Session::new("miso", "t")).expect("miso");
        assert_eq!(miso.value.len(), 40);
        assert!(miso.value.as_slice().iter().all(|v| v.is_finite()));
        let smn = seasonal_mean(&y, 4, &Session::new("smean", "t")).expect("smean");
        assert_eq!(smn.value.len(), 4);
        assert!(smn.value.as_slice().iter().all(|v| v.is_finite()));
        let cfou = CalendarFourier::new(4, 2)
            .in_sample(40, &Session::new("cfou", "t"))
            .expect("cfou");
        assert_eq!(cfou.value.shape(), (40, 4));
        let ar1 = Vector::from_slice(&[0.5]);
        let ma0 = Vector::from_slice(&[0.0]);
        let aacf = arma_acf(&ar1, &ma0, 5, &Session::new("aacf", "t")).expect("aacf");
        assert_eq!(aacf.value.len(), 6);
        assert!((aacf.value[0] - 1.0).abs() < 1e-9);
        assert!((aacf.value[1] - 0.5).abs() < 0.05);
        let one = Vector::from_slice(&[1.0]);
        let lf = lfilter(&ker, &one, &y, &Session::new("lf", "t")).expect("lf");
        assert_eq!(lf.value.len(), 40);
        let aro = ar_select_order(&y, 3, &Session::new("arsel", "t")).expect("arsel");
        assert!(aro.value.order <= 3);
        assert_eq!(aro.value.aic.len(), 4);
        let al = add_lags(&y2, 2, &Session::new("alags", "t")).expect("alags");
        assert_eq!(al.value.shape(), (40, 6));
        let fp = freq_to_period("Q", &Session::new("ftp", "t")).expect("ftp");
        assert!((fp.value - 4.0).abs() < 1e-12);
    }
}
