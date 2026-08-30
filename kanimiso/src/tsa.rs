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
use crate::linalg::{chol_solve, thin_svd};

pub use crate::filters::{bk_filter, cf_filter, FittedLocalLinearTrend, LocalLinearTrend};
use crate::traits::FitSeries;
use crate::validate::{inspect_identification, inspect_xy};
use ojizou_san::Session;
use signlred::{
    scan_finite, slice_stats, Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified,
    Report, Result, Severity,
};

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
        let col = ColumnEnsembleForecaster::new(1)
            .fit(&y2, &Session::new("col", "fit"))
            .expect("colens");
        let cf = col
            .value
            .forecast(3, &Session::new("col", "fc"))
            .expect("colf")
            .value;
        assert_eq!(cf.shape(), (3, 2));
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
    }
}
