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
use crate::linalg::chol_solve;

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
        if issue.code == IssueCode::ResidualTooLarge {
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
}
