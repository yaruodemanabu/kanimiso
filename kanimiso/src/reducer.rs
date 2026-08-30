//! sktime-style reduction: lag features plus recursive multi-step forecasts.
//!
//! A window shorter than the lag order cannot identify the autoregression.
//! Recursive forecasts past the identified horizon are marked
//! [`IssueCode::ForecastHorizonExceedsIdentifiability`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::least_squares;
use crate::traits::FitSeries;
use crate::validate::inspect_identification;
use ojizou_san::Session;
use signlred::{
    Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result, Severity,
};

/// Recursive reduction: \(y_t \sim y_{t-1},\ldots,y_{t-p}\).
#[derive(Clone, Debug)]
pub struct RecursiveReducer {
    /// Autoregressive order (window length).
    pub window: usize,
}

impl Default for RecursiveReducer {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl RecursiveReducer {
    /// Reducer with lag window `p`.
    pub fn new(window: usize) -> Self {
        Self { window }
    }
}

/// Fitted recursive reducer.
#[derive(Clone, Debug)]
pub struct FittedReducer {
    /// Lag coefficients (oldest lag first).
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
    last: Vector,
    /// Lag order.
    pub window: usize,
}

impl FittedReducer {
    /// Recursive `h`-step forecast from the last training window.
    pub fn forecast(&self, horizon: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        if horizon == 0 {
            return ctx.finish(Vector::zeros(0));
        }
        if horizon > self.window.saturating_mul(4).max(8) {
            ctx.push(
                Issue::builder(IssueCode::ForecastHorizonExceedsIdentifiability)
                    .message(format!(
                        "recursive horizon {horizon} ≫ window {}; later steps are iterated noise",
                        self.window
                    ))
                    .meaninglessness(Meaninglessness::new(
                        "recursive multi-step path",
                        "each step feeds predictions into the next lag window",
                        signlred::InterpretiveValue::Misleading,
                        "treat only the first few steps as identified",
                    ))
                    .build(),
            );
        }
        let mut hist = self.last.as_slice().to_vec();
        let mut out = Vector::zeros(horizon);
        for h in 0..horizon {
            let row = Matrix::from_fn(1, self.window, |_, j| {
                hist[hist.len().saturating_sub(self.window) + j]
            });
            let yhat = if row.ncols() == self.coef.len() {
                let mut s = self.intercept;
                for j in 0..self.coef.len() {
                    s += self.coef[j] * row.get(0, j);
                }
                s
            } else {
                hist.last().copied().unwrap_or(0.0)
            };
            out[h] = yhat;
            hist.push(yhat);
        }
        ctx.finish(out)
    }
}

impl FitSeries for RecursiveReducer {
    type Fitted = FittedReducer;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<FittedReducer>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let p = self.window.max(1);
        if y.len() <= p {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "RecursiveReducer window {p} needs more than {p} samples (n={})",
                        y.len()
                    ))
                    .meaninglessness(Meaninglessness::vacuous(
                        "lag autoregression",
                        "n ≤ p leaves every lag column unidentified",
                        "lengthen the series or shorten the window",
                    ))
                    .build(),
            );
            return ctx.finish(FittedReducer {
                coef: Vector::zeros(p),
                intercept: y.mean(),
                last: y.clone(),
                window: p,
            });
        }
        inspect_identification(&mut ctx.report, y.len() - p, p, &ctx.policy);
        let n = y.len() - p;
        let x = Matrix::from_fn(n, p, |i, j| y[i + j]);
        let target = Vector::from_iter((0..n).map(|i| y[i + p]));
        let design = x.with_intercept();
        let mut scratch = signlred::Report::new("reducer", "ols");
        let beta = least_squares(&mut scratch, &design, &target, &ctx.policy);
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::PerfectCollinearity
                    | IssueCode::NearSingular
                    | IssueCode::SingularMatrix
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        if scratch.contains(IssueCode::PerfectCollinearity)
            || scratch.contains(IssueCode::NearSingular)
            || scratch.contains(IssueCode::SingularMatrix)
        {
            ctx.push(
                Issue::builder(IssueCode::HighMulticollinearity)
                    .message("lag columns of a trending series are nearly collinear; the reducer is a difference equation, not a set of partial effects")
                    .meaninglessness(Meaninglessness::new(
                        "lag coefficients",
                        "consecutive levels of a trend share one direction",
                        signlred::InterpretiveValue::Misleading,
                        "forecast the path; do not interpret individual lag coefficients",
                    ))
                    .build(),
            );
        }
        let (intercept, coef) = match beta {
            Some(b) => (b[0], Vector::from_iter((1..b.len()).map(|j| b[j]))),
            None => {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("recursive reducer OLS failed")
                        .build(),
                );
                (target.mean(), Vector::zeros(p))
            }
        };
        ctx.push(
            Issue::builder(IssueCode::AutocorrelatedResiduals)
                .severity(Severity::Advisory)
                .message("recursive reduction treats lagged y as exogenous; residual ACF is not a specification test of the original series")
                .compromise(NumericalCompromise::new(
                    "direct multi-horizon regression",
                    "one-step OLS iterated h times",
                    "prediction error is reused as a regressor",
                    "interval forecasts understate iterated uncertainty",
                ))
                .build(),
        );
        let last = Vector::from_iter(y.as_slice()[y.len() - p..].iter().copied());
        ctx.finish(FittedReducer {
            coef,
            intercept,
            last,
            window: p,
        })
    }
}

/// Direct multi-horizon reduction: one lag-OLS per forecast step.
#[derive(Clone, Debug)]
pub struct DirectReducer {
    /// Lag window.
    pub window: usize,
    /// Number of direct horizons to identify.
    pub horizon: usize,
}

impl Default for DirectReducer {
    fn default() -> Self {
        Self {
            window: 3,
            horizon: 3,
        }
    }
}

impl DirectReducer {
    /// Direct reducer with lag `window` and `horizon` models.
    pub fn new(window: usize, horizon: usize) -> Self {
        Self { window, horizon }
    }
}

/// Fitted direct reducer.
#[derive(Clone, Debug)]
pub struct FittedDirectReducer {
    /// `(coef, intercept)` per horizon (1-based order).
    pub models: Vec<(Vector, f64)>,
    last: Vector,
    /// Lag order.
    pub window: usize,
}

impl FittedDirectReducer {
    /// Direct `h`-step forecast from the last training window (not recursive).
    pub fn forecast(&self, horizon: usize, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("forecast"));
        if horizon == 0 {
            return ctx.finish(Vector::zeros(0));
        }
        if horizon > self.models.len() {
            ctx.push(
                Issue::builder(IssueCode::ForecastHorizonExceedsIdentifiability)
                    .message(format!(
                        "requested horizon {horizon} > identified {}",
                        self.models.len()
                    ))
                    .meaninglessness(Meaninglessness::new(
                        "direct multi-horizon path",
                        "horizons past the fitted set have no model",
                        signlred::InterpretiveValue::Misleading,
                        "only the first fitted horizons are identified",
                    ))
                    .build(),
            );
        }
        let mut out = Vector::zeros(horizon);
        for h in 0..horizon {
            if h >= self.models.len() {
                out[h] = *self.last.as_slice().last().unwrap_or(&0.0);
                continue;
            }
            let (coef, intercept) = &self.models[h];
            let mut s = *intercept;
            let start = self.last.len().saturating_sub(self.window);
            for j in 0..coef.len().min(self.window) {
                s += coef[j] * self.last[start + j];
            }
            out[h] = s;
        }
        ctx.finish(out)
    }
}

impl FitSeries for DirectReducer {
    type Fitted = FittedDirectReducer;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedDirectReducer>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let p = self.window.max(1);
        let hmax = self.horizon.max(1);
        if y.len() <= p + hmax {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "DirectReducer window {p} horizon {hmax} needs n>{} (n={})",
                        p + hmax,
                        y.len()
                    ))
                    .meaninglessness(Meaninglessness::vacuous(
                        "direct horizon regressions",
                        "n ≤ p+H leaves a horizon without rows",
                        "lengthen the series or shorten the window/horizon",
                    ))
                    .build(),
            );
            return ctx.finish(FittedDirectReducer {
                models: vec![(Vector::zeros(p), y.mean()); hmax],
                last: y.clone(),
                window: p,
            });
        }
        let mut models = Vec::with_capacity(hmax);
        for h in 1..=hmax {
            let n = y.len().saturating_sub(p + h - 1);
            if n == 0 {
                models.push((Vector::zeros(p), y.mean()));
                continue;
            }
            let x = Matrix::from_fn(n, p, |i, j| y[i + j]);
            let target = Vector::from_iter((0..n).map(|i| y[i + p + h - 1]));
            let design = x.with_intercept();
            let mut scratch = signlred::Report::new("direct", "ols");
            let beta = least_squares(&mut scratch, &design, &target, &ctx.policy);
            for issue in scratch.issues() {
                if matches!(
                    issue.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::PerfectCollinearity
                        | IssueCode::NearSingular
                        | IssueCode::SingularMatrix
                ) {
                    continue;
                }
                ctx.push(issue.clone());
            }
            if scratch.contains(IssueCode::PerfectCollinearity)
                || scratch.contains(IssueCode::NearSingular)
                || scratch.contains(IssueCode::SingularMatrix)
            {
                ctx.push(
                    Issue::builder(IssueCode::HighMulticollinearity)
                        .message(format!(
                            "direct horizon {h}: lag columns of a trend are nearly collinear"
                        ))
                        .meaninglessness(Meaninglessness::new(
                            "lag coefficients",
                            "consecutive levels of a trend share one direction",
                            signlred::InterpretiveValue::Misleading,
                            "forecast the path; do not interpret individual lag coefficients",
                        ))
                        .build(),
                );
            }
            match beta {
                Some(b) => models.push((
                    Vector::from_iter((1..b.len()).map(|j| b[j])),
                    b.as_slice().first().copied().unwrap_or(0.0),
                )),
                None => models.push((Vector::zeros(p), target.mean())),
            }
        }
        ctx.push(
            Issue::builder(IssueCode::AutocorrelatedResiduals)
                .severity(Severity::Advisory)
                .message("direct reduction fits each horizon independently; residual ACF is not a joint specification test")
                .compromise(NumericalCompromise::new(
                    "joint multi-horizon likelihood",
                    "separate OLS per horizon",
                    "horizons do not share parameters",
                    "the H models are not a single identified dynamic system",
                ))
                .build(),
        );
        let last = Vector::from_iter(y.as_slice()[y.len() - p..].iter().copied());
        ctx.finish(FittedDirectReducer {
            models,
            last,
            window: p,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reducer_follows_a_ramp() {
        let y = Vector::from_iter((0..24).map(|i| i as f64));
        let q = RecursiveReducer::new(2)
            .fit_series(&y, &Session::new("red", "fit"))
            .expect("red");
        let fc = q
            .value
            .forecast(3, &Session::new("red", "fc"))
            .unwrap()
            .value;
        assert_eq!(fc.len(), 3);
        assert!((fc[0] - 24.0).abs() < 1.0, "{:?}", fc.as_slice());
    }

    #[test]
    fn direct_reducer_follows_a_ramp() {
        let y = Vector::from_iter((0..24).map(|i| i as f64));
        let q = DirectReducer::new(2, 3)
            .fit_series(&y, &Session::new("dir", "fit"))
            .expect("dir");
        let fc = q
            .value
            .forecast(3, &Session::new("dir", "fc"))
            .unwrap()
            .value;
        assert_eq!(fc.len(), 3);
        assert!((fc[0] - 24.0).abs() < 1.5, "{:?}", fc.as_slice());
    }
}
