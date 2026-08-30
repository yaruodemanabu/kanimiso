//! Estimator traits. Every method takes an [`ojizou_san::Session`].

use crate::data::{Matrix, Vector};
use ojizou_san::{IncrementalExplain, Session};
use signlred::{Qualified, Result};

/// Batch fit `y ~ X`.
pub trait Fit {
    /// Fitted object stored on the estimator or returned.
    type Fitted;
    /// Fit and return a quality-qualified fitted model.
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session)
        -> Result<Qualified<Self::Fitted>>;
}

/// Unsupervised fit on `X` only.
pub trait FitUnsupervised {
    /// Fitted object.
    type Fitted;
    /// Fit on features alone.
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<Self::Fitted>>;
}

/// Predict from a fitted model.
pub trait Predict {
    /// Prediction type (`Vector` of scores, labels, or forecasts).
    type Output;
    /// Predict.
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Self::Output>>;
}

/// Feature transform.
pub trait Transform {
    /// Transformed matrix.
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>>;
}

/// Incremental / online update. **Must** return explainability.
///
/// A `partial_fit` that only mutates parameters and returns `()` is a contract
/// violation: additional learning has to say *what moved, why, and whether the
/// new state is identified*.
pub trait PartialFit {
    /// Update on a batch and explain the change.
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>>;
}

/// 1-d series fit (time series, HMM observations as a vector, …).
pub trait FitSeries {
    /// Fitted object.
    type Fitted;
    /// Fit on a univariate series.
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<Self::Fitted>>;
}
