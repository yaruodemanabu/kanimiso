//! Verified online estimators and streaming statistics.
//!
//! This module deliberately exposes only algorithms backed by an independent
//! oracle and transactional failure tests. Historical generated and
//! experimental online APIs are preserved in the v0.1 source archive instead
//! of being compiled into the verification-first surface.

mod autocorrelation;
mod common;
mod exponentially_weighted;
mod linear_regression;
mod moments;
mod variance_threshold;
mod weighted_mean;

#[cfg(test)]
mod tests;

pub use autocorrelation::OnlineAutoCorr;
pub use exponentially_weighted::{OnlineEwMean, OnlineEwVar};
pub use linear_regression::LinearRegression;
pub use moments::{OnlineCount, OnlineCovariance, OnlineMean, OnlineSum, OnlineVar};
pub use variance_threshold::OnlineVarianceThreshold;
pub use weighted_mean::OnlineWeightedMean;

pub(crate) use common::{finish_explain, flag_info, inspect_online_xy, reject_explain};
