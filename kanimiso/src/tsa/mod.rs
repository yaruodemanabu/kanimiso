//! Verified and strongly checked time-series and volatility kernels.
//!
//! The active v0.2 surface is intentionally small: exact ARMA recurrences,
//! fixed-order conditional-variance models, and the shared verified filters.
//! Historical forecasting and generated compatibility APIs remain searchable
//! in `generated-v0.1-archive/tsa.rs.txt`.

mod arma;
mod common;
mod egarch;
mod ewma;
mod fiegarch;
mod figarch;
mod garch;

pub use arma::{
    arma2ar, arma2ma, arma_acf, arma_acovf, arma_generate_sample, arma_impulse_response,
};
pub use egarch::{Egarch, FittedEgarch};
pub use ewma::{EwmaVol, FittedEwmaVol};
pub use fiegarch::{Fiegarch, FittedFiegarch};
pub use figarch::{Figarch, FittedFigarch};
pub use garch::{FittedGarch11, Garch11};

pub use crate::filters::{
    bk_filter, cf_filter, lfilter, miso_lfilter, FittedLocalLinearTrend, LocalLinearTrend,
};
