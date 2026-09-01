//! Shared constructors for named models.

use crate::error::{Error, Result};

pub(crate) fn require_pos(name: &str, x: f64) -> Result<()> {
    if x > 0.0 && x.is_finite() {
        Ok(())
    } else {
        Err(Error::param(format!("{name} must be positive")))
    }
}

pub(crate) fn require_nonneg(name: &str, x: f64) -> Result<()> {
    if x >= 0.0 && x.is_finite() {
        Ok(())
    } else {
        Err(Error::param(format!("{name} must be non-negative")))
    }
}

pub(crate) fn require_unit(name: &str, x: f64) -> Result<()> {
    if (0.0..=1.0).contains(&x) {
        Ok(())
    } else {
        Err(Error::param(format!("{name} must lie in [0, 1]")))
    }
}
