//! Combined model + sampling + data object (`setYuima` in YUIMA).

use crate::error::{Error, Result};
use crate::model::Sde;
use crate::path::Path;
use crate::sampling::Sampling;
use crate::scheme::Scheme;
use crate::simulate::{simulate, SimConfig};

/// A YUIMA-style container: model, sampling scheme, and optional data.
#[derive(Clone, Debug)]
pub struct Yuima<M> {
    pub model: M,
    pub sampling: Sampling,
    pub data: Option<Path>,
    pub xinit: Vec<f64>,
}

impl<M: Sde> Yuima<M> {
    pub fn new(model: M, sampling: Sampling, xinit: Vec<f64>) -> Result<Self> {
        model.validate()?;
        if xinit.len() != model.dim() {
            return Err(Error::dim("xinit length must equal model dim"));
        }
        Ok(Self {
            model,
            sampling,
            data: None,
            xinit,
        })
    }

    pub fn with_data(mut self, data: Path) -> Result<Self> {
        if data.dim() != self.model.dim() {
            return Err(Error::dim("data dim != model dim"));
        }
        self.data = Some(data);
        Ok(self)
    }

    /// Simulate and store the path (YUIMA `simulate`).
    pub fn simulate<R: amatsuki::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        scheme: Scheme,
    ) -> Result<&Path> {
        let cfg = SimConfig {
            scheme,
            ..SimConfig::default()
        };
        let path = simulate(&self.model, &self.sampling, &self.xinit, rng, &cfg)?;
        self.data = Some(path);
        Ok(self.data.as_ref().unwrap())
    }

    pub fn data(&self) -> Result<&Path> {
        self.data
            .as_ref()
            .ok_or_else(|| Error::infer("Yuima object has no data; call simulate first"))
    }
}
