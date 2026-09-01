//! scikit-learn-style API: `Estimator`, `Simulator`, `Dataset`, recovery.

use crate::error::{Error, Result};
use crate::infer::{qmle, Fit};
use crate::model::ParametricSde;
use crate::optimize::OptOptions;
use crate::path::Path;

/// Fit on process data and expose parameters (sklearn `estimator.fit`).
pub trait Estimator {
    type Data;
    fn fit(&mut self, data: &Self::Data) -> Result<()>;
    fn fitted_params(&self) -> Result<&[f64]>;
    fn fitted_names(&self) -> &[&'static str];
}

/// Something that can emit a path.
pub trait Simulator {
    fn simulate_path<R: amatsuki::Rng + ?Sized>(&self, rng: &mut R) -> Result<Path>;
}

/// Euler simulator for any [`crate::model::Sde`].
pub struct EulerSimulator<M> {
    pub model: M,
    pub sampling: crate::sampling::Sampling,
    pub x0: Vec<f64>,
    pub cfg: crate::simulate::SimConfig,
}

impl<M: crate::model::Sde> Simulator for EulerSimulator<M> {
    fn simulate_path<R: amatsuki::Rng + ?Sized>(&self, rng: &mut R) -> Result<Path> {
        crate::simulate::simulate(&self.model, &self.sampling, &self.x0, rng, &self.cfg)
    }
}

/// QMLE wrapped as an estimator.
#[derive(Clone, Debug)]
pub struct QmleEstimator<F: ParametricSde> {
    pub family: F,
    pub start: Vec<f64>,
    pub lower: Option<Vec<f64>>,
    pub upper: Option<Vec<f64>>,
    pub opt: OptOptions,
    pub fit: Option<Fit>,
}

impl<F: ParametricSde> QmleEstimator<F> {
    pub fn new(family: F, start: Vec<f64>) -> Self {
        Self {
            family,
            start,
            lower: None,
            upper: None,
            opt: OptOptions::default(),
            fit: None,
        }
    }

    pub fn bounds(mut self, lower: Vec<f64>, upper: Vec<f64>) -> Self {
        self.lower = Some(lower);
        self.upper = Some(upper);
        self
    }
}

impl<F: ParametricSde> Estimator for QmleEstimator<F> {
    type Data = Path;
    fn fit(&mut self, data: &Path) -> Result<()> {
        let lo = self.lower.as_deref();
        let hi = self.upper.as_deref();
        self.fit = Some(qmle(
            &self.family,
            data,
            &self.start,
            lo,
            hi,
            self.opt.clone(),
        )?);
        Ok(())
    }
    fn fitted_params(&self) -> Result<&[f64]> {
        self.fit
            .as_ref()
            .map(|f| f.params.as_slice())
            .ok_or_else(|| Error::infer("estimator has not been fit"))
    }
    fn fitted_names(&self) -> &[&'static str] {
        self.family.param_names()
    }
}

/// Parameter-recovery report (the sklearn “score on a known toy”).
#[derive(Clone, Debug)]
pub struct RecoveryReport {
    pub names: Vec<String>,
    pub truth: Vec<f64>,
    pub fitted: Vec<f64>,
    pub abs_error: Vec<f64>,
    pub max_abs_error: f64,
    pub quasi_loglik: f64,
}

impl RecoveryReport {
    pub fn new(names: &[&str], truth: &[f64], fitted: &[f64], ql: f64) -> Self {
        let abs_error: Vec<f64> = truth
            .iter()
            .zip(fitted.iter())
            .map(|(t, f)| (t - f).abs())
            .collect();
        let max_abs_error = abs_error.iter().copied().fold(0.0, f64::max);
        Self {
            names: names.iter().map(|s| (*s).to_string()).collect(),
            truth: truth.to_vec(),
            fitted: fitted.to_vec(),
            abs_error,
            max_abs_error,
            quasi_loglik: ql,
        }
    }
}

/// Fit `estimator` and compare against known parameters.
pub fn recover<E: Estimator<Data = Path>>(
    estimator: &mut E,
    path: &Path,
    truth: &[f64],
) -> Result<RecoveryReport> {
    estimator.fit(path)?;
    let fitted = estimator.fitted_params()?.to_vec();
    if fitted.len() != truth.len() {
        return Err(Error::dim("truth / fitted length mismatch"));
    }
    Ok(RecoveryReport::new(
        estimator.fitted_names(),
        truth,
        &fitted,
        f64::NAN,
    ))
}

/// Tiny pipeline: simulate → estimate → recover.
pub struct Pipeline<S, E> {
    pub simulator: S,
    pub estimator: E,
}

impl<S, E> Pipeline<S, E>
where
    S: Simulator,
    E: Estimator<Data = Path>,
{
    pub fn run<R: amatsuki::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        truth: &[f64],
    ) -> Result<(Path, RecoveryReport)> {
        let path = self.simulator.simulate_path(rng)?;
        let report = recover(&mut self.estimator, &path, truth)?;
        Ok((path, report))
    }
}

/// Expanding-window time-series splits (sklearn `TimeSeriesSplit`).
///
/// Train is always a prefix of the test block. Future observations are
/// never used for training (no look-ahead). The first block of length
/// `n / (k+1)` is the initial training window, so this returns `k` folds
/// and requires `n ≥ k + 1`.
pub fn time_series_folds(n: usize, k: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if k < 2 || n < k + 1 {
        return Err(Error::infer("need n ≥ k+1 and k ≥ 2 for expanding folds"));
    }
    let fold = n / (k + 1);
    if fold == 0 {
        return Err(Error::infer("folds are empty"));
    }
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let te0 = (i + 1) * fold;
        let te1 = if i + 1 == k { n } else { (i + 2) * fold };
        let test: Vec<usize> = (te0..te1).collect();
        let train: Vec<usize> = (0..te0).collect();
        out.push((train, test));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_cover() {
        let f = time_series_folds(12, 3).unwrap();
        assert_eq!(f.len(), 3);
        for (train, test) in &f {
            assert!(!train.is_empty() && !test.is_empty());
            let tmax = *train.iter().max().unwrap();
            assert!(
                test.iter().all(|&j| j > tmax),
                "look-ahead in {train:?} / {test:?}"
            );
        }
    }
}
