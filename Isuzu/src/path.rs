//! Observed and simulated trajectories.

use crate::error::{Error, Result};
use crate::sampling::Sampling;

/// One trajectory of a `dim`-dimensional process on a time grid.
///
/// Values are stored row-major: `values[i * dim + j]` is coordinate `j` at
/// node `i`.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    times: Vec<f64>,
    values: Vec<f64>,
    dim: usize,
}

impl Path {
    pub fn new(times: Vec<f64>, values: Vec<f64>, dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::dim("state dimension must be positive"));
        }
        if times.len() < 2 {
            return Err(Error::sampling("path needs at least two nodes"));
        }
        if values.len() != times.len() * dim {
            return Err(Error::dim(format!(
                "values length {} != n_nodes {} × dim {}",
                values.len(),
                times.len(),
                dim
            )));
        }
        for (i, &t) in times.iter().enumerate() {
            if !t.is_finite() {
                return Err(Error::sampling("path times must be finite"));
            }
            if i > 0 && times[i] <= times[i - 1] {
                return Err(Error::sampling("path times must be strictly increasing"));
            }
        }
        if values.iter().any(|v| v.is_infinite()) {
            return Err(Error::sampling(
                "path values must not be infinite (NaN marks a missing observation)",
            ));
        }
        Ok(Self { times, values, dim })
    }

    pub fn from_sampling(sampling: &Sampling, values: Vec<f64>, dim: usize) -> Result<Self> {
        Self::new(sampling.times().to_vec(), values, dim)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn n_nodes(&self) -> usize {
        self.times.len()
    }

    pub fn n_steps(&self) -> usize {
        self.times.len() - 1
    }

    pub fn times(&self) -> &[f64] {
        &self.times
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn sampling(&self) -> Result<Sampling> {
        Sampling::irregular(self.times.clone())
    }

    pub fn state(&self, i: usize) -> &[f64] {
        let o = i * self.dim;
        &self.values[o..o + self.dim]
    }

    pub fn state_mut(&mut self, i: usize) -> &mut [f64] {
        let o = i * self.dim;
        &mut self.values[o..o + self.dim]
    }

    /// Univariate series (requires `dim == 1`).
    pub fn as_univariate(&self) -> Result<Vec<f64>> {
        if self.dim != 1 {
            return Err(Error::dim("as_univariate requires dim = 1"));
        }
        Ok(self.values.clone())
    }

    /// Extract coordinate `j` along the whole path.
    pub fn component(&self, j: usize) -> Result<Vec<f64>> {
        if j >= self.dim {
            return Err(Error::dim("component index out of range"));
        }
        Ok((0..self.n_nodes())
            .map(|i| self.values[i * self.dim + j])
            .collect())
    }

    /// Increments of coordinate `j`.
    pub fn increments(&self, j: usize) -> Result<Vec<f64>> {
        let x = self.component(j)?;
        Ok(x.windows(2).map(|w| w[1] - w[0]).collect())
    }

    pub fn terminal(&self) -> &[f64] {
        self.state(self.n_nodes() - 1)
    }

    pub fn initial(&self) -> &[f64] {
        self.state(0)
    }

    /// Restrict to a time window `[t0, t1]` (inclusive on both ends that exist).
    pub fn window(&self, t0: f64, t1: f64) -> Result<Self> {
        if t1 <= t0 {
            return Err(Error::sampling("window requires t1 > t0"));
        }
        let idx: Vec<usize> = self
            .times
            .iter()
            .enumerate()
            .filter(|(_, t)| **t >= t0 && **t <= t1)
            .map(|(i, _)| i)
            .collect();
        if idx.len() < 2 {
            return Err(Error::sampling("window has fewer than two nodes"));
        }
        let times = idx.iter().map(|&i| self.times[i]).collect();
        let mut values = Vec::with_capacity(idx.len() * self.dim);
        for &i in &idx {
            values.extend_from_slice(self.state(i));
        }
        Self::new(times, values, self.dim)
    }

    /// Realized quadratic variation of coordinate `j`.
    pub fn quadratic_variation(&self, j: usize) -> Result<f64> {
        Ok(self.increments(j)?.iter().map(|d| d * d).sum())
    }
}

/// Collection of independent simulated paths (same model and grid).
#[derive(Clone, Debug)]
pub struct Ensemble {
    pub paths: Vec<Path>,
}

impl Ensemble {
    pub fn new(paths: Vec<Path>) -> Result<Self> {
        if paths.is_empty() {
            return Err(Error::sim("ensemble is empty"));
        }
        let dim = paths[0].dim();
        let n = paths[0].n_nodes();
        for p in &paths {
            if p.dim() != dim || p.n_nodes() != n {
                return Err(Error::dim("ensemble paths must share dim and length"));
            }
        }
        Ok(Self { paths })
    }

    pub fn n_paths(&self) -> usize {
        self.paths.len()
    }

    /// Mean of coordinate `j` at the terminal time.
    pub fn terminal_mean(&self, j: usize) -> Result<f64> {
        let mut s = 0.0;
        for p in &self.paths {
            let c = p.component(j)?;
            s += c[c.len() - 1];
        }
        Ok(s / self.paths.len() as f64)
    }

    pub fn terminal_var(&self, j: usize) -> Result<f64> {
        let m = self.terminal_mean(j)?;
        let mut s = 0.0;
        for p in &self.paths {
            let c = p.component(j)?;
            let x = c[c.len() - 1] - m;
            s += x * x;
        }
        Ok(s / self.paths.len() as f64)
    }
}

/// One irregularly sampled univariate series (asynchronous / tick data).
#[derive(Clone, Debug, PartialEq)]
pub struct TickSeries {
    pub times: Vec<f64>,
    pub values: Vec<f64>,
}

impl TickSeries {
    pub fn new(times: Vec<f64>, values: Vec<f64>) -> Result<Self> {
        if times.len() != values.len() || times.len() < 2 {
            return Err(Error::dim("tick series needs matching times/values, n ≥ 2"));
        }
        for w in times.windows(2) {
            if w[1] <= w[0] {
                return Err(Error::sampling("tick times must be strictly increasing"));
            }
        }
        Ok(Self { times, values })
    }

    pub fn n(&self) -> usize {
        self.times.len()
    }

    pub fn increments(&self) -> Vec<f64> {
        self.values.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Shift every timestamp by `theta` (lead-lag experiments).
    pub fn shift_time(&self, theta: f64) -> Self {
        Self {
            times: self.times.iter().map(|t| t + theta).collect(),
            values: self.values.clone(),
        }
    }
}

/// Several possibly asynchronous series (YUIMA `yuima.data`).
#[derive(Clone, Debug)]
pub struct AsyncData {
    pub series: Vec<TickSeries>,
}

impl AsyncData {
    pub fn new(series: Vec<TickSeries>) -> Result<Self> {
        if series.is_empty() {
            return Err(Error::dim("AsyncData needs at least one series"));
        }
        Ok(Self { series })
    }

    pub fn from_path(path: &Path) -> Result<Self> {
        let mut series = Vec::with_capacity(path.dim());
        for j in 0..path.dim() {
            series.push(TickSeries::new(path.times().to_vec(), path.component(j)?)?);
        }
        Self::new(series)
    }

    pub fn dim(&self) -> usize {
        self.series.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_component() {
        let p = Path::new(
            vec![0.0, 1.0, 2.0],
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            2,
        )
        .unwrap();
        assert_eq!(p.component(1).unwrap(), vec![10.0, 20.0, 30.0]);
        assert!((p.quadratic_variation(0).unwrap() - 2.0).abs() < 1e-14);
    }
}
