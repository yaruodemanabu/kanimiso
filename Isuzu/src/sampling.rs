//! Time grids (`setSampling` in YUIMA).

use crate::error::{Error, Result};

/// Observation / simulation grid.
///
/// Regular grids are stored compactly; irregular grids keep every timestamp.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    times: Vec<f64>,
}

impl Sampling {
    /// Regular grid on `[initial, terminal]` with `n` steps (`n + 1` nodes).
    pub fn regular(initial: f64, terminal: f64, n: usize) -> Result<Self> {
        if !initial.is_finite() || !terminal.is_finite() {
            return Err(Error::sampling("initial and terminal must be finite"));
        }
        if terminal <= initial {
            return Err(Error::sampling("terminal must exceed initial"));
        }
        if n == 0 {
            return Err(Error::sampling("n must be positive"));
        }
        let dt = (terminal - initial) / n as f64;
        let mut times = Vec::with_capacity(n + 1);
        for i in 0..=n {
            times.push(initial + dt * i as f64);
        }
        times[n] = terminal;
        Ok(Self { times })
    }

    /// YUIMA-style constructor: `Terminal` horizon, `n` steps, start at 0.
    pub fn from_terminal(terminal: f64, n: usize) -> Result<Self> {
        Self::regular(0.0, terminal, n)
    }

    /// Arbitrary increasing timestamps (irregular sampling).
    pub fn irregular(times: Vec<f64>) -> Result<Self> {
        if times.len() < 2 {
            return Err(Error::sampling("need at least two timestamps"));
        }
        for w in times.windows(2) {
            if !w[0].is_finite() || !w[1].is_finite() {
                return Err(Error::sampling("timestamps must be finite"));
            }
            if w[1] <= w[0] {
                return Err(Error::sampling("timestamps must be strictly increasing"));
            }
        }
        Ok(Self { times })
    }

    /// Number of intervals.
    pub fn n_steps(&self) -> usize {
        self.times.len().saturating_sub(1)
    }

    /// Number of nodes including the initial time.
    pub fn n_nodes(&self) -> usize {
        self.times.len()
    }

    pub fn initial(&self) -> f64 {
        self.times[0]
    }

    pub fn terminal(&self) -> f64 {
        self.times[self.times.len() - 1]
    }

    pub fn horizon(&self) -> f64 {
        self.terminal() - self.initial()
    }

    /// Mean step size.
    pub fn mean_delta(&self) -> f64 {
        self.horizon() / self.n_steps() as f64
    }

    pub fn times(&self) -> &[f64] {
        &self.times
    }

    pub fn delta(&self, i: usize) -> f64 {
        self.times[i + 1] - self.times[i]
    }

    /// Whether every step equals the first (up to a relative tolerance).
    pub fn is_regular(&self) -> bool {
        if self.n_steps() == 0 {
            return true;
        }
        let dt = self.delta(0);
        self.times.windows(2).all(|w| {
            let d = w[1] - w[0];
            (d - dt).abs() <= 1e-12 * (1.0 + dt.abs())
        })
    }

    /// Subsample every `k`-th node (keeps the last node).
    pub fn subsample(&self, k: usize) -> Result<Self> {
        if k == 0 {
            return Err(Error::sampling("subsample stride must be positive"));
        }
        let mut times = self
            .times
            .iter()
            .enumerate()
            .filter(|(i, _)| i % k == 0)
            .map(|(_, t)| *t)
            .collect::<Vec<_>>();
        if times.last() != self.times.last() {
            times.push(self.terminal());
        }
        Self::irregular(times)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_nodes() {
        let s = Sampling::from_terminal(1.0, 100).unwrap();
        assert_eq!(s.n_steps(), 100);
        assert_eq!(s.n_nodes(), 101);
        assert!((s.mean_delta() - 0.01).abs() < 1e-14);
        assert!(s.is_regular());
    }

    #[test]
    fn irregular_rejects_non_monotone() {
        assert!(Sampling::irregular(vec![0.0, 0.5, 0.4]).is_err());
    }
}
