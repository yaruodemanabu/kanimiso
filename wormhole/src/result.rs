//! Common solver result structures.

use faer::Mat;

/// Termination status of an iterative numerical method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverStatus {
    /// The requested stopping criterion was met.
    Converged,
    /// The iteration limit was reached before convergence.
    IterationLimit,
}

/// Dual source and target potentials.
#[derive(Clone, Debug, PartialEq)]
pub struct DualPotentials {
    /// Potential associated with source marginal constraints.
    pub source: Vec<f64>,
    /// Potential associated with target marginal constraints.
    pub target: Vec<f64>,
}

/// A transport plan and diagnostics.
#[derive(Clone, Debug)]
pub struct TransportPlan {
    /// Dense coupling; rows correspond to source atoms.
    pub plan: Mat<f64>,
    /// Linear transport value `sum(plan * cost)`.
    pub value: f64,
    /// Optional dual potentials when the selected solver exposes them.
    pub potentials: Option<DualPotentials>,
    /// Number of outer solver iterations.
    pub iterations: usize,
    /// Last marginal or fixed-point residual.
    pub residual: f64,
    /// Solver termination status.
    pub status: SolverStatus,
}

impl TransportPlan {
    /// Row marginals of the coupling.
    pub fn source_marginal(&self) -> Vec<f64> {
        (0..self.plan.nrows())
            .map(|i| (0..self.plan.ncols()).map(|j| self.plan[(i, j)]).sum())
            .collect()
    }

    /// Column marginals of the coupling.
    pub fn target_marginal(&self) -> Vec<f64> {
        (0..self.plan.ncols())
            .map(|j| (0..self.plan.nrows()).map(|i| self.plan[(i, j)]).sum())
            .collect()
    }
}

/// An estimated barycenter distribution and convergence diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct BarycenterResult {
    /// Barycenter weights on the shared support.
    pub weights: Vec<f64>,
    /// Number of iterations completed.
    pub iterations: usize,
    /// Last fixed-point residual.
    pub residual: f64,
    /// Solver termination status.
    pub status: SolverStatus,
}
