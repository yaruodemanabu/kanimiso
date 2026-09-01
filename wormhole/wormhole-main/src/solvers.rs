//! Unified dispatch over linear optimal-transport solvers.

use crate::error::{Error, Result};
use crate::exact;
use crate::metrics::{self, Metric};
use crate::partial::{self, EntropicPartialOptions};
use crate::result::TransportPlan;
use crate::sinkhorn::{self, SinkhornMethod, SinkhornOptions};
use crate::unbalanced::{self, UnbalancedOptions};
use crate::validate;
use faer::Mat;

/// Convex regularization of a linear transport problem.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Regularization {
    /// Unregularized linear transport.
    #[default]
    None,
    /// Negative-entropy regularization with positive strength.
    Entropy(f64),
}

/// Constraint imposed on source and target marginals.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum MarginalConstraint {
    /// Both marginals must be matched exactly.
    #[default]
    Balanced,
    /// Marginals are relaxed with asymmetric generalized-KL penalties.
    UnbalancedKl {
        /// Source-marginal KL penalty.
        source_penalty: f64,
        /// Target-marginal KL penalty.
        target_penalty: f64,
    },
    /// Move exactly this mass without exceeding either marginal.
    Partial {
        /// Prescribed transported mass.
        mass: f64,
    },
}

/// Explicit algorithm selection for a unified linear solve.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LinearMethod {
    /// Select a solver from regularization and marginal constraints.
    #[default]
    Auto,
    /// Exact min-cost flow.
    Exact,
    /// Multiplicative Sinkhorn scaling.
    SinkhornScaling,
    /// Log-domain Sinkhorn.
    SinkhornLog,
}

/// Unified linear transport options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveOptions {
    /// Plan regularization.
    pub regularization: Regularization,
    /// Marginal constraint or relaxation.
    pub marginal: MarginalConstraint,
    /// Solver override.
    pub method: LinearMethod,
    /// Maximum iterations for iterative methods.
    pub max_iterations: usize,
    /// Stopping tolerance for iterative methods.
    pub tolerance: f64,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            regularization: Regularization::None,
            marginal: MarginalConstraint::Balanced,
            method: LinearMethod::Auto,
            max_iterations: 1_000,
            tolerance: 1e-9,
        }
    }
}

fn marginals<'a>(
    cost: &Mat<f64>,
    source: Option<&'a [f64]>,
    target: Option<&'a [f64]>,
    source_storage: &'a mut Vec<f64>,
    target_storage: &'a mut Vec<f64>,
) -> Result<(&'a [f64], &'a [f64])> {
    let source = match source {
        Some(weights) => weights,
        None => {
            *source_storage = validate::uniform(cost.nrows())?;
            source_storage
        }
    };
    let target = match target {
        Some(weights) => weights,
        None => {
            *target_storage = validate::uniform(cost.ncols())?;
            target_storage
        }
    };
    validate::cost_matrix(cost, source.len(), target.len())?;
    Ok((source, target))
}

/// Solve a general linear discrete transport problem.
pub fn solve(
    cost: &Mat<f64>,
    source: Option<&[f64]>,
    target: Option<&[f64]>,
    options: SolveOptions,
) -> Result<TransportPlan> {
    if options.max_iterations == 0 {
        return Err(Error::InvalidParameter {
            name: "max_iterations",
            requirement: "positive",
        });
    }
    validate::finite_positive(
        options.tolerance,
        "tolerance",
        "finite and strictly positive",
    )?;
    let mut source_storage = Vec::new();
    let mut target_storage = Vec::new();
    let (source, target) = marginals(
        cost,
        source,
        target,
        &mut source_storage,
        &mut target_storage,
    )?;
    match (options.marginal, options.regularization, options.method) {
        (
            MarginalConstraint::Balanced,
            Regularization::None,
            LinearMethod::Auto | LinearMethod::Exact,
        ) => exact::emd(source, target, cost),
        (
            MarginalConstraint::Balanced,
            Regularization::Entropy(regularization),
            LinearMethod::Auto | LinearMethod::SinkhornLog | LinearMethod::SinkhornScaling,
        ) => {
            let method = match options.method {
                LinearMethod::SinkhornScaling => SinkhornMethod::Scaling,
                _ => SinkhornMethod::Log,
            };
            sinkhorn::sinkhorn_with_options(
                source,
                target,
                cost,
                regularization,
                SinkhornOptions {
                    max_iterations: options.max_iterations,
                    tolerance: options.tolerance,
                    method,
                    ..SinkhornOptions::default()
                },
            )
        }
        (
            MarginalConstraint::UnbalancedKl {
                source_penalty,
                target_penalty,
            },
            Regularization::Entropy(regularization),
            LinearMethod::Auto | LinearMethod::SinkhornLog,
        ) => unbalanced::sinkhorn_unbalanced_with_options(
            source,
            target,
            cost,
            regularization,
            UnbalancedOptions {
                source_penalty,
                target_penalty,
                max_iterations: options.max_iterations,
                tolerance: options.tolerance,
                ..UnbalancedOptions::default()
            },
        ),
        (
            MarginalConstraint::Partial { mass },
            Regularization::None,
            LinearMethod::Auto | LinearMethod::Exact,
        ) => partial::partial_wasserstein(source, target, cost, mass),
        (
            MarginalConstraint::Partial { mass },
            Regularization::Entropy(regularization),
            LinearMethod::Auto | LinearMethod::SinkhornScaling,
        ) => partial::entropic_partial_wasserstein(
            source,
            target,
            cost,
            regularization,
            mass,
            EntropicPartialOptions {
                max_iterations: options.max_iterations,
                tolerance: options.tolerance,
            },
        ),
        _ => Err(Error::InvalidParameter {
            name: "method",
            requirement: "compatible with regularization and marginal constraints",
        }),
    }
}

/// Build a ground-cost matrix from sample rows and solve linear transport.
pub fn solve_samples(
    source_samples: &Mat<f64>,
    target_samples: &Mat<f64>,
    source: Option<&[f64]>,
    target: Option<&[f64]>,
    metric: Metric,
    options: SolveOptions,
) -> Result<TransportPlan> {
    let cost = metrics::pairwise(source_samples, target_samples, metric)?;
    solve(&cost, source, target, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_dispatches_unregularized_balanced_to_exact() {
        let cost = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);
        let result = solve(&cost, None, None, SolveOptions::default()).unwrap();
        assert!(result.value.abs() < 1e-12);
        assert!(result.potentials.is_none());
    }

    #[test]
    fn auto_dispatches_entropy_to_sinkhorn() {
        let cost = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);
        let result = solve(
            &cost,
            None,
            None,
            SolveOptions {
                regularization: Regularization::Entropy(0.1),
                ..SolveOptions::default()
            },
        )
        .unwrap();
        assert!(result.potentials.is_some());
    }
}
