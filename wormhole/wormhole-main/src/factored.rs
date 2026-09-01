//! Factored optimal transport through a small intermediate support.

use crate::error::{Error, Result};
use crate::exact;
use crate::metrics::{self, Metric};
use crate::result::SolverStatus;
use crate::sinkhorn;
use crate::validate;
use faer::Mat;

/// Options for factored optimal transport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FactoredOptions {
    /// Number of intermediate atoms.
    pub rank: usize,
    /// Optional positive entropic regularization for each linear subproblem.
    pub regularization: Option<f64>,
    /// Maximum support updates.
    pub max_iterations: usize,
    /// Frobenius support displacement required for convergence.
    pub tolerance: f64,
}

impl Default for FactoredOptions {
    fn default() -> Self {
        Self {
            rank: 100,
            regularization: None,
            max_iterations: 100,
            tolerance: 1e-7,
        }
    }
}

/// Result of a factored transport solve.
#[derive(Clone, Debug)]
pub struct FactoredTransport {
    /// Coupling from source atoms to intermediate atoms.
    pub source_plan: Mat<f64>,
    /// Coupling from intermediate atoms to target atoms.
    pub target_plan: Mat<f64>,
    /// Learned intermediate support.
    pub support: Mat<f64>,
    /// Sum of both linear transport costs.
    pub value: f64,
    /// Number of support updates.
    pub iterations: usize,
    /// Last support displacement.
    pub residual: f64,
    /// Termination status.
    pub status: SolverStatus,
}

fn deterministic_initial_support(source: &Mat<f64>, target: &Mat<f64>, rank: usize) -> Mat<f64> {
    Mat::<f64>::from_fn(rank, source.ncols(), |atom, coordinate| {
        let source_row = atom % source.nrows();
        let target_row = atom % target.nrows();
        0.5 * (source[(source_row, coordinate)] + target[(target_row, coordinate)])
    })
}

/// Solve the factored OT barycenter problem.
pub fn factored_optimal_transport(
    source_samples: &Mat<f64>,
    target_samples: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    initial_support: Option<&Mat<f64>>,
    options: FactoredOptions,
) -> Result<FactoredTransport> {
    validate::samples(source_samples, "factored source samples")?;
    validate::samples(target_samples, "factored target samples")?;
    if source_samples.ncols() != target_samples.ncols() {
        return Err(Error::ShapeMismatch {
            context: "factored feature dimensions",
            left: (source_samples.nrows(), source_samples.ncols()),
            right: (target_samples.nrows(), target_samples.ncols()),
        });
    }
    if options.rank == 0 || options.max_iterations == 0 {
        return Err(Error::InvalidParameter {
            name: "rank and max_iterations",
            requirement: "positive",
        });
    }
    validate::finite_positive(
        options.tolerance,
        "tolerance",
        "finite and strictly positive",
    )?;
    if let Some(regularization) = options.regularization {
        validate::finite_positive(
            regularization,
            "regularization",
            "finite and strictly positive",
        )?;
    }
    let source_owned;
    let target_owned;
    let source_weights = match source_weights {
        Some(weights) => weights,
        None => {
            source_owned = validate::uniform(source_samples.nrows())?;
            &source_owned
        }
    };
    let target_weights = match target_weights {
        Some(weights) => weights,
        None => {
            target_owned = validate::uniform(target_samples.nrows())?;
            &target_owned
        }
    };
    if source_weights.len() != source_samples.nrows()
        || target_weights.len() != target_samples.nrows()
    {
        return Err(Error::ShapeMismatch {
            context: "factored samples and weights",
            left: (source_samples.nrows(), target_samples.nrows()),
            right: (source_weights.len(), target_weights.len()),
        });
    }
    validate::balanced_distributions(source_weights, target_weights)?;
    let intermediate_weights = vec![1.0 / options.rank as f64; options.rank];
    let mut support = match initial_support {
        Some(initial) => {
            validate::samples(initial, "factored initial support")?;
            if initial.nrows() != options.rank || initial.ncols() != source_samples.ncols() {
                return Err(Error::ShapeMismatch {
                    context: "factored initial support",
                    left: (initial.nrows(), initial.ncols()),
                    right: (options.rank, source_samples.ncols()),
                });
            }
            initial.clone()
        }
        None => deterministic_initial_support(source_samples, target_samples, options.rank),
    };
    let mut source_plan = Mat::<f64>::zeros(source_samples.nrows(), options.rank);
    let mut target_plan = Mat::<f64>::zeros(options.rank, target_samples.nrows());
    let mut source_value = 0.0;
    let mut target_value = 0.0;
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        let source_cost = metrics::pairwise(source_samples, &support, Metric::SquaredEuclidean)?;
        let target_cost = metrics::pairwise(&support, target_samples, Metric::SquaredEuclidean)?;
        let source_result = if let Some(regularization) = options.regularization {
            sinkhorn::sinkhorn(
                source_weights,
                &intermediate_weights,
                &source_cost,
                regularization,
            )?
        } else {
            exact::emd(source_weights, &intermediate_weights, &source_cost)?
        };
        let target_result = if let Some(regularization) = options.regularization {
            sinkhorn::sinkhorn(
                &intermediate_weights,
                target_weights,
                &target_cost,
                regularization,
            )?
        } else {
            exact::emd(&intermediate_weights, target_weights, &target_cost)?
        };
        source_plan = source_result.plan;
        target_plan = target_result.plan;
        source_value = source_result.value;
        target_value = target_result.value;
        let next = Mat::<f64>::from_fn(options.rank, source_samples.ncols(), |atom, coordinate| {
            let source_average = (0..source_samples.nrows())
                .map(|row| source_plan[(row, atom)] * source_samples[(row, coordinate)])
                .sum::<f64>();
            let target_average = (0..target_samples.nrows())
                .map(|row| target_plan[(atom, row)] * target_samples[(row, coordinate)])
                .sum::<f64>();
            0.5 * (source_average + target_average) / intermediate_weights[atom]
        });
        residual = 0.0;
        for i in 0..support.nrows() {
            for j in 0..support.ncols() {
                residual += (next[(i, j)] - support[(i, j)]).powi(2);
            }
        }
        residual = residual.sqrt();
        support = next;
        iterations = iteration;
        if residual <= options.tolerance {
            status = SolverStatus::Converged;
            break;
        }
    }
    Ok(FactoredTransport {
        source_plan,
        target_plan,
        support,
        value: source_value + target_value,
        iterations,
        residual,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_one_support_is_midpoint_of_diracs() {
        let source = Mat::<f64>::from_fn(1, 1, |_, _| 0.0);
        let target = Mat::<f64>::from_fn(1, 1, |_, _| 4.0);
        let result = factored_optimal_transport(
            &source,
            &target,
            None,
            None,
            None,
            FactoredOptions {
                rank: 1,
                ..FactoredOptions::default()
            },
        )
        .unwrap();
        assert!((result.support[(0, 0)] - 2.0).abs() < 1e-12);
        assert_eq!(result.status, SolverStatus::Converged);
    }
}
