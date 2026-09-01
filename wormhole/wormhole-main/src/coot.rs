//! Co-optimal transport between matrix rows and columns.

use crate::error::{Error, Result};
use crate::exact;
use crate::result::SolverStatus;
use crate::sinkhorn::{self, SinkhornOptions};
use crate::validate;
use faer::Mat;

/// Data, weights, and optional linear terms for a COOT problem.
#[derive(Clone, Copy, Debug)]
pub struct CootProblem<'a> {
    /// Source data matrix (samples by features).
    pub source: &'a Mat<f64>,
    /// Target data matrix (samples by features).
    pub target: &'a Mat<f64>,
    /// Source sample histogram, uniform when omitted.
    pub source_sample_weights: Option<&'a [f64]>,
    /// Source feature histogram, uniform when omitted.
    pub source_feature_weights: Option<&'a [f64]>,
    /// Target sample histogram, uniform when omitted.
    pub target_sample_weights: Option<&'a [f64]>,
    /// Target feature histogram, uniform when omitted.
    pub target_feature_weights: Option<&'a [f64]>,
    /// Optional linear cost on the sample coupling.
    pub sample_linear_cost: Option<&'a Mat<f64>>,
    /// Optional linear cost on the feature coupling.
    pub feature_linear_cost: Option<&'a Mat<f64>>,
}

impl<'a> CootProblem<'a> {
    /// Construct a uniformly weighted COOT problem without linear terms.
    pub fn new(source: &'a Mat<f64>, target: &'a Mat<f64>) -> Self {
        Self {
            source,
            target,
            source_sample_weights: None,
            source_feature_weights: None,
            target_sample_weights: None,
            target_feature_weights: None,
            sample_linear_cost: None,
            feature_linear_cost: None,
        }
    }
}

/// Block-coordinate and inner OT options for COOT.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CootOptions {
    /// Entropic regularization of the sample coupling; zero selects exact EMD.
    pub sample_regularization: f64,
    /// Entropic regularization of the feature coupling; zero selects exact EMD.
    pub feature_regularization: f64,
    /// Multiplier of `sample_linear_cost`.
    pub sample_linear_weight: f64,
    /// Multiplier of `feature_linear_cost`.
    pub feature_linear_weight: f64,
    /// Maximum block-coordinate iterations.
    pub max_iterations: usize,
    /// L1 sample-plan change required for convergence.
    pub tolerance: f64,
    /// Absolute objective change required for convergence.
    pub objective_tolerance: f64,
    /// Options for entropic inner transport solves.
    pub sinkhorn: SinkhornOptions,
}

impl Default for CootOptions {
    fn default() -> Self {
        Self {
            sample_regularization: 0.0,
            feature_regularization: 0.0,
            sample_linear_weight: 0.0,
            feature_linear_weight: 0.0,
            max_iterations: 100,
            tolerance: 1e-7,
            objective_tolerance: 1e-6,
            sinkhorn: SinkhornOptions {
                max_iterations: 500,
                tolerance: 1e-7,
                ..SinkhornOptions::default()
            },
        }
    }
}

/// Sample/feature couplings and the COOT objective.
#[derive(Clone, Debug)]
pub struct CootTransport {
    /// Coupling between source and target samples.
    pub sample_plan: Mat<f64>,
    /// Coupling between source and target features.
    pub feature_plan: Mat<f64>,
    /// Full COOT objective, including optional linear and KL terms.
    pub value: f64,
    /// Number of block-coordinate iterations.
    pub iterations: usize,
    /// Last L1 sample-coupling change.
    pub residual: f64,
    /// Solver termination status.
    pub status: SolverStatus,
}

fn owned_weights(weights: Option<&[f64]>, size: usize, name: &'static str) -> Result<Vec<f64>> {
    let values = match weights {
        Some(weights) => weights.to_vec(),
        None => validate::uniform(size)?,
    };
    if values.len() != size {
        return Err(Error::ShapeMismatch {
            context: "COOT weights and axis",
            left: (values.len(), 1),
            right: (size, 1),
        });
    }
    validate::distribution(&values, name)?;
    Ok(values)
}

fn validate_options(options: CootOptions) -> Result<()> {
    for (name, value) in [
        ("sample_regularization", options.sample_regularization),
        ("feature_regularization", options.feature_regularization),
        ("sample_linear_weight", options.sample_linear_weight),
        ("feature_linear_weight", options.feature_linear_weight),
    ] {
        validate::finite_non_negative(value, name, "finite and non-negative")?;
    }
    validate::finite_positive(
        options.tolerance,
        "tolerance",
        "finite and strictly positive",
    )?;
    validate::finite_positive(
        options.objective_tolerance,
        "objective_tolerance",
        "finite and strictly positive",
    )?;
    if options.max_iterations == 0 {
        return Err(Error::InvalidParameter {
            name: "max_iterations",
            requirement: "positive",
        });
    }
    Ok(())
}

fn independent(left: &[f64], right: &[f64]) -> Mat<f64> {
    let mass = left.iter().sum::<f64>();
    Mat::<f64>::from_fn(left.len(), right.len(), |i, j| left[i] * right[j] / mass)
}

fn sample_cost(
    source: &Mat<f64>,
    target: &Mat<f64>,
    feature_plan: &Mat<f64>,
    source_feature_weights: &[f64],
    target_feature_weights: &[f64],
    linear: Option<&Mat<f64>>,
    linear_weight: f64,
) -> Mat<f64> {
    Mat::<f64>::from_fn(source.nrows(), target.nrows(), |i, j| {
        let source_square = (0..source.ncols())
            .map(|k| source_feature_weights[k] * source[(i, k)].powi(2))
            .sum::<f64>();
        let target_square = (0..target.ncols())
            .map(|l| target_feature_weights[l] * target[(j, l)].powi(2))
            .sum::<f64>();
        let cross = (0..source.ncols())
            .map(|k| {
                (0..target.ncols())
                    .map(|l| source[(i, k)] * feature_plan[(k, l)] * target[(j, l)])
                    .sum::<f64>()
            })
            .sum::<f64>();
        source_square + target_square - 2.0 * cross
            + linear.map_or(0.0, |cost| linear_weight * cost[(i, j)])
    })
}

fn feature_cost(
    source: &Mat<f64>,
    target: &Mat<f64>,
    sample_plan: &Mat<f64>,
    source_sample_weights: &[f64],
    target_sample_weights: &[f64],
    linear: Option<&Mat<f64>>,
    linear_weight: f64,
) -> Mat<f64> {
    Mat::<f64>::from_fn(source.ncols(), target.ncols(), |k, l| {
        let source_square = (0..source.nrows())
            .map(|i| source_sample_weights[i] * source[(i, k)].powi(2))
            .sum::<f64>();
        let target_square = (0..target.nrows())
            .map(|j| target_sample_weights[j] * target[(j, l)].powi(2))
            .sum::<f64>();
        let cross = (0..source.nrows())
            .map(|i| {
                (0..target.nrows())
                    .map(|j| source[(i, k)] * sample_plan[(i, j)] * target[(j, l)])
                    .sum::<f64>()
            })
            .sum::<f64>();
        source_square + target_square - 2.0 * cross
            + linear.map_or(0.0, |cost| linear_weight * cost[(k, l)])
    })
}

fn solve_block(
    left: &[f64],
    right: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    options: SinkhornOptions,
) -> Result<Mat<f64>> {
    if regularization == 0.0 {
        Ok(exact::emd(left, right, cost)?.plan)
    } else {
        Ok(sinkhorn::sinkhorn_with_options(left, right, cost, regularization, options)?.plan)
    }
}

fn matrix_inner(left: &Mat<f64>, right: &Mat<f64>) -> f64 {
    let mut value = 0.0;
    for j in 0..left.ncols() {
        for i in 0..left.nrows() {
            value += left[(i, j)] * right[(i, j)];
        }
    }
    value
}

fn kl_product_reference(plan: &Mat<f64>, left: &[f64], right: &[f64]) -> f64 {
    let mass = left.iter().sum::<f64>();
    let mut value = 0.0;
    for i in 0..plan.nrows() {
        for j in 0..plan.ncols() {
            let current = plan[(i, j)];
            let reference = left[i] * right[j] / mass;
            if current > 0.0 {
                value += current * (current / reference).ln() - current + reference;
            } else {
                value += reference;
            }
        }
    }
    value
}

/// Compute balanced COOT by alternating exact or entropic OT subproblems.
pub fn co_optimal_transport(
    problem: CootProblem<'_>,
    options: CootOptions,
) -> Result<CootTransport> {
    validate::samples(problem.source, "COOT source matrix")?;
    validate::samples(problem.target, "COOT target matrix")?;
    validate_options(options)?;
    let source_sample_weights = owned_weights(
        problem.source_sample_weights,
        problem.source.nrows(),
        "COOT source sample weights",
    )?;
    let source_feature_weights = owned_weights(
        problem.source_feature_weights,
        problem.source.ncols(),
        "COOT source feature weights",
    )?;
    let target_sample_weights = owned_weights(
        problem.target_sample_weights,
        problem.target.nrows(),
        "COOT target sample weights",
    )?;
    let target_feature_weights = owned_weights(
        problem.target_feature_weights,
        problem.target.ncols(),
        "COOT target feature weights",
    )?;
    validate::balanced_distributions(&source_sample_weights, &target_sample_weights)?;
    validate::balanced_distributions(&source_feature_weights, &target_feature_weights)?;
    let sample_mass = source_sample_weights.iter().sum::<f64>();
    let feature_mass = source_feature_weights.iter().sum::<f64>();
    if (sample_mass - feature_mass).abs() > 1e-10 * sample_mass.max(feature_mass).max(1.0) {
        return Err(Error::MassMismatch {
            source: sample_mass,
            target: feature_mass,
        });
    }
    if let Some(cost) = problem.sample_linear_cost {
        validate::cost_matrix(cost, problem.source.nrows(), problem.target.nrows())?;
    }
    if let Some(cost) = problem.feature_linear_cost {
        validate::cost_matrix(cost, problem.source.ncols(), problem.target.ncols())?;
    }

    let mut sample_plan = independent(&source_sample_weights, &target_sample_weights);
    let mut feature_plan = independent(&source_feature_weights, &target_feature_weights);
    let mut value = f64::INFINITY;
    let mut previous_value = None;
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        let previous_sample = sample_plan.clone();
        let current_sample_cost = sample_cost(
            problem.source,
            problem.target,
            &feature_plan,
            &source_feature_weights,
            &target_feature_weights,
            problem.sample_linear_cost,
            options.sample_linear_weight,
        );
        sample_plan = solve_block(
            &source_sample_weights,
            &target_sample_weights,
            &current_sample_cost,
            options.sample_regularization,
            options.sinkhorn,
        )?;
        let current_feature_cost = feature_cost(
            problem.source,
            problem.target,
            &sample_plan,
            &source_sample_weights,
            &target_sample_weights,
            problem.feature_linear_cost,
            options.feature_linear_weight,
        );
        feature_plan = solve_block(
            &source_feature_weights,
            &target_feature_weights,
            &current_feature_cost,
            options.feature_regularization,
            options.sinkhorn,
        )?;

        value = matrix_inner(&feature_plan, &current_feature_cost);
        if let Some(cost) = problem.sample_linear_cost {
            value += options.sample_linear_weight * matrix_inner(&sample_plan, cost);
        }
        if options.sample_regularization > 0.0 {
            value += options.sample_regularization
                * kl_product_reference(
                    &sample_plan,
                    &source_sample_weights,
                    &target_sample_weights,
                );
        }
        if options.feature_regularization > 0.0 {
            value += options.feature_regularization
                * kl_product_reference(
                    &feature_plan,
                    &source_feature_weights,
                    &target_feature_weights,
                );
        }
        residual = 0.0;
        for j in 0..sample_plan.ncols() {
            for i in 0..sample_plan.nrows() {
                residual += (sample_plan[(i, j)] - previous_sample[(i, j)]).abs();
            }
        }
        iterations = iteration;
        let objective_converged = previous_value
            .is_some_and(|previous: f64| (previous - value).abs() <= options.objective_tolerance);
        if residual <= options.tolerance || objective_converged {
            status = SolverStatus::Converged;
            break;
        }
        previous_value = Some(value);
    }
    Ok(CootTransport {
        sample_plan,
        feature_plan,
        value,
        iterations,
        residual,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_coot_matches_reference_couplings() {
        let source = Mat::<f64>::from_fn(3, 2, |i, j| [[0.0, 1.0], [2.0, 3.0], [4.0, 0.0]][i][j]);
        let target = Mat::<f64>::from_fn(2, 3, |i, j| [[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]][i][j]);
        let problem = CootProblem {
            source: &source,
            target: &target,
            source_sample_weights: Some(&[0.2, 0.5, 0.3]),
            source_feature_weights: Some(&[0.7, 0.3]),
            target_sample_weights: Some(&[0.6, 0.4]),
            target_feature_weights: Some(&[0.2, 0.5, 0.3]),
            sample_linear_cost: None,
            feature_linear_cost: None,
        };
        let result = co_optimal_transport(
            problem,
            CootOptions {
                max_iterations: 1_000,
                tolerance: 1e-12,
                objective_tolerance: 1e-12,
                ..CootOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, SolverStatus::Converged);
        assert!((result.value - 2.45).abs() < 1e-12);
        assert!((result.sample_plan[(1, 0)] - 0.4).abs() < 1e-12);
        assert!((result.feature_plan[(0, 2)] - 0.3).abs() < 1e-12);
    }
}
