//! Entropically regularized balanced optimal transport.

use crate::error::{Error, Result};
use crate::metrics::{self, Metric};
use crate::result::{DualPotentials, SolverStatus, TransportPlan};
use crate::validate;
use faer::Mat;

/// Numerical variant used for Sinkhorn iterations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SinkhornMethod {
    /// Multiplicative matrix scaling.
    Scaling,
    /// Log-domain dual updates, robust at small regularization.
    #[default]
    Log,
}

/// Stopping and implementation options for Sinkhorn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SinkhornOptions {
    /// Maximum number of alternating row/column updates.
    pub max_iterations: usize,
    /// Maximum absolute marginal error required for convergence.
    pub tolerance: f64,
    /// Compute the relatively expensive marginal residual every this many steps.
    pub check_interval: usize,
    /// Numerical Sinkhorn variant.
    pub method: SinkhornMethod,
}

impl Default for SinkhornOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: 1e-9,
            check_interval: 10,
            method: SinkhornMethod::Log,
        }
    }
}

/// Stopping options for coordinate-wise Greenkhorn scaling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GreenkhornOptions {
    /// Maximum number of single-row or single-column updates.
    pub max_iterations: usize,
    /// Maximum absolute marginal error required for convergence.
    pub tolerance: f64,
}

impl Default for GreenkhornOptions {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            tolerance: 1e-9,
        }
    }
}

/// Continuation options for epsilon-scaling Sinkhorn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpsilonScalingOptions {
    /// Regularization used by the first continuation stage.
    pub initial_regularization: f64,
    /// Number of geometrically spaced regularization stages.
    pub stages: usize,
    /// Inner log-domain Sinkhorn options used at every stage.
    pub sinkhorn: SinkhornOptions,
}

impl Default for EpsilonScalingOptions {
    fn default() -> Self {
        Self {
            initial_regularization: 1.0,
            stages: 10,
            sinkhorn: SinkhornOptions {
                method: SinkhornMethod::Log,
                ..SinkhornOptions::default()
            },
        }
    }
}

fn validate_options(regularization: f64, options: SinkhornOptions) -> Result<()> {
    validate::finite_positive(
        regularization,
        "regularization",
        "finite and strictly positive",
    )?;
    validate::finite_positive(
        options.tolerance,
        "tolerance",
        "finite and strictly positive",
    )?;
    if options.max_iterations == 0 {
        return Err(Error::InvalidParameter {
            name: "max_iterations",
            requirement: "positive",
        });
    }
    if options.check_interval == 0 {
        return Err(Error::InvalidParameter {
            name: "check_interval",
            requirement: "positive",
        });
    }
    Ok(())
}

fn validate_greenkhorn_options(regularization: f64, options: GreenkhornOptions) -> Result<()> {
    validate::finite_positive(
        regularization,
        "regularization",
        "finite and strictly positive",
    )?;
    validate::finite_positive(
        options.tolerance,
        "tolerance",
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

/// Solve entropically regularized balanced transport.
pub fn sinkhorn(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
) -> Result<TransportPlan> {
    sinkhorn_with_options(
        source,
        target,
        cost,
        regularization,
        SinkhornOptions::default(),
    )
}

/// Solve entropic transport with explicit implementation and stopping options.
pub fn sinkhorn_with_options(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    options: SinkhornOptions,
) -> Result<TransportPlan> {
    validate::balanced_distributions(source, target)?;
    validate::cost_matrix(cost, source.len(), target.len())?;
    validate_options(regularization, options)?;
    match options.method {
        SinkhornMethod::Scaling => scaling_sinkhorn(source, target, cost, regularization, options),
        SinkhornMethod::Log => log_sinkhorn(source, target, cost, regularization, options),
    }
}

/// Solve Sinkhorn through stable log-domain dual absorption.
///
/// This is the dense, numerically stabilized variant of the same entropic
/// problem and returns dual potentials in cost units.
pub fn sinkhorn_stabilized(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    mut options: SinkhornOptions,
) -> Result<TransportPlan> {
    options.method = SinkhornMethod::Log;
    sinkhorn_with_options(source, target, cost, regularization, options)
}

/// Solve Sinkhorn using geometric epsilon continuation and warm dual potentials.
pub fn sinkhorn_epsilon_scaling(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    options: EpsilonScalingOptions,
) -> Result<TransportPlan> {
    validate::balanced_distributions(source, target)?;
    validate::cost_matrix(cost, source.len(), target.len())?;
    validate_options(regularization, options.sinkhorn)?;
    validate::finite_positive(
        options.initial_regularization,
        "initial_regularization",
        "finite and strictly positive",
    )?;
    if options.stages == 0 {
        return Err(Error::InvalidParameter {
            name: "stages",
            requirement: "positive",
        });
    }
    let initial_regularization = options.initial_regularization.max(regularization);
    let mut potentials = None;
    let mut total_iterations = 0usize;
    let mut final_result = None;
    for stage in 0..options.stages {
        let stage_regularization = if stage + 1 == options.stages {
            regularization
        } else {
            let fraction = stage as f64 / (options.stages - 1) as f64;
            initial_regularization * (regularization / initial_regularization).powf(fraction)
        };
        let result = log_sinkhorn_with_initial(
            source,
            target,
            cost,
            stage_regularization,
            options.sinkhorn,
            potentials.as_ref(),
        )?;
        total_iterations = total_iterations.saturating_add(result.iterations);
        potentials = result.potentials.clone();
        final_result = Some(result);
    }
    let mut result = final_result.expect("validated positive continuation stages");
    result.iterations = total_iterations;
    Ok(result)
}

/// Solve entropic transport with coordinate-wise Greenkhorn updates.
pub fn greenkhorn(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
) -> Result<TransportPlan> {
    greenkhorn_with_options(
        source,
        target,
        cost,
        regularization,
        GreenkhornOptions::default(),
    )
}

/// Solve Greenkhorn with explicit stopping options.
pub fn greenkhorn_with_options(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    options: GreenkhornOptions,
) -> Result<TransportPlan> {
    validate::balanced_distributions(source, target)?;
    validate::cost_matrix(cost, source.len(), target.len())?;
    validate_greenkhorn_options(regularization, options)?;

    let mut minimum_cost = f64::INFINITY;
    for j in 0..cost.ncols() {
        for i in 0..cost.nrows() {
            minimum_cost = minimum_cost.min(cost[(i, j)]);
        }
    }
    let kernel = Mat::<f64>::from_fn(cost.nrows(), cost.ncols(), |i, j| {
        (-(cost[(i, j)] - minimum_cost) / regularization).exp()
    });
    let mut left_scale = vec![1.0 / source.len() as f64; source.len()];
    let mut right_scale = vec![1.0 / target.len() as f64; target.len()];
    let mut plan = plan_from_scalings(&kernel, &left_scale, &right_scale);
    let mut row_violation = (0..source.len())
        .map(|i| (0..target.len()).map(|j| plan[(i, j)]).sum::<f64>() - source[i])
        .collect::<Vec<_>>();
    let mut column_violation = (0..target.len())
        .map(|j| (0..source.len()).map(|i| plan[(i, j)]).sum::<f64>() - target[j])
        .collect::<Vec<_>>();
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;

    for iteration in 1..=options.max_iterations {
        let row = (0..row_violation.len())
            .max_by(|&left, &right| {
                row_violation[left]
                    .abs()
                    .total_cmp(&row_violation[right].abs())
            })
            .expect("validated non-empty source");
        let column = (0..column_violation.len())
            .max_by(|&left, &right| {
                column_violation[left]
                    .abs()
                    .total_cmp(&column_violation[right].abs())
            })
            .expect("validated non-empty target");
        let row_error = row_violation[row].abs();
        let column_error = column_violation[column].abs();
        if row_error.max(column_error) <= options.tolerance {
            status = SolverStatus::Converged;
            break;
        }

        if row_error > column_error {
            let denominator = (0..target.len())
                .map(|j| kernel[(row, j)] * right_scale[j])
                .sum::<f64>();
            if source[row] > 0.0 && (!denominator.is_finite() || denominator <= 0.0) {
                return Err(Error::DidNotConverge {
                    algorithm: "Greenkhorn",
                    iterations: iteration,
                    residual: row_error,
                });
            }
            let next_scale = if source[row] == 0.0 {
                0.0
            } else {
                source[row] / denominator
            };
            let mut row_sum = 0.0;
            for j in 0..target.len() {
                let previous = plan[(row, j)];
                let next = next_scale * kernel[(row, j)] * right_scale[j];
                plan[(row, j)] = next;
                row_sum += next;
                column_violation[j] += next - previous;
            }
            left_scale[row] = next_scale;
            row_violation[row] = row_sum - source[row];
        } else {
            let denominator = (0..source.len())
                .map(|i| kernel[(i, column)] * left_scale[i])
                .sum::<f64>();
            if target[column] > 0.0 && (!denominator.is_finite() || denominator <= 0.0) {
                return Err(Error::DidNotConverge {
                    algorithm: "Greenkhorn",
                    iterations: iteration,
                    residual: column_error,
                });
            }
            let next_scale = if target[column] == 0.0 {
                0.0
            } else {
                target[column] / denominator
            };
            let mut column_sum = 0.0;
            for i in 0..source.len() {
                let previous = plan[(i, column)];
                let next = left_scale[i] * kernel[(i, column)] * next_scale;
                plan[(i, column)] = next;
                column_sum += next;
                row_violation[i] += next - previous;
            }
            right_scale[column] = next_scale;
            column_violation[column] = column_sum - target[column];
        }
        iterations = iteration;
    }

    let residual = marginal_residual(&plan, source, target);
    if residual <= options.tolerance {
        status = SolverStatus::Converged;
    }
    let source_potential = left_scale
        .iter()
        .map(|&value| {
            if value > 0.0 {
                regularization * value.ln() + minimum_cost
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let target_potential = right_scale
        .iter()
        .map(|&value| {
            if value > 0.0 {
                regularization * value.ln()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    Ok(build_result(
        plan,
        cost,
        source_potential,
        target_potential,
        iterations,
        residual,
        status,
    ))
}

fn scaling_sinkhorn(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    options: SinkhornOptions,
) -> Result<TransportPlan> {
    let mut minimum_cost = f64::INFINITY;
    for j in 0..cost.ncols() {
        for i in 0..cost.nrows() {
            minimum_cost = minimum_cost.min(cost[(i, j)]);
        }
    }
    let kernel = Mat::<f64>::from_fn(cost.nrows(), cost.ncols(), |i, j| {
        (-(cost[(i, j)] - minimum_cost) / regularization).exp()
    });
    let mut left_scale = vec![1.0; source.len()];
    let mut right_scale = vec![1.0; target.len()];
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        for i in 0..source.len() {
            if source[i] == 0.0 {
                left_scale[i] = 0.0;
                continue;
            }
            let denominator = (0..target.len())
                .map(|j| kernel[(i, j)] * right_scale[j])
                .sum::<f64>();
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(Error::DidNotConverge {
                    algorithm: "Sinkhorn scaling",
                    iterations: iteration,
                    residual: f64::INFINITY,
                });
            }
            left_scale[i] = source[i] / denominator;
        }
        for j in 0..target.len() {
            if target[j] == 0.0 {
                right_scale[j] = 0.0;
                continue;
            }
            let denominator = (0..source.len())
                .map(|i| kernel[(i, j)] * left_scale[i])
                .sum::<f64>();
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(Error::DidNotConverge {
                    algorithm: "Sinkhorn scaling",
                    iterations: iteration,
                    residual: f64::INFINITY,
                });
            }
            right_scale[j] = target[j] / denominator;
        }
        iterations = iteration;
        if iteration % options.check_interval == 0 || iteration == options.max_iterations {
            let plan = plan_from_scalings(&kernel, &left_scale, &right_scale);
            residual = marginal_residual(&plan, source, target);
            if residual <= options.tolerance {
                status = SolverStatus::Converged;
                break;
            }
        }
    }
    let plan = plan_from_scalings(&kernel, &left_scale, &right_scale);
    let source_potential = left_scale
        .iter()
        .map(|&value| {
            if value > 0.0 {
                regularization * value.ln() + minimum_cost
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let target_potential = right_scale
        .iter()
        .map(|&value| {
            if value > 0.0 {
                regularization * value.ln()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    Ok(build_result(
        plan,
        cost,
        source_potential,
        target_potential,
        iterations,
        residual,
        status,
    ))
}

fn plan_from_scalings(kernel: &Mat<f64>, left: &[f64], right: &[f64]) -> Mat<f64> {
    Mat::<f64>::from_fn(kernel.nrows(), kernel.ncols(), |i, j| {
        left[i] * kernel[(i, j)] * right[j]
    })
}

fn log_sum_exp<I>(values: I) -> f64
where
    I: Iterator<Item = f64> + Clone,
{
    let maximum = values
        .clone()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return maximum;
    }
    maximum
        + values
            .filter(|value| value.is_finite())
            .map(|value| (value - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn log_sinkhorn(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    options: SinkhornOptions,
) -> Result<TransportPlan> {
    log_sinkhorn_with_initial(source, target, cost, regularization, options, None)
}

fn log_sinkhorn_with_initial(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    options: SinkhornOptions,
    initial: Option<&DualPotentials>,
) -> Result<TransportPlan> {
    let (mut source_potential, mut target_potential) = match initial {
        Some(potentials)
            if potentials.source.len() == source.len()
                && potentials.target.len() == target.len() =>
        {
            (potentials.source.clone(), potentials.target.clone())
        }
        Some(potentials) => {
            return Err(Error::ShapeMismatch {
                context: "Sinkhorn warm-start potentials",
                left: (potentials.source.len(), potentials.target.len()),
                right: (source.len(), target.len()),
            });
        }
        None => (vec![0.0; source.len()], vec![0.0; target.len()]),
    };
    for (potential, &weight) in source_potential.iter_mut().zip(source) {
        if weight == 0.0 {
            *potential = f64::NEG_INFINITY;
        } else if !potential.is_finite() {
            return Err(Error::InvalidParameter {
                name: "source warm-start potentials",
                requirement: "finite on positive-mass entries",
            });
        }
    }
    for (potential, &weight) in target_potential.iter_mut().zip(target) {
        if weight == 0.0 {
            *potential = f64::NEG_INFINITY;
        } else if !potential.is_finite() {
            return Err(Error::InvalidParameter {
                name: "target warm-start potentials",
                requirement: "finite on positive-mass entries",
            });
        }
    }
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        for i in 0..source.len() {
            if source[i] == 0.0 {
                continue;
            }
            let normalization = log_sum_exp(
                (0..target.len()).map(|j| (target_potential[j] - cost[(i, j)]) / regularization),
            );
            if !normalization.is_finite() {
                return Err(Error::DidNotConverge {
                    algorithm: "log-domain Sinkhorn",
                    iterations: iteration,
                    residual: f64::INFINITY,
                });
            }
            source_potential[i] = regularization * (source[i].ln() - normalization);
        }
        for j in 0..target.len() {
            if target[j] == 0.0 {
                continue;
            }
            let normalization = log_sum_exp(
                (0..source.len()).map(|i| (source_potential[i] - cost[(i, j)]) / regularization),
            );
            if !normalization.is_finite() {
                return Err(Error::DidNotConverge {
                    algorithm: "log-domain Sinkhorn",
                    iterations: iteration,
                    residual: f64::INFINITY,
                });
            }
            target_potential[j] = regularization * (target[j].ln() - normalization);
        }
        iterations = iteration;
        if iteration % options.check_interval == 0 || iteration == options.max_iterations {
            let plan =
                plan_from_potentials(cost, &source_potential, &target_potential, regularization);
            residual = marginal_residual(&plan, source, target);
            if residual <= options.tolerance {
                status = SolverStatus::Converged;
                break;
            }
        }
    }
    let plan = plan_from_potentials(cost, &source_potential, &target_potential, regularization);
    Ok(build_result(
        plan,
        cost,
        source_potential,
        target_potential,
        iterations,
        residual,
        status,
    ))
}

fn plan_from_potentials(
    cost: &Mat<f64>,
    source_potential: &[f64],
    target_potential: &[f64],
    regularization: f64,
) -> Mat<f64> {
    Mat::<f64>::from_fn(cost.nrows(), cost.ncols(), |i, j| {
        if source_potential[i].is_finite() && target_potential[j].is_finite() {
            ((source_potential[i] + target_potential[j] - cost[(i, j)]) / regularization).exp()
        } else {
            0.0
        }
    })
}

fn marginal_residual(plan: &Mat<f64>, source: &[f64], target: &[f64]) -> f64 {
    let mut residual = 0.0_f64;
    for i in 0..plan.nrows() {
        let sum = (0..plan.ncols()).map(|j| plan[(i, j)]).sum::<f64>();
        residual = residual.max((sum - source[i]).abs());
    }
    for j in 0..plan.ncols() {
        let sum = (0..plan.nrows()).map(|i| plan[(i, j)]).sum::<f64>();
        residual = residual.max((sum - target[j]).abs());
    }
    residual
}

fn build_result(
    plan: Mat<f64>,
    cost: &Mat<f64>,
    source_potential: Vec<f64>,
    target_potential: Vec<f64>,
    iterations: usize,
    residual: f64,
    status: SolverStatus,
) -> TransportPlan {
    let mut value = 0.0;
    for i in 0..plan.nrows() {
        for j in 0..plan.ncols() {
            value += plan[(i, j)] * cost[(i, j)];
        }
    }
    TransportPlan {
        plan,
        value,
        potentials: Some(DualPotentials {
            source: source_potential,
            target: target_potential,
        }),
        iterations,
        residual,
        status,
    }
}

/// Return only the linear cost of the Sinkhorn plan.
pub fn sinkhorn2(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
) -> Result<f64> {
    Ok(sinkhorn(source, target, cost, regularization)?.value)
}

/// Entropic primal objective `���plan,cost��� + reg * sum(plan * (ln(plan)-1))`.
pub fn entropic_objective(plan: &Mat<f64>, cost: &Mat<f64>, regularization: f64) -> Result<f64> {
    validate::finite_positive(
        regularization,
        "regularization",
        "finite and strictly positive",
    )?;
    if plan.nrows() != cost.nrows() || plan.ncols() != cost.ncols() {
        return Err(Error::ShapeMismatch {
            context: "plan and cost",
            left: (plan.nrows(), plan.ncols()),
            right: (cost.nrows(), cost.ncols()),
        });
    }
    validate::cost_matrix(cost, plan.nrows(), plan.ncols())?;
    let mut value = 0.0;
    for i in 0..plan.nrows() {
        for j in 0..plan.ncols() {
            let mass = plan[(i, j)];
            if !mass.is_finite() || mass < 0.0 {
                return Err(Error::InvalidWeight {
                    index: i * plan.ncols() + j,
                    value: mass,
                });
            }
            value += mass * cost[(i, j)];
            if mass > 0.0 {
                value += regularization * mass * (mass.ln() - 1.0);
            }
        }
    }
    Ok(value)
}

/// Sinkhorn transport directly between sample rows.
pub fn empirical_sinkhorn(
    source_samples: &Mat<f64>,
    target_samples: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    metric: Metric,
    regularization: f64,
) -> Result<TransportPlan> {
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
    let cost = metrics::pairwise(source_samples, target_samples, metric)?;
    sinkhorn(source_weights, target_weights, &cost, regularization)
}

/// POT-compatible debiased empirical Sinkhorn divergence.
///
/// POT 0.9.7 combines the linear costs of the three entropic plans and does
/// not include their entropy terms in the returned quantity.
pub fn empirical_sinkhorn_divergence(
    source_samples: &Mat<f64>,
    target_samples: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    metric: Metric,
    regularization: f64,
) -> Result<f64> {
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
    let cross_cost = metrics::pairwise(source_samples, target_samples, metric)?;
    let source_cost = metrics::pairwise_self(source_samples, metric)?;
    let target_cost = metrics::pairwise_self(target_samples, metric)?;
    let cross = sinkhorn(source_weights, target_weights, &cross_cost, regularization)?;
    let source_self = sinkhorn(source_weights, source_weights, &source_cost, regularization)?;
    let target_self = sinkhorn(target_weights, target_weights, &target_cost, regularization)?;
    let value = cross.value - 0.5 * (source_self.value + target_self.value);
    Ok(value.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_sinkhorn_matches_requested_marginals() {
        let source = [0.25, 0.75];
        let target = [0.6, 0.4];
        let cost = Mat::<f64>::from_fn(2, 2, |i, j| (i as f64 - j as f64).powi(2));
        let result = sinkhorn(&source, &target, &cost, 0.2).expect("valid Sinkhorn");
        assert_eq!(result.status, SolverStatus::Converged);
        assert!(result.residual < 1e-8, "residual={}", result.residual);
        for (actual, expected) in result.source_marginal().iter().zip(source) {
            assert!((actual - expected).abs() < 1e-8);
        }
        for (actual, expected) in result.target_marginal().iter().zip(target) {
            assert!((actual - expected).abs() < 1e-8);
        }
    }

    #[test]
    fn scaling_and_log_variants_agree() {
        let source = [0.5, 0.5];
        let target = [0.5, 0.5];
        let cost = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);
        let log = sinkhorn(&source, &target, &cost, 0.5).unwrap();
        let scaling = sinkhorn_with_options(
            &source,
            &target,
            &cost,
            0.5,
            SinkhornOptions {
                method: SinkhornMethod::Scaling,
                ..SinkhornOptions::default()
            },
        )
        .unwrap();
        assert!((log.value - scaling.value).abs() < 1e-9);
    }

    #[test]
    fn greenkhorn_and_log_variants_agree() {
        let source = [0.2, 0.5, 0.3];
        let target = [0.4, 0.1, 0.5];
        let cost = Mat::<f64>::from_fn(3, 3, |i, j| {
            [[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]][i][j]
        });
        let log = sinkhorn(&source, &target, &cost, 0.7).unwrap();
        let green = greenkhorn_with_options(
            &source,
            &target,
            &cost,
            0.7,
            GreenkhornOptions {
                max_iterations: 100_000,
                tolerance: 1e-12,
            },
        )
        .unwrap();
        assert_eq!(green.status, SolverStatus::Converged);
        assert!((log.value - green.value).abs() < 1e-9);
    }

    #[test]
    fn stabilized_and_epsilon_scaling_agree_at_small_regularization() {
        let source = [0.2, 0.5, 0.3];
        let target = [0.4, 0.1, 0.5];
        let cost = Mat::<f64>::from_fn(3, 3, |i, j| {
            [[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]][i][j]
        });
        let stabilized = sinkhorn_stabilized(
            &source,
            &target,
            &cost,
            0.1,
            SinkhornOptions {
                max_iterations: 10_000,
                tolerance: 1e-12,
                ..SinkhornOptions::default()
            },
        )
        .unwrap();
        let continuation = sinkhorn_epsilon_scaling(
            &source,
            &target,
            &cost,
            0.1,
            EpsilonScalingOptions {
                stages: 8,
                sinkhorn: SinkhornOptions {
                    max_iterations: 10_000,
                    tolerance: 1e-12,
                    ..SinkhornOptions::default()
                },
                ..EpsilonScalingOptions::default()
            },
        )
        .unwrap();
        assert_eq!(continuation.status, SolverStatus::Converged);
        assert!((continuation.value - 0.40000000103057687).abs() < 1e-10);
        for i in 0..source.len() {
            for j in 0..target.len() {
                assert!((stabilized.plan[(i, j)] - continuation.plan[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn empirical_divergence_vanishes_on_identity() {
        let sample = Mat::<f64>::from_fn(3, 1, |i, _| i as f64);
        let value = empirical_sinkhorn_divergence(
            &sample,
            &sample,
            None,
            None,
            Metric::SquaredEuclidean,
            0.5,
        )
        .unwrap();
        assert!(value < 1e-9, "value={value}");
    }

    #[test]
    fn empirical_divergence_uses_pot_linear_cost_convention() {
        let source = Mat::<f64>::from_fn(2, 1, |i, _| i as f64);
        let target = Mat::<f64>::from_fn(4, 1, |i, _| i as f64);
        let value = empirical_sinkhorn_divergence(
            &source,
            &target,
            None,
            None,
            Metric::SquaredEuclidean,
            0.5,
        )
        .unwrap();
        assert!((value - 1.4203817720317107).abs() < 1e-8, "value={value}");
    }
}
