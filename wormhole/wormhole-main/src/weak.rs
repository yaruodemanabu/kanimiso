//! Weak optimal transport for empirical measures.

use crate::error::{Error, Result};
use crate::exact;
use crate::result::{SolverStatus, TransportPlan};
use crate::validate;
use faer::Mat;

/// Stopping options for weak optimal transport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeakTransportOptions {
    /// Maximum Frank-Wolfe iterations.
    pub max_iterations: usize,
    /// Relative objective-change threshold.
    pub relative_tolerance: f64,
    /// Absolute objective-change and dual-gap threshold.
    pub absolute_tolerance: f64,
}

impl Default for WeakTransportOptions {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            relative_tolerance: 1e-9,
            absolute_tolerance: 1e-9,
        }
    }
}

fn validate_problem(
    source_samples: &Mat<f64>,
    target_samples: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    options: WeakTransportOptions,
) -> Result<()> {
    validate::samples(source_samples, "weak OT source samples")?;
    validate::samples(target_samples, "weak OT target samples")?;
    if source_samples.ncols() != target_samples.ncols()
        || source_samples.nrows() != source.len()
        || target_samples.nrows() != target.len()
    {
        return Err(Error::ShapeMismatch {
            context: "weak OT samples and weights",
            left: (source_samples.nrows(), source_samples.ncols()),
            right: (target_samples.nrows(), target_samples.ncols()),
        });
    }
    validate::balanced_distributions(source, target)?;
    validate::finite_positive(
        options.relative_tolerance,
        "relative_tolerance",
        "finite and strictly positive",
    )?;
    validate::finite_positive(
        options.absolute_tolerance,
        "absolute_tolerance",
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

fn validate_initial(plan: &Mat<f64>, source: &[f64], target: &[f64]) -> Result<()> {
    if plan.nrows() != source.len() || plan.ncols() != target.len() {
        return Err(Error::ShapeMismatch {
            context: "weak OT initial plan",
            left: (plan.nrows(), plan.ncols()),
            right: (source.len(), target.len()),
        });
    }
    for j in 0..plan.ncols() {
        for i in 0..plan.nrows() {
            if !plan[(i, j)].is_finite() || plan[(i, j)] < 0.0 {
                return Err(Error::InvalidWeight {
                    index: i * plan.ncols() + j,
                    value: plan[(i, j)],
                });
            }
        }
    }
    let residual = marginal_residual(plan, source, target);
    if residual > 1e-9 {
        return Err(Error::Infeasible {
            context: "weak OT initial plan does not have requested marginals",
        });
    }
    Ok(())
}

fn barycentric_error(
    plan: &Mat<f64>,
    source_samples: &Mat<f64>,
    target_samples: &Mat<f64>,
    source: &[f64],
) -> (f64, Mat<f64>) {
    let mut value = 0.0;
    let mut residual = Mat::<f64>::zeros(source_samples.nrows(), source_samples.ncols());
    for i in 0..source_samples.nrows() {
        if source[i] == 0.0 {
            continue;
        }
        for coordinate in 0..source_samples.ncols() {
            let mapped = (0..target_samples.nrows())
                .map(|j| plan[(i, j)] * target_samples[(j, coordinate)])
                .sum::<f64>()
                / source[i];
            residual[(i, coordinate)] = source_samples[(i, coordinate)] - mapped;
            value += source[i] * residual[(i, coordinate)].powi(2);
        }
    }
    (value, residual)
}

fn gradient(residual: &Mat<f64>, target_samples: &Mat<f64>) -> Mat<f64> {
    Mat::<f64>::from_fn(residual.nrows(), target_samples.nrows(), |i, j| {
        -2.0 * (0..residual.ncols())
            .map(|coordinate| residual[(i, coordinate)] * target_samples[(j, coordinate)])
            .sum::<f64>()
    })
}

fn marginal_residual(plan: &Mat<f64>, source: &[f64], target: &[f64]) -> f64 {
    let mut residual = 0.0_f64;
    for i in 0..plan.nrows() {
        let mass = (0..plan.ncols()).map(|j| plan[(i, j)]).sum::<f64>();
        residual = residual.max((mass - source[i]).abs());
    }
    for j in 0..plan.ncols() {
        let mass = (0..plan.nrows()).map(|i| plan[(i, j)]).sum::<f64>();
        residual = residual.max((mass - target[j]).abs());
    }
    residual
}

/// Compute weak optimal transport with the independent coupling initialization.
pub fn weak_optimal_transport(
    source_samples: &Mat<f64>,
    target_samples: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
) -> Result<TransportPlan> {
    weak_optimal_transport_with_options(
        source_samples,
        target_samples,
        source_weights,
        target_weights,
        None,
        WeakTransportOptions::default(),
    )
}

/// Compute weak optimal transport through exact-line-search Frank-Wolfe steps.
pub fn weak_optimal_transport_with_options(
    source_samples: &Mat<f64>,
    target_samples: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    initial: Option<&Mat<f64>>,
    options: WeakTransportOptions,
) -> Result<TransportPlan> {
    let source_storage;
    let source = match source_weights {
        Some(weights) => weights,
        None => {
            source_storage = validate::uniform(source_samples.nrows())?;
            &source_storage
        }
    };
    let target_storage;
    let target = match target_weights {
        Some(weights) => weights,
        None => {
            target_storage = validate::uniform(target_samples.nrows())?;
            &target_storage
        }
    };
    validate_problem(source_samples, target_samples, source, target, options)?;
    let total_mass = source.iter().sum::<f64>();
    let mut plan = match initial {
        Some(initial) => {
            validate_initial(initial, source, target)?;
            initial.clone()
        }
        None => Mat::<f64>::from_fn(source.len(), target.len(), |i, j| {
            source[i] * target[j] / total_mass
        }),
    };
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    let (mut objective, _) = barycentric_error(&plan, source_samples, target_samples, source);
    for iteration in 1..=options.max_iterations {
        let (_, residual) = barycentric_error(&plan, source_samples, target_samples, source);
        let derivative = gradient(&residual, target_samples);
        let vertex = exact::emd(source, target, &derivative)?.plan;
        let direction = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
            vertex[(i, j)] - plan[(i, j)]
        });
        let mut directional_derivative = 0.0;
        for j in 0..plan.ncols() {
            for i in 0..plan.nrows() {
                directional_derivative += derivative[(i, j)] * direction[(i, j)];
            }
        }
        let gap = -directional_derivative;
        if gap <= options.absolute_tolerance {
            status = SolverStatus::Converged;
            break;
        }
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for i in 0..source.len() {
            if source[i] == 0.0 {
                continue;
            }
            for coordinate in 0..source_samples.ncols() {
                let mapped_direction = (0..target.len())
                    .map(|j| direction[(i, j)] * target_samples[(j, coordinate)])
                    .sum::<f64>()
                    / source[i];
                numerator += source[i] * residual[(i, coordinate)] * mapped_direction;
                denominator += source[i] * mapped_direction.powi(2);
            }
        }
        let step = if denominator > 0.0 {
            (numerator / denominator).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if step == 0.0 {
            status = SolverStatus::Converged;
            break;
        }
        for j in 0..plan.ncols() {
            for i in 0..plan.nrows() {
                plan[(i, j)] += step * direction[(i, j)];
            }
        }
        let previous = objective;
        objective = barycentric_error(&plan, source_samples, target_samples, source).0;
        iterations = iteration;
        let absolute_change = (previous - objective).abs();
        let relative_change = absolute_change / previous.abs().max(1.0);
        if absolute_change <= options.absolute_tolerance
            || relative_change <= options.relative_tolerance
        {
            status = SolverStatus::Converged;
            break;
        }
    }
    let residual = marginal_residual(&plan, source, target);
    Ok(TransportPlan {
        plan,
        value: objective,
        potentials: None,
        iterations,
        residual,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weak_transport_matches_one_dimensional_reference() {
        let source = Mat::<f64>::from_fn(3, 1, |i, _| [0.0, 2.0, 5.0][i]);
        let target = Mat::<f64>::from_fn(2, 1, |i, _| [1.0, 4.0][i]);
        let result = weak_optimal_transport_with_options(
            &source,
            &target,
            Some(&[0.2, 0.5, 0.3]),
            Some(&[0.6, 0.4]),
            None,
            WeakTransportOptions {
                max_iterations: 1_000,
                relative_tolerance: 1e-12,
                absolute_tolerance: 1e-12,
            },
        )
        .unwrap();
        assert_eq!(result.status, SolverStatus::Converged);
        assert!((result.value - 0.58).abs() < 1e-10);
        let expected = [[0.2, 0.0], [0.4, 0.1], [0.0, 0.3]];
        for (i, row) in expected.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                assert!((result.plan[(i, j)] - value).abs() < 1e-10);
            }
        }
    }
}
