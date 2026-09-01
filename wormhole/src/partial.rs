//! Partial optimal transport with a prescribed transported mass.

use crate::error::{Error, Result};
use crate::exact;
use crate::result::{SolverStatus, TransportPlan};
use crate::validate;
use faer::Mat;

fn validate_mass(source: &[f64], target: &[f64], transported_mass: f64) -> Result<(f64, f64)> {
    let source_mass = validate::distribution(source, "source distribution")?;
    let target_mass = validate::distribution(target, "target distribution")?;
    if !transported_mass.is_finite()
        || transported_mass <= 0.0
        || transported_mass > source_mass.min(target_mass) + 1e-12
    {
        return Err(Error::InvalidParameter {
            name: "transported_mass",
            requirement: "finite, positive, and no larger than either marginal mass",
        });
    }
    Ok((source_mass, target_mass))
}

/// Solve exact partial Wasserstein transport for a prescribed mass.
///
/// The problem is reduced to balanced min-cost flow with one dummy atom on
/// each side; the returned plan contains only real-to-real transport.
pub fn partial_wasserstein(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    transported_mass: f64,
) -> Result<TransportPlan> {
    validate::cost_matrix(cost, source.len(), target.len())?;
    let (source_mass, target_mass) = validate_mass(source, target, transported_mass)?;
    let mut augmented_source = source.to_vec();
    augmented_source.push(target_mass - transported_mass);
    let mut augmented_target = target.to_vec();
    augmented_target.push(source_mass - transported_mass);
    let maximum_absolute_cost = (0..cost.nrows())
        .flat_map(|i| (0..cost.ncols()).map(move |j| cost[(i, j)].abs()))
        .fold(0.0, f64::max);
    let dummy_penalty = 2.0 * maximum_absolute_cost + 1.0;
    let augmented_cost = Mat::<f64>::from_fn(source.len() + 1, target.len() + 1, |i, j| {
        if i < source.len() && j < target.len() {
            cost[(i, j)]
        } else if i == source.len() && j == target.len() {
            dummy_penalty
        } else {
            0.0
        }
    });
    let augmented = exact::emd(&augmented_source, &augmented_target, &augmented_cost)?;
    let plan = Mat::<f64>::from_fn(source.len(), target.len(), |i, j| augmented.plan[(i, j)]);
    let mut value = 0.0;
    for i in 0..source.len() {
        for j in 0..target.len() {
            value += plan[(i, j)] * cost[(i, j)];
        }
    }
    let actual_mass = (0..plan.nrows())
        .map(|i| (0..plan.ncols()).map(|j| plan[(i, j)]).sum::<f64>())
        .sum::<f64>();
    Ok(TransportPlan {
        plan,
        value,
        potentials: None,
        iterations: augmented.iterations,
        residual: (actual_mass - transported_mass).abs(),
        status: SolverStatus::Converged,
    })
}

/// Return only the exact partial transport objective.
pub fn partial_wasserstein2(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    transported_mass: f64,
) -> Result<f64> {
    Ok(partial_wasserstein(source, target, cost, transported_mass)?.value)
}

/// Solve Lagrangian partial transport with row and column capacity constraints.
///
/// The optimized shifted objective is `sum(plan * (cost - mass_penalty))`;
/// unmatched mass is routed through zero-cost dummy atoms. The returned
/// `value` is the unshifted linear transport cost.
pub fn partial_wasserstein_lagrange(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    mass_penalty: Option<f64>,
) -> Result<TransportPlan> {
    let source_mass = validate::distribution(source, "partial Lagrange source")?;
    let target_mass = validate::distribution(target, "partial Lagrange target")?;
    validate::cost_matrix(cost, source.len(), target.len())?;
    let maximum_cost = (0..cost.nrows())
        .flat_map(|i| (0..cost.ncols()).map(move |j| cost[(i, j)]))
        .fold(f64::NEG_INFINITY, f64::max);
    let mass_penalty = mass_penalty.unwrap_or(maximum_cost + 1.0);
    if !mass_penalty.is_finite() {
        return Err(Error::InvalidParameter {
            name: "mass_penalty",
            requirement: "finite",
        });
    }
    let mut augmented_source = source.to_vec();
    augmented_source.push(target_mass);
    let mut augmented_target = target.to_vec();
    augmented_target.push(source_mass);
    let augmented_cost = Mat::<f64>::from_fn(source.len() + 1, target.len() + 1, |i, j| {
        if i < source.len() && j < target.len() {
            cost[(i, j)] - mass_penalty
        } else {
            0.0
        }
    });
    let augmented = exact::emd(&augmented_source, &augmented_target, &augmented_cost)?;
    let plan = Mat::<f64>::from_fn(source.len(), target.len(), |i, j| augmented.plan[(i, j)]);
    let mut value = 0.0;
    for i in 0..plan.nrows() {
        for j in 0..plan.ncols() {
            value += plan[(i, j)] * cost[(i, j)];
        }
    }
    Ok(TransportPlan {
        plan,
        value,
        potentials: None,
        iterations: augmented.iterations,
        residual: augmented.residual,
        status: augmented.status,
    })
}

/// Options for entropic partial transport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EntropicPartialOptions {
    /// Maximum Dykstra projection iterations.
    pub max_iterations: usize,
    /// Frobenius plan change required for convergence.
    pub tolerance: f64,
}

impl Default for EntropicPartialOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: 1e-10,
        }
    }
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    numerator / denominator.max(f64::MIN_POSITIVE)
}

/// Entropically regularized partial Wasserstein transport.
///
/// This is Dykstra's cyclic Bregman projection onto row-capacity,
/// column-capacity, and total-mass constraints.
pub fn entropic_partial_wasserstein(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    transported_mass: f64,
    options: EntropicPartialOptions,
) -> Result<TransportPlan> {
    validate::cost_matrix(cost, source.len(), target.len())?;
    validate_mass(source, target, transported_mass)?;
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
    let minimum_cost = (0..cost.nrows())
        .flat_map(|i| (0..cost.ncols()).map(move |j| cost[(i, j)]))
        .fold(f64::INFINITY, f64::min);
    let mut plan = Mat::<f64>::from_fn(cost.nrows(), cost.ncols(), |i, j| {
        (-(cost[(i, j)] - minimum_cost) / regularization).exp()
    });
    let kernel_mass = (0..plan.nrows())
        .map(|i| (0..plan.ncols()).map(|j| plan[(i, j)]).sum::<f64>())
        .sum::<f64>();
    if kernel_mass <= 0.0 || !kernel_mass.is_finite() {
        return Err(Error::DidNotConverge {
            algorithm: "entropic partial Wasserstein initialization",
            iterations: 0,
            residual: f64::INFINITY,
        });
    }
    for i in 0..plan.nrows() {
        for j in 0..plan.ncols() {
            plan[(i, j)] *= transported_mass / kernel_mass;
        }
    }
    let mut first_correction = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |_, _| 1.0);
    let mut second_correction = first_correction.clone();
    let mut third_correction = first_correction.clone();
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        let previous = plan.clone();

        let before_first = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
            plan[(i, j)] * first_correction[(i, j)]
        });
        let row_sums = (0..plan.nrows())
            .map(|i| (0..plan.ncols()).map(|j| before_first[(i, j)]).sum::<f64>())
            .collect::<Vec<_>>();
        let after_first = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
            before_first[(i, j)] * safe_ratio(source[i], row_sums[i]).min(1.0)
        });
        for i in 0..plan.nrows() {
            for j in 0..plan.ncols() {
                first_correction[(i, j)] *= safe_ratio(plan[(i, j)], after_first[(i, j)]);
            }
        }

        let before_second = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
            after_first[(i, j)] * second_correction[(i, j)]
        });
        let column_sums = (0..plan.ncols())
            .map(|j| {
                (0..plan.nrows())
                    .map(|i| before_second[(i, j)])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let after_second = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
            before_second[(i, j)] * safe_ratio(target[j], column_sums[j]).min(1.0)
        });
        for i in 0..plan.nrows() {
            for j in 0..plan.ncols() {
                second_correction[(i, j)] *= safe_ratio(after_first[(i, j)], after_second[(i, j)]);
            }
        }

        let before_third = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
            after_second[(i, j)] * third_correction[(i, j)]
        });
        let mass = (0..plan.nrows())
            .map(|i| (0..plan.ncols()).map(|j| before_third[(i, j)]).sum::<f64>())
            .sum::<f64>();
        plan = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
            before_third[(i, j)] * safe_ratio(transported_mass, mass)
        });
        for i in 0..plan.nrows() {
            for j in 0..plan.ncols() {
                third_correction[(i, j)] *= safe_ratio(after_second[(i, j)], plan[(i, j)]);
            }
        }

        residual = 0.0;
        for i in 0..plan.nrows() {
            for j in 0..plan.ncols() {
                residual += (plan[(i, j)] - previous[(i, j)]).powi(2);
            }
        }
        residual = residual.sqrt();
        iterations = iteration;
        if residual <= options.tolerance {
            status = SolverStatus::Converged;
            break;
        }
    }
    let mut value = 0.0;
    for i in 0..plan.nrows() {
        for j in 0..plan.ncols() {
            value += plan[(i, j)] * cost[(i, j)];
        }
    }
    Ok(TransportPlan {
        plan,
        value,
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
    fn exact_partial_moves_requested_mass_only() {
        let source = [0.5, 0.5];
        let target = [0.5, 0.5];
        let cost = Mat::<f64>::from_fn(2, 2, |i, j| [[0.0, 100.0], [100.0, 1.0]][i][j]);
        let result = partial_wasserstein(&source, &target, &cost, 0.5).unwrap();
        let mass = result.source_marginal().iter().copied().sum::<f64>();
        assert!((mass - 0.5).abs() < 1e-12);
        assert!(result.value.abs() < 1e-12);
    }

    #[test]
    fn entropic_partial_respects_caps_and_mass() {
        let source = [0.4, 0.6];
        let target = [0.7, 0.3];
        let cost = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);
        let result = entropic_partial_wasserstein(
            &source,
            &target,
            &cost,
            0.2,
            0.5,
            EntropicPartialOptions::default(),
        )
        .unwrap();
        assert_eq!(result.status, SolverStatus::Converged);
        let rows = result.source_marginal();
        let columns = result.target_marginal();
        assert!(rows
            .iter()
            .zip(source)
            .all(|(&actual, cap)| actual <= cap + 1e-8));
        assert!(columns
            .iter()
            .zip(target)
            .all(|(&actual, cap)| actual <= cap + 1e-8));
        assert!((rows.iter().sum::<f64>() - 0.5).abs() < 1e-8);
    }

    #[test]
    fn lagrange_partial_transport_selects_profitable_mass() {
        let source = [0.1, 0.2];
        let target = [0.1, 0.1];
        let cost = Mat::<f64>::from_fn(2, 2, |i, j| [[0.0, 1.0], [2.0, 3.0]][i][j]);
        let unconstrained = partial_wasserstein_lagrange(&source, &target, &cost, None).unwrap();
        assert!((unconstrained.plan[(0, 0)] - 0.1).abs() < 1e-12);
        assert!((unconstrained.plan[(1, 1)] - 0.1).abs() < 1e-12);
        let penalized = partial_wasserstein_lagrange(&source, &target, &cost, Some(2.0)).unwrap();
        assert!((penalized.plan[(0, 0)] - 0.1).abs() < 1e-12);
        assert!(penalized.plan[(1, 1)].abs() < 1e-12);
    }
}
