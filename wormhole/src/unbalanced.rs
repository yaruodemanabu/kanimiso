//! Unbalanced optimal transport with KL marginal relaxation.

use crate::error::{Error, Result};
use crate::result::{BarycenterResult, DualPotentials, SolverStatus, TransportPlan};
use crate::validate;
use faer::Mat;

/// Reference measure used by the plan regularizer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum UnbalancedRegularization {
    /// `KL(plan, source ⊗ target)`, matching POT 0.9.7's default.
    #[default]
    KullbackLeibler,
    /// Negative entropy, equivalent to KL against an all-ones matrix.
    Entropy,
}

/// Options for KL-relaxed unbalanced Sinkhorn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnbalancedOptions {
    /// KL penalty on the source marginal.
    pub source_penalty: f64,
    /// KL penalty on the target marginal.
    pub target_penalty: f64,
    /// Maximum alternating updates.
    pub max_iterations: usize,
    /// Maximum potential change required for convergence.
    pub tolerance: f64,
    /// Reference measure used to regularize the transport plan.
    pub plan_regularization: UnbalancedRegularization,
}

impl Default for UnbalancedOptions {
    fn default() -> Self {
        Self {
            source_penalty: 1.0,
            target_penalty: 1.0,
            max_iterations: 1_000,
            tolerance: 1e-9,
            plan_regularization: UnbalancedRegularization::KullbackLeibler,
        }
    }
}

/// Stopping options for a fixed-support unbalanced barycenter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnbalancedBarycenterOptions {
    /// Maximum generalized Sinkhorn iterations.
    pub max_iterations: usize,
    /// Relative maximum barycenter change required for convergence.
    pub tolerance: f64,
}

impl Default for UnbalancedBarycenterOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: 1e-6,
        }
    }
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

/// Solve entropic unbalanced OT with KL penalties on both marginals.
pub fn sinkhorn_unbalanced(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    mass_penalty: f64,
) -> Result<TransportPlan> {
    sinkhorn_unbalanced_with_options(
        source,
        target,
        cost,
        regularization,
        UnbalancedOptions {
            source_penalty: mass_penalty,
            target_penalty: mass_penalty,
            ..UnbalancedOptions::default()
        },
    )
}

/// Solve entropic unbalanced OT with asymmetric KL penalties.
pub fn sinkhorn_unbalanced_with_options(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    options: UnbalancedOptions,
) -> Result<TransportPlan> {
    validate::distribution(source, "source distribution")?;
    validate::distribution(target, "target distribution")?;
    validate::cost_matrix(cost, source.len(), target.len())?;
    validate::finite_positive(
        regularization,
        "regularization",
        "finite and strictly positive",
    )?;
    validate::finite_positive(
        options.source_penalty,
        "source_penalty",
        "finite and strictly positive",
    )?;
    validate::finite_positive(
        options.target_penalty,
        "target_penalty",
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

    let source_exponent = options.source_penalty / (options.source_penalty + regularization);
    let target_exponent = options.target_penalty / (options.target_penalty + regularization);
    let mut source_potential = source
        .iter()
        .map(|&weight| if weight > 0.0 { 0.0 } else { f64::NEG_INFINITY })
        .collect::<Vec<_>>();
    let mut target_potential = target
        .iter()
        .map(|&weight| if weight > 0.0 { 0.0 } else { f64::NEG_INFINITY })
        .collect::<Vec<_>>();
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    let effective_cost = |i: usize, j: usize| match options.plan_regularization {
        UnbalancedRegularization::Entropy => cost[(i, j)],
        UnbalancedRegularization::KullbackLeibler => {
            if source[i] > 0.0 && target[j] > 0.0 {
                cost[(i, j)] - regularization * (source[i].ln() + target[j].ln())
            } else {
                f64::INFINITY
            }
        }
    };
    for iteration in 1..=options.max_iterations {
        let previous_source = source_potential.clone();
        let previous_target = target_potential.clone();
        for i in 0..source.len() {
            if source[i] == 0.0 {
                continue;
            }
            let normalization = log_sum_exp(
                (0..target.len())
                    .map(|j| (target_potential[j] - effective_cost(i, j)) / regularization),
            );
            source_potential[i] =
                source_exponent * regularization * (source[i].ln() - normalization);
        }
        for j in 0..target.len() {
            if target[j] == 0.0 {
                continue;
            }
            let normalization = log_sum_exp(
                (0..source.len())
                    .map(|i| (source_potential[i] - effective_cost(i, j)) / regularization),
            );
            target_potential[j] =
                target_exponent * regularization * (target[j].ln() - normalization);
        }
        residual = 0.0_f64;
        for (&next, &previous) in source_potential.iter().zip(&previous_source) {
            if next.is_finite() && previous.is_finite() {
                residual = residual.max((next - previous).abs());
            }
        }
        for (&next, &previous) in target_potential.iter().zip(&previous_target) {
            if next.is_finite() && previous.is_finite() {
                residual = residual.max((next - previous).abs());
            }
        }
        iterations = iteration;
        if residual <= options.tolerance {
            status = SolverStatus::Converged;
            break;
        }
    }

    let plan = Mat::<f64>::from_fn(source.len(), target.len(), |i, j| {
        if source_potential[i].is_finite() && target_potential[j].is_finite() {
            ((source_potential[i] + target_potential[j] - effective_cost(i, j)) / regularization)
                .exp()
        } else {
            0.0
        }
    });
    let mut value = 0.0;
    for i in 0..source.len() {
        for j in 0..target.len() {
            value += plan[(i, j)] * cost[(i, j)];
        }
    }
    Ok(TransportPlan {
        plan,
        value,
        potentials: Some(DualPotentials {
            source: source_potential,
            target: target_potential,
        }),
        iterations,
        residual,
        status,
    })
}

fn generalized_kl(left: &[f64], right: &[f64]) -> Result<f64> {
    if left.len() != right.len() {
        return Err(Error::ShapeMismatch {
            context: "KL vectors",
            left: (1, left.len()),
            right: (1, right.len()),
        });
    }
    let mut value = 0.0;
    for (index, (&p, &q)) in left.iter().zip(right).enumerate() {
        if !p.is_finite() || p < 0.0 {
            return Err(Error::InvalidWeight { index, value: p });
        }
        if !q.is_finite() || q < 0.0 {
            return Err(Error::InvalidWeight { index, value: q });
        }
        if p > 0.0 {
            if q == 0.0 {
                return Ok(f64::INFINITY);
            }
            value += p * (p / q).ln() - p + q;
        } else {
            value += q;
        }
    }
    Ok(value)
}

/// Full POT-default KL objective of an unbalanced plan.
///
/// The plan reference measure is the outer product `source ⊗ target`.
pub fn unbalanced_kl_objective(
    plan: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    source_penalty: f64,
    target_penalty: f64,
) -> Result<f64> {
    validate::distribution(source, "source distribution")?;
    validate::distribution(target, "target distribution")?;
    validate::cost_matrix(cost, source.len(), target.len())?;
    if plan.nrows() != source.len() || plan.ncols() != target.len() {
        return Err(Error::ShapeMismatch {
            context: "unbalanced plan and marginals",
            left: (plan.nrows(), plan.ncols()),
            right: (source.len(), target.len()),
        });
    }
    validate::finite_positive(
        regularization,
        "regularization",
        "finite and strictly positive",
    )?;
    validate::finite_non_negative(source_penalty, "source_penalty", "finite and non-negative")?;
    validate::finite_non_negative(target_penalty, "target_penalty", "finite and non-negative")?;
    let mut source_marginal = vec![0.0; source.len()];
    let mut target_marginal = vec![0.0; target.len()];
    let mut objective = 0.0;
    for i in 0..plan.nrows() {
        for j in 0..plan.ncols() {
            let mass = plan[(i, j)];
            if !mass.is_finite() || mass < 0.0 {
                return Err(Error::InvalidWeight {
                    index: i * plan.ncols() + j,
                    value: mass,
                });
            }
            source_marginal[i] += mass;
            target_marginal[j] += mass;
            objective += mass * cost[(i, j)];
            let reference = source[i] * target[j];
            if mass > 0.0 {
                if reference == 0.0 {
                    return Ok(f64::INFINITY);
                }
                objective += regularization * (mass * (mass / reference).ln() - mass + reference);
            } else {
                objective += regularization * reference;
            }
        }
    }
    objective += source_penalty * generalized_kl(&source_marginal, source)?;
    objective += target_penalty * generalized_kl(&target_marginal, target)?;
    Ok(objective)
}

/// Return only the linear cost of an unbalanced Sinkhorn plan.
pub fn sinkhorn_unbalanced2(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    mass_penalty: f64,
) -> Result<f64> {
    Ok(sinkhorn_unbalanced(source, target, cost, regularization, mass_penalty)?.value)
}

/// Compute a fixed-support entropic unbalanced Wasserstein barycenter.
///
/// Columns of `distributions` are non-negative input histograms. Unlike a
/// balanced barycenter, their total masses need not agree.
pub fn barycenter_unbalanced(
    distributions: &Mat<f64>,
    cost: &Mat<f64>,
    regularization: f64,
    mass_penalty: f64,
    mixture: Option<&[f64]>,
) -> Result<BarycenterResult> {
    barycenter_unbalanced_with_options(
        distributions,
        cost,
        regularization,
        mass_penalty,
        mixture,
        UnbalancedBarycenterOptions::default(),
    )
}

/// Compute an unbalanced barycenter with explicit stopping options.
pub fn barycenter_unbalanced_with_options(
    distributions: &Mat<f64>,
    cost: &Mat<f64>,
    regularization: f64,
    mass_penalty: f64,
    mixture: Option<&[f64]>,
    options: UnbalancedBarycenterOptions,
) -> Result<BarycenterResult> {
    if distributions.nrows() == 0 || distributions.ncols() == 0 {
        return Err(Error::EmptyInput {
            name: "unbalanced barycenter distributions",
        });
    }
    for column in 0..distributions.ncols() {
        let histogram = (0..distributions.nrows())
            .map(|row| distributions[(row, column)])
            .collect::<Vec<_>>();
        validate::distribution(&histogram, "unbalanced barycenter distribution")?;
    }
    validate::cost_matrix(cost, distributions.nrows(), distributions.nrows())?;
    validate::finite_positive(
        regularization,
        "regularization",
        "finite and strictly positive",
    )?;
    validate::finite_positive(mass_penalty, "mass_penalty", "finite and strictly positive")?;
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

    let mut mixture = match mixture {
        Some(weights) => {
            if weights.len() != distributions.ncols() {
                return Err(Error::ShapeMismatch {
                    context: "unbalanced barycenter mixture",
                    left: (distributions.ncols(), 1),
                    right: (weights.len(), 1),
                });
            }
            weights.to_vec()
        }
        None => validate::uniform(distributions.ncols())?,
    };
    let mixture_mass = validate::distribution(&mixture, "unbalanced barycenter mixture")?;
    for weight in &mut mixture {
        *weight /= mixture_mass;
    }

    let dimension = distributions.nrows();
    let count = distributions.ncols();
    let kernel = Mat::<f64>::from_fn(dimension, dimension, |i, j| {
        (-cost[(i, j)] / regularization).exp()
    });
    let exponent = mass_penalty / (mass_penalty + regularization);
    let complement = 1.0 - exponent;
    let mut right_scaling = Mat::<f64>::from_fn(dimension, count, |_, _| 1.0);
    let mut barycenter = vec![1.0; dimension];
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;

    for iteration in 1..=options.max_iterations {
        let previous = barycenter.clone();
        let mut left_scaling = Mat::<f64>::zeros(dimension, count);
        for distribution in 0..count {
            for i in 0..dimension {
                let denominator = (0..dimension)
                    .map(|j| kernel[(i, j)] * right_scaling[(j, distribution)])
                    .sum::<f64>();
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err(Error::DidNotConverge {
                        algorithm: "unbalanced barycenter",
                        iterations: iteration,
                        residual: f64::INFINITY,
                    });
                }
                left_scaling[(i, distribution)] =
                    (distributions[(i, distribution)] / denominator).powf(exponent);
            }
        }
        let mut transported = Mat::<f64>::zeros(dimension, count);
        for distribution in 0..count {
            for j in 0..dimension {
                transported[(j, distribution)] = (0..dimension)
                    .map(|i| kernel[(i, j)] * left_scaling[(i, distribution)])
                    .sum::<f64>();
                if !transported[(j, distribution)].is_finite()
                    || transported[(j, distribution)] <= 0.0
                {
                    return Err(Error::DidNotConverge {
                        algorithm: "unbalanced barycenter",
                        iterations: iteration,
                        residual: f64::INFINITY,
                    });
                }
            }
        }
        for j in 0..dimension {
            barycenter[j] = (0..count)
                .map(|distribution| {
                    mixture[distribution] * transported[(j, distribution)].powf(complement)
                })
                .sum::<f64>()
                .powf(complement.recip());
        }
        for distribution in 0..count {
            for j in 0..dimension {
                right_scaling[(j, distribution)] =
                    (barycenter[j] / transported[(j, distribution)]).powf(exponent);
            }
        }
        let current_maximum = barycenter.iter().copied().fold(0.0_f64, f64::max);
        let previous_maximum = previous.iter().copied().fold(0.0_f64, f64::max);
        residual = barycenter
            .iter()
            .zip(previous)
            .map(|(&next, old)| (next - old).abs())
            .fold(0.0_f64, f64::max)
            / current_maximum.max(previous_maximum).max(1.0);
        iterations = iteration;
        if residual <= options.tolerance && iteration > 10 {
            status = SolverStatus::Converged;
            break;
        }
    }

    Ok(BarycenterResult {
        weights: barycenter,
        iterations,
        residual,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unequal_mass_problem_converges() {
        let source = [1.0, 1.0];
        let target = [0.25, 0.75];
        let cost = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);
        let result = sinkhorn_unbalanced(&source, &target, &cost, 0.2, 1.0)
            .expect("valid unbalanced problem");
        assert_eq!(result.status, SolverStatus::Converged);
        assert!(result.plan[(0, 0)].is_finite());
        assert!(result.plan[(1, 1)].is_finite());
        assert!(result.residual < 1e-8);
    }

    #[test]
    fn larger_mass_penalty_better_matches_marginals() {
        let source = [1.0];
        let target = [2.0];
        let cost = Mat::<f64>::zeros(1, 1);
        let loose = sinkhorn_unbalanced(&source, &target, &cost, 0.5, 0.1).unwrap();
        let strict = sinkhorn_unbalanced(&source, &target, &cost, 0.5, 100.0).unwrap();
        let loose_error = (loose.source_marginal()[0] - source[0]).abs()
            + (loose.target_marginal()[0] - target[0]).abs();
        let strict_error = (strict.source_marginal()[0] - source[0]).abs()
            + (strict.target_marginal()[0] - target[0]).abs();
        assert!(strict_error <= loose_error + 1e-12);
    }

    #[test]
    fn unbalanced_barycenter_accepts_different_input_masses() {
        let distributions = Mat::<f64>::from_fn(2, 2, |i, j| [[1.0, 2.0], [0.0, 1.0]][i][j]);
        let cost = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);
        let result = barycenter_unbalanced(&distributions, &cost, 0.5, 1.0, None).unwrap();
        assert_eq!(result.status, SolverStatus::Converged);
        assert!(result
            .weights
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }
}
