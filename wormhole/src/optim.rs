//! Generic conditional-gradient optimization over the transport polytope.

use crate::error::{Error, Result};
use crate::exact;
use crate::result::SolverStatus;
use crate::validate;
use faer::Mat;

/// Inputs to a regularized balanced transport problem.
#[derive(Clone, Copy, Debug)]
pub struct RegularizedTransportProblem<'a> {
    /// Source histogram.
    pub source: &'a [f64],
    /// Target histogram.
    pub target: &'a [f64],
    /// Linear transport cost.
    pub cost: &'a Mat<f64>,
    /// Positive multiplier of the supplied regularizer.
    pub regularization: f64,
    /// Optional feasible initial coupling.
    pub initial: Option<&'a Mat<f64>>,
}

/// Stopping and Armijo line-search options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConditionalGradientOptions {
    /// Maximum Frank-Wolfe iterations.
    pub max_iterations: usize,
    /// Frank-Wolfe dual-gap threshold.
    pub gap_tolerance: f64,
    /// Absolute objective-change threshold.
    pub objective_tolerance: f64,
    /// Armijo sufficient-decrease coefficient in `(0, 1)`.
    pub armijo: f64,
    /// Multiplicative line-search backtracking in `(0, 1)`.
    pub backtracking: f64,
    /// Maximum backtracking trials per iteration.
    pub max_line_search_iterations: usize,
}

impl Default for ConditionalGradientOptions {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            gap_tolerance: 1e-9,
            objective_tolerance: 1e-9,
            armijo: 1e-4,
            backtracking: 0.5,
            max_line_search_iterations: 50,
        }
    }
}

/// Result of generic regularized transport optimization.
#[derive(Clone, Debug)]
pub struct RegularizedTransport {
    /// Feasible transport plan.
    pub plan: Mat<f64>,
    /// Full linear-plus-regularizer objective.
    pub objective: f64,
    /// Linear transport contribution.
    pub linear_value: f64,
    /// Unscaled regularizer value.
    pub regularizer_value: f64,
    /// Number of Frank-Wolfe steps.
    pub iterations: usize,
    /// Last Frank-Wolfe gap.
    pub gap: f64,
    /// Maximum marginal violation.
    pub residual: f64,
    /// Solver termination status.
    pub status: SolverStatus,
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

fn validate_options(options: ConditionalGradientOptions) -> Result<()> {
    if options.max_iterations == 0 || options.max_line_search_iterations == 0 {
        return Err(Error::InvalidParameter {
            name: "conditional-gradient iteration limits",
            requirement: "positive",
        });
    }
    validate::finite_positive(
        options.gap_tolerance,
        "gap_tolerance",
        "finite and strictly positive",
    )?;
    validate::finite_positive(
        options.objective_tolerance,
        "objective_tolerance",
        "finite and strictly positive",
    )?;
    if !options.armijo.is_finite() || options.armijo <= 0.0 || options.armijo >= 1.0 {
        return Err(Error::InvalidParameter {
            name: "armijo",
            requirement: "finite and strictly between zero and one",
        });
    }
    if !options.backtracking.is_finite()
        || options.backtracking <= 0.0
        || options.backtracking >= 1.0
    {
        return Err(Error::InvalidParameter {
            name: "backtracking",
            requirement: "finite and strictly between zero and one",
        });
    }
    Ok(())
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

fn validate_plan(plan: &Mat<f64>, source: &[f64], target: &[f64]) -> Result<()> {
    if plan.nrows() != source.len() || plan.ncols() != target.len() {
        return Err(Error::ShapeMismatch {
            context: "conditional-gradient initial plan",
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
    if marginal_residual(plan, source, target) > 1e-9 {
        return Err(Error::Infeasible {
            context: "conditional-gradient initial plan has incorrect marginals",
        });
    }
    Ok(())
}

fn evaluate<F>(
    plan: &Mat<f64>,
    cost: &Mat<f64>,
    regularization: f64,
    regularizer: &F,
) -> Result<(f64, f64, f64)>
where
    F: Fn(&Mat<f64>) -> Result<f64>,
{
    let linear = matrix_inner(plan, cost);
    let regularizer_value = regularizer(plan)?;
    if !regularizer_value.is_finite() {
        return Err(Error::InvalidParameter {
            name: "regularizer value",
            requirement: "finite",
        });
    }
    Ok((
        linear + regularization * regularizer_value,
        linear,
        regularizer_value,
    ))
}

/// Minimize a differentiable regularized objective over balanced couplings.
///
/// `regularizer` returns `f(plan)` and `regularizer_gradient` returns `��f(plan)`.
/// Every linear minimization oracle is solved exactly by the workspace EMD
/// implementation.
pub fn conditional_gradient<F, G>(
    problem: RegularizedTransportProblem<'_>,
    regularizer: F,
    regularizer_gradient: G,
    options: ConditionalGradientOptions,
) -> Result<RegularizedTransport>
where
    F: Fn(&Mat<f64>) -> Result<f64>,
    G: Fn(&Mat<f64>) -> Result<Mat<f64>>,
{
    validate::balanced_distributions(problem.source, problem.target)?;
    validate::cost_matrix(problem.cost, problem.source.len(), problem.target.len())?;
    validate::finite_positive(
        problem.regularization,
        "regularization",
        "finite and strictly positive",
    )?;
    validate_options(options)?;
    let total_mass = problem.source.iter().sum::<f64>();
    let mut plan = match problem.initial {
        Some(initial) => {
            validate_plan(initial, problem.source, problem.target)?;
            initial.clone()
        }
        None => Mat::<f64>::from_fn(problem.source.len(), problem.target.len(), |i, j| {
            problem.source[i] * problem.target[j] / total_mass
        }),
    };
    let (mut objective, mut linear_value, mut regularizer_value) =
        evaluate(&plan, problem.cost, problem.regularization, &regularizer)?;
    let mut gap = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;

    for iteration in 1..=options.max_iterations {
        let regularizer_derivative = regularizer_gradient(&plan)?;
        if regularizer_derivative.nrows() != plan.nrows()
            || regularizer_derivative.ncols() != plan.ncols()
        {
            return Err(Error::ShapeMismatch {
                context: "regularizer gradient",
                left: (
                    regularizer_derivative.nrows(),
                    regularizer_derivative.ncols(),
                ),
                right: (plan.nrows(), plan.ncols()),
            });
        }
        let total_gradient = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
            let derivative = regularizer_derivative[(i, j)];
            if !derivative.is_finite() {
                f64::NAN
            } else {
                problem.cost[(i, j)] + problem.regularization * derivative
            }
        });
        if (0..total_gradient.nrows())
            .any(|i| (0..total_gradient.ncols()).any(|j| !total_gradient[(i, j)].is_finite()))
        {
            return Err(Error::InvalidParameter {
                name: "regularizer gradient",
                requirement: "finite",
            });
        }
        let vertex = exact::emd(problem.source, problem.target, &total_gradient)?.plan;
        let direction = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
            vertex[(i, j)] - plan[(i, j)]
        });
        let directional_derivative = matrix_inner(&total_gradient, &direction);
        gap = -directional_derivative;
        if gap <= options.gap_tolerance {
            status = SolverStatus::Converged;
            break;
        }

        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..options.max_line_search_iterations {
            let candidate = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
                plan[(i, j)] + step * direction[(i, j)]
            });
            let values = evaluate(
                &candidate,
                problem.cost,
                problem.regularization,
                &regularizer,
            )?;
            if values.0 <= objective + options.armijo * step * directional_derivative {
                accepted = Some((candidate, values));
                break;
            }
            step *= options.backtracking;
        }
        let Some((candidate, values)) = accepted else {
            return Err(Error::DidNotConverge {
                algorithm: "conditional-gradient line search",
                iterations: iteration,
                residual: gap,
            });
        };
        let previous = objective;
        plan = candidate;
        objective = values.0;
        linear_value = values.1;
        regularizer_value = values.2;
        iterations = iteration;
        if (previous - objective).abs() <= options.objective_tolerance {
            status = SolverStatus::Converged;
            break;
        }
    }

    Ok(RegularizedTransport {
        residual: marginal_residual(&plan, problem.source, problem.target),
        plan,
        objective,
        linear_value,
        regularizer_value,
        iterations,
        gap,
        status,
    })
}

/// Squared-L2 regularized optimal transport.
pub fn squared_l2_transport(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    regularization: f64,
    options: ConditionalGradientOptions,
) -> Result<RegularizedTransport> {
    conditional_gradient(
        RegularizedTransportProblem {
            source,
            target,
            cost,
            regularization,
            initial: None,
        },
        |plan| {
            Ok(0.5
                * (0..plan.nrows())
                    .flat_map(|i| (0..plan.ncols()).map(move |j| plan[(i, j)].powi(2)))
                    .sum::<f64>())
        },
        |plan| Ok(plan.clone()),
        options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_l2_solver_matches_regularized_objective() {
        let source = [0.2, 0.5, 0.3];
        let target = [0.4, 0.1, 0.5];
        let cost = Mat::<f64>::from_fn(3, 3, |i, j| {
            [[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]][i][j]
        });
        let result = squared_l2_transport(
            &source,
            &target,
            &cost,
            2.0,
            ConditionalGradientOptions {
                max_iterations: 1_000,
                gap_tolerance: 1e-12,
                objective_tolerance: 1e-12,
                ..ConditionalGradientOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, SolverStatus::Converged);
        assert!((result.linear_value - 0.4).abs() < 1e-10);
        assert!((result.regularizer_value - 0.11).abs() < 1e-10);
        assert!((result.objective - 0.62).abs() < 1e-10);
    }
}
