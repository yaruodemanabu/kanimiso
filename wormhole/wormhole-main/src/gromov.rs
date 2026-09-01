//! Gromov-Wasserstein and fused Gromov-Wasserstein transport.

use crate::error::{Error, Result};
use crate::exact;
use crate::metrics::{self, Metric};
use crate::partial::{self, EntropicPartialOptions};
use crate::result::SolverStatus;
use crate::sinkhorn;
use crate::validate;
use faer::Mat;

/// Inner linearized solver for Gromov-Wasserstein iterations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum GromovMethod {
    /// Conditional gradient with exact EMD subproblems.
    #[default]
    Exact,
    /// Entropic projected fixed-point steps.
    Entropic {
        /// Positive entropy regularization used by each inner Sinkhorn solve.
        regularization: f64,
    },
}

/// Stopping options for Gromov-Wasserstein solvers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GromovOptions {
    /// Maximum outer iterations.
    pub max_iterations: usize,
    /// Absolute objective change required for convergence.
    pub tolerance: f64,
    /// Linearized subproblem method.
    pub method: GromovMethod,
}

impl Default for GromovOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: 1e-9,
            method: GromovMethod::Exact,
        }
    }
}

/// Result of a quadratic Gromov transport solve.
#[derive(Clone, Debug)]
pub struct GromovResult {
    /// Coupling between source and target structure atoms.
    pub plan: Mat<f64>,
    /// Squared-loss Gromov-Wasserstein part of the objective.
    pub structure_value: f64,
    /// Linear feature-cost part, zero for plain GW.
    pub feature_value: f64,
    /// Combined fused objective.
    pub value: f64,
    /// Number of outer iterations.
    pub iterations: usize,
    /// Last absolute objective change.
    pub residual: f64,
    /// Termination status.
    pub status: SolverStatus,
}

/// Outer options for a fixed-support Gromov-Wasserstein barycenter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GromovBarycenterOptions {
    /// Maximum structure-update iterations.
    pub max_iterations: usize,
    /// Frobenius structure change required for convergence.
    pub tolerance: f64,
    /// Options for each inner Gromov transport solve.
    pub transport: GromovOptions,
}

impl Default for GromovBarycenterOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-9,
            transport: GromovOptions::default(),
        }
    }
}

/// Structure, couplings, and diagnostics of a Gromov barycenter.
#[derive(Clone, Debug)]
pub struct GromovBarycenter {
    /// Learned barycenter structure matrix.
    pub structure: Mat<f64>,
    /// Couplings from barycenter atoms to each input structure.
    pub plans: Vec<Mat<f64>>,
    /// Number of structure updates.
    pub iterations: usize,
    /// Last Frobenius structure change.
    pub residual: f64,
    /// Solver termination status.
    pub status: SolverStatus,
}

/// Inputs to a fixed-support fused Gromov-Wasserstein barycenter.
#[derive(Clone, Copy, Debug)]
pub struct FusedGromovBarycenterProblem<'a> {
    /// Node-feature matrices for each input space.
    pub features: &'a [Mat<f64>],
    /// Structure matrices for each input space.
    pub structures: &'a [Mat<f64>],
    /// Input node histograms, uniform when omitted.
    pub distributions: Option<&'a [Vec<f64>]>,
    /// Fixed node histogram of the barycenter.
    pub barycenter_weights: &'a [f64],
    /// Coefficients of the input spaces, uniform when omitted.
    pub mixture: Option<&'a [f64]>,
    /// Initial barycenter feature matrix.
    pub initial_features: &'a Mat<f64>,
    /// Initial barycenter structure matrix.
    pub initial_structure: &'a Mat<f64>,
}

/// Outer options for a fused Gromov-Wasserstein barycenter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusedGromovBarycenterOptions {
    /// Relative weight of structural distortion versus feature transport.
    pub structure_weight: f64,
    /// Maximum alternating barycenter updates.
    pub max_iterations: usize,
    /// Combined feature/structure Frobenius change required for convergence.
    pub tolerance: f64,
    /// Whether to update the barycenter structure.
    pub update_structure: bool,
    /// Whether to update the barycenter features.
    pub update_features: bool,
    /// Options for each inner fused Gromov transport solve.
    pub transport: GromovOptions,
}

impl Default for FusedGromovBarycenterOptions {
    fn default() -> Self {
        Self {
            structure_weight: 0.5,
            max_iterations: 100,
            tolerance: 1e-9,
            update_structure: true,
            update_features: true,
            transport: GromovOptions::default(),
        }
    }
}

/// Features, structure, couplings, and diagnostics of an FGW barycenter.
#[derive(Clone, Debug)]
pub struct FusedGromovBarycenter {
    /// Learned barycenter node features.
    pub features: Mat<f64>,
    /// Learned barycenter structure.
    pub structure: Mat<f64>,
    /// Couplings from barycenter nodes to each input space.
    pub plans: Vec<Mat<f64>>,
    /// Number of outer updates.
    pub iterations: usize,
    /// Last combined feature/structure Frobenius change.
    pub residual: f64,
    /// Solver termination status.
    pub status: SolverStatus,
}

/// Options for Bregman alternating projected-gradient Gromov transport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BapgOptions {
    /// Positive Bregman proximal regularization.
    pub regularization: f64,
    /// Maximum alternating projection iterations.
    pub max_iterations: usize,
    /// Frobenius plan-change threshold.
    pub tolerance: f64,
    /// Include squared-loss terms depending only on prescribed marginals.
    pub marginal_loss: bool,
}

impl Default for BapgOptions {
    fn default() -> Self {
        Self {
            regularization: 0.1,
            max_iterations: 1_000,
            tolerance: 1e-9,
            marginal_loss: false,
        }
    }
}

/// Linearized subproblem used by partial Gromov transport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PartialGromovMethod {
    /// Exact fixed-mass partial EMD oracle with conditional-gradient updates.
    Exact,
    /// Entropic fixed-mass partial OT projected fixed point.
    Entropic {
        /// Positive entropy regularization.
        regularization: f64,
        /// Dykstra options for each partial OT projection.
        options: EntropicPartialOptions,
    },
}

/// Options for partial GW and FGW.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartialGromovOptions {
    /// Total mass transported between the two structures.
    pub transported_mass: f64,
    /// Maximum outer iterations.
    pub max_iterations: usize,
    /// Absolute objective-change threshold.
    pub tolerance: f64,
    /// Exact or entropic linearized subproblem.
    pub method: PartialGromovMethod,
}

impl Default for PartialGromovOptions {
    fn default() -> Self {
        Self {
            transported_mass: 1.0,
            max_iterations: 1_000,
            tolerance: 1e-9,
            method: PartialGromovMethod::Exact,
        }
    }
}

fn validate_structure(matrix: &Mat<f64>, weights: &[f64], name: &'static str) -> Result<()> {
    validate::samples(matrix, name)?;
    if matrix.nrows() != matrix.ncols() || matrix.nrows() != weights.len() {
        return Err(Error::ShapeMismatch {
            context: "Gromov structure and marginal",
            left: (matrix.nrows(), matrix.ncols()),
            right: (weights.len(), weights.len()),
        });
    }
    Ok(())
}

fn validate_options(options: GromovOptions) -> Result<()> {
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
    if let GromovMethod::Entropic { regularization } = options.method {
        validate::finite_positive(
            regularization,
            "Gromov entropy regularization",
            "finite and strictly positive",
        )?;
    }
    Ok(())
}

fn initial_plan(source: &[f64], target: &[f64], mass: f64) -> Mat<f64> {
    Mat::<f64>::from_fn(source.len(), target.len(), |i, j| {
        source[i] * target[j] / mass
    })
}

fn square_loss_tensor(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    plan: &Mat<f64>,
) -> Mat<f64> {
    let source_constant = (0..source.len())
        .map(|i| {
            (0..source.len())
                .map(|k| source_structure[(i, k)].powi(2) * source[k])
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let target_constant = (0..target.len())
        .map(|j| {
            (0..target.len())
                .map(|l| target_structure[(j, l)].powi(2) * target[l])
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    Mat::<f64>::from_fn(source.len(), target.len(), |i, j| {
        let cross = (0..source.len())
            .map(|k| {
                (0..target.len())
                    .map(|l| source_structure[(i, k)] * plan[(k, l)] * target_structure[(j, l)])
                    .sum::<f64>()
            })
            .sum::<f64>();
        source_constant[i] + target_constant[j] - 2.0 * cross
    })
}

fn structure_objective(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    plan: &Mat<f64>,
) -> f64 {
    let tensor = square_loss_tensor(source_structure, target_structure, source, target, plan);
    let mut value = 0.0;
    for i in 0..plan.nrows() {
        for j in 0..plan.ncols() {
            value += tensor[(i, j)] * plan[(i, j)];
        }
    }
    value.max(0.0)
}

fn feature_objective(feature_cost: Option<&Mat<f64>>, plan: &Mat<f64>) -> f64 {
    let Some(cost) = feature_cost else {
        return 0.0;
    };
    let mut value = 0.0;
    for i in 0..plan.nrows() {
        for j in 0..plan.ncols() {
            value += cost[(i, j)] * plan[(i, j)];
        }
    }
    value
}

fn combined_objective(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    feature_cost: Option<&Mat<f64>>,
    structure_weight: f64,
    plan: &Mat<f64>,
) -> (f64, f64, f64) {
    let structure = structure_objective(source_structure, target_structure, source, target, plan);
    let feature = feature_objective(feature_cost, plan);
    let combined = structure_weight * structure + (1.0 - structure_weight) * feature;
    (structure, feature, combined)
}

/// Solve squared-loss Gromov-Wasserstein transport.
pub fn gromov_wasserstein(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    options: GromovOptions,
) -> Result<GromovResult> {
    solve_fused(
        source_structure,
        target_structure,
        source,
        target,
        None,
        1.0,
        options,
    )
}

/// Return only the squared-loss Gromov-Wasserstein objective.
pub fn gromov_wasserstein2(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    options: GromovOptions,
) -> Result<f64> {
    Ok(gromov_wasserstein(source_structure, target_structure, source, target, options)?.value)
}

/// Solve fused Gromov-Wasserstein transport.
///
/// `structure_weight` corresponds to POT's `alpha`: zero uses only feature
/// cost and one uses only structural distortion.
pub fn fused_gromov_wasserstein(
    feature_cost: &Mat<f64>,
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    structure_weight: f64,
    options: GromovOptions,
) -> Result<GromovResult> {
    solve_fused(
        source_structure,
        target_structure,
        source,
        target,
        Some(feature_cost),
        structure_weight,
        options,
    )
}

/// Return only the fused Gromov-Wasserstein objective.
pub fn fused_gromov_wasserstein2(
    feature_cost: &Mat<f64>,
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    structure_weight: f64,
    options: GromovOptions,
) -> Result<f64> {
    Ok(fused_gromov_wasserstein(
        feature_cost,
        source_structure,
        target_structure,
        source,
        target,
        structure_weight,
        options,
    )?
    .value)
}

/// Compute squared-loss GW with Bregman alternating projected gradients.
pub fn bapg_gromov_wasserstein(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    options: BapgOptions,
) -> Result<GromovResult> {
    bapg_solve(
        source_structure,
        target_structure,
        source,
        target,
        None,
        1.0,
        options,
    )
}

/// Compute squared-loss fused GW with Bregman alternating projected gradients.
pub fn bapg_fused_gromov_wasserstein(
    feature_cost: &Mat<f64>,
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    structure_weight: f64,
    options: BapgOptions,
) -> Result<GromovResult> {
    bapg_solve(
        source_structure,
        target_structure,
        source,
        target,
        Some(feature_cost),
        structure_weight,
        options,
    )
}

struct BapgGradientProblem<'a> {
    source_structure: &'a Mat<f64>,
    target_structure: &'a Mat<f64>,
    source: &'a [f64],
    target: &'a [f64],
    feature_cost: Option<&'a Mat<f64>>,
    structure_weight: f64,
}

fn bapg_gradient(
    problem: &BapgGradientProblem<'_>,
    marginal_loss: bool,
    plan: &Mat<f64>,
) -> Mat<f64> {
    if marginal_loss {
        let tensor = square_loss_tensor(
            problem.source_structure,
            problem.target_structure,
            problem.source,
            problem.target,
            plan,
        );
        Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
            2.0 * problem.structure_weight * tensor[(i, j)]
                + (1.0 - problem.structure_weight)
                    * problem.feature_cost.map_or(0.0, |cost| cost[(i, j)])
        })
    } else {
        Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
            let cross = (0..plan.nrows())
                .map(|k| {
                    (0..plan.ncols())
                        .map(|l| {
                            problem.source_structure[(i, k)]
                                * plan[(k, l)]
                                * problem.target_structure[(j, l)]
                        })
                        .sum::<f64>()
                })
                .sum::<f64>();
            -4.0 * problem.structure_weight * cross
                + (1.0 - problem.structure_weight)
                    * problem.feature_cost.map_or(0.0, |cost| cost[(i, j)])
        })
    }
}

fn bapg_project_rows(
    plan: &Mat<f64>,
    gradient: &Mat<f64>,
    weights: &[f64],
    regularization: f64,
) -> Result<Mat<f64>> {
    let mut output = Mat::<f64>::zeros(plan.nrows(), plan.ncols());
    for i in 0..plan.nrows() {
        if weights[i] == 0.0 {
            continue;
        }
        let maximum = (0..plan.ncols())
            .map(|j| {
                if plan[(i, j)] > 0.0 {
                    plan[(i, j)].ln() - gradient[(i, j)] / regularization
                } else {
                    f64::NEG_INFINITY
                }
            })
            .fold(f64::NEG_INFINITY, f64::max);
        if !maximum.is_finite() {
            return Err(Error::DidNotConverge {
                algorithm: "BAPG row projection",
                iterations: 0,
                residual: f64::INFINITY,
            });
        }
        let normalizer = (0..plan.ncols())
            .map(|j| {
                if plan[(i, j)] > 0.0 {
                    (plan[(i, j)].ln() - gradient[(i, j)] / regularization - maximum).exp()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        for j in 0..plan.ncols() {
            output[(i, j)] = if plan[(i, j)] > 0.0 {
                weights[i] * (plan[(i, j)].ln() - gradient[(i, j)] / regularization - maximum).exp()
                    / normalizer
            } else {
                0.0
            };
        }
    }
    Ok(output)
}

fn bapg_project_columns(
    plan: &Mat<f64>,
    gradient: &Mat<f64>,
    weights: &[f64],
    regularization: f64,
) -> Result<Mat<f64>> {
    let mut output = Mat::<f64>::zeros(plan.nrows(), plan.ncols());
    for j in 0..plan.ncols() {
        if weights[j] == 0.0 {
            continue;
        }
        let maximum = (0..plan.nrows())
            .map(|i| {
                if plan[(i, j)] > 0.0 {
                    plan[(i, j)].ln() - gradient[(i, j)] / regularization
                } else {
                    f64::NEG_INFINITY
                }
            })
            .fold(f64::NEG_INFINITY, f64::max);
        if !maximum.is_finite() {
            return Err(Error::DidNotConverge {
                algorithm: "BAPG column projection",
                iterations: 0,
                residual: f64::INFINITY,
            });
        }
        let normalizer = (0..plan.nrows())
            .map(|i| {
                if plan[(i, j)] > 0.0 {
                    (plan[(i, j)].ln() - gradient[(i, j)] / regularization - maximum).exp()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        for i in 0..plan.nrows() {
            output[(i, j)] = if plan[(i, j)] > 0.0 {
                weights[j] * (plan[(i, j)].ln() - gradient[(i, j)] / regularization - maximum).exp()
                    / normalizer
            } else {
                0.0
            };
        }
    }
    Ok(output)
}

fn bapg_solve(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    feature_cost: Option<&Mat<f64>>,
    structure_weight: f64,
    options: BapgOptions,
) -> Result<GromovResult> {
    let mass = validate::balanced_distributions(source, target)?;
    validate_structure(source_structure, source, "BAPG source structure")?;
    validate_structure(target_structure, target, "BAPG target structure")?;
    if let Some(cost) = feature_cost {
        validate::cost_matrix(cost, source.len(), target.len())?;
    }
    if !structure_weight.is_finite() || !(0.0..=1.0).contains(&structure_weight) {
        return Err(Error::InvalidParameter {
            name: "structure_weight",
            requirement: "finite and between zero and one",
        });
    }
    validate::finite_positive(
        options.regularization,
        "BAPG regularization",
        "finite and strictly positive",
    )?;
    validate::finite_positive(
        options.tolerance,
        "BAPG tolerance",
        "finite and strictly positive",
    )?;
    if options.max_iterations == 0 {
        return Err(Error::InvalidParameter {
            name: "BAPG max_iterations",
            requirement: "positive",
        });
    }

    let mut plan = initial_plan(source, target, mass);
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    let gradient_problem = BapgGradientProblem {
        source_structure,
        target_structure,
        source,
        target,
        feature_cost,
        structure_weight,
    };
    for iteration in 1..=options.max_iterations {
        let previous = plan.clone();
        let row_gradient = bapg_gradient(&gradient_problem, options.marginal_loss, &plan);
        plan = bapg_project_rows(&plan, &row_gradient, source, options.regularization)?;
        let column_gradient = bapg_gradient(&gradient_problem, options.marginal_loss, &plan);
        plan = bapg_project_columns(&plan, &column_gradient, target, options.regularization)?;
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
    let (structure_value, feature_value, value) = combined_objective(
        source_structure,
        target_structure,
        source,
        target,
        feature_cost,
        structure_weight,
        &plan,
    );
    Ok(GromovResult {
        plan,
        structure_value,
        feature_value,
        value,
        iterations,
        residual,
        status,
    })
}

fn partial_structure_objective(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    plan: &Mat<f64>,
) -> f64 {
    let mut value = 0.0;
    for i in 0..plan.nrows() {
        for j in 0..plan.ncols() {
            if plan[(i, j)] == 0.0 {
                continue;
            }
            for k in 0..plan.nrows() {
                for l in 0..plan.ncols() {
                    let difference = source_structure[(i, k)] - target_structure[(j, l)];
                    value += difference.powi(2) * plan[(i, j)] * plan[(k, l)];
                }
            }
        }
    }
    value.max(0.0)
}

fn partial_combined_objective(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    feature_cost: Option<&Mat<f64>>,
    structure_weight: f64,
    plan: &Mat<f64>,
) -> (f64, f64, f64) {
    let structure = partial_structure_objective(source_structure, target_structure, plan);
    let feature = feature_objective(feature_cost, plan);
    (
        structure,
        feature,
        structure_weight * structure + (1.0 - structure_weight) * feature,
    )
}

fn partial_gromov_gradient(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    feature_cost: Option<&Mat<f64>>,
    structure_weight: f64,
    plan: &Mat<f64>,
) -> Mat<f64> {
    Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
        let structure = (0..plan.nrows())
            .map(|k| {
                (0..plan.ncols())
                    .map(|l| {
                        let difference = source_structure[(i, k)] - target_structure[(j, l)];
                        difference.powi(2) * plan[(k, l)]
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();
        2.0 * structure_weight * structure
            + (1.0 - structure_weight) * feature_cost.map_or(0.0, |cost| cost[(i, j)])
    })
}

/// Compute exact or entropic squared-loss partial Gromov-Wasserstein transport.
pub fn partial_gromov_wasserstein(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    options: PartialGromovOptions,
) -> Result<GromovResult> {
    partial_gromov_solve(
        source_structure,
        target_structure,
        source,
        target,
        None,
        1.0,
        options,
    )
}

/// Compute exact or entropic partial fused Gromov-Wasserstein transport.
pub fn partial_fused_gromov_wasserstein(
    feature_cost: &Mat<f64>,
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    structure_weight: f64,
    options: PartialGromovOptions,
) -> Result<GromovResult> {
    partial_gromov_solve(
        source_structure,
        target_structure,
        source,
        target,
        Some(feature_cost),
        structure_weight,
        options,
    )
}

fn partial_gromov_solve(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    feature_cost: Option<&Mat<f64>>,
    structure_weight: f64,
    options: PartialGromovOptions,
) -> Result<GromovResult> {
    let source_mass = validate::distribution(source, "partial Gromov source")?;
    let target_mass = validate::distribution(target, "partial Gromov target")?;
    validate_structure(source_structure, source, "partial Gromov source structure")?;
    validate_structure(target_structure, target, "partial Gromov target structure")?;
    if let Some(cost) = feature_cost {
        validate::cost_matrix(cost, source.len(), target.len())?;
    }
    if !structure_weight.is_finite() || !(0.0..=1.0).contains(&structure_weight) {
        return Err(Error::InvalidParameter {
            name: "structure_weight",
            requirement: "finite and between zero and one",
        });
    }
    if !options.transported_mass.is_finite()
        || options.transported_mass <= 0.0
        || options.transported_mass > source_mass.min(target_mass) + 1e-12
    {
        return Err(Error::InvalidParameter {
            name: "transported_mass",
            requirement: "finite, positive, and no larger than either marginal mass",
        });
    }
    if options.max_iterations == 0 {
        return Err(Error::InvalidParameter {
            name: "max_iterations",
            requirement: "positive",
        });
    }
    validate::finite_positive(
        options.tolerance,
        "partial Gromov tolerance",
        "finite and strictly positive",
    )?;
    if let PartialGromovMethod::Entropic { regularization, .. } = options.method {
        validate::finite_positive(
            regularization,
            "partial Gromov regularization",
            "finite and strictly positive",
        )?;
    }

    let mut plan = Mat::<f64>::from_fn(source.len(), target.len(), |i, j| {
        options.transported_mass * source[i] * target[j] / (source_mass * target_mass)
    });
    let (_, _, mut objective) = partial_combined_objective(
        source_structure,
        target_structure,
        feature_cost,
        structure_weight,
        &plan,
    );
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        let gradient = partial_gromov_gradient(
            source_structure,
            target_structure,
            feature_cost,
            structure_weight,
            &plan,
        );
        let candidate = match options.method {
            PartialGromovMethod::Exact => {
                partial::partial_wasserstein(source, target, &gradient, options.transported_mass)?
                    .plan
            }
            PartialGromovMethod::Entropic {
                regularization,
                options: entropic_options,
            } => {
                partial::entropic_partial_wasserstein(
                    source,
                    target,
                    &gradient,
                    regularization,
                    options.transported_mass,
                    entropic_options,
                )?
                .plan
            }
        };
        let next = match options.method {
            PartialGromovMethod::Exact => {
                let direction = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
                    candidate[(i, j)] - plan[(i, j)]
                });
                let mut directional_derivative = 0.0;
                for i in 0..plan.nrows() {
                    for j in 0..plan.ncols() {
                        directional_derivative += gradient[(i, j)] * direction[(i, j)];
                    }
                }
                let endpoint = partial_combined_objective(
                    source_structure,
                    target_structure,
                    feature_cost,
                    structure_weight,
                    &candidate,
                )
                .2;
                let quadratic = endpoint - objective - directional_derivative;
                let step = if quadratic > 0.0 {
                    (-directional_derivative / (2.0 * quadratic)).clamp(0.0, 1.0)
                } else if endpoint < objective {
                    1.0
                } else {
                    0.0
                };
                Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
                    plan[(i, j)] + step * direction[(i, j)]
                })
            }
            PartialGromovMethod::Entropic { .. } => candidate,
        };
        let next_objective = partial_combined_objective(
            source_structure,
            target_structure,
            feature_cost,
            structure_weight,
            &next,
        )
        .2;
        residual = (objective - next_objective).abs();
        plan = next;
        objective = next_objective;
        iterations = iteration;
        if residual <= options.tolerance {
            status = SolverStatus::Converged;
            break;
        }
    }
    let (structure_value, feature_value, value) = partial_combined_objective(
        source_structure,
        target_structure,
        feature_cost,
        structure_weight,
        &plan,
    );
    Ok(GromovResult {
        plan,
        structure_value,
        feature_value,
        value,
        iterations,
        residual,
        status,
    })
}

fn solve_fused(
    source_structure: &Mat<f64>,
    target_structure: &Mat<f64>,
    source: &[f64],
    target: &[f64],
    feature_cost: Option<&Mat<f64>>,
    structure_weight: f64,
    options: GromovOptions,
) -> Result<GromovResult> {
    let mass = validate::balanced_distributions(source, target)?;
    validate_structure(source_structure, source, "source structure")?;
    validate_structure(target_structure, target, "target structure")?;
    validate_options(options)?;
    if !structure_weight.is_finite() || !(0.0..=1.0).contains(&structure_weight) {
        return Err(Error::InvalidParameter {
            name: "structure_weight",
            requirement: "finite and between zero and one",
        });
    }
    if let Some(cost) = feature_cost {
        validate::cost_matrix(cost, source.len(), target.len())?;
    }
    let mut plan = initial_plan(source, target, mass);
    let (_, _, mut objective) = combined_objective(
        source_structure,
        target_structure,
        source,
        target,
        feature_cost,
        structure_weight,
        &plan,
    );
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        let tensor = square_loss_tensor(source_structure, target_structure, source, target, &plan);
        let gradient = Mat::<f64>::from_fn(source.len(), target.len(), |i, j| {
            2.0 * structure_weight * tensor[(i, j)]
                + (1.0 - structure_weight) * feature_cost.map_or(0.0, |cost| cost[(i, j)])
        });
        let candidate = match options.method {
            GromovMethod::Exact => exact::emd(source, target, &gradient)?.plan,
            GromovMethod::Entropic { regularization } => {
                sinkhorn::sinkhorn(source, target, &gradient, regularization)?.plan
            }
        };
        let next = match options.method {
            GromovMethod::Exact => {
                let direction = Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
                    candidate[(i, j)] - plan[(i, j)]
                });
                let directional_derivative = (0..plan.nrows())
                    .map(|i| {
                        (0..plan.ncols())
                            .map(|j| gradient[(i, j)] * direction[(i, j)])
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                let (_, _, endpoint) = combined_objective(
                    source_structure,
                    target_structure,
                    source,
                    target,
                    feature_cost,
                    structure_weight,
                    &candidate,
                );
                let quadratic = endpoint - objective - directional_derivative;
                let step = if quadratic > 0.0 {
                    (-directional_derivative / (2.0 * quadratic)).clamp(0.0, 1.0)
                } else if endpoint < objective {
                    1.0
                } else {
                    0.0
                };
                Mat::<f64>::from_fn(plan.nrows(), plan.ncols(), |i, j| {
                    plan[(i, j)] + step * direction[(i, j)]
                })
            }
            GromovMethod::Entropic { .. } => candidate,
        };
        let (_, _, next_objective) = combined_objective(
            source_structure,
            target_structure,
            source,
            target,
            feature_cost,
            structure_weight,
            &next,
        );
        residual = (objective - next_objective).abs();
        plan = next;
        objective = next_objective;
        iterations = iteration;
        if residual <= options.tolerance {
            status = SolverStatus::Converged;
            break;
        }
    }
    let (structure_value, feature_value, value) = combined_objective(
        source_structure,
        target_structure,
        source,
        target,
        feature_cost,
        structure_weight,
        &plan,
    );
    Ok(GromovResult {
        plan,
        structure_value,
        feature_value,
        value,
        iterations,
        residual,
        status,
    })
}

/// Compute a squared-loss fixed-support Gromov-Wasserstein barycenter.
pub fn gromov_barycenter(
    structures: &[Mat<f64>],
    distributions: Option<&[Vec<f64>]>,
    barycenter_weights: &[f64],
    mixture: Option<&[f64]>,
    initial_structure: &Mat<f64>,
    options: GromovBarycenterOptions,
) -> Result<GromovBarycenter> {
    if structures.is_empty() {
        return Err(Error::EmptyInput {
            name: "Gromov barycenter structures",
        });
    }
    let barycenter_mass = validate::distribution(barycenter_weights, "Gromov barycenter weights")?;
    if barycenter_weights.iter().any(|&weight| weight <= 0.0) {
        return Err(Error::InvalidParameter {
            name: "Gromov barycenter weights",
            requirement: "strictly positive",
        });
    }
    validate_structure(
        initial_structure,
        barycenter_weights,
        "initial barycenter structure",
    )?;
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
    let distribution_storage;
    let distributions = match distributions {
        Some(distributions) => distributions,
        None => {
            distribution_storage = structures
                .iter()
                .map(|structure| validate::uniform(structure.nrows()))
                .collect::<Result<Vec<_>>>()?;
            &distribution_storage
        }
    };
    if distributions.len() != structures.len() {
        return Err(Error::ShapeMismatch {
            context: "Gromov barycenter structures and distributions",
            left: (structures.len(), 1),
            right: (distributions.len(), 1),
        });
    }
    for (structure, distribution) in structures.iter().zip(distributions) {
        validate_structure(structure, distribution, "Gromov barycenter input structure")?;
        let mass = validate::distribution(distribution, "Gromov barycenter distribution")?;
        if (mass - barycenter_mass).abs() > 1e-10 * mass.max(barycenter_mass).max(1.0) {
            return Err(Error::MassMismatch {
                source: barycenter_mass,
                target: mass,
            });
        }
    }
    let mixture_storage;
    let mixture = match mixture {
        Some(mixture) => mixture,
        None => {
            mixture_storage = validate::uniform(structures.len())?;
            &mixture_storage
        }
    };
    if mixture.len() != structures.len() {
        return Err(Error::ShapeMismatch {
            context: "Gromov barycenter mixture",
            left: (mixture.len(), 1),
            right: (structures.len(), 1),
        });
    }
    let mixture_mass = validate::distribution(mixture, "Gromov barycenter mixture")?;
    let mixture = mixture
        .iter()
        .map(|weight| weight / mixture_mass)
        .collect::<Vec<_>>();

    let mut structure = initial_structure.clone();
    let mut plans = Vec::new();
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        plans = structures
            .iter()
            .zip(distributions)
            .map(|(target_structure, target_weights)| {
                gromov_wasserstein(
                    &structure,
                    target_structure,
                    barycenter_weights,
                    target_weights,
                    options.transport,
                )
                .map(|result| result.plan)
            })
            .collect::<Result<Vec<_>>>()?;
        let next = Mat::<f64>::from_fn(structure.nrows(), structure.ncols(), |a, b| {
            structures
                .iter()
                .zip(&plans)
                .zip(&mixture)
                .map(|((input, plan), &coefficient)| {
                    let transported = (0..input.nrows())
                        .map(|i| {
                            (0..input.ncols())
                                .map(|j| plan[(a, i)] * input[(i, j)] * plan[(b, j)])
                                .sum::<f64>()
                        })
                        .sum::<f64>();
                    coefficient * transported
                })
                .sum::<f64>()
                / (barycenter_weights[a] * barycenter_weights[b])
        });
        residual = 0.0;
        for i in 0..structure.nrows() {
            for j in 0..structure.ncols() {
                residual += (next[(i, j)] - structure[(i, j)]).powi(2);
            }
        }
        residual = residual.sqrt();
        structure = next;
        iterations = iteration;
        if residual <= options.tolerance {
            status = SolverStatus::Converged;
            break;
        }
    }
    Ok(GromovBarycenter {
        structure,
        plans,
        iterations,
        residual,
        status,
    })
}

/// Compute a squared-feature-loss fixed-support fused Gromov barycenter.
pub fn fused_gromov_barycenter(
    problem: FusedGromovBarycenterProblem<'_>,
    options: FusedGromovBarycenterOptions,
) -> Result<FusedGromovBarycenter> {
    if problem.features.is_empty() || problem.structures.is_empty() {
        return Err(Error::EmptyInput {
            name: "fused Gromov barycenter inputs",
        });
    }
    if problem.features.len() != problem.structures.len() {
        return Err(Error::ShapeMismatch {
            context: "fused Gromov features and structures",
            left: (problem.features.len(), 1),
            right: (problem.structures.len(), 1),
        });
    }
    if !options.structure_weight.is_finite() || !(0.0..=1.0).contains(&options.structure_weight) {
        return Err(Error::InvalidParameter {
            name: "structure_weight",
            requirement: "finite and between zero and one",
        });
    }
    if !options.update_features && !options.update_structure {
        return Err(Error::InvalidParameter {
            name: "FGW barycenter updates",
            requirement: "enabled for features, structure, or both",
        });
    }
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
    let barycenter_mass =
        validate::distribution(problem.barycenter_weights, "FGW barycenter weights")?;
    if problem
        .barycenter_weights
        .iter()
        .any(|&weight| weight <= 0.0)
    {
        return Err(Error::InvalidParameter {
            name: "FGW barycenter weights",
            requirement: "strictly positive",
        });
    }
    validate_structure(
        problem.initial_structure,
        problem.barycenter_weights,
        "initial FGW barycenter structure",
    )?;
    validate::samples(problem.initial_features, "initial FGW barycenter features")?;
    if problem.initial_features.nrows() != problem.barycenter_weights.len() {
        return Err(Error::ShapeMismatch {
            context: "initial FGW features and weights",
            left: (
                problem.initial_features.nrows(),
                problem.initial_features.ncols(),
            ),
            right: (
                problem.barycenter_weights.len(),
                problem.initial_features.ncols(),
            ),
        });
    }
    let dimensions = problem.initial_features.ncols();
    let distribution_storage;
    let distributions = match problem.distributions {
        Some(distributions) => distributions,
        None => {
            distribution_storage = problem
                .structures
                .iter()
                .map(|structure| validate::uniform(structure.nrows()))
                .collect::<Result<Vec<_>>>()?;
            &distribution_storage
        }
    };
    if distributions.len() != problem.structures.len() {
        return Err(Error::ShapeMismatch {
            context: "FGW structures and distributions",
            left: (problem.structures.len(), 1),
            right: (distributions.len(), 1),
        });
    }
    for ((features, structure), distribution) in problem
        .features
        .iter()
        .zip(problem.structures)
        .zip(distributions)
    {
        validate::samples(features, "FGW barycenter input features")?;
        validate_structure(structure, distribution, "FGW barycenter input structure")?;
        if features.nrows() != structure.nrows() || features.ncols() != dimensions {
            return Err(Error::ShapeMismatch {
                context: "FGW input features and structure",
                left: (features.nrows(), features.ncols()),
                right: (structure.nrows(), dimensions),
            });
        }
        let mass = validate::distribution(distribution, "FGW barycenter distribution")?;
        if (mass - barycenter_mass).abs() > 1e-10 * mass.max(barycenter_mass).max(1.0) {
            return Err(Error::MassMismatch {
                source: barycenter_mass,
                target: mass,
            });
        }
    }
    let mixture_storage;
    let mixture = match problem.mixture {
        Some(mixture) => mixture,
        None => {
            mixture_storage = validate::uniform(problem.structures.len())?;
            &mixture_storage
        }
    };
    if mixture.len() != problem.structures.len() {
        return Err(Error::ShapeMismatch {
            context: "FGW barycenter mixture",
            left: (mixture.len(), 1),
            right: (problem.structures.len(), 1),
        });
    }
    let mixture_mass = validate::distribution(mixture, "FGW barycenter mixture")?;
    let mixture = mixture
        .iter()
        .map(|weight| weight / mixture_mass)
        .collect::<Vec<_>>();

    let mut features = problem.initial_features.clone();
    let mut structure = problem.initial_structure.clone();
    let mut plans = Vec::new();
    let mut iterations = 0;
    let mut residual = f64::INFINITY;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        plans = problem
            .features
            .iter()
            .zip(problem.structures)
            .zip(distributions)
            .map(|((target_features, target_structure), target_weights)| {
                let feature_cost =
                    metrics::pairwise(&features, target_features, Metric::SquaredEuclidean)?;
                fused_gromov_wasserstein(
                    &feature_cost,
                    &structure,
                    target_structure,
                    problem.barycenter_weights,
                    target_weights,
                    options.structure_weight,
                    options.transport,
                )
                .map(|result| result.plan)
            })
            .collect::<Result<Vec<_>>>()?;

        let next_features = if options.update_features {
            Mat::<f64>::from_fn(features.nrows(), features.ncols(), |i, coordinate| {
                problem
                    .features
                    .iter()
                    .zip(&plans)
                    .zip(&mixture)
                    .map(|((input, plan), &coefficient)| {
                        coefficient
                            * (0..input.nrows())
                                .map(|j| plan[(i, j)] * input[(j, coordinate)])
                                .sum::<f64>()
                    })
                    .sum::<f64>()
                    / problem.barycenter_weights[i]
            })
        } else {
            features.clone()
        };
        let next_structure = if options.update_structure {
            Mat::<f64>::from_fn(structure.nrows(), structure.ncols(), |a, b| {
                problem
                    .structures
                    .iter()
                    .zip(&plans)
                    .zip(&mixture)
                    .map(|((input, plan), &coefficient)| {
                        let transported = (0..input.nrows())
                            .map(|i| {
                                (0..input.ncols())
                                    .map(|j| plan[(a, i)] * input[(i, j)] * plan[(b, j)])
                                    .sum::<f64>()
                            })
                            .sum::<f64>();
                        coefficient * transported
                    })
                    .sum::<f64>()
                    / (problem.barycenter_weights[a] * problem.barycenter_weights[b])
            })
        } else {
            structure.clone()
        };
        residual = 0.0;
        for j in 0..features.ncols() {
            for i in 0..features.nrows() {
                residual += (next_features[(i, j)] - features[(i, j)]).powi(2);
            }
        }
        for j in 0..structure.ncols() {
            for i in 0..structure.nrows() {
                residual += (next_structure[(i, j)] - structure[(i, j)]).powi(2);
            }
        }
        residual = residual.sqrt();
        features = next_features;
        structure = next_structure;
        iterations = iteration;
        if residual <= options.tolerance {
            status = SolverStatus::Converged;
            break;
        }
    }
    Ok(FusedGromovBarycenter {
        features,
        structure,
        plans,
        iterations,
        residual,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isomorphic_two_point_spaces_have_zero_gw() {
        let structure = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);
        let weights = [0.5, 0.5];
        let result = gromov_wasserstein(
            &structure,
            &structure,
            &weights,
            &weights,
            GromovOptions::default(),
        )
        .unwrap();
        assert!(result.value < 1e-10, "value={}", result.value);
        for i in 0..2 {
            let row = (0..2).map(|j| result.plan[(i, j)]).sum::<f64>();
            assert!((row - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn feature_only_fused_problem_matches_linear_cost() {
        let structure = Mat::<f64>::zeros(2, 2);
        let feature = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);
        let weights = [0.5, 0.5];
        let result = fused_gromov_wasserstein(
            &feature,
            &structure,
            &structure,
            &weights,
            &weights,
            0.0,
            GromovOptions::default(),
        )
        .unwrap();
        assert!(result.value < 1e-12);
        assert!((result.plan[(0, 0)] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn fixed_support_barycenter_matches_reference_update() {
        let first = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);
        let second = Mat::<f64>::from_fn(3, 3, |i, j| {
            [[0.0, 2.0, 3.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]][i][j]
        });
        let initial = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 1.5 });
        let result = gromov_barycenter(
            &[first, second],
            Some(&[vec![0.5, 0.5], vec![0.2, 0.5, 0.3]]),
            &[0.4, 0.6],
            Some(&[0.25, 0.75]),
            &initial,
            GromovBarycenterOptions {
                max_iterations: 1_000,
                tolerance: 1e-12,
                ..GromovBarycenterOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, SolverStatus::Converged);
        let expected = [[1.125, 4.0 / 3.0], [4.0 / 3.0, 5.0 / 18.0]];
        for (i, row) in expected.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                assert!((result.structure[(i, j)] - value).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fused_barycenter_updates_features_and_structure() {
        let first_features = Mat::<f64>::from_fn(2, 1, |i, _| [0.0, 2.0][i]);
        let first_structure = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);
        let second_features = Mat::<f64>::from_fn(3, 1, |i, _| [1.0, 3.0, 5.0][i]);
        let second_structure = Mat::<f64>::from_fn(3, 3, |i, j| {
            [[0.0, 2.0, 3.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]][i][j]
        });
        let initial_features = Mat::<f64>::from_fn(2, 1, |i, _| [0.5, 3.0][i]);
        let initial_structure = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 1.5 });
        let result = fused_gromov_barycenter(
            FusedGromovBarycenterProblem {
                features: &[first_features, second_features],
                structures: &[first_structure, second_structure],
                distributions: Some(&[vec![0.5, 0.5], vec![0.2, 0.5, 0.3]]),
                barycenter_weights: &[0.4, 0.6],
                mixture: Some(&[0.25, 0.75]),
                initial_features: &initial_features,
                initial_structure: &initial_structure,
            },
            FusedGromovBarycenterOptions {
                structure_weight: 0.6,
                max_iterations: 1_000,
                tolerance: 1e-12,
                ..FusedGromovBarycenterOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, SolverStatus::Converged);
        assert!((result.features[(0, 0)] - 1.5).abs() < 1e-12);
        assert!((result.features[(1, 0)] - 41.0 / 12.0).abs() < 1e-12);
        assert!((result.structure[(0, 0)] - 0.75).abs() < 1e-12);
        assert!((result.structure[(1, 1)] - 4.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn bapg_fused_solver_matches_reference_fixed_point() {
        let source_structure = Mat::<f64>::from_fn(3, 3, |i, j| {
            [[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]][i][j]
        });
        let target_structure = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 2.0 });
        let feature_cost =
            Mat::<f64>::from_fn(3, 2, |i, j| [[0.0, 1.0], [1.0, 0.0], [2.0, 0.5]][i][j]);
        let result = bapg_fused_gromov_wasserstein(
            &feature_cost,
            &source_structure,
            &target_structure,
            &[0.2, 0.5, 0.3],
            &[0.6, 0.4],
            0.6,
            BapgOptions {
                regularization: 0.5,
                max_iterations: 10_000,
                tolerance: 1e-12,
                marginal_loss: true,
            },
        )
        .unwrap();
        assert_eq!(result.status, SolverStatus::Converged);
        assert!((result.value - 0.4544104167895048).abs() < 1e-10);
        assert!((result.plan[(0, 0)] - 0.10450772839556065).abs() < 1e-10);
        assert!((result.plan[(1, 0)] - 0.49549227160443937).abs() < 1e-10);
        assert!((result.plan[(2, 1)] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn exact_partial_fused_solver_moves_requested_mass() {
        let source_structure = Mat::<f64>::from_fn(3, 3, |i, j| {
            [[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]][i][j]
        });
        let target_structure = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 2.0 });
        let feature_cost =
            Mat::<f64>::from_fn(3, 2, |i, j| [[0.0, 1.0], [1.0, 0.0], [2.0, 0.5]][i][j]);
        let result = partial_fused_gromov_wasserstein(
            &feature_cost,
            &source_structure,
            &target_structure,
            &[0.2, 0.5, 0.3],
            &[0.6, 0.4],
            0.6,
            PartialGromovOptions {
                transported_mass: 0.7,
                tolerance: 1e-12,
                ..PartialGromovOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, SolverStatus::Converged);
        assert!((result.value - 0.352).abs() < 1e-10);
        let mass = (0..result.plan.nrows())
            .map(|i| {
                (0..result.plan.ncols())
                    .map(|j| result.plan[(i, j)])
                    .sum::<f64>()
            })
            .sum::<f64>();
        assert!((mass - 0.7).abs() < 1e-12);
    }
}
