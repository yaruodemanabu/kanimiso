//! Wasserstein barycenters on fixed and free supports.

use crate::error::{Error, Result};
use crate::exact;
use crate::metrics::{self, Metric};
use crate::result::{BarycenterResult, SolverStatus};
use crate::validate;
use faer::Mat;

/// Stopping options for fixed-support entropic barycenters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarycenterOptions {
    /// Maximum Bregman projection iterations.
    pub max_iterations: usize,
    /// Maximum coordinate change required for convergence.
    pub tolerance: f64,
}

impl Default for BarycenterOptions {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            tolerance: 1e-10,
        }
    }
}

fn mixture_weights(count: usize, supplied: Option<&[f64]>) -> Result<Vec<f64>> {
    let mut result = match supplied {
        Some(weights) => {
            if weights.len() != count {
                return Err(Error::ShapeMismatch {
                    context: "barycenter distributions and mixture weights",
                    left: (count, 1),
                    right: (weights.len(), 1),
                });
            }
            weights.to_vec()
        }
        None => validate::uniform(count)?,
    };
    let sum = validate::distribution(&result, "barycenter mixture weights")?;
    for value in &mut result {
        *value /= sum;
    }
    Ok(result)
}

fn validate_histogram_columns(distributions: &Mat<f64>) -> Result<f64> {
    if distributions.nrows() == 0 || distributions.ncols() == 0 {
        return Err(Error::EmptyInput {
            name: "barycenter distributions",
        });
    }
    let mut reference_mass: Option<f64> = None;
    for column in 0..distributions.ncols() {
        let weights: Vec<_> = (0..distributions.nrows())
            .map(|row| distributions[(row, column)])
            .collect();
        let mass = validate::distribution(&weights, "barycenter distribution")?;
        if let Some(reference) = reference_mass {
            if (mass - reference).abs() > 1e-9 * mass.abs().max(reference).max(1.0) {
                return Err(Error::MassMismatch {
                    source: reference,
                    target: mass,
                });
            }
        } else {
            reference_mass = Some(mass);
        }
    }
    Ok(reference_mass.unwrap_or(1.0))
}

/// Entropic Wasserstein barycenter for histograms sharing one support.
///
/// Columns of `distributions` are input histograms and `cost` is the
/// support-to-support ground cost.
pub fn barycenter(
    distributions: &Mat<f64>,
    cost: &Mat<f64>,
    regularization: f64,
    mixture: Option<&[f64]>,
) -> Result<BarycenterResult> {
    barycenter_with_options(
        distributions,
        cost,
        regularization,
        mixture,
        BarycenterOptions::default(),
    )
}

/// Fixed-support entropic barycenter with explicit stopping options.
pub fn barycenter_with_options(
    distributions: &Mat<f64>,
    cost: &Mat<f64>,
    regularization: f64,
    mixture: Option<&[f64]>,
    options: BarycenterOptions,
) -> Result<BarycenterResult> {
    let mass = validate_histogram_columns(distributions)?;
    validate::cost_matrix(cost, distributions.nrows(), distributions.nrows())?;
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
    let mixture = mixture_weights(distributions.ncols(), mixture)?;
    let n = distributions.nrows();
    let count = distributions.ncols();
    let minimum_cost = (0..cost.nrows())
        .flat_map(|i| (0..cost.ncols()).map(move |j| cost[(i, j)]))
        .fold(f64::INFINITY, f64::min);
    let kernel = Mat::<f64>::from_fn(n, n, |i, j| {
        (-(cost[(i, j)] - minimum_cost) / regularization).exp()
    });
    let mut right_scaling = Mat::<f64>::from_fn(n, count, |_, _| 1.0);
    let mut left_scaling = Mat::<f64>::zeros(n, count);
    let mut current = vec![mass / n as f64; n];
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        let previous = current.clone();
        for distribution in 0..count {
            for i in 0..n {
                let denominator = (0..n)
                    .map(|j| kernel[(i, j)] * right_scaling[(j, distribution)])
                    .sum::<f64>();
                if denominator <= 0.0 || !denominator.is_finite() {
                    return Err(Error::DidNotConverge {
                        algorithm: "entropic barycenter",
                        iterations: iteration,
                        residual: f64::INFINITY,
                    });
                }
                left_scaling[(i, distribution)] = distributions[(i, distribution)] / denominator;
            }
        }
        let mut transported = Mat::<f64>::zeros(n, count);
        for distribution in 0..count {
            for j in 0..n {
                transported[(j, distribution)] = (0..n)
                    .map(|i| kernel[(i, j)] * left_scaling[(i, distribution)])
                    .sum::<f64>();
            }
        }
        for j in 0..n {
            let logarithm = (0..count)
                .map(|distribution| {
                    mixture[distribution]
                        * transported[(j, distribution)].max(f64::MIN_POSITIVE).ln()
                })
                .sum::<f64>();
            current[j] = logarithm.exp();
        }
        let current_mass = current.iter().sum::<f64>();
        if current_mass <= 0.0 || !current_mass.is_finite() {
            return Err(Error::DidNotConverge {
                algorithm: "entropic barycenter",
                iterations: iteration,
                residual: f64::INFINITY,
            });
        }
        for value in &mut current {
            *value *= mass / current_mass;
        }
        for distribution in 0..count {
            for j in 0..n {
                right_scaling[(j, distribution)] =
                    current[j] / transported[(j, distribution)].max(f64::MIN_POSITIVE);
            }
        }
        residual = current
            .iter()
            .zip(&previous)
            .map(|(&next, &old)| (next - old).abs())
            .fold(0.0, f64::max);
        iterations = iteration;
        if residual <= options.tolerance {
            status = SolverStatus::Converged;
            break;
        }
    }
    Ok(BarycenterResult {
        weights: current,
        iterations,
        residual,
        status,
    })
}

/// Options for a free-support Wasserstein barycenter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FreeSupportOptions {
    /// Maximum block-coordinate iterations.
    pub max_iterations: usize,
    /// Frobenius support displacement required for convergence.
    pub tolerance: f64,
}

impl Default for FreeSupportOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-7,
        }
    }
}

/// Result of free-support barycenter optimization.
#[derive(Clone, Debug)]
pub struct FreeSupportBarycenter {
    /// Learned barycenter atom coordinates.
    pub support: Mat<f64>,
    /// Number of block-coordinate iterations.
    pub iterations: usize,
    /// Last Frobenius displacement of the support.
    pub residual: f64,
    /// Solver termination status.
    pub status: SolverStatus,
}

/// Compute a non-regularized free-support Wasserstein barycenter.
///
/// `initial_support` contains the barycenter atoms.  Its fixed atom weights
/// are supplied separately and remain unchanged.
pub fn free_support_barycenter(
    supports: &[Mat<f64>],
    distributions: &[Vec<f64>],
    barycenter_weights: &[f64],
    initial_support: &Mat<f64>,
    mixture: Option<&[f64]>,
    options: FreeSupportOptions,
) -> Result<FreeSupportBarycenter> {
    if supports.is_empty() {
        return Err(Error::EmptyInput {
            name: "free-support input distributions",
        });
    }
    if supports.len() != distributions.len() {
        return Err(Error::ShapeMismatch {
            context: "free-support locations and distributions",
            left: (supports.len(), 1),
            right: (distributions.len(), 1),
        });
    }
    validate::samples(initial_support, "initial barycenter support")?;
    if barycenter_weights.len() != initial_support.nrows() {
        return Err(Error::ShapeMismatch {
            context: "barycenter support and weights",
            left: (initial_support.nrows(), 1),
            right: (barycenter_weights.len(), 1),
        });
    }
    validate::distribution(barycenter_weights, "barycenter weights")?;
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
    let dimensions = initial_support.ncols();
    let mixture = mixture_weights(supports.len(), mixture)?;
    for (support, weights) in supports.iter().zip(distributions) {
        validate::samples(support, "free-support samples")?;
        if support.ncols() != dimensions || support.nrows() != weights.len() {
            return Err(Error::ShapeMismatch {
                context: "free-support dimensions and weights",
                left: (support.nrows(), support.ncols()),
                right: (weights.len(), dimensions),
            });
        }
        validate::balanced_distributions(barycenter_weights, weights)?;
    }

    let mut barycenter = initial_support.clone();
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        let previous = barycenter.clone();
        let mut next = Mat::<f64>::zeros(barycenter.nrows(), dimensions);
        for (distribution_index, (support, weights)) in
            supports.iter().zip(distributions).enumerate()
        {
            let cost = metrics::pairwise(&barycenter, support, Metric::SquaredEuclidean)?;
            let plan = exact::emd(barycenter_weights, weights, &cost)?;
            for i in 0..barycenter.nrows() {
                for coordinate in 0..dimensions {
                    let transported = (0..support.nrows())
                        .map(|j| plan.plan[(i, j)] * support[(j, coordinate)])
                        .sum::<f64>();
                    next[(i, coordinate)] +=
                        mixture[distribution_index] * transported / barycenter_weights[i];
                }
            }
        }
        residual = 0.0;
        for i in 0..next.nrows() {
            for j in 0..next.ncols() {
                residual += (next[(i, j)] - previous[(i, j)]).powi(2);
            }
        }
        residual = residual.sqrt();
        barycenter = next;
        iterations = iteration;
        if residual <= options.tolerance {
            status = SolverStatus::Converged;
            break;
        }
    }
    Ok(FreeSupportBarycenter {
        support: barycenter,
        iterations,
        residual,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_regularization_barycenter_preserves_identical_distribution() {
        let distributions = Mat::<f64>::from_fn(3, 2, |i, _| [0.2, 0.3, 0.5][i]);
        let cost = Mat::<f64>::from_fn(3, 3, |i, j| i.abs_diff(j).pow(2) as f64);
        let result = barycenter(&distributions, &cost, 0.01, None).unwrap();
        for (actual, expected) in result.weights.iter().zip([0.2, 0.3, 0.5]) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn free_support_mean_of_two_diracs() {
        let supports = vec![
            Mat::<f64>::from_fn(1, 1, |_, _| 0.0),
            Mat::<f64>::from_fn(1, 1, |_, _| 2.0),
        ];
        let distributions = vec![vec![1.0], vec![1.0]];
        let initial = Mat::<f64>::from_fn(1, 1, |_, _| 0.0);
        let result = free_support_barycenter(
            &supports,
            &distributions,
            &[1.0],
            &initial,
            None,
            FreeSupportOptions::default(),
        )
        .unwrap();
        assert!((result.support[(0, 0)] - 1.0).abs() < 1e-12);
    }
}
