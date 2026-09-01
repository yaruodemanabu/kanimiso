//! Low-rank cost and kernel factorizations for scalable transport.

use crate::error::{Error, Result};
use crate::result::SolverStatus;
use crate::validate;
use faer::Mat;

/// A rectangular matrix represented as `left * right���`.
#[derive(Clone, Debug)]
pub struct LowRankFactors {
    /// Left factor with one row per source sample.
    pub left: Mat<f64>,
    /// Right factor with one row per target sample.
    pub right: Mat<f64>,
}

impl LowRankFactors {
    /// Materialize the represented dense matrix.
    pub fn dense(&self) -> Mat<f64> {
        Mat::<f64>::from_fn(self.left.nrows(), self.right.nrows(), |i, j| {
            (0..self.left.ncols())
                .map(|rank| self.left[(i, rank)] * self.right[(j, rank)])
                .sum()
        })
    }
}

/// Stopping options for Sinkhorn scaling of a low-rank kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LowRankKernelOptions {
    /// Maximum alternating scaling iterations.
    pub max_iterations: usize,
    /// L2 target-marginal error required for convergence.
    pub tolerance: f64,
    /// Evaluate the marginal error every this many iterations.
    pub check_interval: usize,
}

impl Default for LowRankKernelOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: 1e-9,
            check_interval: 10,
        }
    }
}

/// Scaled factors and diagnostics for a low-rank kernel transport plan.
#[derive(Clone, Debug)]
pub struct LowRankKernelTransport {
    /// Row scaling applied to the left kernel factor.
    pub source_scaling: Vec<f64>,
    /// Row scaling applied to the right kernel factor.
    pub target_scaling: Vec<f64>,
    /// Factor `diag(source_scaling) * kernel_left`.
    pub left: Mat<f64>,
    /// Factor `diag(target_scaling) * kernel_right`.
    pub right: Mat<f64>,
    /// Number of alternating updates.
    pub iterations: usize,
    /// Last target-marginal L2 residual.
    pub residual: f64,
    /// Solver termination status.
    pub status: SolverStatus,
}

impl LowRankKernelTransport {
    /// Materialize the dense transport plan.
    pub fn dense_plan(&self) -> Mat<f64> {
        LowRankFactors {
            left: self.left.clone(),
            right: self.right.clone(),
        }
        .dense()
    }
}

fn validate_factors(left: &Mat<f64>, right: &Mat<f64>) -> Result<()> {
    validate::samples(left, "left low-rank factor")?;
    validate::samples(right, "right low-rank factor")?;
    if left.ncols() != right.ncols() {
        return Err(Error::ShapeMismatch {
            context: "low-rank factor dimensions",
            left: (left.nrows(), left.ncols()),
            right: (right.nrows(), right.ncols()),
        });
    }
    Ok(())
}

/// Factor a squared-Euclidean cost matrix exactly as `left * right���`.
///
/// The factors have `dimension + 2` columns. With `rescale = true`, each
/// factor is divided by the square root of its maximum entry, matching POT.
pub fn squared_euclidean_cost_factors(
    source: &Mat<f64>,
    target: &Mat<f64>,
    rescale: bool,
) -> Result<LowRankFactors> {
    validate::samples(source, "source samples")?;
    validate::samples(target, "target samples")?;
    if source.ncols() != target.ncols() {
        return Err(Error::ShapeMismatch {
            context: "low-rank squared-Euclidean samples",
            left: (source.nrows(), source.ncols()),
            right: (target.nrows(), target.ncols()),
        });
    }
    let dimensions = source.ncols();
    let mut left = Mat::<f64>::from_fn(source.nrows(), dimensions + 2, |i, j| match j {
        0 => (0..dimensions)
            .map(|coordinate| source[(i, coordinate)].powi(2))
            .sum(),
        1 => 1.0,
        _ => -2.0 * source[(i, j - 2)],
    });
    let mut right = Mat::<f64>::from_fn(target.nrows(), dimensions + 2, |i, j| match j {
        0 => 1.0,
        1 => (0..dimensions)
            .map(|coordinate| target[(i, coordinate)].powi(2))
            .sum(),
        _ => target[(i, j - 2)],
    });
    if rescale {
        let mut left_maximum = f64::NEG_INFINITY;
        for j in 0..left.ncols() {
            for i in 0..left.nrows() {
                left_maximum = left_maximum.max(left[(i, j)]);
            }
        }
        let mut right_maximum = f64::NEG_INFINITY;
        for j in 0..right.ncols() {
            for i in 0..right.nrows() {
                right_maximum = right_maximum.max(right[(i, j)]);
            }
        }
        let left_scale = left_maximum.sqrt();
        let right_scale = right_maximum.sqrt();
        for j in 0..left.ncols() {
            for i in 0..left.nrows() {
                left[(i, j)] /= left_scale;
            }
        }
        for j in 0..right.ncols() {
            for i in 0..right.nrows() {
                right[(i, j)] /= right_scale;
            }
        }
    }
    Ok(LowRankFactors { left, right })
}

fn factor_transpose_vector(factor: &Mat<f64>, vector: &[f64]) -> Vec<f64> {
    (0..factor.ncols())
        .map(|rank| {
            (0..factor.nrows())
                .map(|row| factor[(row, rank)] * vector[row])
                .sum()
        })
        .collect()
}

fn factor_vector(factor: &Mat<f64>, vector: &[f64]) -> Vec<f64> {
    (0..factor.nrows())
        .map(|row| {
            (0..factor.ncols())
                .map(|rank| factor[(row, rank)] * vector[rank])
                .sum()
        })
        .collect()
}

/// Scale a non-negative low-rank kernel to prescribed marginals.
///
/// The implicit kernel is `kernel_left * kernel_right���`; the dense kernel and
/// dense plan are never needed during iterations.
pub fn sinkhorn_low_rank_kernel(
    kernel_left: &Mat<f64>,
    kernel_right: &Mat<f64>,
    source: Option<&[f64]>,
    target: Option<&[f64]>,
    options: LowRankKernelOptions,
) -> Result<LowRankKernelTransport> {
    validate_factors(kernel_left, kernel_right)?;
    if options.max_iterations == 0 || options.check_interval == 0 {
        return Err(Error::InvalidParameter {
            name: "max_iterations and check_interval",
            requirement: "positive",
        });
    }
    validate::finite_positive(
        options.tolerance,
        "tolerance",
        "finite and strictly positive",
    )?;
    for factor in [kernel_left, kernel_right] {
        for j in 0..factor.ncols() {
            for i in 0..factor.nrows() {
                if factor[(i, j)] < 0.0 {
                    return Err(Error::InvalidParameter {
                        name: "low-rank kernel factors",
                        requirement: "non-negative",
                    });
                }
            }
        }
    }
    let source_storage;
    let source = match source {
        Some(weights) => weights,
        None => {
            source_storage = validate::uniform(kernel_left.nrows())?;
            &source_storage
        }
    };
    let target_storage;
    let target = match target {
        Some(weights) => weights,
        None => {
            target_storage = validate::uniform(kernel_right.nrows())?;
            &target_storage
        }
    };
    if source.len() != kernel_left.nrows() || target.len() != kernel_right.nrows() {
        return Err(Error::ShapeMismatch {
            context: "low-rank kernel factors and marginals",
            left: (kernel_left.nrows(), kernel_right.nrows()),
            right: (source.len(), target.len()),
        });
    }
    validate::balanced_distributions(source, target)?;

    let mut source_scaling = vec![1.0 / source.len() as f64; source.len()];
    let mut target_scaling = vec![1.0 / target.len() as f64; target.len()];
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        let kernel_transpose_source = factor_vector(
            kernel_right,
            &factor_transpose_vector(kernel_left, &source_scaling),
        );
        for j in 0..target.len() {
            let denominator = kernel_transpose_source[j];
            if target[j] > 0.0 && (!denominator.is_finite() || denominator <= 0.0) {
                return Err(Error::DidNotConverge {
                    algorithm: "low-rank kernel Sinkhorn",
                    iterations: iteration,
                    residual,
                });
            }
            target_scaling[j] = if target[j] == 0.0 {
                0.0
            } else {
                target[j] / denominator
            };
        }
        let kernel_target = factor_vector(
            kernel_left,
            &factor_transpose_vector(kernel_right, &target_scaling),
        );
        for i in 0..source.len() {
            let denominator = kernel_target[i];
            if source[i] > 0.0 && (!denominator.is_finite() || denominator <= 0.0) {
                return Err(Error::DidNotConverge {
                    algorithm: "low-rank kernel Sinkhorn",
                    iterations: iteration,
                    residual,
                });
            }
            source_scaling[i] = if source[i] == 0.0 {
                0.0
            } else {
                source[i] / denominator
            };
        }
        iterations = iteration;
        if iteration == 1
            || iteration % options.check_interval == 0
            || iteration == options.max_iterations
        {
            let target_marginal_kernel = factor_vector(
                kernel_right,
                &factor_transpose_vector(kernel_left, &source_scaling),
            );
            residual = target_marginal_kernel
                .iter()
                .zip(&target_scaling)
                .zip(target)
                .map(|((&kernel_value, &scaling), &weight)| {
                    (scaling * kernel_value - weight).powi(2)
                })
                .sum::<f64>()
                .sqrt();
            if residual <= options.tolerance {
                status = SolverStatus::Converged;
                break;
            }
        }
    }
    let left = Mat::<f64>::from_fn(kernel_left.nrows(), kernel_left.ncols(), |i, j| {
        source_scaling[i] * kernel_left[(i, j)]
    });
    let right = Mat::<f64>::from_fn(kernel_right.nrows(), kernel_right.ncols(), |i, j| {
        target_scaling[i] * kernel_right[(i, j)]
    });
    Ok(LowRankKernelTransport {
        source_scaling,
        target_scaling,
        left,
        right,
        iterations,
        residual,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_euclidean_factors_reconstruct_cost() {
        let source = Mat::<f64>::from_fn(3, 2, |i, j| [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]][i][j]);
        let target = Mat::<f64>::from_fn(2, 2, |i, j| [[1.0, 1.0], [2.0, 0.0]][i][j]);
        let factors = squared_euclidean_cost_factors(&source, &target, false).unwrap();
        let dense = factors.dense();
        let expected = [[2.0, 4.0], [1.0, 1.0], [2.0, 8.0]];
        for i in 0..3 {
            for j in 0..2 {
                assert!((dense[(i, j)] - expected[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn low_rank_kernel_scaling_matches_both_marginals() {
        let left = Mat::<f64>::from_fn(3, 2, |i, j| [[1.0, 0.2], [0.5, 1.0], [0.2, 0.7]][i][j]);
        let right = Mat::<f64>::from_fn(2, 2, |i, j| [[0.8, 0.3], [0.4, 1.1]][i][j]);
        let result = sinkhorn_low_rank_kernel(
            &left,
            &right,
            Some(&[0.2, 0.5, 0.3]),
            Some(&[0.6, 0.4]),
            LowRankKernelOptions {
                max_iterations: 10_000,
                tolerance: 1e-13,
                ..LowRankKernelOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, SolverStatus::Converged);
        let plan = result.dense_plan();
        for (i, expected) in [0.2, 0.5, 0.3].iter().enumerate() {
            let actual = (0..plan.ncols()).map(|j| plan[(i, j)]).sum::<f64>();
            assert!((actual - expected).abs() < 1e-12);
        }
        for (j, expected) in [0.6, 0.4].iter().enumerate() {
            let actual = (0..plan.nrows()).map(|i| plan[(i, j)]).sum::<f64>();
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
