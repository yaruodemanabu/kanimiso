//! Closed-form Gaussian and Bures-Wasserstein quantities.

use crate::error::{Error, Result};
use crate::result::SolverStatus;
use crate::validate;
use faer::{Mat, Side};

fn validate_square(matrix: &Mat<f64>, name: &'static str) -> Result<()> {
    validate::samples(matrix, name)?;
    if matrix.nrows() != matrix.ncols() {
        return Err(Error::ShapeMismatch {
            context: "square covariance",
            left: (matrix.nrows(), matrix.ncols()),
            right: (matrix.nrows(), matrix.nrows()),
        });
    }
    Ok(())
}

fn symmetric_eigen(matrix: &Mat<f64>) -> Result<(Vec<f64>, Mat<f64>)> {
    let decomposition =
        matrix
            .self_adjoint_eigen(Side::Lower)
            .map_err(|_| Error::LinearAlgebra {
                operation: "self-adjoint eigendecomposition",
            })?;
    let vectors = decomposition.U().to_owned();
    let values =
        matrix
            .self_adjoint_eigenvalues(Side::Lower)
            .map_err(|_| Error::LinearAlgebra {
                operation: "self-adjoint eigenvalues",
            })?;
    Ok((values, vectors))
}

fn spectral_power(matrix: &Mat<f64>, power: f64, inverse: bool) -> Result<Mat<f64>> {
    validate_square(matrix, "positive-semidefinite matrix")?;
    let (values, vectors) = symmetric_eigen(matrix)?;
    let largest = values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = 1e-12 * largest.max(1.0);
    if values.iter().any(|&value| value < -tolerance) {
        return Err(Error::InvalidParameter {
            name: "covariance",
            requirement: "positive semidefinite",
        });
    }
    let transformed = values
        .iter()
        .map(|&value| {
            let value = value.max(0.0);
            if inverse && value <= tolerance {
                0.0
            } else {
                value.powf(power)
            }
        })
        .collect::<Vec<_>>();
    Ok(Mat::<f64>::from_fn(
        matrix.nrows(),
        matrix.ncols(),
        |i, j| {
            (0..matrix.nrows())
                .map(|k| vectors[(i, k)] * transformed[k] * vectors[(j, k)])
                .sum()
        },
    ))
}

fn matrix_product(left: &Mat<f64>, right: &Mat<f64>) -> Result<Mat<f64>> {
    if left.ncols() != right.nrows() {
        return Err(Error::ShapeMismatch {
            context: "matrix product",
            left: (left.nrows(), left.ncols()),
            right: (right.nrows(), right.ncols()),
        });
    }
    // Delegate the dense product itself to faer.
    Ok(left * right)
}

fn sandwich(left: &Mat<f64>, middle: &Mat<f64>) -> Result<Mat<f64>> {
    let first = matrix_product(left, middle)?;
    matrix_product(&first, &left.transpose().to_owned())
}

fn trace(matrix: &Mat<f64>) -> f64 {
    (0..matrix.nrows().min(matrix.ncols()))
        .map(|index| matrix[(index, index)])
        .sum()
}

/// Positive-semidefinite square root computed by `faer` eigendecomposition.
pub fn psd_sqrt(matrix: &Mat<f64>) -> Result<Mat<f64>> {
    spectral_power(matrix, 0.5, false)
}

/// Moore-Penrose inverse square root of a positive-semidefinite matrix.
pub fn psd_inverse_sqrt(matrix: &Mat<f64>) -> Result<Mat<f64>> {
    spectral_power(matrix, -0.5, true)
}

/// Bures distance between covariance matrices.
pub fn bures_distance(source: &Mat<f64>, target: &Mat<f64>) -> Result<f64> {
    validate_square(source, "source covariance")?;
    validate_square(target, "target covariance")?;
    if source.nrows() != target.nrows() {
        return Err(Error::ShapeMismatch {
            context: "Bures covariance dimensions",
            left: (source.nrows(), source.ncols()),
            right: (target.nrows(), target.ncols()),
        });
    }
    let source_root = psd_sqrt(source)?;
    let middle = sandwich(&source_root, target)?;
    let middle_root = psd_sqrt(&middle)?;
    let squared = trace(source) + trace(target) - 2.0 * trace(&middle_root);
    Ok(squared.max(0.0).sqrt())
}

/// `W₂` distance between multivariate Gaussian distributions.
pub fn bures_wasserstein_distance(
    source_mean: &[f64],
    target_mean: &[f64],
    source_covariance: &Mat<f64>,
    target_covariance: &Mat<f64>,
) -> Result<f64> {
    if source_mean.len() != target_mean.len()
        || source_covariance.nrows() != source_mean.len()
        || target_covariance.nrows() != target_mean.len()
    {
        return Err(Error::ShapeMismatch {
            context: "Gaussian means and covariances",
            left: (source_mean.len(), source_covariance.nrows()),
            right: (target_mean.len(), target_covariance.nrows()),
        });
    }
    for (index, value) in source_mean.iter().chain(target_mean).copied().enumerate() {
        if !value.is_finite() {
            return Err(Error::InvalidCost {
                row: 0,
                column: index,
                value,
            });
        }
    }
    let mean_squared = source_mean
        .iter()
        .zip(target_mean)
        .map(|(&left, &right)| (left - right).powi(2))
        .sum::<f64>();
    let covariance = bures_distance(source_covariance, target_covariance)?;
    Ok((mean_squared + covariance * covariance).sqrt())
}

/// Affine optimal map between non-degenerate Gaussian distributions.
#[derive(Clone, Debug)]
pub struct GaussianMapping {
    /// Linear part of the affine map, with target rows and source columns.
    pub linear: Mat<f64>,
    /// Offset such that `map(x) = linear * x + offset`.
    pub offset: Vec<f64>,
}

impl GaussianMapping {
    /// Apply the affine mapping to one feature vector.
    pub fn apply(&self, sample: &[f64]) -> Result<Vec<f64>> {
        if sample.len() != self.linear.ncols() {
            return Err(Error::ShapeMismatch {
                context: "Gaussian map and sample",
                left: (self.linear.nrows(), self.linear.ncols()),
                right: (sample.len(), 1),
            });
        }
        Ok((0..self.linear.nrows())
            .map(|i| {
                self.offset[i]
                    + (0..self.linear.ncols())
                        .map(|j| self.linear[(i, j)] * sample[j])
                        .sum::<f64>()
            })
            .collect())
    }
}

/// Compute the affine Bures-Wasserstein map between two Gaussians.
pub fn bures_wasserstein_mapping(
    source_mean: &[f64],
    target_mean: &[f64],
    source_covariance: &Mat<f64>,
    target_covariance: &Mat<f64>,
) -> Result<GaussianMapping> {
    let _ = bures_wasserstein_distance(
        source_mean,
        target_mean,
        source_covariance,
        target_covariance,
    )?;
    let source_root = psd_sqrt(source_covariance)?;
    let source_inverse_root = psd_inverse_sqrt(source_covariance)?;
    let middle = sandwich(&source_root, target_covariance)?;
    let middle_root = psd_sqrt(&middle)?;
    let first = matrix_product(&source_inverse_root, &middle_root)?;
    let linear = matrix_product(&first, &source_inverse_root)?;
    let mapped_source_mean = (0..linear.nrows())
        .map(|i| {
            (0..linear.ncols())
                .map(|j| linear[(i, j)] * source_mean[j])
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let offset = target_mean
        .iter()
        .zip(mapped_source_mean)
        .map(|(&target, mapped)| target - mapped)
        .collect();
    Ok(GaussianMapping { linear, offset })
}

/// Weighted empirical mean and maximum-likelihood covariance.
pub fn empirical_gaussian(
    samples: &Mat<f64>,
    weights: Option<&[f64]>,
) -> Result<(Vec<f64>, Mat<f64>)> {
    validate::samples(samples, "Gaussian samples")?;
    let owned;
    let weights = match weights {
        Some(weights) => weights,
        None => {
            owned = validate::uniform(samples.nrows())?;
            &owned
        }
    };
    if weights.len() != samples.nrows() {
        return Err(Error::ShapeMismatch {
            context: "Gaussian samples and weights",
            left: (samples.nrows(), 1),
            right: (weights.len(), 1),
        });
    }
    let mass = validate::distribution(weights, "Gaussian weights")?;
    let mean = (0..samples.ncols())
        .map(|j| {
            (0..samples.nrows())
                .map(|i| weights[i] * samples[(i, j)])
                .sum::<f64>()
                / mass
        })
        .collect::<Vec<_>>();
    let covariance = Mat::<f64>::from_fn(samples.ncols(), samples.ncols(), |j, k| {
        (0..samples.nrows())
            .map(|i| weights[i] * (samples[(i, j)] - mean[j]) * (samples[(i, k)] - mean[k]))
            .sum::<f64>()
            / mass
    });
    Ok((mean, covariance))
}

/// Empirical Gaussian Bures-Wasserstein distance between two samples.
pub fn empirical_bures_wasserstein_distance(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
) -> Result<f64> {
    let (source_mean, source_covariance) = empirical_gaussian(source, source_weights)?;
    let (target_mean, target_covariance) = empirical_gaussian(target, target_weights)?;
    bures_wasserstein_distance(
        &source_mean,
        &target_mean,
        &source_covariance,
        &target_covariance,
    )
}

fn descending_psd_eigen(matrix: &Mat<f64>) -> Result<(Vec<f64>, Mat<f64>)> {
    validate_square(matrix, "Gaussian Gromov covariance")?;
    let (values, vectors) = symmetric_eigen(matrix)?;
    let largest = values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = 1e-12 * largest.max(1.0);
    if values.iter().any(|&value| value < -tolerance) {
        return Err(Error::InvalidParameter {
            name: "Gaussian Gromov covariance",
            requirement: "positive semidefinite",
        });
    }
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| values[right].total_cmp(&values[left]));
    let sorted_values = order
        .iter()
        .map(|&index| values[index].max(0.0))
        .collect::<Vec<_>>();
    let sorted_vectors = Mat::<f64>::from_fn(vectors.nrows(), vectors.ncols(), |i, j| {
        vectors[(i, order[j])]
    });
    Ok((sorted_values, sorted_vectors))
}

/// Closed-form Gaussian Gromov-Wasserstein distance between covariances.
pub fn gaussian_gromov_wasserstein_distance(
    source_covariance: &Mat<f64>,
    target_covariance: &Mat<f64>,
) -> Result<f64> {
    let (source_values, _) = descending_psd_eigen(source_covariance)?;
    let (target_values, _) = descending_psd_eigen(target_covariance)?;
    let (larger, smaller) = if source_values.len() >= target_values.len() {
        (&source_values, &target_values)
    } else {
        (&target_values, &source_values)
    };
    let trace_difference = larger.iter().sum::<f64>() - smaller.iter().sum::<f64>();
    let matched = larger
        .iter()
        .zip(smaller)
        .map(|(&left, &right)| (left - right).powi(2))
        .sum::<f64>();
    let unmatched = larger[smaller.len()..]
        .iter()
        .map(|value| value.powi(2))
        .sum::<f64>();
    Ok((4.0 * trace_difference.powi(2) + 8.0 * matched + 8.0 * unmatched).sqrt())
}

/// Closed-form affine Gaussian Gromov-Wasserstein map.
///
/// The mapping pairs covariance eigendirections in descending eigenvalue order.
/// `signs` controls the orientation of each paired direction and defaults to
/// positive signs.
pub fn gaussian_gromov_wasserstein_mapping(
    source_mean: &[f64],
    target_mean: &[f64],
    source_covariance: &Mat<f64>,
    target_covariance: &Mat<f64>,
    signs: Option<&[f64]>,
) -> Result<GaussianMapping> {
    if source_mean.len() != source_covariance.nrows()
        || target_mean.len() != target_covariance.nrows()
    {
        return Err(Error::ShapeMismatch {
            context: "Gaussian Gromov means and covariances",
            left: (source_mean.len(), source_covariance.nrows()),
            right: (target_mean.len(), target_covariance.nrows()),
        });
    }
    for (index, &value) in source_mean.iter().chain(target_mean).enumerate() {
        if !value.is_finite() {
            return Err(Error::InvalidCost {
                row: 0,
                column: index,
                value,
            });
        }
    }
    let (source_values, source_vectors) = descending_psd_eigen(source_covariance)?;
    let (target_values, target_vectors) = descending_psd_eigen(target_covariance)?;
    let paired = source_values.len().min(target_values.len());
    let signs_storage;
    let signs = match signs {
        Some(signs) => signs,
        None => {
            signs_storage = vec![1.0; paired];
            &signs_storage
        }
    };
    if signs.len() != paired || signs.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidParameter {
            name: "Gaussian Gromov mapping signs",
            requirement: "one finite value per paired eigendirection",
        });
    }
    let scale = source_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = 1e-12 * scale.max(1.0);
    if source_values[..paired]
        .iter()
        .any(|&value| value <= tolerance)
    {
        return Err(Error::InvalidParameter {
            name: "source Gaussian Gromov covariance",
            requirement: "positive definite on paired eigendirections",
        });
    }
    let ratios = (0..paired)
        .map(|index| signs[index] * (target_values[index] / source_values[index]).sqrt())
        .collect::<Vec<_>>();
    let linear = Mat::<f64>::from_fn(target_mean.len(), source_mean.len(), |i, j| {
        (0..paired)
            .map(|index| target_vectors[(i, index)] * ratios[index] * source_vectors[(j, index)])
            .sum()
    });
    let offset = (0..target_mean.len())
        .map(|i| {
            target_mean[i]
                - (0..source_mean.len())
                    .map(|j| linear[(i, j)] * source_mean[j])
                    .sum::<f64>()
        })
        .collect();
    Ok(GaussianMapping { linear, offset })
}

/// Empirical Gaussian Gromov-Wasserstein distance.
pub fn empirical_gaussian_gromov_wasserstein_distance(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
) -> Result<f64> {
    let (_, source_covariance) = empirical_gaussian(source, source_weights)?;
    let (_, target_covariance) = empirical_gaussian(target, target_weights)?;
    gaussian_gromov_wasserstein_distance(&source_covariance, &target_covariance)
}

/// Empirical affine Gaussian Gromov-Wasserstein map.
pub fn empirical_gaussian_gromov_wasserstein_mapping(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    signs: Option<&[f64]>,
) -> Result<GaussianMapping> {
    let (source_mean, source_covariance) = empirical_gaussian(source, source_weights)?;
    let (target_mean, target_covariance) = empirical_gaussian(target, target_weights)?;
    gaussian_gromov_wasserstein_mapping(
        &source_mean,
        &target_mean,
        &source_covariance,
        &target_covariance,
        signs,
    )
}

/// Stopping options for Gaussian Bures-Wasserstein barycenters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianBarycenterOptions {
    /// Maximum covariance fixed-point iterations.
    pub max_iterations: usize,
    /// Frobenius covariance change required for convergence.
    pub tolerance: f64,
}

impl Default for GaussianBarycenterOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-9,
        }
    }
}

/// Mean, covariance, and diagnostics of a Gaussian barycenter.
#[derive(Clone, Debug)]
pub struct GaussianBarycenter {
    /// Weighted Euclidean barycenter of input means.
    pub mean: Vec<f64>,
    /// Bures-Wasserstein barycenter covariance.
    pub covariance: Mat<f64>,
    /// Number of covariance iterations.
    pub iterations: usize,
    /// Last Frobenius covariance change.
    pub residual: f64,
    /// Termination status.
    pub status: SolverStatus,
}

/// Compute a Bures-Wasserstein barycenter of Gaussian distributions.
pub fn bures_wasserstein_barycenter(
    means: &[Vec<f64>],
    covariances: &[Mat<f64>],
    mixture: Option<&[f64]>,
    options: GaussianBarycenterOptions,
) -> Result<GaussianBarycenter> {
    if means.is_empty() || covariances.is_empty() {
        return Err(Error::EmptyInput {
            name: "Gaussian barycenter inputs",
        });
    }
    if means.len() != covariances.len() {
        return Err(Error::ShapeMismatch {
            context: "Gaussian barycenter means and covariances",
            left: (means.len(), 1),
            right: (covariances.len(), 1),
        });
    }
    let dimensions = means[0].len();
    if dimensions == 0 {
        return Err(Error::EmptyInput {
            name: "Gaussian dimensions",
        });
    }
    for (mean, covariance) in means.iter().zip(covariances) {
        validate_square(covariance, "Gaussian barycenter covariance")?;
        if mean.len() != dimensions || covariance.nrows() != dimensions {
            return Err(Error::ShapeMismatch {
                context: "Gaussian barycenter dimensions",
                left: (mean.len(), covariance.nrows()),
                right: (dimensions, dimensions),
            });
        }
    }
    let mut mixture = mixture.map_or_else(
        || validate::uniform(means.len()),
        |weights| Ok(weights.to_vec()),
    )?;
    if mixture.len() != means.len() {
        return Err(Error::ShapeMismatch {
            context: "Gaussian barycenter mixture",
            left: (mixture.len(), 1),
            right: (means.len(), 1),
        });
    }
    let mixture_sum = validate::distribution(&mixture, "Gaussian barycenter mixture")?;
    for weight in &mut mixture {
        *weight /= mixture_sum;
    }
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
    let mean = (0..dimensions)
        .map(|j| {
            means
                .iter()
                .zip(&mixture)
                .map(|(current, &weight)| weight * current[j])
                .sum()
        })
        .collect::<Vec<_>>();
    let mut covariance = Mat::<f64>::zeros(dimensions, dimensions);
    for (current, &weight) in covariances.iter().zip(&mixture) {
        for i in 0..dimensions {
            for j in 0..dimensions {
                covariance[(i, j)] += weight * current[(i, j)];
            }
        }
    }
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    let mut status = SolverStatus::IterationLimit;
    for iteration in 1..=options.max_iterations {
        let root = psd_sqrt(&covariance)?;
        let inverse_root = psd_inverse_sqrt(&covariance)?;
        let mut average_root = Mat::<f64>::zeros(dimensions, dimensions);
        for (current, &weight) in covariances.iter().zip(&mixture) {
            let transported_root = psd_sqrt(&sandwich(&root, current)?)?;
            for i in 0..dimensions {
                for j in 0..dimensions {
                    average_root[(i, j)] += weight * transported_root[(i, j)];
                }
            }
        }
        let squared = matrix_product(&average_root, &average_root)?;
        let first = matrix_product(&inverse_root, &squared)?;
        let next = matrix_product(&first, &inverse_root)?;
        residual = 0.0;
        for i in 0..dimensions {
            for j in 0..dimensions {
                residual += (next[(i, j)] - covariance[(i, j)]).powi(2);
            }
        }
        residual = residual.sqrt();
        covariance = next;
        iterations = iteration;
        if residual <= options.tolerance {
            status = SolverStatus::Converged;
            break;
        }
    }
    Ok(GaussianBarycenter {
        mean,
        covariance,
        iterations,
        residual,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_dimensional_bures_has_closed_form() {
        let first = Mat::<f64>::from_fn(1, 1, |_, _| 4.0);
        let second = Mat::<f64>::from_fn(1, 1, |_, _| 9.0);
        assert!((bures_distance(&first, &second).unwrap() - 1.0).abs() < 1e-12);
        let value = bures_wasserstein_distance(&[0.0], &[2.0], &first, &second).unwrap();
        assert!((value - 5.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn gaussian_mapping_matches_mean_and_scale() {
        let first = Mat::<f64>::from_fn(1, 1, |_, _| 4.0);
        let second = Mat::<f64>::from_fn(1, 1, |_, _| 9.0);
        let mapping = bures_wasserstein_mapping(&[1.0], &[5.0], &first, &second).unwrap();
        assert!((mapping.linear[(0, 0)] - 1.5).abs() < 1e-12);
        assert!((mapping.apply(&[1.0]).unwrap()[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn empirical_gaussian_recovers_simple_moments() {
        let sample = Mat::<f64>::from_fn(2, 1, |i, _| (2 * i) as f64);
        let (mean, covariance) = empirical_gaussian(&sample, None).unwrap();
        assert_eq!(mean, vec![1.0]);
        assert!((covariance[(0, 0)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_gromov_distance_and_map_have_closed_form_invariants() {
        let source = Mat::<f64>::from_fn(2, 2, |i, j| [[4.0, 1.0], [1.0, 1.0]][i][j]);
        let target = Mat::<f64>::from_fn(1, 1, |_, _| 9.0);
        let distance = gaussian_gromov_wasserstein_distance(&source, &target).unwrap();
        assert!((distance - 15.633307652783936).abs() < 1e-12);
        let mapping =
            gaussian_gromov_wasserstein_mapping(&[1.0, -1.0], &[2.0], &source, &target, None)
                .unwrap();
        assert!((mapping.apply(&[1.0, -1.0]).unwrap()[0] - 2.0).abs() < 1e-12);
        let first = &mapping.linear * &source;
        let mapped_covariance = &first * mapping.linear.transpose();
        assert!((mapped_covariance[(0, 0)] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn identical_gaussian_barycenter_is_fixed_point() {
        let covariance = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.0 });
        let result = bures_wasserstein_barycenter(
            &[vec![1.0, 2.0], vec![1.0, 2.0]],
            &[covariance.clone(), covariance.clone()],
            None,
            GaussianBarycenterOptions::default(),
        )
        .unwrap();
        assert_eq!(result.mean, vec![1.0, 2.0]);
        for i in 0..2 {
            for j in 0..2 {
                assert!((result.covariance[(i, j)] - covariance[(i, j)]).abs() < 1e-9);
            }
        }
    }
}
