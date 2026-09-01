//! Kernel functions and kernel-derived quantities.
//!
//! `coronel` owns kernel computation for the sibling `wormhole` and
//! `jelly-wave` crates.  Dense inputs and outputs use [`faer::Mat`] so the
//! crate can be embedded without converting through a second matrix type.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use faer::Mat;
use std::fmt;

/// Errors returned by kernel calculations.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// At least one input has no rows or columns.
    EmptyInput,
    /// Sample vectors have different dimensions.
    DimensionMismatch {
        /// Left-hand dimension.
        left: usize,
        /// Right-hand dimension.
        right: usize,
    },
    /// An input contains NaN or infinity.
    NonFiniteInput {
        /// Matrix row containing the invalid value.
        row: usize,
        /// Matrix column containing the invalid value.
        column: usize,
    },
    /// A kernel parameter is outside its mathematical domain.
    InvalidParameter(&'static str),
    /// An unbiased statistic was requested with fewer than two samples.
    TooFewSamples,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "kernel input must be non-empty"),
            Self::DimensionMismatch { left, right } => {
                write!(f, "feature dimensions differ: {left} != {right}")
            }
            Self::NonFiniteInput { row, column } => {
                write!(f, "kernel input at ({row}, {column}) is not finite")
            }
            Self::InvalidParameter(name) => write!(f, "invalid kernel parameter: {name}"),
            Self::TooFewSamples => {
                write!(f, "an unbiased estimate requires at least two samples")
            }
        }
    }
}

impl std::error::Error for Error {}

/// A positive-semidefinite kernel or commonly used neural tangent similarity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kernel {
    /// Dot-product kernel, `x · y`.
    Linear,
    /// Polynomial kernel, `(gamma * x · y + coef0)^degree`.
    Polynomial {
        /// Positive polynomial degree.
        degree: u32,
        /// Dot-product scale.
        gamma: f64,
        /// Additive offset.
        coef0: f64,
    },
    /// Gaussian radial-basis kernel, `exp(-gamma * ||x-y||²)`.
    Rbf {
        /// Positive inverse squared length scale.
        gamma: f64,
    },
    /// Laplacian kernel, `exp(-gamma * ||x-y||₁)`.
    Laplacian {
        /// Positive inverse length scale.
        gamma: f64,
    },
    /// Hyperbolic tangent similarity, `tanh(gamma * x · y + coef0)`.
    Sigmoid {
        /// Dot-product scale.
        gamma: f64,
        /// Additive offset.
        coef0: f64,
    },
    /// Cosine similarity with a zero-safe convention.
    Cosine,
    /// Exponentiated chi-squared kernel for non-negative features.
    ChiSquared {
        /// Positive distance scale.
        gamma: f64,
    },
}

impl Kernel {
    fn validate(self) -> Result<(), Error> {
        match self {
            Self::Linear | Self::Cosine => Ok(()),
            Self::Polynomial {
                degree,
                gamma,
                coef0,
            } => {
                if degree == 0 {
                    Err(Error::InvalidParameter("degree must be positive"))
                } else if !gamma.is_finite() {
                    Err(Error::InvalidParameter("gamma must be finite"))
                } else if !coef0.is_finite() {
                    Err(Error::InvalidParameter("coef0 must be finite"))
                } else {
                    Ok(())
                }
            }
            Self::Rbf { gamma } | Self::Laplacian { gamma } | Self::ChiSquared { gamma } => {
                if gamma.is_finite() && gamma > 0.0 {
                    Ok(())
                } else {
                    Err(Error::InvalidParameter(
                        "gamma must be finite and strictly positive",
                    ))
                }
            }
            Self::Sigmoid { gamma, coef0 } => {
                if gamma.is_finite() && coef0.is_finite() {
                    Ok(())
                } else {
                    Err(Error::InvalidParameter(
                        "gamma and coef0 must both be finite",
                    ))
                }
            }
        }
    }
}

fn validate_matrix(x: &Mat<f64>) -> Result<(), Error> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(Error::EmptyInput);
    }
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            if !x[(i, j)].is_finite() {
                return Err(Error::NonFiniteInput { row: i, column: j });
            }
        }
    }
    Ok(())
}

fn validate_vector(x: &[f64]) -> Result<(), Error> {
    if x.is_empty() {
        return Err(Error::EmptyInput);
    }
    for (column, value) in x.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::NonFiniteInput { row: 0, column });
        }
    }
    Ok(())
}

/// Evaluate `kernel` between two feature vectors.
pub fn value(kernel: Kernel, x: &[f64], y: &[f64]) -> Result<f64, Error> {
    kernel.validate()?;
    validate_vector(x)?;
    validate_vector(y)?;
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch {
            left: x.len(),
            right: y.len(),
        });
    }
    value_unchecked(kernel, x, y)
}

fn value_unchecked(kernel: Kernel, x: &[f64], y: &[f64]) -> Result<f64, Error> {
    let mut dot = 0.0;
    let mut squared = 0.0;
    let mut l1 = 0.0;
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut chi_squared = 0.0;
    for (&a, &b) in x.iter().zip(y) {
        dot += a * b;
        let delta = a - b;
        squared += delta * delta;
        l1 += delta.abs();
        nx += a * a;
        ny += b * b;
        if matches!(kernel, Kernel::ChiSquared { .. }) {
            if a < 0.0 || b < 0.0 {
                return Err(Error::InvalidParameter(
                    "chi-squared inputs must be non-negative",
                ));
            }
            let denominator = a + b;
            if denominator > 0.0 {
                chi_squared += delta * delta / denominator;
            }
        }
    }
    let result = match kernel {
        Kernel::Linear => dot,
        Kernel::Polynomial {
            degree,
            gamma,
            coef0,
        } => (gamma * dot + coef0).powi(degree as i32),
        Kernel::Rbf { gamma } => (-gamma * squared).exp(),
        Kernel::Laplacian { gamma } => (-gamma * l1).exp(),
        Kernel::Sigmoid { gamma, coef0 } => (gamma * dot + coef0).tanh(),
        Kernel::Cosine => {
            let denominator = (nx * ny).sqrt();
            if denominator > 0.0 {
                dot / denominator
            } else if nx == 0.0 && ny == 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Kernel::ChiSquared { gamma } => (-gamma * chi_squared).exp(),
    };
    Ok(result)
}

fn row(x: &Mat<f64>, index: usize) -> Vec<f64> {
    (0..x.ncols()).map(|j| x[(index, j)]).collect()
}

/// Compute the pairwise kernel matrix between rows of `x` and rows of `y`.
pub fn pairwise(kernel: Kernel, x: &Mat<f64>, y: &Mat<f64>) -> Result<Mat<f64>, Error> {
    kernel.validate()?;
    validate_matrix(x)?;
    validate_matrix(y)?;
    if x.ncols() != y.ncols() {
        return Err(Error::DimensionMismatch {
            left: x.ncols(),
            right: y.ncols(),
        });
    }
    let xs: Vec<_> = (0..x.nrows()).map(|i| row(x, i)).collect();
    let ys: Vec<_> = (0..y.nrows()).map(|i| row(y, i)).collect();
    let mut first_error = None;
    let result = Mat::<f64>::from_fn(x.nrows(), y.nrows(), |i, j| {
        match value_unchecked(kernel, &xs[i], &ys[j]) {
            Ok(value) => value,
            Err(error) => {
                first_error = Some(error);
                f64::NAN
            }
        }
    });
    match first_error {
        Some(error) => Err(error),
        None => Ok(result),
    }
}

/// Compute a square Gram matrix between rows of `x`.
pub fn gram(kernel: Kernel, x: &Mat<f64>) -> Result<Mat<f64>, Error> {
    kernel.validate()?;
    validate_matrix(x)?;
    let rows: Vec<_> = (0..x.nrows()).map(|i| row(x, i)).collect();
    let n = rows.len();
    let mut result = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let entry = value_unchecked(kernel, &rows[i], &rows[j])?;
            result[(i, j)] = entry;
            result[(j, i)] = entry;
        }
    }
    Ok(result)
}

/// Double-center a Gram matrix as `H K H`.
pub fn center_gram(input: &Mat<f64>) -> Result<Mat<f64>, Error> {
    validate_matrix(input)?;
    if input.nrows() != input.ncols() {
        return Err(Error::DimensionMismatch {
            left: input.nrows(),
            right: input.ncols(),
        });
    }
    let n = input.nrows();
    let mut row_means = vec![0.0; n];
    let mut column_means = vec![0.0; n];
    let mut mean = 0.0;
    for i in 0..n {
        for j in 0..n {
            let entry = input[(i, j)];
            row_means[i] += entry;
            column_means[j] += entry;
            mean += entry;
        }
    }
    let scale = 1.0 / n as f64;
    for item in &mut row_means {
        *item *= scale;
    }
    for item in &mut column_means {
        *item *= scale;
    }
    mean *= scale * scale;
    Ok(Mat::<f64>::from_fn(n, n, |i, j| {
        input[(i, j)] - row_means[i] - column_means[j] + mean
    }))
}

/// Kernel-induced distance `sqrt(k(x,x) + k(y,y) - 2 k(x,y))`.
pub fn distance(kernel: Kernel, x: &[f64], y: &[f64]) -> Result<f64, Error> {
    let xx = value(kernel, x, x)?;
    let yy = value(kernel, y, y)?;
    let xy = value(kernel, x, y)?;
    Ok((xx + yy - 2.0 * xy).max(0.0).sqrt())
}

/// Squared maximum mean discrepancy between two empirical samples.
///
/// The unbiased estimate removes Gram-matrix diagonal terms and requires two
/// rows in each sample.  The biased estimate includes those terms.
pub fn maximum_mean_discrepancy_squared(
    kernel: Kernel,
    x: &Mat<f64>,
    y: &Mat<f64>,
    unbiased: bool,
) -> Result<f64, Error> {
    let kxx = gram(kernel, x)?;
    let kyy = gram(kernel, y)?;
    let kxy = pairwise(kernel, x, y)?;
    let n = x.nrows();
    let m = y.nrows();
    if unbiased && (n < 2 || m < 2) {
        return Err(Error::TooFewSamples);
    }
    let mut xx = 0.0;
    let mut yy = 0.0;
    let mut xy = 0.0;
    for i in 0..n {
        for j in 0..n {
            if !unbiased || i != j {
                xx += kxx[(i, j)];
            }
        }
    }
    for i in 0..m {
        for j in 0..m {
            if !unbiased || i != j {
                yy += kyy[(i, j)];
            }
        }
    }
    for i in 0..n {
        for j in 0..m {
            xy += kxy[(i, j)];
        }
    }
    let xx_denominator = if unbiased { n * (n - 1) } else { n * n };
    let yy_denominator = if unbiased { m * (m - 1) } else { m * m };
    Ok(xx / xx_denominator as f64 + yy / yy_denominator as f64 - 2.0 * xy / (n * m) as f64)
}

/// Frobenius alignment between centered kernel matrices.
pub fn centered_alignment(left: &Mat<f64>, right: &Mat<f64>) -> Result<f64, Error> {
    if left.nrows() != right.nrows() || left.ncols() != right.ncols() {
        return Err(Error::DimensionMismatch {
            left: left.nrows().saturating_mul(left.ncols()),
            right: right.nrows().saturating_mul(right.ncols()),
        });
    }
    let left = center_gram(left)?;
    let right = center_gram(right)?;
    let mut numerator = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for j in 0..left.ncols() {
        for i in 0..left.nrows() {
            let a = left[(i, j)];
            let b = right[(i, j)];
            numerator += a * b;
            left_norm += a * a;
            right_norm += b * b;
        }
    }
    let denominator = (left_norm * right_norm).sqrt();
    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbf_gram_is_symmetric_with_unit_diagonal() {
        let x = Mat::<f64>::from_fn(3, 2, |i, j| (i + j) as f64);
        let k = gram(Kernel::Rbf { gamma: 0.5 }, &x).expect("valid Gram");
        for i in 0..3 {
            assert!((k[(i, i)] - 1.0).abs() < 1e-12);
            for j in 0..3 {
                assert!((k[(i, j)] - k[(j, i)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn mmd_vanishes_for_same_biased_sample() {
        let x = Mat::<f64>::from_fn(3, 1, |i, _| i as f64);
        let mmd = maximum_mean_discrepancy_squared(Kernel::Rbf { gamma: 1.0 }, &x, &x, false)
            .expect("valid MMD");
        assert!(mmd.abs() < 1e-12);
    }

    #[test]
    fn rejects_negative_chi_squared_features() {
        let error = value(Kernel::ChiSquared { gamma: 1.0 }, &[-1.0], &[1.0])
            .expect_err("negative histograms are invalid");
        assert!(matches!(error, Error::InvalidParameter(_)));
    }

    #[test]
    fn every_kernel_variant_matches_its_definition() {
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];
        assert_eq!(value(Kernel::Linear, &x, &y).unwrap(), 11.0);
        assert!(
            (value(
                Kernel::Polynomial {
                    degree: 2,
                    gamma: 0.5,
                    coef0: 1.0,
                },
                &x,
                &y,
            )
            .unwrap()
                - 42.25)
                .abs()
                < 1e-12
        );
        assert!(
            (value(Kernel::Rbf { gamma: 0.5 }, &x, &y).unwrap() - (-4.0_f64).exp()).abs() < 1e-12
        );
        assert!(
            (value(Kernel::Laplacian { gamma: 0.5 }, &x, &y).unwrap() - (-2.0_f64).exp()).abs()
                < 1e-12
        );
        assert!(
            (value(
                Kernel::Sigmoid {
                    gamma: 0.1,
                    coef0: 0.2,
                },
                &x,
                &y,
            )
            .unwrap()
                - 1.3_f64.tanh())
            .abs()
                < 1e-12
        );
        assert!((value(Kernel::Cosine, &x, &y).unwrap() - 11.0 / 125.0_f64.sqrt()).abs() < 1e-12);
        assert!(
            (value(Kernel::ChiSquared { gamma: 1.0 }, &x, &y).unwrap() - (-5.0_f64 / 3.0).exp())
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn centered_gram_distance_and_alignment_have_expected_invariants() {
        let x = Mat::<f64>::from_fn(3, 2, |i, j| [[0.0, 1.0], [1.0, 2.0], [3.0, 1.0]][i][j]);
        let raw = gram(Kernel::Linear, &x).unwrap();
        let centered = center_gram(&raw).unwrap();
        for i in 0..centered.nrows() {
            let row_sum = (0..centered.ncols()).map(|j| centered[(i, j)]).sum::<f64>();
            let column_sum = (0..centered.nrows()).map(|j| centered[(j, i)]).sum::<f64>();
            assert!(row_sum.abs() < 1e-12);
            assert!(column_sum.abs() < 1e-12);
        }
        assert!((centered_alignment(&raw, &raw).unwrap() - 1.0).abs() < 1e-12);
        assert!(
            (distance(Kernel::Linear, &[1.0, 2.0], &[3.0, 4.0]).unwrap() - 8.0_f64.sqrt()).abs()
                < 1e-12
        );
    }

    #[test]
    fn unbiased_mmd_excludes_both_gram_diagonals() {
        let x = Mat::<f64>::from_fn(2, 1, |i, _| i as f64);
        let y = Mat::<f64>::from_fn(2, 1, |i, _| (i + 2) as f64);
        let value = maximum_mean_discrepancy_squared(Kernel::Linear, &x, &y, true).unwrap();
        assert!((value - 3.5).abs() < 1e-12);
    }
}
