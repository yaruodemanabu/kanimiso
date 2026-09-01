//! Ground metrics and pairwise cost matrices.

use crate::error::{Error, Result};
use crate::validate;
use faer::Mat;

/// Built-in metrics for rows of dense matrices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric {
    /// Sum of squared coordinate differences.
    SquaredEuclidean,
    /// Euclidean (`L²`) distance.
    Euclidean,
    /// Manhattan (`L¹`) distance.
    Manhattan,
    /// Minkowski distance with the supplied finite exponent `p >= 1`.
    Minkowski(f64),
    /// Maximum absolute coordinate difference.
    Chebyshev,
    /// One minus cosine similarity.
    Cosine,
    /// One minus Pearson correlation.
    Correlation,
    /// Canberra distance.
    Canberra,
    /// Bray-Curtis dissimilarity.
    BrayCurtis,
    /// Fraction of unequal coordinates.
    Hamming,
    /// Square root of Jensen-Shannon divergence.
    JensenShannon,
}

/// Metric choices supported by the batched pairwise API.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BatchMetric {
    /// Any metric supported by [`distance`].
    Distance(Metric),
    /// POT-compatible `KL(right || left)` with a `1e-10` logarithm offset.
    KullbackLeibler,
}

impl From<Metric> for BatchMetric {
    fn from(metric: Metric) -> Self {
        Self::Distance(metric)
    }
}

fn validate_metric(metric: Metric) -> Result<()> {
    if let Metric::Minkowski(p) = metric {
        if !p.is_finite() || p < 1.0 {
            return Err(Error::InvalidParameter {
                name: "Minkowski exponent",
                requirement: "finite and at least one",
            });
        }
    }
    Ok(())
}

fn finite_vector(vector: &[f64], name: &'static str) -> Result<()> {
    if vector.is_empty() {
        return Err(Error::EmptyInput { name });
    }
    for (index, value) in vector.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(Error::InvalidCost {
                row: 0,
                column: index,
                value,
            });
        }
    }
    Ok(())
}

/// Compute a metric between two dense vectors.
pub fn distance(left: &[f64], right: &[f64], metric: Metric) -> Result<f64> {
    validate_metric(metric)?;
    finite_vector(left, "left vector")?;
    finite_vector(right, "right vector")?;
    if left.len() != right.len() {
        return Err(Error::ShapeMismatch {
            context: "metric vectors",
            left: (1, left.len()),
            right: (1, right.len()),
        });
    }
    distance_unchecked(left, right, metric)
}

fn distance_unchecked(left: &[f64], right: &[f64], metric: Metric) -> Result<f64> {
    match metric {
        Metric::SquaredEuclidean => Ok(left
            .iter()
            .zip(right)
            .map(|(&a, &b)| {
                let difference = a - b;
                difference * difference
            })
            .sum()),
        Metric::Euclidean => Ok(distance_unchecked(left, right, Metric::SquaredEuclidean)?.sqrt()),
        Metric::Manhattan => Ok(left.iter().zip(right).map(|(&a, &b)| (a - b).abs()).sum()),
        Metric::Minkowski(p) => Ok(left
            .iter()
            .zip(right)
            .map(|(&a, &b)| (a - b).abs().powf(p))
            .sum::<f64>()
            .powf(1.0 / p)),
        Metric::Chebyshev => Ok(left
            .iter()
            .zip(right)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0, f64::max)),
        Metric::Cosine => {
            let mut dot = 0.0;
            let mut left_norm = 0.0;
            let mut right_norm = 0.0;
            for (&a, &b) in left.iter().zip(right) {
                dot += a * b;
                left_norm += a * a;
                right_norm += b * b;
            }
            let denominator = (left_norm * right_norm).sqrt();
            if denominator > 0.0 {
                Ok((1.0 - dot / denominator).clamp(0.0, 2.0))
            } else if left_norm == 0.0 && right_norm == 0.0 {
                Ok(0.0)
            } else {
                Ok(1.0)
            }
        }
        Metric::Correlation => {
            let left_mean = left.iter().sum::<f64>() / left.len() as f64;
            let right_mean = right.iter().sum::<f64>() / right.len() as f64;
            let mut numerator = 0.0;
            let mut left_norm = 0.0;
            let mut right_norm = 0.0;
            for (&a, &b) in left.iter().zip(right) {
                let centered_left = a - left_mean;
                let centered_right = b - right_mean;
                numerator += centered_left * centered_right;
                left_norm += centered_left * centered_left;
                right_norm += centered_right * centered_right;
            }
            let denominator = (left_norm * right_norm).sqrt();
            if denominator > 0.0 {
                Ok((1.0 - numerator / denominator).clamp(0.0, 2.0))
            } else if left == right {
                Ok(0.0)
            } else {
                Ok(1.0)
            }
        }
        Metric::Canberra => Ok(left
            .iter()
            .zip(right)
            .map(|(&a, &b)| {
                let denominator = a.abs() + b.abs();
                if denominator > 0.0 {
                    (a - b).abs() / denominator
                } else {
                    0.0
                }
            })
            .sum()),
        Metric::BrayCurtis => {
            let numerator = left
                .iter()
                .zip(right)
                .map(|(&a, &b)| (a - b).abs())
                .sum::<f64>();
            let denominator = left
                .iter()
                .zip(right)
                .map(|(&a, &b)| (a + b).abs())
                .sum::<f64>();
            if denominator > 0.0 {
                Ok(numerator / denominator)
            } else {
                Ok(0.0)
            }
        }
        Metric::Hamming => {
            let unequal = left
                .iter()
                .zip(right)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            Ok(unequal as f64 / left.len() as f64)
        }
        Metric::JensenShannon => jensen_shannon(left, right),
    }
}

fn jensen_shannon(left: &[f64], right: &[f64]) -> Result<f64> {
    if left.iter().chain(right).any(|&value| value < 0.0) {
        return Err(Error::InvalidParameter {
            name: "Jensen-Shannon inputs",
            requirement: "non-negative",
        });
    }
    let left_sum = left.iter().sum::<f64>();
    let right_sum = right.iter().sum::<f64>();
    if left_sum <= 0.0 || right_sum <= 0.0 {
        return Err(Error::InvalidParameter {
            name: "Jensen-Shannon inputs",
            requirement: "have positive sums",
        });
    }
    let mut divergence = 0.0;
    for (&left_value, &right_value) in left.iter().zip(right) {
        let p = left_value / left_sum;
        let q = right_value / right_sum;
        let midpoint = 0.5 * (p + q);
        if p > 0.0 {
            divergence += 0.5 * p * (p / midpoint).ln();
        }
        if q > 0.0 {
            divergence += 0.5 * q * (q / midpoint).ln();
        }
    }
    Ok(divergence.max(0.0).sqrt())
}

fn matrix_row(matrix: &Mat<f64>, index: usize) -> Vec<f64> {
    (0..matrix.ncols())
        .map(|column| matrix[(index, column)])
        .collect()
}

/// Compute all distances between rows of `left` and rows of `right`.
pub fn pairwise(left: &Mat<f64>, right: &Mat<f64>, metric: Metric) -> Result<Mat<f64>> {
    validate_metric(metric)?;
    validate::samples(left, "left samples")?;
    validate::samples(right, "right samples")?;
    if left.ncols() != right.ncols() {
        return Err(Error::ShapeMismatch {
            context: "sample feature dimensions",
            left: (left.nrows(), left.ncols()),
            right: (right.nrows(), right.ncols()),
        });
    }
    let left_rows: Vec<_> = (0..left.nrows())
        .map(|index| matrix_row(left, index))
        .collect();
    let right_rows: Vec<_> = (0..right.nrows())
        .map(|index| matrix_row(right, index))
        .collect();
    let mut output = Mat::<f64>::zeros(left.nrows(), right.nrows());
    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            output[(i, j)] = distance_unchecked(&left_rows[i], &right_rows[j], metric)?;
        }
    }
    Ok(output)
}

/// Compute a square pairwise distance matrix between rows of one matrix.
pub fn pairwise_self(samples: &Mat<f64>, metric: Metric) -> Result<Mat<f64>> {
    validate_metric(metric)?;
    validate::samples(samples, "samples")?;
    let rows: Vec<_> = (0..samples.nrows())
        .map(|index| matrix_row(samples, index))
        .collect();
    let mut output = Mat::<f64>::zeros(samples.nrows(), samples.nrows());
    for i in 0..samples.nrows() {
        for j in 0..i {
            let value = distance_unchecked(&rows[i], &rows[j], metric)?;
            output[(i, j)] = value;
            output[(j, i)] = value;
        }
    }
    Ok(output)
}

/// Compute the POT-compatible reverse KL cost between all sample rows.
///
/// Element `(i, j)` is `KL(right[j] || left[i])`, matching
/// `ot.dist_batch(..., metric="kl")`.
pub fn pairwise_kullback_leibler(left: &Mat<f64>, right: &Mat<f64>) -> Result<Mat<f64>> {
    validate::samples(left, "left KL samples")?;
    validate::samples(right, "right KL samples")?;
    if left.ncols() != right.ncols() {
        return Err(Error::ShapeMismatch {
            context: "KL sample feature dimensions",
            left: (left.nrows(), left.ncols()),
            right: (right.nrows(), right.ncols()),
        });
    }
    for matrix in [left, right] {
        for j in 0..matrix.ncols() {
            for i in 0..matrix.nrows() {
                if matrix[(i, j)] < 0.0 {
                    return Err(Error::InvalidParameter {
                        name: "KL samples",
                        requirement: "non-negative",
                    });
                }
            }
        }
    }
    const LOG_OFFSET: f64 = 1e-10;
    Ok(Mat::<f64>::from_fn(left.nrows(), right.nrows(), |i, j| {
        (0..left.ncols())
            .map(|coordinate| {
                let source = left[(i, coordinate)];
                let target = right[(j, coordinate)];
                target * ((target + LOG_OFFSET).ln() - (source + LOG_OFFSET).ln())
            })
            .sum()
    }))
}

/// Compute corresponding pairwise matrices for batches of sample matrices.
///
/// Each input slice element represents one `(samples, features)` batch item.
/// If `right` is `None`, each left item is compared with itself.
pub fn pairwise_batch(
    left: &[Mat<f64>],
    right: Option<&[Mat<f64>]>,
    metric: BatchMetric,
) -> Result<Vec<Mat<f64>>> {
    if left.is_empty() {
        return Err(Error::EmptyInput {
            name: "left sample batch",
        });
    }
    if let Some(right) = right {
        if left.len() != right.len() {
            return Err(Error::ShapeMismatch {
                context: "pairwise batch lengths",
                left: (left.len(), 1),
                right: (right.len(), 1),
            });
        }
        left.iter()
            .zip(right)
            .map(|(left, right)| match metric {
                BatchMetric::Distance(metric) => pairwise(left, right, metric),
                BatchMetric::KullbackLeibler => pairwise_kullback_leibler(left, right),
            })
            .collect()
    } else {
        left.iter()
            .map(|samples| match metric {
                BatchMetric::Distance(metric) => pairwise_self(samples, metric),
                BatchMetric::KullbackLeibler => pairwise_kullback_leibler(samples, samples),
            })
            .collect()
    }
}

/// Compute pairwise squared Mahalanobis distances using a precision matrix.
pub fn mahalanobis_pairwise(
    left: &Mat<f64>,
    right: &Mat<f64>,
    precision: &Mat<f64>,
) -> Result<Mat<f64>> {
    validate::samples(left, "left samples")?;
    validate::samples(right, "right samples")?;
    validate::samples(precision, "precision matrix")?;
    let dimensions = left.ncols();
    if right.ncols() != dimensions
        || precision.nrows() != dimensions
        || precision.ncols() != dimensions
    {
        return Err(Error::ShapeMismatch {
            context: "Mahalanobis samples and precision",
            left: (left.nrows(), dimensions),
            right: (precision.nrows(), precision.ncols()),
        });
    }
    let mut output = Mat::<f64>::zeros(left.nrows(), right.nrows());
    let mut difference = vec![0.0; dimensions];
    let mut transformed = vec![0.0; dimensions];
    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            for k in 0..dimensions {
                difference[k] = left[(i, k)] - right[(j, k)];
            }
            for row in 0..dimensions {
                transformed[row] = (0..dimensions)
                    .map(|column| precision[(row, column)] * difference[column])
                    .sum();
            }
            output[(i, j)] = difference
                .iter()
                .zip(&transformed)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .max(0.0);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_distances_match_hand_calculation() {
        let left = [0.0, 3.0];
        let right = [4.0, 0.0];
        assert_eq!(
            distance(&left, &right, Metric::SquaredEuclidean).unwrap(),
            25.0
        );
        assert_eq!(distance(&left, &right, Metric::Euclidean).unwrap(), 5.0);
        assert_eq!(distance(&left, &right, Metric::Manhattan).unwrap(), 7.0);
        assert_eq!(distance(&left, &right, Metric::Chebyshev).unwrap(), 4.0);
    }

    #[test]
    fn pairwise_self_is_symmetric() {
        let samples = Mat::<f64>::from_fn(3, 2, |i, j| (i * 2 + j) as f64);
        let distances = pairwise_self(&samples, Metric::Euclidean).unwrap();
        for i in 0..3 {
            assert_eq!(distances[(i, i)], 0.0);
            for j in 0..3 {
                assert_eq!(distances[(i, j)], distances[(j, i)]);
            }
        }
    }

    #[test]
    fn jensen_shannon_identity_is_zero() {
        let p = [0.25, 0.75];
        assert_eq!(distance(&p, &p, Metric::JensenShannon).unwrap(), 0.0);
    }

    #[test]
    fn batches_preserve_independent_pairwise_shapes() {
        let first = Mat::<f64>::from_fn(2, 2, |i, j| (i + j) as f64);
        let second = Mat::<f64>::from_fn(3, 2, |i, j| (2 * i + j) as f64);
        let result = pairwise_batch(
            &[first, second],
            None,
            BatchMetric::Distance(Metric::SquaredEuclidean),
        )
        .unwrap();
        assert_eq!((result[0].nrows(), result[0].ncols()), (2, 2));
        assert_eq!((result[1].nrows(), result[1].ncols()), (3, 3));
    }
}
