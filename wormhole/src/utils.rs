//! Numerical support utilities shared by optimal-transport algorithms.

use crate::error::{Error, Result};
use crate::validate;
use faer::{Mat, Side};

/// Axis used by sparse-simplex matrix projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionAxis {
    /// Project the flattened matrix as one vector.
    All,
    /// Project every matrix row independently.
    Rows,
    /// Project every matrix column independently.
    Columns,
}

/// Cost-matrix normalization compatible with POT's named transformations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CostNormalization {
    /// Divide by the median matrix entry.
    Median,
    /// Divide by the maximum matrix entry.
    Maximum,
    /// Apply `ln(1 + cost)` elementwise.
    Log,
    /// Apply `ln(1 + ln(1 + cost))` elementwise.
    LogLog,
}

fn validate_vector(values: &[f64], name: &'static str) -> Result<()> {
    if values.is_empty() {
        return Err(Error::EmptyInput { name });
    }
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::InvalidWeight { index, value });
        }
    }
    Ok(())
}

fn validate_matrix(matrix: &Mat<f64>, name: &'static str) -> Result<()> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Err(Error::EmptyInput { name });
    }
    for j in 0..matrix.ncols() {
        for i in 0..matrix.nrows() {
            let value = matrix[(i, j)];
            if !value.is_finite() {
                return Err(Error::InvalidCost {
                    row: i,
                    column: j,
                    value,
                });
            }
        }
    }
    Ok(())
}

/// Orthogonally project a vector onto a non-negative simplex.
pub fn project_simplex(values: &[f64], mass: f64) -> Result<Vec<f64>> {
    validate_vector(values, "simplex values")?;
    validate::finite_non_negative(mass, "mass", "finite and non-negative")?;
    if mass == 0.0 {
        return Ok(vec![0.0; values.len()]);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    let mut cumulative = 0.0;
    let mut active = 0;
    let mut theta = 0.0;
    for (index, &value) in sorted.iter().enumerate() {
        cumulative += value;
        let candidate = (cumulative - mass) / (index + 1) as f64;
        if value > candidate {
            active = index + 1;
            theta = candidate;
        }
    }
    if active == 0 {
        return Err(Error::Infeasible {
            context: "simplex projection has no active coordinates",
        });
    }
    Ok(values
        .iter()
        .map(|&value| (value - theta).max(0.0))
        .collect())
}

/// Project each column of a matrix onto a simplex of common mass.
pub fn project_simplex_columns(matrix: &Mat<f64>, mass: f64) -> Result<Mat<f64>> {
    validate_matrix(matrix, "simplex matrix")?;
    let mut output = Mat::<f64>::zeros(matrix.nrows(), matrix.ncols());
    for column in 0..matrix.ncols() {
        let values = (0..matrix.nrows())
            .map(|row| matrix[(row, column)])
            .collect::<Vec<_>>();
        let projected = project_simplex(&values, mass)?;
        for row in 0..matrix.nrows() {
            output[(row, column)] = projected[row];
        }
    }
    Ok(output)
}

/// Project a vector onto a simplex with at most `max_nonzero` active entries.
pub fn project_sparse_simplex(values: &[f64], mass: f64, max_nonzero: usize) -> Result<Vec<f64>> {
    validate_vector(values, "sparse simplex values")?;
    if max_nonzero == 0 {
        return Err(Error::InvalidParameter {
            name: "max_nonzero",
            requirement: "positive",
        });
    }
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| values[right].total_cmp(&values[left]));
    order.truncate(max_nonzero.min(values.len()));
    let selected = order.iter().map(|&index| values[index]).collect::<Vec<_>>();
    let projected = project_simplex(&selected, mass)?;
    let mut output = vec![0.0; values.len()];
    for (&index, &value) in order.iter().zip(&projected) {
        output[index] = value;
    }
    Ok(output)
}

/// Sparse-simplex projection of a matrix along the selected axis.
pub fn project_sparse_simplex_matrix(
    matrix: &Mat<f64>,
    mass: f64,
    max_nonzero: usize,
    axis: ProjectionAxis,
) -> Result<Mat<f64>> {
    validate_matrix(matrix, "sparse simplex matrix")?;
    match axis {
        ProjectionAxis::All => {
            let values = (0..matrix.nrows())
                .flat_map(|row| (0..matrix.ncols()).map(move |column| matrix[(row, column)]))
                .collect::<Vec<_>>();
            let projected = project_sparse_simplex(&values, mass, max_nonzero)?;
            Ok(Mat::<f64>::from_fn(
                matrix.nrows(),
                matrix.ncols(),
                |row, column| projected[row * matrix.ncols() + column],
            ))
        }
        ProjectionAxis::Rows => {
            let mut output = Mat::<f64>::zeros(matrix.nrows(), matrix.ncols());
            for row in 0..matrix.nrows() {
                let values = (0..matrix.ncols())
                    .map(|column| matrix[(row, column)])
                    .collect::<Vec<_>>();
                let projected = project_sparse_simplex(&values, mass, max_nonzero)?;
                for column in 0..matrix.ncols() {
                    output[(row, column)] = projected[column];
                }
            }
            Ok(output)
        }
        ProjectionAxis::Columns => {
            let mut output = Mat::<f64>::zeros(matrix.nrows(), matrix.ncols());
            for column in 0..matrix.ncols() {
                let values = (0..matrix.nrows())
                    .map(|row| matrix[(row, column)])
                    .collect::<Vec<_>>();
                let projected = project_sparse_simplex(&values, mass, max_nonzero)?;
                for row in 0..matrix.nrows() {
                    output[(row, column)] = projected[row];
                }
            }
            Ok(output)
        }
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

/// Normalize a cost matrix with an automatically selected scale.
pub fn normalize_cost(cost: &Mat<f64>, normalization: CostNormalization) -> Result<Mat<f64>> {
    Ok(normalize_cost_with_value(cost, normalization, None)?.0)
}

/// Normalize a cost matrix and return the scale used by divide normalizations.
///
/// `value` overrides the automatic median or maximum scale. It is ignored by
/// logarithmic transformations.
pub fn normalize_cost_with_value(
    cost: &Mat<f64>,
    normalization: CostNormalization,
    value: Option<f64>,
) -> Result<(Mat<f64>, Option<f64>)> {
    validate_matrix(cost, "cost matrix")?;
    match normalization {
        CostNormalization::Median | CostNormalization::Maximum => {
            let scale = match value {
                Some(scale) => scale,
                None if normalization == CostNormalization::Median => {
                    let mut entries = (0..cost.nrows())
                        .flat_map(|row| (0..cost.ncols()).map(move |column| cost[(row, column)]))
                        .collect::<Vec<_>>();
                    median(&mut entries)
                }
                None => (0..cost.nrows())
                    .flat_map(|row| (0..cost.ncols()).map(move |column| cost[(row, column)]))
                    .fold(f64::NEG_INFINITY, f64::max),
            };
            if !scale.is_finite() || scale == 0.0 {
                return Err(Error::InvalidParameter {
                    name: "cost normalization scale",
                    requirement: "finite and non-zero",
                });
            }
            Ok((
                Mat::<f64>::from_fn(cost.nrows(), cost.ncols(), |i, j| cost[(i, j)] / scale),
                Some(scale),
            ))
        }
        CostNormalization::Log | CostNormalization::LogLog => {
            let mut output = Mat::<f64>::zeros(cost.nrows(), cost.ncols());
            for j in 0..cost.ncols() {
                for i in 0..cost.nrows() {
                    let first = cost[(i, j)].ln_1p();
                    let transformed = if normalization == CostNormalization::Log {
                        first
                    } else {
                        first.ln_1p()
                    };
                    if !transformed.is_finite() {
                        return Err(Error::InvalidParameter {
                            name: "cost entries",
                            requirement: "inside the selected logarithm domain",
                        });
                    }
                    output[(i, j)] = transformed;
                }
            }
            Ok((output, None))
        }
    }
}

fn validate_symmetric(matrix: &Mat<f64>, name: &'static str) -> Result<()> {
    validate_matrix(matrix, name)?;
    if matrix.nrows() != matrix.ncols() {
        return Err(Error::ShapeMismatch {
            context: "symmetric matrix",
            left: (matrix.nrows(), matrix.ncols()),
            right: (matrix.nrows(), matrix.nrows()),
        });
    }
    let scale = (0..matrix.nrows())
        .flat_map(|i| (0..matrix.ncols()).map(move |j| matrix[(i, j)].abs()))
        .fold(1.0_f64, f64::max);
    for i in 0..matrix.nrows() {
        for j in 0..i {
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > 1e-12 * scale {
                return Err(Error::InvalidParameter {
                    name,
                    requirement: "symmetric within numerical tolerance",
                });
            }
        }
    }
    Ok(())
}

/// Project a symmetric matrix onto matrices with eigenvalues at least `minimum`.
pub fn project_psd(matrix: &Mat<f64>, minimum: f64) -> Result<Mat<f64>> {
    validate_symmetric(matrix, "PSD projection input")?;
    if !minimum.is_finite() {
        return Err(Error::InvalidParameter {
            name: "minimum eigenvalue",
            requirement: "finite",
        });
    }
    let decomposition =
        matrix
            .self_adjoint_eigen(Side::Lower)
            .map_err(|_| Error::LinearAlgebra {
                operation: "PSD projection eigendecomposition",
            })?;
    let vectors = decomposition.U();
    let values =
        matrix
            .self_adjoint_eigenvalues(Side::Lower)
            .map_err(|_| Error::LinearAlgebra {
                operation: "PSD projection eigenvalues",
            })?;
    Ok(Mat::<f64>::from_fn(
        matrix.nrows(),
        matrix.ncols(),
        |i, j| {
            (0..matrix.nrows())
                .map(|k| vectors[(i, k)] * values[k].max(minimum) * vectors[(j, k)])
                .sum()
        },
    ))
}

/// Bures-Wasserstein exponential map `(I + tangent) covariance (I + tangent)`.
pub fn bures_exponential(covariance: &Mat<f64>, tangent: &Mat<f64>) -> Result<Mat<f64>> {
    validate_symmetric(covariance, "Bures covariance")?;
    validate_symmetric(tangent, "Bures tangent")?;
    if covariance.nrows() != tangent.nrows() {
        return Err(Error::ShapeMismatch {
            context: "Bures exponential matrices",
            left: (covariance.nrows(), covariance.ncols()),
            right: (tangent.nrows(), tangent.ncols()),
        });
    }
    let eigenvalues = covariance
        .self_adjoint_eigenvalues(Side::Lower)
        .map_err(|_| Error::LinearAlgebra {
            operation: "Bures covariance eigenvalues",
        })?;
    let scale = eigenvalues.iter().copied().fold(1.0_f64, f64::max);
    if eigenvalues.iter().any(|&value| value < -1e-12 * scale) {
        return Err(Error::InvalidParameter {
            name: "Bures covariance",
            requirement: "positive semidefinite",
        });
    }
    let transform = Mat::<f64>::from_fn(tangent.nrows(), tangent.ncols(), |i, j| {
        tangent[(i, j)] + if i == j { 1.0 } else { 0.0 }
    });
    let first = &transform * covariance;
    Ok(&first * &transform)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_projections_match_pot_example_values() {
        let values = [-0.5, 0.3, 1.2];
        let dense = project_simplex(&values, 1.0).unwrap();
        assert!((dense[0] - 0.0).abs() < 1e-12);
        assert!((dense[1] - 0.05).abs() < 1e-12);
        assert!((dense[2] - 0.95).abs() < 1e-12);
        assert_eq!(
            project_sparse_simplex(&values, 1.0, 1).unwrap(),
            vec![0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn named_cost_normalizations_match_pot() {
        let cost = Mat::<f64>::from_fn(2, 2, |i, j| [[0.0, 1.0], [3.0, 7.0]][i][j]);
        let median = normalize_cost(&cost, CostNormalization::Median).unwrap();
        assert_eq!(median[(0, 1)], 0.5);
        assert_eq!(median[(1, 0)], 1.5);
        assert_eq!(median[(1, 1)], 3.5);
        let maximum = normalize_cost(&cost, CostNormalization::Maximum).unwrap();
        assert!((maximum[(0, 1)] - 1.0 / 7.0).abs() < 1e-12);
        let loglog = normalize_cost(&cost, CostNormalization::LogLog).unwrap();
        assert!((loglog[(1, 1)] - 1.1247482629090362).abs() < 1e-12);
    }

    #[test]
    fn psd_projection_and_bures_exponential_match_pot() {
        let indefinite = Mat::<f64>::from_fn(2, 2, |i, j| [[1.0, 2.0], [2.0, 1.0]][i][j]);
        let projected = project_psd(&indefinite, 0.0).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((projected[(i, j)] - 1.5).abs() < 1e-12);
            }
        }
        let covariance = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { [4.0, 9.0][i] } else { 0.0 });
        let tangent = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { [0.5, -0.5][i] } else { 0.0 });
        let exponential = bures_exponential(&covariance, &tangent).unwrap();
        assert!((exponential[(0, 0)] - 9.0).abs() < 1e-12);
        assert!((exponential[(1, 1)] - 2.25).abs() < 1e-12);
        assert!(exponential[(0, 1)].abs() < 1e-12);
    }
}
