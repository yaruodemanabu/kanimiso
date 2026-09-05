use crate::numeric::CompensatedSum;
use crate::{Error, MatrixView, Result};

pub(crate) fn validate_training_matrix<M: MatrixView>(matrix: &M) -> Result<()> {
    if matrix.nrows() == 0 {
        return Err(Error::EmptyTrainingData);
    }
    if matrix.ncols() == 0 {
        return Err(Error::EmptyFeatures);
    }
    validate_finite(matrix)
}

pub(crate) fn validate_prediction_matrix<M: MatrixView>(
    matrix: &M,
    expected_columns: usize,
) -> Result<()> {
    if matrix.ncols() != expected_columns {
        return Err(Error::FeatureCount {
            expected: expected_columns,
            actual: matrix.ncols(),
        });
    }
    validate_finite(matrix)
}

fn validate_finite<M: MatrixView>(matrix: &M) -> Result<()> {
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            if !matrix.get(row, column).is_finite() {
                return Err(Error::NonFiniteFeature { row, column });
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_targets(rows: usize, targets: &[f64]) -> Result<()> {
    validate_target_length(rows, targets.len())?;
    if let Some(row) = targets.iter().position(|value| !value.is_finite()) {
        return Err(Error::NonFiniteTarget { row });
    }
    Ok(())
}

pub(crate) fn validate_target_length(rows: usize, targets: usize) -> Result<()> {
    if targets != rows {
        return Err(Error::TargetLength {
            expected: rows,
            actual: targets,
        });
    }
    Ok(())
}

pub(crate) fn weights(rows: usize, supplied: Option<&[f64]>) -> Result<Vec<f64>> {
    let weights = supplied.map_or_else(|| vec![1.0; rows], <[f64]>::to_vec);
    if weights.len() != rows {
        return Err(Error::WeightLength {
            expected: rows,
            actual: weights.len(),
        });
    }
    let mut total = CompensatedSum::default();
    for (row, &weight) in weights.iter().enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::InvalidWeight { row });
        }
        total.add(weight, "sample-weight summation")?;
    }
    if total.total("sample-weight summation")? == 0.0 {
        return Err(Error::NoPositiveWeight);
    }
    Ok(weights)
}
