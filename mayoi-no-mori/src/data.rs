//! Matrix adapters and shared input validation.

use crate::{Error, Result};
use oldwood::MatrixView;

pub(crate) struct IndexedRows<'a, M: MatrixView + ?Sized> {
    source: &'a M,
    rows: &'a [usize],
}

impl<'a, M: MatrixView + ?Sized> IndexedRows<'a, M> {
    pub(crate) fn new(source: &'a M, rows: &'a [usize]) -> Self {
        Self { source, rows }
    }
}

impl<M: MatrixView + ?Sized> MatrixView for IndexedRows<'_, M> {
    fn nrows(&self) -> usize {
        self.rows.len()
    }

    fn ncols(&self) -> usize {
        self.source.ncols()
    }

    fn get(&self, row: usize, column: usize) -> f64 {
        self.source.get(self.rows[row], column)
    }
}

pub(crate) fn validate_training<M: MatrixView + ?Sized>(
    x: &M,
    target_len: usize,
    sample_weight: Option<&[f64]>,
) -> Result<()> {
    if x.nrows() == 0 {
        return Err(Error::EmptyTrainingData);
    }
    if x.ncols() == 0 {
        return Err(Error::EmptyFeatures);
    }
    if target_len != x.nrows() {
        return Err(Error::Length {
            name: "target",
            expected: x.nrows(),
            actual: target_len,
        });
    }
    if let Some(weights) = sample_weight {
        if weights.len() != x.nrows() {
            return Err(Error::Length {
                name: "sample_weight",
                expected: x.nrows(),
                actual: weights.len(),
            });
        }
        let mut positive = false;
        for (index, &weight) in weights.iter().enumerate() {
            if !weight.is_finite() {
                return Err(Error::NonFinite {
                    name: "sample_weight",
                    index,
                });
            }
            if weight < 0.0 {
                return Err(Error::NegativeWeight { index });
            }
            positive |= weight > 0.0;
        }
        if !positive {
            return Err(Error::NoPositiveWeight);
        }
    }
    Ok(())
}

pub(crate) fn checked_weights(rows: usize, weights: Option<&[f64]>) -> Vec<f64> {
    weights.map_or_else(|| vec![1.0; rows], <[f64]>::to_vec)
}

pub(crate) fn validate_finite_features<M: MatrixView + ?Sized>(x: &M) -> Result<()> {
    for row in 0..x.nrows() {
        for column in 0..x.ncols() {
            if !x.get(row, column).is_finite() {
                return Err(Error::NonFinite {
                    name: "feature",
                    index: row.saturating_mul(x.ncols()).saturating_add(column),
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_regression_target(target: &[f64]) -> Result<()> {
    for (index, &value) in target.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::NonFinite {
                name: "target",
                index,
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_predict<M: MatrixView + ?Sized>(x: &M, features: usize) -> Result<()> {
    if x.ncols() != features {
        return Err(Error::FeatureCount {
            expected: features,
            actual: x.ncols(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oldwood::DenseMatrix;

    fn matrix(rows: usize, columns: usize, values: &[f64]) -> DenseMatrix {
        DenseMatrix::from_row_major(rows, columns, values.to_vec()).expect("valid fixture")
    }

    #[test]
    fn training_shape_and_weight_failures_are_typed() {
        let empty_rows = matrix(0, 1, &[]);
        assert_eq!(
            validate_training(&empty_rows, 0, None),
            Err(Error::EmptyTrainingData)
        );
        let empty_columns = matrix(1, 0, &[]);
        assert_eq!(
            validate_training(&empty_columns, 1, None),
            Err(Error::EmptyFeatures)
        );
        let x = matrix(2, 1, &[0.0, 1.0]);
        assert!(matches!(
            validate_training(&x, 1, None),
            Err(Error::Length { name: "target", .. })
        ));
        assert!(matches!(
            validate_training(&x, 2, Some(&[1.0])),
            Err(Error::Length {
                name: "sample_weight",
                ..
            })
        ));
        assert_eq!(
            validate_training(&x, 2, Some(&[0.0, 0.0])),
            Err(Error::NoPositiveWeight)
        );
        assert_eq!(
            validate_training(&x, 2, Some(&[-1.0, 1.0])),
            Err(Error::NegativeWeight { index: 0 })
        );
        assert_eq!(
            validate_training(&x, 2, Some(&[f64::NAN, 1.0])),
            Err(Error::NonFinite {
                name: "sample_weight",
                index: 0
            })
        );
    }

    #[test]
    fn finite_feature_target_and_prediction_checks_report_positions() {
        let feature = matrix(2, 2, &[0.0, 1.0, f64::INFINITY, 3.0]);
        assert_eq!(
            validate_finite_features(&feature),
            Err(Error::NonFinite {
                name: "feature",
                index: 2
            })
        );
        assert_eq!(
            validate_regression_target(&[0.0, f64::NAN]),
            Err(Error::NonFinite {
                name: "target",
                index: 1
            })
        );
        assert_eq!(
            validate_predict(&feature, 1),
            Err(Error::FeatureCount {
                expected: 1,
                actual: 2
            })
        );
    }
}
