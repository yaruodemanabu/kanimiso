use crate::{Error, Result};
pub use tsutsumi::MatrixView;

/// Owned row-major dense matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseMatrix {
    rows: usize,
    columns: usize,
    values: Vec<f64>,
}

impl DenseMatrix {
    /// Constructs a row-major matrix after checking its storage length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidMatrixStorage`] when `rows * columns`
    /// overflows or differs from `values.len()`.
    pub fn from_row_major(rows: usize, columns: usize, values: Vec<f64>) -> Result<Self> {
        let Some(expected) = rows.checked_mul(columns) else {
            return Err(Error::InvalidMatrixStorage {
                rows,
                columns,
                values: values.len(),
            });
        };
        if values.len() != expected {
            return Err(Error::InvalidMatrixStorage {
                rows,
                columns,
                values: values.len(),
            });
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    /// Returns the row-major backing slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }
}

impl MatrixView for DenseMatrix {
    fn nrows(&self) -> usize {
        self.rows
    }

    fn ncols(&self) -> usize {
        self.columns
    }

    fn get(&self, row: usize, column: usize) -> f64 {
        self.values[row * self.columns + column]
    }
}
