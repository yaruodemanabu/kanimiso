/// Read-only two-dimensional feature source.
///
/// Implementations must return a value for every `row < nrows()` and
/// `column < ncols()`. Consumers must never call [`MatrixView::get`] outside those
/// bounds.
pub trait MatrixView {
    /// Number of rows.
    fn nrows(&self) -> usize;

    /// Number of columns.
    fn ncols(&self) -> usize;

    /// Value at `(row, column)`.
    fn get(&self, row: usize, column: usize) -> f64;
}

impl<T: MatrixView + ?Sized> MatrixView for &T {
    fn nrows(&self) -> usize {
        (**self).nrows()
    }

    fn ncols(&self) -> usize {
        (**self).ncols()
    }

    fn get(&self, row: usize, column: usize) -> f64 {
        (**self).get(row, column)
    }
}
