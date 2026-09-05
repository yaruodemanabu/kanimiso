//! Dense design matrices and vectors backed by [`faer::Mat`].

use faer::Mat;
use std::fmt;

/// Column-major dense matrix of `f64`, optionally named.
#[derive(Clone, Debug)]
pub struct Matrix {
    inner: Mat<f64>,
    /// Optional feature / column names.
    pub col_names: Option<Vec<String>>,
}

impl Matrix {
    /// `n` × `p` zeros.
    pub fn zeros(n: usize, p: usize) -> Self {
        Self {
            inner: Mat::<f64>::zeros(n, p),
            col_names: None,
        }
    }

    /// Build from a function of `(row, col)`.
    pub fn from_fn(n: usize, p: usize, f: impl FnMut(usize, usize) -> f64) -> Self {
        Self {
            inner: Mat::<f64>::from_fn(n, p, f),
            col_names: None,
        }
    }

    /// Row-major slice `data.len() == n * p`.
    pub fn from_row_major(n: usize, p: usize, data: &[f64]) -> Self {
        assert_eq!(
            data.len(),
            n.saturating_mul(p),
            "row-major length must be n*p"
        );
        Self::from_fn(n, p, |i, j| data[i * p + j])
    }

    /// One column from a vector (n × 1).
    pub fn from_vector(v: &Vector) -> Self {
        Self::from_fn(v.len(), 1, |i, _| v[i])
    }

    /// Rows and columns.
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// Columns.
    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    /// `(n, p)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Read `X[i, j]`.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.inner[(i, j)]
    }

    /// Write `X[i, j]`.
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.inner[(i, j)] = v;
    }

    /// Borrow the faer matrix.
    pub fn inner(&self) -> &Mat<f64> {
        &self.inner
    }

    /// Mutable faer matrix.
    pub fn inner_mut(&mut self) -> &mut Mat<f64> {
        &mut self.inner
    }

    /// Consume and return the faer matrix.
    pub fn into_inner(self) -> Mat<f64> {
        self.inner
    }

    /// Wrap an existing faer matrix.
    pub fn from_faer(inner: Mat<f64>) -> Self {
        Self {
            inner,
            col_names: None,
        }
    }

    /// Copy column `j`.
    pub fn column(&self, j: usize) -> Vector {
        Vector::from_iter((0..self.nrows()).map(|i| self.get(i, j)))
    }

    /// Copy row `i`.
    pub fn row(&self, i: usize) -> Vector {
        Vector::from_iter((0..self.ncols()).map(|j| self.get(i, j)))
    }

    /// Prepend a column of ones (intercept).
    pub fn with_intercept(&self) -> Self {
        let (n, p) = self.shape();
        let mut out = Self::from_fn(
            n,
            p + 1,
            |i, j| {
                if j == 0 {
                    1.0
                } else {
                    self.get(i, j - 1)
                }
            },
        );
        if let Some(names) = &self.col_names {
            let mut nms = Vec::with_capacity(p + 1);
            nms.push("intercept".into());
            nms.extend(names.iter().cloned());
            out.col_names = Some(nms);
        }
        out
    }

    /// Center each column; returns the column means.
    pub fn centered(&self) -> (Self, Vector) {
        let (n, p) = self.shape();
        let mut means = Vector::zeros(p);
        if n == 0 {
            return (self.clone(), means);
        }
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += self.get(i, j);
            }
            means[j] = s / n as f64;
        }
        let out = Self::from_fn(n, p, |i, j| self.get(i, j) - means[j]);
        (out, means)
    }

    /// Frobenius norm.
    pub fn frobenius(&self) -> f64 {
        let mut s = 0.0;
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                let v = self.get(i, j);
                s += v * v;
            }
        }
        s.sqrt()
    }

    /// Flatten row-major.
    pub fn to_row_major(&self) -> Vec<f64> {
        let (n, p) = self.shape();
        let mut out = Vec::with_capacity(n * p);
        for i in 0..n {
            for j in 0..p {
                out.push(self.get(i, j));
            }
        }
        out
    }

    /// Matrix-vector product `X w`.
    pub fn matvec(&self, w: &Vector) -> Vector {
        assert_eq!(w.len(), self.ncols());
        let mut out = Vector::zeros(self.nrows());
        for i in 0..self.nrows() {
            let mut s = 0.0;
            for j in 0..self.ncols() {
                s += self.get(i, j) * w[j];
            }
            out[i] = s;
        }
        out
    }

    /// `Xᵀ v`.
    pub fn matvec_t(&self, v: &Vector) -> Vector {
        assert_eq!(v.len(), self.nrows());
        let mut out = Vector::zeros(self.ncols());
        for j in 0..self.ncols() {
            let mut s = 0.0;
            for i in 0..self.nrows() {
                s += self.get(i, j) * v[i];
            }
            out[j] = s;
        }
        out
    }

    /// Gram matrix `XᵀX` via faer multiply.
    pub fn gram(&self) -> Mat<f64> {
        self.inner.transpose() * &self.inner
    }

    /// Attach column names.
    pub fn with_col_names(mut self, names: Vec<String>) -> Self {
        self.col_names = Some(names);
        self
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix({}x{})", self.nrows(), self.ncols())
    }
}

impl crate::MatrixView for Matrix {
    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn get(&self, row: usize, column: usize) -> f64 {
        self.get(row, column)
    }
}

/// Dense real vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Vector {
    data: Vec<f64>,
}

impl Vector {
    /// Zeros.
    pub fn zeros(n: usize) -> Self {
        Self { data: vec![0.0; n] }
    }

    /// Filled.
    pub fn filled(n: usize, v: f64) -> Self {
        Self { data: vec![v; n] }
    }

    /// From a slice.
    pub fn from_slice(data: &[f64]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }

    /// Length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Empty?
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrow the storage.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Mutable storage.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// L2 norm.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Max abs.
    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0, |a, &x| a.max(x.abs()))
    }

    /// Dot product.
    pub fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len());
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }

    /// `self + other`.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        Self::from_iter(self.data.iter().zip(&other.data).map(|(a, b)| a + b))
    }

    /// `self - other`.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        Self::from_iter(self.data.iter().zip(&other.data).map(|(a, b)| a - b))
    }

    /// Scalar multiply.
    pub fn scale(&self, s: f64) -> Self {
        Self::from_iter(self.data.iter().map(|x| x * s))
    }

    /// Mean of finite entries.
    pub fn mean(&self) -> f64 {
        let st = signlred::slice_stats(&self.data);
        st.mean
    }

    /// Sample std of finite entries.
    pub fn std(&self) -> f64 {
        signlred::slice_stats(&self.data).std()
    }

    /// Convert to an n×1 matrix.
    pub fn to_matrix(&self) -> Matrix {
        Matrix::from_vector(self)
    }
}

impl FromIterator<f64> for Vector {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        Self {
            data: iter.into_iter().collect(),
        }
    }
}

impl std::ops::Index<usize> for Vector {
    type Output = f64;
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}

impl std::ops::IndexMut<usize> for Vector {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.data[i]
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector(len={})", self.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_identity() {
        let x = Matrix::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let w = Vector::from_slice(&[3.0, 4.0]);
        let y = x.matvec(&w);
        assert!((y[0] - 3.0).abs() < 1e-12);
        assert!((y[1] - 4.0).abs() < 1e-12);
    }
}
