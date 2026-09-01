//! Small dense linear-algebra helpers on top of faer (Pure Rust).

use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::Side;

use crate::error::{Error, Result};

pub use faer::{Col, Mat, Scale};

/// `n × m` matrix filled with zeros.
pub fn mat_zeros(nrows: usize, ncols: usize) -> Mat<f64> {
    Mat::zeros(nrows, ncols)
}

/// `n × n` identity.
pub fn mat_identity(n: usize) -> Mat<f64> {
    Mat::identity(n, n)
}

/// Row-major slice → dense matrix.
pub fn mat_from_row_slice(nrows: usize, ncols: usize, data: &[f64]) -> Mat<f64> {
    assert_eq!(data.len(), nrows * ncols, "row-major slice length");
    Mat::from_fn(nrows, ncols, |i, j| data[i * ncols + j])
}

/// Column vector of zeros.
pub fn col_zeros(n: usize) -> Col<f64> {
    Col::zeros(n)
}

/// Column vector from a slice.
pub fn col_from_slice(data: &[f64]) -> Col<f64> {
    Col::from_fn(data.len(), |i| data[i])
}

/// Owned `Vec` copy of a column.
pub fn col_to_vec(x: &Col<f64>) -> Vec<f64> {
    (0..x.nrows()).map(|i| x[i]).collect()
}

/// `x · y`.
pub fn dot(x: &Col<f64>, y: &Col<f64>) -> f64 {
    assert_eq!(x.nrows(), y.nrows());
    let mut s = 0.0;
    for i in 0..x.nrows() {
        s += x[i] * y[i];
    }
    s
}

/// `s A`.
pub fn scale_mat(a: &Mat<f64>, s: f64) -> Mat<f64> {
    Scale(s) * a
}

/// `A x` for a column `x` stored as a slice.
pub fn matvec_slice(a: &Mat<f64>, x: &[f64]) -> Vec<f64> {
    let xv = col_from_slice(x);
    let y = a * &xv;
    col_to_vec(&y)
}

/// Cholesky factor of a symmetric positive-definite matrix (`L` with `A = L Lᵀ`).
pub fn cholesky(a: &Mat<f64>) -> Result<Mat<f64>> {
    let llt = a
        .llt(Side::Lower)
        .map_err(|_| Error::numeric("Cholesky failed (matrix not SPD)"))?;
    Ok(llt.L().to_owned())
}

/// Solve `A x = b` for SPD `A` via Cholesky.
pub fn solve_spd(a: &Mat<f64>, b: &Col<f64>) -> Result<Col<f64>> {
    let llt = a
        .llt(Side::Lower)
        .map_err(|_| Error::numeric("SPD solve failed"))?;
    Ok(llt.solve(b))
}

/// Forward-solve `L y = b` for lower-triangular `L`.
///
/// Returns [`Error::Numeric`] if a diagonal entry is zero (singular `L`).
pub fn solve_lower(l: &Mat<f64>, b: &Col<f64>) -> Result<Col<f64>> {
    let n = l.nrows();
    if b.nrows() != n {
        return Err(Error::dim("solve_lower: L and b dimension mismatch"));
    }
    let mut y = col_zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[(i, k)] * y[k];
        }
        let d = l[(i, i)];
        if !d.is_finite() || d.abs() == 0.0 {
            return Err(Error::numeric("solve_lower: zero or non-finite pivot"));
        }
        y[i] = s / d;
    }
    Ok(y)
}

/// Log-determinant of an SPD matrix via Cholesky (`2 Σ log Lᵢᵢ`).
pub fn logdet_spd(a: &Mat<f64>) -> Result<f64> {
    let l = cholesky(a)?;
    let mut s = 0.0;
    for i in 0..l.nrows() {
        let d = l[(i, i)];
        if d <= 0.0 {
            return Err(Error::numeric("non-positive Cholesky diagonal"));
        }
        s += d.ln();
    }
    Ok(2.0 * s)
}

/// `Σ = σ σᵀ` from a `n × m` diffusion matrix stored row-major.
pub fn gram_rowmajor(sigma: &[f64], n: usize, m: usize) -> Mat<f64> {
    let s = mat_from_row_slice(n, m, sigma);
    &s * s.transpose()
}

/// Matrix exponential via scaling-and-squaring + Taylor (small dense).
pub fn expm(a: &Mat<f64>) -> Mat<f64> {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols());
    let nrm = one_norm(a);
    let s = if nrm <= 1e-16 {
        0
    } else {
        nrm.log2().ceil().max(0.0) as u32
    };
    let b = Scale(0.5f64.powi(s as i32)) * a;
    let mut term = mat_identity(n);
    let mut sum = mat_identity(n);
    for k in 1..28 {
        term = &term * &b;
        term = Scale(1.0 / k as f64) * &term;
        sum = &sum + &term;
    }
    for _ in 0..s {
        sum = &sum * &sum;
    }
    sum
}

fn one_norm(a: &Mat<f64>) -> f64 {
    let mut nrm: f64 = 0.0;
    for j in 0..a.ncols() {
        let mut s: f64 = 0.0;
        for i in 0..a.nrows() {
            s += a[(i, j)].abs();
        }
        nrm = nrm.max(s);
    }
    nrm
}

/// Spectral radius (max |λ|).
pub fn spectral_radius(a: &Mat<f64>) -> Result<f64> {
    let ev = complex_eigenvalues(a)?;
    ev.iter()
        .map(|z| z.norm())
        .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| Error::numeric("empty eigenvalue set"))
}

/// Whether every eigenvalue has strictly negative real part.
pub fn is_hurwitz(a: &Mat<f64>) -> bool {
    match complex_eigenvalues(a) {
        Ok(ev) => ev.iter().all(|z| z.re < -1e-14),
        Err(_) => false,
    }
}

fn complex_eigenvalues(a: &Mat<f64>) -> Result<Vec<faer::c64>> {
    a.eigenvalues()
        .map_err(|_| Error::numeric("eigendecomposition failed"))
}

/// Dense inverse via partial-pivoting LU, or `None` if singular / non-finite.
///
/// Checks that every entry of the candidate inverse is finite and that
/// `‖A Â − I‖_∞` is at most a generous multiple of machine epsilon.
pub fn try_inverse(a: &Mat<f64>) -> Option<Mat<f64>> {
    if a.nrows() != a.ncols() || a.nrows() == 0 {
        return None;
    }
    let n = a.nrows();
    for i in 0..n {
        for j in 0..n {
            if !a[(i, j)].is_finite() {
                return None;
            }
        }
    }
    let inv = a.partial_piv_lu().inverse();
    for i in 0..n {
        for j in 0..n {
            if !inv[(i, j)].is_finite() {
                return None;
            }
        }
    }
    let prod = a * &inv;
    let mut err: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let target = if i == j { 1.0 } else { 0.0 };
            err = err.max((prod[(i, j)] - target).abs());
        }
    }
    let tol = 1e-8 * (n as f64) * (1.0 + one_norm(a) * one_norm(&inv));
    if err > tol {
        return None;
    }
    Some(inv)
}

/// Exact linear-SDE discretisation of Van Loan (1978).
///
/// For `dX = (A X + b) dt + G dW` over a step `Δt`,
///
/// ```text
/// F = exp(A Δt),
/// u = ∫_0^{Δt} exp(A s) b ds,
/// Q = ∫_0^{Δt} exp(A s) G Gᵀ exp(Aᵀ s) ds.
/// ```
///
/// `F` and `u` come from the affine block exponential
/// `exp([[A, b], [0, 0]] Δt)`. `Q` comes from
/// `exp([[−A, GGᵀ], [0, Aᵀ]] Δt)` via `Q = F Ψ₁₂`.
pub fn van_loan_discretize(
    a: &Mat<f64>,
    b: &Col<f64>,
    g: &Mat<f64>,
    dt: f64,
) -> Result<(Mat<f64>, Col<f64>, Mat<f64>)> {
    let n = a.nrows();
    if a.ncols() != n || b.nrows() != n || g.nrows() != n {
        return Err(Error::dim("van_loan: A, b, G dimension mismatch"));
    }
    if !dt.is_finite() || dt < 0.0 {
        return Err(Error::param("van_loan: dt must be finite and ≥ 0"));
    }
    if dt == 0.0 {
        return Ok((mat_identity(n), col_zeros(n), mat_zeros(n, n)));
    }
    let (f, u) = affine_expm_step(a, b, dt);
    let gg = g * g.transpose();
    let mut m = mat_zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = -a[(i, j)] * dt;
            m[(i, n + j)] = gg[(i, j)] * dt;
            m[(n + i, n + j)] = a[(j, i)] * dt;
        }
    }
    let em = expm(&m);
    let mut psi12 = mat_zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            psi12[(i, j)] = em[(i, n + j)];
        }
    }
    let mut q = &f * &psi12;
    for i in 0..n {
        for j in i..n {
            let v = 0.5 * (q[(i, j)] + q[(j, i)]);
            q[(i, j)] = v;
            q[(j, i)] = v;
        }
    }
    Ok((f, u, q))
}

/// `∫_0^{Δt} exp(A s) x ds` via the affine block exponential.
pub fn integrate_expm(a: &Mat<f64>, x: &Col<f64>, dt: f64) -> Result<Col<f64>> {
    let n = a.nrows();
    if a.ncols() != n || x.nrows() != n {
        return Err(Error::dim("integrate_expm: dimension mismatch"));
    }
    if !dt.is_finite() || dt < 0.0 {
        return Err(Error::param("integrate_expm: dt must be finite and ≥ 0"));
    }
    if dt == 0.0 {
        return Ok(col_zeros(n));
    }
    Ok(affine_expm_step(a, x, dt).1)
}

fn affine_expm_step(a: &Mat<f64>, b: &Col<f64>, dt: f64) -> (Mat<f64>, Col<f64>) {
    let n = a.nrows();
    let mut au = mat_zeros(n + 1, n + 1);
    for i in 0..n {
        for j in 0..n {
            au[(i, j)] = a[(i, j)] * dt;
        }
        au[(i, n)] = b[i] * dt;
    }
    let eu = expm(&au);
    let mut f = mat_zeros(n, n);
    let mut u = col_zeros(n);
    for i in 0..n {
        for j in 0..n {
            f[(i, j)] = eu[(i, j)];
        }
        u[i] = eu[(i, n)];
    }
    (f, u)
}

/// Regularize a symmetric matrix by adding `ε I` until Cholesky succeeds.
pub fn spd_regularize(mut a: Mat<f64>, eps0: f64) -> Result<Mat<f64>> {
    let n = a.nrows();
    let mut eps = eps0;
    for _ in 0..12 {
        if a.llt(Side::Lower).is_ok() {
            return Ok(a);
        }
        for i in 0..n {
            a[(i, i)] += eps;
        }
        eps *= 10.0;
    }
    Err(Error::numeric("unable to regularize matrix to SPD"))
}

/// Diagnostics from a thin QR / ridge least-squares solve.
#[derive(Clone, Debug)]
pub struct LeastSquaresFit {
    pub beta: Col<f64>,
    pub residual_norm: f64,
    pub rank: usize,
    pub condition: f64,
    pub used_ridge: bool,
    pub ridge: f64,
}

/// Householder thin QR least squares for `min ‖A x − b‖₂`.
///
/// Columns are scaled to unit Euclidean norm before the factorization.
/// If the (scaled) R-diagonal reveals rank deficiency, a ridge
/// `λ = ε · ‖A‖_F²` is added on the normal equations as a fallback.
pub fn qr_least_squares(a: &Mat<f64>, b: &Col<f64>, ridge: Option<f64>) -> Result<LeastSquaresFit> {
    let m = a.nrows();
    let n = a.ncols();
    if b.nrows() != m {
        return Err(Error::dim("least squares: A and b row mismatch"));
    }
    if m == 0 || n == 0 {
        return Err(Error::dim("least squares: empty design"));
    }
    // Column scaling. Keep `scaled` intact for the ridge fallback; Householder
    // overwrites `work`.
    let mut scale = vec![1.0; n];
    let mut scaled = a.clone();
    for j in 0..n {
        let mut nrm = 0.0;
        for i in 0..m {
            nrm += scaled[(i, j)] * scaled[(i, j)];
        }
        nrm = nrm.sqrt();
        if nrm > 1e-15 {
            scale[j] = nrm;
            for i in 0..m {
                scaled[(i, j)] /= nrm;
            }
        } else {
            scale[j] = 1.0;
        }
    }
    let mut work = scaled.clone();
    let mut rdiag = vec![0.0; n];
    let kmax = n.min(m);
    let mut qtb = b.clone();
    for k in 0..kmax {
        // Householder reflector for column k from row k down.
        let mut nrm = 0.0;
        for i in k..m {
            nrm += work[(i, k)] * work[(i, k)];
        }
        nrm = nrm.sqrt();
        if nrm < 1e-14 {
            rdiag[k] = 0.0;
            continue;
        }
        let sign = if work[(k, k)] >= 0.0 { 1.0 } else { -1.0 };
        let u1 = work[(k, k)] + sign * nrm;
        let mut beta = 1.0;
        work[(k, k)] = -sign * nrm;
        rdiag[k] = work[(k, k)];
        for i in (k + 1)..m {
            work[(i, k)] /= u1;
            beta += work[(i, k)] * work[(i, k)];
        }
        let inv = 2.0 / beta;
        for j in (k + 1)..n {
            let mut dot = work[(k, j)];
            for i in (k + 1)..m {
                dot += work[(i, k)] * work[(i, j)];
            }
            dot *= inv;
            work[(k, j)] -= dot;
            for i in (k + 1)..m {
                work[(i, j)] -= work[(i, k)] * dot;
            }
        }
        let mut dotb = qtb[k];
        for i in (k + 1)..m {
            dotb += work[(i, k)] * qtb[i];
        }
        dotb *= inv;
        qtb[k] -= dotb;
        for i in (k + 1)..m {
            qtb[i] -= work[(i, k)] * dotb;
        }
    }
    let mut rank = 0usize;
    let mut rmin = f64::INFINITY;
    let mut rmax = 0.0_f64;
    for k in 0..kmax {
        let d = rdiag[k].abs();
        if d > 1e-12 {
            rank += 1;
            rmin = rmin.min(d);
            rmax = rmax.max(d);
        }
    }
    let cond = if rmin > 0.0 && rmin.is_finite() {
        rmax / rmin
    } else {
        f64::INFINITY
    };
    // `ridge` is the fallback ε used only when R is rank-deficient. Passing
    // `Some(ε)` must not force Tikhonov on a well-conditioned design — that
    // path used to rebuild AᵀA from the destroyed Householder workspace.
    let singular = rank < n || !cond.is_finite() || cond > 1e10;
    let mut used_ridge = singular;
    let lambda = if singular {
        ridge.unwrap_or(1e-8) * (1.0 + frobenius(a) * frobenius(a))
    } else {
        0.0
    };
    let mut beta = col_zeros(n);
    if !used_ridge {
        // Back-substitution on R (upper, stored in work + rdiag).
        for i in (0..kmax).rev() {
            if rdiag[i].abs() < 1e-14 {
                used_ridge = true;
                break;
            }
            let mut s = qtb[i];
            for j in (i + 1)..n {
                s -= work[(i, j)] * beta[j];
            }
            beta[i] = s / rdiag[i];
        }
    }
    if used_ridge {
        // (AᵀA + λI) x = Aᵀb on the *scaled* columns, then unscale.
        let mut ata = mat_zeros(n, n);
        let mut atb = col_zeros(n);
        for i in 0..n {
            for j in i..n {
                let mut s = 0.0;
                for p in 0..m {
                    s += scaled[(p, i)] * scaled[(p, j)];
                }
                if i == j {
                    s += lambda;
                }
                ata[(i, j)] = s;
                ata[(j, i)] = s;
            }
            let mut s = 0.0;
            for p in 0..m {
                s += scaled[(p, i)] * b[p];
            }
            atb[i] = s;
        }
        beta = solve_spd(&ata, &atb)?;
    }
    for j in 0..n {
        beta[j] /= scale[j];
    }
    let ax = a * &beta;
    let mut r2 = 0.0;
    for i in 0..m {
        let d = ax[i] - b[i];
        r2 += d * d;
    }
    Ok(LeastSquaresFit {
        beta,
        residual_norm: r2.sqrt(),
        rank,
        condition: cond,
        used_ridge,
        ridge: lambda,
    })
}

fn frobenius(a: &Mat<f64>) -> f64 {
    let mut s = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            s += a[(i, j)] * a[(i, j)];
        }
    }
    s.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logdet_identity() {
        let a = mat_identity(3);
        assert!((logdet_spd(&a).unwrap()).abs() < 1e-14);
    }

    #[test]
    fn expm_zero_is_identity() {
        let a = mat_zeros(2, 2);
        let e = expm(&a);
        assert!((e[(0, 0)] - 1.0).abs() < 1e-14);
        assert!(e[(0, 1)].abs() < 1e-14);
    }

    #[test]
    fn hurwitz_scalar() {
        let a = mat_from_row_slice(1, 1, &[-1.5]);
        assert!(is_hurwitz(&a));
        let b = mat_from_row_slice(1, 1, &[0.2]);
        assert!(!is_hurwitz(&b));
    }

    #[test]
    fn try_inverse_rejects_singular() {
        let z = mat_zeros(2, 2);
        assert!(try_inverse(&z).is_none());
        let i = mat_identity(2);
        let inv = try_inverse(&i).unwrap();
        assert!((inv[(0, 0)] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn solve_lower_rejects_zero_pivot() {
        let l = mat_from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        let b = col_from_slice(&[1.0, 1.0]);
        assert!(solve_lower(&l, &b).is_err());
    }

    #[test]
    fn van_loan_ou_and_nilpotent() {
        // Scalar OU: A=−κ, b=0, G=σ, Q = σ²(1−e^{−2κΔ})/(2κ)
        let a = mat_from_row_slice(1, 1, &[-1.2]);
        let b = col_from_slice(&[0.0]);
        let g = mat_from_row_slice(1, 1, &[0.5]);
        let (_f, _u, q) = van_loan_discretize(&a, &b, &g, 1.0).unwrap();
        let exact = 0.25 * (1.0 - (-2.4_f64).exp()) / 2.4;
        assert!(
            (q[(0, 0)] - exact).abs() < 1e-10,
            "Q {} vs {exact}",
            q[(0, 0)]
        );
        // Nilpotent double integrator: A=[[0,1],[0,0]], b=[0,1]
        let a2 = mat_from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b2 = col_from_slice(&[0.0, 1.0]);
        let g2 = mat_zeros(2, 1);
        let dt = 1.0;
        let (f2, u2, _) = van_loan_discretize(&a2, &b2, &g2, dt).unwrap();
        assert!((f2[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((f2[(0, 1)] - dt).abs() < 1e-12);
        assert!((u2[0] - 0.5 * dt * dt).abs() < 1e-12);
        assert!((u2[1] - dt).abs() < 1e-12);
    }

    #[test]
    fn qr_least_squares_exact_and_ridge() {
        // 3 x = (1,2,3) is exact for A = I stacked.
        let a = mat_from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let b = col_from_slice(&[1.0, 2.0, 3.0]);
        let fit = qr_least_squares(&a, &b, None).unwrap();
        assert_eq!(fit.rank, 2);
        assert!((fit.beta[0] - 1.0).abs() < 1e-10);
        assert!((fit.beta[1] - 2.0).abs() < 1e-10);
        // Duplicate columns: rank 1, ridge fallback.
        let sing = mat_from_row_slice(3, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
        let fit2 = qr_least_squares(&sing, &b, None).unwrap();
        assert!(fit2.used_ridge);
        assert!(fit2.residual_norm.is_finite());
        // A well-conditioned line with Some(ε) must still use QR, not ridge.
        let a3 = mat_from_row_slice(4, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let b3 = col_from_slice(&[1.0, 4.0, 7.0, 10.0]);
        let fit3 = qr_least_squares(&a3, &b3, Some(1e-8)).unwrap();
        assert!(!fit3.used_ridge);
        assert!((fit3.beta[0] - 1.0).abs() < 1e-10);
        assert!((fit3.beta[1] - 3.0).abs() < 1e-10);
    }
}
