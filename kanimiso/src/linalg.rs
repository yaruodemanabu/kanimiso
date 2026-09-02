//! `faer`-backed linear algebra with a [`signlred`] quality contract.

use crate::data::{Matrix, Vector};
use faer::linalg::solvers::{Solve, SolveLstsq};
use faer::{Mat, Side};
use signlred::{
    classify_condition_number, condition_issue, Issue, IssueCode, NumericalCompromise, Policy,
    RankHint, Report,
};

/// Thin SVD factors used by PCA / ridge / pseudoinverse.
#[derive(Clone, Debug)]
pub(crate) struct ThinSvd {
    /// Left singular vectors (n × r).
    pub u: Mat<f64>,
    /// Singular values, nonincreasing, length r.
    pub singular_values: Vec<f64>,
    /// Right singular vectors (p × r).
    pub v: Mat<f64>,
}

/// Least-squares coefficients and the decomposition that certifies them.
#[derive(Clone, Debug)]
pub(crate) struct LeastSquaresSolution {
    pub(crate) coefficients: Vector,
    pub(crate) decomposition: ThinSvd,
    pub(crate) rank: usize,
}

/// Result of one Cholesky factorization with one or more right-hand sides.
#[derive(Clone, Debug)]
pub(crate) struct SpdSolveSolution {
    pub(crate) solution: Matrix,
    pub(crate) log_determinant: f64,
}

impl ThinSvd {
    /// Numerical rank at `tol * σ_max`.
    pub(crate) fn rank(&self, rel_tol: f64) -> usize {
        let max = self.singular_values.first().copied().unwrap_or(0.0);
        if max <= 0.0 {
            return 0;
        }
        self.singular_values
            .iter()
            .filter(|s| **s > rel_tol * max)
            .count()
    }

    /// κ = σ_max / σ_min (∞ if rank 0).
    pub(crate) fn condition_number(&self) -> f64 {
        let max = self.singular_values.first().copied().unwrap_or(0.0);
        let min = self.singular_values.last().copied().unwrap_or(0.0);
        if max <= 0.0 || min <= 0.0 {
            f64::INFINITY
        } else {
            max / min
        }
    }
}

/// Compute the thin SVD of `x`, recording non-convergence.
pub(crate) fn thin_svd(report: &mut Report, x: &Matrix, policy: &Policy) -> Option<ThinSvd> {
    match x.inner().thin_svd() {
        Ok(svd) => {
            let u = svd.U().to_owned();
            let v = svd.V().to_owned();
            let s = match x.inner().singular_values() {
                Ok(s) => s,
                Err(_) => {
                    report.push_with_policy(
                        policy.clone(),
                        Issue::builder(IssueCode::SvdDidNotConverge)
                            .message("singular values failed after a successful thin SVD")
                            .build(),
                    );
                    return None;
                }
            };
            Some(ThinSvd {
                u,
                singular_values: s,
                v,
            })
        }
        Err(_) => {
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::SvdDidNotConverge)
                    .message("faer thin SVD did not converge")
                    .build(),
            );
            None
        }
    }
}

/// Least-squares `min ‖Xβ − y‖₂` with rank / condition diagnostics.
pub(crate) fn least_squares(
    report: &mut Report,
    x: &Matrix,
    y: &Vector,
    policy: &Policy,
) -> Option<Vector> {
    least_squares_with_diagnostics(report, x, y, policy).map(|solution| solution.coefficients)
}

/// Least-squares solve retaining the SVD used for rank-aware diagnostics.
pub(crate) fn least_squares_with_diagnostics(
    report: &mut Report,
    x: &Matrix,
    y: &Vector,
    policy: &Policy,
) -> Option<LeastSquaresSolution> {
    let (n, p) = x.shape();
    if y.len() != n {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!("lstsq y.len()={} X is {n}×{p}", y.len()))
                .build(),
        );
        return None;
    }
    if n == 0 || p == 0 {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::EmptyMatrix)
                .message("lstsq on an empty design")
                .build(),
        );
        return None;
    }
    if n < p {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::UnderdeterminedSystem)
                .message(format!(
                    "n={n} < p={p}: infinite OLS solutions; min-norm SVD used"
                ))
                .compromise(NumericalCompromise::new(
                    "unique OLS solution of a full-column-rank tall system",
                    "thin-SVD minimum-norm least squares",
                    "the design is fat or square-deficient",
                    "coefficients are a particular solution; they are not the unique OLS estimand",
                ))
                .build(),
        );
    }

    let svd = thin_svd(report, x, policy)?;
    let kappa = svd.condition_number();
    let rank = svd.rank(policy.rank_tol_relative);

    if rank == 0 {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::RankZero)
                .message("X is the zero operator at working precision")
                .metric("condition_number", kappa)
                .build(),
        );
        return None;
    }
    if rank < p {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::RankDeficient)
                .message(format!("numerical rank {rank} < p={p}"))
                .metric("rank", rank as f64)
                .metric("p", p as f64)
                .compromise(NumericalCompromise::new(
                    "full-column-rank OLS",
                    format!("truncated SVD pseudoinverse at rank {rank}"),
                    "one or more singular values are below the relative cutoff",
                    "coefficients in the numerical null space are unidentified",
                ))
                .build(),
        );
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::PseudoinverseUsed)
                .message("Moore–Penrose / truncated inverse substituted for (X'X)⁻¹")
                .compromise(NumericalCompromise::new(
                    "(XᵀX)⁻¹ Xᵀ y",
                    "Σ_{i=1..r} (u_iᵀ y / σ_i) v_i",
                    "XᵀX is singular at working precision",
                    "do not interpret dropped-direction coefficients; they were set to the min-norm choice 0",
                ))
                .metric("rank", rank as f64)
                .build(),
        );
    } else if let Some(issue) = condition_issue(kappa, policy) {
        // Full numerical rank: κ documents ill-conditioning. Do not emit
        // RankZero/NearSingular from an infinite κ when rank < p — that is
        // already recorded as RankDeficient + PseudoinverseUsed.
        report.push_with_policy(policy.clone(), issue);
    }

    let hint = classify_condition_number(kappa, policy);
    let beta = match hint {
        RankHint::Full | RankHint::Ill if n >= p && rank == p => {
            let qr = x.inner().qr();
            let rhs = y.to_matrix();
            let sol = qr.solve_lstsq(rhs.inner());
            let mut b = Vector::zeros(p);
            for j in 0..p {
                b[j] = sol[(j, 0)];
            }
            if matches!(hint, RankHint::Ill) {
                report.push_with_policy(
                    policy.clone(),
                    Issue::builder(IssueCode::IllConditioned)
                        .message(format!("QR lstsq with κ={kappa:.4e}"))
                        .metric("condition_number", kappa)
                        .build(),
                );
            }
            b
        }
        _ => svd_solve(&svd, y, rank),
    };

    let pred = x.matvec(&beta);
    let resid = y.sub(&pred);
    let stationarity = relative_least_squares_stationarity(x, y, &pred, &resid);
    if stationarity > policy.residual_tol && rank == p && n >= p {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::ResidualTooLarge)
                .message(format!(
                    "relative normal-equation residual {stationarity:.4e} > {}",
                    policy.residual_tol
                ))
                .metric("relative_normal_residual", stationarity)
                .build(),
        );
    }
    if !beta.as_slice().iter().all(|v| v.is_finite()) {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::NonFiniteOutput)
                .message("least-squares coefficients contain NaN/Inf")
                .build(),
        );
        return None;
    }
    Some(LeastSquaresSolution {
        coefficients: beta,
        decomposition: svd,
        rank,
    })
}

/// Scale-free first-order optimality residual for `min ||X beta - y||_2`.
///
/// A statistical least-squares residual is generally nonzero. The quantity
/// that must be small at a minimizer is `X^T residual`. Scaling both operands
/// before the dot products avoids overflowing this diagnostic for otherwise
/// finite inputs.
fn relative_least_squares_stationarity(
    x: &Matrix,
    y: &Vector,
    fitted: &Vector,
    residual: &Vector,
) -> f64 {
    let mut x_scale = 0.0_f64;
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            x_scale = x_scale.max(x.get(i, j).abs());
        }
    }
    let response_scale = y.max_abs().max(fitted.max_abs()).max(residual.max_abs());
    if x_scale == 0.0 || response_scale == 0.0 {
        return 0.0;
    }

    let mut x_norm_squared = 0.0;
    let mut y_norm_squared = 0.0;
    let mut fitted_norm_squared = 0.0;
    for i in 0..x.nrows() {
        let scaled_y = y[i] / response_scale;
        let scaled_fitted = fitted[i] / response_scale;
        y_norm_squared += scaled_y * scaled_y;
        fitted_norm_squared += scaled_fitted * scaled_fitted;
        for j in 0..x.ncols() {
            let scaled_x = x.get(i, j) / x_scale;
            x_norm_squared += scaled_x * scaled_x;
        }
    }
    let mut gradient_norm_squared = 0.0;
    for j in 0..x.ncols() {
        let mut component = 0.0;
        for i in 0..x.nrows() {
            component += (x.get(i, j) / x_scale) * (residual[i] / response_scale);
        }
        gradient_norm_squared += component * component;
    }
    let denominator = x_norm_squared.sqrt() * (y_norm_squared.sqrt() + fitted_norm_squared.sqrt());
    if denominator > 0.0 {
        gradient_norm_squared.sqrt() / denominator
    } else if gradient_norm_squared == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

fn svd_solve(svd: &ThinSvd, y: &Vector, rank: usize) -> Vector {
    let r = svd.singular_values.len().min(svd.u.ncols()).min(rank);
    let p = svd.v.nrows();
    let mut beta = Vector::zeros(p);
    for k in 0..r {
        let sigma = svd.singular_values[k];
        if sigma <= 0.0 {
            continue;
        }
        let mut uty = 0.0;
        for i in 0..y.len().min(svd.u.nrows()) {
            uty += svd.u[(i, k)] * y[i];
        }
        let scale = uty / sigma;
        for j in 0..p {
            beta[j] += svd.v[(j, k)] * scale;
        }
    }
    beta
}

/// Ridge: `(XᵀX + λI)β = Xᵀy`. Falls back to SVD if Cholesky fails.
pub(crate) fn ridge_solve(
    report: &mut Report,
    x: &Matrix,
    y: &Vector,
    lambda: f64,
    policy: &Policy,
) -> Option<Vector> {
    if !lambda.is_finite() || lambda < 0.0 {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!(
                    "ridge λ={lambda} is not a finite non-negative number"
                ))
                .build(),
        );
        return None;
    }
    let (n, p) = x.shape();
    if y.len() != n {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::DimensionMismatch)
                .message("ridge y length ≠ n")
                .build(),
        );
        return None;
    }
    let mut gram = x.gram();
    for i in 0..p {
        gram[(i, i)] += lambda;
    }
    match gram.llt(Side::Lower) {
        Ok(chol) => {
            let rhs = x.matvec_t(y).to_matrix();
            let sol = chol.solve(&rhs.inner());
            let mut b = Vector::zeros(p);
            for j in 0..p {
                b[j] = sol[(j, 0)];
            }
            if !b.as_slice().iter().all(|v| v.is_finite()) {
                report.push_with_policy(
                    policy.clone(),
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message("ridge coefficients non-finite")
                        .build(),
                );
                return None;
            }
            Some(b)
        }
        Err(_) => {
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::CholeskyFailed)
                    .message("ridge Gram Cholesky failed; jitter then SVD fallback")
                    .build(),
            );
            for i in 0..p {
                gram[(i, i)] += policy.rank_tol_relative.max(1e-12);
            }
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::JitterInjected)
                    .message("added diagonal jitter after Cholesky failure")
                    .compromise(NumericalCompromise::new(
                        format!("Cholesky(XᵀX + {lambda} I)"),
                        "jittered Gram or SVD ridge",
                        "the regularized Gram was not SPD at working precision",
                        "the estimand is a slightly different ridge; SEs from (XᵀX+λI)⁻¹ are approximate",
                    ))
                    .build(),
            );
            match gram.llt(Side::Lower) {
                Ok(chol) => {
                    let rhs = x.matvec_t(y).to_matrix();
                    let sol = chol.solve(&rhs.inner());
                    let mut b = Vector::zeros(p);
                    for j in 0..p {
                        b[j] = sol[(j, 0)];
                    }
                    Some(b)
                }
                Err(_) => {
                    report.push_with_policy(
                        policy.clone(),
                        Issue::builder(IssueCode::RidgeFallbackUsed)
                            .message("SVD ridge: β = V (σ²/(σ²+λ)) Σ⁺ Uᵀ y")
                            .compromise(NumericalCompromise::new(
                                "Cholesky ridge",
                                "filtered SVD ridge",
                                "Gram remained non-SPD after jitter",
                                "same ridge estimand in exact arithmetic; filtering is the stable form",
                            ))
                            .build(),
                    );
                    svd_ridge(report, x, y, lambda, policy)
                }
            }
        }
    }
}

fn svd_ridge(
    report: &mut Report,
    x: &Matrix,
    y: &Vector,
    lambda: f64,
    policy: &Policy,
) -> Option<Vector> {
    let svd = thin_svd(report, x, policy)?;
    let p = svd.v.nrows();
    let mut beta = Vector::zeros(p);
    let r = svd.singular_values.len().min(svd.u.ncols());
    for k in 0..r {
        let sigma = svd.singular_values[k];
        let filt = (sigma * sigma) / (sigma * sigma + lambda);
        if !filt.is_finite() {
            continue;
        }
        let mut uty = 0.0;
        for i in 0..y.len().min(svd.u.nrows()) {
            uty += svd.u[(i, k)] * y[i];
        }
        let scale = if sigma > 0.0 { filt * uty / sigma } else { 0.0 };
        for j in 0..p {
            beta[j] += svd.v[(j, k)] * scale;
        }
    }
    Some(beta)
}

/// Symmetric eigendecomposition of a p×p matrix (lower triangle read).
pub(crate) fn symmetric_eigen(
    report: &mut Report,
    a: &Mat<f64>,
    policy: &Policy,
) -> Option<(Vec<f64>, Mat<f64>)> {
    match a.self_adjoint_eigen(Side::Lower) {
        Ok(ev) => {
            let u = ev.U().to_owned();
            let vals = match a.self_adjoint_eigenvalues(Side::Lower) {
                Ok(v) => v,
                Err(_) => {
                    report.push_with_policy(
                        policy.clone(),
                        Issue::builder(IssueCode::EigenDidNotConverge).build(),
                    );
                    return None;
                }
            };
            Some((vals, u))
        }
        Err(_) => {
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::EigenDidNotConverge)
                    .message("self-adjoint eigensolver failed")
                    .build(),
            );
            None
        }
    }
}

/// SPD solve `A x = b` via Cholesky; records failure.
pub(crate) fn chol_solve(
    report: &mut Report,
    a: &Mat<f64>,
    b: &Vector,
    policy: &Policy,
) -> Option<Vector> {
    let rhs = b.to_matrix();
    let solved = chol_solve_matrix(report, a, &rhs, policy)?;
    Some(solved.solution.column(0))
}

/// SPD solve `A X = B` with one factorization, its log determinant, and a
/// scale-free solve-residual diagnostic.
pub(crate) fn chol_solve_matrix(
    report: &mut Report,
    a: &Mat<f64>,
    b: &Matrix,
    policy: &Policy,
) -> Option<SpdSolveSolution> {
    chol_solve_matrix_with_context(report, a, b, policy, "SPD solve")
}

/// Contextual SPD solve used by algorithms that must identify which system
/// failed.  The factorization is performed once, and a single augmented solve
/// handles both the caller's right-hand sides and an identity matrix.  The
/// identity block supplies an inverse-norm condition estimate without a second
/// factorization or solve call.
pub(crate) fn chol_solve_matrix_with_context(
    report: &mut Report,
    a: &Mat<f64>,
    b: &Matrix,
    policy: &Policy,
    context: &str,
) -> Option<SpdSolveSolution> {
    if a.nrows() == 0 || a.nrows() != a.ncols() || b.nrows() != a.nrows() {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "{context}: requires square non-empty A and B.nrows=A.nrows; A={}x{}, B={}x{}",
                    a.nrows(),
                    a.ncols(),
                    b.nrows(),
                    b.ncols()
                ))
                .build(),
        );
        return None;
    }
    if !faer_matrix_is_finite(a) || !matrix_is_finite(b) {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::NonFiniteInput)
                .message(format!(
                    "{context}: coefficient matrix or right-hand side contains NaN/infinity"
                ))
                .build(),
        );
        return None;
    }
    let chol = match a.llt(Side::Lower) {
        Ok(chol) => chol,
        Err(_) => {
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::CholeskyFailed)
                    .message(format!(
                        "{context}: Cholesky refused the requested {}x{} SPD system",
                        a.nrows(),
                        a.ncols()
                    ))
                    .build(),
            );
            return None;
        }
    };
    let dimension = a.nrows();
    let rhs_columns = b.ncols();
    let augmented_rhs = Matrix::from_fn(dimension, rhs_columns + dimension, |i, j| {
        if j < rhs_columns {
            b.get(i, j)
        } else if i == j - rhs_columns {
            1.0
        } else {
            0.0
        }
    });
    let augmented_solution = Matrix::from_faer(chol.solve(augmented_rhs.inner()));
    if !matrix_is_finite(&augmented_solution) {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::NonFiniteOutput)
                .message(format!(
                    "{context}: Cholesky solve produced NaN or infinity"
                ))
                .build(),
        );
        return None;
    }
    let solution = Matrix::from_fn(dimension, rhs_columns, |i, j| augmented_solution.get(i, j));
    let inverse = Matrix::from_fn(dimension, dimension, |i, j| {
        augmented_solution.get(i, rhs_columns + j)
    });

    let mut log_determinant = 0.0;
    let lower = chol.L();
    for i in 0..a.nrows() {
        let diagonal = lower[(i, i)];
        if !diagonal.is_finite() || diagonal <= 0.0 {
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::NonPositiveDefinite)
                    .message(format!(
                        "{context}: Cholesky factor has a non-positive or non-finite diagonal at index {i}"
                    ))
                    .build(),
            );
            return None;
        }
        log_determinant += 2.0 * diagonal.ln();
    }
    if !log_determinant.is_finite() {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::NonFiniteOutput)
                .message(format!(
                    "{context}: accumulated log determinant is {log_determinant}"
                ))
                .build(),
        );
        return None;
    }

    let condition_number = faer_matrix_infinity_norm(a) * matrix_infinity_norm(&inverse);
    if let Some(mut issue) = condition_issue(condition_number, policy) {
        issue.message = format!("{context}: {}", issue.message);
        report.push_with_policy(policy.clone(), issue);
    }

    let residual = relative_square_solve_residual(a, &augmented_solution, &augmented_rhs);
    if !residual.is_finite() || residual > policy.residual_tol {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::ResidualTooLarge)
                .message(format!(
                    "{context}: relative augmented SPD solve residual {residual:.4e} > {}",
                    policy.residual_tol
                ))
                .metric("relative_solve_residual", residual)
                .build(),
        );
    }
    Some(SpdSolveSolution {
        solution,
        log_determinant,
    })
}

/// Matrix product computed by faer's dense matrix kernel.
pub(crate) fn matrix_multiply(left: &Matrix, right: &Matrix) -> Matrix {
    assert_eq!(left.ncols(), right.nrows());
    Matrix::from_faer(left.inner() * right.inner())
}

/// Elementwise matrix sum computed by faer.
pub(crate) fn matrix_add(left: &Matrix, right: &Matrix) -> Matrix {
    assert_eq!(left.shape(), right.shape());
    Matrix::from_faer(left.inner() + right.inner())
}

/// Elementwise matrix difference computed by faer.
pub(crate) fn matrix_subtract(left: &Matrix, right: &Matrix) -> Matrix {
    assert_eq!(left.shape(), right.shape());
    Matrix::from_faer(left.inner() - right.inner())
}

/// Owned transpose computed through faer's matrix view.
pub(crate) fn matrix_transpose(matrix: &Matrix) -> Matrix {
    Matrix::from_faer(matrix.inner().transpose().to_owned())
}

/// Average a square matrix with its transpose without overflowing finite pairs.
pub(crate) fn matrix_symmetrized(matrix: &Matrix) -> Matrix {
    assert_eq!(matrix.nrows(), matrix.ncols());
    Matrix::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
        if i == j {
            return matrix.get(i, i);
        }
        // Read each mirrored pair in one canonical order so both output
        // entries are bit-identical.  Halving opposite signs before adding
        // avoids overflow; for equal signs, the difference cannot overflow
        // and preserves tiny equal values without a halving underflow.
        let (first, second) = if i > j {
            (matrix.get(i, j), matrix.get(j, i))
        } else {
            (matrix.get(j, i), matrix.get(i, j))
        };
        let (lower, upper) = if first <= second {
            (first, second)
        } else {
            (second, first)
        };
        if lower == upper {
            lower
        } else if lower.is_sign_positive() == upper.is_sign_positive() {
            lower + (upper - lower) * 0.5
        } else {
            lower * 0.5 + upper * 0.5
        }
    })
}

/// Whether every entry of a dense matrix is finite.
pub(crate) fn matrix_is_finite(matrix: &Matrix) -> bool {
    (0..matrix.nrows()).all(|i| (0..matrix.ncols()).all(|j| matrix.get(i, j).is_finite()))
}

/// Whether every entry of a dense vector is finite.
pub(crate) fn vector_is_finite(vector: &Vector) -> bool {
    vector.as_slice().iter().all(|value| value.is_finite())
}

/// Copy a vector into one row of a dense matrix.
pub(crate) fn matrix_write_row(matrix: &mut Matrix, row: usize, values: &Vector) {
    assert!(row < matrix.nrows());
    assert_eq!(values.len(), matrix.ncols());
    for column in 0..values.len() {
        matrix.inner_mut()[(row, column)] = values[column];
    }
}

fn faer_matrix_is_finite(matrix: &Mat<f64>) -> bool {
    (0..matrix.nrows()).all(|i| (0..matrix.ncols()).all(|j| matrix[(i, j)].is_finite()))
}

fn faer_matrix_infinity_norm(matrix: &Mat<f64>) -> f64 {
    (0..matrix.nrows())
        .map(|i| (0..matrix.ncols()).map(|j| matrix[(i, j)].abs()).sum())
        .fold(0.0, f64::max)
}

fn matrix_infinity_norm(matrix: &Matrix) -> f64 {
    (0..matrix.nrows())
        .map(|i| (0..matrix.ncols()).map(|j| matrix.get(i, j).abs()).sum())
        .fold(0.0, f64::max)
}

fn relative_square_solve_residual(a: &Mat<f64>, x: &Matrix, b: &Matrix) -> f64 {
    let mut a_scale = 0.0_f64;
    let mut x_scale = 0.0_f64;
    let mut b_scale = 0.0_f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            a_scale = a_scale.max(a[(i, j)].abs());
        }
    }
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            x_scale = x_scale.max(x.get(i, j).abs());
            b_scale = b_scale.max(b.get(i, j).abs());
        }
    }
    if a_scale == 0.0 || x_scale == 0.0 {
        return if b_scale == 0.0 { 0.0 } else { 1.0 };
    }
    let product_log_scale = a_scale.ln() + x_scale.ln();
    let rhs_log_scale = if b_scale > 0.0 {
        b_scale.ln()
    } else {
        f64::NEG_INFINITY
    };
    let common_log_scale = product_log_scale.max(rhs_log_scale);
    let product_factor = (product_log_scale - common_log_scale).exp();
    let rhs_factor = if b_scale > 0.0 {
        (rhs_log_scale - common_log_scale).exp()
    } else {
        0.0
    };
    let mut residual_squared = 0.0;
    let mut a_squared = 0.0;
    let mut x_squared = 0.0;
    let mut b_squared = 0.0;
    for i in 0..a.nrows() {
        for j in 0..x.ncols() {
            let mut product = 0.0;
            for k in 0..a.ncols() {
                product += (a[(i, k)] / a_scale) * (x.get(k, j) / x_scale);
            }
            let scaled_product = product * product_factor;
            let scaled_b = if b_scale > 0.0 {
                (b.get(i, j) / b_scale) * rhs_factor
            } else {
                0.0
            };
            let difference = scaled_product - scaled_b;
            residual_squared += difference * difference;
            b_squared += scaled_b * scaled_b;
        }
    }
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let value = a[(i, j)] / a_scale;
            a_squared += value * value;
        }
    }
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            let value = x.get(i, j) / x_scale;
            x_squared += value * value;
        }
    }
    let product_norm = a_squared.sqrt() * x_squared.sqrt() * product_factor;
    let denominator = product_norm + b_squared.sqrt();
    if denominator > 0.0 {
        residual_squared.sqrt() / denominator
    } else if residual_squared == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use signlred::Report;

    #[test]
    fn lstsq_recovers_line() {
        // y = 2 + 3x
        let x = Matrix::from_fn(5, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let y = Vector::from_iter((0..5).map(|i| 2.0 + 3.0 * i as f64));
        let mut report = Report::new("ols", "fit");
        let policy = Policy::default();
        let b = least_squares(&mut report, &x, &y, &policy).expect("lstsq");
        assert!((b[0] - 2.0).abs() < 1e-8, "{:?}", b.as_slice());
        assert!((b[1] - 3.0).abs() < 1e-8, "{:?}", b.as_slice());
    }

    #[test]
    fn lstsq_accepts_noise_when_the_normal_equations_are_solved() {
        let x = Matrix::from_fn(9, 2, |i, j| if j == 0 { 1.0 } else { i as f64 - 4.0 });
        let y = Vector::from_slice(&[1.4, 0.1, 2.2, 1.0, 3.3, 4.8, 4.1, 6.7, 5.9]);
        let mut report = Report::new("noisy-ols", "fit");
        let policy = Policy::default();
        let beta = least_squares(&mut report, &x, &y, &policy).expect("noisy least squares");
        let residual = y.sub(&x.matvec(&beta));
        let ordinary_relative_residual = residual.norm() / (1.0 + y.norm());
        let fitted = x.matvec(&beta);
        let stationarity = relative_least_squares_stationarity(&x, &y, &fitted, &residual);
        assert!(ordinary_relative_residual > policy.residual_tol);
        // Measured 1.296e-16 on 2026-09-03; tolerance is approximately 4x.
        assert!(stationarity <= 5.2e-16);
        assert!(!report.contains(IssueCode::ResidualTooLarge));
    }

    #[test]
    fn spd_solve_reports_condition_even_when_the_residual_is_exact() {
        let a = Mat::<f64>::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (1, 1) => 1e-12,
            _ => 0.0,
        });
        let b = Matrix::from_row_major(2, 1, &[1.0, 1e-12]);
        let policy = Policy {
            condition_number_warn: 1e10,
            condition_number_error: 1e16,
            ..Policy::default()
        };
        let mut report = Report::new("spd", "condition");
        let solved = chol_solve_matrix_with_context(
            &mut report,
            &a,
            &b,
            &policy,
            "test innovation system at time 7; dimension=2, observed_columns=[0, 3]",
        )
        .expect("diagonal SPD solve");

        assert!(matrix_is_finite(&solved.solution));
        assert!(report.contains(IssueCode::IllConditioned));
        assert!(!report.contains(IssueCode::ResidualTooLarge));
        let issue = report
            .issues()
            .iter()
            .find(|issue| issue.code == IssueCode::IllConditioned)
            .expect("condition issue");
        assert!(issue.message.contains("time 7"));
        assert!(issue.message.contains("observed_columns=[0, 3]"));
    }

    #[test]
    fn symmetrization_preserves_finite_extremes_without_intermediate_overflow() {
        let maximum_symmetric =
            Matrix::from_row_major(2, 2, &[f64::MAX, f64::MAX, f64::MAX, f64::MAX]);
        let symmetric = matrix_symmetrized(&maximum_symmetric);
        assert!(matrix_is_finite(&symmetric));
        assert_eq!(symmetric.to_row_major(), vec![f64::MAX; 4]);

        let maximum_antisymmetric =
            Matrix::from_row_major(2, 2, &[f64::MAX, f64::MAX, -f64::MAX, f64::MAX]);
        let antisymmetric = matrix_symmetrized(&maximum_antisymmetric);
        assert!(matrix_is_finite(&antisymmetric));
        assert_eq!(
            antisymmetric.to_row_major(),
            vec![f64::MAX, 0.0, 0.0, f64::MAX]
        );

        let smallest_subnormal = f64::from_bits(1);
        let tiny_symmetric = Matrix::from_row_major(
            2,
            2,
            &[
                smallest_subnormal,
                smallest_subnormal,
                smallest_subnormal,
                smallest_subnormal,
            ],
        );
        let tiny = matrix_symmetrized(&tiny_symmetric);
        assert!(tiny
            .to_row_major()
            .iter()
            .all(|value| value.to_bits() == smallest_subnormal.to_bits()));
    }
}
