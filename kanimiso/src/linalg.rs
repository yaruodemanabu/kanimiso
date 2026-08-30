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
pub struct ThinSvd {
    /// Left singular vectors (n × r).
    pub u: Mat<f64>,
    /// Singular values, nonincreasing, length r.
    pub singular_values: Vec<f64>,
    /// Right singular vectors (p × r).
    pub v: Mat<f64>,
}

impl ThinSvd {
    /// Numerical rank at `tol * σ_max`.
    pub fn rank(&self, rel_tol: f64) -> usize {
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
    pub fn condition_number(&self) -> f64 {
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
pub fn thin_svd(report: &mut Report, x: &Matrix, policy: &Policy) -> Option<ThinSvd> {
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
pub fn least_squares(
    report: &mut Report,
    x: &Matrix,
    y: &Vector,
    policy: &Policy,
) -> Option<Vector> {
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
    if let Some(issue) = condition_issue(kappa, policy) {
        report.push_with_policy(policy.clone(), issue);
    }
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
    let rel = resid.norm() / (1.0 + y.norm());
    if rel > policy.residual_tol && rank == p && n >= p {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::ResidualTooLarge)
                .message(format!(
                    "relative residual {rel:.4e} > {}",
                    policy.residual_tol
                ))
                .metric("relative_residual", rel)
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
    Some(beta)
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
pub fn ridge_solve(
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
pub fn symmetric_eigen(
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
pub fn chol_solve(
    report: &mut Report,
    a: &Mat<f64>,
    b: &Vector,
    policy: &Policy,
) -> Option<Vector> {
    match a.llt(Side::Lower) {
        Ok(chol) => {
            let rhs = b.to_matrix();
            let sol = chol.solve(rhs.inner());
            let mut x = Vector::zeros(b.len());
            for i in 0..b.len() {
                x[i] = sol[(i, 0)];
            }
            Some(x)
        }
        Err(_) => {
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::CholeskyFailed)
                    .message("requested SPD solve but Cholesky refused the matrix")
                    .build(),
            );
            None
        }
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
}
