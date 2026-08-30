//! Johansen cointegration and the resulting VECM (statsmodels `coint_johansen`).
//!
//! The reduced-rank regression of \(\Delta Y_t\) on \(Y_{t-1}\) is solved by
//! SVD of the whitened moment matrices, not by a silent eigen call. Rank-0
//! (no cointegration) is a valid finding; asking a VECM for \(r>0\) when every
//! eigenvalue is numerically zero is a vacuous error-correction term.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::thin_svd;
use crate::validate::{inspect_identification, inspect_xy};
use faer::Mat;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result};

/// Johansen trace / max-eigen test (constant in the cointegrating relation).
#[derive(Clone, Debug, Default)]
pub struct Johansen {
    /// Include an unrestricted intercept in \(\Delta Y\) (Case 3 / “c”).
    pub detrend: bool,
}

impl Johansen {
    /// Intercept-in-CE Johansen test.
    pub fn new() -> Self {
        Self { detrend: true }
    }

    /// Fit on an \(T \times K\) level series (columns are variables).
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedJohansen>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let (n, k) = y.shape();
        if n < 8 || k < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message(format!("Johansen needs T≥8 and K≥2; got T={n} K={k}"))
                    .meaninglessness(Meaninglessness::vacuous(
                        "Johansen eigenvalues",
                        "the residual moment matrices are unidentified at this (T, K)",
                        "collect a longer multivariate series",
                    ))
                    .build(),
            );
            return ctx.finish(empty_johansen(k, n));
        }
        inspect_identification(&mut ctx.report, n.saturating_sub(1), k, &ctx.policy);

        let tstar = n - 1;
        let mut dy = Matrix::zeros(tstar, k);
        let mut ylag = Matrix::zeros(tstar, k);
        for t in 1..n {
            for j in 0..k {
                dy.set(t - 1, j, y.get(t, j) - y.get(t - 1, j));
                ylag.set(t - 1, j, y.get(t - 1, j));
            }
        }
        if self.detrend {
            demean_cols(&mut dy);
            demean_cols(&mut ylag);
            ctx.push(
                Issue::builder(IssueCode::TruncatedSvdUsed)
                    .severity(signlred::Severity::Advisory)
                    .message("series are demeaned (Case 3: intercept in the CE and in ΔY)")
                    .compromise(NumericalCompromise::new(
                        "Johansen with a fully specified deterministic menu",
                        "column-demean ΔY and Y_{t-1} before forming Sij",
                        "trend / restricted-constant cases are not estimated",
                        "trace critical values are the MHM 1999 intercept table, not χ²",
                    ))
                    .build(),
            );
        }

        let s00 = moment(&dy, &dy);
        let s11 = moment(&ylag, &ylag);
        let s01 = moment(&dy, &ylag);

        let Some(w00) = inv_sqrt(&mut ctx, &s00, "S00") else {
            return ctx.finish(empty_johansen(k, n));
        };
        let Some(w11) = inv_sqrt(&mut ctx, &s11, "S11") else {
            return ctx.finish(empty_johansen(k, n));
        };
        // Q = S00^{-1/2} S01 S11^{-1/2}
        let q = mul3(&w00, &s01, &w11);
        let qmat = Matrix::from_fn(k, k, |i, j| q[(i, j)]);
        let Some(svd) = thin_svd(&mut ctx.report, &qmat, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::EigenDidNotConverge)
                    .message("whitened Johansen SVD failed")
                    .build(),
            );
            return ctx.finish(empty_johansen(k, n));
        };

        let mut eigs: Vec<f64> = svd
            .singular_values
            .iter()
            .map(|s| (s * s).clamp(0.0, 1.0 - 1e-15))
            .collect();
        // Pad if thin SVD dropped zeros.
        while eigs.len() < k {
            eigs.push(0.0);
        }
        eigs.truncate(k);

        // β = S11^{-1/2} V  (columns = cointegrating vectors, largest λ first)
        let rmax = svd.v.ncols().min(k);
        let mut beta = Matrix::zeros(k, k);
        for j in 0..rmax {
            for i in 0..k {
                let mut acc = 0.0;
                for t in 0..k {
                    acc += w11[(i, t)] * svd.v[(t, j)];
                }
                beta.set(i, j, acc);
            }
        }
        normalize_beta(&mut beta);

        let mut trace = Vector::zeros(k);
        let mut maxeig = Vector::zeros(k);
        let tf = tstar as f64;
        for r in 0..k {
            let mut acc = 0.0;
            for i in r..k {
                acc += -(1.0 - eigs[i]).max(1e-15).ln();
            }
            trace[r] = tf * acc;
            maxeig[r] = tf * (-(1.0 - eigs[r]).max(1e-15).ln());
        }

        let mut suggested = 0usize;
        for r in 0..k {
            let crit = mhm_trace_5(k - r);
            if trace[r] > crit {
                suggested = r + 1;
            } else {
                break;
            }
        }
        suggested = suggested.min(k.saturating_sub(1));

        if eigs.iter().all(|l| *l < 1e-10) {
            ctx.push(
                Issue::builder(IssueCode::FeatureTargetIndependence)
                    .severity(signlred::Severity::Warning)
                    .message(
                        "every Johansen eigenvalue is ~0; no cointegrating relation is identified",
                    )
                    .meaninglessness(Meaninglessness::new(
                        "cointegrating vectors",
                        "the levels have no reduced-rank link at working precision",
                        signlred::InterpretiveValue::Misleading,
                        "report r=0; a VECM with r>0 would invent an error-correction term",
                    ))
                    .build(),
            );
        }

        ctx.finish(FittedJohansen {
            eigenvalues: Vector::from_iter(eigs),
            trace,
            maxeig,
            beta,
            suggested_rank: suggested,
            t: tstar,
            k,
            last: Matrix::from_fn(1, k, |_, j| y.get(n - 1, j)),
            s01: mat_to_matrix(&s01),
            s11: mat_to_matrix(&s11),
        })
    }
}

/// Fitted Johansen spectrum.
#[derive(Clone, Debug)]
pub struct FittedJohansen {
    /// \(\lambda_1 \ge \cdots \ge \lambda_K \in [0,1)\).
    pub eigenvalues: Vector,
    /// Trace statistic for \(H_0: r = 0,1,\ldots,K-1\).
    pub trace: Vector,
    /// Max-eigen statistic for the same nulls.
    pub maxeig: Vector,
    /// Cointegrating vectors (columns), largest \(\lambda\) first.
    pub beta: Matrix,
    /// Smallest r whose remaining trace fails the 5% MHM table.
    pub suggested_rank: usize,
    /// Effective sample \(T-1\).
    pub t: usize,
    /// Number of series.
    pub k: usize,
    last: Matrix,
    s01: Matrix,
    s11: Matrix,
}

impl FittedJohansen {
    /// Build a VECM of rank `r` from this spectrum (α, β, Π).
    pub fn vecm(&self, r: usize, session: &Session) -> Result<Qualified<FittedVecm>> {
        let mut ctx = FitCtx::with_session(session.child("vecm"));
        let r = r.min(self.k);
        if r == 0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .severity(signlred::Severity::Warning)
                    .message(
                        "VECM rank 0 is a VAR in differences; there is no error-correction term",
                    )
                    .meaninglessness(Meaninglessness::new(
                        "Π = αβ′",
                        "r=0 forces Π=0 by construction",
                        signlred::InterpretiveValue::Misleading,
                        "use a differenced VAR, or raise r only if the trace test supports it",
                    ))
                    .build(),
            );
        }
        let mut beta = Matrix::zeros(self.k, r.max(1));
        for j in 0..r {
            for i in 0..self.k {
                beta.set(i, j, self.beta.get(i, j));
            }
        }
        if r == 0 {
            beta = Matrix::zeros(self.k, 0);
        }
        let alpha = loading_alpha(&mut ctx, &self.s01, &self.s11, &beta, r);
        let mut pi = Matrix::zeros(self.k, self.k);
        for i in 0..self.k {
            for j in 0..self.k {
                let mut s = 0.0;
                for c in 0..r {
                    s += alpha.get(i, c) * beta.get(j, c);
                }
                pi.set(i, j, s);
            }
        }
        ctx.finish(FittedVecm {
            rank: r,
            alpha,
            beta,
            pi,
            last: self.last.clone(),
        })
    }
}

/// Vector error-correction model \(\Delta Y_t = \Pi Y_{t-1}\) (no short-run lags).
#[derive(Clone, Debug)]
pub struct Vecm {
    /// Cointegrating rank.
    pub rank: usize,
}

impl Default for Vecm {
    fn default() -> Self {
        Self { rank: 1 }
    }
}

impl Vecm {
    /// VECM of the given rank.
    pub fn new(rank: usize) -> Self {
        Self { rank }
    }

    /// Estimate Π via Johansen and keep the first `rank` relations.
    pub fn fit(&self, y: &Matrix, session: &Session) -> Result<Qualified<FittedVecm>> {
        let j = Johansen::new().fit(y, session)?;
        let (inner, report) = j.into_parts();
        if self.rank > inner.suggested_rank {
            // Re-open a session to attach the extra warning onto the VECM fit.
            let mut ctx = FitCtx::with_session(session.child("vecm"));
            for issue in report.issues() {
                ctx.push(issue.clone());
            }
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .severity(signlred::Severity::Warning)
                    .message(format!(
                        "requested r={} exceeds the Johansen 5% suggestion r={}",
                        self.rank, inner.suggested_rank
                    ))
                    .metric("requested_rank", self.rank as f64)
                    .metric("suggested_rank", inner.suggested_rank as f64)
                    .meaninglessness(Meaninglessness::new(
                        "extra cointegrating vectors",
                        "the additional columns of β correspond to eigenvalues the trace test does not support",
                        signlred::InterpretiveValue::Misleading,
                        "report the suggested rank, or justify a structural r",
                    ))
                    .build(),
            );
            let q = inner.vecm(self.rank, session)?;
            let (v, _) = q.into_parts();
            return ctx.finish(v);
        }
        inner.vecm(self.rank, session)
    }
}

/// Fitted VECM (Π = αβ′, no short-run Γ lags).
#[derive(Clone, Debug)]
pub struct FittedVecm {
    /// Cointegrating rank.
    pub rank: usize,
    /// Adjustment loadings \(K \times r\).
    pub alpha: Matrix,
    /// Cointegrating vectors \(K \times r\).
    pub beta: Matrix,
    /// \(\Pi = \alpha\beta'\).
    pub pi: Matrix,
    last: Matrix,
}

impl FittedVecm {
    /// Iterate \(y_t = y_{t-1} + \Pi y_{t-1}\) for `h` steps.
    pub fn forecast(&self, h: usize, session: &Session) -> Result<Qualified<Matrix>> {
        let ctx = FitCtx::with_session(session.child("forecast"));
        let k = self.pi.nrows();
        let mut y = Vector::from_iter((0..k).map(|j| self.last.get(0, j)));
        let mut out = Matrix::zeros(h, k);
        for t in 0..h {
            let mut dy = Vector::zeros(k);
            for i in 0..k {
                let mut s = 0.0;
                for j in 0..k {
                    s += self.pi.get(i, j) * y[j];
                }
                dy[i] = s;
            }
            for i in 0..k {
                y[i] += dy[i];
                out.set(t, i, y[i]);
            }
        }
        ctx.finish(out)
    }
}

fn empty_johansen(k: usize, n: usize) -> FittedJohansen {
    FittedJohansen {
        eigenvalues: Vector::zeros(k),
        trace: Vector::zeros(k),
        maxeig: Vector::zeros(k),
        beta: Matrix::zeros(k, k),
        suggested_rank: 0,
        t: n.saturating_sub(1),
        k,
        last: Matrix::zeros(1, k),
        s01: Matrix::zeros(k, k),
        s11: Matrix::zeros(k, k),
    }
}

fn demean_cols(x: &mut Matrix) {
    let (n, p) = x.shape();
    if n == 0 {
        return;
    }
    for j in 0..p {
        let mut s = 0.0;
        for i in 0..n {
            s += x.get(i, j);
        }
        let m = s / n as f64;
        for i in 0..n {
            x.set(i, j, x.get(i, j) - m);
        }
    }
}

fn moment(a: &Matrix, b: &Matrix) -> Mat<f64> {
    let n = a.nrows().max(1) as f64;
    let mut g = Mat::<f64>::zeros(a.ncols(), b.ncols());
    for i in 0..a.ncols() {
        for j in 0..b.ncols() {
            let mut s = 0.0;
            for t in 0..a.nrows() {
                s += a.get(t, i) * b.get(t, j);
            }
            g[(i, j)] = s / n;
        }
    }
    g
}

fn mat_to_matrix(a: &Mat<f64>) -> Matrix {
    Matrix::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)])
}

fn mul3(a: &Mat<f64>, b: &Mat<f64>, c: &Mat<f64>) -> Mat<f64> {
    let tmp = a * b;
    tmp * c
}

fn inv_sqrt(ctx: &mut FitCtx, a: &Mat<f64>, name: &str) -> Option<Mat<f64>> {
    let m = Matrix::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let svd = thin_svd(&mut ctx.report, &m, &ctx.policy)?;
    let r = svd.rank(ctx.policy.rank_tol_relative);
    if r < a.nrows() {
        ctx.push(
            Issue::builder(IssueCode::RankDeficient)
                .message(format!(
                    "Johansen {name} has numerical rank {r} < {}",
                    a.nrows()
                ))
                .compromise(NumericalCompromise::new(
                    format!("{name}^{{-1/2}} at full rank"),
                    format!("truncated SVD inverse square root, rank {r}"),
                    "the residual moment matrix is singular at working precision",
                    "eigenvalues in the dropped subspace are unidentified",
                ))
                .metric("rank", r as f64)
                .build(),
        );
    }
    if r == 0 {
        ctx.push(
            Issue::builder(IssueCode::RankZero)
                .message(format!("Johansen {name} is the zero operator"))
                .build(),
        );
        return None;
    }
    let p = a.nrows();
    let mut w = Mat::<f64>::zeros(p, p);
    for k in 0..r.min(svd.singular_values.len()) {
        let s = svd.singular_values[k];
        if s <= 0.0 {
            continue;
        }
        let scale = 1.0 / s.sqrt();
        for i in 0..p.min(svd.u.nrows()) {
            for j in 0..p.min(svd.u.nrows()) {
                w[(i, j)] += svd.u[(i, k)] * scale * svd.u[(j, k)];
            }
        }
    }
    Some(w)
}

fn normalize_beta(beta: &mut Matrix) {
    let (k, r) = beta.shape();
    for j in 0..r {
        let mut piv = 0usize;
        let mut best = 0.0;
        for i in 0..k {
            if beta.get(i, j).abs() > best {
                best = beta.get(i, j).abs();
                piv = i;
            }
        }
        if best > 1e-15 {
            let s = beta.get(piv, j);
            for i in 0..k {
                beta.set(i, j, beta.get(i, j) / s);
            }
        }
    }
}

fn loading_alpha(ctx: &mut FitCtx, s01: &Matrix, s11: &Matrix, beta: &Matrix, r: usize) -> Matrix {
    if r == 0 {
        return Matrix::zeros(s01.nrows(), 0);
    }
    // α = S01 β (β' S11 β)^{-1}
    let mut bsb = Matrix::zeros(r, r);
    for a in 0..r {
        for b in 0..r {
            let mut s = 0.0;
            for i in 0..beta.nrows() {
                let mut s11bi = 0.0;
                for t in 0..s11.ncols() {
                    s11bi += s11.get(i, t) * beta.get(t, b);
                }
                s += beta.get(i, a) * s11bi;
            }
            bsb.set(a, b, s);
        }
    }
    let ident = Matrix::from_fn(r, r, |i, j| if i == j { 1.0 } else { 0.0 });
    let mut rhs = Vector::zeros(r);
    let mut inv = Matrix::zeros(r, r);
    for col in 0..r {
        for i in 0..r {
            rhs[i] = ident.get(i, col);
        }
        if let Some(sol) = crate::linalg::least_squares(&mut ctx.report, &bsb, &rhs, &ctx.policy) {
            for i in 0..r {
                inv.set(i, col, sol[i]);
            }
        }
    }
    let mut s01b = Matrix::zeros(s01.nrows(), r);
    for i in 0..s01.nrows() {
        for c in 0..r {
            let mut s = 0.0;
            for j in 0..s01.ncols() {
                s += s01.get(i, j) * beta.get(j, c);
            }
            s01b.set(i, c, s);
        }
    }
    Matrix::from_fn(s01.nrows(), r, |i, c| {
        let mut s = 0.0;
        for t in 0..r {
            s += s01b.get(i, t) * inv.get(t, c);
        }
        s
    })
}

/// MacKinnon–Haug–Michelis (1999) 5% trace critical values, intercept case,
/// for `n_minus_r` = K − r residual unit roots. Values beyond the table are
/// linearly extrapolated and flagged by the caller via the compromise on
/// `Johansen::fit`.
fn mhm_trace_5(n_minus_r: usize) -> f64 {
    match n_minus_r {
        0 => 0.0,
        1 => 3.76,
        2 => 15.41,
        3 => 29.80,
        4 => 47.85,
        5 => 69.82,
        6 => 95.75,
        n => 95.75 + 28.0 * (n as f64 - 6.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cointegrated_pair(n: usize) -> Matrix {
        // y1 random walk, y2 = y1 + noise  ⇒  one cointegrating relation.
        let mut y1 = 0.0;
        Matrix::from_fn(n, 2, |t, j| {
            if j == 0 {
                y1 += 0.15 * (((t * 17 + 3) % 7) as f64 - 3.0);
                y1
            } else {
                y1 + 0.05 * (((t * 11 + 5) % 5) as f64 - 2.0)
            }
        })
    }

    #[test]
    fn johansen_finds_one_relation() {
        let y = cointegrated_pair(80);
        let q = Johansen::new()
            .fit(&y, &Session::new("joh", "fit"))
            .expect("johansen");
        assert_eq!(q.value.k, 2);
        assert!(q.value.eigenvalues[0] > q.value.eigenvalues[1]);
        assert!(
            q.value.suggested_rank >= 1,
            "trace={:?} eigs={:?}",
            q.value.trace.as_slice(),
            q.value.eigenvalues.as_slice()
        );
        let vecm = q.value.vecm(1, &Session::new("joh", "vecm")).expect("vecm");
        assert_eq!(vecm.value.rank, 1);
        assert_eq!(vecm.value.pi.nrows(), 2);
        let fc = vecm.value.forecast(3, &Session::new("joh", "fc")).unwrap();
        assert_eq!(fc.value.nrows(), 3);
    }

    #[test]
    fn short_series_is_unidentified() {
        let y = Matrix::from_fn(4, 2, |i, j| (i + j) as f64);
        let err = Johansen::new()
            .fit(&y, &Session::new("joh", "fit"))
            .unwrap_err();
        assert_eq!(err.primary().code, IssueCode::InsufficientSample);
    }
}
