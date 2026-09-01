//! Linear and quadratic discriminant analysis.
//!
//! A single training class records [`IssueCode::SingleClass`]. A class with
//! no finite rows records [`IssueCode::EmptyClass`]. A class-conditional
//! covariance that is not SPD records [`IssueCode::EmissionDegenerate`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::symmetric_eigen;
use crate::traits::{Fit, Predict};
use crate::validate::{inspect_classes, inspect_identification, inspect_xy};
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Qualified, Result};

fn labels_of(y: &Vector) -> Vec<i64> {
    y.as_slice()
        .iter()
        .map(|&v| if v.is_finite() { v.round() as i64 } else { 0 })
        .collect()
}

fn class_rows(y: &[i64], c: i64) -> Vec<usize> {
    y.iter()
        .enumerate()
        .filter(|(_, &lab)| lab == c)
        .map(|(i, _)| i)
        .collect()
}

fn class_mean(x: &Matrix, rows: &[usize]) -> Vector {
    let p = x.ncols();
    let mut m = Vector::zeros(p);
    if rows.is_empty() {
        return m;
    }
    for &i in rows {
        for j in 0..p {
            m[j] += x.get(i, j);
        }
    }
    m.scale(1.0 / rows.len() as f64)
}

fn scatter(x: &Matrix, rows: &[usize], mean: &Vector) -> Mat<f64> {
    let p = x.ncols();
    let mut s = Mat::<f64>::zeros(p, p);
    for &i in rows {
        for a in 0..p {
            let da = x.get(i, a) - mean[a];
            for b in 0..=a {
                let db = x.get(i, b) - mean[b];
                s[(a, b)] += da * db;
                if a != b {
                    s[(b, a)] += da * db;
                }
            }
        }
    }
    s
}

fn invert_or_pinv(ctx: &mut FitCtx, a: &Mat<f64>) -> Option<Mat<f64>> {
    let p = a.nrows();
    match a.llt(Side::Lower) {
        Ok(chol) => {
            let mut out = Mat::<f64>::zeros(p, p);
            for j in 0..p {
                let mut e = Vector::zeros(p);
                e[j] = 1.0;
                let sol = chol.solve(e.to_matrix().inner());
                for i in 0..p {
                    out[(i, j)] = sol[(i, 0)];
                }
            }
            Some(out)
        }
        Err(_) => {
            ctx.push(
                Issue::builder(IssueCode::EmissionDegenerate)
                    .message("class covariance is not SPD; using a jittered inverse")
                    .build(),
            );
            let mut b = a.clone();
            let eps = ctx.policy.rank_tol_relative.max(1e-8);
            for i in 0..p {
                b[(i, i)] += eps;
            }
            match b.llt(Side::Lower) {
                Ok(chol) => {
                    let mut out = Mat::<f64>::zeros(p, p);
                    for j in 0..p {
                        let mut e = Vector::zeros(p);
                        e[j] = 1.0;
                        let sol = chol.solve(e.to_matrix().inner());
                        for i in 0..p {
                            out[(i, j)] = sol[(i, 0)];
                        }
                    }
                    Some(out)
                }
                Err(_) => {
                    let Some((vals, vecs)) = symmetric_eigen(&mut ctx.report, &b, &ctx.policy)
                    else {
                        return None;
                    };
                    let mut out = Mat::<f64>::zeros(p, p);
                    for k in 0..vals.len().min(vecs.ncols()) {
                        if vals[k] <= eps {
                            continue;
                        }
                        let inv = 1.0 / vals[k];
                        for i in 0..p {
                            for j in 0..p {
                                out[(i, j)] += inv * vecs[(i, k)] * vecs[(j, k)];
                            }
                        }
                    }
                    Some(out)
                }
            }
        }
    }
}

fn logdet_spd(ctx: &mut FitCtx, a: &Mat<f64>) -> f64 {
    match a.self_adjoint_eigenvalues(Side::Lower) {
        Ok(vals) => {
            let mut s = 0.0;
            for &v in &vals {
                if v <= 0.0 {
                    ctx.push(
                        Issue::builder(IssueCode::EmissionDegenerate)
                            .message("non-positive eigenvalue in a class covariance")
                            .build(),
                    );
                    s += (v.abs().max(1e-12)).ln();
                } else {
                    s += v.ln();
                }
            }
            s
        }
        Err(_) => {
            ctx.push(Issue::builder(IssueCode::EigenDidNotConverge).build());
            f64::NAN
        }
    }
}

/// Linear discriminant analysis (shared pooled covariance).
#[derive(Clone, Debug)]
pub(crate) struct LinearDiscriminantAnalysis {
    /// Shrinkage toward a scaled identity on the pooled covariance (`0` = none).
    pub shrinkage: f64,
}

impl Default for LinearDiscriminantAnalysis {
    fn default() -> Self {
        Self { shrinkage: 0.05 }
    }
}

impl LinearDiscriminantAnalysis {
    /// Unshrunk LDA.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Fitted LDA.
#[derive(Clone, Debug)]
pub(crate) struct FittedLda {
    /// Sorted class labels.
    pub classes: Vec<i64>,
    /// Class means (`K × p`).
    pub means: Matrix,
    /// Class priors.
    pub priors: Vector,
    /// Pooled covariance.
    pub covariance: Matrix,
    /// Discriminant directions (`p × r`) from the generalised eigenproblem.
    pub scalings: Matrix,
}

impl FittedLda {
    fn log_scores(&self, x: &Matrix, prec: &Mat<f64>) -> Matrix {
        let k = self.classes.len();
        let p = x.ncols().min(self.means.ncols());
        let mut out = Matrix::zeros(x.nrows(), k);
        for i in 0..x.nrows() {
            for c in 0..k {
                let mut d = Vector::zeros(p);
                for j in 0..p {
                    d[j] = x.get(i, j) - self.means.get(c, j);
                }
                let mut qd = 0.0;
                for a in 0..p {
                    let mut s = 0.0;
                    for b in 0..p {
                        s += prec[(a, b)] * d[b];
                    }
                    qd += d[a] * s;
                }
                let prior = self.priors[c].max(1e-15).ln();
                out.set(i, c, prior - 0.5 * qd);
            }
        }
        out
    }
}

impl Predict for FittedLda {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if self.classes.is_empty() {
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let cov = Mat::<f64>::from_fn(self.covariance.nrows(), self.covariance.ncols(), |i, j| {
            self.covariance.get(i, j)
        });
        let prec = invert_or_pinv(&mut ctx, &cov).unwrap_or_else(|| {
            Mat::<f64>::from_fn(
                cov.nrows(),
                cov.ncols(),
                |i, j| if i == j { 1.0 } else { 0.0 },
            )
        });
        let sc = self.log_scores(x, &prec);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut best = 0usize;
            let mut val = f64::NEG_INFINITY;
            for c in 0..self.classes.len() {
                if sc.get(i, c) > val {
                    val = sc.get(i, c);
                    best = c;
                }
            }
            self.classes[best] as f64
        }));
        ctx.finish(y)
    }
}

impl Fit for LinearDiscriminantAnalysis {
    type Fitted = FittedLda;
    fn fit(&self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedLda>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        let labs = labels_of(y);
        let p = x.ncols();
        let k = classes.len();
        if k < 2 {
            return ctx.finish(FittedLda {
                classes,
                means: Matrix::zeros(1, p),
                priors: Vector::filled(1, 1.0),
                covariance: Matrix::zeros(p, p),
                scalings: Matrix::zeros(p, 0),
            });
        }
        let n = x.nrows() as f64;
        let mut means = Matrix::zeros(k, p);
        let mut priors = Vector::zeros(k);
        let mut sw = Mat::<f64>::zeros(p, p);
        let mut overall = Vector::zeros(p);
        for j in 0..p {
            overall[j] = x.column(j).mean();
        }
        for (c, &lab) in classes.iter().enumerate() {
            let rows = class_rows(&labs, lab);
            if rows.is_empty() {
                ctx.push(
                    Issue::builder(IssueCode::EmptyClass)
                        .message(format!("LDA class {lab} has no rows"))
                        .build(),
                );
            }
            let m = class_mean(x, &rows);
            for j in 0..p {
                means.set(c, j, m[j]);
            }
            priors[c] = rows.len() as f64 / n.max(1.0);
            let s = scatter(x, &rows, &m);
            for a in 0..p {
                for b in 0..p {
                    sw[(a, b)] += s[(a, b)];
                }
            }
        }
        let df = (x.nrows() as isize - k as isize).max(1) as f64;
        for a in 0..p {
            for b in 0..p {
                sw[(a, b)] /= df;
            }
        }
        // Floor the diagonal so a rank-deficient pooled scatter still yields a
        // usable (shrunk) inverse rather than aborting the fit.
        for i in 0..p {
            if sw[(i, i)] <= ctx.policy.near_zero_variance {
                sw[(i, i)] += 1e-6;
            }
        }
        if self.shrinkage > 0.0 {
            let mut tr = 0.0;
            for i in 0..p {
                tr += sw[(i, i)];
            }
            let mu = if p > 0 { tr / p as f64 } else { 0.0 };
            let a = self.shrinkage.clamp(0.0, 1.0);
            for i in 0..p {
                for j in 0..p {
                    let t = if i == j { mu } else { 0.0 };
                    sw[(i, j)] = (1.0 - a) * sw[(i, j)] + a * t;
                }
            }
        }
        let mut sb = Mat::<f64>::zeros(p, p);
        for c in 0..k {
            let nk = priors[c] * n;
            for a in 0..p {
                let da = means.get(c, a) - overall[a];
                for b in 0..p {
                    let db = means.get(c, b) - overall[b];
                    sb[(a, b)] += nk * da * db;
                }
            }
        }
        // Whitened between-scatter: Sw^{-1/2} Sb Sw^{-T/2}.
        let Some((evals, evecs)) = symmetric_eigen(&mut ctx.report, &sw, &ctx.policy) else {
            return ctx.finish(FittedLda {
                classes,
                means,
                priors,
                covariance: Matrix::from_fn(p, p, |i, j| sw[(i, j)]),
                scalings: Matrix::zeros(p, 0),
            });
        };
        let mut whiten = Mat::<f64>::zeros(p, p);
        for k_e in 0..evals.len().min(evecs.ncols()) {
            if evals[k_e] <= ctx.policy.near_zero_variance {
                continue;
            }
            let s = 1.0 / evals[k_e].sqrt();
            for i in 0..p {
                for j in 0..p {
                    whiten[(i, j)] += s * evecs[(i, k_e)] * evecs[(j, k_e)];
                }
            }
        }
        let mut mid = Mat::<f64>::zeros(p, p);
        for i in 0..p {
            for j in 0..p {
                let mut s = 0.0;
                for a in 0..p {
                    for b in 0..p {
                        s += whiten[(i, a)] * sb[(a, b)] * whiten[(b, j)];
                    }
                }
                mid[(i, j)] = s;
            }
        }
        let r = (k - 1).min(p);
        let mut scalings = Matrix::zeros(p, r);
        if let Some((vals, vecs)) = symmetric_eigen(&mut ctx.report, &mid, &ctx.policy) {
            let mut pairs: Vec<(f64, usize)> = vals
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (c, &(_, idx)) in pairs.iter().take(r).enumerate() {
                // scaling column = Sw^{-1/2} u
                for i in 0..p {
                    let mut s = 0.0;
                    for j in 0..p {
                        s += whiten[(i, j)] * vecs[(j, idx)];
                    }
                    scalings.set(i, c, s);
                }
            }
        }
        ctx.finish(FittedLda {
            classes,
            means,
            priors,
            covariance: Matrix::from_fn(p, p, |i, j| sw[(i, j)]),
            scalings,
        })
    }
}

/// Quadratic discriminant analysis (per-class covariance).
#[derive(Clone, Debug)]
pub(crate) struct QuadraticDiscriminantAnalysis {
    /// Diagonal jitter added to every class covariance.
    pub reg_param: f64,
}

impl Default for QuadraticDiscriminantAnalysis {
    fn default() -> Self {
        Self { reg_param: 0.0 }
    }
}

impl QuadraticDiscriminantAnalysis {
    /// Unregularized QDA.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Fitted QDA.
#[derive(Clone, Debug)]
pub(crate) struct FittedQda {
    /// Sorted class labels.
    pub classes: Vec<i64>,
    /// Class means (`K × p`).
    pub means: Matrix,
    /// Class priors.
    pub priors: Vector,
    /// Per-class covariance matrices.
    pub covariances: Vec<Matrix>,
}

impl Predict for FittedQda {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let k = self.classes.len();
        let p = x.ncols();
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut best = 0usize;
            let mut val = f64::NEG_INFINITY;
            for c in 0..k {
                let cov = &self.covariances.get(c);
                let Some(cov) = cov else {
                    continue;
                };
                let a = Mat::<f64>::from_fn(cov.nrows(), cov.ncols(), |r, c2| cov.get(r, c2));
                let mut d = Vector::zeros(p.min(self.means.ncols()));
                for j in 0..d.len() {
                    d[j] = x.get(i, j) - self.means.get(c, j);
                }
                let prec = invert_or_pinv(&mut ctx, &a);
                let mut qd = d.dot(&d);
                if let Some(prec) = prec {
                    qd = 0.0;
                    for a_i in 0..d.len() {
                        let mut s = 0.0;
                        for b in 0..d.len().min(prec.ncols()) {
                            s += prec[(a_i, b)] * d[b];
                        }
                        qd += d[a_i] * s;
                    }
                }
                let ld = logdet_spd(&mut ctx, &a);
                let score = self.priors[c].max(1e-15).ln() - 0.5 * ld - 0.5 * qd;
                if score > val {
                    val = score;
                    best = c;
                }
            }
            self.classes.get(best).copied().unwrap_or(0) as f64
        }));
        ctx.finish(y)
    }
}

impl Fit for QuadraticDiscriminantAnalysis {
    type Fitted = FittedQda;
    fn fit(&self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedQda>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        let labs = labels_of(y);
        let p = x.ncols();
        let k = classes.len();
        let n = x.nrows() as f64;
        if k < 2 {
            return ctx.finish(FittedQda {
                classes,
                means: Matrix::zeros(1, p),
                priors: Vector::filled(1, 1.0),
                covariances: vec![Matrix::zeros(p, p)],
            });
        }
        let mut means = Matrix::zeros(k, p);
        let mut priors = Vector::zeros(k);
        let mut covariances = Vec::with_capacity(k);
        for (c, &lab) in classes.iter().enumerate() {
            let rows = class_rows(&labs, lab);
            if rows.is_empty() {
                ctx.push(
                    Issue::builder(IssueCode::EmptyClass)
                        .message(format!("QDA class {lab} is empty"))
                        .build(),
                );
            }
            if rows.len() <= p {
                ctx.push(
                    Issue::builder(IssueCode::EmissionDegenerate)
                        .message(format!(
                            "QDA class {lab} has n_k={} ≤ p={p}; covariance is singular",
                            rows.len()
                        ))
                        .build(),
                );
            }
            let m = class_mean(x, &rows);
            for j in 0..p {
                means.set(c, j, m[j]);
            }
            priors[c] = rows.len() as f64 / n.max(1.0);
            let mut s = scatter(x, &rows, &m);
            let den = (rows.len() as isize - 1).max(1) as f64;
            for a in 0..p {
                for b in 0..p {
                    s[(a, b)] /= den;
                    if a == b {
                        s[(a, b)] += self.reg_param.max(0.0);
                    }
                }
            }
            covariances.push(Matrix::from_fn(p, p, |i, j| s[(i, j)]));
        }
        ctx.finish(FittedQda {
            classes,
            means,
            priors,
            covariances,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    fn two_blobs() -> (Matrix, Vector) {
        let x = Matrix::from_fn(20, 2, |i, j| {
            let c = if i < 10 { 0.0 } else { 4.0 };
            if j == 0 {
                c + 0.2 * ((i % 5) as f64 - 2.0)
            } else {
                c + 0.15 * ((i / 2) as f64 - 5.0)
            }
        });
        let y = Vector::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        (x, y)
    }

    #[test]
    fn lda_separates_blobs() {
        let (x, y) = two_blobs();
        let q = LinearDiscriminantAnalysis::new()
            .fit(&x, &y, &Session::new("lda", "fit"))
            .unwrap();
        let pred = q
            .value
            .predict(&x, &Session::new("lda", "pred"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..y.len() {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 18, "ok={ok}");
        assert_eq!(q.value.scalings.ncols() >= 1, true);
    }

    #[test]
    fn qda_separates_blobs() {
        let (x, y) = two_blobs();
        let q = QuadraticDiscriminantAnalysis { reg_param: 1e-3 }
            .fit(&x, &y, &Session::new("qda", "fit"))
            .unwrap();
        let pred = q
            .value
            .predict(&x, &Session::new("qda", "pred"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..y.len() {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 16, "ok={ok}");
    }

    #[test]
    fn single_class_is_diagnosed() {
        let x = Matrix::from_fn(8, 2, |i, j| (i + j) as f64);
        let y = Vector::filled(8, 1.0);
        let err = LinearDiscriminantAnalysis::new()
            .fit(&x, &y, &Session::new("lda", "one"))
            .unwrap_err();
        assert!(
            err.report.contains(IssueCode::SingleClass)
                || err.primary().code == IssueCode::SingleClass
                || err.report.contains(IssueCode::ConstantTarget)
        );
    }
}
