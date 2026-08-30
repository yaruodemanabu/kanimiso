//! Random-intercept and random-slope mixed models (statsmodels `MixedLM`).
//!
//! Identification uses the within (group-demeaned) OLS estimand for the
//! intercept-only model. Random slopes use the Swamy (1970) group-wise OLS
//! estimator: a group with \(n_g \le p+1\) cannot identify an intercept plus
//! \(p\) slopes and is dropped with a warning. A single group, or groups of
//! size 1 only, makes the random intercept unidentified.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::least_squares;
use crate::traits::Predict;
use crate::validate::inspect_xy;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result};
use std::collections::BTreeMap;

/// Random-intercept / random-slope linear mixed model.
#[derive(Clone, Debug)]
pub struct MixedLM {
    /// Include a global intercept in the within design.
    pub fit_intercept: bool,
    /// If true, estimate a random coefficient on every column of `X` (Swamy).
    pub random_slopes: bool,
}

impl Default for MixedLM {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            random_slopes: false,
        }
    }
}

impl MixedLM {
    /// Default random-intercept model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit `y | groups` with a random intercept per group.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMixed>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if self.random_slopes {
            return self.fit_random_slopes(x, y, groups, ctx);
        }
        if groups.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("groups length ≠ n")
                    .build(),
            );
            return ctx.finish(empty_mixed(x));
        }
        let mut sizes: BTreeMap<i64, usize> = BTreeMap::new();
        for &g in groups.as_slice() {
            if !g.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteInput)
                        .message("group labels contain NaN/Inf")
                        .build(),
                );
                break;
            }
            *sizes.entry(g.round() as i64).or_insert(0) += 1;
        }
        let n_groups = sizes.len();
        if n_groups <= 1 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("a random intercept is unidentified with a single group")
                    .meaninglessness(Meaninglessness::vacuous(
                        "random intercept variance",
                        "u_i is not separable from the residual when there is only one cluster",
                        "use OLS, or collect more groups",
                    ))
                    .build(),
            );
            return ctx.finish(empty_mixed(x));
        }
        let n_singletons = sizes.values().filter(|&&c| c <= 1).count();
        if n_singletons == n_groups {
            ctx.push(
                Issue::builder(IssueCode::IncrementalUnidentifiable)
                    .message("every group has size 1; the within estimand is empty")
                    .meaninglessness(Meaninglessness::vacuous(
                        "within-group slopes",
                        "no repeated measures; group demeaning zeros every row",
                        "need groups with size ≥ 2, or fit a between model only",
                    ))
                    .build(),
            );
        }
        if n_groups < 5 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(signlred::Severity::Warning)
                    .message(format!(
                        "{n_groups} groups is a thin sample for a variance component"
                    ))
                    .metric("n_groups", n_groups as f64)
                    .build(),
            );
        }

        // Group means
        let mut sum_x: BTreeMap<i64, Vector> = BTreeMap::new();
        let mut sum_y: BTreeMap<i64, f64> = BTreeMap::new();
        for i in 0..y.len() {
            let g = groups[i].round() as i64;
            let entry = sum_x.entry(g).or_insert_with(|| Vector::zeros(x.ncols()));
            for j in 0..x.ncols() {
                entry[j] += x.get(i, j);
            }
            *sum_y.entry(g).or_insert(0.0) += y[i];
        }
        let mut xw = Matrix::zeros(x.nrows(), x.ncols());
        let mut yw = Vector::zeros(y.len());
        for i in 0..y.len() {
            let g = groups[i].round() as i64;
            let c = *sizes.get(&g).unwrap_or(&1) as f64;
            let mx = sum_x.get(&g).unwrap();
            let my = *sum_y.get(&g).unwrap_or(&0.0) / c;
            yw[i] = y[i] - my;
            for j in 0..x.ncols() {
                xw.set(i, j, x.get(i, j) - mx[j] / c);
            }
        }
        let design = if self.fit_intercept {
            // within intercept is identically 0 after demeaning
            ctx.push(
                Issue::builder(IssueCode::OneHotFullRankViolation)
                    .severity(signlred::Severity::Advisory)
                    .message("within design has no intercept: group demeaning kills the constant")
                    .compromise(NumericalCompromise::new(
                        "random-intercept model with a global intercept",
                        "within OLS without an intercept column",
                        "the intercept is absorbed into u_i + grand mean",
                        "the reported intercept is the grand mean, not a within slope",
                    ))
                    .build(),
            );
            xw
        } else {
            xw
        };
        let Some(coef) = least_squares(&mut ctx.report, &design, &yw, &ctx.policy) else {
            ctx.push(Issue::builder(IssueCode::UnidentifiedModel).build());
            return ctx.finish(empty_mixed(x));
        };
        // Variance components (Swamy–Arora style, simplified)
        let resid_w = yw.sub(&design.matvec(&coef));
        let sse_w = resid_w.dot(&resid_w);
        let df_w = (y.len() as f64 - n_groups as f64 - x.ncols() as f64).max(1.0);
        let sigma2 = sse_w / df_w;
        let mut between_y = Vector::zeros(n_groups);
        let mut between_x = Matrix::zeros(n_groups, x.ncols());
        for (k, (&g, &c)) in sizes.iter().enumerate() {
            let mx = sum_x.get(&g).unwrap();
            between_y[k] = *sum_y.get(&g).unwrap() / c as f64;
            for j in 0..x.ncols() {
                between_x.set(k, j, mx[j] / c as f64);
            }
        }
        let mut trial = signlred::Report::new("mixedlm", "between");
        let between = least_squares(&mut trial, &between_x, &between_y, &ctx.policy);
        let tau2 = match &between {
            Some(b) => {
                let r = between_y.sub(&between_x.matvec(b));
                (r.dot(&r) / (n_groups as f64 - 1.0) - sigma2).max(0.0)
            }
            None => {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("between regression failed; τ² left at 0")
                        .build(),
                );
                0.0
            }
        };
        if tau2 <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::DegenerateDistribution)
                    .message(
                        "estimated random-intercept variance is ~0; the model collapsed to OLS",
                    )
                    .metric("tau2", tau2)
                    .build(),
            );
        }
        let intercept = y.mean();
        ctx.finish(FittedMixed {
            coef,
            intercept,
            sigma2,
            tau2,
            slope_tau2: Vector::zeros(x.ncols()),
            n_groups,
            n_unidentified_groups: 0,
            n: y.len(),
        })
    }

    fn fit_random_slopes(
        &self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        mut ctx: FitCtx,
    ) -> Result<Qualified<FittedMixed>> {
        if groups.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("groups length ≠ n")
                    .build(),
            );
            return ctx.finish(empty_mixed(x));
        }
        let mut members: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (i, &g) in groups.as_slice().iter().enumerate() {
            if !g.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteInput)
                        .message("group labels contain NaN/Inf")
                        .build(),
                );
                return ctx.finish(empty_mixed(x));
            }
            members.entry(g.round() as i64).or_default().push(i);
        }
        let p = x.ncols();
        let need = p + 1 + 1; // intercept + slopes + residual df
        let mut betas: Vec<Vector> = Vec::new();
        let mut sigmas: Vec<f64> = Vec::new();
        let mut n_unid = 0usize;
        for (g, idx) in &members {
            if idx.len() < need {
                n_unid += 1;
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .severity(signlred::Severity::Warning)
                        .message(format!(
                            "group {g} has n={} ≤ p+1={}; its random slope is unidentified",
                            idx.len(),
                            p + 1
                        ))
                        .meaninglessness(Meaninglessness::new(
                            "group-wise OLS slope",
                            "n_g ≤ p+1 leaves no residual degree of freedom for a group-specific (intercept, slope) vector",
                            signlred::InterpretiveValue::Misleading,
                            "this group is dropped from Δ; remaining identified groups still identify the Swamy mean",
                        ))
                        .metric("n_g", idx.len() as f64)
                        .build(),
                );
                continue;
            }
            let design = Matrix::from_fn(idx.len(), p + 1, |r, c| {
                if c == 0 {
                    1.0
                } else {
                    x.get(idx[r], c - 1)
                }
            });
            let yg = Vector::from_iter(idx.iter().map(|&i| y[i]));
            let mut scratch = signlred::Report::new("mixed", "group_ols");
            match crate::linalg::least_squares(&mut scratch, &design, &yg, &ctx.policy) {
                Some(b) => {
                    let fit = design.matvec(&b);
                    let mut sse = 0.0;
                    for i in 0..yg.len() {
                        let e = yg[i] - fit[i];
                        sse += e * e;
                    }
                    let df = (idx.len() - (p + 1)) as f64;
                    sigmas.push(sse / df.max(1.0));
                    betas.push(b);
                }
                None => {
                    n_unid += 1;
                    ctx.push(
                        Issue::builder(IssueCode::CholeskyFailed)
                            .severity(signlred::Severity::Warning)
                            .message(format!("group {g} OLS failed to factor"))
                            .build(),
                    );
                }
            }
        }
        let n_groups = members.len();
        if betas.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message(format!(
                        "random slopes need ≥2 identified groups; got {} (unidentified={n_unid})",
                        betas.len()
                    ))
                    .meaninglessness(Meaninglessness::vacuous(
                        "between-group slope covariance",
                        "Δ = Cov(β_g) is unidentified with fewer than two group OLS fits",
                        "collect more groups with n_g > p+1, or fit random intercepts only",
                    ))
                    .build(),
            );
            return ctx.finish(empty_mixed(x));
        }
        if n_groups < 5 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(signlred::Severity::Warning)
                    .message(format!(
                        "{n_groups} groups is a thin sample for a random-slope covariance"
                    ))
                    .metric("n_groups", n_groups as f64)
                    .build(),
            );
        }
        let dim = p + 1;
        let mut mean = Vector::zeros(dim);
        for b in &betas {
            for j in 0..dim {
                mean[j] += b[j];
            }
        }
        let inv_g = 1.0 / betas.len() as f64;
        for j in 0..dim {
            mean[j] *= inv_g;
        }
        let mut var = Vector::zeros(dim);
        if betas.len() > 1 {
            let den = (betas.len() - 1) as f64;
            for b in &betas {
                for j in 0..dim {
                    let d = b[j] - mean[j];
                    var[j] += d * d;
                }
            }
            for j in 0..dim {
                var[j] /= den;
            }
        }
        let sigma2 = if sigmas.is_empty() {
            f64::NAN
        } else {
            sigmas.iter().sum::<f64>() / sigmas.len() as f64
        };
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(signlred::Severity::Advisory)
                .message("Swamy random-slope means are not REML; SEs are not reported")
                .compromise(NumericalCompromise::new(
                    "REML / ML MixedLM with random slopes",
                    "Swamy (1970) average of identified group-wise OLS coefficients",
                    "a joint sparse-precision LMM is not estimated",
                    "treat τ² as a between-group variance of OLS slopes, not as a REML variance component",
                ))
                .build(),
        );
        let intercept = if self.fit_intercept { mean[0] } else { 0.0 };
        let coef = if self.fit_intercept {
            Vector::from_iter((1..dim).map(|j| mean[j]))
        } else {
            Vector::from_iter((1..dim).map(|j| mean[j]))
        };
        ctx.finish(FittedMixed {
            coef,
            intercept,
            sigma2,
            tau2: var[0],
            slope_tau2: Vector::from_iter((1..dim).map(|j| var[j])),
            n_groups,
            n_unidentified_groups: n_unid,
            n: y.len(),
        })
    }
}

/// Fitted random-intercept / random-slope model.
#[derive(Clone, Debug)]
pub struct FittedMixed {
    /// Fixed slopes (within OLS, or Swamy mean of group slopes).
    pub coef: Vector,
    /// Grand mean (intercept-only) or Swamy mean intercept (random slopes).
    pub intercept: f64,
    /// Residual variance.
    pub sigma2: f64,
    /// Random-intercept variance.
    pub tau2: f64,
    /// Between-group variance of each slope (empty when `random_slopes` is off).
    pub slope_tau2: Vector,
    /// Number of groups.
    pub n_groups: usize,
    /// Groups dropped because \(n_g \le p+1\).
    pub n_unidentified_groups: usize,
    /// Sample size.
    pub n: usize,
}

impl Predict for FittedMixed {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(signlred::Severity::Advisory)
                .message("predict uses the fixed part only (E[u_i]=0); it is not a BLUP")
                .build(),
        );
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            y[i] += self.intercept;
        }
        ctx.finish(y)
    }
}

fn empty_mixed(x: &Matrix) -> FittedMixed {
    FittedMixed {
        coef: Vector::zeros(x.ncols()),
        intercept: 0.0,
        sigma2: f64::NAN,
        tau2: f64::NAN,
        slope_tau2: Vector::zeros(x.ncols()),
        n_groups: 0,
        n_unidentified_groups: 0,
        n: x.nrows(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_intercept_recovers_slope() {
        // two groups, y = 2x + u_g
        let x = Matrix::from_fn(10, 1, |i, _| (i % 5) as f64);
        let y = Vector::from_iter((0..10).map(|i| {
            let u = if i < 5 { 5.0 } else { -5.0 };
            2.0 * (i % 5) as f64 + u
        }));
        let g = Vector::from_iter((0..10).map(|i| if i < 5 { 0.0 } else { 1.0 }));
        let q = MixedLM::new()
            .fit(&x, &y, &g, &Session::new("mixed", "fit"))
            .expect("mixed");
        assert!(
            (q.value.coef[0] - 2.0).abs() < 1e-6,
            "{:?}",
            q.value.coef.as_slice()
        );
        assert!(q.value.tau2 > 1.0, "tau2={}", q.value.tau2);
        assert_eq!(q.value.n_groups, 2);
    }

    #[test]
    fn random_slopes_recover_between_variance() {
        // group 0: y = 1 + 2x; group 1: y = 1 + 6x; n_g = 8 > p+1
        let x = Matrix::from_fn(16, 1, |i, _| (i % 8) as f64);
        let y = Vector::from_iter((0..16).map(|i| {
            let slope = if i < 8 { 2.0 } else { 6.0 };
            1.0 + slope * (i % 8) as f64
        }));
        let g = Vector::from_iter((0..16).map(|i| if i < 8 { 0.0 } else { 1.0 }));
        let q = MixedLM {
            random_slopes: true,
            ..MixedLM::default()
        }
        .fit(&x, &y, &g, &Session::new("mixed", "slope"))
        .expect("slope");
        assert!(
            (q.value.coef[0] - 4.0).abs() < 1e-6,
            "pooled slope {:?}",
            q.value.coef.as_slice()
        );
        assert!(
            q.value.slope_tau2[0] > 4.0,
            "slope_tau2={}",
            q.value.slope_tau2[0]
        );
        assert_eq!(q.value.n_unidentified_groups, 0);
    }

    #[test]
    fn random_slopes_tiny_groups_abort() {
        let x = Matrix::from_fn(4, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..4).map(|i| i as f64));
        let g = Vector::from_iter((0..4).map(|i| if i < 2 { 0.0 } else { 1.0 }));
        let err = MixedLM {
            random_slopes: true,
            ..MixedLM::default()
        }
        .fit(&x, &y, &g, &Session::new("mixed", "slope"))
        .unwrap_err();
        assert_eq!(err.primary().code, IssueCode::UnidentifiedModel);
    }

    #[test]
    fn single_group_is_unidentified() {
        let x = Matrix::from_fn(6, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..6).map(|i| i as f64));
        let g = Vector::filled(6, 1.0);
        let err = MixedLM::new()
            .fit(&x, &y, &g, &Session::new("mixed", "fit"))
            .unwrap_err();
        assert_eq!(err.primary().code, IssueCode::UnidentifiedModel);
    }
}
