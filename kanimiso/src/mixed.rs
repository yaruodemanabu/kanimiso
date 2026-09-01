//! Random-intercept and random-slope mixed models (statsmodels `MixedLM`).
//!
//! Identification uses the within (group-demeaned) OLS estimand for the
//! intercept-only model. Random slopes use the Swamy (1970) group-wise OLS
//! estimator: a group with \(n_g \le p+1\) cannot identify an intercept plus
//! \(p\) slopes and is dropped with a warning. A single group, or groups of
//! size 1 only, makes the random intercept unidentified.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{chol_solve, least_squares};
use crate::traits::Predict;
use crate::validate::inspect_xy;
use faer::Mat;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result};
use std::collections::BTreeMap;

/// Random-intercept / random-slope linear mixed model.
#[derive(Clone, Debug)]
pub(crate) struct MixedLM {
    /// Include a global intercept in the within design.
    pub fit_intercept: bool,
    /// If true, estimate a random coefficient on every column of `X` (Swamy).
    pub random_slopes: bool,
    /// Iterate GLS + Henderson/EM variance updates with REML denominators.
    ///
    /// This is **not** full observed-information REML: the information matrix
    /// of \((\sigma^2,\tau^2)\) is not formed, and random slopes stay on the
    /// Swamy path.
    pub reml: bool,
}

impl Default for MixedLM {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            random_slopes: false,
            reml: false,
        }
    }
}

impl MixedLM {
    /// Default random-intercept model.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Fit `y | groups` with a random intercept per group.
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMixed>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if self.random_slopes {
            if self.reml {
                ctx.push(
                    Issue::builder(IssueCode::PValueUnreliable)
                        .severity(signlred::Severity::Advisory)
                        .message("REML is not implemented for random slopes; Swamy group OLS is used")
                        .compromise(NumericalCompromise::new(
                            "REML MixedLM with a random-slope covariance",
                            "Swamy (1970) average of identified group-wise OLS coefficients",
                            "a joint sparse-precision LMM is not estimated",
                            "treat τ² as a between-group variance of OLS slopes, not as a REML variance component",
                        ))
                        .build(),
                );
            }
            return self.fit_random_slopes(x, y, groups, ctx);
        }
        if self.reml {
            return self.fit_reml(x, y, groups, ctx);
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
            used_reml: false,
        })
    }

    fn fit_reml(
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
        let n_groups = members.len();
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
        let n_singletons = members.values().filter(|idx| idx.len() <= 1).count();
        if n_singletons == n_groups {
            ctx.push(
                Issue::builder(IssueCode::IncrementalUnidentifiable)
                    .message("every group has size 1; the within estimand is empty")
                    .meaninglessness(Meaninglessness::vacuous(
                        "within-group slopes",
                        "no repeated measures; REML τ² is not separable from σ²",
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
                        "{n_groups} groups is a thin sample for a REML variance component"
                    ))
                    .metric("n_groups", n_groups as f64)
                    .build(),
            );
        }
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        let p = design.ncols();
        let mut scratch = signlred::Report::new("mixed", "reml_start");
        let mut beta = least_squares(&mut scratch, &design, y, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(p));
        let resid0 = y.sub(&design.matvec(&beta));
        let mut sigma2 = (resid0.dot(&resid0) / (y.len() as f64 - p as f64).max(1.0)).max(1e-8);
        let mut tau2 = sigma2;
        let mut converged = false;
        for it in 0..40 {
            match gls_random_intercept(&design, y, &members, sigma2, tau2, &ctx.policy) {
                Some(b) => beta = b,
                None => {
                    ctx.push(
                        Issue::builder(IssueCode::CholeskyFailed)
                            .severity(signlred::Severity::Warning)
                            .message("REML GLS Hessian refused Cholesky; keeping the last β")
                            .compromise(NumericalCompromise::new(
                                "GLS with V_g = σ²I + τ²11'",
                                "previous iterate of β",
                                "X'V⁻¹X was not SPD at working precision",
                                "the reported point is not a GLS/REML stationary point",
                            ))
                            .build(),
                    );
                    break;
                }
            }
            let (s2, t2) = reml_em_variances(&design, y, &beta, &members, sigma2, tau2, p);
            let ds = (s2 - sigma2).abs() + (t2 - tau2).abs();
            sigma2 = s2;
            tau2 = t2;
            ctx.session.step(it as u64, ds, Some(ds));
            if ds < 1e-8 && it > 0 {
                ctx.session.converged("MixedLM REML EM", it as u64);
                converged = true;
                break;
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .severity(signlred::Severity::Warning)
                    .message("REML GLS/EM did not meet the tolerance")
                    .build(),
            );
        }
        if tau2 <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::DegenerateDistribution)
                    .message("REML τ² collapsed to ~0; the model is OLS with a GLS path")
                    .metric("tau2", tau2)
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(signlred::Severity::Advisory)
                .message("REML here is Henderson/EM for a random intercept, not observed-information REML")
                .compromise(NumericalCompromise::new(
                    "observed-information REML of (β, σ², τ²)",
                    "iterated GLS plus EM with denominators n−p and G−1",
                    "the Hessian of the restricted likelihood is not formed",
                    "SEs are not reported; treat (σ², τ²) as moment/EM estimates",
                ))
                .build(),
        );
        let intercept = if self.fit_intercept && !beta.is_empty() {
            beta[0]
        } else {
            0.0
        };
        let coef = if self.fit_intercept {
            Vector::from_iter((1..beta.len()).map(|j| beta[j]))
        } else {
            beta
        };
        ctx.finish(FittedMixed {
            coef,
            intercept,
            sigma2,
            tau2,
            slope_tau2: Vector::zeros(x.ncols()),
            n_groups,
            n_unidentified_groups: 0,
            n: y.len(),
            used_reml: true,
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
            used_reml: false,
        })
    }
}

/// Fitted random-intercept / random-slope model.
#[derive(Clone, Debug)]
pub(crate) struct FittedMixed {
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
    /// True when the REML GLS/EM path produced this fit.
    pub used_reml: bool,
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
        used_reml: false,
    }
}

fn gls_random_intercept(
    design: &Matrix,
    y: &Vector,
    members: &BTreeMap<i64, Vec<usize>>,
    sigma2: f64,
    tau2: f64,
    policy: &signlred::Policy,
) -> Option<Vector> {
    let p = design.ncols();
    let mut xtvx = vec![0.0; p * p];
    let mut xtvy = Vector::zeros(p);
    let s2 = sigma2.max(1e-12);
    let t2 = tau2.max(0.0);
    for idx in members.values() {
        let ng = idx.len() as f64;
        let a = 1.0 / s2;
        let b = t2 / (s2 * (s2 + ng * t2)).max(1e-18);
        let mut x1 = Vector::zeros(p);
        let mut y1 = 0.0;
        for &i in idx {
            y1 += y[i];
            for j in 0..p {
                x1[j] += design.get(i, j);
            }
        }
        for &i in idx {
            for j in 0..p {
                let xij = design.get(i, j);
                xtvy[j] += a * xij * y[i];
                for k in 0..p {
                    xtvx[j * p + k] += a * xij * design.get(i, k);
                }
            }
        }
        for j in 0..p {
            xtvy[j] -= b * x1[j] * y1;
            for k in 0..p {
                xtvx[j * p + k] -= b * x1[j] * x1[k];
            }
        }
    }
    let mut a = Mat::<f64>::zeros(p, p);
    for j in 0..p {
        for k in 0..p {
            a[(j, k)] = xtvx[j * p + k];
        }
        a[(j, j)] += 1e-12;
    }
    let mut scratch = signlred::Report::new("mixed", "reml_gls");
    chol_solve(&mut scratch, &a, &xtvy, policy)
}

fn reml_em_variances(
    design: &Matrix,
    y: &Vector,
    beta: &Vector,
    members: &BTreeMap<i64, Vec<usize>>,
    sigma2: f64,
    tau2: f64,
    npar: usize,
) -> (f64, f64) {
    let s2 = sigma2.max(1e-12);
    let t2 = tau2.max(0.0);
    let mut sse = 0.0;
    let mut tau_acc = 0.0;
    let mut n = 0usize;
    let g = members.len().max(1);
    for idx in members.values() {
        let ng = idx.len() as f64;
        let mut ebar = 0.0;
        let mut e = Vec::with_capacity(idx.len());
        for &i in idx {
            let mut xb = 0.0;
            for j in 0..beta.len().min(design.ncols()) {
                xb += design.get(i, j) * beta[j];
            }
            let ei = y[i] - xb;
            e.push(ei);
            ebar += ei;
            n += 1;
        }
        ebar /= ng.max(1.0);
        let w = t2 / (t2 + s2 / ng.max(1.0));
        let u = w * ebar;
        let var_u = t2 * s2 / (s2 + ng * t2).max(1e-18);
        for ei in e {
            let r = ei - u;
            sse += r * r;
        }
        sse += ng * var_u;
        tau_acc += u * u + var_u;
    }
    let sigma_new = (sse / (n as f64 - npar as f64).max(1.0)).max(1e-12);
    let tau_new = (tau_acc / (g as f64 - 1.0).max(1.0)).max(0.0);
    (sigma_new, tau_new)
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

    #[test]
    fn reml_recovers_the_within_slope() {
        let x = Matrix::from_fn(10, 1, |i, _| (i % 5) as f64);
        let y = Vector::from_iter((0..10).map(|i| {
            let u = if i < 5 { 5.0 } else { -5.0 };
            2.0 * (i % 5) as f64 + u
        }));
        let g = Vector::from_iter((0..10).map(|i| if i < 5 { 0.0 } else { 1.0 }));
        let q = MixedLM {
            reml: true,
            ..MixedLM::default()
        }
        .fit(&x, &y, &g, &Session::new("mixed", "reml"))
        .expect("reml");
        assert!(q.value.used_reml);
        assert!(
            (q.value.coef[0] - 2.0).abs() < 0.15,
            "reml slope {:?}",
            q.value.coef.as_slice()
        );
        assert!(q.value.tau2 > 1.0, "tau2={}", q.value.tau2);
        assert!(q.value.sigma2.is_finite());
    }
}
