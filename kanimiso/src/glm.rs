//! Extra GLMs and linear online classifiers (probit, NB2, SGD, PA regression).
//!
//! Probit and negative-binomial fits are IRLS. Perfect separation and a
//! non-positive count series abort. SGD / PA expose
//! [`ojizou_san::IncrementalExplain`] on every `partial_fit`.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{chol_solve, least_squares};
use crate::special::{ln_gamma, norm_cdf};
use crate::traits::{Fit, PartialFit, Predict};
use crate::validate::{inspect_classes, inspect_identification, inspect_xy};
use faer::Mat;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{
    IncrementalQuality, InterpretiveValue, Issue, IssueCode, Meaninglessness, NumericalCompromise,
    Qualified, Result, Severity,
};
use std::collections::BTreeMap;

const INV_SQRT_2PI: f64 = 0.3989422804014327;

fn norm_pdf(z: f64) -> f64 {
    if !z.is_finite() {
        return 0.0;
    }
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn dummy_explain(update: u64, batch: usize, n_seen: u64) -> IncrementalExplain {
    IncrementalExplain::from_quality(
        IncrementalQuality::new(update, batch, n_seen),
        "nothing",
        "the update was rejected",
        "invalid",
        "invalid",
    )
}

/// Fitted GLM coefficients (probit / NB2).
#[derive(Clone, Debug)]
pub struct FittedGlm {
    /// Slopes.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
    /// Extra scalar: 1 for probit, \(\alpha\) for NB2.
    pub dispersion: f64,
}

impl Predict for FittedGlm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("GLM predict column count ≠ coef")
                    .build(),
            );
        }
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            y[i] += self.intercept;
        }
        ctx.finish(y)
    }
}

/// Binary probit GLM (statsmodels `Probit`) via IRLS.
#[derive(Clone, Debug)]
pub struct ProbitRegression {
    /// Max IRLS iterations.
    pub max_iter: usize,
    /// Prepend an intercept.
    pub fit_intercept: bool,
}

impl Default for ProbitRegression {
    fn default() -> Self {
        Self {
            max_iter: 40,
            fit_intercept: true,
        }
    }
}

impl ProbitRegression {
    /// Default probit.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for ProbitRegression {
    type Fitted = FittedGlm;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedGlm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        if counts.len() != 2 {
            if counts.len() > 2 {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("Probit is binary; K>2 is not a joint ordered probit")
                        .meaninglessness(Meaninglessness::new(
                            "probit coefficients",
                            "the latent Gaussian is identified for a single cut",
                            InterpretiveValue::Misleading,
                            "use MultinomialLogistic or an ordered model",
                        ))
                        .build(),
                );
            }
            return ctx.finish(FittedGlm {
                coef: Vector::zeros(x.ncols()),
                intercept: 0.0,
                dispersion: 1.0,
            });
        }
        let pos = counts[1].0;
        let y01 = Vector::from_iter(y.as_slice().iter().map(|&v| {
            if v.round() as i64 == pos {
                1.0
            } else {
                0.0
            }
        }));
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
        let mut beta = Vector::zeros(design.ncols());
        let mut converged = false;
        for it in 0..self.max_iter.max(1) {
            let mut xs = Matrix::zeros(design.nrows(), design.ncols());
            let mut z = Vector::zeros(y01.len());
            let mut sep = false;
            for i in 0..y01.len() {
                let mut eta = 0.0;
                for j in 0..design.ncols() {
                    eta += design.get(i, j) * beta[j];
                }
                eta = eta.clamp(-8.0, 8.0);
                let mu = norm_cdf(eta).clamp(1e-12, 1.0 - 1e-12);
                let dens = norm_pdf(eta).max(1e-12);
                if (y01[i] > 0.5 && mu > 1.0 - 1e-8) || (y01[i] < 0.5 && mu < 1e-8) {
                    sep = true;
                }
                let w = (dens * dens / (mu * (1.0 - mu))).max(1e-12);
                let sw = w.sqrt();
                z[i] = (eta + (y01[i] - mu) / dens) * sw;
                for j in 0..design.ncols() {
                    xs.set(i, j, design.get(i, j) * sw);
                }
            }
            if sep {
                ctx.push(
                    Issue::builder(IssueCode::QuasiCompleteSeparation)
                        .message("probit IRLS approached Φ(η)∈{0,1}; the MLE is diverging")
                        .build(),
                );
            }
            let mut scratch = signlred::Report::new("probit", "irls");
            let Some(next) = least_squares(&mut scratch, &xs, &z, &ctx.policy) else {
                break;
            };
            for issue in scratch.issues() {
                if issue.code == IssueCode::ResidualTooLarge {
                    continue;
                }
                ctx.push(issue.clone());
            }
            let d = next.sub(&beta).norm();
            beta = next;
            ctx.session.step(it as u64, d, None);
            if d < 1e-8 {
                ctx.session.converged("probit IRLS", it as u64);
                converged = true;
                break;
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("probit IRLS did not meet the tolerance")
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(Severity::Advisory)
                .message("probit coefficients are IRLS; no Hessian SEs are attached")
                .compromise(NumericalCompromise::new(
                    "observed-information probit MLE",
                    "IRLS with Φ / φ working weights",
                    "the last weighted LS is the score equation, not a sandwich",
                    "do not treat these as OLS t-statistics",
                ))
                .build(),
        );
        let (intercept, coef) = if self.fit_intercept {
            (beta[0], Vector::from_iter((1..beta.len()).map(|j| beta[j])))
        } else {
            (0.0, beta)
        };
        ctx.finish(FittedGlm {
            coef,
            intercept,
            dispersion: 1.0,
        })
    }
}

/// Negative-binomial GLM (NB2, log link) with a moment \(\alpha\).
#[derive(Clone, Debug)]
pub struct NegativeBinomialRegressor {
    /// Fixed \(\alpha = \mathrm{Var}/\mu^2 - 1/\mu\). `None` ⇒ moment estimate.
    pub alpha: Option<f64>,
    /// Max IRLS iterations.
    pub max_iter: usize,
    /// Prepend an intercept.
    pub fit_intercept: bool,
}

impl Default for NegativeBinomialRegressor {
    fn default() -> Self {
        Self {
            alpha: None,
            max_iter: 40,
            fit_intercept: true,
        }
    }
}

impl NegativeBinomialRegressor {
    /// NB2 with a moment-estimated dispersion.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for NegativeBinomialRegressor {
    type Fitted = FittedGlm;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedGlm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        for (i, &yi) in y.as_slice().iter().enumerate() {
            if yi < 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NonPositiveSeries)
                        .message(format!("NegativeBinomial y[{i}]={yi} < 0"))
                        .build(),
                );
                break;
            }
        }
        if ctx.report.contains(IssueCode::ConstantTarget)
            || ctx.report.contains(IssueCode::EmptyMatrix)
        {
            return ctx.finish(FittedGlm {
                coef: Vector::zeros(x.ncols()),
                intercept: y.mean().max(1e-8).ln(),
                dispersion: f64::NAN,
            });
        }
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
        let mut beta = Vector::zeros(design.ncols());
        beta[0] = y.mean().max(1e-6).ln();
        let mut alpha = self.alpha.unwrap_or(0.0).max(0.0);
        let mut converged = false;
        for it in 0..self.max_iter.max(1) {
            let mut xs = Matrix::zeros(design.nrows(), design.ncols());
            let mut z = Vector::zeros(y.len());
            let mut mus = Vector::zeros(y.len());
            for i in 0..y.len() {
                let mut eta = 0.0;
                for j in 0..design.ncols() {
                    eta += design.get(i, j) * beta[j];
                }
                let mu = eta.exp().max(1e-12);
                mus[i] = mu;
                let w = (mu / (1.0 + alpha * mu)).max(1e-12);
                let sw = w.sqrt();
                z[i] = (eta + (y[i] - mu) / mu) * sw;
                for j in 0..design.ncols() {
                    xs.set(i, j, design.get(i, j) * sw);
                }
            }
            if self.alpha.is_none() {
                let mut num = 0.0;
                let mut den = 0.0;
                for i in 0..y.len() {
                    let mu = mus[i];
                    let e = y[i] - mu;
                    num += e * e - mu;
                    den += mu * mu;
                }
                alpha = (num / den.max(1e-12)).max(0.0);
            }
            let mut scratch = signlred::Report::new("nb2", "irls");
            let Some(next) = least_squares(&mut scratch, &xs, &z, &ctx.policy) else {
                break;
            };
            for issue in scratch.issues() {
                if issue.code == IssueCode::ResidualTooLarge {
                    continue;
                }
                ctx.push(issue.clone());
            }
            let d = next.sub(&beta).norm();
            beta = next;
            ctx.session.step(it as u64, d, Some(alpha));
            if d < 1e-8 {
                ctx.session.converged("NB2 IRLS", it as u64);
                converged = true;
                break;
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("NB2 IRLS did not meet the tolerance")
                    .build(),
            );
        }
        if alpha <= 1e-12 {
            ctx.push(
                Issue::builder(IssueCode::DegenerateDistribution)
                    .severity(Severity::Advisory)
                    .message("NB2 α≈0; the fit collapsed to a Poisson GLM")
                    .compromise(NumericalCompromise::new(
                        "overdispersed NB2",
                        "moment α floor at 0",
                        "the sample variance is not larger than the mean",
                        "coefficients are Poisson IRLS, not a genuine NB2 MLE",
                    ))
                    .build(),
            );
        }
        let (intercept, coef) = if self.fit_intercept {
            (beta[0], Vector::from_iter((1..beta.len()).map(|j| beta[j])))
        } else {
            (0.0, beta)
        };
        ctx.finish(FittedGlm {
            coef,
            intercept,
            dispersion: alpha,
        })
    }
}

/// Mini-batch SGD classifier (hinge or log loss).
#[derive(Clone, Debug)]
pub struct SgdClassifier {
    /// Learning rate.
    pub learning_rate: f64,
    /// ℓ2 penalty.
    pub l2: f64,
    /// `"hinge"` or `"log"`.
    pub loss: SgdLoss,
    /// Intercept.
    pub fit_intercept: bool,
    coef: Vector,
    intercept: f64,
    classes: Vec<i64>,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

/// Loss for [`SgdClassifier`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SgdLoss {
    /// Linear SVM hinge.
    Hinge,
    /// Logistic log-loss.
    Log,
}

impl Default for SgdClassifier {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            l2: 1e-4,
            loss: SgdLoss::Hinge,
            fit_intercept: true,
            coef: Vector::zeros(0),
            intercept: 0.0,
            classes: Vec::new(),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl SgdClassifier {
    /// Default hinge SGD.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current slopes.
    pub fn coef(&self) -> &Vector {
        &self.coef
    }
}

impl PartialFit for SgdClassifier {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(
                Issue::builder(IssueCode::MissingTarget)
                    .message("SgdClassifier.partial_fit requires y")
                    .build(),
            );
            return ctx.finish(dummy_explain(self.updates, 0, self.n_seen));
        };
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        if self.classes.is_empty() {
            self.classes = counts.iter().map(|(c, _)| *c).collect();
        } else {
            for (c, _) in &counts {
                if !self.classes.contains(c) {
                    ctx.push(
                        Issue::builder(IssueCode::LabelSpaceExpandedOnline)
                            .message(format!("SGD saw a new class {c}"))
                            .build(),
                    );
                    self.classes.push(*c);
                    self.classes.sort_unstable();
                }
            }
        }
        if self.classes.len() > 2 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .severity(Severity::Warning)
                    .message("SgdClassifier is binary; extra labels are folded into ±1")
                    .meaninglessness(Meaninglessness::new(
                        "SGD decision scores",
                        "hinge/log SGD is derived for a single hyperplane",
                        InterpretiveValue::Misleading,
                        "use MultinomialLogistic for K>2",
                    ))
                    .build(),
            );
        }
        if !self.initialized {
            self.coef = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if self.coef.len() != x.ncols() {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message("SgdClassifier feature count changed")
                    .build(),
            );
            return ctx.finish(dummy_explain(self.updates, x.nrows(), self.n_seen));
        }
        let pos = *self.classes.last().unwrap_or(&1);
        let before = self.coef.clone();
        let mut loss_before = 0.0;
        let mut loss_after = 0.0;
        let mut info = 0.0;
        for i in 0..x.nrows() {
            let yi = if y[i].round() as i64 == pos {
                1.0
            } else {
                -1.0
            };
            let mut pred = self.intercept;
            for j in 0..x.ncols() {
                pred += self.coef[j] * x.get(i, j);
            }
            let margin = yi * pred;
            match self.loss {
                SgdLoss::Hinge => {
                    loss_before += (1.0 - margin).max(0.0);
                    if margin < 1.0 {
                        info += 1.0;
                        for j in 0..x.ncols() {
                            self.coef[j] +=
                                self.learning_rate * (yi * x.get(i, j) - self.l2 * self.coef[j]);
                        }
                        if self.fit_intercept {
                            self.intercept += self.learning_rate * yi;
                        }
                    } else if self.l2 > 0.0 {
                        for j in 0..x.ncols() {
                            self.coef[j] -= self.learning_rate * self.l2 * self.coef[j];
                        }
                    }
                }
                SgdLoss::Log => {
                    let p = sigmoid(pred);
                    let y01 = if yi > 0.0 { 1.0 } else { 0.0 };
                    loss_before += if y01 > 0.5 {
                        -(p.max(1e-15).ln())
                    } else {
                        -((1.0 - p).max(1e-15).ln())
                    };
                    let g = p - y01;
                    info += g * g;
                    for j in 0..x.ncols() {
                        self.coef[j] -=
                            self.learning_rate * (g * x.get(i, j) + self.l2 * self.coef[j]);
                    }
                    if self.fit_intercept {
                        self.intercept -= self.learning_rate * g;
                    }
                }
            }
        }
        for i in 0..x.nrows() {
            let yi = if y[i].round() as i64 == pos {
                1.0
            } else {
                -1.0
            };
            let mut pred = self.intercept;
            for j in 0..x.ncols() {
                pred += self.coef[j] * x.get(i, j);
            }
            match self.loss {
                SgdLoss::Hinge => loss_after += (1.0 - yi * pred).max(0.0),
                SgdLoss::Log => {
                    let p = sigmoid(pred);
                    let y01 = if yi > 0.0 { 1.0 } else { 0.0 };
                    loss_after += if y01 > 0.5 {
                        -(p.max(1e-15).ln())
                    } else {
                        -((1.0 - p).max(1e-15).ln())
                    };
                }
            }
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.coef.sub(&before);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.norm());
        q.parameter_delta_max = Some(delta.max_abs());
        q.loss_before = Some(loss_before);
        q.loss_after = Some(loss_after);
        q.information_gain = Some(info);
        q.still_identified = self.n_seen as usize > x.ncols();
        q.warmup = self.n_seen < 5;
        q.explanation = format!(
            "SGD {:?} update: {} rows, η={}, Δθ_ℓ2={:.4e}, loss {:.6e} → {:.6e}",
            self.loss,
            x.nrows(),
            self.learning_rate,
            delta.norm(),
            loss_before,
            loss_after
        );
        if q.is_uninformative(ctx.policy.uninformative_info_eps) {
            ctx.push(
                Issue::builder(IssueCode::UpdateWithZeroInformation)
                    .incremental(q.clone())
                    .message("this SGD classification batch did not move the parameters")
                    .build(),
            );
        }
        if q.warmup {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .incremental(q.clone())
                    .message("SgdClassifier n_seen < 5; scores are not inferential")
                    .build(),
            );
        }
        let expl = IncrementalExplain::from_quality(
            q,
            format!("coef[{}] and intercept", self.coef.len()),
            format!("{:?} subgradient plus ℓ2", self.loss),
            format!("loss={loss_before:.6e}"),
            format!("loss={loss_after:.6e}"),
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

impl Predict for SgdClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let pos = *self.classes.last().unwrap_or(&1) as f64;
        let neg = *self.classes.first().unwrap_or(&0) as f64;
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut s = self.intercept;
            for j in 0..x.ncols().min(self.coef.len()) {
                s += self.coef[j] * x.get(i, j);
            }
            if s >= 0.0 {
                pos
            } else {
                neg
            }
        }));
        ctx.finish(out)
    }
}

/// ε-insensitive passive-aggressive regressor (Crammer et al.).
#[derive(Clone, Debug)]
pub struct PassiveAggressiveRegressor {
    /// Aggressiveness \(C\).
    pub c: f64,
    /// Insensitivity tube \(\varepsilon\).
    pub epsilon: f64,
    coef: Vector,
    intercept: f64,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for PassiveAggressiveRegressor {
    fn default() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.1,
            coef: Vector::zeros(0),
            intercept: 0.0,
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl PassiveAggressiveRegressor {
    /// PA-I regressor with aggressiveness `c`.
    pub fn new(c: f64) -> Self {
        Self {
            c,
            ..Self::default()
        }
    }

    /// Current slopes.
    pub fn coef(&self) -> &Vector {
        &self.coef
    }
}

impl PartialFit for PassiveAggressiveRegressor {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(
                Issue::builder(IssueCode::MissingTarget)
                    .message("PassiveAggressiveRegressor.partial_fit requires y")
                    .build(),
            );
            return ctx.finish(dummy_explain(self.updates, 0, self.n_seen));
        };
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if !self.initialized {
            self.coef = Vector::zeros(x.ncols());
            self.initialized = true;
        } else if self.coef.len() != x.ncols() {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message("PA regressor feature count changed")
                    .build(),
            );
            return ctx.finish(dummy_explain(self.updates, x.nrows(), self.n_seen));
        }
        let before = self.coef.clone();
        let mut loss_before = 0.0;
        let mut n_hit = 0.0;
        for i in 0..x.nrows() {
            let mut pred = self.intercept;
            for j in 0..x.ncols() {
                pred += self.coef[j] * x.get(i, j);
            }
            let err = y[i] - pred;
            let slack = err.abs() - self.epsilon;
            loss_before += slack.max(0.0);
            if slack <= 0.0 {
                continue;
            }
            n_hit += 1.0;
            let mut nrm2 = 1.0;
            for j in 0..x.ncols() {
                let v = x.get(i, j);
                nrm2 += v * v;
            }
            let tau = (slack / nrm2).min(self.c);
            let sgn = if err >= 0.0 { 1.0 } else { -1.0 };
            for j in 0..x.ncols() {
                self.coef[j] += tau * sgn * x.get(i, j);
            }
            self.intercept += tau * sgn;
        }
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let delta = self.coef.sub(&before);
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.parameter_delta_norm = Some(delta.norm());
        q.parameter_delta_max = Some(delta.max_abs());
        q.loss_before = Some(loss_before);
        q.information_gain = Some(n_hit);
        q.still_identified = self.n_seen as usize > x.ncols();
        q.warmup = self.n_seen < 5;
        q.explanation = format!(
            "PA-I regressor: {n_hit} of {} rows outside ε={}, C={}, ||Δθ||={:.4e}",
            x.nrows(),
            self.epsilon,
            self.c,
            delta.norm()
        );
        if n_hit == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UpdateWithZeroInformation)
                    .incremental(q.clone())
                    .message("every residual was inside the ε-tube; PA did not move")
                    .build(),
            );
        }
        let expl = IncrementalExplain::from_quality(
            q,
            format!("coef[{}] and intercept", self.coef.len()),
            "PA-I ε-insensitive closed-form step",
            format!("tube_loss={loss_before:.6e}"),
            format!("rows_outside_tube={n_hit}"),
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

impl Predict for PassiveAggressiveRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if !self.initialized {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            y[i] += self.intercept;
        }
        ctx.finish(y)
    }
}

/// NB2 log-likelihood helper used by tests / diagnostics.
pub fn nb2_loglik(y: &Vector, mu: &Vector, alpha: f64) -> f64 {
    let a = alpha.max(1e-12);
    let mut s = 0.0;
    for i in 0..y.len().min(mu.len()) {
        let m = mu[i].max(1e-12);
        let r = 1.0 / a;
        // y log(μ/(μ+r)) + r log(r/(μ+r)) + ln Γ(y+r) − ln Γ(r) − ln Γ(y+1)
        s += y[i] * (m / (m + r)).ln() + r * (r / (m + r)).ln() + ln_gamma(y[i] + r)
            - ln_gamma(r)
            - ln_gamma(y[i] + 1.0);
    }
    s
}

/// Ordered logit (cumulative / proportional-odds) via gradient ascent.
///
/// Classes must be ordered by their integer labels. A single class leaves
/// \((\beta,\theta)\) unidentified.
#[derive(Clone, Debug)]
pub struct OrderedLogit {
    /// Learning rate.
    pub eta: f64,
    /// Max gradient steps.
    pub max_iter: usize,
    /// ℓ₂ on the slopes.
    pub ridge: f64,
}

impl Default for OrderedLogit {
    fn default() -> Self {
        Self {
            eta: 0.05,
            max_iter: 200,
            ridge: 1e-3,
        }
    }
}

impl OrderedLogit {
    /// Default ordered logit.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted ordered logit.
#[derive(Clone, Debug)]
pub struct FittedOrderedLogit {
    /// Slopes.
    pub coef: Vector,
    /// Increasing cutpoints \(\theta_1 < \cdots < \theta_{K-1}\).
    pub thresholds: Vector,
    /// Sorted class labels.
    pub classes: Vec<i64>,
}

impl FittedOrderedLogit {
    /// Predicted class (argmax of category probabilities).
    pub fn predict_label(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("ordered logit column count ≠ coef")
                    .build(),
            );
        }
        let k = self.classes.len().max(1);
        let mut out = Vector::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut xb = 0.0;
            for j in 0..self.coef.len().min(x.ncols()) {
                xb += self.coef[j] * x.get(i, j);
            }
            let mut best = 0usize;
            let mut bp = f64::NEG_INFINITY;
            for c in 0..k {
                let lo = if c == 0 {
                    0.0
                } else {
                    sigmoid(self.thresholds[c - 1] - xb)
                };
                let hi = if c + 1 == k {
                    1.0
                } else {
                    sigmoid(self.thresholds[c] - xb)
                };
                let p = (hi - lo).max(0.0);
                if p > bp {
                    bp = p;
                    best = c;
                }
            }
            out[i] = self.classes.get(best).copied().unwrap_or(0) as f64;
        }
        ctx.finish(out)
    }
}

impl Fit for OrderedLogit {
    type Fitted = FittedOrderedLogit;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedOrderedLogit>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        if classes.len() < 2 {
            return ctx.finish(FittedOrderedLogit {
                coef: Vector::zeros(x.ncols()),
                thresholds: Vector::zeros(0),
                classes,
            });
        }
        let k = classes.len();
        let mut yidx = vec![0usize; y.len()];
        for i in 0..y.len() {
            let lab = y[i].round() as i64;
            yidx[i] = classes.iter().position(|&c| c == lab).unwrap_or(0);
        }
        let p = x.ncols();
        let mut coef = Vector::zeros(p);
        let mut thr = Vector::zeros(k - 1);
        for t in 0..k - 1 {
            let cum = counts.iter().take(t + 1).map(|(_, c)| *c).sum::<usize>() as f64
                / y.len().max(1) as f64;
            let q = cum.clamp(1e-3, 1.0 - 1e-3);
            thr[t] = (q / (1.0 - q)).ln();
            if t > 0 && thr[t] <= thr[t - 1] {
                thr[t] = thr[t - 1] + 0.2;
            }
        }
        let eta = self.eta.max(1e-6);
        let ridge = self.ridge.max(0.0);
        for it in 0..self.max_iter.max(1) {
            let mut g_b = Vector::zeros(p);
            let mut g_t = Vector::zeros(k - 1);
            let mut ll = 0.0;
            for i in 0..x.nrows().min(yidx.len()) {
                let mut xb = 0.0;
                for j in 0..p {
                    xb += coef[j] * x.get(i, j);
                }
                let c = yidx[i];
                let (p_c, d_beta, d_thr) = ordered_prob_grad(c, k, xb, &thr);
                ll += (p_c.max(1e-15)).ln();
                let inv = 1.0 / p_c.max(1e-15);
                for j in 0..p {
                    g_b[j] += inv * d_beta * x.get(i, j);
                }
                for t in 0..k - 1 {
                    g_t[t] += inv * d_thr[t];
                }
            }
            for j in 0..p {
                g_b[j] -= ridge * coef[j];
                coef[j] += eta * g_b[j] / x.nrows().max(1) as f64;
            }
            for t in 0..k - 1 {
                thr[t] += eta * g_t[t] / x.nrows().max(1) as f64;
            }
            for t in 1..k - 1 {
                if thr[t] <= thr[t - 1] + 1e-4 {
                    thr[t] = thr[t - 1] + 1e-4;
                }
            }
            ctx.session.step(it as u64, -ll, None);
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(Severity::Advisory)
                .message(
                    "ordered logit SEs are not reported; this is a gradient point, not IRLS MLE",
                )
                .compromise(NumericalCompromise::new(
                    "IRLS / Newton ordered logit",
                    "gradient ascent on the proportional-odds likelihood",
                    "the information matrix is not inverted",
                    "thresholds are ordered by projection, not by a constrained Hessian",
                ))
                .build(),
        );
        ctx.finish(FittedOrderedLogit {
            coef,
            thresholds: thr,
            classes,
        })
    }
}

fn ordered_prob_grad(c: usize, k: usize, xb: f64, thr: &Vector) -> (f64, f64, Vector) {
    let mut d_thr = Vector::zeros(k.saturating_sub(1));
    let sig = |z: f64| sigmoid(z);
    let dsig = |z: f64| {
        let s = sigmoid(z);
        s * (1.0 - s)
    };
    if k < 2 {
        return (1.0, 0.0, d_thr);
    }
    if c == 0 {
        let z = thr[0] - xb;
        let p = sig(z);
        d_thr[0] = dsig(z);
        return (p, -dsig(z), d_thr);
    }
    if c + 1 == k {
        let z = thr[k - 2] - xb;
        let p = 1.0 - sig(z);
        d_thr[k - 2] = -dsig(z);
        return (p, dsig(z), d_thr);
    }
    let zu = thr[c] - xb;
    let zl = thr[c - 1] - xb;
    let p = sig(zu) - sig(zl);
    d_thr[c] = dsig(zu);
    d_thr[c - 1] = -dsig(zl);
    (p, -(dsig(zu) - dsig(zl)), d_thr)
}

/// Linear GEE with exchangeable working correlation (Liang–Zeger).
#[derive(Clone, Debug)]
pub struct Gee {
    /// IRLS / GLS iterations.
    pub max_iter: usize,
    /// Include an intercept.
    pub fit_intercept: bool,
}

impl Default for Gee {
    fn default() -> Self {
        Self {
            max_iter: 25,
            fit_intercept: true,
        }
    }
}

impl Gee {
    /// Default exchangeable linear GEE.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit `y | groups` under an exchangeable working `V`.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedGee>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if groups.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("GEE groups length ≠ n")
                    .build(),
            );
            return ctx.finish(FittedGee {
                coef: Vector::zeros(x.ncols()),
                intercept: y.mean(),
                sigma2: f64::NAN,
                rho: f64::NAN,
                n_groups: 0,
            });
        }
        let mut members: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (i, &g) in groups.as_slice().iter().enumerate() {
            if !g.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteInput)
                        .message("GEE group labels contain NaN/Inf")
                        .build(),
                );
                break;
            }
            members.entry(g.round() as i64).or_default().push(i);
        }
        let n_groups = members.len();
        if n_groups <= 1 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("exchangeable GEE is unidentified with a single group")
                    .meaninglessness(Meaninglessness::vacuous(
                        "working correlation ρ",
                        "one cluster cannot separate ρ from the residual scale",
                        "collect more groups",
                    ))
                    .build(),
            );
            return ctx.finish(FittedGee {
                coef: Vector::zeros(x.ncols()),
                intercept: y.mean(),
                sigma2: f64::NAN,
                rho: f64::NAN,
                n_groups,
            });
        }
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        let p = design.ncols();
        let mut scratch = signlred::Report::new("gee", "ols");
        let mut beta = least_squares(&mut scratch, &design, y, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(p));
        let mut sigma2 = 1.0;
        let mut rho = 0.1;
        for it in 0..self.max_iter.max(1) {
            if let Some(b) = gee_gls(&design, y, &members, sigma2, rho, &ctx.policy) {
                beta = b;
            } else {
                ctx.push(
                    Issue::builder(IssueCode::CholeskyFailed)
                        .severity(Severity::Warning)
                        .message("GEE GLS Hessian refused Cholesky")
                        .build(),
                );
                break;
            }
            let (s2, r) = gee_moments(&design, y, &beta, &members);
            sigma2 = s2;
            rho = r.clamp(-0.99, 0.99);
            ctx.session.step(it as u64, s2, Some(r.abs()));
        }
        if rho.abs() >= 0.99 {
            ctx.push(
                Issue::builder(IssueCode::DegenerateDistribution)
                    .message("GEE ρ hit the ±0.99 bound; the working V is nearly singular")
                    .metric("rho", rho)
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(Severity::Advisory)
                .message("GEE reports a working-correlation point, not a robust sandwich")
                .compromise(NumericalCompromise::new(
                    "Liang–Zeger sandwich GEE",
                    "iterated GLS with exchangeable V and moment ρ",
                    "the bread/meat sandwich is not formed",
                    "treat ρ as a working parameter, not as a causal intra-class correlation",
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
        ctx.finish(FittedGee {
            coef,
            intercept,
            sigma2,
            rho,
            n_groups,
        })
    }
}

/// Fitted exchangeable linear GEE.
#[derive(Clone, Debug)]
pub struct FittedGee {
    /// Fixed slopes.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
    /// Residual scale.
    pub sigma2: f64,
    /// Working exchangeable correlation.
    pub rho: f64,
    /// Number of groups.
    pub n_groups: usize,
}

impl Predict for FittedGee {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("GEE predict column count ≠ coef")
                    .build(),
            );
        }
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            y[i] += self.intercept;
        }
        ctx.finish(y)
    }
}

/// Zero-inflated Poisson (Lambert): intercept-only inflate + Poisson count GLM.
#[derive(Clone, Debug)]
pub struct ZeroInflatedPoisson {
    /// EM / IRLS cycles.
    pub max_iter: usize,
    /// Count-model intercept.
    pub fit_intercept: bool,
}

impl Default for ZeroInflatedPoisson {
    fn default() -> Self {
        Self {
            max_iter: 25,
            fit_intercept: true,
        }
    }
}

impl ZeroInflatedPoisson {
    /// Default ZIP.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted ZIP.
#[derive(Clone, Debug)]
pub struct FittedZip {
    /// Count-model slopes.
    pub coef: Vector,
    /// Count-model intercept.
    pub intercept: f64,
    /// Structural-zero probability (intercept-only).
    pub inflate_pi: f64,
}

impl Predict for FittedZip {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("ZIP predict column count ≠ coef")
                    .build(),
            );
        }
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            let mu = (y[i] + self.intercept).exp().max(1e-12);
            y[i] = (1.0 - self.inflate_pi) * mu;
        }
        ctx.finish(y)
    }
}

impl Fit for ZeroInflatedPoisson {
    type Fitted = FittedZip;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedZip>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        for (i, &yi) in y.as_slice().iter().enumerate() {
            if yi < 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NonPositiveSeries)
                        .message(format!("ZIP y[{i}]={yi} < 0"))
                        .build(),
                );
                break;
            }
        }
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
        let n = y.len();
        let n0 = y.as_slice().iter().filter(|v| **v <= 0.0).count() as f64;
        let mut pi = (n0 / n.max(1) as f64).clamp(0.05, 0.8);
        let mut beta = Vector::zeros(design.ncols());
        if self.fit_intercept && !beta.is_empty() {
            let ypos: Vec<f64> = y.as_slice().iter().copied().filter(|v| *v > 0.0).collect();
            let m = if ypos.is_empty() {
                1.0
            } else {
                ypos.iter().sum::<f64>() / ypos.len() as f64
            };
            beta[0] = m.max(1e-3).ln();
        }
        if n0 == n as f64 {
            ctx.push(
                Issue::builder(IssueCode::DegenerateDistribution)
                    .message("ZIP: every count is zero")
                    .meaninglessness(Meaninglessness::vacuous(
                        "ZIP rates",
                        "a zero series does not identify a count mean",
                        "collect positive counts",
                    ))
                    .build(),
            );
        }
        if n0 == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::MixtureWeightCollapsed)
                    .message("ZIP inflate is unidentified: there are no zeros")
                    .build(),
            );
            pi = 0.0;
        }
        for it in 0..self.max_iter.max(1) {
            let mut z = Vector::zeros(n);
            let mut tau_sum = 0.0;
            for i in 0..n {
                let mut eta = 0.0;
                for j in 0..design.ncols() {
                    eta += design.get(i, j) * beta[j];
                }
                let mu = eta.exp().max(1e-12);
                if y[i] <= 0.0 {
                    let p0 = (-mu).exp();
                    z[i] = pi / (pi + (1.0 - pi) * p0).max(1e-12);
                } else {
                    z[i] = 0.0;
                }
                tau_sum += z[i];
            }
            pi = (tau_sum / n as f64).clamp(1e-6, 1.0 - 1e-6);
            let mut xs = Matrix::zeros(n, design.ncols());
            let mut rhs = Vector::zeros(n);
            for i in 0..n {
                let mut eta = 0.0;
                for j in 0..design.ncols() {
                    eta += design.get(i, j) * beta[j];
                }
                let mu = eta.exp().max(1e-12);
                let w = ((1.0 - z[i]) * mu).max(1e-12);
                let sw = w.sqrt();
                rhs[i] = (eta + (y[i] - mu) / mu) * sw;
                for j in 0..design.ncols() {
                    xs.set(i, j, design.get(i, j) * sw);
                }
            }
            let mut scratch = signlred::Report::new("zip", "irls");
            let Some(next) = least_squares(&mut scratch, &xs, &rhs, &ctx.policy) else {
                break;
            };
            for issue in scratch.issues() {
                if matches!(
                    issue.code,
                    IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
                ) {
                    continue;
                }
                ctx.push(issue.clone());
            }
            let d = next.sub(&beta).norm();
            beta = next;
            ctx.session.step(it as u64, d, Some(pi));
            if d < 1e-7 {
                ctx.session.converged("ZIP EM", it as u64);
                break;
            }
        }
        let (intercept, coef) = if self.fit_intercept {
            (
                beta.as_slice().first().copied().unwrap_or(0.0),
                Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            )
        } else {
            (0.0, beta)
        };
        ctx.finish(FittedZip {
            coef,
            intercept,
            inflate_pi: pi,
        })
    }
}

/// Zero-inflated negative binomial (NB2 count + intercept-only inflate).
#[derive(Clone, Debug)]
pub struct ZeroInflatedNegativeBinomial {
    /// EM / IRLS cycles.
    pub max_iter: usize,
    /// Count-model intercept.
    pub fit_intercept: bool,
}

impl Default for ZeroInflatedNegativeBinomial {
    fn default() -> Self {
        Self {
            max_iter: 25,
            fit_intercept: true,
        }
    }
}

impl ZeroInflatedNegativeBinomial {
    /// Default ZINB.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted ZINB.
#[derive(Clone, Debug)]
pub struct FittedZinb {
    /// Count-model slopes.
    pub coef: Vector,
    /// Count-model intercept.
    pub intercept: f64,
    /// Structural-zero probability.
    pub inflate_pi: f64,
    /// NB2 dispersion \(\alpha > 0\) (\(\mathrm{Var}=\mu+\alpha\mu^2\)).
    pub alpha: f64,
}

impl Predict for FittedZinb {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("ZINB predict column count ≠ coef")
                    .build(),
            );
        }
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            let mu = (y[i] + self.intercept).exp().max(1e-12);
            y[i] = (1.0 - self.inflate_pi) * mu;
        }
        ctx.finish(y)
    }
}

impl Fit for ZeroInflatedNegativeBinomial {
    type Fitted = FittedZinb;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedZinb>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        for (i, &yi) in y.as_slice().iter().enumerate() {
            if yi < 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NonPositiveSeries)
                        .message(format!("ZINB y[{i}]={yi} < 0"))
                        .build(),
                );
                break;
            }
        }
        let design = if self.fit_intercept {
            x.with_intercept()
        } else {
            x.clone()
        };
        inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
        let n = y.len();
        let n0 = y.as_slice().iter().filter(|v| **v <= 0.0).count() as f64;
        let mut pi = (n0 / n.max(1) as f64).clamp(0.05, 0.8);
        let mean = y.mean().max(1e-3);
        let var = y.std() * y.std();
        let mut alpha = ((var / mean - 1.0) / mean).clamp(1e-3, 20.0);
        let mut beta = Vector::zeros(design.ncols());
        if self.fit_intercept && !beta.is_empty() {
            beta[0] = mean.ln();
        }
        if n0 == n as f64 {
            ctx.push(
                Issue::builder(IssueCode::DegenerateDistribution)
                    .message("ZINB: every count is zero")
                    .meaninglessness(Meaninglessness::vacuous(
                        "ZINB rates",
                        "a zero series does not identify a count mean or α",
                        "collect positive counts",
                    ))
                    .build(),
            );
        }
        if n0 == 0.0 {
            ctx.push(
                Issue::builder(IssueCode::MixtureWeightCollapsed)
                    .message("ZINB inflate is unidentified: there are no zeros")
                    .build(),
            );
            pi = 0.0;
        }
        for it in 0..self.max_iter.max(1) {
            let r = 1.0 / alpha.max(1e-6);
            let mut z = Vector::zeros(n);
            let mut tau_sum = 0.0;
            for i in 0..n {
                let mut eta = 0.0;
                for j in 0..design.ncols() {
                    eta += design.get(i, j) * beta[j];
                }
                let mu = eta.exp().max(1e-12);
                if y[i] <= 0.0 {
                    let p0 = (r / (r + mu)).powf(r);
                    z[i] = pi / (pi + (1.0 - pi) * p0).max(1e-12);
                } else {
                    z[i] = 0.0;
                }
                tau_sum += z[i];
            }
            pi = (tau_sum / n as f64).clamp(1e-6, 1.0 - 1e-6);
            let mut xs = Matrix::zeros(n, design.ncols());
            let mut rhs = Vector::zeros(n);
            let mut mom_num = 0.0;
            let mut mom_den = 0.0;
            for i in 0..n {
                let mut eta = 0.0;
                for j in 0..design.ncols() {
                    eta += design.get(i, j) * beta[j];
                }
                let mu = eta.exp().max(1e-12);
                let var_i = (mu + alpha * mu * mu).max(1e-8);
                let w = ((1.0 - z[i]) * mu * mu / var_i).max(1e-12);
                let sw = w.sqrt();
                rhs[i] = (eta + (y[i] - mu) / mu) * sw;
                for j in 0..design.ncols() {
                    xs.set(i, j, design.get(i, j) * sw);
                }
                let wi = 1.0 - z[i];
                mom_num += wi * (y[i] - mu) * (y[i] - mu);
                mom_den += wi * mu * mu;
            }
            if mom_den > 0.0 {
                alpha = ((mom_num / mom_den.max(1e-8)) - 1.0 / mean)
                    .abs()
                    .clamp(1e-4, 20.0);
            }
            let mut scratch = signlred::Report::new("zinb", "irls");
            let Some(next) = least_squares(&mut scratch, &xs, &rhs, &ctx.policy) else {
                break;
            };
            for issue in scratch.issues() {
                if matches!(
                    issue.code,
                    IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
                ) {
                    continue;
                }
                ctx.push(issue.clone());
            }
            let d = next.sub(&beta).norm();
            beta = next;
            ctx.session.step(it as u64, d, Some(pi));
            if d < 1e-7 {
                ctx.session.converged("ZINB EM", it as u64);
                break;
            }
        }
        let (intercept, coef) = if self.fit_intercept {
            (
                beta.as_slice().first().copied().unwrap_or(0.0),
                Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            )
        } else {
            (0.0, beta)
        };
        ctx.finish(FittedZinb {
            coef,
            intercept,
            inflate_pi: pi,
            alpha,
        })
    }
}

/// Weibull AFT (uncensored): \(\log T = x^\top\beta + \sigma\varepsilon\), \(\varepsilon\sim\) Gumbel.
#[derive(Clone, Debug)]
pub struct WeibullAft {
    /// Gradient steps.
    pub max_iter: usize,
}

impl Default for WeibullAft {
    fn default() -> Self {
        Self { max_iter: 40 }
    }
}

impl WeibullAft {
    /// Default Weibull AFT.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Weibull AFT.
#[derive(Clone, Debug)]
pub struct FittedWeibullAft {
    /// Slopes.
    pub coef: Vector,
    /// Intercept on the log-time scale.
    pub intercept: f64,
    /// Scale \(\sigma > 0\).
    pub sigma: f64,
}

impl Predict for FittedWeibullAft {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("WeibullAFT predict column count ≠ coef")
                    .build(),
            );
        }
        let mut y = x.matvec(&self.coef);
        for i in 0..y.len() {
            y[i] = (y[i] + self.intercept).exp();
        }
        ctx.finish(y)
    }
}

impl Fit for WeibullAft {
    type Fitted = FittedWeibullAft;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedWeibullAft>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        for (i, &yi) in y.as_slice().iter().enumerate() {
            if yi <= 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NonPositiveSeries)
                        .message(format!("Weibull AFT y[{i}]={yi} is not strictly positive"))
                        .build(),
                );
                break;
            }
        }
        let design = x.with_intercept();
        inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
        let n = y.len().min(design.nrows());
        let logy = Vector::from_iter(y.as_slice().iter().map(|v| v.max(1e-12).ln()));
        let mut beta = Vector::zeros(design.ncols());
        beta[0] = logy.mean();
        let mut log_sigma = logy.std().max(0.1).ln().clamp(-2.0, 2.0);
        for it in 0..self.max_iter.max(1) {
            let sigma = log_sigma.exp().clamp(1e-3, 10.0);
            let mut g_beta = Vector::zeros(design.ncols());
            let mut g_ls = 0.0;
            for i in 0..n {
                let mut eta = 0.0;
                for j in 0..design.ncols() {
                    eta += design.get(i, j) * beta[j];
                }
                let z = ((logy[i] - eta) / sigma).clamp(-20.0, 20.0);
                let ez = z.exp();
                for j in 0..design.ncols() {
                    g_beta[j] += (ez - 1.0) * design.get(i, j) / sigma;
                }
                g_ls += -1.0 + (ez - 1.0) * z;
            }
            let step = 0.01 / (n as f64).sqrt();
            let mut next = beta.clone();
            for j in 0..next.len() {
                let g = g_beta[j].clamp(-1e3, 1e3);
                next[j] += step * g;
            }
            let next_ls = (log_sigma + step * (g_ls / n as f64).clamp(-1e3, 1e3)).clamp(-2.0, 2.0);
            if next.as_slice().iter().all(|v| v.is_finite()) && next_ls.is_finite() {
                beta = next;
                log_sigma = next_ls;
            } else {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("Weibull AFT step was non-finite; last finite iterate kept")
                        .build(),
                );
                break;
            }
            let gn = g_beta.norm() + g_ls.abs();
            ctx.session.step(it as u64, gn, None);
            if gn < 1e-4 {
                ctx.session.converged("Weibull AFT gradient", it as u64);
                break;
            }
        }
        let sigma = log_sigma.exp().clamp(1e-3, 10.0);
        if !sigma.is_finite() || beta.as_slice().iter().any(|v| !v.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("Weibull AFT ended on a non-finite iterate; parameters were clamped")
                    .build(),
            );
            beta = Vector::zeros(design.ncols());
            beta[0] = logy.mean();
        }
        ctx.finish(FittedWeibullAft {
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            sigma,
        })
    }
}

/// Exponential AFT: Weibull AFT with fixed scale `σ = 1` (log-OLS on `log T`).
///
/// `y ≤ 0` is [`IssueCode::NonPositiveSeries`] (Error).
#[derive(Clone, Debug)]
pub struct ExponentialAft {
    /// Unused (kept for a Weibull-compatible constructor surface).
    pub max_iter: usize,
}

impl Default for ExponentialAft {
    fn default() -> Self {
        Self { max_iter: 1 }
    }
}

impl ExponentialAft {
    /// Default exponential AFT.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for ExponentialAft {
    type Fitted = FittedWeibullAft;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedWeibullAft>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        for (i, &yi) in y.as_slice().iter().enumerate() {
            if yi <= 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NonPositiveSeries)
                        .message(format!(
                            "Exponential AFT y[{i}]={yi} is not strictly positive"
                        ))
                        .build(),
                );
                break;
            }
        }
        let design = x.with_intercept();
        inspect_identification(&mut ctx.report, design.nrows(), design.ncols(), &ctx.policy);
        let logy = Vector::from_iter(y.as_slice().iter().map(|v| v.max(1e-12).ln()));
        let mut scratch = signlred::Report::new("exp_aft", "ols");
        let beta = least_squares(&mut scratch, &design, &logy, &ctx.policy).unwrap_or_else(|| {
            let mut b = Vector::zeros(design.ncols());
            b[0] = logy.mean();
            b
        });
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Exponential AFT fixes σ=1; only the log-mean is estimated")
                .compromise(NumericalCompromise::new(
                    "Weibull AFT with free scale",
                    "log-OLS with σ pinned at 1",
                    "the residual scale is not identified from the data",
                    "do not read this as a full Weibull MLE",
                ))
                .build(),
        );
        ctx.finish(FittedWeibullAft {
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            sigma: 1.0,
        })
    }
}

fn gee_gls(
    design: &Matrix,
    y: &Vector,
    members: &BTreeMap<i64, Vec<usize>>,
    sigma2: f64,
    rho: f64,
    policy: &signlred::Policy,
) -> Option<Vector> {
    let p = design.ncols();
    let mut xtvx = vec![0.0; p * p];
    let mut xtvy = Vector::zeros(p);
    let s2 = sigma2.max(1e-12);
    let r = rho.clamp(-0.99, 0.99);
    let om = (1.0 - r).max(1e-8);
    for idx in members.values() {
        let ng = idx.len() as f64;
        let a = 1.0 / (s2 * om);
        let b = r / (s2 * om * (om + ng * r).max(1e-12));
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
    let mut scratch = signlred::Report::new("gee", "gls");
    chol_solve(&mut scratch, &a, &xtvy, policy)
}

fn gee_moments(
    design: &Matrix,
    y: &Vector,
    beta: &Vector,
    members: &BTreeMap<i64, Vec<usize>>,
) -> (f64, f64) {
    let mut sse = 0.0;
    let mut n: f64 = 0.0;
    let mut num = 0.0;
    let mut pairs = 0.0;
    for idx in members.values() {
        let mut e = Vec::with_capacity(idx.len());
        for &i in idx {
            let mut xb = 0.0;
            for j in 0..beta.len().min(design.ncols()) {
                xb += design.get(i, j) * beta[j];
            }
            let ei = y[i] - xb;
            e.push(ei);
            sse += ei * ei;
            n += 1.0;
        }
        for a in 0..e.len() {
            for b in (a + 1)..e.len() {
                num += e[a] * e[b];
                pairs += 1.0;
            }
        }
    }
    let sigma2 = (sse / n.max(1.0)).max(1e-12);
    let rho = if pairs > 0.0 {
        num / (pairs * sigma2)
    } else {
        0.0
    };
    (sigma2, rho)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::PartialFit;

    #[test]
    fn probit_separates_overlapping_blobs() {
        let x = Matrix::from_fn(24, 1, |i, _| {
            if i < 12 {
                -1.2 + 0.15 * ((i % 4) as f64)
            } else {
                1.2 + 0.15 * ((i % 4) as f64)
            }
        });
        let y = Vector::from_iter((0..24).map(|i| if i < 12 { 0.0 } else { 1.0 }));
        let q = ProbitRegression::new()
            .fit(&x, &y, &Session::new("probit", "fit"))
            .expect("probit");
        let score = q
            .value
            .predict(&x, &Session::new("probit", "p"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..24 {
            let pred = if score[i] >= 0.0 { 1.0 } else { 0.0 };
            if (pred - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 20, "ok={ok}");
    }

    #[test]
    fn nb2_fits_a_log_mean() {
        let x = Matrix::from_fn(16, 1, |i, _| (i as f64) * 0.1);
        let y = Vector::from_iter((0..16).map(|i| (2.0 * (0.1 * i as f64).exp()).round().max(0.0)));
        let q = NegativeBinomialRegressor::new()
            .fit(&x, &y, &Session::new("nb2", "fit"))
            .expect("nb2");
        assert!(q.value.coef[0].is_finite());
        assert!(q.value.dispersion.is_finite());
    }

    #[test]
    fn sgd_and_pa_online() {
        let x = Matrix::from_fn(20, 1, |i, _| if i < 10 { -1.5 } else { 1.5 });
        let yb = Vector::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        let y = Vector::from_iter((0..20).map(|i| 0.5 * i as f64));
        let mut clf = SgdClassifier::new();
        clf.partial_fit(&x, Some(&yb), &Session::new("sgd", "pf"))
            .expect("sgd");
        let pred = clf.predict(&x, &Session::new("sgd", "p")).unwrap().value;
        assert_eq!(pred.len(), 20);
        let mut pa = PassiveAggressiveRegressor::new(1.0);
        pa.partial_fit(&x, Some(&y), &Session::new("pa", "pf"))
            .expect("pa");
        let hat = pa.predict(&x, &Session::new("pa", "p")).unwrap().value;
        assert!(hat.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ordered_logit_ranks_a_line() {
        let x = Matrix::from_fn(24, 1, |i, _| i as f64 + 0.3 * ((i % 3) as f64));
        let y = Vector::from_iter((0..24).map(|i| {
            if i < 8 {
                0.0
            } else if i < 16 {
                1.0
            } else {
                2.0
            }
        }));
        let q = OrderedLogit::new()
            .fit(&x, &y, &Session::new("ol", "fit"))
            .expect("ol");
        let pred = q
            .value
            .predict_label(&x, &Session::new("ol", "p"))
            .unwrap()
            .value;
        assert!((pred[0] - 0.0).abs() < 0.5 || (pred[1] - 0.0).abs() < 0.5);
        assert!((pred[23] - 2.0).abs() < 0.5 || (pred[22] - 2.0).abs() < 0.5);
        assert_eq!(q.value.thresholds.len(), 2);
        assert!(q.value.thresholds[1] > q.value.thresholds[0]);
    }

    #[test]
    fn gee_recovers_slope() {
        let x = Matrix::from_fn(10, 1, |i, _| (i % 5) as f64);
        let y = Vector::from_iter((0..10).map(|i| {
            let u = if i < 5 { 5.0 } else { -5.0 };
            2.0 * (i % 5) as f64 + u
        }));
        let g = Vector::from_iter((0..10).map(|i| if i < 5 { 0.0 } else { 1.0 }));
        let q = Gee::new()
            .fit(&x, &y, &g, &Session::new("gee", "fit"))
            .expect("gee");
        assert!(
            (q.value.coef[0] - 2.0).abs() < 0.25,
            "{:?}",
            q.value.coef.as_slice()
        );
        assert!(q.value.rho.is_finite());
        assert_eq!(q.value.n_groups, 2);
    }

    #[test]
    fn zip_and_weibull() {
        let x = Matrix::from_fn(30, 1, |i, _| (i % 6) as f64);
        let y = Vector::from_iter((0..30).map(|i| {
            if i % 5 == 0 {
                0.0
            } else {
                (0.4 * (i % 6) as f64).exp().round()
            }
        }));
        let z = ZeroInflatedPoisson::new()
            .fit(&x, &y, &Session::new("zip", "fit"))
            .expect("zip");
        assert!(z.value.inflate_pi > 0.0 && z.value.inflate_pi < 1.0);
        let yt = Vector::from_iter((0..30).map(|i| (0.8 + 0.15 * (i % 6) as f64).exp()));
        let w = WeibullAft::new()
            .fit(&x, &yt, &Session::new("aft", "fit"))
            .expect("aft");
        assert!(w.value.sigma > 0.0 && w.value.sigma.is_finite());
        let pred = w
            .value
            .predict(&x, &Session::new("aft", "p"))
            .unwrap()
            .value;
        assert!(pred.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let zb = ZeroInflatedNegativeBinomial::new()
            .fit(&x, &y, &Session::new("zinb", "fit"))
            .expect("zinb");
        assert!(zb.value.inflate_pi >= 0.0 && zb.value.inflate_pi <= 1.0);
        assert!(zb.value.alpha > 0.0 && zb.value.alpha.is_finite());
        let ea = ExponentialAft::new()
            .fit(&x, &yt, &Session::new("eaft", "fit"))
            .expect("eaft");
        assert!((ea.value.sigma - 1.0).abs() < 1e-12);
        let ep = ea
            .value
            .predict(&x, &Session::new("eaft", "p"))
            .unwrap()
            .value;
        assert!(ep.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
    }
}
