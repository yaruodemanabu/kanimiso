//! Extra GLMs and linear online classifiers (probit, NB2, SGD, PA regression).
//!
//! Probit and negative-binomial fits are IRLS. Perfect separation and a
//! non-positive count series abort. SGD / PA expose
//! [`ojizou_san::IncrementalExplain`] on every `partial_fit`.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::least_squares;
use crate::special::{ln_gamma, norm_cdf};
use crate::traits::{Fit, PartialFit, Predict};
use crate::validate::{inspect_classes, inspect_identification, inspect_xy};
use ojizou_san::{IncrementalExplain, Session};
use signlred::{
    IncrementalQuality, InterpretiveValue, Issue, IssueCode, Meaninglessness, NumericalCompromise,
    Qualified, Result, Severity,
};

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
}
