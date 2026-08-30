//! Classifier façade: re-exports plus ridge, perceptron, PA, and a dummy.
//!
//! Re-exports the crate's tree / SVM / naïve-Bayes / logistic estimators so
//! a single module matches the sklearn `sklearn.linear_model` + `svm` + `tree`
//! + `naive_bayes` import surface.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linear_model::{FittedPenalized, Ridge};
use crate::traits::{Fit, Predict};
use crate::validate::{inspect_classes, inspect_xy};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Qualified, Result};

pub use crate::glm::{FittedGlm, ProbitRegression, SgdClassifier, SgdLoss};
pub use crate::histgb::{FittedHistGbc, HistGradientBoostingClassifier};
pub use crate::linear_model::{FittedLogistic, LogisticRegression};
pub use crate::naive_bayes::{
    BernoulliNB, CategoricalNB, ComplementNB, FittedBernoulliNB, FittedCategoricalNB,
    FittedDiscreteNB, FittedGaussianNB, GaussianNB, MultinomialNB,
};
pub use crate::svm::{FittedLinearSvc, FittedSvc, LinearSvc, Svc};
pub use crate::tree::{
    AdaBoostClassifier, AdaBoostRegressor, DecisionTreeClassifier, ExtraTreesClassifier,
    FittedAdaBoost, FittedAdaBoostRegressor, FittedForestClassifier, FittedGbc,
    FittedTreeClassifier, GradientBoostingClassifier, RandomForestClassifier,
};

fn labels_of(y: &Vector) -> Vec<i64> {
    y.as_slice()
        .iter()
        .map(|&v| if v.is_finite() { v.round() as i64 } else { 0 })
        .collect()
}

fn to_pm(y: f64, classes: &[i64]) -> f64 {
    let lab = y.round() as i64;
    if classes.len() >= 2 && lab == classes[classes.len() - 1] {
        1.0
    } else if classes.len() == 2 && lab == classes[0] {
        -1.0
    } else if y >= 0.5 {
        1.0
    } else {
        -1.0
    }
}

pub(crate) fn from_score(s: f64, classes: &[i64]) -> f64 {
    let pos = *classes.last().unwrap_or(&1) as f64;
    let neg = *classes.first().unwrap_or(&0) as f64;
    if s >= 0.0 {
        pos
    } else {
        neg
    }
}

/// Ridge classifier: sign of a ridge regressor on `±1` labels.
#[derive(Clone, Debug)]
pub struct RidgeClassifier {
    /// ℓ₂ penalty.
    pub alpha: f64,
}

impl Default for RidgeClassifier {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl RidgeClassifier {
    /// Ridge classifier with the given `α`.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

/// Fitted ridge classifier.
#[derive(Clone, Debug)]
pub struct FittedRidgeClassifier {
    pub(crate) inner: FittedPenalized,
    /// Training classes (sorted).
    pub classes: Vec<i64>,
}

impl FittedRidgeClassifier {
    /// Build from an already-fitted ridge on `±1` labels.
    pub(crate) fn from_penalized(inner: FittedPenalized, classes: Vec<i64>) -> Self {
        Self { inner, classes }
    }

    /// Decision scores (ridge prediction on the `±1` scale).
    pub fn decision_function(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.inner.predict(x, session)
    }
}

impl Predict for FittedRidgeClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let raw = match self.inner.predict(x, &session.child("ridge")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vector::zeros(x.nrows())
            }
        };
        let y = Vector::from_iter(raw.as_slice().iter().map(|&s| from_score(s, &self.classes)));
        ctx.finish(y)
    }
}

impl Fit for RidgeClassifier {
    type Fitted = FittedRidgeClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRidgeClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        let ypm = Vector::from_iter(y.as_slice().iter().map(|&v| to_pm(v, &classes)));
        let inner = match Ridge::new(self.alpha).fit(x, &ypm, &session.child("ridge")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedPenalized {
                    coef: Vector::zeros(x.ncols()),
                    intercept: 0.0,
                    alpha: self.alpha,
                    l1_ratio: 0.0,
                }
            }
        };
        ctx.finish(FittedRidgeClassifier { inner, classes })
    }
}

/// Batch perceptron (Rosenblatt) on `±1` labels.
#[derive(Clone, Debug)]
pub struct Perceptron {
    /// Learning rate.
    pub eta0: f64,
    /// Passes over the data.
    pub max_iter: usize,
    /// Fit an intercept.
    pub fit_intercept: bool,
}

impl Default for Perceptron {
    fn default() -> Self {
        Self {
            eta0: 1.0,
            max_iter: 20,
            fit_intercept: true,
        }
    }
}

impl Perceptron {
    /// Default perceptron.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted linear classifier (`w, b`).
#[derive(Clone, Debug)]
pub struct FittedLinearClassifier {
    /// Slopes.
    pub coef: Vector,
    /// Intercept.
    pub intercept: f64,
    /// Training classes.
    pub classes: Vec<i64>,
}

impl Predict for FittedLinearClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.ncols() != self.coef.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("linear classifier predict shape mismatch")
                    .build(),
            );
        }
        let mut s = x.matvec(&self.coef);
        for i in 0..s.len() {
            s[i] = from_score(s[i] + self.intercept, &self.classes);
        }
        ctx.finish(s)
    }
}

impl Fit for Perceptron {
    type Fitted = FittedLinearClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinearClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        let (n, p) = x.shape();
        let mut w = Vector::zeros(p);
        let mut b = 0.0;
        let labs = labels_of(y);
        let mut converged = false;
        for it in 0..self.max_iter.max(1) {
            let mut err = 0usize;
            for i in 0..n {
                let yi = to_pm(labs[i] as f64, &classes);
                let mut s = b;
                for j in 0..p {
                    s += w[j] * x.get(i, j);
                }
                if yi * s <= 0.0 {
                    err += 1;
                    for j in 0..p {
                        w[j] += self.eta0 * yi * x.get(i, j);
                    }
                    if self.fit_intercept {
                        b += self.eta0 * yi;
                    }
                }
            }
            ctx.session.step(it as u64, err as f64, None);
            if err == 0 {
                ctx.session
                    .converged("perceptron zero training error", it as u64);
                converged = true;
                break;
            }
        }
        if !converged {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("perceptron did not linearly separate the sample")
                    .build(),
            );
        }
        ctx.finish(FittedLinearClassifier {
            coef: w,
            intercept: b,
            classes,
        })
    }
}

/// Batch passive-aggressive classifier (PA-I).
#[derive(Clone, Debug)]
pub struct PassiveAggressive {
    /// Aggressiveness `C` (step-size cap).
    pub c: f64,
    /// Passes over the data.
    pub max_iter: usize,
    /// Fit an intercept.
    pub fit_intercept: bool,
}

impl Default for PassiveAggressive {
    fn default() -> Self {
        Self {
            c: 1.0,
            max_iter: 20,
            fit_intercept: true,
        }
    }
}

impl PassiveAggressive {
    /// PA-I with aggressiveness `c`.
    pub fn new(c: f64) -> Self {
        Self {
            c,
            ..Self::default()
        }
    }
}

impl Fit for PassiveAggressive {
    type Fitted = FittedLinearClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLinearClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        let (n, p) = x.shape();
        let mut w = Vector::zeros(p);
        let mut b = 0.0;
        let labs = labels_of(y);
        for it in 0..self.max_iter.max(1) {
            let mut loss_sum = 0.0;
            for i in 0..n {
                let yi = to_pm(labs[i] as f64, &classes);
                let mut s = b;
                let mut nrm2 = if self.fit_intercept { 1.0 } else { 0.0 };
                for j in 0..p {
                    s += w[j] * x.get(i, j);
                    nrm2 += x.get(i, j) * x.get(i, j);
                }
                let loss = (1.0 - yi * s).max(0.0);
                loss_sum += loss;
                if loss > 0.0 && nrm2 > 0.0 {
                    let tau = (loss / nrm2).min(self.c.max(0.0));
                    for j in 0..p {
                        w[j] += tau * yi * x.get(i, j);
                    }
                    if self.fit_intercept {
                        b += tau * yi;
                    }
                }
            }
            ctx.session.step(it as u64, loss_sum, None);
        }
        ctx.finish(FittedLinearClassifier {
            coef: w,
            intercept: b,
            classes,
        })
    }
}

/// Dummy classifier: always predict the most frequent training label.
#[derive(Clone, Debug, Default)]
pub struct DummyClassifier {}

impl DummyClassifier {
    /// Most-frequent dummy.
    pub fn new() -> Self {
        Self {}
    }
}

/// Fitted dummy classifier.
#[derive(Clone, Debug)]
pub struct FittedDummyClassifier {
    /// Majority label.
    pub label: f64,
    /// Training classes.
    pub classes: Vec<i64>,
}

impl Predict for FittedDummyClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        ctx.finish(Vector::filled(x.nrows(), self.label))
    }
}

impl Fit for DummyClassifier {
    type Fitted = FittedDummyClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedDummyClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let (lab, _) = counts
            .iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
            .copied()
            .unwrap_or((0, 0));
        ctx.finish(FittedDummyClassifier {
            label: lab as f64,
            classes: counts.iter().map(|(c, _)| *c).collect(),
        })
    }
}

/// Platt scaling: \(P(y=1 \mid s) = \sigma(A s + B)\) (sklearn `CalibratedClassifierCV`).
///
/// `scores` are the uncalibrated decision values. A single class makes \(A, B\)
/// unidentified.
#[derive(Clone, Debug, Default)]
pub struct PlattCalibrator {
    /// Max IRLS iterations.
    pub max_iter: usize,
}

impl PlattCalibrator {
    /// Default Platt calibrator.
    pub fn new() -> Self {
        Self { max_iter: 40 }
    }

    /// Fit \(A, B\) on `(scores, labels)`.
    pub fn fit(
        &mut self,
        scores: &Vector,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPlatt>> {
        let x = Matrix::from_vector(scores);
        let mut lr = crate::linear_model::LogisticRegression {
            c_inv: 1e-6,
            max_iter: self.max_iter.max(1),
            ..crate::linear_model::LogisticRegression::default()
        };
        let q = lr.fit(&x, y, session)?;
        Ok(q.map(|m| FittedPlatt {
            a: if m.coef.is_empty() { 0.0 } else { m.coef[0] },
            b: m.intercept,
            classes: m.classes,
        }))
    }
}

/// Fitted Platt map.
#[derive(Clone, Debug)]
pub struct FittedPlatt {
    /// Slope on the score.
    pub a: f64,
    /// Intercept.
    pub b: f64,
    /// Training classes.
    pub classes: Vec<i64>,
}

/// Calibration map used by [`CalibratedClassifierCV`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Platt logistic map on OOF scores.
    Platt,
    /// Isotonic regression on OOF scores.
    Isotonic,
}

/// K-fold OOF calibration of a ridge classifier (sklearn `CalibratedClassifierCV`).
#[derive(Clone, Debug)]
pub struct CalibratedClassifierCV {
    /// How to map scores to probabilities.
    pub method: CalibrationMethod,
    /// Number of stratified folds for OOF scores.
    pub n_splits: usize,
    /// Base ridge penalty.
    pub alpha: f64,
}

impl Default for CalibratedClassifierCV {
    fn default() -> Self {
        Self {
            method: CalibrationMethod::Platt,
            n_splits: 3,
            alpha: 1.0,
        }
    }
}

impl CalibratedClassifierCV {
    /// Platt-calibrated ridge classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted calibrated classifier.
#[derive(Clone, Debug)]
pub struct FittedCalibrated {
    /// Base classifier refit on the full sample.
    pub base: FittedRidgeClassifier,
    platt: Option<FittedPlatt>,
    isotonic: Option<crate::linear_model::FittedIsotonic>,
    /// Training classes.
    pub classes: Vec<i64>,
}

impl FittedCalibrated {
    /// Calibrated \(P(\text{last class})\).
    pub fn predict_proba(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let scores = self.base.decision_function(x, &session.child("score"))?;
        if let Some(p) = &self.platt {
            return p.predict_proba(&scores.value, session);
        }
        if let Some(iso) = &self.isotonic {
            let ctx = FitCtx::with_session(session.child("isotonic"));
            let raw = iso.predict_1d(&scores.value);
            let out = Vector::from_iter(raw.as_slice().iter().map(|&v| v.clamp(0.0, 1.0)));
            return ctx.finish(out);
        }
        let ctx = FitCtx::with_session(session.child("fallback"));
        ctx.finish(Vector::from_iter(
            scores
                .value
                .as_slice()
                .iter()
                .map(|&s| 1.0 / (1.0 + (-s).exp())),
        ))
    }
}

impl Predict for FittedCalibrated {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let p = self.predict_proba(x, session)?;
        Ok(p.map(|prob| {
            Vector::from_iter(
                prob.as_slice()
                    .iter()
                    .map(|&v| from_score(if v >= 0.5 { 1.0 } else { -1.0 }, &self.classes)),
            )
        }))
    }
}

impl Fit for CalibratedClassifierCV {
    type Fitted = FittedCalibrated;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedCalibrated>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        if ctx.report.contains(IssueCode::SingleClass) || ctx.report.contains(IssueCode::EmptyClass)
        {
            return ctx.finish(FittedCalibrated {
                base: FittedRidgeClassifier::from_penalized(
                    FittedPenalized {
                        coef: Vector::zeros(x.ncols()),
                        intercept: 0.0,
                        alpha: self.alpha,
                        l1_ratio: 0.0,
                    },
                    counts.iter().map(|(c, _)| *c).collect(),
                ),
                platt: None,
                isotonic: None,
                classes: counts.iter().map(|(c, _)| *c).collect(),
            });
        }
        let splitter = crate::model_selection::StratifiedKFold::new(self.n_splits.max(2));
        let folds = match splitter.split(y, &session.child("cv")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vec::new()
            }
        };
        let mut oof_s = Vec::new();
        let mut oof_y = Vec::new();
        for (i, fold) in folds.iter().enumerate() {
            let xt = crate::model_selection::take_rows(x, &fold.train);
            let yt = crate::model_selection::take_vec(y, &fold.train);
            let xv = crate::model_selection::take_rows(x, &fold.test);
            let yv = crate::model_selection::take_vec(y, &fold.test);
            let mut base = RidgeClassifier::new(self.alpha);
            match base.fit(&xt, &yt, &session.child(format!("fold_{i}"))) {
                Ok(q) => match q.value.decision_function(&xv, &session.child("score")) {
                    Ok(s) => {
                        for t in 0..s.value.len().min(yv.len()) {
                            oof_s.push(s.value[t]);
                            oof_y.push(yv[t]);
                        }
                    }
                    Err(e) => ctx.push(e.primary),
                },
                Err(e) => ctx.push(e.primary),
            }
        }
        if oof_s.len() < 4 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(signlred::Severity::Warning)
                    .message("too few OOF scores; calibration map is barely identified")
                    .build(),
            );
        }
        let scores = Vector::from_iter(oof_s);
        let yo = Vector::from_iter(oof_y);
        let mut platt = None;
        let mut isotonic = None;
        match self.method {
            CalibrationMethod::Platt => {
                match PlattCalibrator::new().fit(&scores, &yo, &session.child("platt")) {
                    Ok(q) => platt = Some(q.value),
                    Err(e) => ctx.push(e.primary),
                }
            }
            CalibrationMethod::Isotonic => {
                match crate::linear_model::IsotonicRegression::new().fit_1d(
                    &scores,
                    &yo,
                    &session.child("iso"),
                ) {
                    Ok(q) => isotonic = Some(q.value),
                    Err(e) => ctx.push(e.primary),
                }
            }
        }
        ctx.push(
            Issue::builder(IssueCode::TargetLeakageSuspected)
                .severity(signlred::Severity::Advisory)
                .message("the calibrator is fit on OOF scores; the base is then refit on all rows")
                .build(),
        );
        let base = match RidgeClassifier::new(self.alpha).fit(x, y, &session.child("refit")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedRidgeClassifier::from_penalized(
                    FittedPenalized {
                        coef: Vector::zeros(x.ncols()),
                        intercept: 0.0,
                        alpha: self.alpha,
                        l1_ratio: 0.0,
                    },
                    counts.iter().map(|(c, _)| *c).collect(),
                )
            }
        };
        let classes = base.classes.clone();
        ctx.finish(FittedCalibrated {
            base,
            platt,
            isotonic,
            classes,
        })
    }
}

impl FittedPlatt {
    /// Calibrated \(P(\text{last class} \mid s)\).
    pub fn predict_proba(&self, scores: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        let out = Vector::from_iter(scores.as_slice().iter().map(|&s| {
            let z = self.a * s + self.b;
            if z >= 0.0 {
                let e = (-z).exp();
                1.0 / (1.0 + e)
            } else {
                let e = z.exp();
                e / (1.0 + e)
            }
        }));
        ctx.finish(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    fn sep() -> (Matrix, Vector) {
        let x = Matrix::from_fn(16, 1, |i, _| if i < 8 { -1.5 } else { 1.5 });
        let y = Vector::from_iter((0..16).map(|i| if i < 8 { 0.0 } else { 1.0 }));
        (x, y)
    }

    #[test]
    fn ridge_perceptron_pa_separate() {
        let (x, y) = sep();
        for (name, pred) in [
            (
                "ridge",
                RidgeClassifier::new(0.1)
                    .fit(&x, &y, &Session::new("clf", "ridge"))
                    .unwrap()
                    .value
                    .predict(&x, &Session::new("clf", "p"))
                    .unwrap()
                    .value,
            ),
            (
                "perc",
                Perceptron::new()
                    .fit(&x, &y, &Session::new("clf", "perc"))
                    .unwrap()
                    .value
                    .predict(&x, &Session::new("clf", "p"))
                    .unwrap()
                    .value,
            ),
            (
                "pa",
                PassiveAggressive::new(1.0)
                    .fit(&x, &y, &Session::new("clf", "pa"))
                    .unwrap()
                    .value
                    .predict(&x, &Session::new("clf", "p"))
                    .unwrap()
                    .value,
            ),
        ] {
            let mut ok = 0;
            for i in 0..y.len() {
                if (pred[i] - y[i]).abs() < 0.5 {
                    ok += 1;
                }
            }
            assert!(ok >= 14, "{name} ok={ok}");
        }
    }

    #[test]
    fn dummy_is_majority() {
        let x = Matrix::zeros(5, 1);
        let y = Vector::from_slice(&[0.0, 1.0, 1.0, 1.0, 0.0]);
        let q = DummyClassifier::new()
            .fit(&x, &y, &Session::new("clf", "dummy"))
            .unwrap();
        assert!((q.value.label - 1.0).abs() < 1e-12);
        let p = q
            .value
            .predict(&x, &Session::new("clf", "p"))
            .unwrap()
            .value;
        assert!(p.as_slice().iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn reexports_exist() {
        let _ = LogisticRegression::new();
        let _ = GaussianNB::default();
        let _ = DecisionTreeClassifier::default();
        let _ = LinearSvc::default();
    }

    #[test]
    fn calibrated_cv_platt_on_overlapping() {
        let x = Matrix::from_fn(24, 1, |i, _| {
            if i % 2 == 0 {
                -0.6 + 0.12 * ((i / 2) % 6) as f64
            } else {
                0.6 + 0.12 * ((i / 2) % 6) as f64
            }
        });
        let y = Vector::from_iter((0..24).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }));
        let q = CalibratedClassifierCV::new()
            .fit(&x, &y, &Session::new("cal", "fit"))
            .expect("cal");
        let p = q
            .value
            .predict_proba(&x, &Session::new("cal", "p"))
            .unwrap()
            .value;
        assert!(p[1] > p[0], "p0={} p1={}", p[0], p[1]);
        let pred = q
            .value
            .predict(&x, &Session::new("cal", "lab"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..24 {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 16, "ok={ok}");
    }

    #[test]
    fn platt_maps_separated_scores() {
        let scores = Vector::from_iter((0..20).map(|i| {
            if i < 10 {
                -1.0 + 0.15 * i as f64
            } else {
                0.4 + 0.15 * (i - 10) as f64
            }
        }));
        let y = Vector::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        let q = PlattCalibrator::new()
            .fit(&scores, &y, &Session::new("platt", "fit"))
            .expect("platt");
        let p = q
            .value
            .predict_proba(&scores, &Session::new("platt", "p"))
            .unwrap()
            .value;
        assert!(p[0] < 0.3, "p0={}", p[0]);
        assert!(p[15] > 0.7, "p15={}", p[15]);
    }
}
