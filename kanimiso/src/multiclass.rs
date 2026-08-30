//! One-vs-rest and one-vs-one reductions (sklearn `multiclass`).
//!
//! Each binary subproblem is a [`crate::classification::RidgeClassifier`]. A
//! single class on the original `y` is vacuous ([`IssueCode::SingleClass`]).
//! Do not treat `n_classes` as a linear-model parameter count for
//! identification: a 3-class problem on 24 rows is identified.

use crate::classification::{FittedRidgeClassifier, RidgeClassifier};
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::{Fit, Predict};
use crate::validate::{inspect_classes, inspect_xy};
use ojizou_san::Session;
use signlred::{
    Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result, Severity,
};

/// One-vs-rest ridge reduction.
#[derive(Clone, Debug)]
pub struct OneVsRestClassifier {
    /// Shared ridge `α`.
    pub alpha: f64,
}

impl Default for OneVsRestClassifier {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl OneVsRestClassifier {
    /// OvR with ridge `α`.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

/// Fitted OvR: one binary ridge per class.
#[derive(Clone, Debug)]
pub struct FittedOvr {
    /// Class ids in the same order as [`Self::estimators`].
    pub classes: Vec<i64>,
    /// One vs-rest ridge per class.
    pub estimators: Vec<FittedRidgeClassifier>,
}

impl Fit for OneVsRestClassifier {
    type Fitted = FittedOvr;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedOvr>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        if classes.len() < 2 {
            return ctx.finish(FittedOvr {
                classes,
                estimators: Vec::new(),
            });
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("OvR fits independent binary ridges; scores are not a joint softmax")
                .compromise(NumericalCompromise::new(
                    "multinomial logistic / joint posterior",
                    "independent ridge-vs-rest scores",
                    "the binary problems do not share a likelihood",
                    "do not read argmax scores as class probabilities",
                ))
                .build(),
        );
        let mut estimators = Vec::with_capacity(classes.len());
        for (k, &c) in classes.iter().enumerate() {
            let yb = Vector::from_iter(y.as_slice().iter().map(|&v| {
                if v.is_finite() && v.round() as i64 == c {
                    1.0
                } else {
                    0.0
                }
            }));
            match RidgeClassifier::new(self.alpha).fit(x, &yb, &session.child(format!("ovr_{k}"))) {
                Ok(q) => estimators.push(q.value),
                Err(e) => {
                    ctx.push(
                        Issue::builder(IssueCode::UnidentifiedModel)
                            .severity(Severity::Warning)
                            .message(format!(
                                "OvR class {c} ridge aborted; using a constant rule"
                            ))
                            .build(),
                    );
                    let _ = e;
                    estimators.push(FittedRidgeClassifier::from_penalized(
                        crate::linear_model::FittedPenalized {
                            coef: Vector::zeros(x.ncols()),
                            intercept: 0.0,
                            alpha: self.alpha,
                            l1_ratio: 0.0,
                        },
                        vec![0, 1],
                    ));
                }
            }
        }
        ctx.finish(FittedOvr {
            classes,
            estimators,
        })
    }
}

impl Predict for FittedOvr {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.estimators.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::StaleState)
                    .message("OvR has no binary estimators")
                    .meaninglessness(Meaninglessness::vacuous(
                        "OvR labels",
                        "no binary problem was identified",
                        "fit on a y with at least two classes",
                    ))
                    .build(),
            );
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut best = vec![(f64::NEG_INFINITY, self.classes[0]); x.nrows()];
        for (est, &c) in self.estimators.iter().zip(&self.classes) {
            let scores = match est.decision_function(x, &session.child("ovr_score")) {
                Ok(q) => q.value,
                Err(e) => {
                    ctx.push(e.primary);
                    Vector::zeros(x.nrows())
                }
            };
            for i in 0..x.nrows().min(scores.len()) {
                if scores[i] > best[i].0 {
                    best[i] = (scores[i], c);
                }
            }
        }
        ctx.finish(Vector::from_iter(best.into_iter().map(|(_, c)| c as f64)))
    }
}

/// One-vs-one ridge reduction.
#[derive(Clone, Debug)]
pub struct OneVsOneClassifier {
    /// Shared ridge `α`.
    pub alpha: f64,
}

impl Default for OneVsOneClassifier {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl OneVsOneClassifier {
    /// OvO with ridge `α`.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

/// One pairwise estimator.
#[derive(Clone, Debug)]
pub struct OvoPair {
    /// Left class.
    pub a: i64,
    /// Right class.
    pub b: i64,
    /// Ridge on the pair (scores > 0 → `b`).
    pub model: FittedRidgeClassifier,
}

/// Fitted OvO vote.
#[derive(Clone, Debug)]
pub struct FittedOvo {
    /// Training classes.
    pub classes: Vec<i64>,
    /// Pairwise ridges.
    pub pairs: Vec<OvoPair>,
}

impl Fit for OneVsOneClassifier {
    type Fitted = FittedOvo;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedOvo>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        if classes.len() < 2 {
            return ctx.finish(FittedOvo {
                classes,
                pairs: Vec::new(),
            });
        }
        let mut pairs = Vec::new();
        for i in 0..classes.len() {
            for j in (i + 1)..classes.len() {
                let a = classes[i];
                let b = classes[j];
                let mut rows = Vec::new();
                for r in 0..y.len().min(x.nrows()) {
                    if !y[r].is_finite() {
                        continue;
                    }
                    let lab = y[r].round() as i64;
                    if lab == a || lab == b {
                        rows.push(r);
                    }
                }
                if rows.len() < 4 {
                    ctx.push(
                        Issue::builder(IssueCode::InsufficientSample)
                            .severity(Severity::Warning)
                            .message(format!("OvO pair ({a},{b}) has only {} rows", rows.len()))
                            .build(),
                    );
                    continue;
                }
                let xs = Matrix::from_fn(rows.len(), x.ncols(), |t, c| x.get(rows[t], c));
                let ys = Vector::from_iter(rows.iter().map(|&r| {
                    if y[r].round() as i64 == b {
                        1.0
                    } else {
                        0.0
                    }
                }));
                match RidgeClassifier::new(self.alpha).fit(
                    &xs,
                    &ys,
                    &session.child(format!("ovo_{a}_{b}")),
                ) {
                    Ok(q) => pairs.push(OvoPair {
                        a,
                        b,
                        model: q.value,
                    }),
                    Err(_) => {
                        ctx.push(
                            Issue::builder(IssueCode::UnidentifiedModel)
                                .severity(Severity::Warning)
                                .message(format!("OvO pair ({a},{b}) ridge aborted"))
                                .build(),
                        );
                    }
                }
            }
        }
        if pairs.is_empty() && classes.len() >= 2 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("OvO produced no pairwise estimators")
                    .build(),
            );
        }
        ctx.finish(FittedOvo { classes, pairs })
    }
}

impl Predict for FittedOvo {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.pairs.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::StaleState)
                    .message("OvO has no pairwise estimators")
                    .build(),
            );
            let fill = self.classes.first().copied().unwrap_or(0) as f64;
            return ctx.finish(Vector::filled(x.nrows(), fill));
        }
        let mut votes = vec![std::collections::BTreeMap::<i64, usize>::new(); x.nrows()];
        for (pidx, pair) in self.pairs.iter().enumerate() {
            match pair
                .model
                .predict(x, &session.child(format!("ovo_p{pidx}")))
            {
                Ok(q) => {
                    for i in 0..x.nrows().min(q.value.len()) {
                        let lab = if q.value[i] >= 0.5 { pair.b } else { pair.a };
                        *votes[i].entry(lab).or_insert(0) += 1;
                    }
                }
                Err(e) => ctx.push(e.primary),
            }
        }
        let y = Vector::from_iter(votes.iter().map(|m| {
            m.iter()
                .max_by_key(|(_, c)| *c)
                .map(|(k, _)| *k as f64)
                .unwrap_or(0.0)
        }));
        ctx.finish(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Fit, Predict};

    #[test]
    fn ovr_and_ovo_three_blocks() {
        let x = Matrix::from_fn(30, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..30).map(|i| (i / 10) as f64));
        let ovr = OneVsRestClassifier::new(0.1)
            .fit(&x, &y, &Session::new("ovr", "fit"))
            .expect("ovr");
        assert_eq!(ovr.value.classes.len(), 3);
        let p = ovr
            .value
            .predict(&x, &Session::new("ovr", "p"))
            .expect("ovrp")
            .value;
        let acc = (0..30).filter(|&i| (p[i] - y[i]).abs() < 0.5).count();
        assert!(acc >= 20, "ovr acc={acc}");
        let ovo = OneVsOneClassifier::new(0.1)
            .fit(&x, &y, &Session::new("ovo", "fit"))
            .expect("ovo");
        assert!(!ovo.value.pairs.is_empty());
        let q = ovo
            .value
            .predict(&x, &Session::new("ovo", "p"))
            .expect("ovop")
            .value;
        let acc2 = (0..30).filter(|&i| (q[i] - y[i]).abs() < 0.5).count();
        assert!(acc2 >= 18, "ovo acc={acc2}");
    }
}
