//! Additive regression using centered, scaled linear-spline terms.

use crate::annotation::{context, note, stopped};
use crate::regression::{validate_prediction, validate_training};
use crate::{
    AnalysisOptions, Annotation, Family, FittedRegression, GeneralizedLinearModel, Matrix,
    Qualified, Result, Session, Topic, Vector,
};
use signlred::{Issue, IssueCode};

/// One feature's continuous piecewise-linear additive effect.
#[derive(Clone, Debug)]
pub struct SplineTerm {
    /// Feature column in the original matrix.
    pub feature: usize,
    /// Strictly increasing interior knots in original feature units.
    pub knots: Vec<f64>,
}

/// Gaussian LAM or canonical GAM, selected through `family`.
#[derive(Clone, Debug, Default)]
pub struct AdditiveModel {
    /// Gaussian for LAM; Binomial or Poisson for canonical GAM.
    pub family: Family,
    /// Distinct feature terms; each contributes a linear basis and hinge bases.
    pub terms: Vec<SplineTerm>,
    /// L2 penalty on all centered/scaled basis coefficients (not the intercept).
    pub penalty: f64,
    /// Intercept and quality controls.
    pub options: AnalysisOptions,
}

/// Gaussian by default; set terms and penalty before fitting a LAM.
pub type LinearAdditiveModel = AdditiveModel;
/// Select `family` at runtime to fit the corresponding GAM.
pub type GeneralizedAdditiveModel = AdditiveModel;

#[derive(Clone, Debug)]
struct BasisColumn {
    feature: usize,
    knot: Option<f64>,
    center: f64,
    scale: f64,
}

/// Fitted additive effects with the exact training basis retained.
#[derive(Clone, Debug)]
pub struct FittedAdditive {
    /// Fitted basis model; coefficients and inference refer to basis columns.
    pub regression: FittedRegression,
    /// Term specification in original feature units.
    pub terms: Vec<SplineTerm>,
    /// Basis and interpretation notes.
    pub annotations: Vec<Annotation>,
    basis: Vec<BasisColumn>,
    features: usize,
    options: AnalysisOptions,
}

impl AdditiveModel {
    /// Fit a centered linear-spline LAM/GAM with explicit knots and penalty.
    pub fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedAdditive>> {
        let mut ctx = context(session, &self.options);
        if self.terms.is_empty() || !self.options.fit_intercept {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message(
                        "additive fitting requires an intercept and at least one explicit term",
                    )
                    .build(),
            );
        }
        let mut features = std::collections::BTreeSet::new();
        for term in &self.terms {
            if term.feature >= x.ncols()
                || !features.insert(term.feature)
                || term.knots.iter().any(|v| !v.is_finite())
                || term.knots.windows(2).any(|v| v[0] >= v[1])
            {
                ctx.push(Issue::builder(IssueCode::InvalidParameter).message("additive terms require distinct valid features and strictly increasing finite knots").build());
            }
        }
        if stopped(&ctx) {
            return Err(ctx.finish_failure());
        }
        // Only selected terms determine identification. Unused constant or
        // collinear columns must not invalidate an otherwise identified basis.
        let selected = Matrix::from_fn(x.nrows(), self.terms.len(), |i, j| {
            x.get(i, self.terms[j].feature)
        });
        validate_training(&mut ctx, &selected, y, self.family, true);
        if !tsutsumi::linalg::matrix_is_finite(x) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteInput)
                    .message("additive input features must be finite, including unused columns")
                    .build(),
            );
        }
        if stopped(&ctx) {
            return Err(ctx.finish_failure());
        }
        let bounds = crate::regression::feature_bounds(x);
        let mut basis = Vec::new();
        for term in &self.terms {
            for knot in std::iter::once(None).chain(term.knots.iter().copied().map(Some)) {
                if knot.is_some_and(|k| k <= bounds[term.feature].0 || k >= bounds[term.feature].1)
                {
                    ctx.push(
                        Issue::builder(IssueCode::InvalidParameter)
                            .message(
                                "spline knots must be strictly inside the training feature range",
                            )
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                }
                let raw: Vec<f64> = (0..x.nrows())
                    .map(|i| basis_value(x.get(i, term.feature), knot))
                    .collect();
                let center = raw.iter().sum::<f64>() / x.nrows() as f64;
                let scale = raw.iter().map(|v| (v - center).abs()).fold(0.0, f64::max);
                if !center.is_finite()
                    || !scale.is_finite()
                    || scale <= ctx.policy.near_zero_variance
                {
                    ctx.push(
                        Issue::builder(IssueCode::UnidentifiedModel)
                            .message(
                                "an additive basis column has zero or unrepresentable variation",
                            )
                            .build(),
                    );
                    return Err(ctx.finish_failure());
                }
                basis.push(BasisColumn {
                    feature: term.feature,
                    knot,
                    center,
                    scale,
                });
            }
        }
        let expanded = transform(x, &basis);
        let estimator = GeneralizedLinearModel {
            family: self.family,
            options: self.options.clone(),
        };
        let fitted = if self.penalty == 0.0 {
            estimator.fit(&expanded, y, &session.child("basis_fit"))
        } else {
            estimator.fit_penalized(&expanded, y, self.penalty, &session.child("basis_fit"))
        }?;
        ctx.report.merge(fitted.report);
        let mut annotations = fitted.value.annotations.clone();
        annotations.push(Annotation::new(Topic::Assumptions,
            format!("Continuous piecewise-linear splines, {} explicit basis columns, no interactions; basis columns are centered and scaled using training data only. Penalty={}.",basis.len(),self.penalty),
            "Choose knots and smoothing strength using a separate validation procedure; inspect residual structure and sensitivity to knot placement."));
        annotations.push(Annotation::new(Topic::Interpretation,
            "Term effects add on the linear-predictor scale. Individual spline coefficients are not feature-level marginal effects; extrapolation continues the boundary slope.",
            "Use term_effects to inspect whole feature curves, and report the response link and observed range."));
        note(&mut ctx,IssueCode::PValueUnreliable,"Additive-model inference is conditional on the supplied knots and penalty; smoothing selection uncertainty is not included");
        ctx.finish(FittedAdditive {
            regression: fitted.value,
            terms: self.terms.clone(),
            annotations,
            basis,
            features: x.ncols(),
            options: self.options.clone(),
        })
    }
}

impl FittedAdditive {
    /// Predict on the response scale using the stored training transform.
    pub fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = context(session, &self.options);
        validate_prediction(&mut ctx, x, self.features);
        if stopped(&ctx) {
            return Err(ctx.finish_failure());
        }
        let result = self
            .regression
            .predict(&transform(x, &self.basis), &session.child("basis_predict"))?;
        ctx.report.merge(result.report);
        ctx.finish(result.value)
    }

    /// One column per additive term; row sums plus the intercept equal eta.
    pub fn term_effects(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = context(session, &self.options);
        validate_prediction(&mut ctx, x, self.features);
        if stopped(&ctx) {
            return Err(ctx.finish_failure());
        }
        let basis = transform(x, &self.basis);
        let effects = Matrix::from_fn(x.nrows(), self.terms.len(), |i, t| {
            self.basis
                .iter()
                .enumerate()
                .filter(|(_, b)| b.feature == self.terms[t].feature)
                .map(|(j, _)| basis.get(i, j) * self.regression.beta[j + 1])
                .sum()
        });
        if !tsutsumi::linalg::matrix_is_finite(&effects) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("additive effect overflow")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        ctx.finish(effects)
    }
}

fn basis_value(x: f64, knot: Option<f64>) -> f64 {
    knot.map_or(x, |k| (x - k).max(0.0))
}
fn transform(x: &Matrix, basis: &[BasisColumn]) -> Matrix {
    Matrix::from_fn(x.nrows(), basis.len(), |i, j| {
        (basis_value(x.get(i, basis[j].feature), basis[j].knot) - basis[j].center) / basis[j].scale
    })
}
