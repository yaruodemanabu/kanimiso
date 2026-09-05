//! Exact interventional SHAP for a fitted linear predictor.

use crate::annotation::{context, note, stopped};
use crate::regression::validate_prediction;
use crate::{Annotation, FittedRegression, Matrix, Qualified, Result, Session, Topic};
use signlred::{Issue, IssueCode};

/// Additive feature attributions on the model's linear-predictor scale.
#[derive(Clone, Debug)]
pub struct LinearExplanation {
    /// Expected linear predictor under the supplied background.
    pub base_value: f64,
    /// Rows are observations and columns follow the original feature order.
    pub contributions: Matrix,
    /// Empirical background feature means used by this explanation.
    pub background_mean: Vec<f64>,
    /// Assumptions and interpretation of the selected SHAP definition.
    pub annotations: Vec<Annotation>,
}

/// Compute `beta_j * (x_j - E_background[X_j])` for the linear model.
///
/// This is interventional SHAP for the linear predictor, including log-odds
/// or log-mean for nonlinear response links. It is not conditional SHAP.
pub fn linear_shap(
    model: &FittedRegression,
    background: &Matrix,
    x: &Matrix,
    session: &Session,
) -> Result<Qualified<LinearExplanation>> {
    let mut ctx = context(session, &model.options);
    validate_prediction(&mut ctx, x, model.bounds.len());
    validate_prediction(&mut ctx, background, model.bounds.len());
    if background.nrows() == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("SHAP background requires at least one row")
                .build(),
        );
    }
    if stopped(&ctx) {
        return Err(ctx.finish_failure());
    }
    let means: Vec<f64> = (0..background.ncols())
        .map(|j| {
            (0..background.nrows())
                .map(|i| background.get(i, j) / background.nrows() as f64)
                .sum()
        })
        .collect();
    let offset = usize::from(model.fit_intercept);
    let intercept = if model.fit_intercept {
        model.beta[0]
    } else {
        0.0
    };
    let base = intercept
        + means
            .iter()
            .enumerate()
            .map(|(j, &mean)| model.beta[j + offset] * mean)
            .sum::<f64>();
    let contributions = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
        model.beta[j + offset] * (x.get(i, j) - means[j])
    });
    if !base.is_finite() || !tsutsumi::linalg::matrix_is_finite(&contributions) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteOutput)
                .message("linear SHAP arithmetic is unrepresentable")
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    let annotations=vec![
        Annotation::new(Topic::Interpretation,"Interventional SHAP attributes the fitted linear predictor, not causality. Correlated features are not conditioned upon, so conditional/correlation-dependent SHAP can allocate credit differently.","State the background population and SHAP definition; do not interpret attribution as a causal effect."),
        Annotation::new(Topic::Assumptions,format!("Background has {} rows. The baseline and contributions are on the {:?} linear-predictor scale; nonlinear inverse links do not preserve additive attribution.",background.nrows(),model.family),"For logit/count models report contributions as log-odds/log-mean; inspect sensitivity to the background dataset."),
    ];
    note(&mut ctx,IssueCode::CausalClaimUnidentified,"SHAP is a decomposition of a fitted predictor under a chosen background, not an identified causal effect");
    ctx.finish(LinearExplanation {
        base_value: base,
        contributions,
        background_mean: means,
        annotations,
    })
}
