//! Interpretive notes that travel with every fitted analysis.

use crate::{Policy, Session};
use signlred::{Issue, IssueCode, Severity};
use tsutsumi::FitCtx;

/// Why a note matters to the analyst.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topic {
    /// Sampling, distribution, link, and design assumptions.
    Assumptions,
    /// Evidence and regularity required by uncertainty estimates.
    Inference,
    /// Sensitivity to observations, collinearity, or extrapolation.
    Sensitivity,
    /// Algorithmic approximation or convergence evidence.
    Computation,
    /// Meaning of a prediction or feature attribution.
    Interpretation,
}

/// A result-specific statement and an action the analyst can take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Annotation {
    /// Subject of this note.
    pub topic: Topic,
    /// Human-readable condition, assumption, or limitation.
    pub statement: String,
    /// Suggested inspection or next analysis.
    pub action: String,
}

impl Annotation {
    pub(crate) fn new(
        topic: Topic,
        statement: impl Into<String>,
        action: impl Into<String>,
    ) -> Self {
        Self {
            topic,
            statement: statement.into(),
            action: action.into(),
        }
    }
}

/// Controls convergence and the numerical quality contract for regression.
#[derive(Clone, Debug)]
pub struct AnalysisOptions {
    /// Include an intercept separately from the supplied feature columns.
    pub fit_intercept: bool,
    /// Maximum IRLS iterations; mixed models use this times their dimension.
    pub max_iterations: usize,
    /// Shared numerical thresholds and abort policy.
    pub policy: Policy,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            max_iterations: 200,
            policy: Policy::default(),
        }
    }
}

pub(crate) fn context(session: &Session, options: &AnalysisOptions) -> FitCtx {
    let mut ctx = FitCtx::with_session(session.clone());
    ctx.policy = options.policy.clone();
    if options.max_iterations == 0
        || options.policy.cf_max_iter == 0
        || [
            options.policy.optimizer_parameter_tol,
            options.policy.optimizer_objective_tol,
            options.policy.rank_tol_relative,
            options.policy.residual_tol,
            options.policy.near_zero_variance,
            options.policy.probability_sum_tol,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
    {
        ctx.push(
            Issue::builder(IssueCode::InvalidParameter)
                .message("iteration limits and numerical tolerances must be positive and finite")
                .build(),
        );
    }
    ctx
}

pub(crate) fn stopped(ctx: &FitCtx) -> bool {
    ctx.report.issues().iter().any(|issue| {
        ctx.policy.must_abort(issue)
            || matches!(
                issue.code,
                IssueCode::DimensionMismatch
                    | IssueCode::EmptyMatrix
                    | IssueCode::NonFiniteInput
                    | IssueCode::InvalidParameter
                    | IssueCode::ConstantTarget
                    | IssueCode::AllMissing
            )
    })
}

pub(crate) fn note(ctx: &mut FitCtx, code: IssueCode, message: impl Into<String>) {
    ctx.push(
        Issue::builder(code)
            .severity(Severity::Advisory)
            .message(message)
            .build(),
    );
}

pub(crate) fn basic_annotations() -> Vec<Annotation> {
    vec![
        Annotation::new(Topic::Interpretation,
            "Coefficients describe conditional association on the model's linear-predictor scale; they do not identify a causal effect.",
            "Check confounding, sampling, treatment assignment, and feature construction before making an effect claim."),
        Annotation::new(Topic::Inference,
            "In-sample fit and nominal coefficient tests do not account for variable selection, repeated model searches, or dataset reuse.",
            "Use a held-out validation design and pre-specify or adjust the family of tests."),
    ]
}
