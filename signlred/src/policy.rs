//! Thresholds that decide abort vs. warn for numerical and statistical quality.

use crate::codes::IssueCode;
use crate::issue::Issue;
use crate::meaningless::InterpretiveValue;
use crate::severity::Severity;

/// Tunable quality contract used by [`crate::Report::finish_with_policy`].
///
/// Defaults are deliberately strict: unidentified models, constant targets,
/// and singular solves abort. Ill-conditioning and numerical compromises warn
/// but stay attached to the value.
#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    /// Issues at this severity or worse (`Fatal` < `Error` < …) abort.
    pub abort_at: Severity,
    /// If true, Vacuous/False meaninglessness aborts even when the code's
    /// default severity is only a warning. Misleading diagnoses stay warnings:
    /// the number exists, but it answers the wrong question.
    pub abort_on_meaningless: bool,
    /// Condition number at or above this value is [`IssueCode::IllConditioned`].
    pub condition_number_warn: f64,
    /// Condition number at or above this value is [`IssueCode::NearSingular`].
    pub condition_number_error: f64,
    /// Relative singular-value cutoff for numerical rank (`σ_i > cutoff · σ_max`).
    pub rank_tol_relative: f64,
    /// Residual ‖Ax−b‖ / (‖A‖‖x‖+‖b‖) above this is [`IssueCode::ResidualTooLarge`].
    pub residual_tol: f64,
    /// Minimum observations per estimated parameter (unregularized models).
    pub min_samples_per_parameter: f64,
    /// Feature standard deviation below this is near-zero variance.
    pub near_zero_variance: f64,
    /// VIF at or above this is [`IssueCode::HighMulticollinearity`].
    pub vif_warn: f64,
    /// |corr| at or above this between distinct columns is perfect collinearity.
    pub collinearity_corr: f64,
    /// Effective sample size below this after forgetting is insufficient.
    pub min_effective_sample: f64,
    /// |information_gain| below this is an uninformative online update.
    pub uninformative_info_eps: f64,
    /// Parameter-delta L2 relative to ‖θ‖ above this is an anomalous jump.
    pub anomalous_jump_rel: f64,
    /// Minority-class fraction below this is severe imbalance.
    pub imbalance_warn: f64,
    /// |R² − 1| below this is treated as R² = 1.
    pub r2_one_tol: f64,
    /// |R²| below this is treated as R² = 0.
    pub r2_zero_tol: f64,
    /// Optional floor on a log-probability. `None` means do not clamp.
    /// Applying a floor must record [`crate::CompromiseKind::ProbabilityClamped`].
    pub log_prob_floor: Option<f64>,
    /// Scale factor below this emits [`crate::IssueCode::ForwardUnderflow`].
    pub underflow_guard: f64,
    /// Maximum backward-difference order (`WindowLag`). Above this is a `Failure`.
    pub max_difference_order: usize,
    /// Continued-fraction relative stop for `betainc_reg` / `gamma_p`.
    pub cf_tol: f64,
    /// Continued-fraction iteration cap for `betainc_reg` / `gamma_p`.
    pub cf_max_iter: usize,
    /// Floor applied to HMM emission variances in the M-step (not in `log_prob`).
    pub emission_var_floor: f64,
    /// Additive floor when renormalising HMM start / transition rows.
    pub transition_floor: f64,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            abort_at: Severity::Error,
            abort_on_meaningless: true,
            condition_number_warn: 1e10,
            condition_number_error: 1e16,
            rank_tol_relative: 1e-12,
            residual_tol: 1e-6,
            min_samples_per_parameter: 5.0,
            near_zero_variance: 1e-15,
            vif_warn: 10.0,
            collinearity_corr: 1.0 - 1e-12,
            min_effective_sample: 5.0,
            uninformative_info_eps: 1e-15,
            anomalous_jump_rel: 10.0,
            imbalance_warn: 0.05,
            r2_one_tol: 1e-12,
            r2_zero_tol: 1e-12,
            log_prob_floor: None,
            underflow_guard: 1e-300,
            max_difference_order: 8,
            cf_tol: 1e-15,
            cf_max_iter: 300,
            emission_var_floor: 1e-6,
            transition_floor: 1e-8,
        }
    }
}

impl Policy {
    /// A still-strict policy that allows meaningless-but-flagged values through
    /// as `Qualified` (used only in tests that need the number).
    pub fn warn_meaningless() -> Self {
        Self {
            abort_on_meaningless: false,
            ..Self::default()
        }
    }

    fn is_vacuous_or_false(issue: &Issue) -> bool {
        match issue.meaninglessness.as_ref().map(|m| m.interpretive_value) {
            Some(InterpretiveValue::Vacuous | InterpretiveValue::False) => true,
            _ => matches!(
                issue.code,
                IssueCode::MeaninglessFit
                    | IssueCode::ConstantTarget
                    | IssueCode::PredictionsAreConstant
                    | IssueCode::RankZero
            ),
        }
    }

    /// Rewrite an issue's severity according to this policy.
    pub fn apply(&self, mut issue: Issue) -> Issue {
        if self.abort_on_meaningless
            && Self::is_vacuous_or_false(&issue)
            && issue.severity > Severity::Error
        {
            issue.severity = Severity::Error;
        }
        if issue.code == IssueCode::IllConditioned {
            if let Some((_, kappa)) = issue.metrics.iter().find(|(k, _)| k == "condition_number") {
                if *kappa >= self.condition_number_error {
                    issue.code = IssueCode::NearSingular;
                    issue.severity = IssueCode::NearSingular.default_severity();
                    issue.title = issue.code.as_str().to_string();
                }
            }
        }
        issue
    }

    /// Whether an issue, after [`Self::apply`], must abort.
    pub fn must_abort(&self, issue: &Issue) -> bool {
        issue.severity.is_at_least(self.abort_at)
            || (self.abort_on_meaningless && Self::is_vacuous_or_false(issue))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_fields_match_agents_d8() {
        let p = Policy::default();
        assert!(p.log_prob_floor.is_none());
        assert_eq!(p.underflow_guard, 1e-300);
        assert_eq!(p.max_difference_order, 8);
        assert_eq!(p.cf_tol, 1e-15);
        assert_eq!(p.cf_max_iter, 300);
        assert_eq!(p.emission_var_floor, 1e-6);
        assert_eq!(p.transition_floor, 1e-8);
    }
}
