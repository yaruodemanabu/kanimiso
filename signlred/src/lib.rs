//! Quality-aware error handling for machine learning and linear algebra.
//!
//! `signlred` owns the question *“can this numerical or statistical result be
//! trusted, and if not, why?”*. Ordinary I/O or programmer errors are in scope
//! only insofar as they poison a computation. The crate is deliberately stricter
//! than typical application error types: a successful factorization that is
//! rank-deficient, a regression with a constant target, or an online update that
//! added no information are first-class failures or warnings.
//!
//! # Contract
//!
//! 1. Fatal / error-severity issues abort via [`Failure`]. The value is not
//!    returned.
//! 2. Warnings, advisories, and info stay attached to [`Qualified<T>`].
//! 3. Every numerical *compromise* (a different computation than the caller
//!    asked for) must be recorded as [`NumericalCompromise`].
//! 4. Every statistically vacuous result must carry [`Meaninglessness`].
//! 5. Incremental / online updates must carry [`IncrementalQuality`].

#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Failure embeds the full Report so callers can see every preceding warning.
// Boxing it would shrink the Err variant but hide that the ledger is the value.
#![allow(clippy::result_large_err)]

mod codes;
mod compromise;
mod domain;
mod failure;
mod guards;
mod incremental;
mod issue;
mod location;
mod meaningless;
mod policy;
mod qualified;
mod report;
mod severity;

pub use codes::IssueCode;
pub use compromise::{CompromiseKind, NumericalCompromise};
pub use domain::Domain;
pub use failure::Failure;
pub use guards::{
    classify_condition_number, condition_issue, constant_feature_issue, constant_target_issue,
    insufficient_sample, scan_finite, slice_stats, FiniteScan, RankHint, SliceStats,
};
pub use incremental::IncrementalQuality;
pub use issue::Issue;
pub use location::Location;
pub use meaningless::{InterpretiveValue, Meaninglessness};
pub use policy::Policy;
pub use qualified::Qualified;
pub use report::Report;
pub use severity::Severity;

/// Result alias used by every quality-aware computation.
pub type Result<T> = core::result::Result<T, Failure>;

/// Convenience constructor for a new report bound to an algorithm and operation.
pub fn report(algorithm: impl Into<String>, operation: impl Into<String>) -> Report {
    Report::new(algorithm, operation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_target_is_meaningless_and_aborts_under_default_policy() {
        let mut report = Report::new("ols", "fit");
        report.set_sample_shape(10, 3);
        report.push(
            Issue::builder(IssueCode::ConstantTarget)
                .message("every y value equals 1.0")
                .meaninglessness(Meaninglessness {
                    what_was_computed: "ordinary least squares coefficients".into(),
                    why_meaningless: "a constant target has no residual variance; R² is undefined or vacuously 1, and slope identification is empty".into(),
                    interpretive_value: InterpretiveValue::Vacuous,
                    suggested_action: "inspect the target construction; do not interpret coefficients as effects".into(),
                })
                .metric("target_std", 0.0)
                .build(),
        );
        let err = report
            .finish_with_policy(Policy::default(), ())
            .unwrap_err();
        assert_eq!(err.primary().code, IssueCode::ConstantTarget);
        assert!(err.primary().meaninglessness.is_some());
    }

    #[test]
    fn ill_conditioned_warning_survives_as_qualified() {
        let mut report = Report::new("ridge", "fit");
        report.push(
            Issue::builder(IssueCode::IllConditioned)
                .message("κ ≈ 1e12")
                .compromise(NumericalCompromise {
                    original_intent: "solve (XᵀX)β = Xᵀy exactly".into(),
                    actual_computation: "QR with residual and condition warning".into(),
                    why_necessary: "XᵀX is not safely invertible in f64".into(),
                    expected_error_bound: Some(1e-4),
                    assumptions_violated: vec!["well-conditioned Gram matrix".into()],
                    interpretability_impact:
                        "coefficient signs and magnitudes can flip under tiny perturbations".into(),
                    kind: CompromiseKind::Unspecified,
                })
                .metric("condition_number", 1e12)
                .build(),
        );
        let q = report.finish(()).unwrap();
        assert!(q.report.has_warning());
        assert_eq!(q.report.compromises().count(), 1);
    }
}
