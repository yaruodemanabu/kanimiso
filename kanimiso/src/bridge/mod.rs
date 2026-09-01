//! Zero-copy boundary with the wormhole family (AGENTS.md §3.3).
//!
//! Other kanimiso modules must see [`signlred`] types only. Wormhole `Error`
//! values are mapped here and never re-exported.

use signlred::{Failure, Issue, IssueCode, NumericalCompromise};

/// Map a wormhole error into a kanimiso [`Failure`].
///
/// [`wormhole::Error::InvalidParameter`] is treated as a caller bug (not a
/// [`signlred::Qualified`] warning). Iterative residuals become
/// [`IssueCode::ResidualTooLarge`] / [`IssueCode::MaxIterReached`] with a
/// [`NumericalCompromise`].
pub(crate) fn failure_from_wormhole(
    algorithm: impl Into<String>,
    operation: impl Into<String>,
    err: wormhole::Error,
) -> Failure {
    let algorithm = algorithm.into();
    let operation = operation.into();
    let issue = issue_from_wormhole(&err);
    Failure::from_issue(algorithm, operation, issue)
}

fn issue_from_wormhole(err: &wormhole::Error) -> Issue {
    match err {
        wormhole::Error::EmptyInput { name } => Issue::builder(IssueCode::EmptyMatrix)
            .message(format!("{name} must be non-empty"))
            .build(),
        wormhole::Error::ShapeMismatch {
            context,
            left,
            right,
        } => Issue::builder(IssueCode::DimensionMismatch)
            .message(format!(
                "shape mismatch for {context}: left={left:?}, right={right:?}"
            ))
            .build(),
        wormhole::Error::InvalidWeight { index, value } => {
            let code = if value.is_finite() {
                IssueCode::InvalidWeight
            } else {
                IssueCode::NonFiniteInput
            };
            Issue::builder(code)
                .message(format!("weight {index} is invalid: {value}"))
                .metric("index", *index as f64)
                .metric("value", *value)
                .build()
        }
        wormhole::Error::InvalidCost { row, column, value } => {
            Issue::builder(IssueCode::NonFiniteInput)
                .message(format!("cost at ({row}, {column}) is invalid: {value}"))
                .metric("row", *row as f64)
                .metric("column", *column as f64)
                .metric("value", *value)
                .build()
        }
        wormhole::Error::InvalidParameter { name, requirement } => {
            Issue::builder(IssueCode::MeaninglessFit)
                .message(format!(
            "invalid parameter {name}: must be {requirement} (caller bug; not a Qualified warning)"
        ))
                .build()
        }
        wormhole::Error::DidNotConverge {
            algorithm,
            iterations,
            residual,
        } => {
            let code = if *residual > 0.0 {
                IssueCode::ResidualTooLarge
            } else {
                IssueCode::MaxIterReached
            };
            Issue::builder(code)
                .message(format!(
                    "{algorithm} did not converge after {iterations} iterations (residual={residual})"
                ))
                .compromise(NumericalCompromise::new(
                    format!("{algorithm} to residual tolerance"),
                    format!("stopped after {iterations} iterations"),
                    format!("residual {residual} exceeded the solver tolerance"),
                    "the plan is the last iterate, not a certified optimum",
                ))
                .metric("iterations", *iterations as f64)
                .metric("residual", *residual)
                .build()
        }
        wormhole::Error::MassMismatch { source, target } => {
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "balanced masses differ: source={source}, target={target}"
                ))
                .metric("source_mass", *source)
                .metric("target_mass", *target)
                .build()
        }
        wormhole::Error::Infeasible { context } => Issue::builder(IssueCode::UnidentifiedModel)
            .message(format!("infeasible transport: {context}"))
            .build(),
        wormhole::Error::LinearAlgebra { operation } => Issue::builder(IssueCode::SingularMatrix)
            .message(format!("faer linear algebra failed during {operation}"))
            .build(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_maps_to_empty_matrix() {
        let err = failure_from_wormhole(
            "bridge",
            "map",
            wormhole::Error::EmptyInput { name: "source" },
        );
        assert_eq!(err.primary().code, IssueCode::EmptyMatrix);
    }

    #[test]
    fn nonfinite_weight_maps_to_non_finite_input() {
        let err = failure_from_wormhole(
            "bridge",
            "map",
            wormhole::Error::InvalidWeight {
                index: 0,
                value: f64::NAN,
            },
        );
        assert_eq!(err.primary().code, IssueCode::NonFiniteInput);
    }

    #[test]
    fn did_not_converge_records_compromise() {
        let err = failure_from_wormhole(
            "bridge",
            "map",
            wormhole::Error::DidNotConverge {
                algorithm: "sinkhorn",
                iterations: 10,
                residual: 1e-3,
            },
        );
        assert_eq!(err.primary().code, IssueCode::ResidualTooLarge);
        assert!(err.primary().numerical_compromise.is_some());
    }
}
