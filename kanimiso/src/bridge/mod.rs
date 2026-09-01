//! Zero-copy boundary with the wormhole family (AGENTS.md §3.3).
//!
//! Other kanimiso modules must see [`signlred`] types only. Wormhole `Error`
//! values are mapped here and never re-exported.

use crate::context::FitCtx;
use crate::data::Matrix;
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

pub(crate) fn issue_from_wormhole(err: &wormhole::Error) -> Issue {
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

/// Map a coronel error into a kanimiso [`Failure`].
pub(crate) fn failure_from_coronel(
    algorithm: impl Into<String>,
    operation: impl Into<String>,
    err: coronel::Error,
) -> Failure {
    Failure::from_issue(algorithm, operation, issue_from_coronel(&err))
}

pub(crate) fn issue_from_coronel(err: &coronel::Error) -> Issue {
    match err {
        coronel::Error::EmptyInput => Issue::builder(IssueCode::EmptyMatrix)
            .message("kernel input must be non-empty")
            .build(),
        coronel::Error::DimensionMismatch { left, right } => {
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!("feature dimensions differ: {left} != {right}"))
                .build()
        }
        coronel::Error::NonFiniteInput { row, column } => Issue::builder(IssueCode::NonFiniteInput)
            .message(format!("kernel input at ({row}, {column}) is not finite"))
            .metric("row", *row as f64)
            .metric("column", *column as f64)
            .build(),
        coronel::Error::InvalidParameter(name) => Issue::builder(IssueCode::MeaninglessFit)
            .message(format!("invalid kernel parameter: {name}"))
            .build(),
        coronel::Error::TooFewSamples => Issue::builder(IssueCode::EmptyMatrix)
            .message("an unbiased kernel statistic needs at least two samples")
            .build(),
    }
}

/// Map a jelly-wave error into a kanimiso [`Failure`].
pub(crate) fn failure_from_jelly_wave(
    algorithm: impl Into<String>,
    operation: impl Into<String>,
    err: jelly_wave::Error,
) -> Failure {
    Failure::from_issue(algorithm, operation, issue_from_jelly_wave(&err))
}

pub(crate) fn issue_from_jelly_wave(err: &jelly_wave::Error) -> Issue {
    match err {
        jelly_wave::Error::EmptyInput => Issue::builder(IssueCode::EmptyMatrix)
            .message("waveforms must be non-empty")
            .build(),
        jelly_wave::Error::NonFiniteInput { index } => Issue::builder(IssueCode::NonFiniteInput)
            .message(format!("waveform sample {index} is not finite"))
            .metric("index", *index as f64)
            .build(),
        jelly_wave::Error::InvalidParameter(parameter) => Issue::builder(IssueCode::MeaninglessFit)
            .message(format!("invalid waveform parameter: {parameter}"))
            .build(),
        jelly_wave::Error::ShapeMismatch { left, right } => {
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "waveform matrix shapes differ: {left:?} != {right:?}"
                ))
                .build()
        }
    }
}

/// Pairwise row distances via wormhole. Shape / domain errors become issues
/// and a zero matrix so existing score APIs stay `Qualified`.
pub(crate) fn pairwise_metric(
    ctx: &mut FitCtx,
    left: &Matrix,
    right: &Matrix,
    metric: wormhole::metrics::Metric,
) -> Matrix {
    match wormhole::metrics::pairwise(left.inner(), right.inner(), metric) {
        Ok(mat) => Matrix::from_faer(mat),
        Err(err) => {
            ctx.push(issue_from_wormhole(&err));
            Matrix::zeros(left.nrows(), right.nrows())
        }
    }
}

/// Pairwise kernel matrix via coronel.
pub(crate) fn pairwise_kernel(
    ctx: &mut FitCtx,
    left: &Matrix,
    right: &Matrix,
    kernel: coronel::Kernel,
) -> Matrix {
    match coronel::pairwise(kernel, left.inner(), right.inner()) {
        Ok(mat) => Matrix::from_faer(mat),
        Err(err) => {
            ctx.push(issue_from_coronel(&err));
            Matrix::zeros(left.nrows(), right.nrows())
        }
    }
}

/// One paired (row-aligned) distance via wormhole.
pub(crate) fn paired_metric(
    ctx: &mut FitCtx,
    left: &Matrix,
    right: &Matrix,
    metric: wormhole::metrics::Metric,
) -> crate::data::Vector {
    use crate::data::Vector;
    if left.nrows() != right.nrows() || left.ncols() != right.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "paired distances shapes {:?} vs {:?}",
                    left.shape(),
                    right.shape()
                ))
                .build(),
        );
        return Vector::zeros(left.nrows().min(right.nrows()));
    }
    let mut out = Vector::zeros(left.nrows());
    for i in 0..left.nrows() {
        let a: Vec<f64> = (0..left.ncols()).map(|j| left.get(i, j)).collect();
        let b: Vec<f64> = (0..right.ncols()).map(|j| right.get(i, j)).collect();
        match wormhole::metrics::distance(&a, &b, metric) {
            Ok(v) => out[i] = v,
            Err(err) => {
                ctx.push(issue_from_wormhole(&err));
                out[i] = f64::NAN;
            }
        }
    }
    out
}

/// DTW / ERP / Fréchet / Soft-DTW via jelly-wave. Empty or invalid input is
/// `NaN` plus an issue (the public tslearn wrappers stay `Qualified`).
pub(crate) fn wave_dtw(left: &[f64], right: &[f64]) -> std::result::Result<f64, jelly_wave::Error> {
    jelly_wave::dtw(left, right)
}

pub(crate) fn wave_dtw_path(
    left: &[f64],
    right: &[f64],
) -> std::result::Result<jelly_wave::DtwAlignment, jelly_wave::Error> {
    jelly_wave::dtw_with_options(left, right, jelly_wave::DtwOptions::default())
}

pub(crate) fn wave_erp(
    left: &[f64],
    right: &[f64],
    gap: f64,
) -> std::result::Result<f64, jelly_wave::Error> {
    jelly_wave::erp_distance(left, right, gap)
}

pub(crate) fn wave_frechet(
    left: &[f64],
    right: &[f64],
) -> std::result::Result<f64, jelly_wave::Error> {
    jelly_wave::discrete_frechet(left, right)
}

pub(crate) fn wave_soft_dtw(
    left: &[f64],
    right: &[f64],
    gamma: f64,
) -> std::result::Result<f64, jelly_wave::Error> {
    jelly_wave::soft_dtw(left, right, gamma, jelly_wave::LocalCost::Absolute)
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
    fn coronel_empty_maps_to_empty_matrix() {
        let err = failure_from_coronel("bridge", "map", coronel::Error::EmptyInput);
        assert_eq!(err.primary().code, IssueCode::EmptyMatrix);
    }

    #[test]
    fn jelly_wave_empty_maps_to_empty_matrix() {
        let err = failure_from_jelly_wave("bridge", "map", jelly_wave::Error::EmptyInput);
        assert_eq!(err.primary().code, IssueCode::EmptyMatrix);
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
