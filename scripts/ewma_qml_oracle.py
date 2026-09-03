#!/usr/bin/env python3
"""Generate and verify the EWMA Gaussian QML Tier-0 golden file.

This is deliberately a Python-standard-library oracle.  It mirrors the
mathematical recurrence in ``kanimiso/src/tsa.rs`` without mirroring the Rust
optimizer:

* demean the observations;
* divide residuals by their maximum absolute value;
* initialize normalized variance with the mean normalized squared residual;
* use ``h[t] = lambda * h[t-1] + (1-lambda) * x[t-1]**2``; and
* minimize ``sum((ln(h[t]) + x[t]**2 / h[t]) / 2)``; and
* report full NLL by adding the lambda-invariant ``n * ln(scale)`` term.

The production ``ewma_qml_objective`` minimizes the normalized profile using
an algebraically equivalent log-variance recurrence.  The high-precision
Decimal oracle deliberately evaluates the physical-variance recurrence above;
``binary64_replay`` mirrors the production log recurrence, including computing
normalized residual squares from logarithm differences so that division does
not erase representable small residuals, and exponential physical rescaling
exactly.  The test-only ``ewma_nll`` helper adds
``n * ln(scale)`` for replay; profile and full NLL have the same minimizing
lambda.  The additive ``n * ln(2*pi) / 2`` Gaussian constant is intentionally
absent from both.  Decimal arithmetic and an analytic objective derivative are
used so that the oracle does not depend on binary64 arithmetic or the
production Nelder-Mead implementation.

This validates kanimiso's recurrence, initialization, and configured bounds.
It is not an output-parity oracle for arch ``EWMAVariance``, whose backcast
initialization and estimation bounds differ.

Commands::

    python scripts/ewma_qml_oracle.py emit golden/ewma_qml.json
    python scripts/ewma_qml_oracle.py check golden/ewma_qml.json

Both commands run the full oracle twice (80 and 120 decimal digits).  ``check``
also requires a byte-for-byte match with the committed canonical JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Iterable, Sequence


ORACLE_PRECISION = 80
VERIFICATION_PRECISION = 120
OUTPUT_SIGNIFICANT_DIGITS = 50
DERIVATIVE_SCAN_INTERVALS = 16_384
ROOT_BRACKET_WIDTH = Decimal("1e-65")
EXTENDED_REAL_ZERO_COUNT = 1_198
EXTENDED_REAL_GRID_INTERVALS = 64


@dataclass(frozen=True)
class CaseSpec:
    name: str
    purpose: str
    observations: tuple[str, ...]
    bounds: tuple[str, str]


@dataclass(frozen=True)
class PreparedSeries:
    mean: Decimal
    residuals: tuple[Decimal, ...]
    scale: Decimal
    normalized: tuple[Decimal, ...]
    initial_variance: Decimal | None


@dataclass(frozen=True)
class Evaluation:
    nll: Decimal
    derivative: Decimal
    normalized_variances: tuple[Decimal, ...]
    sigma2: tuple[Decimal, ...]


@dataclass(frozen=True)
class Optimum:
    selection: str
    lambda_: Decimal
    evaluation: Evaluation
    lower_evaluation: Evaluation
    upper_evaluation: Evaluation
    stationary_points: tuple[Decimal, ...]
    maximum_root_bracket_width: Decimal


@dataclass(frozen=True)
class ExtendedRealEvidence:
    prepared: PreparedSeries
    lower_lambda: Decimal
    upper_lambda: Decimal
    lower_evaluation: Evaluation
    upper_evaluation: Evaluation
    grid_best_lambda: Decimal
    grid_best_evaluation: Evaluation
    grid_second_lambda: Decimal
    grid_second_evaluation: Evaluation


CASES = (
    CaseSpec(
        name="default_bounds_interior",
        purpose="Interior optimum used by the existing independent-oracle regression.",
        observations=(
            "8",
            "-8",
            "1",
            "-1",
            "0.5",
            "-0.5",
            "0.25",
            "-0.25",
            "0.25",
            "-0.25",
            "-2",
            "2",
        ),
        bounds=("0.5", "0.999"),
    ),
    CaseSpec(
        name="default_lower_bound",
        purpose="Monotonically preferred lower endpoint of the default interval.",
        observations=("8", "-8", "4", "-4", "2", "-2", "1", "-1"),
        bounds=("0.5", "0.999"),
    ),
    CaseSpec(
        name="default_upper_bound",
        purpose="Alternating shock scales select the upper endpoint of the default interval.",
        observations=(
            "8",
            "-1",
            "-8",
            "1",
            "8",
            "-1",
            "-8",
            "1",
            "8",
            "-1",
            "-8",
            "1",
        ),
        bounds=("0.5", "0.999"),
    ),
    CaseSpec(
        name="constant_unidentified",
        purpose="Demeaning leaves zero residuals, so lambda is unidentified.",
        observations=("7", "7", "7", "7"),
        bounds=("0.5", "0.999"),
    ),
    CaseSpec(
        name="custom_bounds_truncate_interior_optimum",
        purpose="A custom interval below the unconstrained fixture optimum selects its upper endpoint.",
        observations=(
            "8",
            "-8",
            "1",
            "-1",
            "0.5",
            "-0.5",
            "0.25",
            "-0.25",
            "0.25",
            "-0.25",
            "-2",
            "2",
        ),
        bounds=("0.55", "0.60"),
    ),
)


EXTENDED_REAL_CASE = CaseSpec(
    name="late_shock_extended_real_barrier",
    purpose=(
        "A long zero run makes the lower-end binary64 objective overflow while "
        "the Decimal objective remains finite and the constrained optimum is the "
        "active upper endpoint."
    ),
    observations=("1", "-1")
    + ("0",) * EXTENDED_REAL_ZERO_COUNT
    + ("1", "-1"),
    bounds=("0.5", "0.999"),
)


def _prepare(spec: CaseSpec) -> PreparedSeries:
    observations = tuple(Decimal(value) for value in spec.observations)
    count = Decimal(len(observations))
    mean = sum(observations, Decimal(0)) / count
    residuals = tuple(value - mean for value in observations)
    scale = max(abs(value) for value in residuals)
    if scale == 0:
        return PreparedSeries(mean, residuals, scale, (), None)
    normalized = tuple(value / scale for value in residuals)
    initial = sum((value * value for value in normalized), Decimal(0)) / count
    if initial <= 0:
        raise ArithmeticError(f"{spec.name}: non-positive initial variance")
    return PreparedSeries(mean, residuals, scale, normalized, initial)


def _evaluate(prepared: PreparedSeries, lambda_: Decimal) -> Evaluation:
    if prepared.initial_variance is None:
        raise ValueError("constant series has no identified EWMA objective")

    one = Decimal(1)
    half = Decimal("0.5")
    variance = prepared.initial_variance
    variance_derivative = Decimal(0)
    nll = Decimal(0)
    derivative = Decimal(0)
    variances: list[Decimal] = []

    for time, residual in enumerate(prepared.normalized):
        if variance <= 0:
            raise ArithmeticError("EWMA variance became non-positive")
        residual_square = residual * residual
        variances.append(variance)
        nll += half * (variance.ln() + residual_square / variance)
        derivative += half * variance_derivative * (
            one / variance - residual_square / (variance * variance)
        )

        if time + 1 < len(prepared.normalized):
            next_variance_derivative = (
                variance - residual_square + lambda_ * variance_derivative
            )
            variance = lambda_ * variance + (one - lambda_) * residual_square
            variance_derivative = next_variance_derivative

    nll += Decimal(len(prepared.normalized)) * prepared.scale.ln()
    scale_square = prepared.scale * prepared.scale
    sigma2 = tuple(scale_square * value for value in variances)
    return Evaluation(nll, derivative, tuple(variances), sigma2)


def _derivative(prepared: PreparedSeries, lambda_: Decimal) -> Decimal:
    """Evaluate only d(NLL)/d(lambda), avoiding Decimal logarithms in the scan."""
    if prepared.initial_variance is None:
        raise ValueError("constant series has no identified EWMA objective")
    one = Decimal(1)
    half = Decimal("0.5")
    variance = prepared.initial_variance
    variance_derivative = Decimal(0)
    derivative = Decimal(0)
    for time, residual in enumerate(prepared.normalized):
        residual_square = residual * residual
        derivative += half * variance_derivative * (
            one / variance - residual_square / (variance * variance)
        )
        if time + 1 < len(prepared.normalized):
            variance_derivative = (
                variance - residual_square + lambda_ * variance_derivative
            )
            variance = lambda_ * variance + (one - lambda_) * residual_square
    return derivative


def _bisect_derivative_root(
    prepared: PreparedSeries,
    left: Decimal,
    right: Decimal,
    left_derivative: Decimal,
    right_derivative: Decimal,
) -> tuple[Decimal, Decimal]:
    if left_derivative == 0:
        return left, Decimal(0)
    if right_derivative == 0:
        return right, Decimal(0)
    if left_derivative.is_signed() == right_derivative.is_signed():
        raise ValueError("derivative root is not bracketed")

    while right - left > ROOT_BRACKET_WIDTH:
        middle = (left + right) / Decimal(2)
        middle_derivative = _derivative(prepared, middle)
        if middle_derivative == 0:
            return middle, Decimal(0)
        if left_derivative.is_signed() == middle_derivative.is_signed():
            left = middle
            left_derivative = middle_derivative
        else:
            right = middle
            right_derivative = middle_derivative
    width = right - left
    return (left + right) / Decimal(2), width


def _deduplicate(values: Iterable[Decimal]) -> tuple[Decimal, ...]:
    ordered = sorted(values)
    result: list[Decimal] = []
    threshold = ROOT_BRACKET_WIDTH * Decimal(2)
    for value in ordered:
        if not result or abs(value - result[-1]) > threshold:
            result.append(value)
    return tuple(result)


def _minimize(prepared: PreparedSeries, bounds: tuple[str, str]) -> Optimum:
    lower, upper = (Decimal(value) for value in bounds)
    if not (Decimal(0) < lower < upper < Decimal(1)):
        raise ValueError(f"invalid search bounds {bounds!r}")

    step = (upper - lower) / Decimal(DERIVATIVE_SCAN_INTERVALS)
    previous_lambda = lower
    previous_derivative = _derivative(prepared, previous_lambda)
    roots: list[Decimal] = []
    root_widths: list[Decimal] = []

    for index in range(1, DERIVATIVE_SCAN_INTERVALS + 1):
        lambda_ = lower + step * Decimal(index)
        current_derivative = _derivative(prepared, lambda_)
        if previous_derivative == 0:
            roots.append(previous_lambda)
            root_widths.append(Decimal(0))
        elif current_derivative == 0:
            roots.append(lambda_)
            root_widths.append(Decimal(0))
        elif previous_derivative.is_signed() != current_derivative.is_signed():
            root, width = _bisect_derivative_root(
                prepared,
                previous_lambda,
                lambda_,
                previous_derivative,
                current_derivative,
            )
            roots.append(root)
            root_widths.append(width)
        previous_lambda = lambda_
        previous_derivative = current_derivative

    stationary_points = _deduplicate(roots)
    candidate_lambdas = (lower, upper, *stationary_points)
    candidates = tuple((value, _evaluate(prepared, value)) for value in candidate_lambdas)
    best_lambda, best_evaluation = min(candidates, key=lambda item: item[1].nll)
    selection = (
        "lower_bound"
        if best_lambda == lower
        else "upper_bound"
        if best_lambda == upper
        else "interior"
    )

    return Optimum(
        selection=selection,
        lambda_=best_lambda,
        evaluation=best_evaluation,
        lower_evaluation=_evaluate(prepared, lower),
        upper_evaluation=_evaluate(prepared, upper),
        stationary_points=stationary_points,
        maximum_root_bracket_width=max(root_widths, default=Decimal(0)),
    )


def _compute_extended_real_evidence(precision: int) -> ExtendedRealEvidence:
    with localcontext() as context:
        context.prec = precision
        context.rounding = ROUND_HALF_EVEN
        prepared = _prepare(EXTENDED_REAL_CASE)
        lower, upper = (Decimal(value) for value in EXTENDED_REAL_CASE.bounds)
        step = (upper - lower) / Decimal(EXTENDED_REAL_GRID_INTERVALS)
        grid = tuple(
            (
                lower + step * Decimal(index),
                _evaluate(prepared, lower + step * Decimal(index)),
            )
            for index in range(EXTENDED_REAL_GRID_INTERVALS + 1)
        )
        ordered = sorted(grid, key=lambda item: item[1].nll)
        return ExtendedRealEvidence(
            prepared=prepared,
            lower_lambda=lower,
            upper_lambda=upper,
            lower_evaluation=grid[0][1],
            upper_evaluation=grid[-1][1],
            grid_best_lambda=ordered[0][0],
            grid_best_evaluation=ordered[0][1],
            grid_second_lambda=ordered[1][0],
            grid_second_evaluation=ordered[1][1],
        )


def _decimal_text(value: Decimal, digits: int = OUTPUT_SIGNIFICANT_DIGITS) -> str:
    if value == 0:
        return "0"
    with localcontext() as context:
        context.prec = digits
        rounded = +value
    text = format(rounded, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _compact_decimal_text(
    value: Decimal, digits: int = OUTPUT_SIGNIFICANT_DIGITS
) -> str:
    """Use fixed notation for ordinary values and scientific notation for tails."""
    if value == 0:
        return "0"
    if -6 <= value.adjusted() < digits:
        return _decimal_text(value, digits)
    with localcontext() as context:
        context.prec = digits
        rounded = +value
    return format(rounded, "E").replace("E", "e")


def _scientific_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value, ".6E").replace("E", "e")


def _decimal_list(values: Sequence[Decimal]) -> list[str]:
    return [_decimal_text(value) for value in values]


def _binary64_exp(value: float) -> float:
    """Match Rust `f64::exp`: overflow is positive infinity, not an exception."""
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _binary64_replay(spec: CaseSpec, lambda_: Decimal) -> dict[str, object]:
    observations = [float(value) for value in spec.observations]
    mean = sum(observations) / len(observations)
    residuals = [value - mean for value in observations]
    scale = max(abs(value) for value in residuals)
    log_scale = math.log(scale)
    normalized_log_squares = [
        -math.inf if value == 0.0 else 2.0 * (math.log(abs(value)) - log_scale)
        for value in residuals
    ]
    maximum_log_square = max(normalized_log_squares)
    log_initial_variance = maximum_log_square + math.log(
        sum(_binary64_exp(value - maximum_log_square) for value in normalized_log_squares)
    ) - math.log(len(normalized_log_squares))
    nll = 0.0
    lambda_f64 = float(lambda_)
    log_lambda = math.log(lambda_f64)
    log_one_minus_lambda = math.log1p(-lambda_f64)
    log_variances = [log_initial_variance] * len(residuals)

    for time in range(1, len(residuals)):
        terms = (
            log_lambda + log_variances[time - 1],
            log_one_minus_lambda + normalized_log_squares[time - 1],
        )
        maximum = max(terms)
        shifted_sum = 0.0
        for value in terms:
            shifted_sum += _binary64_exp(value - maximum)
        log_variances[time] = maximum + math.log(shifted_sum)

    for normalized_log_square, log_variance in zip(
        normalized_log_squares, log_variances
    ):
        standardized_square = (
            0.0
            if normalized_log_square == -math.inf
            else _binary64_exp(normalized_log_square - log_variance)
        )
        nll += 0.5 * (log_variance + standardized_square)
    nll += len(residuals) * log_scale
    log_scale_square = 2.0 * log_scale
    sigma2 = [_binary64_exp(log_scale_square + value) for value in log_variances]
    return {
        "lambda": repr(lambda_f64),
        "nll": repr(nll),
        "normalized_log_variances": [repr(value) for value in log_variances],
        "sigma2": [repr(value) for value in sigma2],
    }


def _success_case_payload(
    spec: CaseSpec, prepared: PreparedSeries, optimum: Optimum
) -> dict[str, object]:
    replay = _binary64_replay(spec, optimum.lambda_)
    replay_nll_error = abs(
        Decimal.from_float(float(replay["nll"])) - optimum.evaluation.nll
    )
    replay_lambda_error = abs(
        Decimal.from_float(float(replay["lambda"])) - optimum.lambda_
    )
    replay_sigma2_errors = [
        abs(Decimal.from_float(float(actual)) - expected)
        for actual, expected in zip(replay["sigma2"], optimum.evaluation.sigma2)
    ]

    return {
        "name": spec.name,
        "purpose": spec.purpose,
        "input": {
            "observations": list(spec.observations),
            "search_bounds": list(spec.bounds),
        },
        "preprocessing": {
            "mean": _decimal_text(prepared.mean),
            "residuals": _decimal_list(prepared.residuals),
            "scale": _decimal_text(prepared.scale),
            "normalized_residuals": _decimal_list(prepared.normalized),
            "initial_normalized_variance": _decimal_text(
                prepared.initial_variance or Decimal(0)
            ),
        },
        "expected": {
            "outcome": "success",
            "selection": optimum.selection,
            "lambda": _decimal_text(optimum.lambda_),
            "nll_without_gaussian_constant": _decimal_text(optimum.evaluation.nll),
            "normalized_variances": _decimal_list(
                optimum.evaluation.normalized_variances
            ),
            "sigma2": _decimal_list(optimum.evaluation.sigma2),
            "objective_at_lower_bound": _decimal_text(
                optimum.lower_evaluation.nll
            ),
            "objective_at_upper_bound": _decimal_text(
                optimum.upper_evaluation.nll
            ),
            "derivative_at_selected_lambda": _scientific_text(
                optimum.evaluation.derivative
            ),
        },
        "stationary_point_scan": {
            "stationary_points": _decimal_list(optimum.stationary_points),
            "stationary_point_count": len(optimum.stationary_points),
            "maximum_root_bracket_width": _scientific_text(
                optimum.maximum_root_bracket_width
            ),
        },
        "binary64_replay": {
            **replay,
            "measured_lambda_abs_error": _scientific_text(replay_lambda_error),
            "measured_nll_abs_error": _scientific_text(replay_nll_error),
            "measured_max_sigma2_abs_error": _scientific_text(
                max(replay_sigma2_errors, default=Decimal(0))
            ),
        },
    }


def _constant_case_payload(spec: CaseSpec, prepared: PreparedSeries) -> dict[str, object]:
    return {
        "name": spec.name,
        "purpose": spec.purpose,
        "input": {
            "observations": list(spec.observations),
            "search_bounds": list(spec.bounds),
        },
        "preprocessing": {
            "mean": _decimal_text(prepared.mean),
            "residuals": _decimal_list(prepared.residuals),
            "scale": "0",
        },
        "expected": {
            "outcome": "failure",
            "issue_code": "UnidentifiedModel",
            "reason": "Demeaning produces only zero residuals; every lambda has the same zero-variance recurrence.",
        },
    }


def _binary64_status(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "positive_infinity"
    if value == -math.inf:
        return "negative_infinity"
    return "finite"


def _binary64_endpoint_payload(lambda_: Decimal) -> dict[str, object]:
    replay = _binary64_replay(EXTENDED_REAL_CASE, lambda_)
    nll = float(replay["nll"])
    sigma2 = [float(value) for value in replay["sigma2"]]
    nll_status = _binary64_status(nll)
    nonfinite_sigma2 = sum(not math.isfinite(value) for value in sigma2)
    zero_sigma2 = sum(value == 0.0 for value in sigma2)
    return {
        "lambda": replay["lambda"],
        "nll_status": nll_status,
        "nll_without_gaussian_constant": replay["nll"]
        if nll_status == "finite"
        else "+inf"
        if nll_status == "positive_infinity"
        else replay["nll"],
        "sigma2_status": "finite"
        if nonfinite_sigma2 == 0
        else "contains_nonfinite",
        "zero_sigma2_count": zero_sigma2,
        "nonfinite_sigma2_count": nonfinite_sigma2,
    }


def _extended_precision_verification(
    primary: ExtendedRealEvidence, verification: ExtendedRealEvidence
) -> dict[str, str]:
    lower_scale = max(
        Decimal(1),
        abs(primary.lower_evaluation.nll),
        abs(verification.lower_evaluation.nll),
    )
    lower_relative_drift = (
        abs(primary.lower_evaluation.nll - verification.lower_evaluation.nll)
        / lower_scale
    )
    upper_absolute_drift = abs(
        primary.upper_evaluation.nll - verification.upper_evaluation.nll
    )
    derivative_absolute_drift = abs(
        primary.upper_evaluation.derivative
        - verification.upper_evaluation.derivative
    )
    grid_lambda_drift = abs(primary.grid_best_lambda - verification.grid_best_lambda)
    result = {
        "lower_nll_relative_drift": _scientific_text(lower_relative_drift),
        "upper_nll_abs_drift": _scientific_text(upper_absolute_drift),
        "upper_derivative_abs_drift": _scientific_text(derivative_absolute_drift),
        "grid_minimum_lambda_abs_drift": _scientific_text(grid_lambda_drift),
    }
    if lower_relative_drift >= Decimal("1e-65"):
        raise AssertionError(f"extended lower-NLL precision verification failed: {result}")
    if upper_absolute_drift >= Decimal("1e-65"):
        raise AssertionError(f"extended upper-NLL precision verification failed: {result}")
    if derivative_absolute_drift >= Decimal("1e-60"):
        raise AssertionError(
            f"extended upper-derivative precision verification failed: {result}"
        )
    if grid_lambda_drift != 0:
        raise AssertionError(f"extended grid minimum changed with precision: {result}")
    return result


def _extended_real_payload(
    evidence: ExtendedRealEvidence, precision_drift: dict[str, str]
) -> dict[str, object]:
    if evidence.grid_best_lambda != evidence.upper_lambda:
        raise AssertionError(
            "late-shock Decimal grid minimum must occur at the upper endpoint"
        )
    upper_kkt_satisfied = evidence.upper_evaluation.derivative <= 0
    if not upper_kkt_satisfied:
        raise AssertionError(
            "late-shock upper endpoint must satisfy the one-sided KKT condition"
        )
    grid_gap = (
        evidence.grid_second_evaluation.nll - evidence.grid_best_evaluation.nll
    )
    return {
        "name": EXTENDED_REAL_CASE.name,
        "purpose": EXTENDED_REAL_CASE.purpose,
        "input_recipe": {
            "expression": "[1, -1] + [0] * 1198 + [1, -1]",
            "prefix": ["1", "-1"],
            "zero_count": EXTENDED_REAL_ZERO_COUNT,
            "suffix": ["1", "-1"],
            "expanded_length": len(EXTENDED_REAL_CASE.observations),
            "search_bounds": list(EXTENDED_REAL_CASE.bounds),
        },
        "preprocessing": {
            "mean": _decimal_text(evidence.prepared.mean),
            "scale": _decimal_text(evidence.prepared.scale),
            "initial_normalized_variance": _decimal_text(
                evidence.prepared.initial_variance or Decimal(0)
            ),
        },
        "decimal_evidence": {
            "arithmetic": "decimal.Decimal with ROUND_HALF_EVEN",
            "precision_digits": ORACLE_PRECISION,
            "objective": "full NLL without the lambda-invariant Gaussian constant; scale=1 makes it equal to the normalized profile",
            "endpoint_nlls": {
                "lower_bound": _compact_decimal_text(evidence.lower_evaluation.nll),
                "upper_bound": _compact_decimal_text(evidence.upper_evaluation.nll),
            },
            "derivative_at_upper_bound": _decimal_text(
                evidence.upper_evaluation.derivative
            ),
            "grid_evidence": {
                "intervals": EXTENDED_REAL_GRID_INTERVALS,
                "minimum_lambda": _decimal_text(evidence.grid_best_lambda),
                "minimum_nll": _decimal_text(evidence.grid_best_evaluation.nll),
                "second_best_lambda": _decimal_text(evidence.grid_second_lambda),
                "nll_gap_to_second_best": _decimal_text(grid_gap),
                "unique_minimum": grid_gap > 0,
            },
            "one_sided_kkt": {
                "condition": "At an upper bound in a scalar minimization problem, d(NLL)/d(lambda) <= 0.",
                "satisfied": upper_kkt_satisfied,
            },
            "conclusion": "The 64-interval Decimal grid selects lambda=0.999 uniquely, and the negative endpoint derivative satisfies the one-sided KKT condition for an active upper bound.",
            "scope_limit": "Grid evidence plus the endpoint KKT condition documents this fixture; it is not a general proof that an arbitrary objective has no unsampled narrow basin.",
            "precision_error_measurement": precision_drift,
        },
        "binary64_endpoint_statuses": {
            "lower_bound": _binary64_endpoint_payload(evidence.lower_lambda),
            "upper_bound": _binary64_endpoint_payload(evidence.upper_lambda),
        },
    }


def _compute_cases(precision: int) -> tuple[list[dict[str, object]], dict[str, Optimum]]:
    payloads: list[dict[str, object]] = []
    optima: dict[str, Optimum] = {}
    with localcontext() as context:
        context.prec = precision
        context.rounding = ROUND_HALF_EVEN
        for spec in CASES:
            prepared = _prepare(spec)
            if prepared.scale == 0:
                payloads.append(_constant_case_payload(spec, prepared))
                continue
            optimum = _minimize(prepared, spec.bounds)
            optima[spec.name] = optimum
            payloads.append(_success_case_payload(spec, prepared, optimum))
    return payloads, optima


def _precision_verification(
    primary: dict[str, Optimum], verification: dict[str, Optimum]
) -> dict[str, str]:
    lambda_errors: list[Decimal] = []
    nll_errors: list[Decimal] = []
    sigma2_errors: list[Decimal] = []
    for name, primary_optimum in primary.items():
        verification_optimum = verification[name]
        lambda_errors.append(abs(primary_optimum.lambda_ - verification_optimum.lambda_))
        nll_errors.append(
            abs(primary_optimum.evaluation.nll - verification_optimum.evaluation.nll)
        )
        sigma2_errors.extend(
            abs(left - right)
            for left, right in zip(
                primary_optimum.evaluation.sigma2,
                verification_optimum.evaluation.sigma2,
            )
        )
    result = {
        "max_lambda_abs_drift": _scientific_text(max(lambda_errors, default=Decimal(0))),
        "max_nll_abs_drift": _scientific_text(max(nll_errors, default=Decimal(0))),
        "max_sigma2_abs_drift": _scientific_text(max(sigma2_errors, default=Decimal(0))),
    }
    if max(lambda_errors, default=Decimal(0)) >= Decimal("1e-55"):
        raise AssertionError(f"lambda precision verification failed: {result}")
    if max(nll_errors, default=Decimal(0)) >= Decimal("1e-70"):
        raise AssertionError(f"NLL precision verification failed: {result}")
    if max(sigma2_errors, default=Decimal(0)) >= Decimal("1e-55"):
        raise AssertionError(f"sigma2 precision verification failed: {result}")
    return result


def build_payload() -> dict[str, object]:
    cases, primary = _compute_cases(ORACLE_PRECISION)
    _, verification = _compute_cases(VERIFICATION_PRECISION)
    precision_drift = _precision_verification(primary, verification)
    extended_primary = _compute_extended_real_evidence(ORACLE_PRECISION)
    extended_verification = _compute_extended_real_evidence(VERIFICATION_PRECISION)
    extended_precision_drift = _extended_precision_verification(
        extended_primary, extended_verification
    )
    expected_selections = {
        "default_bounds_interior": "interior",
        "default_lower_bound": "lower_bound",
        "default_upper_bound": "upper_bound",
        "custom_bounds_truncate_interior_optimum": "upper_bound",
    }
    for name, selection in expected_selections.items():
        if primary[name].selection != selection:
            raise AssertionError(
                f"{name}: expected {selection}, got {primary[name].selection}"
            )

    return {
        "schema_version": 1,
        "generator": "scripts/ewma_qml_oracle.py emit",
        "source_contract": {
            "implementation": "kanimiso/src/tsa.rs",
            "compatibility_scope": "Kanimiso recurrence, full-sample mean-square initialization, and configured bounds; not output parity with arch EWMAVariance backcast initialization or bounds.",
            "production_functions": [
                "ewma_normalized_log_variance",
                "ewma_profile_from_log_variance",
                "ewma_qml_objective",
                "ewma_sigma2",
            ],
            "test_replay_function": "ewma_nll (test-only full-NLL helper)",
            "inspected_on": "2026-09-02",
            "recurrence": "h[0]=mean(x^2); h[t]=lambda*h[t-1]+(1-lambda)*x[t-1]^2",
            "production_log_recurrence": "log_x2[t]=2*(log(abs(residual[t]))-log(scale)); log_h[0]=logsumexp(log_x2)-log(n); log_h[t]=logsumexp(log(lambda)+log_h[t-1], log(1-lambda)+log_x2[t-1]); a zero shock contributes negative infinity",
            "production_selection_objective": "0.5*sum(ln(h[t])+x[t]^2/h[t])",
            "reported_full_nll": "production_selection_objective+n*ln(max_abs_demeaned_residual)",
            "selection_equivalence": "The added scale term is constant in lambda, so profile and full NLL select the same lambda.",
            "gaussian_constant": "omitted to match the production objective",
        },
        "oracle": {
            "dependencies": "Python standard library only",
            "arithmetic": "decimal.Decimal with ROUND_HALF_EVEN",
            "precision_digits": ORACLE_PRECISION,
            "verification_precision_digits": VERIFICATION_PRECISION,
            "output_significant_digits": OUTPUT_SIGNIFICANT_DIGITS,
            "selection_method": "Scan the analytic first derivative on the full closed interval, bisect every detected sign-changing bracket, then compare those stationary-point candidates and both endpoints.",
            "derivative_scan_intervals": DERIVATIVE_SCAN_INTERVALS,
            "root_bracket_target_width": str(ROOT_BRACKET_WIDTH),
            "scan_limitation": "This dense deterministic scan is a fixture-selection heuristic, not a general global-optimality certificate: a finite sign scan cannot prove the absence of paired narrow roots or roots where the derivative does not change sign.",
            "precision_error_method": "Regenerate independently at 120 digits and compare unrounded lambda, NLL, and sigma2 values from the 80-digit run.",
            "precision_error_measurement": precision_drift,
        },
        "tolerance_policy": {
            "rule": "For each Rust assertion, measure binary64/optimizer error against this Decimal oracle, use 3-4 times that error, and retain the measurement in the test comment (AGENTS.md R9).",
            "binary64_measurements": "Each successful case records direct recurrence replay error; optimizer lambda error must be measured by the Rust consumer because this oracle intentionally does not reproduce Nelder-Mead.",
        },
        "case_count": len(cases),
        "cases": cases,
        "extended_real_case": _extended_real_payload(
            extended_primary, extended_precision_drift
        ),
    }


def render_payload() -> str:
    return json.dumps(
        build_payload(), ensure_ascii=True, indent=2, sort_keys=True
    ) + "\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def emit(path: Path) -> None:
    rendered = render_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {len(CASES)} cases -> {path}")
    print(f"sha256 { _sha256(rendered) }")


def check(path: Path) -> None:
    expected = render_payload()
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        raise SystemExit(
            f"{path} is stale: expected sha256 {_sha256(expected)}, "
            f"found {_sha256(actual)}"
        )
    print(f"ok {path}: {len(CASES)} cases, sha256 {_sha256(actual)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit_parser = subparsers.add_parser("emit", help="write canonical golden JSON")
    emit_parser.add_argument("path", type=Path)
    check_parser = subparsers.add_parser("check", help="verify canonical golden JSON")
    check_parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "emit":
        emit(arguments.path)
    else:
        check(arguments.path)


if __name__ == "__main__":
    main()
