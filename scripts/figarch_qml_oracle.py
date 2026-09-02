#!/usr/bin/env python3
"""Independent Decimal oracle for BBM/arch FIGARCH(1,d,1) Gaussian QML.

This script has no third-party dependency and does not reproduce kanimiso's
optimizer.  It evaluates the power-2 FIGARCH filter in Decimal arithmetic,
checks the lambda recurrence against an independently assembled ``a_j``
polynomial/convolution path, and emits deterministic JSON fixtures.

The finite-filter contract is explicit.  With ``K`` retained ARCH-infinity
coefficients and BBM mean-square backcast ``b``::

    delta_1 = d
    lambda_1 = d + phi - beta
    delta_j = ((j - 1 - d) / j) delta_(j-1)
    lambda_j = beta lambda_(j-1) + delta_j - phi delta_(j-1)
    h_t = omega / (1 - beta)
          + sum_{j=1..K} lambda_j * (e_(t-j)^2 if t >= j else b)

The omitted ``j > K`` tail is neither renormalized nor silently replaced.
Gaussian QML includes ``t=0`` and omits only the parameter-independent
``log(2*pi)`` constant.  Fixed cases are residual-kernel probes.  Full-QMLE
cases store observations and apply the public estimator's arithmetic-mean
demeaning exactly once before fitting volatility parameters.

Usage::

    python scripts/figarch_qml_oracle.py emit golden/figarch_qml.json
    python scripts/figarch_qml_oracle.py fast-check golden/figarch_qml.json
    python scripts/figarch_qml_oracle.py deep-check golden/figarch_qml.json

``deep-check`` builds the complete unformatted Decimal tree independently at
80 and 120 digits, compares exact schemas and ordered lists while traversing
mapping keys in sorted order, and only then formats either tree for JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence


PRIMARY_PRECISION = 80
VERIFICATION_PRECISION = 120
OUTPUT_SIGNIFICANT_DIGITS = 34
DEFAULT_TRUNCATION = 1_000
RAW_RELATIVE_TOLERANCE = Decimal("1e-36")
RAW_ABSOLUTE_FLOOR = Decimal("1e-30")
RAW_ABSOLUTE_TOLERANCE = Decimal("1e-44")
OPTIMIZER_TOLERANCE = Decimal("2e-21")
KKT_TOLERANCE = Decimal("2e-14")
OPEN_BOUNDARY_GUARD = Decimal("1e-11")
OMEGA_LOWER_GUARD = Decimal("1e-13")
MAX_BFGS_ITERATIONS = 180
MAX_LINE_SEARCH_STEPS = 90
EXPECTED_ARTIFACT_SHA256 = "a0d0e5f024b2d6cb686be80278c8209b5e506856177659666b531f9f5caad878"

ZERO = Decimal(0)
ONE = Decimal(1)
TWO = Decimal(2)
HALF = Decimal("0.5")


@dataclass(frozen=True)
class PreparedSeries:
    residuals: tuple[Decimal, ...]
    scale: Decimal
    scale_square: Decimal
    normalized: tuple[Decimal, ...]
    squares: tuple[Decimal, ...]
    backcast: Decimal
    log_scale: Decimal


@dataclass(frozen=True)
class FixedSpec:
    name: str
    purpose: str
    residuals: tuple[str, ...]
    truncation: int
    omega: str
    d: str
    phi: str
    beta: str


@dataclass(frozen=True)
class SimulationSpec:
    name: str
    purpose: str
    seed: int
    count: int
    burn: int
    truncation: int
    omega: str
    d: str
    phi: str
    beta: str
    location: str


@dataclass(frozen=True)
class QmlSpec:
    name: str
    purpose: str
    observations: tuple[str, ...]
    truncation: int
    simulation_parameters: tuple[str, str, str, str, str]


@dataclass(frozen=True)
class FaceSpec:
    name: str
    d_status: str
    phi_status: str
    beta_status: str


@dataclass(frozen=True)
class Decoded:
    omega: Decimal
    d: Decimal
    phi: Decimal
    beta: Decimal
    jacobian: tuple[tuple[Decimal, ...], ...]
    interior_probabilities: tuple[Decimal, ...]


@dataclass(frozen=True)
class Evaluation:
    objective: Decimal
    gradient: tuple[Decimal, Decimal, Decimal, Decimal]
    deltas: tuple[Decimal, ...]
    lambdas: tuple[Decimal, ...]
    normalized_variances: tuple[Decimal, ...]
    physical_variances: tuple[Decimal, ...]
    one_step_normalized_variance: Decimal
    one_step_physical_variance: Decimal


@dataclass(frozen=True)
class Candidate:
    face: FaceSpec
    coordinates: tuple[Decimal, ...]
    decoded: Decoded
    evaluation: Evaluation
    transformed_gradient: tuple[Decimal, ...]
    iterations: int
    converged: bool
    guarded: bool
    termination: str


@dataclass(frozen=True)
class SolveResult:
    selected: Candidate
    candidates: tuple[Candidate, ...]
    grid_evaluations: int


FIXED_SPECS = (
    FixedSpec(
        "hand_computable",
        "Exact rational delta/lambda/path fixture; also checks that t=0 and one-step forecasting use BBM backcast.",
        ("1", "2"),
        3,
        "1",
        "0.5",
        "0.125",
        "0.25",
    ),
    FixedSpec(
        "interior",
        "Ordinary admissible interior parameters and asymmetric residual magnitudes.",
        ("0.4", "-1.2", "0.7", "-0.3", "1.8", "-0.9", "0.2", "-1.5"),
        12,
        "0.09",
        "0.4",
        "0.2",
        "0.45",
    ),
    FixedSpec(
        "d_zero_garch_face",
        "The exact d=0 GARCH face, alpha=phi-beta, within the BBM sufficient wedge.",
        ("0.5", "-0.8", "1.4", "-0.2", "0.9", "-1.1", "0.3"),
        16,
        "0.12",
        "0",
        "0.35",
        "0.2",
    ),
    FixedSpec(
        "d_one_igarch_face",
        "The exact d=1, phi=0 IGARCH boundary with beta strictly below one.",
        ("0.2", "-0.6", "1.7", "-0.5", "0.8", "-1.3", "0.4"),
        16,
        "0.025",
        "1",
        "0",
        "0.8",
    ),
    FixedSpec(
        "phi_zero_face",
        "Exact phi=0 boundary with positive fractional order and beta.",
        ("0.6", "-1.4", "0.1", "-0.7", "1.2", "-0.4", "0.9"),
        18,
        "0.07",
        "0.45",
        "0",
        "0.3",
    ),
    FixedSpec(
        "beta_zero_face",
        "Exact beta=0 boundary with interior d and phi.",
        ("0.3", "-0.9", "1.6", "-0.2", "0.5", "-1.1", "0.8"),
        18,
        "0.08",
        "0.3",
        "0.2",
        "0",
    ),
    FixedSpec(
        "beta_upper_face",
        "Exact beta=d+phi positivity boundary.",
        ("0.8", "-0.3", "1.5", "-1.0", "0.2", "-0.6", "1.1"),
        20,
        "0.055",
        "0.35",
        "0.1",
        "0.45",
    ),
    FixedSpec(
        "phi_upper_face",
        "Exact phi=(1-d)/2 positivity boundary.",
        ("0.7", "-1.5", "0.25", "-0.5", "1.3", "-0.8", "0.4"),
        20,
        "0.065",
        "0.4",
        "0.3",
        "0.2",
    ),
    FixedSpec(
        "near_persistent_default_k",
        "Near d=1 and beta=d+phi with the explicit default K=1000; the omitted tail is not renormalized.",
        ("0.15", "-0.45", "1.9", "-0.7", "0.35", "-1.2", "0.6", "-0.25"),
        DEFAULT_TRUNCATION,
        "0.0002",
        "0.998",
        "0.0008",
        "0.9987",
    ),
)


SIMULATION_SPECS = (
    SimulationSpec(
        "qml_interior_simulated",
        "Full exhaustive-face QMLE on a deterministic interior FIGARCH simulation.",
        1729,
        44,
        90,
        42,
        "0.045",
        "0.42",
        "0.16",
        "0.38",
        "1.25",
    ),
    SimulationSpec(
        "qml_d_zero_simulated",
        "Full exhaustive-face QMLE on a deterministic d=0 GARCH-face simulation.",
        2718,
        44,
        90,
        42,
        "0.07",
        "0",
        "0.32",
        "0.18",
        "-0.8",
    ),
    SimulationSpec(
        "qml_beta_zero_simulated",
        "Full exhaustive-face QMLE on a deterministic beta=0 FIGARCH simulation.",
        31415,
        44,
        90,
        42,
        "0.04",
        "0.55",
        "0.12",
        "0",
        "0.35",
    ),
    SimulationSpec(
        "qml_all_interior_simulated",
        "Full exhaustive-face QMLE whose attained Decimal solution is interior in d, phi, and beta.",
        17,
        100,
        120,
        12,
        "0.045",
        "0.42",
        "0.16",
        "0.38",
        "1.25",
    ),
)


def _face_specs() -> tuple[FaceSpec, ...]:
    """Enumerate every non-degenerate exact face of the sufficient wedge.

    ``d=0, beta=phi`` has lambda_j=0 for every j, so every such face is the
    same constant-variance model after omega is profiled.  Those aliases are
    represented once by ``d_lower__phi_lower__beta_lower``.  At ``d=1`` the
    phi lower and upper faces coincide at zero, and ``beta=d+phi=1`` is outside
    the strict beta<1 domain; both duplicates are therefore omitted.
    """

    faces: list[FaceSpec] = []
    statuses = ("lower", "interior", "upper")
    for d_status in statuses:
        for phi_status in statuses:
            if d_status == "upper" and phi_status != "lower":
                continue
            for beta_status in statuses:
                if d_status == "upper" and beta_status == "upper":
                    continue
                if d_status == "lower" and phi_status == "lower" and beta_status != "lower":
                    continue
                if d_status == "lower" and beta_status == "upper":
                    continue
                name = f"d_{d_status}__phi_{phi_status}__beta_{beta_status}"
                faces.append(FaceSpec(name, d_status, phi_status, beta_status))
    result = tuple(faces)
    if len(result) != 16:
        raise AssertionError(f"face topology changed: {len(result)}")
    return result


FACE_SPECS = _face_specs()


def _prepare(values: Sequence[str | Decimal]) -> PreparedSeries:
    residuals = tuple(Decimal(value) for value in values)
    if not residuals:
        raise ValueError("FIGARCH requires at least one residual")
    if any(not value.is_finite() for value in residuals):
        raise ValueError("residuals must be finite")
    scale = max(abs(value) for value in residuals)
    if scale == ZERO:
        raise ValueError("all-zero residual series has no attained omega>0 QMLE")
    normalized = tuple(value / scale for value in residuals)
    squares = tuple(value * value for value in normalized)
    backcast = sum(squares, ZERO) / Decimal(len(squares))
    return PreparedSeries(
        residuals,
        scale,
        scale * scale,
        normalized,
        squares,
        backcast,
        scale.ln(),
    )


def _valid_parameters(
    omega: Decimal, d: Decimal, phi: Decimal, beta: Decimal, truncation: int
) -> bool:
    return (
        truncation >= 1
        and omega.is_finite()
        and d.is_finite()
        and phi.is_finite()
        and beta.is_finite()
        and omega > ZERO
        and ZERO <= d <= ONE
        and ZERO <= phi <= (ONE - d) / TWO
        and ZERO <= beta <= d + phi
        and beta < ONE
    )


def _weights_with_derivatives(
    d: Decimal, phi: Decimal, beta: Decimal, truncation: int
) -> tuple[
    tuple[Decimal, ...],
    tuple[Decimal, ...],
    tuple[tuple[Decimal, Decimal, Decimal], ...],
]:
    """Primary delta/lambda recurrence and d/phi/beta derivatives."""

    deltas = [d]
    delta_d = [ONE]
    lambdas = [d + phi - beta]
    derivatives: list[tuple[Decimal, Decimal, Decimal]] = [(ONE, ONE, -ONE)]
    for lag in range(2, truncation + 1):
        divisor = Decimal(lag)
        ratio = (Decimal(lag - 1) - d) / divisor
        previous_delta = deltas[-1]
        current_delta = previous_delta * ratio
        current_delta_d = delta_d[-1] * ratio - previous_delta / divisor
        previous_lambda = lambdas[-1]
        previous_derivative = derivatives[-1]
        current_lambda = (
            beta * previous_lambda + current_delta - phi * previous_delta
        )
        current_derivative = (
            beta * previous_derivative[0] + current_delta_d - phi * delta_d[-1],
            beta * previous_derivative[1] - previous_delta,
            previous_lambda + beta * previous_derivative[2],
        )
        deltas.append(current_delta)
        delta_d.append(current_delta_d)
        lambdas.append(current_lambda)
        derivatives.append(current_derivative)
    return tuple(deltas), tuple(lambdas), tuple(derivatives)


def _alternate_a_lambda_weights(
    d: Decimal, phi: Decimal, beta: Decimal, truncation: int
) -> tuple[Decimal, ...]:
    """Independent polynomial ``a_j`` path followed by direct convolution.

    The code forms coefficients of ``(1-L)^d`` rather than calling the primary
    delta helper.  It then forms

      ``a(L) = (1-beta L) - (1-phi L)(1-L)^d``

    and convolves ``a_j`` with ``1/(1-beta L)`` directly.  It deliberately does
    not use the lambda recurrence.
    """

    fractional = [ONE]
    for lag in range(1, truncation + 1):
        fractional.append(
            fractional[-1]
            * (Decimal(lag - 1) - d)
            / Decimal(lag)
        )
    forcing: list[Decimal] = []
    for lag in range(1, truncation + 1):
        coefficient = -fractional[lag] + phi * fractional[lag - 1]
        if lag == 1:
            coefficient -= beta
        forcing.append(coefficient)
    lambdas: list[Decimal] = []
    for lag in range(1, truncation + 1):
        total = ZERO
        beta_power = ONE
        for forcing_lag in range(lag, 0, -1):
            total += beta_power * forcing[forcing_lag - 1]
            beta_power *= beta
        lambdas.append(total)
    return tuple(lambdas)


def _path_from_weights(
    prepared: PreparedSeries,
    omega: Decimal,
    beta: Decimal,
    lambdas: Sequence[Decimal],
) -> tuple[tuple[Decimal, ...], Decimal]:
    intercept = omega / (ONE - beta)
    path: list[Decimal] = []
    count = len(prepared.squares)
    for time in range(count + 1):
        variance = intercept
        for lag, weight in enumerate(lambdas, start=1):
            index = time - lag
            shock = prepared.squares[index] if index >= 0 else prepared.backcast
            variance += weight * shock
        if variance <= ZERO or not variance.is_finite():
            raise ArithmeticError("non-positive or non-finite FIGARCH variance")
        path.append(variance)
    return tuple(path[:-1]), path[-1]


def _evaluate_normalized(
    prepared: PreparedSeries,
    omega: Decimal,
    d: Decimal,
    phi: Decimal,
    beta: Decimal,
    truncation: int,
) -> Evaluation:
    if not _valid_parameters(omega, d, phi, beta, truncation):
        raise ValueError("parameter outside FIGARCH sufficient domain")
    deltas, lambdas, lambda_derivatives = _weights_with_derivatives(
        d, phi, beta, truncation
    )
    # The sufficient wedge implies nonnegative ARCH-infinity coefficients.
    if any(weight < ZERO for weight in lambdas):
        raise ArithmeticError("negative lambda inside stated sufficient wedge")

    intercept = omega / (ONE - beta)
    intercept_derivative = (
        ONE / (ONE - beta),
        ZERO,
        ZERO,
        omega / ((ONE - beta) * (ONE - beta)),
    )
    gradient = [ZERO, ZERO, ZERO, ZERO]
    objective = Decimal(len(prepared.squares)) * prepared.log_scale
    normalized_variances: list[Decimal] = []

    for time, observed_square in enumerate(prepared.squares):
        variance = intercept
        variance_derivative = list(intercept_derivative)
        for lag, (weight, derivative) in enumerate(
            zip(lambdas, lambda_derivatives), start=1
        ):
            index = time - lag
            shock = prepared.squares[index] if index >= 0 else prepared.backcast
            variance += weight * shock
            variance_derivative[1] += derivative[0] * shock
            variance_derivative[2] += derivative[1] * shock
            variance_derivative[3] += derivative[2] * shock
        if variance <= ZERO or not variance.is_finite():
            raise ArithmeticError("non-positive or non-finite FIGARCH variance")
        objective += HALF * (variance.ln() + observed_square / variance)
        score = HALF * (variance - observed_square) / (variance * variance)
        for index in range(4):
            gradient[index] += score * variance_derivative[index]
        normalized_variances.append(variance)

    one_step = intercept
    time = len(prepared.squares)
    for lag, weight in enumerate(lambdas, start=1):
        index = time - lag
        shock = prepared.squares[index] if index >= 0 else prepared.backcast
        one_step += weight * shock
    if one_step <= ZERO or not one_step.is_finite():
        raise ArithmeticError("invalid one-step FIGARCH variance")

    physical = tuple(value * prepared.scale_square for value in normalized_variances)
    return Evaluation(
        objective,
        tuple(gradient),  # type: ignore[arg-type]
        deltas,
        lambdas,
        tuple(normalized_variances),
        physical,
        one_step,
        one_step * prepared.scale_square,
    )


def _objective_or_infinity(
    prepared: PreparedSeries,
    omega: Decimal,
    d: Decimal,
    phi: Decimal,
    beta: Decimal,
    truncation: int,
    *,
    alternate: bool = False,
) -> Decimal:
    if not _valid_parameters(omega, d, phi, beta, truncation):
        return Decimal("Infinity")
    try:
        if alternate:
            lambdas = _alternate_a_lambda_weights(d, phi, beta, truncation)
            path, _ = _path_from_weights(prepared, omega, beta, lambdas)
            return Decimal(len(path)) * prepared.log_scale + sum(
                (
                    HALF * (variance.ln() + square / variance)
                    for variance, square in zip(path, prepared.squares)
                ),
                ZERO,
            )
        return _evaluate_normalized(
            prepared, omega, d, phi, beta, truncation
        ).objective
    except (ArithmeticError, InvalidOperation, ValueError):
        return Decimal("Infinity")


def _sigmoid(value: Decimal) -> Decimal:
    if value >= ZERO:
        exp_negative = (-value).exp()
        return ONE / (ONE + exp_negative)
    exp_positive = value.exp()
    return exp_positive / (ONE + exp_positive)


def _logit(value: Decimal) -> Decimal:
    if not ZERO < value < ONE:
        raise ValueError("logit input must be interior")
    return (value / (ONE - value)).ln()


def _decode(face: FaceSpec, coordinates: Sequence[Decimal]) -> Decoded:
    expected = 1 + sum(
        status == "interior"
        for status in (face.d_status, face.phi_status, face.beta_status)
    )
    if len(coordinates) != expected:
        raise ValueError(f"{face.name}: expected {expected} coordinates")
    size = len(coordinates)
    cursor = 1
    omega = coordinates[0].exp()
    omega_jacobian = [ZERO] * size
    omega_jacobian[0] = omega
    probabilities: list[Decimal] = []

    d_jacobian = [ZERO] * size
    if face.d_status == "lower":
        d = ZERO
    elif face.d_status == "upper":
        d = ONE
    else:
        d = _sigmoid(coordinates[cursor])
        probabilities.append(d)
        d_jacobian[cursor] = d * (ONE - d)
        cursor += 1

    phi_bound = (ONE - d) / TWO
    phi_bound_jacobian = [-value / TWO for value in d_jacobian]
    phi_jacobian = [ZERO] * size
    if face.phi_status == "lower":
        phi = ZERO
    elif face.phi_status == "upper":
        phi = phi_bound
        phi_jacobian = list(phi_bound_jacobian)
    else:
        ratio = _sigmoid(coordinates[cursor])
        probabilities.append(ratio)
        phi = phi_bound * ratio
        phi_jacobian = [value * ratio for value in phi_bound_jacobian]
        phi_jacobian[cursor] += phi_bound * ratio * (ONE - ratio)
        cursor += 1

    beta_bound = d + phi
    beta_bound_jacobian = [
        left + right for left, right in zip(d_jacobian, phi_jacobian)
    ]
    beta_jacobian = [ZERO] * size
    if face.beta_status == "lower":
        beta = ZERO
    elif face.beta_status == "upper":
        beta = beta_bound
        beta_jacobian = list(beta_bound_jacobian)
    else:
        ratio = _sigmoid(coordinates[cursor])
        probabilities.append(ratio)
        beta = beta_bound * ratio
        beta_jacobian = [value * ratio for value in beta_bound_jacobian]
        beta_jacobian[cursor] += beta_bound * ratio * (ONE - ratio)
        cursor += 1

    if cursor != size:
        raise AssertionError("decode coordinate cursor mismatch")
    if not _valid_parameters(omega, d, phi, beta, 1):
        raise ValueError(f"decoded invalid face point for {face.name}")
    return Decoded(
        omega,
        d,
        phi,
        beta,
        (
            tuple(omega_jacobian),
            tuple(d_jacobian),
            tuple(phi_jacobian),
            tuple(beta_jacobian),
        ),
        tuple(probabilities),
    )


def _transformed_evaluation(
    prepared: PreparedSeries,
    face: FaceSpec,
    coordinates: Sequence[Decimal],
    truncation: int,
) -> tuple[Decoded, Evaluation, tuple[Decimal, ...]]:
    decoded = _decode(face, coordinates)
    evaluation = _evaluate_normalized(
        prepared,
        decoded.omega,
        decoded.d,
        decoded.phi,
        decoded.beta,
        truncation,
    )
    transformed = tuple(
        sum(
            evaluation.gradient[row] * decoded.jacobian[row][column]
            for row in range(4)
        )
        for column in range(len(coordinates))
    )
    return decoded, evaluation, transformed


def _dot(left: Sequence[Decimal], right: Sequence[Decimal]) -> Decimal:
    return sum((a * b for a, b in zip(left, right)), ZERO)


def _identity(size: int) -> list[list[Decimal]]:
    return [
        [ONE if row == column else ZERO for column in range(size)]
        for row in range(size)
    ]


def _mat_vec(
    matrix: Sequence[Sequence[Decimal]], vector: Sequence[Decimal]
) -> list[Decimal]:
    return [
        sum((value * item for value, item in zip(row, vector)), ZERO)
        for row in matrix
    ]


def _inverse_bfgs_update(
    inverse_hessian: Sequence[Sequence[Decimal]],
    step: Sequence[Decimal],
    gradient_delta: Sequence[Decimal],
) -> list[list[Decimal]]:
    size = len(step)
    curvature = _dot(gradient_delta, step)
    scale = max(ONE, *(abs(value) for value in step), *(abs(value) for value in gradient_delta))
    if curvature <= Decimal("1e-55") * scale * scale:
        return _identity(size)
    rho = ONE / curvature
    left = [
        [
            (ONE if row == column else ZERO)
            - rho * step[row] * gradient_delta[column]
            for column in range(size)
        ]
        for row in range(size)
    ]
    right = [
        [
            (ONE if row == column else ZERO)
            - rho * gradient_delta[row] * step[column]
            for column in range(size)
        ]
        for row in range(size)
    ]
    middle = [
        [
            sum(
                (left[row][index] * inverse_hessian[index][column] for index in range(size)),
                ZERO,
            )
            for column in range(size)
        ]
        for row in range(size)
    ]
    return [
        [
            sum(
                (middle[row][index] * right[index][column] for index in range(size)),
                ZERO,
            )
            + rho * step[row] * step[column]
            for column in range(size)
        ]
        for row in range(size)
    ]


def _guarded(prepared: PreparedSeries, decoded: Decoded) -> bool:
    if decoded.omega <= OMEGA_LOWER_GUARD * prepared.backcast:
        return True
    return any(
        probability <= OPEN_BOUNDARY_GUARD
        or ONE - probability <= OPEN_BOUNDARY_GUARD
        for probability in decoded.interior_probabilities
    )


def _bfgs(
    prepared: PreparedSeries,
    face: FaceSpec,
    initial: Sequence[Decimal],
    truncation: int,
) -> Candidate:
    coordinates = list(initial)
    decoded, evaluation, transformed_tuple = _transformed_evaluation(
        prepared, face, coordinates, truncation
    )
    gradient = list(transformed_tuple)
    inverse_hessian = _identity(len(coordinates))
    termination = "max_iterations"
    iterations = 0

    for iteration in range(MAX_BFGS_ITERATIONS):
        iterations = iteration
        if _guarded(prepared, decoded):
            termination = "open_boundary_guard"
            break
        if max(abs(value) for value in gradient) <= OPTIMIZER_TOLERANCE:
            termination = "gradient"
            break
        direction = [-value for value in _mat_vec(inverse_hessian, gradient)]
        directional = _dot(gradient, direction)
        if directional >= ZERO:
            direction = [-value for value in gradient]
            directional = -_dot(gradient, gradient)
            inverse_hessian = _identity(len(coordinates))
        accepted = False
        step_size = ONE
        for _ in range(MAX_LINE_SEARCH_STEPS):
            trial = [
                old + step_size * delta
                for old, delta in zip(coordinates, direction)
            ]
            try:
                next_decoded, next_evaluation, next_gradient_tuple = (
                    _transformed_evaluation(prepared, face, trial, truncation)
                )
            except (ArithmeticError, InvalidOperation, ValueError):
                step_size *= HALF
                continue
            if next_evaluation.objective <= (
                evaluation.objective
                + Decimal("1e-4") * step_size * directional
            ):
                accepted = True
                break
            step_size *= HALF
        if not accepted:
            termination = "line_search"
            break
        next_coordinates = trial
        next_gradient = list(next_gradient_tuple)
        actual_step = [
            new - old for new, old in zip(next_coordinates, coordinates)
        ]
        gradient_delta = [
            new - old for new, old in zip(next_gradient, gradient)
        ]
        inverse_hessian = _inverse_bfgs_update(
            inverse_hessian, actual_step, gradient_delta
        )
        coordinates = next_coordinates
        decoded = next_decoded
        evaluation = next_evaluation
        gradient = next_gradient
    else:
        iterations = MAX_BFGS_ITERATIONS

    guarded = _guarded(prepared, decoded)
    tangent_small = max(abs(value) for value in gradient) <= KKT_TOLERANCE
    converged = tangent_small and not guarded
    return Candidate(
        face,
        tuple(coordinates),
        decoded,
        evaluation,
        tuple(gradient),
        iterations,
        converged,
        guarded,
        termination,
    )


def _seed_coordinates(
    prepared: PreparedSeries,
    face: FaceSpec,
    omega_multiplier: Decimal,
    d_probability: Decimal,
    phi_probability: Decimal,
    beta_probability: Decimal,
) -> tuple[Decimal, ...]:
    coordinates = [ZERO]
    if face.d_status == "interior":
        coordinates.append(_logit(d_probability))
    if face.phi_status == "interior":
        coordinates.append(_logit(phi_probability))
    if face.beta_status == "interior":
        coordinates.append(_logit(beta_probability))
    decoded = _decode(face, coordinates)
    omega = prepared.backcast * omega_multiplier * (ONE - decoded.beta)
    coordinates[0] = omega.ln()
    return tuple(coordinates)


def _face_seed_grid(
    prepared: PreparedSeries, face: FaceSpec, truncation: int
) -> tuple[tuple[Decimal, ...], int]:
    omega_values = tuple(map(Decimal, ("0.015", "0.07", "0.3")))
    d_values = tuple(map(Decimal, ("0.18", "0.50", "0.82")))
    phi_values = tuple(map(Decimal, ("0.20", "0.55", "0.85")))
    beta_values = tuple(map(Decimal, ("0.18", "0.55", "0.88")))
    d_grid = d_values if face.d_status == "interior" else (HALF,)
    phi_grid = phi_values if face.phi_status == "interior" else (HALF,)
    beta_grid = beta_values if face.beta_status == "interior" else (HALF,)
    best_coordinates: tuple[Decimal, ...] | None = None
    best_objective = Decimal("Infinity")
    evaluations = 0
    for omega_multiplier, d_probability, phi_probability, beta_probability in product(
        omega_values, d_grid, phi_grid, beta_grid
    ):
        coordinates = _seed_coordinates(
            prepared,
            face,
            omega_multiplier,
            d_probability,
            phi_probability,
            beta_probability,
        )
        decoded = _decode(face, coordinates)
        objective = _objective_or_infinity(
            prepared,
            decoded.omega,
            decoded.d,
            decoded.phi,
            decoded.beta,
            truncation,
        )
        evaluations += 1
        if objective < best_objective:
            best_objective = objective
            best_coordinates = coordinates
    if best_coordinates is None:
        raise ArithmeticError(f"no finite seed on {face.name}")
    return best_coordinates, evaluations


def _solve(prepared: PreparedSeries, truncation: int) -> SolveResult:
    candidates: list[Candidate] = []
    grid_evaluations = 0
    for face in FACE_SPECS:
        seed, evaluations = _face_seed_grid(prepared, face, truncation)
        grid_evaluations += evaluations
        candidates.append(_bfgs(prepared, face, seed, truncation))
    eligible = [candidate for candidate in candidates if candidate.converged]
    if not eligible:
        summary = ", ".join(
            f"{item.face.name}:{item.termination}:g={max(abs(v) for v in item.transformed_gradient)}"
            for item in candidates
        )
        raise ArithmeticError(f"no attained face candidate ({summary})")
    minimum = min(candidate.evaluation.objective for candidate in eligible)
    tie = Decimal("1e-18") * max(ONE, abs(minimum))
    equivalent = [
        candidate
        for candidate in eligible
        if candidate.evaluation.objective <= minimum + tie
    ]
    selected = min(
        equivalent,
        key=lambda item: (len(item.coordinates), item.face.name),
    )
    return SolveResult(selected, tuple(candidates), grid_evaluations)


def _lcg_surrogates(count: int, seed: int) -> tuple[Decimal, ...]:
    modulus = 2_147_483_647
    multiplier = 48_271
    state = seed % modulus
    values: list[Decimal] = []
    denominator = Decimal(modulus)
    for _ in range(count):
        total = ZERO
        for _ in range(12):
            state = (multiplier * state) % modulus
            total += Decimal(state) / denominator
        values.append(total - Decimal(6))
    mean = sum(values, ZERO) / Decimal(count)
    centered = [value - mean for value in values]
    rms = (sum((value * value for value in centered), ZERO) / Decimal(count)).sqrt()
    return tuple(value / rms for value in centered)


def _simulate(spec: SimulationSpec) -> QmlSpec:
    with localcontext() as context:
        context.prec = 140
        context.Emax = 999_999_999
        context.Emin = -999_999_999
        omega, d, phi, beta = map(
            Decimal, (spec.omega, spec.d, spec.phi, spec.beta)
        )
        _, lambdas, _ = _weights_with_derivatives(
            d, phi, beta, spec.truncation
        )
        innovations = _lcg_surrogates(spec.burn + spec.count, spec.seed)
        squares: list[Decimal] = []
        residuals: list[Decimal] = []
        intercept = omega / (ONE - beta)
        for innovation in innovations:
            variance = intercept
            for lag, weight in enumerate(lambdas, start=1):
                shock = squares[-lag] if lag <= len(squares) else ONE
                variance += weight * shock
            residual = variance.sqrt() * innovation
            residuals.append(residual)
            squares.append(residual * residual)
        kept = residuals[spec.burn :]
        location = Decimal(spec.location)
        texts = tuple(format(value + location, ".45g") for value in kept)
    return QmlSpec(
        spec.name,
        spec.purpose,
        texts,
        spec.truncation,
        (spec.omega, spec.d, spec.phi, spec.beta, spec.location),
    )


def _qml_specs() -> tuple[QmlSpec, ...]:
    return tuple(_simulate(spec) for spec in SIMULATION_SPECS)


def _weight_indices(truncation: int) -> tuple[int, ...]:
    if truncation <= 24:
        return tuple(range(1, truncation + 1))
    requested = (1, 2, 3, 4, 5, 10, 20, 50, 100, 250, 500, 1_000)
    return tuple(index for index in requested if index <= truncation)


def _checkpoints(values: Sequence[Decimal], indices: Sequence[int]) -> list[dict[str, object]]:
    return [{"lag": index, "value": values[index - 1]} for index in indices]


def _alternate_error(
    d: Decimal, phi: Decimal, beta: Decimal, truncation: int, primary: Sequence[Decimal]
) -> Decimal:
    alternate = _alternate_a_lambda_weights(d, phi, beta, truncation)
    return max((abs(left - right) for left, right in zip(primary, alternate)), default=ZERO)


def _fixed_record(spec: FixedSpec) -> dict[str, object]:
    prepared = _prepare(spec.residuals)
    omega_physical, d, phi, beta = map(
        Decimal, (spec.omega, spec.d, spec.phi, spec.beta)
    )
    omega_normalized = omega_physical / prepared.scale_square
    evaluation = _evaluate_normalized(
        prepared, omega_normalized, d, phi, beta, spec.truncation
    )
    alternate = _alternate_a_lambda_weights(d, phi, beta, spec.truncation)
    alternate_path, alternate_forecast = _path_from_weights(
        prepared, omega_normalized, beta, alternate
    )
    path_error = max(
        (
            abs(left - right)
            for left, right in zip(evaluation.normalized_variances, alternate_path)
        ),
        default=ZERO,
    )
    forecast_error = abs(evaluation.one_step_normalized_variance - alternate_forecast)
    indices = _weight_indices(spec.truncation)
    if spec.name == "hand_computable":
        expected_lambdas = tuple(map(Decimal, ("0.375", "0.15625", "0.0859375")))
        if evaluation.lambdas != expected_lambdas:
            raise AssertionError("hand lambda fixture changed")
        tolerance = Decimal(10) ** Decimal(-max(20, PRIMARY_PRECISION - 10))
        # Residual normalization divides physical variances by scale^2 = 4.
        expected_path = (Decimal(2209) / Decimal(3072), Decimal(1777) / Decimal(3072))
        expected_forecast = Decimal(2461) / Decimal(3072)
        if max(
            abs(left - right)
            for left, right in zip(evaluation.normalized_variances, expected_path)
        ) > tolerance:
            raise AssertionError("hand variance path changed")
        if abs(evaluation.one_step_normalized_variance - expected_forecast) > tolerance:
            raise AssertionError("hand one-step fixture changed")
    return {
        "name": spec.name,
        "purpose": spec.purpose,
        "residuals": list(spec.residuals),
        "truncation": spec.truncation,
        "scale": prepared.scale,
        "backcast_normalized": prepared.backcast,
        "parameters": {
            "omega_physical": omega_physical,
            "omega_normalized": omega_normalized,
            "d": d,
            "phi": phi,
            "beta": beta,
        },
        "objective": evaluation.objective,
        "delta_checkpoints": _checkpoints(evaluation.deltas, indices),
        "lambda_checkpoints": _checkpoints(evaluation.lambdas, indices),
        "lambda_sum_retained": sum(evaluation.lambdas, ZERO),
        "normalized_variances": list(evaluation.normalized_variances),
        "physical_variances": list(evaluation.physical_variances),
        "one_step_forecast": {
            "normalized_variance": evaluation.one_step_normalized_variance,
            "physical_variance": evaluation.one_step_physical_variance,
        },
        "alternate_a_path": {
            "lambda_max_absolute_error": max(
                (abs(left - right) for left, right in zip(evaluation.lambdas, alternate)),
                default=ZERO,
            ),
            "variance_path_max_absolute_error": path_error,
            "forecast_absolute_error": forecast_error,
        },
    }


def _candidate_record(candidate: Candidate, prepared: PreparedSeries) -> dict[str, object]:
    decoded = candidate.decoded
    return {
        "face": candidate.face.name,
        "d_status": candidate.face.d_status,
        "phi_status": candidate.face.phi_status,
        "beta_status": candidate.face.beta_status,
        "omega_normalized": decoded.omega,
        "omega_physical": decoded.omega * prepared.scale_square,
        "d": decoded.d,
        "phi": decoded.phi,
        "beta": decoded.beta,
        "objective": candidate.evaluation.objective,
        "tangent_gradient_max_absolute": max(
            (abs(value) for value in candidate.transformed_gradient), default=ZERO
        ),
        "iterations": candidate.iterations,
        "converged": candidate.converged,
        "guarded": candidate.guarded,
        "termination": candidate.termination,
    }


def _qml_record(spec: QmlSpec) -> dict[str, object]:
    observations = tuple(Decimal(value) for value in spec.observations)
    mean = sum(observations, ZERO) / Decimal(len(observations))
    demeaned = tuple(value - mean for value in observations)
    prepared = _prepare(demeaned)
    solved = _solve(prepared, spec.truncation)
    selected = solved.selected
    evaluation = selected.evaluation
    indices = _weight_indices(spec.truncation)
    alternate = _alternate_a_lambda_weights(
        selected.decoded.d,
        selected.decoded.phi,
        selected.decoded.beta,
        spec.truncation,
    )
    alternate_path, alternate_forecast = _path_from_weights(
        prepared,
        selected.decoded.omega,
        selected.decoded.beta,
        alternate,
    )
    return {
        "name": spec.name,
        "purpose": spec.purpose,
        "observations": list(spec.observations),
        "mean": mean,
        "demeaned_residuals": list(demeaned),
        "truncation": spec.truncation,
        "simulation_parameters": {
            "omega": Decimal(spec.simulation_parameters[0]),
            "d": Decimal(spec.simulation_parameters[1]),
            "phi": Decimal(spec.simulation_parameters[2]),
            "beta": Decimal(spec.simulation_parameters[3]),
            "location": Decimal(spec.simulation_parameters[4]),
        },
        "scale": prepared.scale,
        "backcast_normalized": prepared.backcast,
        "fit": _candidate_record(selected, prepared),
        "lambda_checkpoints": _checkpoints(evaluation.lambdas, indices),
        "lambda_sum_retained": sum(evaluation.lambdas, ZERO),
        "normalized_variances": list(evaluation.normalized_variances),
        "physical_variances": list(evaluation.physical_variances),
        "one_step_forecast": {
            "normalized_variance": evaluation.one_step_normalized_variance,
            "physical_variance": evaluation.one_step_physical_variance,
        },
        "alternate_a_path": {
            "lambda_max_absolute_error": max(
                (abs(left - right) for left, right in zip(evaluation.lambdas, alternate)),
                default=ZERO,
            ),
            "variance_path_max_absolute_error": max(
                (
                    abs(left - right)
                    for left, right in zip(
                        evaluation.normalized_variances, alternate_path
                    )
                ),
                default=ZERO,
            ),
            "forecast_absolute_error": abs(
                evaluation.one_step_normalized_variance - alternate_forecast
            ),
        },
        "face_candidates": [
            _candidate_record(candidate, prepared) for candidate in solved.candidates
        ],
        "face_search": {
            "enumerated_face_count": len(FACE_SPECS),
            "grid_evaluations": solved.grid_evaluations,
            "local_refinements_per_face": 1,
            "optimizer": "Decimal analytic-gradient inverse-BFGS with Armijo backtracking",
            "global_optimality_claimed": False,
            "face_contract": "All 16 non-degenerate lower/interior/upper d-phi-beta faces are refined; open-face boundary limits are covered by the corresponding exact adjacent face.",
        },
    }


def _invalid_records() -> list[dict[str, object]]:
    prepared = _prepare(FIXED_SPECS[1].residuals)
    candidates = (
        ("omega_zero", "0", "0.4", "0.2", "0.45", 12),
        ("omega_negative", "-0.1", "0.4", "0.2", "0.45", 12),
        ("d_negative", "0.1", "-0.01", "0", "0", 12),
        ("d_above_one", "0.1", "1.01", "0", "0", 12),
        ("phi_negative", "0.1", "0.4", "-0.01", "0", 12),
        ("phi_above_wedge", "0.1", "0.4", "0.31", "0.2", 12),
        ("beta_negative", "0.1", "0.4", "0.2", "-0.01", 12),
        ("beta_above_d_plus_phi", "0.1", "0.4", "0.2", "0.61", 12),
        ("beta_equal_one", "0.1", "1", "0", "1", 12),
        ("zero_truncation", "0.1", "0.4", "0.2", "0.45", 0),
    )
    records: list[dict[str, object]] = []
    for name, omega, d, phi, beta, truncation in candidates:
        objective = _objective_or_infinity(
            prepared,
            Decimal(omega),
            Decimal(d),
            Decimal(phi),
            Decimal(beta),
            truncation,
        )
        if objective != Decimal("Infinity"):
            raise AssertionError(f"invalid candidate {name} did not hit +infinity")
        records.append(
            {
                "name": name,
                "omega_normalized": Decimal(omega),
                "d": Decimal(d),
                "phi": Decimal(phi),
                "beta": Decimal(beta),
                "truncation": truncation,
                "objective": objective,
            }
        )
    return records


def _gradient_audit() -> Decimal:
    spec = FIXED_SPECS[1]
    prepared = _prepare(spec.residuals)
    omega_physical, d, phi, beta = map(
        Decimal, (spec.omega, spec.d, spec.phi, spec.beta)
    )
    parameters = [omega_physical / prepared.scale_square, d, phi, beta]
    analytic = _evaluate_normalized(
        prepared, *parameters, spec.truncation
    ).gradient
    step = Decimal("1e-14")
    maximum = ZERO
    for index in range(4):
        values: list[Decimal] = []
        for multiplier in (-2, -1, 1, 2):
            trial = list(parameters)
            trial[index] += Decimal(multiplier) * step
            values.append(
                _objective_or_infinity(
                    prepared, *trial, spec.truncation, alternate=True
                )
            )
        minus_two, minus_one, plus_one, plus_two = values
        numeric = (
            minus_two
            - Decimal(8) * minus_one
            + Decimal(8) * plus_one
            - plus_two
        ) / (Decimal(12) * step)
        maximum = max(maximum, abs(numeric - analytic[index]))
    return maximum


def _cross_case_checks(
    fixed_records: Sequence[dict[str, object]],
    qml_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_spec = FIXED_SPECS[1]
    base_prepared = _prepare(base_spec.residuals)
    omega_physical, d, phi, beta = map(
        Decimal, (base_spec.omega, base_spec.d, base_spec.phi, base_spec.beta)
    )
    base = _evaluate_normalized(
        base_prepared,
        omega_physical / base_prepared.scale_square,
        d,
        phi,
        beta,
        base_spec.truncation,
    )
    signed_values = tuple(format(-Decimal(value), "f") for value in base_spec.residuals)
    signed_prepared = _prepare(signed_values)
    signed = _evaluate_normalized(
        signed_prepared,
        omega_physical / signed_prepared.scale_square,
        d,
        phi,
        beta,
        base_spec.truncation,
    )
    sign_error = max(
        (
            abs(left - right)
            for left, right in zip(
                base.normalized_variances, signed.normalized_variances
            )
        ),
        default=ZERO,
    )
    sign_error = max(sign_error, abs(base.objective - signed.objective))

    factor = Decimal("7.25")
    scaled_values = tuple(
        format(Decimal(value) * factor, "f") for value in base_spec.residuals
    )
    scaled_prepared = _prepare(scaled_values)
    scaled = _evaluate_normalized(
        scaled_prepared,
        omega_physical * factor * factor / scaled_prepared.scale_square,
        d,
        phi,
        beta,
        base_spec.truncation,
    )
    normalized_scale_error = max(
        (
            abs(left - right)
            for left, right in zip(
                base.normalized_variances, scaled.normalized_variances
            )
        ),
        default=ZERO,
    )
    physical_scale_error = max(
        (
            abs(right - left * factor * factor)
            for left, right in zip(base.physical_variances, scaled.physical_variances)
        ),
        default=ZERO,
    )
    expected_objective_shift = Decimal(len(base_spec.residuals)) * factor.ln()
    objective_scale_error = abs(
        (scaled.objective - base.objective) - expected_objective_shift
    )

    alternate_errors: list[Decimal] = []
    for record in (*fixed_records, *qml_records):
        block = record["alternate_a_path"]
        if not isinstance(block, dict):
            raise AssertionError("alternate block type")
        alternate_errors.extend(
            block[key]
            for key in (
                "lambda_max_absolute_error",
                "variance_path_max_absolute_error",
                "forecast_absolute_error",
            )
        )
    demeaned_sum_error = max(
        (
            abs(sum(record["demeaned_residuals"], ZERO))
            for record in qml_records
        ),
        default=ZERO,
    )
    hand = fixed_records[0]
    return {
        "face_names": [face.name for face in FACE_SPECS],
        "degenerate_face_rule": "At d=0 and beta=phi every retained lambda is zero, so all such aliases are represented by the constant face d_lower__phi_lower__beta_lower; at d=1 phi=0 is the sole phi face and beta=1 is inadmissible.",
        "hand_expected_lambdas": [
            Decimal("0.375"),
            Decimal("0.15625"),
            Decimal("0.0859375"),
        ],
        "hand_expected_normalized_variances": [
            Decimal(2209) / Decimal(3072),
            Decimal(1777) / Decimal(3072),
        ],
        "hand_expected_one_step_normalized_variance": Decimal(2461) / Decimal(3072),
        "hand_record_name": hand["name"],
        "sign_invariance_max_absolute_error": sign_error,
        "scale_factor": factor,
        "scale_normalized_path_max_absolute_error": normalized_scale_error,
        "scale_physical_path_max_absolute_error": physical_scale_error,
        "scale_objective_shift_absolute_error": objective_scale_error,
        "alternate_a_path_global_max_absolute_error": max(alternate_errors, default=ZERO),
        "analytic_gradient_vs_alternate_five_point_max_absolute_error": _gradient_audit(),
        "qml_demeaned_residual_sum_max_absolute": demeaned_sum_error,
    }


def _build_raw(precision: int) -> dict[str, object]:
    with localcontext() as context:
        context.prec = precision
        context.Emax = 999_999_999
        context.Emin = -999_999_999
        fixed_records = [_fixed_record(spec) for spec in FIXED_SPECS]
        qml_specs = _qml_specs()
        qml_records = [_qml_record(spec) for spec in qml_specs]
        cross = _cross_case_checks(fixed_records, qml_records)
        return {
            "schema_version": 1,
            "oracle": "independent Decimal BBM/arch FIGARCH(1,d,1) power-2 Gaussian QML",
            "provenance": {
                "reference": "Baillie, Bollerslev and Mikkelsen (1996), Journal of Econometrics 74, equations 8-10",
                "reference_url": "https://public.econ.duke.edu/~boller/Published_Papers/joe_96a.pdf",
                "generator": "scripts/figarch_qml_oracle.py",
                "emit_command": "python scripts/figarch_qml_oracle.py emit golden/figarch_qml.json",
                "dependencies": "Python standard library only; decimal.Decimal arithmetic",
            },
            "contract": {
                "input": "Fixed cases are residual-kernel probes. QML cases store observations, subtract their Decimal arithmetic mean exactly once, emit that mean and the demeaned residuals, and then fit only volatility parameters.",
                "power": "2",
                "backcast": "Arithmetic mean of all supplied residual squares, used for every unavailable lag.",
                "weights": "delta_1=d; lambda_1=d+phi-beta; delta_j=((j-1-d)/j)delta_(j-1); lambda_j=beta lambda_(j-1)+delta_j-phi delta_(j-1).",
                "finite_filter": "Retain j=1..K, omit j>K without renormalization; K is explicit and defaults to 1000.",
                "variance": "h_t=omega/(1-beta)+sum_{j=1..K}lambda_j*(observed prior square or BBM backcast).",
                "objective": "0.5*sum_t(log(h_t)+e_t^2/h_t), including t=0 and omitting only the parameter-independent Gaussian constant.",
                "domain": "omega>0; 0<=d<=1; 0<=phi<=(1-d)/2; 0<=beta<=d+phi; beta<1.",
                "forecast": "One-step forecast applies the identical finite-K equation at t=n.",
                "normalization": "Residuals are divided by max(abs(e)); omega and variances are transformed exactly, and n*log(scale) restores the physical-scale objective.",
                "alternate_check": "A separate (1-L)^d polynomial constructs a_j and direct beta convolution; it does not call the primary lambda recurrence.",
                "optimizer": "Every non-degenerate lower/interior/upper coefficient face receives deterministic grid seeding and one Decimal inverse-BFGS refinement; this is exhaustive face topology, not a global-optimality certificate. The oracle searches log(omega); log(omega/(1-beta)) is a bijective coordinate change for beta<1 and has the same physical optimum.",
            },
            "constants": {
                "default_truncation": DEFAULT_TRUNCATION,
                "primary_decimal_digits": PRIMARY_PRECISION,
                "verification_decimal_digits": VERIFICATION_PRECISION,
                "output_significant_digits": OUTPUT_SIGNIFICANT_DIGITS,
                "optimizer_tolerance": OPTIMIZER_TOLERANCE,
                "kkt_tolerance": KKT_TOLERANCE,
                "open_boundary_guard": OPEN_BOUNDARY_GUARD,
                "omega_lower_guard_relative_to_backcast": OMEGA_LOWER_GUARD,
                "face_count": len(FACE_SPECS),
            },
            "fixed_case_count": len(FIXED_SPECS),
            "fixed_cases": fixed_records,
            "invalid_candidate_count": 10,
            "invalid_candidates": _invalid_records(),
            "qml_case_count": len(qml_specs),
            "qml_cases": qml_records,
            "cross_case_checks": cross,
            "precision_recheck": {
                "primary_decimal_digits": PRIMARY_PRECISION,
                "verification_decimal_digits": VERIFICATION_PRECISION,
                "raw_relative_tolerance": RAW_RELATIVE_TOLERANCE,
                "near_zero_absolute_floor": RAW_ABSOLUTE_FLOOR,
                "near_zero_absolute_tolerance": RAW_ABSOLUTE_TOLERANCE,
                "comparison": "Before formatting, exact schemas and list lengths/order are checked; dict keys are traversed sorted; Decimal leaves use relative error away from the near-zero floor and absolute error below it.",
                "max_raw_relative_or_absolute_error": ZERO,
                "formatted_output_equality_required": True,
            },
        }


def _format_decimal(value: Decimal) -> str:
    if not value.is_finite():
        if value.is_nan():
            return "nan"
        return "+inf" if value > ZERO else "-inf"
    # Values below the independently checked raw absolute tolerance are only
    # arithmetic residue (for example, two equivalent lambda constructions).
    if abs(value) < RAW_ABSOLUTE_TOLERANCE:
        return "0"
    return format(value, f".{OUTPUT_SIGNIFICANT_DIGITS}g")


def _format_tree(value: object) -> object:
    if isinstance(value, Decimal):
        return _format_decimal(value)
    if isinstance(value, dict):
        return {key: _format_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_format_tree(item) for item in value]
    return value


def _compare_raw(primary: object, verification: object, path: str = "") -> Decimal:
    """Compare raw trees, with exact structure and sorted mapping traversal."""

    if isinstance(primary, Decimal):
        if not isinstance(verification, Decimal):
            raise AssertionError(f"{path}: Decimal/type mismatch")
        if primary.is_nan() or verification.is_nan():
            if not (primary.is_nan() and verification.is_nan()):
                raise AssertionError(f"{path}: NaN mismatch")
            return ZERO
        if primary.is_infinite() or verification.is_infinite():
            if primary != verification:
                raise AssertionError(f"{path}: infinity mismatch")
            return ZERO
        difference = abs(primary - verification)
        magnitude = max(abs(primary), abs(verification))
        if magnitude <= RAW_ABSOLUTE_FLOOR:
            if difference > RAW_ABSOLUTE_TOLERANCE:
                raise AssertionError(
                    f"{path}: absolute drift {difference} > {RAW_ABSOLUTE_TOLERANCE}"
                )
            return difference
        relative = difference / magnitude
        if relative > RAW_RELATIVE_TOLERANCE:
            raise AssertionError(
                f"{path}: relative drift {relative} > {RAW_RELATIVE_TOLERANCE}"
            )
        return relative
    if isinstance(primary, dict):
        if not isinstance(verification, dict):
            raise AssertionError(f"{path}: mapping/type mismatch")
        primary_keys = sorted(primary)
        verification_keys = sorted(verification)
        if primary_keys != verification_keys:
            raise AssertionError(f"{path}: mapping keys differ")
        maximum = ZERO
        for key in primary_keys:
            maximum = max(
                maximum,
                _compare_raw(primary[key], verification[key], f"{path}/{key}"),
            )
        return maximum
    if isinstance(primary, (tuple, list)):
        if not isinstance(verification, (tuple, list)):
            raise AssertionError(f"{path}: list/type mismatch")
        if len(primary) != len(verification):
            raise AssertionError(f"{path}: list length mismatch")
        maximum = ZERO
        for index, (left, right) in enumerate(zip(primary, verification)):
            maximum = max(
                maximum,
                _compare_raw(left, right, f"{path}/{index}"),
            )
        return maximum
    if type(primary) is not type(verification) or primary != verification:
        raise AssertionError(f"{path}: {primary!r} != {verification!r}")
    return ZERO


def _require_keys(value: object, keys: Iterable[str], path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError(f"{path}: expected mapping")
    expected = set(keys)
    actual = set(value)
    if actual != expected or len(value) != len(expected):
        raise AssertionError(
            f"{path}: keys differ missing={sorted(expected-actual)} extra={sorted(actual-expected)}"
        )
    return value


def _numeric(value: object, path: str, *, allow_infinity: bool = False) -> Decimal:
    if not isinstance(value, str):
        raise AssertionError(f"{path}: expected numeric string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise AssertionError(f"{path}: invalid numeric string") from error
    if parsed.is_nan() or (parsed.is_infinite() and not allow_infinity):
        raise AssertionError(f"{path}: non-finite numeric string")
    return parsed


def _numeric_list(value: object, path: str, length: int) -> list[object]:
    if not isinstance(value, list) or len(value) != length:
        raise AssertionError(f"{path}: expected list length {length}")
    for index, item in enumerate(value):
        _numeric(item, f"{path}/{index}")
    return value


def _validate_parameters(value: object, path: str) -> None:
    block = _require_keys(
        value,
        ("omega_physical", "omega_normalized", "d", "phi", "beta"),
        path,
    )
    for key, item in block.items():
        _numeric(item, f"{path}/{key}")


def _validate_alternate(value: object, path: str) -> None:
    block = _require_keys(
        value,
        (
            "lambda_max_absolute_error",
            "variance_path_max_absolute_error",
            "forecast_absolute_error",
        ),
        path,
    )
    for key, item in block.items():
        _numeric(item, f"{path}/{key}")


def _validate_checkpoints(value: object, path: str, indices: Sequence[int]) -> None:
    if not isinstance(value, list) or len(value) != len(indices):
        raise AssertionError(f"{path}: checkpoint length")
    for position, (item, expected_lag) in enumerate(zip(value, indices)):
        block = _require_keys(item, ("lag", "value"), f"{path}/{position}")
        if block["lag"] != expected_lag or type(block["lag"]) is not int:
            raise AssertionError(f"{path}/{position}/lag")
        _numeric(block["value"], f"{path}/{position}/value")


def _validate_candidate(value: object, path: str, expected_face: str | None = None) -> dict[str, object]:
    keys = (
        "face",
        "d_status",
        "phi_status",
        "beta_status",
        "omega_normalized",
        "omega_physical",
        "d",
        "phi",
        "beta",
        "objective",
        "tangent_gradient_max_absolute",
        "iterations",
        "converged",
        "guarded",
        "termination",
    )
    block = _require_keys(value, keys, path)
    if expected_face is not None and block["face"] != expected_face:
        raise AssertionError(f"{path}/face")
    if block["face"] not in {face.name for face in FACE_SPECS}:
        raise AssertionError(f"{path}/face unknown")
    for key in ("d_status", "phi_status", "beta_status"):
        if block[key] not in ("lower", "interior", "upper"):
            raise AssertionError(f"{path}/{key}")
    for key in (
        "omega_normalized",
        "omega_physical",
        "d",
        "phi",
        "beta",
        "objective",
        "tangent_gradient_max_absolute",
    ):
        _numeric(block[key], f"{path}/{key}")
    if type(block["iterations"]) is not int or block["iterations"] < 0:
        raise AssertionError(f"{path}/iterations")
    if type(block["converged"]) is not bool or type(block["guarded"]) is not bool:
        raise AssertionError(f"{path}: boolean")
    if not isinstance(block["termination"], str):
        raise AssertionError(f"{path}/termination")
    return block


def _validate_schema(document: object) -> dict[str, object]:
    top_keys = (
        "schema_version",
        "oracle",
        "provenance",
        "contract",
        "constants",
        "fixed_case_count",
        "fixed_cases",
        "invalid_candidate_count",
        "invalid_candidates",
        "qml_case_count",
        "qml_cases",
        "cross_case_checks",
        "precision_recheck",
    )
    top = _require_keys(document, top_keys, "")
    if top["schema_version"] != 1 or type(top["schema_version"]) is not int:
        raise AssertionError("/schema_version")
    if not isinstance(top["oracle"], str):
        raise AssertionError("/oracle")
    provenance = _require_keys(
        top["provenance"],
        ("reference", "reference_url", "generator", "emit_command", "dependencies"),
        "/provenance",
    )
    if any(not isinstance(item, str) for item in provenance.values()):
        raise AssertionError("/provenance")
    contract_keys = (
        "input",
        "power",
        "backcast",
        "weights",
        "finite_filter",
        "variance",
        "objective",
        "domain",
        "forecast",
        "normalization",
        "alternate_check",
        "optimizer",
    )
    contract = _require_keys(top["contract"], contract_keys, "/contract")
    if any(not isinstance(item, str) for item in contract.values()):
        raise AssertionError("/contract")
    constants = _require_keys(
        top["constants"],
        (
            "default_truncation",
            "primary_decimal_digits",
            "verification_decimal_digits",
            "output_significant_digits",
            "optimizer_tolerance",
            "kkt_tolerance",
            "open_boundary_guard",
            "omega_lower_guard_relative_to_backcast",
            "face_count",
        ),
        "/constants",
    )
    integer_constants = {
        "default_truncation": DEFAULT_TRUNCATION,
        "primary_decimal_digits": PRIMARY_PRECISION,
        "verification_decimal_digits": VERIFICATION_PRECISION,
        "output_significant_digits": OUTPUT_SIGNIFICANT_DIGITS,
        "face_count": len(FACE_SPECS),
    }
    for key, expected in integer_constants.items():
        if constants[key] != expected or type(constants[key]) is not int:
            raise AssertionError(f"/constants/{key}")
    for key in (
        "optimizer_tolerance",
        "kkt_tolerance",
        "open_boundary_guard",
        "omega_lower_guard_relative_to_backcast",
    ):
        _numeric(constants[key], f"/constants/{key}")

    if top["fixed_case_count"] != len(FIXED_SPECS):
        raise AssertionError("/fixed_case_count")
    fixed_cases = top["fixed_cases"]
    if not isinstance(fixed_cases, list) or len(fixed_cases) != len(FIXED_SPECS):
        raise AssertionError("/fixed_cases")
    fixed_keys = (
        "name",
        "purpose",
        "residuals",
        "truncation",
        "scale",
        "backcast_normalized",
        "parameters",
        "objective",
        "delta_checkpoints",
        "lambda_checkpoints",
        "lambda_sum_retained",
        "normalized_variances",
        "physical_variances",
        "one_step_forecast",
        "alternate_a_path",
    )
    for index, (case, spec) in enumerate(zip(fixed_cases, FIXED_SPECS)):
        path = f"/fixed_cases/{index}"
        block = _require_keys(case, fixed_keys, path)
        if block["name"] != spec.name or block["purpose"] != spec.purpose:
            raise AssertionError(f"{path}: identity")
        if block["residuals"] != list(spec.residuals):
            raise AssertionError(f"{path}/residuals")
        if block["truncation"] != spec.truncation:
            raise AssertionError(f"{path}/truncation")
        for key in ("scale", "backcast_normalized", "objective", "lambda_sum_retained"):
            _numeric(block[key], f"{path}/{key}")
        _validate_parameters(block["parameters"], f"{path}/parameters")
        indices = _weight_indices(spec.truncation)
        _validate_checkpoints(block["delta_checkpoints"], f"{path}/delta_checkpoints", indices)
        _validate_checkpoints(block["lambda_checkpoints"], f"{path}/lambda_checkpoints", indices)
        _numeric_list(block["normalized_variances"], f"{path}/normalized_variances", len(spec.residuals))
        _numeric_list(block["physical_variances"], f"{path}/physical_variances", len(spec.residuals))
        forecast = _require_keys(
            block["one_step_forecast"],
            ("normalized_variance", "physical_variance"),
            f"{path}/one_step_forecast",
        )
        for key, item in forecast.items():
            _numeric(item, f"{path}/one_step_forecast/{key}")
        _validate_alternate(block["alternate_a_path"], f"{path}/alternate_a_path")

    invalid = top["invalid_candidates"]
    if top["invalid_candidate_count"] != 10 or not isinstance(invalid, list) or len(invalid) != 10:
        raise AssertionError("/invalid_candidates")
    invalid_keys = ("name", "omega_normalized", "d", "phi", "beta", "truncation", "objective")
    for index, item in enumerate(invalid):
        path = f"/invalid_candidates/{index}"
        block = _require_keys(item, invalid_keys, path)
        if not isinstance(block["name"], str):
            raise AssertionError(f"{path}/name")
        for key in ("omega_normalized", "d", "phi", "beta"):
            _numeric(block[key], f"{path}/{key}")
        if type(block["truncation"]) is not int:
            raise AssertionError(f"{path}/truncation")
        if _numeric(block["objective"], f"{path}/objective", allow_infinity=True) != Decimal("Infinity"):
            raise AssertionError(f"{path}/objective")

    qml_specs = _qml_specs()
    if top["qml_case_count"] != len(qml_specs):
        raise AssertionError("/qml_case_count")
    qml_cases = top["qml_cases"]
    if not isinstance(qml_cases, list) or len(qml_cases) != len(qml_specs):
        raise AssertionError("/qml_cases")
    qml_keys = (
        "name",
        "purpose",
        "observations",
        "mean",
        "demeaned_residuals",
        "truncation",
        "simulation_parameters",
        "scale",
        "backcast_normalized",
        "fit",
        "lambda_checkpoints",
        "lambda_sum_retained",
        "normalized_variances",
        "physical_variances",
        "one_step_forecast",
        "alternate_a_path",
        "face_candidates",
        "face_search",
    )
    for index, (case, spec) in enumerate(zip(qml_cases, qml_specs)):
        path = f"/qml_cases/{index}"
        block = _require_keys(case, qml_keys, path)
        if block["name"] != spec.name or block["purpose"] != spec.purpose:
            raise AssertionError(f"{path}: identity")
        if block["observations"] != list(spec.observations) or block["truncation"] != spec.truncation:
            raise AssertionError(f"{path}: fixture")
        _numeric(block["mean"], f"{path}/mean")
        _numeric_list(
            block["demeaned_residuals"],
            f"{path}/demeaned_residuals",
            len(spec.observations),
        )
        simulation = _require_keys(
            block["simulation_parameters"],
            ("omega", "d", "phi", "beta", "location"),
            f"{path}/simulation_parameters",
        )
        for key, item in simulation.items():
            _numeric(item, f"{path}/simulation_parameters/{key}")
        for key in ("scale", "backcast_normalized", "lambda_sum_retained"):
            _numeric(block[key], f"{path}/{key}")
        fit = _validate_candidate(block["fit"], f"{path}/fit")
        if fit["converged"] is not True:
            raise AssertionError(f"{path}/fit/converged")
        indices = _weight_indices(spec.truncation)
        _validate_checkpoints(block["lambda_checkpoints"], f"{path}/lambda_checkpoints", indices)
        _numeric_list(block["normalized_variances"], f"{path}/normalized_variances", len(spec.observations))
        _numeric_list(block["physical_variances"], f"{path}/physical_variances", len(spec.observations))
        forecast = _require_keys(
            block["one_step_forecast"],
            ("normalized_variance", "physical_variance"),
            f"{path}/one_step_forecast",
        )
        for key, item in forecast.items():
            _numeric(item, f"{path}/one_step_forecast/{key}")
        _validate_alternate(block["alternate_a_path"], f"{path}/alternate_a_path")
        candidates = block["face_candidates"]
        if not isinstance(candidates, list) or len(candidates) != len(FACE_SPECS):
            raise AssertionError(f"{path}/face_candidates")
        for candidate_index, (candidate, face) in enumerate(zip(candidates, FACE_SPECS)):
            _validate_candidate(candidate, f"{path}/face_candidates/{candidate_index}", face.name)
        search = _require_keys(
            block["face_search"],
            (
                "enumerated_face_count",
                "grid_evaluations",
                "local_refinements_per_face",
                "optimizer",
                "global_optimality_claimed",
                "face_contract",
            ),
            f"{path}/face_search",
        )
        if search["enumerated_face_count"] != len(FACE_SPECS):
            raise AssertionError(f"{path}/face_search/enumerated_face_count")
        if type(search["grid_evaluations"]) is not int or search["grid_evaluations"] <= 0:
            raise AssertionError(f"{path}/face_search/grid_evaluations")
        if search["local_refinements_per_face"] != 1 or search["global_optimality_claimed"] is not False:
            raise AssertionError(f"{path}/face_search semantics")
        if not isinstance(search["optimizer"], str) or not isinstance(search["face_contract"], str):
            raise AssertionError(f"{path}/face_search text")

    cross_keys = (
        "face_names",
        "degenerate_face_rule",
        "hand_expected_lambdas",
        "hand_expected_normalized_variances",
        "hand_expected_one_step_normalized_variance",
        "hand_record_name",
        "sign_invariance_max_absolute_error",
        "scale_factor",
        "scale_normalized_path_max_absolute_error",
        "scale_physical_path_max_absolute_error",
        "scale_objective_shift_absolute_error",
        "alternate_a_path_global_max_absolute_error",
        "analytic_gradient_vs_alternate_five_point_max_absolute_error",
        "qml_demeaned_residual_sum_max_absolute",
    )
    cross = _require_keys(top["cross_case_checks"], cross_keys, "/cross_case_checks")
    if cross["face_names"] != [face.name for face in FACE_SPECS]:
        raise AssertionError("/cross_case_checks/face_names")
    if not isinstance(cross["degenerate_face_rule"], str) or cross["hand_record_name"] != "hand_computable":
        raise AssertionError("/cross_case_checks text")
    _numeric_list(cross["hand_expected_lambdas"], "/cross_case_checks/hand_expected_lambdas", 3)
    _numeric_list(cross["hand_expected_normalized_variances"], "/cross_case_checks/hand_expected_normalized_variances", 2)
    for key in cross_keys[4:5] + cross_keys[6:]:
        if key not in ("hand_record_name", "degenerate_face_rule", "face_names"):
            _numeric(cross[key], f"/cross_case_checks/{key}")

    precision = _require_keys(
        top["precision_recheck"],
        (
            "primary_decimal_digits",
            "verification_decimal_digits",
            "raw_relative_tolerance",
            "near_zero_absolute_floor",
            "near_zero_absolute_tolerance",
            "comparison",
            "max_raw_relative_or_absolute_error",
            "formatted_output_equality_required",
        ),
        "/precision_recheck",
    )
    if precision["primary_decimal_digits"] != PRIMARY_PRECISION or precision["verification_decimal_digits"] != VERIFICATION_PRECISION:
        raise AssertionError("/precision_recheck digits")
    for key in (
        "raw_relative_tolerance",
        "near_zero_absolute_floor",
        "near_zero_absolute_tolerance",
        "max_raw_relative_or_absolute_error",
    ):
        _numeric(precision[key], f"/precision_recheck/{key}")
    if not isinstance(precision["comparison"], str) or precision["formatted_output_equality_required"] is not True:
        raise AssertionError("/precision_recheck semantics")
    return top


def _canonical_bytes(document: dict[str, object]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _first_difference(left: object, right: object, path: str = "") -> str:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left):
            if left[key] != right[key]:
                return _first_difference(left[key], right[key], f"{path}/{key}")
    elif isinstance(left, list) and isinstance(right, list):
        for index, (first, second) in enumerate(zip(left, right)):
            if first != second:
                return _first_difference(first, second, f"{path}/{index}")
    return f"{path}: {left!r} != {right!r}"


def _deep_build() -> tuple[dict[str, object], Decimal]:
    primary = _build_raw(PRIMARY_PRECISION)
    verification = _build_raw(VERIFICATION_PRECISION)
    with localcontext() as context:
        context.prec = 150
        agreement = _compare_raw(primary, verification)
    primary_precision = primary["precision_recheck"]
    verification_precision = verification["precision_recheck"]
    if not isinstance(primary_precision, dict) or not isinstance(verification_precision, dict):
        raise AssertionError("precision block type")
    # Preserve measured (nonzero) audit magnitudes as already-canonical numeric
    # strings.  Other exact-zero checks may still use the global tiny-value
    # formatting rule.
    agreement_text = format(agreement, f".{OUTPUT_SIGNIFICANT_DIGITS}g")
    primary_precision["max_raw_relative_or_absolute_error"] = agreement_text
    verification_precision["max_raw_relative_or_absolute_error"] = agreement_text
    primary_cross = primary["cross_case_checks"]
    verification_cross = verification["cross_case_checks"]
    if not isinstance(primary_cross, dict) or not isinstance(verification_cross, dict):
        raise AssertionError("cross-case block type")
    gradient_measurement = verification_cross[
        "analytic_gradient_vs_alternate_five_point_max_absolute_error"
    ]
    if not isinstance(gradient_measurement, Decimal):
        raise AssertionError("gradient measurement type")
    gradient_text = format(
        gradient_measurement, f".{OUTPUT_SIGNIFICANT_DIGITS}g"
    )
    primary_cross[
        "analytic_gradient_vs_alternate_five_point_max_absolute_error"
    ] = gradient_text
    verification_cross[
        "analytic_gradient_vs_alternate_five_point_max_absolute_error"
    ] = gradient_text
    formatted_primary = _format_tree(primary)
    formatted_verification = _format_tree(verification)
    if formatted_primary != formatted_verification:
        raise AssertionError(
            "80/120 Decimal trees differ after independent formatting at "
            + _first_difference(formatted_primary, formatted_verification)
        )
    if not isinstance(formatted_primary, dict):
        raise AssertionError("formatted root type")
    _validate_schema(formatted_primary)
    return formatted_primary, agreement


def _load(path: Path) -> tuple[dict[str, object], bytes]:
    data = path.read_bytes()
    document = json.loads(data.decode("utf-8"))
    if not isinstance(document, dict):
        raise AssertionError("artifact root must be a mapping")
    return document, data


def emit(path: Path) -> None:
    started = time.perf_counter()
    document, agreement = _deep_build()
    data = _canonical_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(
        f"emitted {path} sha256={_sha256(data)} "
        f"runtime={time.perf_counter()-started:.3f}s raw_max={agreement}"
    )


def fast_check(path: Path) -> None:
    document, data = _load(path)
    _validate_schema(document)
    if data != _canonical_bytes(document):
        raise AssertionError("artifact is not canonical sorted pretty-printed UTF-8 JSON")
    digest = _sha256(data)
    if EXPECTED_ARTIFACT_SHA256 == "TO_BE_REPLACED_AFTER_DEEP_CHECK":
        raise AssertionError(f"artifact hash is not pinned; measured {digest}")
    if digest != EXPECTED_ARTIFACT_SHA256:
        raise AssertionError(
            f"artifact SHA-256 {digest} != pinned {EXPECTED_ARTIFACT_SHA256}"
        )
    print(f"fast-check ok: {path} sha256={digest}")


def deep_check(path: Path) -> None:
    started = time.perf_counter()
    expected, agreement = _deep_build()
    actual, data = _load(path)
    _validate_schema(actual)
    if actual != expected:
        raise AssertionError(
            "artifact differs from regenerated oracle at "
            + _first_difference(actual, expected)
        )
    if data != _canonical_bytes(actual):
        raise AssertionError("artifact bytes are not canonical")
    digest = _sha256(data)
    if (
        EXPECTED_ARTIFACT_SHA256 != "TO_BE_REPLACED_AFTER_DEEP_CHECK"
        and digest != EXPECTED_ARTIFACT_SHA256
    ):
        raise AssertionError(
            f"artifact SHA-256 {digest} != pinned {EXPECTED_ARTIFACT_SHA256}"
        )
    print(
        f"deep-check ok: sha256={digest} "
        f"runtime={time.perf_counter()-started:.3f}s raw_max={agreement}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("emit", "fast-check", "deep-check", "check"))
    parser.add_argument(
        "path", nargs="?", type=Path, default=Path("golden/figarch_qml.json")
    )
    arguments = parser.parse_args()
    if arguments.command == "emit":
        emit(arguments.path)
    elif arguments.command == "fast-check":
        fast_check(arguments.path)
    else:
        deep_check(arguments.path)


if __name__ == "__main__":
    main()
