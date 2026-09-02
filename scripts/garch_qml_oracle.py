#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = []
# ///
"""Generate and verify the GARCH(1,1) Gaussian-QML Tier-0 golden file.

The oracle is deliberately independent of kanimiso's Nelder--Mead solver and
has no third-party dependency.  It uses Decimal arithmetic, an analytic QML
gradient, a deterministic coefficient/omega basin grid followed by BFGS, and
explicit enumeration of the ``alpha = 0`` and ``beta = 0`` faces.

The model contract checked here is:

* demean the observations;
* normalize residuals by their maximum absolute value;
* initialize ``h[0]`` with the full-sample mean normalized squared residual,
  independently of the fitted parameters;
* recurse with
  ``h[t] = omega + alpha * e[t-1]**2 + beta * h[t-1]``;
* require ``omega > 0``, ``alpha >= 0``, ``beta >= 0``, and
  ``alpha + beta < 1``; and
* minimize ``sum((ln(h[t]) + e[t]**2 / h[t]) / 2)``.

The additive Gaussian constant and ``n * ln(scale)`` are omitted because they
do not affect the minimizing parameters.  Decimal evaluates the ordinary
positive recurrence.  The binary64 replay instead evaluates the production
form expected in Rust: residual squares are formed from logarithm differences
and the variance recursion uses log-sum-exp.  ``math.exp`` overflow is an
ordered positive-infinity objective, not an exception or NaN.

Commands::

    python scripts/garch_qml_oracle.py emit golden/garch_qml.json
    python scripts/garch_qml_oracle.py check golden/garch_qml.json
    python scripts/garch_qml_oracle.py deep golden/garch_qml.json

``check`` performs solver-free schema, canonical-byte, and pinned-hash checks.
``deep`` solves every estimable fixture independently at 80 and 120 digits;
``fast-check`` and ``deep-check`` remain descriptive compatibility aliases.
Canonical values are emitted only after the two precision runs agree beyond
the number of digits stored in the JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Iterable, Sequence


ORACLE_PRECISION = 80
VERIFICATION_PRECISION = 120
OUTPUT_SIGNIFICANT_DIGITS = 24
AGREEMENT_SIGNIFICANT_DIGITS = 26
SCHEMA_VERSION = 2
PYTHON_REQUIRES = ">=3.11,<3.13"
DEFAULT_OUTPUT = Path("golden/garch_qml.json")
MAX_BFGS_ITERATIONS = 360
MAX_LINE_SEARCH_STEPS = 120
KKT_TOLERANCE = Decimal("1e-26")
AGREEMENT_ABSOLUTE_TOLERANCE = Decimal("1e-70")
OPEN_BOUNDARY_GUARD = Decimal("1e-18")
# Cross-host maxima measured on CPython 3.12/3.11 were 4.976e-14,
# 3.148e-15, and 3.280e-15 respectively; bounds retain 3--4x margin.
BINARY64_OBJECTIVE_ERROR_BOUND = Decimal("1.99e-13")
BINARY64_LOG_VARIANCE_ERROR_BOUND = Decimal("1.259e-14")
BINARY64_SIGMA2_RELATIVE_ERROR_BOUND = Decimal("1.3e-14")
# Update this only after reviewing the digest printed by ``emit``.
EXPECTED_ARTIFACT_SHA256 = (
    "3f0485c877b2f11bce5690b00a9e9ac17352f1281657e020cb555a7d90ae8e56"
)

ZERO = Decimal(0)
ONE = Decimal(1)
HALF = Decimal("0.5")
TWO = Decimal(2)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    purpose: str
    observations: tuple[str, ...] | None
    expected_selection: str
    compact_input: dict[str, object] | None = None
    probe_parameters: tuple[str, str, str] | None = None


@dataclass(frozen=True)
class PreparedSeries:
    mean: Decimal
    residuals: tuple[Decimal, ...]
    scale: Decimal
    normalized: tuple[Decimal, ...]
    initial_variance: Decimal | None


@dataclass(frozen=True)
class Evaluation:
    objective: Decimal
    gradient: tuple[Decimal, Decimal, Decimal]
    normalized_variances: tuple[Decimal, ...]
    sigma2: tuple[Decimal, ...]


@dataclass(frozen=True)
class Candidate:
    selection: str
    coordinates: tuple[Decimal, ...]
    omega: Decimal
    alpha: Decimal
    beta: Decimal
    evaluation: Evaluation
    transformed_gradient_norm: Decimal
    iterations: int
    converged: bool
    open_boundary_guard_rejected: bool
    open_boundary_escape_detected: bool


@dataclass(frozen=True)
class SolveResult:
    candidate: Candidate
    candidates: tuple[Candidate, ...]
    basin_grid: dict[str, object]


def _fixed_synthetic_observations(
    omega: str,
    alpha: str,
    beta: str,
    innovations: Sequence[str],
    repeats: int,
    initial_multiplier: str = "1",
) -> tuple[str, ...]:
    """Create fixed decimal-string observations outside either oracle context."""
    with localcontext() as ctx:
        ctx.prec = 60
        w = Decimal(omega)
        a = Decimal(alpha)
        b = Decimal(beta)
        unconditional = w / (ONE - a - b)
        variance = unconditional * Decimal(initial_multiplier)
        out: list[str] = []
        for _ in range(repeats):
            for innovation_text in innovations:
                innovation = Decimal(innovation_text)
                residual = variance.sqrt() * innovation
                out.append(format(residual, ".18g"))
                variance = w + a * residual * residual + b * variance
        return tuple(out)


def _deterministic_innovations(count: int, seed: int) -> tuple[str, ...]:
    """Irwin--Hall normal surrogates from an integer-only LCG."""
    modulus = 1 << 32
    state = seed & 0xFFFF_FFFF
    raw: list[Decimal] = []
    with localcontext() as ctx:
        ctx.prec = 60
        denominator = Decimal(modulus)
        for _ in range(count):
            total = ZERO
            for _ in range(12):
                state = (1_664_525 * state + 1_013_904_223) & 0xFFFF_FFFF
                total += Decimal(state) / denominator
            raw.append(total - Decimal(6))
        mean = sum(raw, ZERO) / Decimal(count)
        centered = [value - mean for value in raw]
        variance = sum((value * value for value in centered), ZERO) / Decimal(count)
        scale = variance.sqrt()
        return tuple(format(value / scale, ".18g") for value in centered)


CASES = (
    CaseSpec(
        name="interior",
        purpose="Well-identified interior GARCH optimum with both dynamic coefficients positive.",
        observations=_fixed_synthetic_observations(
            "0.08", "0.12", "0.72", _deterministic_innovations(160, 0x5A17), 1
        ),
        expected_selection="interior",
    ),
    CaseSpec(
        name="alpha_boundary",
        purpose="A deterministic anti-clustered innovation path selects alpha=0 while beta remains on its free face.",
        observations=_deterministic_innovations(96, 4),
        expected_selection="alpha_zero",
    ),
    CaseSpec(
        name="beta_boundary",
        purpose="An ARCH-generated path selects beta=0 with a positive shock coefficient.",
        observations=_fixed_synthetic_observations(
            "0.20", "0.48", "0", _deterministic_innovations(160, 0xBE7A), 1
        ),
        expected_selection="beta_zero",
    ),
    CaseSpec(
        name="near_integrated",
        purpose="A persistent but strictly stationary interior optimum exercises small slack.",
        observations=_fixed_synthetic_observations(
            "0.015", "0.045", "0.94", _deterministic_innovations(240, 0x91A7), 1
        ),
        expected_selection="interior",
    ),
    CaseSpec(
        name="constant_unidentified",
        purpose="Demeaning leaves no scale, so all variance parameters are unidentified.",
        observations=("7", "7", "7", "7", "7", "7", "7", "7"),
        expected_selection="failure",
    ),
    CaseSpec(
        name="long_quiet_then_shock",
        purpose="A valid tiny-omega candidate is finite in Decimal but overflows binary64 QML to +infinity and must remain a dominated extended-real barrier.",
        observations=None,
        compact_input={
            "kind": "prefix_repeat_suffix",
            "prefix": ["1", "-1"],
            "repeat_value": "0",
            "repeat_count": 1198,
            "suffix": ["1", "-1"],
        },
        expected_selection="probe",
        probe_parameters=("1e-320", "0.25", "0.50"),
    ),
)


def _expand_observations(spec: CaseSpec) -> tuple[str, ...]:
    if spec.observations is not None:
        return spec.observations
    if spec.compact_input is None:
        raise ValueError(f"{spec.name}: missing observations")
    payload = spec.compact_input
    if payload.get("kind") != "prefix_repeat_suffix":
        raise ValueError(f"{spec.name}: unsupported compact input")
    prefix = tuple(str(value) for value in payload["prefix"])
    repeated = (str(payload["repeat_value"]),) * int(payload["repeat_count"])
    suffix = tuple(str(value) for value in payload["suffix"])
    return prefix + repeated + suffix


def _prepare(spec: CaseSpec) -> PreparedSeries:
    observations = tuple(Decimal(value) for value in _expand_observations(spec))
    count = Decimal(len(observations))
    mean = sum(observations, ZERO) / count
    residuals = tuple(value - mean for value in observations)
    scale = max(abs(value) for value in residuals)
    if scale == 0:
        return PreparedSeries(mean, residuals, scale, (), None)
    normalized = tuple(value / scale for value in residuals)
    initial = sum((value * value for value in normalized), ZERO) / count
    if initial <= 0:
        raise ArithmeticError(f"{spec.name}: non-positive initial variance")
    return PreparedSeries(mean, residuals, scale, normalized, initial)


def _evaluate(
    prepared: PreparedSeries,
    omega: Decimal,
    alpha: Decimal,
    beta: Decimal,
) -> Evaluation:
    if prepared.initial_variance is None:
        raise ValueError("constant series has no identified GARCH objective")
    if not omega.is_finite() or omega <= 0:
        raise ValueError("omega must be finite and positive")
    if not alpha.is_finite() or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if not beta.is_finite() or beta < 0 or alpha + beta >= 1:
        raise ValueError("beta must be non-negative and persistence below one")

    variance = prepared.initial_variance
    derivative = [ZERO, ZERO, ZERO]
    objective = ZERO
    gradient = [ZERO, ZERO, ZERO]
    variances: list[Decimal] = []

    for time, residual in enumerate(prepared.normalized):
        if variance <= 0 or not variance.is_finite():
            raise ArithmeticError("GARCH variance became non-positive or non-finite")
        residual_square = residual * residual
        variances.append(variance)
        objective += HALF * (variance.ln() + residual_square / variance)
        score_factor = HALF * (ONE / variance - residual_square / (variance * variance))
        for index in range(3):
            gradient[index] += score_factor * derivative[index]

        if time + 1 < len(prepared.normalized):
            previous_variance = variance
            previous_derivative = derivative
            variance = omega + alpha * residual_square + beta * previous_variance
            derivative = [
                ONE + beta * previous_derivative[0],
                residual_square + beta * previous_derivative[1],
                previous_variance + beta * previous_derivative[2],
            ]

    scale_square = prepared.scale * prepared.scale
    return Evaluation(
        objective=objective,
        gradient=(gradient[0], gradient[1], gradient[2]),
        normalized_variances=tuple(variances),
        sigma2=tuple(scale_square * value for value in variances),
    )


def _objective_only(
    prepared: PreparedSeries,
    omega: Decimal,
    alpha: Decimal,
    beta: Decimal,
) -> Decimal:
    """Evaluate the same QML recurrence without allocating gradient diagnostics."""
    if prepared.initial_variance is None:
        raise ValueError("constant series has no identified GARCH objective")
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        raise ValueError("parameters violate the strict stationary GARCH domain")
    variance = prepared.initial_variance
    objective = ZERO
    for time, residual in enumerate(prepared.normalized):
        residual_square = residual * residual
        objective += HALF * (variance.ln() + residual_square / variance)
        if time + 1 < len(prepared.normalized):
            variance = omega + alpha * residual_square + beta * variance
    return objective


def _sigmoid(value: Decimal) -> Decimal:
    if value >= 0:
        tail = (-value).exp()
        return ONE / (ONE + tail)
    head = value.exp()
    return head / (ONE + head)


def _logit(value: Decimal) -> Decimal:
    if not (ZERO < value < ONE):
        raise ValueError("logit input must be in (0, 1)")
    return value.ln() - (ONE - value).ln()


def _decode(
    selection: str, coordinates: Sequence[Decimal]
) -> tuple[Decimal, Decimal, Decimal]:
    omega = coordinates[0].exp()
    if selection == "interior":
        persistence = _sigmoid(coordinates[1])
        share = _sigmoid(coordinates[2])
        alpha = persistence * share
        beta = persistence * (ONE - share)
    elif selection == "alpha_zero":
        alpha = ZERO
        beta = _sigmoid(coordinates[1])
    elif selection == "beta_zero":
        alpha = _sigmoid(coordinates[1])
        beta = ZERO
    elif selection == "corner":
        alpha = ZERO
        beta = ZERO
    else:
        raise ValueError(f"unknown selection {selection}")
    return omega, alpha, beta


def _transformed_evaluation(
    prepared: PreparedSeries, selection: str, coordinates: Sequence[Decimal]
) -> tuple[Evaluation, tuple[Decimal, ...]]:
    omega, alpha, beta = _decode(selection, coordinates)
    evaluation = _evaluate(prepared, omega, alpha, beta)
    g_omega, g_alpha, g_beta = evaluation.gradient
    if selection == "interior":
        persistence = alpha + beta
        share = alpha / persistence
        gradient = (
            g_omega * omega,
            persistence
            * (ONE - persistence)
            * (g_alpha * share + g_beta * (ONE - share)),
            persistence * share * (ONE - share) * (g_alpha - g_beta),
        )
    elif selection == "alpha_zero":
        gradient = (g_omega * omega, g_beta * beta * (ONE - beta))
    elif selection == "beta_zero":
        gradient = (g_omega * omega, g_alpha * alpha * (ONE - alpha))
    else:
        gradient = (g_omega * omega,)
    return evaluation, gradient


def _dot(left: Sequence[Decimal], right: Sequence[Decimal]) -> Decimal:
    return sum((a * b for a, b in zip(left, right)), ZERO)


def _mat_vec(matrix: Sequence[Sequence[Decimal]], vector: Sequence[Decimal]) -> list[Decimal]:
    return [_dot(row, vector) for row in matrix]


def _identity(size: int) -> list[list[Decimal]]:
    return [[ONE if row == column else ZERO for column in range(size)] for row in range(size)]


def _inverse_bfgs_update(
    inverse_hessian: Sequence[Sequence[Decimal]],
    step: Sequence[Decimal],
    gradient_delta: Sequence[Decimal],
) -> list[list[Decimal]]:
    size = len(step)
    curvature = _dot(gradient_delta, step)
    scale = max(ONE, max(abs(value) for value in step), max(abs(value) for value in gradient_delta))
    if curvature <= Decimal("1e-70") * scale * scale:
        return _identity(size)
    rho = ONE / curvature
    left = [
        [
            (ONE if row == column else ZERO) - rho * step[row] * gradient_delta[column]
            for column in range(size)
        ]
        for row in range(size)
    ]
    right = [
        [
            (ONE if row == column else ZERO) - rho * gradient_delta[row] * step[column]
            for column in range(size)
        ]
        for row in range(size)
    ]
    middle = [
        [sum((left[row][k] * inverse_hessian[k][column] for k in range(size)), ZERO)
         for column in range(size)]
        for row in range(size)
    ]
    updated = [
        [sum((middle[row][k] * right[k][column] for k in range(size)), ZERO)
         + rho * step[row] * step[column]
         for column in range(size)]
        for row in range(size)
    ]
    return updated


def _bfgs(
    prepared: PreparedSeries,
    selection: str,
    initial: Sequence[Decimal],
    gradient_tolerance: Decimal,
) -> Candidate:
    coordinates = list(initial)
    evaluation, gradient_tuple = _transformed_evaluation(prepared, selection, coordinates)
    gradient = list(gradient_tuple)
    inverse_hessian = _identity(len(coordinates))
    converged = False
    iterations = 0

    for iteration in range(MAX_BFGS_ITERATIONS):
        iterations = iteration
        omega, alpha, beta = _decode(selection, coordinates)
        initial_variance = prepared.initial_variance
        if initial_variance is None:
            raise ValueError("constant series")
        near_open_boundary = omega / initial_variance <= OPEN_BOUNDARY_GUARD
        if selection == "interior":
            near_open_boundary = near_open_boundary or (
                min(alpha, beta, ONE - alpha - beta) <= OPEN_BOUNDARY_GUARD
            )
        elif selection == "alpha_zero":
            near_open_boundary = near_open_boundary or (
                min(beta, ONE - beta) <= OPEN_BOUNDARY_GUARD
            )
        elif selection == "beta_zero":
            near_open_boundary = near_open_boundary or (
                min(alpha, ONE - alpha) <= OPEN_BOUNDARY_GUARD
            )
        if near_open_boundary:
            break
        gradient_norm = max(abs(value) for value in gradient)
        if gradient_norm <= gradient_tolerance:
            converged = True
            break

        direction = [-value for value in _mat_vec(inverse_hessian, gradient)]
        directional_derivative = _dot(gradient, direction)
        if directional_derivative >= 0:
            direction = [-value for value in gradient]
            directional_derivative = -_dot(gradient, gradient)
            inverse_hessian = _identity(len(coordinates))

        step_size = ONE
        accepted = False
        candidate_evaluation = evaluation
        candidate_gradient: tuple[Decimal, ...] = tuple(gradient)
        candidate_coordinates = coordinates
        armijo = Decimal("1e-4")
        for _ in range(MAX_LINE_SEARCH_STEPS):
            trial = [
                coordinate + step_size * delta
                for coordinate, delta in zip(coordinates, direction)
            ]
            try:
                trial_evaluation, trial_gradient = _transformed_evaluation(
                    prepared, selection, trial
                )
            except (ArithmeticError, ValueError):
                step_size *= HALF
                continue
            if trial_evaluation.objective <= (
                evaluation.objective + armijo * step_size * directional_derivative
            ):
                accepted = True
                candidate_evaluation = trial_evaluation
                candidate_gradient = trial_gradient
                candidate_coordinates = trial
                break
            step_size *= HALF

        if not accepted:
            break

        actual_step = [
            new - old for new, old in zip(candidate_coordinates, coordinates)
        ]
        gradient_delta = [
            new - old for new, old in zip(candidate_gradient, gradient)
        ]
        inverse_hessian = _inverse_bfgs_update(
            inverse_hessian, actual_step, gradient_delta
        )
        coordinates = list(candidate_coordinates)
        evaluation = candidate_evaluation
        gradient = list(candidate_gradient)
    else:
        iterations = MAX_BFGS_ITERATIONS

    gradient_norm = max(abs(value) for value in gradient)
    omega, alpha, beta = _decode(selection, coordinates)
    initial_variance = prepared.initial_variance
    if initial_variance is None:
        raise ValueError("constant series")
    attained = omega / initial_variance > OPEN_BOUNDARY_GUARD
    if selection == "interior":
        attained = attained and min(alpha, beta, ONE - alpha - beta) > OPEN_BOUNDARY_GUARD
    elif selection == "alpha_zero":
        attained = attained and min(beta, ONE - beta) > OPEN_BOUNDARY_GUARD
    elif selection == "beta_zero":
        attained = attained and min(alpha, ONE - alpha) > OPEN_BOUNDARY_GUARD
    open_boundary_escape = _objective_only(
        prepared, omega * HALF, alpha, beta
    ) < evaluation.objective - KKT_TOLERANCE
    if selection == "interior":
        persistence = alpha + beta
        share = alpha / persistence
        probe_persistence = (ONE + persistence) * HALF
        probe_alpha = probe_persistence * share
        probe_beta = probe_persistence * (ONE - share)
        open_boundary_escape = open_boundary_escape or (
            _objective_only(prepared, omega, probe_alpha, probe_beta)
            < evaluation.objective - KKT_TOLERANCE
        )
    elif selection == "alpha_zero":
        open_boundary_escape = open_boundary_escape or (
            _objective_only(prepared, omega, ZERO, (ONE + beta) * HALF)
            < evaluation.objective - KKT_TOLERANCE
        )
    elif selection == "beta_zero":
        open_boundary_escape = open_boundary_escape or (
            _objective_only(prepared, omega, (ONE + alpha) * HALF, ZERO)
            < evaluation.objective - KKT_TOLERANCE
        )
    return Candidate(
        selection=selection,
        coordinates=tuple(coordinates),
        omega=omega,
        alpha=alpha,
        beta=beta,
        evaluation=evaluation,
        transformed_gradient_norm=gradient_norm,
        iterations=iterations,
        converged=converged and attained and not open_boundary_escape,
        open_boundary_guard_rejected=not attained,
        open_boundary_escape_detected=open_boundary_escape,
    )


def _initial_coordinates(
    prepared: PreparedSeries,
    selection: str,
    alpha: Decimal,
    beta: Decimal,
    omega_multiplier: Decimal,
) -> tuple[Decimal, ...]:
    if prepared.initial_variance is None:
        raise ValueError("constant series")
    slack = ONE - alpha - beta
    omega = prepared.initial_variance * slack * omega_multiplier
    if omega <= 0:
        raise ValueError("non-positive seed omega")
    if selection == "interior":
        persistence = alpha + beta
        share = alpha / persistence
        return (omega.ln(), _logit(persistence), _logit(share))
    if selection == "alpha_zero":
        return (omega.ln(), _logit(beta))
    if selection == "beta_zero":
        return (omega.ln(), _logit(alpha))
    return (omega.ln(),)


def _full_kkt_satisfied(candidate: Candidate) -> bool:
    if not candidate.converged or candidate.transformed_gradient_norm > KKT_TOLERANCE:
        return False
    _, g_alpha, g_beta = candidate.evaluation.gradient
    if candidate.selection == "alpha_zero":
        return g_alpha >= -KKT_TOLERANCE
    if candidate.selection == "beta_zero":
        return g_beta >= -KKT_TOLERANCE
    if candidate.selection == "corner":
        return g_alpha >= -KKT_TOLERANCE and g_beta >= -KKT_TOLERANCE
    return True


def _solve(prepared: PreparedSeries, precision: int) -> SolveResult:
    del precision
    if prepared.initial_variance is None:
        raise ValueError("constant series")
    persistence_grid = (
        Decimal("0.05"),
        Decimal("0.25"),
        Decimal("0.50"),
        Decimal("0.75"),
        Decimal("0.90"),
        Decimal("0.97"),
        Decimal("0.99"),
    )
    share_grid = (
        Decimal("0.10"),
        Decimal("0.50"),
        Decimal("0.90"),
    )
    omega_multipliers = (
        Decimal("0.10"),
        Decimal("0.50"),
        ONE,
        Decimal("2"),
        Decimal("10"),
    )
    face_seeds: dict[str, tuple[Decimal, Decimal, Decimal] | None] = {
        "interior": None,
        "alpha_zero": None,
        "beta_zero": None,
        "corner": None,
    }
    face_grid_objectives: dict[str, Decimal] = {
        selection: Decimal("Infinity") for selection in face_seeds
    }
    grid_evaluations = 0

    coefficient_grid: list[tuple[str, Decimal, Decimal]] = [("corner", ZERO, ZERO)]
    for persistence in persistence_grid:
        coefficient_grid.append(("alpha_zero", ZERO, persistence))
        coefficient_grid.append(("beta_zero", persistence, ZERO))
        for share in share_grid:
            alpha = persistence * share
            coefficient_grid.append(("interior", alpha, persistence - alpha))

    for selection, alpha, beta in coefficient_grid:
        slack = ONE - alpha - beta
        for multiplier in omega_multipliers:
            omega = prepared.initial_variance * slack * multiplier
            objective = _objective_only(prepared, omega, alpha, beta)
            grid_evaluations += 1
            if objective < face_grid_objectives[selection]:
                face_grid_objectives[selection] = objective
                face_seeds[selection] = (alpha, beta, multiplier)

    candidates: list[Candidate] = []
    for selection, seed in face_seeds.items():
        if seed is None:
            raise ArithmeticError(f"no {selection} basin-grid seed")
        alpha, beta, multiplier = seed
        initial = _initial_coordinates(prepared, selection, alpha, beta, multiplier)
        candidates.append(
            _bfgs(prepared, selection, initial, KKT_TOLERANCE)
        )

    eligible = [candidate for candidate in candidates if _full_kkt_satisfied(candidate)]
    if not eligible:
        raise ArithmeticError("no attained face KKT point")
    objective_scale = max(
        ONE, min(candidate.evaluation.objective.copy_abs() for candidate in eligible)
    )
    tie_tolerance = Decimal(10) ** Decimal(-(OUTPUT_SIGNIFICANT_DIGITS - 4)) * objective_scale
    minimum = min(candidate.evaluation.objective for candidate in eligible)
    equivalent = [
        candidate
        for candidate in eligible
        if candidate.evaluation.objective - minimum <= tie_tolerance
    ]
    complexity = {"corner": 0, "alpha_zero": 1, "beta_zero": 1, "interior": 2}
    selected = min(
        equivalent,
        key=lambda candidate: (complexity[candidate.selection], candidate.selection),
    )
    grid_minimum_selection = min(face_grid_objectives, key=face_grid_objectives.get)
    if selected.evaluation.objective > face_grid_objectives[grid_minimum_selection] + tie_tolerance:
        raise AssertionError("refined optimum is worse than the deterministic basin grid")
    basin_grid = {
        "evaluations": grid_evaluations,
        "persistence_values": list(persistence_grid),
        "share_values": list(share_grid),
        "omega_multipliers": list(omega_multipliers),
        "best_face": grid_minimum_selection,
        "best_objective": face_grid_objectives[grid_minimum_selection],
        "face_best_objectives": dict(face_grid_objectives),
    }
    return SolveResult(selected, tuple(candidates), basin_grid)


def _safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _float_logsumexp(values: Iterable[float]) -> float:
    materialized = tuple(values)
    maximum = max(materialized, default=-math.inf)
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in materialized))


def _float_welford_mean(values: Sequence[float]) -> float:
    mean = 0.0
    for count, value in enumerate(values, 1):
        mean += (value - mean) / count
    return mean


def _binary64_replay(
    spec: CaseSpec,
    omega_text: str,
    alpha_text: str,
    beta_text: str,
) -> dict[str, object]:
    observations = [float(value) for value in _expand_observations(spec)]
    mean = _float_welford_mean(observations)
    residuals = [value - mean for value in observations]
    scale = max(abs(value) for value in residuals)
    omega = float(omega_text)
    alpha = float(alpha_text)
    beta = float(beta_text)
    if scale == 0.0:
        return {"outcome": "unidentified"}

    log_scale = math.log(scale)
    log_squares = [
        -math.inf if value == 0.0 else 2.0 * (math.log(abs(value)) - log_scale)
        for value in residuals
    ]
    log_initial = _float_logsumexp(log_squares) - math.log(len(log_squares))
    log_omega = math.log(omega) if omega > 0.0 else -math.inf
    log_alpha = math.log(alpha) if alpha > 0.0 else -math.inf
    log_beta = math.log(beta) if beta > 0.0 else -math.inf
    log_variances = [log_initial]
    for time in range(1, len(residuals)):
        log_variances.append(
            _float_logsumexp(
                (
                    log_omega,
                    log_alpha + log_squares[time - 1],
                    log_beta + log_variances[time - 1],
                )
            )
        )

    objective = 0.0
    first_infinite_standardized_square: int | None = None
    infinite_standardized_squares = 0
    for time, (log_square, log_variance) in enumerate(zip(log_squares, log_variances)):
        standardized_square = (
            0.0 if log_square == -math.inf else _safe_exp(log_square - log_variance)
        )
        if standardized_square == math.inf:
            infinite_standardized_squares += 1
            if first_infinite_standardized_square is None:
                first_infinite_standardized_square = time
        objective += 0.5 * (log_variance + standardized_square)
    log_scale_square = 2.0 * log_scale
    sigma2 = [_safe_exp(log_scale_square + value) for value in log_variances]
    return {
        "outcome": "positive_infinity" if objective == math.inf else "finite",
        "first_infinite_standardized_square": first_infinite_standardized_square,
        "infinite_standardized_squares": infinite_standardized_squares,
        "_objective_float": objective,
        "_log_variances_float": log_variances,
        "_sigma2_float": sigma2,
    }


def _decimal_text(value: Decimal) -> str:
    if value.is_infinite():
        return "-Infinity" if value.is_signed() else "Infinity"
    if value.is_nan():
        return "NaN"
    return format(value, f".{OUTPUT_SIGNIFICANT_DIGITS}g")


def _float_error(actual: float, expected: Decimal) -> Decimal:
    if actual == math.inf:
        return Decimal("Infinity")
    return abs(Decimal.from_float(actual) - expected)


def _relative_error(actual: float, expected: Decimal) -> Decimal:
    absolute = _float_error(actual, expected)
    if absolute.is_infinite():
        return absolute
    return absolute if expected == 0 else absolute / abs(expected)


def _candidate_summary(candidate: Candidate) -> dict[str, object]:
    gradient = candidate.evaluation.gradient
    return {
        "selection": candidate.selection,
        "objective": candidate.evaluation.objective,
        "omega_normalized": candidate.omega,
        "alpha": candidate.alpha,
        "beta": candidate.beta,
        "stationarity_slack": ONE - candidate.alpha - candidate.beta,
        "physical_gradient": list(gradient),
        "transformed_gradient_norm": candidate.transformed_gradient_norm,
        "iterations": candidate.iterations,
        "converged": candidate.converged,
        "open_boundary_guard_rejected": candidate.open_boundary_guard_rejected,
        "open_boundary_escape_detected": candidate.open_boundary_escape_detected,
        "full_kkt_satisfied": _full_kkt_satisfied(candidate),
    }


def _empty_binary64_measurements() -> dict[str, Decimal]:
    return {
        "objective_abs_error": ZERO,
        "log_variance_abs_error": ZERO,
        "sigma2_relative_error": ZERO,
    }


def _case_payload(
    spec: CaseSpec, precision: int
) -> tuple[dict[str, object], dict[str, Decimal]]:
    """Build an unformatted Decimal payload and non-canonical host measurements."""
    prepared = _prepare(spec)
    input_payload: dict[str, object]
    if spec.compact_input is not None:
        input_payload = dict(spec.compact_input)
    else:
        input_payload = {"kind": "literal", "observations": list(spec.observations or ())}

    payload: dict[str, object] = {
        "name": spec.name,
        "purpose": spec.purpose,
        "input": input_payload,
    }
    if spec.expected_selection == "failure":
        if prepared.initial_variance is not None:
            raise AssertionError(f"{spec.name}: expected an unidentified constant series")
        payload["expected"] = {
            "outcome": "failure",
            "issue_code": "unidentified_model",
            "mean": prepared.mean,
            "scale": prepared.scale,
        }
        payload["binary64_replay"] = {"outcome": "unidentified"}
        return payload, _empty_binary64_measurements()

    if spec.expected_selection == "probe":
        if spec.probe_parameters is None:
            raise AssertionError(f"{spec.name}: probe parameters missing")
        omega, alpha, beta = (Decimal(value) for value in spec.probe_parameters)
        evaluation = _evaluate(prepared, omega, alpha, beta)
        replay = _binary64_replay(
            spec, spec.probe_parameters[0], spec.probe_parameters[1], spec.probe_parameters[2]
        )
        if replay["outcome"] != "positive_infinity":
            raise AssertionError(f"{spec.name}: binary64 probe did not overflow to +infinity")
        stable_replay = {
            key: value for key, value in replay.items() if not key.startswith("_")
        }
        payload["expected"] = {
            "outcome": "extended_real_probe",
            "mean": prepared.mean,
            "scale": prepared.scale,
            "initial_normalized_variance": prepared.initial_variance,
            "parameters": {
                "omega_normalized": omega,
                "alpha": alpha,
                "beta": beta,
            },
            "decimal_objective": evaluation.objective,
            "decimal_objective_is_finite": evaluation.objective.is_finite(),
            "binary64_objective": "Infinity",
            "interpretation": "valid candidate is a dominated positive-infinity barrier in binary64",
        }
        payload["binary64_replay"] = stable_replay
        return payload, _empty_binary64_measurements()

    solved = _solve(prepared, precision)
    selected = solved.candidate
    if selected.selection != spec.expected_selection:
        summaries = [
            (candidate.selection, str(candidate.evaluation.objective))
            for candidate in solved.candidates
        ]
        raise AssertionError(
            f"{spec.name}: expected {spec.expected_selection}, selected {selected.selection}; {summaries}"
        )
    if not _full_kkt_satisfied(selected):
        raise AssertionError(
            f"{spec.name}: selected candidate lacks an attained full KKT point"
        )
    if not (
        selected.omega > 0
        and selected.alpha >= 0
        and selected.beta >= 0
        and selected.alpha + selected.beta < 1
    ):
        raise AssertionError(f"{spec.name}: selected parameters violate the model domain")
    evaluation = selected.evaluation
    replay = _binary64_replay(
        spec,
        str(selected.omega),
        str(selected.alpha),
        str(selected.beta),
    )
    if replay["outcome"] != "finite":
        raise AssertionError(f"{spec.name}: selected optimum is not finite in binary64")

    max_log_variance_error = ZERO
    max_sigma2_relative_error = ZERO
    replay_log_variances = replay.pop("_log_variances_float")
    replay_sigma2 = replay.pop("_sigma2_float")
    replay_objective = replay.pop("_objective_float")
    if not isinstance(replay_log_variances, list) or not isinstance(replay_sigma2, list):
        raise AssertionError(f"{spec.name}: invalid private binary64 replay vectors")
    if not isinstance(replay_objective, float):
        raise AssertionError(f"{spec.name}: invalid private binary64 replay objective")
    if len(replay_log_variances) != len(evaluation.normalized_variances):
        raise AssertionError(f"{spec.name}: binary64 log-variance length mismatch")
    if len(replay_sigma2) != len(evaluation.sigma2):
        raise AssertionError(f"{spec.name}: binary64 sigma2 length mismatch")
    for actual, expected in zip(replay_log_variances, evaluation.normalized_variances):
        max_log_variance_error = max(
            max_log_variance_error, _float_error(actual, expected.ln())
        )
    for actual, expected in zip(replay_sigma2, evaluation.sigma2):
        max_sigma2_relative_error = max(
            max_sigma2_relative_error, _relative_error(actual, expected)
        )
    objective_error = _float_error(replay_objective, evaluation.objective)
    if objective_error > BINARY64_OBJECTIVE_ERROR_BOUND:
        raise AssertionError(
            f"{spec.name}: binary64 objective error {objective_error} exceeds "
            f"{BINARY64_OBJECTIVE_ERROR_BOUND}"
        )
    if max_log_variance_error > BINARY64_LOG_VARIANCE_ERROR_BOUND:
        raise AssertionError(
            f"{spec.name}: binary64 log-variance error {max_log_variance_error} exceeds "
            f"{BINARY64_LOG_VARIANCE_ERROR_BOUND}"
        )
    if max_sigma2_relative_error > BINARY64_SIGMA2_RELATIVE_ERROR_BOUND:
        raise AssertionError(
            f"{spec.name}: binary64 sigma2 error {max_sigma2_relative_error} exceeds "
            f"{BINARY64_SIGMA2_RELATIVE_ERROR_BOUND}"
        )

    g_omega, g_alpha, g_beta = evaluation.gradient
    if selected.selection == "interior":
        kkt = {
            "kind": "interior_stationarity",
            "tolerance": KKT_TOLERANCE,
            "free_transformed_gradient_norm": selected.transformed_gradient_norm,
            "satisfied": selected.transformed_gradient_norm <= KKT_TOLERANCE,
        }
    elif selected.selection == "alpha_zero":
        if g_alpha < -KKT_TOLERANCE:
            raise AssertionError(f"{spec.name}: alpha lower-bound KKT condition failed")
        kkt = {
            "kind": "alpha_lower_bound",
            "tolerance": KKT_TOLERANCE,
            "free_transformed_gradient_norm": selected.transformed_gradient_norm,
            "alpha_one_sided_derivative": g_alpha,
            "free_stationarity_satisfied": (
                selected.transformed_gradient_norm <= KKT_TOLERANCE
            ),
            "alpha_kkt_satisfied": g_alpha >= -KKT_TOLERANCE,
        }
    elif selected.selection == "beta_zero":
        if g_beta < -KKT_TOLERANCE:
            raise AssertionError(f"{spec.name}: beta lower-bound KKT condition failed")
        kkt = {
            "kind": "beta_lower_bound",
            "tolerance": KKT_TOLERANCE,
            "free_transformed_gradient_norm": selected.transformed_gradient_norm,
            "beta_one_sided_derivative": g_beta,
            "free_stationarity_satisfied": (
                selected.transformed_gradient_norm <= KKT_TOLERANCE
            ),
            "beta_kkt_satisfied": g_beta >= -KKT_TOLERANCE,
        }
    else:
        if g_alpha < -KKT_TOLERANCE or g_beta < -KKT_TOLERANCE:
            raise AssertionError(f"{spec.name}: corner KKT condition failed")
        kkt = {
            "kind": "alpha_beta_corner",
            "tolerance": KKT_TOLERANCE,
            "free_transformed_gradient_norm": selected.transformed_gradient_norm,
            "alpha_one_sided_derivative": g_alpha,
            "beta_one_sided_derivative": g_beta,
            "free_stationarity_satisfied": (
                selected.transformed_gradient_norm <= KKT_TOLERANCE
            ),
            "alpha_kkt_satisfied": g_alpha >= -KKT_TOLERANCE,
            "beta_kkt_satisfied": g_beta >= -KKT_TOLERANCE,
        }

    payload["expected"] = {
        "outcome": "success",
        "selection": selected.selection,
        "mean": prepared.mean,
        "scale": prepared.scale,
        "initial_normalized_variance": prepared.initial_variance,
        "omega_normalized": selected.omega,
        "omega_physical": selected.omega * prepared.scale * prepared.scale,
        "alpha": selected.alpha,
        "beta": selected.beta,
        "stationarity_slack": ONE - selected.alpha - selected.beta,
        "objective_without_constants": evaluation.objective,
        "physical_gradient": list(evaluation.gradient),
        "kkt": kkt,
        "normalized_variances": list(evaluation.normalized_variances),
        "sigma2": list(evaluation.sigma2),
        "face_candidates": [
            _candidate_summary(candidate) for candidate in solved.candidates
        ],
        "basin_grid": solved.basin_grid,
    }
    replay["objective_abs_error_certified_below"] = BINARY64_OBJECTIVE_ERROR_BOUND
    replay["max_log_variance_abs_error_certified_below"] = (
        BINARY64_LOG_VARIANCE_ERROR_BOUND
    )
    replay["max_sigma2_relative_error_certified_below"] = (
        BINARY64_SIGMA2_RELATIVE_ERROR_BOUND
    )
    payload["binary64_replay"] = replay
    return payload, {
        "objective_abs_error": objective_error,
        "log_variance_abs_error": max_log_variance_error,
        "sigma2_relative_error": max_sigma2_relative_error,
    }


def _build_payload(
    precision: int,
) -> tuple[dict[str, object], dict[str, Decimal]]:
    with localcontext() as ctx:
        ctx.prec = precision
        built = [_case_payload(spec, precision) for spec in CASES]
    cases = [case for case, _ in built]
    maxima = _empty_binary64_measurements()
    for _, measurements in built:
        for key, value in measurements.items():
            maxima[key] = max(maxima[key], value)
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle": {
            "name": "independent_decimal_garch11_gaussian_qml",
            "implementation": "Python standard library Decimal analytic-gradient face BFGS after deterministic coefficient/omega basin grid",
            "python_requires": PYTHON_REQUIRES,
            "precision_digits": ORACLE_PRECISION,
            "verification_precision_digits": VERIFICATION_PRECISION,
            "normalization": "demean, divide residuals by max absolute residual, form binary64 squares from logarithm differences",
            "initial_variance": "full-sample mean normalized squared residual, independent of parameters",
            "recurrence": "h[t] = omega + alpha * e[t-1]^2 + beta * h[t-1]",
            "constraints": "omega > 0; alpha >= 0; beta >= 0; alpha + beta < 1",
            "objective": "0.5 * sum(log(h[t]) + e[t]^2 / h[t]); Gaussian constant and n*log(scale) omitted",
            "selection": "independently grid interior, alpha=0, beta=0, and alpha=beta=0 faces; refine each face and prefer a lower-dimensional objective-equivalent attained KKT point",
            "open_boundary_rule": "omega/initial_variance and every free coefficient/slack must exceed 1e-18; transformed-gradient saturation at an open boundary is never convergence",
            "kkt_tolerance": KKT_TOLERANCE,
            "binary64_replay": "host libm values are measured and checked against fixed certified bounds but excluded from canonical JSON; only outcome/index/count and the bounds are canonical",
            "external_dependencies": [],
        },
        "case_count": len(cases),
        "cases": cases,
    }, maxima


def _format_payload(value: object) -> object:
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, dict):
        return {key: _format_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_format_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_format_payload(item) for item in value]
    return value


def _verify_precision_agreement(
    ordinary: dict[str, object], verification: dict[str, object]
) -> dict[str, object]:
    tolerance = Decimal(10) ** Decimal(-AGREEMENT_SIGNIFICANT_DIGITS)
    maximum_relative = ZERO
    maximum_absolute = ZERO
    worst_path = ""
    compared_decimals = 0
    compared_nodes = 0

    def visit(left: object, right: object, path: str) -> None:
        nonlocal maximum_relative, maximum_absolute, worst_path
        nonlocal compared_decimals, compared_nodes
        compared_nodes += 1
        if type(left) is not type(right):
            raise AssertionError(
                f"80/120 schema type mismatch at {path}: "
                f"{type(left).__name__} vs {type(right).__name__}"
            )
        if isinstance(left, dict):
            if set(left) != set(right):
                missing = sorted(set(left) - set(right))
                extra = sorted(set(right) - set(left))
                raise AssertionError(
                    f"80/120 schema key mismatch at {path}: missing={missing}, extra={extra}"
                )
            for key in sorted(left):
                visit(left[key], right[key], f"{path}/{key}")
            return
        if isinstance(left, (list, tuple)):
            if len(left) != len(right):
                raise AssertionError(
                    f"80/120 list length mismatch at {path}: {len(left)} vs {len(right)}"
                )
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{path}/{index}")
            return
        if isinstance(left, Decimal):
            compared_decimals += 1
            if not left.is_finite() or not right.is_finite():
                if left != right:
                    raise AssertionError(
                        f"80/120 non-finite mismatch at {path}: {left} vs {right}"
                    )
                return
            absolute = abs(left - right)
            relative = absolute / max(ONE, abs(left), abs(right))
            maximum_absolute = max(maximum_absolute, absolute)
            if relative > maximum_relative:
                maximum_relative = relative
                worst_path = path
            allowed = max(
                AGREEMENT_ABSOLUTE_TOLERANCE,
                tolerance * max(abs(left), abs(right)),
            )
            if absolute > allowed:
                raise AssertionError(
                    f"80/120 Decimal disagreement at {path}: {left} vs {right}; "
                    f"absolute={absolute}, allowed={allowed}"
                )
            return
        if left != right:
            raise AssertionError(
                f"80/120 scalar mismatch at {path}: {left!r} vs {right!r}"
            )

    visit(ordinary, verification, "")
    return {
        "agreement_significant_digits": AGREEMENT_SIGNIFICANT_DIGITS,
        "absolute_tolerance_near_zero": AGREEMENT_ABSOLUTE_TOLERANCE,
        "compared_decimal_fields": compared_decimals,
        "compared_schema_nodes": compared_nodes,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_scaled_difference": maximum_relative,
        "worst_path": worst_path,
    }


def _canonical_payload() -> tuple[dict[str, object], dict[str, Decimal]]:
    ordinary, ordinary_maxima = _build_payload(ORACLE_PRECISION)
    verification, verification_maxima = _build_payload(VERIFICATION_PRECISION)
    agreement = _verify_precision_agreement(ordinary, verification)
    ordinary["precision_recheck"] = agreement
    ordinary["binary64_replay_error_bounds"] = {
        "objective_abs_error": BINARY64_OBJECTIVE_ERROR_BOUND,
        "log_variance_abs_error": BINARY64_LOG_VARIANCE_ERROR_BOUND,
        "sigma2_relative_error": BINARY64_SIGMA2_RELATIVE_ERROR_BOUND,
        "canonicalization": "exact host-libm measurements are intentionally excluded from byte-for-byte artifact equality",
    }
    maxima = {
        key: max(ordinary_maxima[key], verification_maxima[key])
        for key in ordinary_maxima
    }
    formatted = _format_payload(ordinary)
    if not isinstance(formatted, dict):
        raise AssertionError("formatted oracle payload is not an object")
    return formatted, maxima


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _require_keys(value: object, expected: set[str], path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError(f"{path}: expected object, got {type(value).__name__}")
    actual = set(value)
    if actual != expected:
        raise AssertionError(
            f"{path}: keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _require_decimal_text(value: object, path: str) -> Decimal:
    if not isinstance(value, str):
        raise AssertionError(f"{path}: expected Decimal string")
    try:
        parsed = Decimal(value)
    except Exception as error:
        raise AssertionError(f"{path}: invalid Decimal string {value!r}") from error
    if not parsed.is_finite():
        raise AssertionError(f"{path}: expected finite Decimal string")
    return parsed


def _reject_json_floats(value: object, path: str = "") -> None:
    if isinstance(value, float):
        raise AssertionError(f"{path}: JSON floating number is forbidden; use a string")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_json_floats(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_json_floats(item, f"{path}/{index}")


def _validate_decimal_fields(
    value: dict[str, object], fields: Iterable[str], path: str
) -> None:
    for field in fields:
        _require_decimal_text(value[field], f"{path}/{field}")


def _validate_success_case(
    case: dict[str, object], spec: CaseSpec, path: str
) -> None:
    expected = _require_keys(
        case["expected"],
        {
            "outcome",
            "selection",
            "mean",
            "scale",
            "initial_normalized_variance",
            "omega_normalized",
            "omega_physical",
            "alpha",
            "beta",
            "stationarity_slack",
            "objective_without_constants",
            "physical_gradient",
            "kkt",
            "normalized_variances",
            "sigma2",
            "face_candidates",
            "basin_grid",
        },
        f"{path}/expected",
    )
    if expected["outcome"] != "success" or expected["selection"] != spec.expected_selection:
        raise AssertionError(f"{path}: unexpected success selection")
    _validate_decimal_fields(
        expected,
        (
            "mean",
            "scale",
            "initial_normalized_variance",
            "omega_normalized",
            "omega_physical",
            "alpha",
            "beta",
            "stationarity_slack",
            "objective_without_constants",
        ),
        f"{path}/expected",
    )
    observation_count = len(_expand_observations(spec))
    for field, required_length in (
        ("physical_gradient", 3),
        ("normalized_variances", observation_count),
        ("sigma2", observation_count),
    ):
        values = expected[field]
        if not isinstance(values, list) or len(values) != required_length:
            raise AssertionError(f"{path}/expected/{field}: invalid list length")
        for index, value in enumerate(values):
            _require_decimal_text(value, f"{path}/expected/{field}/{index}")

    kkt = expected["kkt"]
    common_kkt = {"kind", "tolerance", "free_transformed_gradient_norm"}
    if spec.expected_selection == "interior":
        kkt = _require_keys(kkt, common_kkt | {"satisfied"}, f"{path}/expected/kkt")
        if kkt["kind"] != "interior_stationarity" or kkt["satisfied"] is not True:
            raise AssertionError(f"{path}/expected/kkt: invalid interior result")
    elif spec.expected_selection == "alpha_zero":
        kkt = _require_keys(
            kkt,
            common_kkt
            | {
                "alpha_one_sided_derivative",
                "free_stationarity_satisfied",
                "alpha_kkt_satisfied",
            },
            f"{path}/expected/kkt",
        )
        if (
            kkt["kind"] != "alpha_lower_bound"
            or kkt["free_stationarity_satisfied"] is not True
            or kkt["alpha_kkt_satisfied"] is not True
        ):
            raise AssertionError(f"{path}/expected/kkt: invalid alpha-bound result")
        _require_decimal_text(
            kkt["alpha_one_sided_derivative"],
            f"{path}/expected/kkt/alpha_one_sided_derivative",
        )
    elif spec.expected_selection == "beta_zero":
        kkt = _require_keys(
            kkt,
            common_kkt
            | {
                "beta_one_sided_derivative",
                "free_stationarity_satisfied",
                "beta_kkt_satisfied",
            },
            f"{path}/expected/kkt",
        )
        if (
            kkt["kind"] != "beta_lower_bound"
            or kkt["free_stationarity_satisfied"] is not True
            or kkt["beta_kkt_satisfied"] is not True
        ):
            raise AssertionError(f"{path}/expected/kkt: invalid beta-bound result")
        _require_decimal_text(
            kkt["beta_one_sided_derivative"],
            f"{path}/expected/kkt/beta_one_sided_derivative",
        )
    _require_decimal_text(kkt["tolerance"], f"{path}/expected/kkt/tolerance")
    _require_decimal_text(
        kkt["free_transformed_gradient_norm"],
        f"{path}/expected/kkt/free_transformed_gradient_norm",
    )

    candidates = expected["face_candidates"]
    selections = ("interior", "alpha_zero", "beta_zero", "corner")
    if not isinstance(candidates, list) or len(candidates) != len(selections):
        raise AssertionError(f"{path}/expected/face_candidates: expected four faces")
    for index, (candidate, selection) in enumerate(zip(candidates, selections)):
        candidate_path = f"{path}/expected/face_candidates/{index}"
        candidate = _require_keys(
            candidate,
            {
                "selection",
                "objective",
                "omega_normalized",
                "alpha",
                "beta",
                "stationarity_slack",
                "physical_gradient",
                "transformed_gradient_norm",
                "iterations",
                "converged",
                "open_boundary_guard_rejected",
                "open_boundary_escape_detected",
                "full_kkt_satisfied",
            },
            candidate_path,
        )
        if candidate["selection"] != selection:
            raise AssertionError(f"{candidate_path}: face order mismatch")
        if (
            type(candidate["iterations"]) is not int
            or type(candidate["converged"]) is not bool
            or type(candidate["open_boundary_guard_rejected"]) is not bool
            or type(candidate["open_boundary_escape_detected"]) is not bool
            or type(candidate["full_kkt_satisfied"]) is not bool
        ):
            raise AssertionError(f"{candidate_path}: invalid iteration/convergence type")
        _validate_decimal_fields(
            candidate,
            (
                "objective",
                "omega_normalized",
                "alpha",
                "beta",
                "stationarity_slack",
                "transformed_gradient_norm",
            ),
            candidate_path,
        )
        gradient = candidate["physical_gradient"]
        if not isinstance(gradient, list) or len(gradient) != 3:
            raise AssertionError(f"{candidate_path}/physical_gradient: invalid length")
        for gradient_index, value in enumerate(gradient):
            _require_decimal_text(value, f"{candidate_path}/physical_gradient/{gradient_index}")

    basin = _require_keys(
        expected["basin_grid"],
        {
            "evaluations",
            "persistence_values",
            "share_values",
            "omega_multipliers",
            "best_face",
            "best_objective",
            "face_best_objectives",
        },
        f"{path}/expected/basin_grid",
    )
    if (
        type(basin["evaluations"]) is not int
        or basin["evaluations"] != 180
        or basin["best_face"] not in selections
    ):
        raise AssertionError(f"{path}/expected/basin_grid: invalid evidence summary")
    for field, length in (
        ("persistence_values", 7),
        ("share_values", 3),
        ("omega_multipliers", 5),
    ):
        values = basin[field]
        if not isinstance(values, list) or len(values) != length:
            raise AssertionError(f"{path}/expected/basin_grid/{field}: invalid length")
        for index, value in enumerate(values):
            _require_decimal_text(value, f"{path}/expected/basin_grid/{field}/{index}")
    _require_decimal_text(basin["best_objective"], f"{path}/expected/basin_grid/best_objective")
    face_best = _require_keys(
        basin["face_best_objectives"], set(selections), f"{path}/expected/basin_grid/face_best_objectives"
    )
    for selection, value in face_best.items():
        _require_decimal_text(value, f"{path}/expected/basin_grid/face_best_objectives/{selection}")

    replay = _require_keys(
        case["binary64_replay"],
        {
            "outcome",
            "first_infinite_standardized_square",
            "infinite_standardized_squares",
            "objective_abs_error_certified_below",
            "max_log_variance_abs_error_certified_below",
            "max_sigma2_relative_error_certified_below",
        },
        f"{path}/binary64_replay",
    )
    if (
        replay["outcome"] != "finite"
        or replay["first_infinite_standardized_square"] is not None
        or type(replay["infinite_standardized_squares"]) is not int
        or replay["infinite_standardized_squares"] != 0
    ):
        raise AssertionError(f"{path}/binary64_replay: invalid finite replay status")
    _validate_decimal_fields(
        replay,
        (
            "objective_abs_error_certified_below",
            "max_log_variance_abs_error_certified_below",
            "max_sigma2_relative_error_certified_below",
        ),
        f"{path}/binary64_replay",
    )
    if (
        Decimal(replay["objective_abs_error_certified_below"])
        != BINARY64_OBJECTIVE_ERROR_BOUND
        or Decimal(replay["max_log_variance_abs_error_certified_below"])
        != BINARY64_LOG_VARIANCE_ERROR_BOUND
        or Decimal(replay["max_sigma2_relative_error_certified_below"])
        != BINARY64_SIGMA2_RELATIVE_ERROR_BOUND
    ):
        raise AssertionError(f"{path}/binary64_replay: certified bound mismatch")


def _validate_artifact_schema(payload: object) -> dict[str, object]:
    """Fast, solver-free validation of the committed artifact's complete shape."""
    _reject_json_floats(payload)
    root = _require_keys(
        payload,
        {
            "schema_version",
            "oracle",
            "case_count",
            "cases",
            "precision_recheck",
            "binary64_replay_error_bounds",
        },
        "",
    )
    if (
        type(root["schema_version"]) is not int
        or root["schema_version"] != SCHEMA_VERSION
        or type(root["case_count"]) is not int
        or root["case_count"] != len(CASES)
    ):
        raise AssertionError("invalid schema version or case count")
    oracle = _require_keys(
        root["oracle"],
        {
            "name",
            "implementation",
            "python_requires",
            "precision_digits",
            "verification_precision_digits",
            "normalization",
            "initial_variance",
            "recurrence",
            "constraints",
            "objective",
            "selection",
            "open_boundary_rule",
            "kkt_tolerance",
            "binary64_replay",
            "external_dependencies",
        },
        "/oracle",
    )
    if (
        oracle["python_requires"] != PYTHON_REQUIRES
        or type(oracle["precision_digits"]) is not int
        or oracle["precision_digits"] != ORACLE_PRECISION
        or type(oracle["verification_precision_digits"]) is not int
        or oracle["verification_precision_digits"] != VERIFICATION_PRECISION
        or oracle["external_dependencies"] != []
    ):
        raise AssertionError("/oracle: precision or dependency contract mismatch")
    for field in (
        "name",
        "implementation",
        "python_requires",
        "normalization",
        "initial_variance",
        "recurrence",
        "constraints",
        "objective",
        "selection",
        "open_boundary_rule",
        "binary64_replay",
    ):
        if not isinstance(oracle[field], str):
            raise AssertionError(f"/oracle/{field}: expected string")
    if _require_decimal_text(oracle["kkt_tolerance"], "/oracle/kkt_tolerance") != KKT_TOLERANCE:
        raise AssertionError("/oracle/kkt_tolerance: value mismatch")

    cases = root["cases"]
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise AssertionError("/cases: invalid length")
    for index, (case_value, spec) in enumerate(zip(cases, CASES)):
        path = f"/cases/{index}"
        case = _require_keys(
            case_value, {"name", "purpose", "input", "expected", "binary64_replay"}, path
        )
        if case["name"] != spec.name or case["purpose"] != spec.purpose:
            raise AssertionError(f"{path}: fixture identity mismatch")
        expected_input = (
            dict(spec.compact_input)
            if spec.compact_input is not None
            else {"kind": "literal", "observations": list(spec.observations or ())}
        )
        if case["input"] != expected_input:
            raise AssertionError(f"{path}/input: fixture data mismatch")
        if spec.expected_selection not in {"failure", "probe"}:
            _validate_success_case(case, spec, path)
        elif spec.expected_selection == "failure":
            failure = _require_keys(
                case["expected"], {"outcome", "issue_code", "mean", "scale"}, f"{path}/expected"
            )
            if (
                failure["outcome"] != "failure"
                or failure["issue_code"] != "unidentified_model"
                or case["binary64_replay"] != {"outcome": "unidentified"}
            ):
                raise AssertionError(f"{path}: invalid constant failure")
            _validate_decimal_fields(failure, ("mean", "scale"), f"{path}/expected")
        else:
            probe = _require_keys(
                case["expected"],
                {
                    "outcome",
                    "mean",
                    "scale",
                    "initial_normalized_variance",
                    "parameters",
                    "decimal_objective",
                    "decimal_objective_is_finite",
                    "binary64_objective",
                    "interpretation",
                },
                f"{path}/expected",
            )
            if (
                probe["outcome"] != "extended_real_probe"
                or probe["decimal_objective_is_finite"] is not True
                or probe["binary64_objective"] != "Infinity"
            ):
                raise AssertionError(f"{path}/expected: invalid extended-real result")
            _validate_decimal_fields(
                probe,
                ("mean", "scale", "initial_normalized_variance", "decimal_objective"),
                f"{path}/expected",
            )
            parameters = _require_keys(
                probe["parameters"], {"omega_normalized", "alpha", "beta"}, f"{path}/expected/parameters"
            )
            _validate_decimal_fields(
                parameters, ("omega_normalized", "alpha", "beta"), f"{path}/expected/parameters"
            )
            replay = _require_keys(
                case["binary64_replay"],
                {"outcome", "first_infinite_standardized_square", "infinite_standardized_squares"},
                f"{path}/binary64_replay",
            )
            if (
                type(replay["first_infinite_standardized_square"]) is not int
                or type(replay["infinite_standardized_squares"]) is not int
            ):
                raise AssertionError(f"{path}/binary64_replay: invalid overflow index/count type")
            if replay != {
                "outcome": "positive_infinity",
                "first_infinite_standardized_square": 1200,
                "infinite_standardized_squares": 1,
            }:
                raise AssertionError(f"{path}/binary64_replay: invalid overflow evidence")

    precision = _require_keys(
        root["precision_recheck"],
        {
            "agreement_significant_digits",
            "absolute_tolerance_near_zero",
            "compared_decimal_fields",
            "compared_schema_nodes",
            "maximum_absolute_difference",
            "maximum_scaled_difference",
            "worst_path",
        },
        "/precision_recheck",
    )
    if (
        type(precision["agreement_significant_digits"]) is not int
        or precision["agreement_significant_digits"] != AGREEMENT_SIGNIFICANT_DIGITS
    ):
        raise AssertionError("/precision_recheck: agreement digit mismatch")
    if type(precision["compared_decimal_fields"]) is not int or type(precision["compared_schema_nodes"]) is not int:
        raise AssertionError("/precision_recheck: invalid counters")
    _validate_decimal_fields(
        precision,
        ("absolute_tolerance_near_zero", "maximum_absolute_difference", "maximum_scaled_difference"),
        "/precision_recheck",
    )
    if not isinstance(precision["worst_path"], str):
        raise AssertionError("/precision_recheck/worst_path: expected string")
    if (
        Decimal(precision["absolute_tolerance_near_zero"])
        != AGREEMENT_ABSOLUTE_TOLERANCE
        or Decimal(precision["maximum_scaled_difference"])
        > Decimal(10) ** Decimal(-AGREEMENT_SIGNIFICANT_DIGITS)
        or precision["compared_decimal_fields"] <= 0
        or precision["compared_schema_nodes"] <= 0
    ):
        raise AssertionError("/precision_recheck: invalid agreement evidence")
    bounds = _require_keys(
        root["binary64_replay_error_bounds"],
        {"objective_abs_error", "log_variance_abs_error", "sigma2_relative_error", "canonicalization"},
        "/binary64_replay_error_bounds",
    )
    _validate_decimal_fields(
        bounds,
        ("objective_abs_error", "log_variance_abs_error", "sigma2_relative_error"),
        "/binary64_replay_error_bounds",
    )
    if (
        Decimal(bounds["objective_abs_error"]) != BINARY64_OBJECTIVE_ERROR_BOUND
        or Decimal(bounds["log_variance_abs_error"])
        != BINARY64_LOG_VARIANCE_ERROR_BOUND
        or Decimal(bounds["sigma2_relative_error"])
        != BINARY64_SIGMA2_RELATIVE_ERROR_BOUND
    ):
        raise AssertionError("/binary64_replay_error_bounds: value mismatch")
    if not isinstance(bounds["canonicalization"], str):
        raise AssertionError("/binary64_replay_error_bounds/canonicalization: expected string")
    return root


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return _canonical_json(payload).encode("utf-8")


def _load_artifact(output: Path) -> tuple[dict[str, object], bytes]:
    data = output.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"{output} is not valid UTF-8 JSON") from error
    return _validate_artifact_schema(payload), data


def _check_pinned_digest(output: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    if EXPECTED_ARTIFACT_SHA256 == "TO_BE_REPLACED_AFTER_EMIT":
        raise SystemExit(f"{output} hash is not pinned; measured sha256={digest}")
    if digest != EXPECTED_ARTIFACT_SHA256:
        raise SystemExit(
            f"{output} hash {digest} does not match pinned "
            f"{EXPECTED_ARTIFACT_SHA256}"
        )
    return digest


def _print_binary64_measurements(maxima: dict[str, Decimal]) -> None:
    print(
        "host binary64 measured maxima (not canonicalized): "
        + ", ".join(
            f"{key}={_decimal_text(value)}" for key, value in maxima.items()
        )
    )


def _fast_check(output: Path) -> None:
    payload, data = _load_artifact(output)
    if data != _canonical_bytes(payload):
        raise SystemExit(f"{output} is not canonical sorted LF-terminated UTF-8 JSON")
    digest = _check_pinned_digest(output, data)
    print(f"checked {output} ({len(CASES)} cases, sha256={digest})")


def _deep_check(output: Path) -> None:
    existing_payload, existing = _load_artifact(output)
    if existing != _canonical_bytes(existing_payload):
        raise SystemExit(f"{output} is not canonical sorted LF-terminated UTF-8 JSON")
    _check_pinned_digest(output, existing)
    started = time.perf_counter()
    payload, maxima = _canonical_payload()
    _validate_artifact_schema(payload)
    canonical = _canonical_bytes(payload)
    if existing != canonical:
        raise SystemExit(
            f"{output} is stale; run: python {Path(__file__)} emit {output}"
        )
    digest = _check_pinned_digest(output, canonical)
    print(
        f"deep-checked {output} ({len(CASES)} cases, sha256={digest}, "
        f"runtime={time.perf_counter() - started:.3f}s)"
    )
    _print_binary64_measurements(maxima)


def _emit(output: Path) -> None:
    started = time.perf_counter()
    payload, maxima = _canonical_payload()
    _validate_artifact_schema(payload)
    canonical = _canonical_bytes(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical)
    digest = hashlib.sha256(canonical).hexdigest()
    print(
        f"wrote {output} ({len(CASES)} cases, sha256={digest}, "
        f"runtime={time.perf_counter() - started:.3f}s)"
    )
    _print_binary64_measurements(maxima)


def _run(command: str, output: Path) -> None:
    if command == "emit":
        _emit(output)
    elif command in {"check", "fast-check"}:
        _fast_check(output)
    elif command in {"deep", "deep-check"}:
        _deep_check(output)
    else:
        raise AssertionError(f"unhandled command {command}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("emit", "check", "fast-check", "deep", "deep-check")
    )
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    _run(arguments.command, arguments.output)


if __name__ == "__main__":
    main()
