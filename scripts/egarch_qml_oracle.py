#!/usr/bin/env python3
"""Independent Decimal oracle for the kanimiso EGARCH Gaussian-QML contract.

This file intentionally has no third-party dependency and does not share code
with the Rust optimizer.  It solves the normalized EGARCH(1,1) problem on the
open ``0 <= beta < 1`` domain by enumerating the exact ``beta = 0`` face and
the interior separately.  A deterministic dense grid supplies basin evidence;
Decimal BFGS refines the best seed on each face.

Contract
--------

For observations ``y`` the oracle forms ``e = y - mean(y)``, sets
``s = max(abs(e))`` and ``x = e / s``, and uses the parameter-independent
initial value ``L[0] = log(mean(x*x))``.  With
``c = sqrt(2/pi)`` and ``z[t] = x[t] * exp(-L[t]/2)``,

``L[t+1] = omega + alpha * (abs(z[t]) - c) + gamma*z[t] + beta*L[t]``.

It minimizes ``0.5*sum(L[t] + x[t]**2*exp(-L[t]))``.  The Gaussian constant
and ``n*log(s)`` are omitted.  ``alpha`` and ``gamma`` are unrestricted,
``beta`` is in ``[0,1)``, and the interior uses the long-run coordinate
``mu = omega/(1-beta)``.  The physical intercept is
``omega + 2*(1-beta)*log(s)``.

Commands::

    python scripts/egarch_qml_oracle.py emit golden/egarch_qml.json
    python scripts/egarch_qml_oracle.py fast-check golden/egarch_qml.json
    python scripts/egarch_qml_oracle.py deep-check golden/egarch_qml.json

``deep-check`` independently solves at 80 and 120 digits, compares raw Decimal
results before formatting, audits the analytic gradient and reflection
identity, and then checks the canonical JSON.  Host binary64 replay errors are
reported to stdout but never serialized as canonical measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Iterable, Sequence


ORACLE_PRECISION = 80
VERIFICATION_PRECISION = 120
OUTPUT_SIGNIFICANT_DIGITS = 24
AGREEMENT_SIGNIFICANT_DIGITS = 25
MAX_BFGS_ITERATIONS = 280
MAX_LINE_SEARCH_STEPS = 140
OPTIMIZER_GRADIENT_TOLERANCE = Decimal("1e-30")
KKT_TOLERANCE = Decimal("1e-24")
RAW_AGREEMENT_ABSOLUTE_TOLERANCE = Decimal("1e-68")
OPEN_BOUNDARY_GUARD = Decimal("1e-16")

# Updated only after emit/deep-check.  A placeholder deliberately makes
# fast-check fail, preventing an unreviewed artifact from becoming canonical.
EXPECTED_ARTIFACT_SHA256 = (
    "b913315b96541603da586f03720c29679750952780791b8fb5d21783ed501af0"
)

SQRT_2_OVER_PI_TEXT = (
    "0.79788456080286535587989211986876373695171726232986931533185165934131585179860367700250466781461387286060511772527036537102198390911167448598"
)

# Bundled CPython 3.12 maxima were 8.395e-14, 1.144e-15, 1.124e-15,
# and 4.346e-16 respectively.  The certified bounds retain a 4x margin;
# measured host values remain stdout-only and are never canonical fields.
BINARY64_OBJECTIVE_ABS_BOUND = Decimal("3.36e-13")
BINARY64_LOG_VARIANCE_ABS_BOUND = Decimal("4.58e-15")
BINARY64_VARIANCE_REL_BOUND = Decimal("4.50e-15")
BINARY64_FORECAST_REL_BOUND = Decimal("1.74e-15")

ZERO = Decimal(0)
ONE = Decimal(1)
TWO = Decimal(2)
HALF = Decimal("0.5")


@dataclass(frozen=True)
class CaseSpec:
    name: str
    purpose: str
    observations: tuple[str, ...]
    expected_selection: str
    expected_property: str
    reflection_of: str | None = None
    probe_parameters: tuple[str, str, str, str] | None = None


@dataclass(frozen=True)
class PreparedSeries:
    mean: Decimal
    residuals: tuple[Decimal, ...]
    scale: Decimal
    normalized: tuple[Decimal, ...]
    initial_log_variance: Decimal | None


@dataclass(frozen=True)
class Evaluation:
    objective: Decimal
    physical_gradient: tuple[Decimal, Decimal, Decimal, Decimal]
    log_variances: tuple[Decimal, ...]
    physical_variances: tuple[Decimal, ...]
    one_step_log_variance: Decimal
    one_step_physical_variance: Decimal


@dataclass(frozen=True)
class Candidate:
    selection: str
    coordinates: tuple[Decimal, ...]
    mu: Decimal
    omega: Decimal
    alpha: Decimal
    gamma: Decimal
    beta: Decimal
    evaluation: Evaluation
    transformed_gradient: tuple[Decimal, ...]
    iterations: int
    converged: bool
    open_boundary_guard_rejected: bool
    open_boundary_escape_detected: bool
    beta_zero_directional_derivative: Decimal | None


@dataclass(frozen=True)
class SolveResult:
    selected: Candidate
    face_candidates: tuple[Candidate, ...]
    basin_grid: dict[str, object]


def _decimal_pi() -> Decimal:
    """Gauss--Legendre pi, converging quadratically in Decimal arithmetic."""
    a = ONE
    b = ONE / TWO.sqrt()
    t = ONE / Decimal(4)
    p = ONE
    # Seven rounds already cover 100+ digits; ten give ample guard digits.
    for _ in range(10):
        next_a = (a + b) / TWO
        b = (a * b).sqrt()
        t -= p * (a - next_a) * (a - next_a)
        a = next_a
        p *= TWO
    return (a + b) * (a + b) / (Decimal(4) * t)


def _sqrt_2_over_pi() -> Decimal:
    calculated = (TWO / _decimal_pi()).sqrt()
    embedded = Decimal(SQRT_2_OVER_PI_TEXT)
    # Audit all digits that the active context can support, leaving guard room
    # for the independent pi calculation and Decimal rounding.
    from decimal import getcontext

    checked_digits = min(getcontext().prec - 8, 120)
    tolerance = Decimal(10) ** Decimal(-checked_digits)
    if abs(calculated - embedded) > tolerance:
        raise AssertionError("embedded sqrt(2/pi) failed Decimal audit")
    return calculated


def _lcg_normal_surrogates(count: int, seed: int) -> tuple[Decimal, ...]:
    """Deterministic centered/unit-RMS Irwin--Hall normal surrogates."""
    modulus = 1 << 32
    state = seed & 0xFFFF_FFFF
    raw: list[Decimal] = []
    with localcontext() as ctx:
        ctx.prec = 70
        denominator = Decimal(modulus)
        for _ in range(count):
            total = ZERO
            for _ in range(12):
                state = (1_664_525 * state + 1_013_904_223) & 0xFFFF_FFFF
                total += Decimal(state) / denominator
            raw.append(total - Decimal(6))
        mean = sum(raw, ZERO) / Decimal(count)
        centered = [value - mean for value in raw]
        rms = (sum((value * value for value in centered), ZERO) / Decimal(count)).sqrt()
        return tuple(value / rms for value in centered)


def _simulate_egarch_observations(
    mu: str,
    alpha: str,
    gamma: str,
    beta: str,
    innovations: Sequence[Decimal],
) -> tuple[str, ...]:
    """Create fixed decimal observations; it is not used by either solver."""
    with localcontext() as ctx:
        ctx.prec = 70
        m = Decimal(mu)
        a = Decimal(alpha)
        g = Decimal(gamma)
        b = Decimal(beta)
        c = Decimal(SQRT_2_OVER_PI_TEXT)
        omega = (ONE - b) * m
        logh = m
        residuals: list[Decimal] = []
        for z in innovations:
            residuals.append((logh / TWO).exp() * z)
            logh = omega + a * (abs(z) - c) + g * z + b * logh
        # Center before freezing the strings so the fixture does not rely on a
        # nonzero simulated sample mean.  The oracle still demeans independently.
        mean = sum(residuals, ZERO) / Decimal(len(residuals))
        return tuple(format(value - mean, ".28g") for value in residuals)


def _negate_strings(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(format(-Decimal(value), "f") for value in values)


_LEVERAGED = _simulate_egarch_observations(
    "-0.35", "0.19", "-0.24", "0.78", _lcg_normal_surrogates(180, 0xE6A4_1101)
)
_GAMMA_ZERO = _simulate_egarch_observations(
    "-0.18", "0.23", "0", "0.64",
    _lcg_normal_surrogates(260, 0x6A77_0011),
)
_BETA_ZERO = _simulate_egarch_observations(
    "-0.30", "0.28", "-0.14", "0",
    _lcg_normal_surrogates(400, 0xDE7A_0047),
)
_NEAR_INTEGRATED = _simulate_egarch_observations(
    "-0.55", "0.10", "-0.16", "0.97",
    _lcg_normal_surrogates(320, 0x97E6_A2C4),
)
_LONG_QUIET = (
    ("1", "-1") + ("0",) * 1198 + ("1", "-1")
)


CASES = (
    CaseSpec(
        "leveraged",
        "Interior EGARCH fit with negative leverage coefficient gamma.",
        _LEVERAGED,
        "interior",
        "gamma_negative",
    ),
    CaseSpec(
        "leveraged_reflected",
        "Exact sign reflection; gamma must change sign and all variance results remain equal.",
        _negate_strings(_LEVERAGED),
        "interior",
        "reflection",
        reflection_of="leveraged",
    ),
    CaseSpec(
        "gamma_zero",
        "Data generated with symmetric-news gamma equal to zero.",
        _GAMMA_ZERO,
        "interior",
        "gamma_near_zero",
    ),
    CaseSpec(
        "beta_zero",
        "Exact beta=0 face with a nonnegative one-sided KKT derivative.",
        _BETA_ZERO,
        "beta_zero",
        "beta_zero_face",
    ),
    CaseSpec(
        "near_integrated",
        "Attained high-persistence optimum, separated from the open beta=1 boundary.",
        _NEAR_INTEGRATED,
        "interior",
        "beta_near_point_97",
    ),
    CaseSpec(
        "constant_failure",
        "Demeaning makes scale zero, so volatility parameters are unidentified.",
        tuple("7.125" for _ in range(24)),
        "unidentified",
        "constant_failure",
    ),
    CaseSpec(
        "long_quiet_extreme",
        "Decimal remains finite while binary64 standardized-square replay reaches +infinity.",
        _LONG_QUIET,
        "probe",
        "extended_real",
        probe_parameters=("-800", "0.10", "-0.10", "0.99"),
    ),
)


def _prepare(observations: Sequence[str]) -> PreparedSeries:
    values = tuple(Decimal(value) for value in observations)
    if not values:
        raise ValueError("empty series")
    mean = sum(values, ZERO) / Decimal(len(values))
    residuals = tuple(value - mean for value in values)
    scale = max(abs(value) for value in residuals)
    if scale == ZERO:
        return PreparedSeries(mean, residuals, scale, tuple(ZERO for _ in values), None)
    normalized = tuple(value / scale for value in residuals)
    mean_square = sum((value * value for value in normalized), ZERO) / Decimal(len(values))
    return PreparedSeries(mean, residuals, scale, normalized, mean_square.ln())


def _decode(selection: str, coordinates: Sequence[Decimal]) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
    if selection == "interior":
        mu, alpha, gamma, beta_logit = coordinates
        if beta_logit >= ZERO:
            exp_neg = (-beta_logit).exp()
            beta = ONE / (ONE + exp_neg)
        else:
            exp_pos = beta_logit.exp()
            beta = exp_pos / (ONE + exp_pos)
    elif selection == "beta_zero":
        mu, alpha, gamma = coordinates
        beta = ZERO
    else:
        raise ValueError(f"unknown face {selection!r}")
    omega = (ONE - beta) * mu
    return mu, omega, alpha, gamma, beta


def _objective_only(
    prepared: PreparedSeries,
    mu: Decimal,
    alpha: Decimal,
    gamma: Decimal,
    beta: Decimal,
) -> Decimal:
    if prepared.initial_log_variance is None or beta < ZERO or beta >= ONE:
        return Decimal("Infinity")
    try:
        c = Decimal(SQRT_2_OVER_PI_TEXT)
        omega = (ONE - beta) * mu
        logh = prepared.initial_log_variance
        objective = ZERO
        for x in prepared.normalized:
            z = x * (-logh / TWO).exp()
            standardized_square = z * z
            objective += HALF * (logh + standardized_square)
            logh = omega + alpha * (abs(z) - c) + gamma * z + beta * logh
        return objective
    except (ArithmeticError, InvalidOperation):
        return Decimal("Infinity")


def _decimal_extended_exp(value: Decimal) -> Decimal:
    """Decimal exp with the model's ordered extended-real overflow contract."""
    try:
        return value.exp()
    except (ArithmeticError, InvalidOperation):
        return Decimal("Infinity") if value > ZERO else ZERO


def _evaluate(
    prepared: PreparedSeries,
    mu: Decimal,
    alpha: Decimal,
    gamma: Decimal,
    beta: Decimal,
) -> Evaluation:
    if prepared.initial_log_variance is None:
        raise ValueError("constant series")
    if beta < ZERO or beta >= ONE:
        raise ValueError("beta outside [0,1)")
    c = Decimal(SQRT_2_OVER_PI_TEXT)
    omega = (ONE - beta) * mu
    logh = prepared.initial_log_variance
    derivative = [ZERO, ZERO, ZERO, ZERO]  # omega, alpha, gamma, beta
    gradient = [ZERO, ZERO, ZERO, ZERO]
    objective = ZERO
    log_path: list[Decimal] = []
    physical_path: list[Decimal] = []
    scale_square = prepared.scale * prepared.scale

    for x in prepared.normalized:
        z = x * (-logh / TWO).exp()
        standardized_square = z * z
        objective += HALF * (logh + standardized_square)
        score_factor = HALF * (ONE - standardized_square)
        for index in range(4):
            gradient[index] += score_factor * derivative[index]
        log_path.append(logh)
        physical_path.append(scale_square * _decimal_extended_exp(logh))

        q = abs(z) - c
        feedback = beta - HALF * (alpha * abs(z) + gamma * z)
        direct = (ONE, q, z, logh)
        derivative = [direct[i] + feedback * derivative[i] for i in range(4)]
        logh = omega + alpha * q + gamma * z + beta * logh

    return Evaluation(
        objective,
        tuple(gradient),  # type: ignore[arg-type]
        tuple(log_path),
        tuple(physical_path),
        logh,
        scale_square * _decimal_extended_exp(logh),
    )


def _transformed_evaluation(
    prepared: PreparedSeries,
    selection: str,
    coordinates: Sequence[Decimal],
) -> tuple[Evaluation, tuple[Decimal, ...]]:
    mu, omega, alpha, gamma, beta = _decode(selection, coordinates)
    del omega
    evaluation = _evaluate(prepared, mu, alpha, gamma, beta)
    g_omega, g_alpha, g_gamma, g_beta = evaluation.physical_gradient
    g_mu = (ONE - beta) * g_omega
    if selection == "beta_zero":
        return evaluation, (g_mu, g_alpha, g_gamma)
    beta_factor = beta * (ONE - beta)
    g_beta_logit = beta_factor * (g_beta - mu * g_omega)
    return evaluation, (g_mu, g_alpha, g_gamma, g_beta_logit)


def _dot(left: Sequence[Decimal], right: Sequence[Decimal]) -> Decimal:
    return sum((a * b for a, b in zip(left, right)), ZERO)


def _identity(size: int) -> list[list[Decimal]]:
    return [[ONE if row == column else ZERO for column in range(size)] for row in range(size)]


def _mat_vec(matrix: Sequence[Sequence[Decimal]], vector: Sequence[Decimal]) -> list[Decimal]:
    return [sum((value * item for value, item in zip(row, vector)), ZERO) for row in matrix]


def _inverse_bfgs_update(
    inverse_hessian: Sequence[Sequence[Decimal]],
    step: Sequence[Decimal],
    gradient_delta: Sequence[Decimal],
) -> list[list[Decimal]]:
    size = len(step)
    curvature = _dot(gradient_delta, step)
    scale = max(ONE, *(abs(value) for value in step), *(abs(value) for value in gradient_delta))
    if curvature <= Decimal("1e-64") * scale * scale:
        return _identity(size)
    rho = ONE / curvature
    left = [[(ONE if r == c else ZERO) - rho * step[r] * gradient_delta[c] for c in range(size)] for r in range(size)]
    right = [[(ONE if r == c else ZERO) - rho * gradient_delta[r] * step[c] for c in range(size)] for r in range(size)]
    middle = [[sum((left[r][k] * inverse_hessian[k][c] for k in range(size)), ZERO) for c in range(size)] for r in range(size)]
    return [[sum((middle[r][k] * right[k][c] for k in range(size)), ZERO) + rho * step[r] * step[c] for c in range(size)] for r in range(size)]


def _bfgs(
    prepared: PreparedSeries,
    selection: str,
    initial: Sequence[Decimal],
) -> Candidate:
    coordinates = list(initial)
    evaluation, gradient_tuple = _transformed_evaluation(prepared, selection, coordinates)
    gradient = list(gradient_tuple)
    inverse_hessian = _identity(len(coordinates))
    converged = False
    iterations = 0

    for iteration in range(MAX_BFGS_ITERATIONS):
        iterations = iteration
        mu, omega, alpha, gamma, beta = _decode(selection, coordinates)
        del omega, alpha, gamma
        near_open_boundary = selection == "interior" and min(beta, ONE - beta) <= OPEN_BOUNDARY_GUARD
        if near_open_boundary:
            break
        if max(abs(value) for value in gradient) <= OPTIMIZER_GRADIENT_TOLERANCE:
            converged = True
            break

        direction = [-value for value in _mat_vec(inverse_hessian, gradient)]
        directional_derivative = _dot(gradient, direction)
        if directional_derivative >= ZERO:
            direction = [-value for value in gradient]
            directional_derivative = -_dot(gradient, gradient)
            inverse_hessian = _identity(len(coordinates))

        accepted = False
        step_size = ONE
        next_evaluation = evaluation
        next_gradient: tuple[Decimal, ...] = tuple(gradient)
        next_coordinates = coordinates
        for _ in range(MAX_LINE_SEARCH_STEPS):
            trial = [value + step_size * delta for value, delta in zip(coordinates, direction)]
            try:
                trial_evaluation, trial_gradient = _transformed_evaluation(prepared, selection, trial)
            except (ArithmeticError, InvalidOperation, ValueError):
                step_size *= HALF
                continue
            if trial_evaluation.objective <= evaluation.objective + Decimal("1e-4") * step_size * directional_derivative:
                accepted = True
                next_evaluation = trial_evaluation
                next_gradient = trial_gradient
                next_coordinates = trial
                break
            step_size *= HALF
        if not accepted:
            break

        actual_step = [new - old for new, old in zip(next_coordinates, coordinates)]
        gradient_delta = [new - old for new, old in zip(next_gradient, gradient)]
        inverse_hessian = _inverse_bfgs_update(inverse_hessian, actual_step, gradient_delta)
        coordinates = list(next_coordinates)
        evaluation = next_evaluation
        gradient = list(next_gradient)
    else:
        iterations = MAX_BFGS_ITERATIONS

    mu, omega, alpha, gamma, beta = _decode(selection, coordinates)
    boundary_rejected = selection == "interior" and min(beta, ONE - beta) <= OPEN_BOUNDARY_GUARD
    # A direct, finite displacement toward beta=1 detects candidates following
    # an unattained open-boundary infimum even if the logit gradient is tiny.
    open_escape = False
    if selection == "interior":
        gap = ONE - beta
        probe_betas = tuple(ONE - gap / divisor for divisor in (TWO, Decimal(4), Decimal(8)))
        open_escape = any(
            _objective_only(prepared, mu, alpha, gamma, probe_beta)
            < evaluation.objective - KKT_TOLERANCE
            for probe_beta in probe_betas
        )
    g_omega, _, _, g_beta = evaluation.physical_gradient
    beta_zero_direction = g_beta - mu * g_omega if selection == "beta_zero" else None
    return Candidate(
        selection,
        tuple(coordinates),
        mu,
        omega,
        alpha,
        gamma,
        beta,
        evaluation,
        tuple(gradient),
        iterations,
        converged and not boundary_rejected and not open_escape,
        boundary_rejected,
        open_escape,
        beta_zero_direction,
    )


def _logit(probability: Decimal) -> Decimal:
    return (probability / (ONE - probability)).ln()


def _initial_coordinates(selection: str, mu: Decimal, alpha: Decimal, gamma: Decimal, beta: Decimal) -> tuple[Decimal, ...]:
    if selection == "interior":
        return (mu, alpha, gamma, _logit(beta))
    return (mu, alpha, gamma)


def _eligible(candidate: Candidate) -> bool:
    if not candidate.converged or max(abs(value) for value in candidate.transformed_gradient) > KKT_TOLERANCE:
        return False
    if candidate.selection == "beta_zero":
        direction = candidate.beta_zero_directional_derivative
        return direction is not None and direction >= -KKT_TOLERANCE
    return True


def _insert_best(entries: list[tuple[Decimal, tuple[Decimal, Decimal, Decimal, Decimal]]], item: tuple[Decimal, tuple[Decimal, Decimal, Decimal, Decimal]], limit: int = 4) -> None:
    entries.append(item)
    entries.sort(key=lambda value: value[0])
    del entries[limit:]


def _solve(prepared: PreparedSeries) -> SolveResult:
    if prepared.initial_log_variance is None:
        raise ValueError("constant series")
    mu_offsets = tuple(Decimal(value) for value in ("-4", "-2", "0", "2", "4"))
    alpha_values = tuple(Decimal(value) for value in ("-0.30", "0", "0.12", "0.30", "0.60"))
    gamma_values = tuple(Decimal(value) for value in ("-0.40", "-0.15", "0", "0.15", "0.40"))
    beta_values = tuple(Decimal(value) for value in ("0.03", "0.30", "0.60", "0.82", "0.94", "0.98"))
    best: dict[str, list[tuple[Decimal, tuple[Decimal, Decimal, Decimal, Decimal]]]] = {"interior": [], "beta_zero": []}
    evaluations = 0
    l0 = prepared.initial_log_variance
    for offset in mu_offsets:
        mu = l0 + offset
        for alpha in alpha_values:
            for gamma in gamma_values:
                face_objective = _objective_only(prepared, mu, alpha, gamma, ZERO)
                _insert_best(best["beta_zero"], (face_objective, (mu, alpha, gamma, ZERO)))
                evaluations += 1
                for beta in beta_values:
                    objective = _objective_only(prepared, mu, alpha, gamma, beta)
                    _insert_best(best["interior"], (objective, (mu, alpha, gamma, beta)))
                    evaluations += 1

    face_candidates: list[Candidate] = []
    refinement_summaries: dict[str, list[Candidate]] = {}
    for selection in ("interior", "beta_zero"):
        refined: list[Candidate] = []
        # The dense grid is independent basin evidence.  Its best point on
        # each explicitly enumerated face is refined by Decimal BFGS.
        for _, seed in best[selection][:1]:
            refined.append(_bfgs(prepared, selection, _initial_coordinates(selection, *seed)))
        refinement_summaries[selection] = refined
        eligible = [candidate for candidate in refined if _eligible(candidate)]
        if eligible:
            face_candidates.append(min(eligible, key=lambda candidate: candidate.evaluation.objective))
        else:
            face_candidates.append(min(refined, key=lambda candidate: candidate.evaluation.objective))

    eligible_faces = [candidate for candidate in face_candidates if _eligible(candidate)]
    if not eligible_faces:
        details = ", ".join(
            f"{c.selection}:conv={c.converged},grad={max(abs(v) for v in c.transformed_gradient)}"
            for c in face_candidates
        )
        raise ArithmeticError(f"no attained KKT candidate ({details})")
    minimum = min(candidate.evaluation.objective for candidate in eligible_faces)
    tie_tolerance = Decimal("1e-20") * max(ONE, abs(minimum))
    equivalent = [candidate for candidate in eligible_faces if candidate.evaluation.objective - minimum <= tie_tolerance]
    complexity = {"beta_zero": 0, "interior": 1}
    selected = min(equivalent, key=lambda candidate: complexity[candidate.selection])
    grid_best = min((entry for entries in best.values() for entry in entries), key=lambda item: item[0])[0]
    if selected.evaluation.objective > grid_best + tie_tolerance:
        raise AssertionError("refined EGARCH solution is worse than dense grid")

    basin_grid: dict[str, object] = {
        "evaluations": evaluations,
        "mu_offsets": mu_offsets,
        "alpha_values": alpha_values,
        "gamma_values": gamma_values,
        "beta_values": beta_values,
        "face_grid_best_objectives": {key: values[0][0] for key, values in best.items()},
        "local_refinements_per_face": 1,
        "local_refinement_objectives": {
            key: tuple(candidate.evaluation.objective for candidate in values)
            for key, values in refinement_summaries.items()
        },
    }
    return SolveResult(selected, tuple(face_candidates), basin_grid)


def _format_decimal(value: Decimal) -> str:
    if not value.is_finite():
        return "+inf" if value > ZERO else "-inf"
    if value == ZERO:
        return "0"
    return format(value, f".{OUTPUT_SIGNIFICANT_DIGITS}g")


def _format_decimal_tree(value: object) -> object:
    if isinstance(value, Decimal):
        return _format_decimal(value)
    if isinstance(value, dict):
        return {key: _format_decimal_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_format_decimal_tree(item) for item in value]
    return value


def _candidate_record(candidate: Candidate, prepared: PreparedSeries) -> dict[str, object]:
    g_omega, g_alpha, g_gamma, g_beta = candidate.evaluation.physical_gradient
    omega_physical = candidate.omega + TWO * (ONE - candidate.beta) * prepared.scale.ln()
    mu_physical = candidate.mu + TWO * prepared.scale.ln()
    return {
        "selection": candidate.selection,
        "mu_normalized": candidate.mu,
        "mu_physical": mu_physical,
        "omega_normalized": candidate.omega,
        "omega_physical": omega_physical,
        "alpha": candidate.alpha,
        "gamma": candidate.gamma,
        "beta": candidate.beta,
        "persistence_slack": ONE - candidate.beta,
        "objective": candidate.evaluation.objective,
        "physical_gradient": {
            "omega": g_omega,
            "alpha": g_alpha,
            "gamma": g_gamma,
            "beta": g_beta,
        },
        "transformed_gradient": candidate.transformed_gradient,
        "transformed_gradient_norm": max(abs(value) for value in candidate.transformed_gradient),
        "iterations": candidate.iterations,
        "converged": candidate.converged,
        "open_boundary_guard_rejected": candidate.open_boundary_guard_rejected,
        "open_boundary_escape_detected": candidate.open_boundary_escape_detected,
        "beta_zero_directional_derivative": candidate.beta_zero_directional_derivative,
    }


def _success_record(spec: CaseSpec, prepared: PreparedSeries, solved: SolveResult) -> dict[str, object]:
    selected = solved.selected
    record: dict[str, object] = {
        "name": spec.name,
        "purpose": spec.purpose,
        "observations": list(spec.observations),
        "expected_selection": spec.expected_selection,
        "expected_property": spec.expected_property,
        "reflection_of": spec.reflection_of,
        "outcome": "success",
        "mean": prepared.mean,
        "scale": prepared.scale,
        "initial_log_variance": prepared.initial_log_variance,
        "fit": _candidate_record(selected, prepared),
        "normalized_log_variances": selected.evaluation.log_variances,
        "physical_variances": selected.evaluation.physical_variances,
        "one_step_forecast": {
            "normalized_log_variance": selected.evaluation.one_step_log_variance,
            "physical_log_variance": selected.evaluation.one_step_log_variance + TWO * prepared.scale.ln(),
            "physical_variance": selected.evaluation.one_step_physical_variance,
        },
        "face_candidates": [_candidate_record(candidate, prepared) for candidate in solved.face_candidates],
        "basin_grid": solved.basin_grid,
    }
    return record


def _probe_record(spec: CaseSpec, prepared: PreparedSeries) -> dict[str, object]:
    if spec.probe_parameters is None:
        raise AssertionError("probe parameters missing")
    mu, alpha, gamma, beta = (Decimal(value) for value in spec.probe_parameters)
    evaluation = _evaluate(prepared, mu, alpha, gamma, beta)
    return {
        "name": spec.name,
        "purpose": spec.purpose,
        "observations": list(spec.observations),
        "expected_selection": spec.expected_selection,
        "expected_property": spec.expected_property,
        "reflection_of": spec.reflection_of,
        "outcome": "extended_real_probe",
        "mean": prepared.mean,
        "scale": prepared.scale,
        "initial_log_variance": prepared.initial_log_variance,
        "parameters": {"mu_normalized": mu, "omega_normalized": (ONE - beta) * mu, "alpha": alpha, "gamma": gamma, "beta": beta},
        "decimal_objective": evaluation.objective,
        "decimal_objective_is_finite": evaluation.objective.is_finite(),
        "normalized_log_variances": evaluation.log_variances,
        "one_step_forecast": {
            "normalized_log_variance": evaluation.one_step_log_variance,
            "physical_variance": evaluation.one_step_physical_variance,
        },
    }


def _failure_record(spec: CaseSpec, prepared: PreparedSeries) -> dict[str, object]:
    return {
        "name": spec.name,
        "purpose": spec.purpose,
        "observations": list(spec.observations),
        "expected_selection": spec.expected_selection,
        "expected_property": spec.expected_property,
        "reflection_of": spec.reflection_of,
        "outcome": "unidentified_constant_series",
        "mean": prepared.mean,
        "scale": prepared.scale,
        "failure": "demeaned maximum absolute residual is zero",
    }


def _assert_case_properties(raw_cases: Sequence[dict[str, object]]) -> dict[str, object]:
    by_name = {str(case["name"]): case for case in raw_cases}
    leveraged = by_name["leveraged"]
    reflected = by_name["leveraged_reflected"]
    gamma_zero = by_name["gamma_zero"]
    beta_zero = by_name["beta_zero"]
    near = by_name["near_integrated"]
    leveraged_fit = leveraged["fit"]
    reflected_fit = reflected["fit"]
    gamma_zero_fit = gamma_zero["fit"]
    beta_zero_fit = beta_zero["fit"]
    near_fit = near["fit"]
    assert isinstance(leveraged_fit, dict) and isinstance(reflected_fit, dict)
    assert isinstance(gamma_zero_fit, dict) and isinstance(beta_zero_fit, dict) and isinstance(near_fit, dict)
    if not (leveraged_fit["gamma"] < ZERO):
        raise AssertionError("leveraged fixture did not select gamma < 0")
    if abs(gamma_zero_fit["gamma"]) > Decimal("0.12"):
        raise AssertionError("gamma-zero fixture estimate is not near zero")
    if beta_zero_fit["selection"] != "beta_zero" or beta_zero_fit["beta"] != ZERO:
        raise AssertionError("beta-zero fixture did not select exact face")
    if not (Decimal("0.94") < near_fit["beta"] < Decimal("0.995")):
        raise AssertionError("near-integrated fixture did not recover high persistence")

    pairs = (
        (leveraged_fit["mu_normalized"], reflected_fit["mu_normalized"]),
        (leveraged_fit["omega_normalized"], reflected_fit["omega_normalized"]),
        (leveraged_fit["alpha"], reflected_fit["alpha"]),
        (leveraged_fit["beta"], reflected_fit["beta"]),
        (leveraged_fit["objective"], reflected_fit["objective"]),
    )
    reflection_error = max(abs(left - right) for left, right in pairs)
    reflection_error = max(reflection_error, abs(leveraged_fit["gamma"] + reflected_fit["gamma"]))
    left_path = leveraged["normalized_log_variances"]
    right_path = reflected["normalized_log_variances"]
    assert isinstance(left_path, tuple) and isinstance(right_path, tuple)
    reflection_error = max(reflection_error, max(abs(left - right) for left, right in zip(left_path, right_path)))
    if reflection_error > Decimal("1e-23"):
        raise AssertionError(f"reflection identity error {reflection_error}")
    return {
        "leveraged_gamma_negative": True,
        "gamma_zero_absolute_bound": Decimal("0.12"),
        "beta_zero_face_selected": True,
        "near_integrated_beta_interval": (Decimal("0.94"), Decimal("0.995")),
        "reflection_max_absolute_error": reflection_error,
    }


def _build_raw(precision: int) -> dict[str, object]:
    with localcontext() as ctx:
        ctx.prec = precision
        ctx.Emax = 999_999_999
        ctx.Emin = -999_999_999
        c = _sqrt_2_over_pi()
        raw_cases: list[dict[str, object]] = []
        for spec in CASES:
            prepared = _prepare(spec.observations)
            if spec.expected_selection == "unidentified":
                raw_cases.append(_failure_record(spec, prepared))
            elif spec.expected_selection == "probe":
                raw_cases.append(_probe_record(spec, prepared))
            else:
                solved = _solve(prepared)
                if solved.selected.selection != spec.expected_selection:
                    raise AssertionError(
                        f"{spec.name}: selected {solved.selected.selection}, expected {spec.expected_selection}"
                    )
                raw_cases.append(_success_record(spec, prepared, solved))
        cross_checks = _assert_case_properties(raw_cases)
        return {
            "schema_version": 1,
            "oracle": "independent Decimal EGARCH(1,1) Gaussian QML",
            "contract": {
                "centering": "demean observations",
                "normalization": "divide residuals by max absolute residual",
                "initialization": "fixed log(mean(normalized_residual^2)), parameter independent",
                "recurrence": "L[t+1]=omega+alpha*(abs(z[t])-sqrt(2/pi))+gamma*z[t]+beta*L[t]",
                "standardized_residual": "z[t]=x[t]*exp(-L[t]/2)",
                "objective": "0.5*sum(L[t]+x[t]^2*exp(-L[t])); Gaussian constant and n*log(scale) omitted",
                "domain": "alpha,gamma unrestricted; beta in [0,1); exact beta=0 face plus interior",
                "coordinates": "mu=omega/(1-beta); omega_physical=omega_normalized+2*(1-beta)*log(scale)",
            },
            "constants": {
                "sqrt_2_over_pi_embedded": SQRT_2_OVER_PI_TEXT,
                "sqrt_2_over_pi_decimal_audit": c,
                "kkt_tolerance": KKT_TOLERANCE,
                "open_boundary_guard": OPEN_BOUNDARY_GUARD,
                "output_significant_digits": OUTPUT_SIGNIFICANT_DIGITS,
            },
            "case_count": len(raw_cases),
            "cases": raw_cases,
            "cross_case_checks": cross_checks,
            "precision_recheck": {
                "primary_decimal_digits": ORACLE_PRECISION,
                "verification_decimal_digits": VERIFICATION_PRECISION,
                "raw_agreement_significant_digits": AGREEMENT_SIGNIFICANT_DIGITS,
                "comparison": "raw Decimal trees compared before canonical formatting",
            },
            "binary64_replay": {
                "canonical_measurements": "none; host observations are printed by deep-check",
                "objective_absolute_error_bound": BINARY64_OBJECTIVE_ABS_BOUND,
                "log_variance_absolute_error_bound": BINARY64_LOG_VARIANCE_ABS_BOUND,
                "variance_relative_error_bound": BINARY64_VARIANCE_REL_BOUND,
                "one_step_forecast_relative_error_bound": BINARY64_FORECAST_REL_BOUND,
                "extended_real_rule": "math.exp overflow is ordered +infinity, never NaN or an exception",
            },
            "rust_kernel_tolerances": {
                "measurement_date": "2026-09-02",
                "measurement_basis": "measured against the Rust EGARCH kernel replay",
                "certification": "empirical Rust binary64 errors, not Decimal certification",
                "tolerance_policy": "approximately four times each measured maximum",
                "measured_maxima": {
                    "objective_absolute": Decimal("5.684341886080802e-14"),
                    "log_variance_absolute": Decimal("1.3322676295501878e-15"),
                    "physical_variance_relative": Decimal("1.5543122344752192e-15"),
                    "one_step_forecast_relative": Decimal("1.1102230246251565e-15"),
                    "physical_gradient_absolute": Decimal("1.191935439237568e-12"),
                },
                "objective_absolute": Decimal("2.28e-13"),
                "log_variance_absolute": Decimal("5.34e-15"),
                "physical_variance_relative": Decimal("6.22e-15"),
                "one_step_forecast_relative": Decimal("4.45e-15"),
                "physical_gradient_absolute": Decimal("4.77e-12"),
            },
            "rust_fit_tolerances": {
                "measurement_date": "2026-09-02",
                "measurement_basis": "measured against the Rust EGARCH fit and forecast replay",
                "certification": "empirical Rust optimizer errors, not Decimal certification",
                "tolerance_policy": "current Rust tests use approximately four times each measured maximum",
                "measured_maxima": {
                    "coefficient_absolute": Decimal("2.2463785886994714e-8"),
                    "omega_absolute": Decimal("9.488963226278457e-9"),
                    "objective_absolute": Decimal("2.2737367544323206e-13"),
                    "log_variance_absolute": Decimal("7.842927929324617e-8"),
                    "physical_variance_relative": Decimal("7.842927662871091e-8"),
                    "one_step_forecast_relative": Decimal("1.9368434078792518e-8"),
                },
                "coefficient_absolute": Decimal("8.99e-8"),
                "omega_absolute": Decimal("3.80e-8"),
                "objective_absolute": Decimal("9.10e-13"),
                "log_variance_absolute": Decimal("3.14e-7"),
                "physical_variance_relative": Decimal("3.14e-7"),
                "one_step_forecast_relative": Decimal("7.75e-8"),
            },
        }


def _float_safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _float_welford_mean(values: Sequence[float]) -> float:
    mean = 0.0
    for count, value in enumerate(values, 1):
        mean += (value - mean) / count
    return mean


def _binary64_replay(spec: CaseSpec, fit_or_parameters: dict[str, object]) -> dict[str, object]:
    observations = [float(value) for value in spec.observations]
    mean = _float_welford_mean(observations)
    residuals = [value - mean for value in observations]
    scale = max(abs(value) for value in residuals)
    if scale == 0.0:
        return {"outcome": "unidentified"}
    normalized = [value / scale for value in residuals]
    mean_square = math.fsum(value * value for value in normalized) / len(normalized)
    logh = math.log(mean_square)
    mu = float(fit_or_parameters["mu_normalized"])
    alpha = float(fit_or_parameters["alpha"])
    gamma = float(fit_or_parameters["gamma"])
    beta = float(fit_or_parameters["beta"])
    omega = (1.0 - beta) * mu
    c = float(SQRT_2_OVER_PI_TEXT)
    objective = 0.0
    log_path: list[float] = []
    variance_path: list[float] = []
    first_infinite_standardized_square: int | None = None
    infinite_standardized_squares = 0
    scale_square = scale * scale
    for index, x in enumerate(normalized):
        standardized_square = 0.0 if x == 0.0 else _float_safe_exp(2.0 * math.log(abs(x)) - logh)
        if standardized_square == math.inf:
            infinite_standardized_squares += 1
            if first_infinite_standardized_square is None:
                first_infinite_standardized_square = index
        objective += 0.5 * (logh + standardized_square)
        log_path.append(logh)
        variance_path.append(scale_square * _float_safe_exp(logh))
        if x == 0.0:
            z = 0.0
        else:
            magnitude = _float_safe_exp(math.log(abs(x)) - 0.5 * logh)
            z = math.copysign(magnitude, x)
        logh = omega + alpha * (abs(z) - c) + gamma * z + beta * logh
    forecast = scale_square * _float_safe_exp(logh)
    return {
        "outcome": "positive_infinity" if objective == math.inf else "finite",
        "objective": objective,
        "log_variances": log_path,
        "physical_variances": variance_path,
        "one_step_forecast": forecast,
        "first_infinite_standardized_square": first_infinite_standardized_square,
        "infinite_standardized_squares": infinite_standardized_squares,
    }


def _relative_error(actual: float, expected: Decimal) -> Decimal:
    if expected == ZERO:
        return abs(Decimal.from_float(actual))
    return abs(Decimal.from_float(actual) - expected) / abs(expected)


def _binary64_measurements(raw: dict[str, object]) -> dict[str, Decimal]:
    maxima = {"objective": ZERO, "log_variance": ZERO, "variance_relative": ZERO, "forecast_relative": ZERO}
    cases = raw["cases"]
    assert isinstance(cases, list)
    by_spec = {spec.name: spec for spec in CASES}
    for case in cases:
        assert isinstance(case, dict)
        spec = by_spec[str(case["name"])]
        if case["outcome"] == "success":
            fit = case["fit"]
            assert isinstance(fit, dict)
            replay = _binary64_replay(spec, fit)
            if replay["outcome"] != "finite":
                raise AssertionError(f"{spec.name}: finite optimum replay became nonfinite")
            maxima["objective"] = max(maxima["objective"], abs(Decimal.from_float(replay["objective"]) - fit["objective"]))
            for actual, expected in zip(replay["log_variances"], case["normalized_log_variances"]):
                maxima["log_variance"] = max(maxima["log_variance"], abs(Decimal.from_float(actual) - expected))
            for actual, expected in zip(replay["physical_variances"], case["physical_variances"]):
                maxima["variance_relative"] = max(maxima["variance_relative"], _relative_error(actual, expected))
            forecast = case["one_step_forecast"]
            assert isinstance(forecast, dict)
            maxima["forecast_relative"] = max(maxima["forecast_relative"], _relative_error(replay["one_step_forecast"], forecast["physical_variance"]))
        elif case["outcome"] == "extended_real_probe":
            parameters = case["parameters"]
            assert isinstance(parameters, dict)
            replay = _binary64_replay(spec, parameters)
            if replay["outcome"] != "positive_infinity":
                raise AssertionError("extreme probe did not replay as binary64 +infinity")
            if replay["first_infinite_standardized_square"] is None:
                raise AssertionError("extreme probe did not locate first +infinity")
    if maxima["objective"] > BINARY64_OBJECTIVE_ABS_BOUND:
        raise AssertionError(f"binary64 objective bound exceeded: {maxima['objective']}")
    if maxima["log_variance"] > BINARY64_LOG_VARIANCE_ABS_BOUND:
        raise AssertionError(f"binary64 log-variance bound exceeded: {maxima['log_variance']}")
    if maxima["variance_relative"] > BINARY64_VARIANCE_REL_BOUND:
        raise AssertionError(f"binary64 variance bound exceeded: {maxima['variance_relative']}")
    if maxima["forecast_relative"] > BINARY64_FORECAST_REL_BOUND:
        raise AssertionError(f"binary64 forecast bound exceeded: {maxima['forecast_relative']}")
    return maxima


def _compare_raw(primary: object, verification: object, path: str = "") -> Decimal:
    if isinstance(primary, Decimal):
        if not isinstance(verification, Decimal):
            raise AssertionError(f"{path}: Decimal/type mismatch")
        if not primary.is_finite() or not verification.is_finite():
            if primary != verification:
                raise AssertionError(f"{path}: nonfinite mismatch")
            return ZERO
        error = abs(primary - verification)
        scale = max(ONE, abs(primary), abs(verification))
        tolerance = max(RAW_AGREEMENT_ABSOLUTE_TOLERANCE, Decimal(10) ** Decimal(-AGREEMENT_SIGNIFICANT_DIGITS) * scale)
        if error > tolerance:
            raise AssertionError(f"{path}: raw 80/120 mismatch {error} > {tolerance}")
        return error / scale
    if isinstance(primary, dict):
        if not isinstance(verification, dict) or sorted(primary) != sorted(verification):
            raise AssertionError(f"{path}: mapping schema mismatch")
        return max(
            (_compare_raw(primary[key], verification[key], f"{path}/{key}") for key in sorted(primary)),
            default=ZERO,
        )
    if isinstance(primary, (list, tuple)):
        if not isinstance(verification, type(primary)) or len(primary) != len(verification):
            raise AssertionError(f"{path}: sequence schema mismatch")
        return max((_compare_raw(a, b, f"{path}/{index}") for index, (a, b) in enumerate(zip(primary, verification))), default=ZERO)
    if type(primary) is not type(verification) or primary != verification:
        raise AssertionError(f"{path}: scalar mismatch {primary!r} != {verification!r}")
    return ZERO


def _gradient_audit(raw: dict[str, object]) -> Decimal:
    maxima = ZERO
    cases = raw["cases"]
    assert isinstance(cases, list)
    by_spec = {spec.name: spec for spec in CASES}
    step = Decimal("1e-18")
    for case in cases:
        assert isinstance(case, dict)
        if case["outcome"] != "success":
            continue
        spec = by_spec[str(case["name"])]
        prepared = _prepare(spec.observations)
        fit = case["fit"]
        assert isinstance(fit, dict)
        base = [fit["omega_normalized"], fit["alpha"], fit["gamma"], fit["beta"]]
        analytic = fit["physical_gradient"]
        assert isinstance(analytic, dict)
        keys = ("omega", "alpha", "gamma", "beta")
        for index, key in enumerate(keys):
            plus = list(base)
            minus = list(base)
            plus[index] += step
            minus[index] -= step
            # Audit the physical parameter gradient, not the long-run transform.
            def physical_objective(values: Sequence[Decimal]) -> Decimal:
                omega, alpha, gamma, beta = values
                if beta >= ONE:
                    return Decimal("Infinity")
                mu = omega / (ONE - beta)
                return _objective_only(prepared, mu, alpha, gamma, beta)
            if key == "beta" and base[index] == ZERO:
                plus_two = list(base)
                plus_two[index] += TWO * step
                numeric = (
                    -Decimal(3) * physical_objective(base)
                    + Decimal(4) * physical_objective(plus)
                    - physical_objective(plus_two)
                ) / (TWO * step)
            else:
                numeric = (physical_objective(plus) - physical_objective(minus)) / (TWO * step)
            error = abs(numeric - analytic[key])
            maxima = max(maxima, error)
            if error > Decimal("2e-16"):
                raise AssertionError(f"{spec.name}/{key}: analytic gradient error {error}")
    return maxima


TOP_KEYS = (
    "schema_version", "oracle", "contract", "constants", "case_count", "cases",
    "cross_case_checks", "precision_recheck", "binary64_replay", "rust_kernel_tolerances",
    "rust_fit_tolerances",
)
COMMON_CASE_KEYS = (
    "name", "purpose", "observations", "expected_selection", "expected_property",
    "reflection_of", "outcome",
)
CONTRACT_KEYS = (
    "centering", "normalization", "initialization", "recurrence",
    "standardized_residual", "objective", "domain", "coordinates",
)
CONSTANT_KEYS = (
    "sqrt_2_over_pi_embedded", "sqrt_2_over_pi_decimal_audit", "kkt_tolerance",
    "open_boundary_guard", "output_significant_digits",
)
FIT_KEYS = (
    "selection", "mu_normalized", "mu_physical", "omega_normalized",
    "omega_physical", "alpha", "gamma", "beta", "persistence_slack",
    "objective", "physical_gradient", "transformed_gradient",
    "transformed_gradient_norm", "iterations", "converged",
    "open_boundary_guard_rejected", "open_boundary_escape_detected",
    "beta_zero_directional_derivative",
)
GRADIENT_KEYS = ("omega", "alpha", "gamma", "beta")
BASIN_KEYS = (
    "evaluations", "mu_offsets", "alpha_values", "gamma_values", "beta_values",
    "face_grid_best_objectives", "local_refinements_per_face", "local_refinement_objectives",
)


def _require_keys(mapping: object, keys: Sequence[str], path: str) -> dict[str, object]:
    if not isinstance(mapping, dict) or sorted(mapping) != sorted(keys):
        actual = sorted(mapping) if isinstance(mapping, dict) else type(mapping).__name__
        raise AssertionError(f"{path}: exact keys mismatch: {actual!r}")
    return mapping


def _numeric_string(value: object, path: str, finite: bool = True) -> None:
    if not isinstance(value, str):
        raise AssertionError(f"{path}: expected numeric string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise AssertionError(f"{path}: invalid Decimal string") from exc
    if finite and not parsed.is_finite():
        raise AssertionError(f"{path}: expected finite Decimal string")


def _validate_numeric_tree(value: object, path: str) -> None:
    if isinstance(value, str):
        _numeric_string(value, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_numeric_tree(item, f"{path}/{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_numeric_tree(item, f"{path}/{key}")
    elif isinstance(value, (bool, int)) or value is None:
        return
    else:
        raise AssertionError(f"{path}: invalid canonical type {type(value).__name__}")


def _validate_numeric_list(value: object, path: str, length: int | None = None) -> list[object]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        raise AssertionError(f"{path}: numeric list/length")
    for index, item in enumerate(value):
        _numeric_string(item, f"{path}/{index}")
    return value


def _validate_fit(value: object, path: str) -> dict[str, object]:
    fit = _require_keys(value, FIT_KEYS, path)
    if fit["selection"] not in ("interior", "beta_zero"):
        raise AssertionError(f"{path}/selection")
    numeric_keys = (
        "mu_normalized", "mu_physical", "omega_normalized", "omega_physical",
        "alpha", "gamma", "beta", "persistence_slack", "objective",
        "transformed_gradient_norm",
    )
    for key in numeric_keys:
        _numeric_string(fit[key], f"{path}/{key}")
    gradient = _require_keys(fit["physical_gradient"], GRADIENT_KEYS, f"{path}/physical_gradient")
    for key in GRADIENT_KEYS:
        _numeric_string(gradient[key], f"{path}/physical_gradient/{key}")
    expected_gradient_length = 4 if fit["selection"] == "interior" else 3
    _validate_numeric_list(fit["transformed_gradient"], f"{path}/transformed_gradient", expected_gradient_length)
    if type(fit["iterations"]) is not int or fit["iterations"] < 0:
        raise AssertionError(f"{path}/iterations")
    for key in ("converged", "open_boundary_guard_rejected", "open_boundary_escape_detected"):
        if type(fit[key]) is not bool:
            raise AssertionError(f"{path}/{key}")
    directional = fit["beta_zero_directional_derivative"]
    if fit["selection"] == "beta_zero":
        _numeric_string(directional, f"{path}/beta_zero_directional_derivative")
    elif directional is not None:
        raise AssertionError(f"{path}/beta_zero_directional_derivative")
    return fit


def _validate_basin(value: object, path: str) -> None:
    basin = _require_keys(value, BASIN_KEYS, path)
    if type(basin["evaluations"]) is not int or basin["evaluations"] != 875:
        raise AssertionError(f"{path}/evaluations")
    expected_lengths = {"mu_offsets": 5, "alpha_values": 5, "gamma_values": 5, "beta_values": 6}
    for key, length in expected_lengths.items():
        _validate_numeric_list(basin[key], f"{path}/{key}", length)
    face_best = _require_keys(
        basin["face_grid_best_objectives"], ("interior", "beta_zero"),
        f"{path}/face_grid_best_objectives",
    )
    refinements = _require_keys(
        basin["local_refinement_objectives"], ("interior", "beta_zero"),
        f"{path}/local_refinement_objectives",
    )
    for key in ("interior", "beta_zero"):
        _numeric_string(face_best[key], f"{path}/face_grid_best_objectives/{key}")
        _validate_numeric_list(refinements[key], f"{path}/local_refinement_objectives/{key}", 1)
    if type(basin["local_refinements_per_face"]) is not int or basin["local_refinements_per_face"] != 1:
        raise AssertionError(f"{path}/local_refinements_per_face")


def _validate_rust_tolerances(
    value: object,
    path: str,
    metric_keys: Sequence[str],
) -> None:
    outer_keys = (
        "measurement_date", "measurement_basis", "certification", "tolerance_policy",
        "measured_maxima", *metric_keys,
    )
    block = _require_keys(value, outer_keys, path)
    if block["measurement_date"] != "2026-09-02":
        raise AssertionError(f"{path}/measurement_date")
    if not isinstance(block["measurement_basis"], str) or "Rust EGARCH" not in block["measurement_basis"]:
        raise AssertionError(f"{path}/measurement_basis")
    if not isinstance(block["certification"], str) or "not Decimal certification" not in block["certification"]:
        raise AssertionError(f"{path}/certification")
    if not isinstance(block["tolerance_policy"], str) or "four times" not in block["tolerance_policy"]:
        raise AssertionError(f"{path}/tolerance_policy")
    maxima = _require_keys(block["measured_maxima"], metric_keys, f"{path}/measured_maxima")
    for key in metric_keys:
        _numeric_string(maxima[key], f"{path}/measured_maxima/{key}")
        _numeric_string(block[key], f"{path}/{key}")
        measured = Decimal(maxima[key])
        tolerance = Decimal(block[key])
        ratio = tolerance / measured
        if not (Decimal("3.9") <= ratio <= Decimal("4.1")):
            raise AssertionError(f"{path}/{key}: tolerance is not approximately 4x")


def _validate_schema(document: object) -> dict[str, object]:
    top = _require_keys(document, TOP_KEYS, "")
    if type(top["schema_version"]) is not int or top["schema_version"] != 1:
        raise AssertionError("/schema_version")
    if not isinstance(top["oracle"], str):
        raise AssertionError("/oracle")
    contract = _require_keys(top["contract"], CONTRACT_KEYS, "/contract")
    if any(not isinstance(contract[key], str) for key in CONTRACT_KEYS):
        raise AssertionError("/contract: prose strings")
    constants = _require_keys(top["constants"], CONSTANT_KEYS, "/constants")
    for key in CONSTANT_KEYS[:-1]:
        _numeric_string(constants[key], f"/constants/{key}")
    if type(constants["output_significant_digits"]) is not int or constants["output_significant_digits"] != OUTPUT_SIGNIFICANT_DIGITS:
        raise AssertionError("/constants/output_significant_digits")
    if type(top["case_count"]) is not int or top["case_count"] != len(CASES):
        raise AssertionError("/case_count")
    cases = top["cases"]
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise AssertionError("/cases")
    if [case.get("name") if isinstance(case, dict) else None for case in cases] != [spec.name for spec in CASES]:
        raise AssertionError("/cases: names/order")
    for index, (case, spec) in enumerate(zip(cases, CASES)):
        path = f"/cases/{index}"
        if not isinstance(case, dict):
            raise AssertionError(f"{path}: mapping required")
        if any(key not in case for key in COMMON_CASE_KEYS):
            raise AssertionError(f"{path}: common keys")
        if not isinstance(case["observations"], list) or case["observations"] != list(spec.observations):
            raise AssertionError(f"{path}/observations")
        if any(not isinstance(value, str) for value in case["observations"]):
            raise AssertionError(f"{path}/observations: string list")
        if case["expected_selection"] != spec.expected_selection or case["expected_property"] != spec.expected_property or case["reflection_of"] != spec.reflection_of:
            raise AssertionError(f"{path}: spec metadata")
        expected_keys: tuple[str, ...]
        if case["outcome"] == "success":
            expected_keys = COMMON_CASE_KEYS + (
                "mean", "scale", "initial_log_variance", "fit", "normalized_log_variances",
                "physical_variances", "one_step_forecast", "face_candidates", "basin_grid",
            )
            if len(case["normalized_log_variances"]) != len(spec.observations) or len(case["physical_variances"]) != len(spec.observations):
                raise AssertionError(f"{path}: path length")
            fit = case["fit"]
            _validate_fit(fit, f"{path}/fit")
            if fit["selection"] != spec.expected_selection:
                raise AssertionError(f"{path}/fit/selection")
            if not isinstance(case["face_candidates"], list) or len(case["face_candidates"]) != 2:
                raise AssertionError(f"{path}/face_candidates")
            for face_index, face in enumerate(case["face_candidates"]):
                _validate_fit(face, f"{path}/face_candidates/{face_index}")
            if [face["selection"] for face in case["face_candidates"]] != ["interior", "beta_zero"]:
                raise AssertionError(f"{path}/face_candidates: face order")
            for key in ("mean", "scale", "initial_log_variance"):
                _numeric_string(case[key], f"{path}/{key}")
            _validate_numeric_list(case["normalized_log_variances"], f"{path}/normalized_log_variances", len(spec.observations))
            _validate_numeric_list(case["physical_variances"], f"{path}/physical_variances", len(spec.observations))
            forecast = _require_keys(
                case["one_step_forecast"],
                ("normalized_log_variance", "physical_log_variance", "physical_variance"),
                f"{path}/one_step_forecast",
            )
            for key in forecast:
                _numeric_string(forecast[key], f"{path}/one_step_forecast/{key}")
            _validate_basin(case["basin_grid"], f"{path}/basin_grid")
        elif case["outcome"] == "unidentified_constant_series":
            expected_keys = COMMON_CASE_KEYS + ("mean", "scale", "failure")
            _numeric_string(case["mean"], f"{path}/mean")
            _numeric_string(case["scale"], f"{path}/scale")
            if not isinstance(case["failure"], str):
                raise AssertionError(f"{path}/failure")
        elif case["outcome"] == "extended_real_probe":
            expected_keys = COMMON_CASE_KEYS + (
                "mean", "scale", "initial_log_variance", "parameters", "decimal_objective",
                "decimal_objective_is_finite", "normalized_log_variances", "one_step_forecast",
            )
            if len(case["normalized_log_variances"]) != len(spec.observations):
                raise AssertionError(f"{path}: probe path length")
            for key in ("mean", "scale", "initial_log_variance", "decimal_objective"):
                _numeric_string(case[key], f"{path}/{key}")
            if type(case["decimal_objective_is_finite"]) is not bool or not case["decimal_objective_is_finite"]:
                raise AssertionError(f"{path}/decimal_objective_is_finite")
            parameters = _require_keys(
                case["parameters"],
                ("mu_normalized", "omega_normalized", "alpha", "gamma", "beta"),
                f"{path}/parameters",
            )
            for key in parameters:
                _numeric_string(parameters[key], f"{path}/parameters/{key}")
            _validate_numeric_list(case["normalized_log_variances"], f"{path}/normalized_log_variances", len(spec.observations))
            forecast = _require_keys(
                case["one_step_forecast"], ("normalized_log_variance", "physical_variance"),
                f"{path}/one_step_forecast",
            )
            for key in forecast:
                if key == "physical_variance":
                    if forecast[key] != "+inf":
                        raise AssertionError(f"{path}/one_step_forecast/physical_variance")
                else:
                    _numeric_string(forecast[key], f"{path}/one_step_forecast/{key}")
        else:
            raise AssertionError(f"{path}/outcome")
        _require_keys(case, expected_keys, path)
    cross = _require_keys(
        top["cross_case_checks"],
        ("leveraged_gamma_negative", "gamma_zero_absolute_bound", "beta_zero_face_selected",
         "near_integrated_beta_interval", "reflection_max_absolute_error"),
        "/cross_case_checks",
    )
    if type(cross["leveraged_gamma_negative"]) is not bool or type(cross["beta_zero_face_selected"]) is not bool:
        raise AssertionError("/cross_case_checks: flags")
    _numeric_string(cross["gamma_zero_absolute_bound"], "/cross_case_checks/gamma_zero_absolute_bound")
    _validate_numeric_list(cross["near_integrated_beta_interval"], "/cross_case_checks/near_integrated_beta_interval", 2)
    _numeric_string(cross["reflection_max_absolute_error"], "/cross_case_checks/reflection_max_absolute_error")
    precision = _require_keys(
        top["precision_recheck"],
        ("primary_decimal_digits", "verification_decimal_digits", "raw_agreement_significant_digits", "comparison"),
        "/precision_recheck",
    )
    if (precision["primary_decimal_digits"], precision["verification_decimal_digits"], precision["raw_agreement_significant_digits"]) != (ORACLE_PRECISION, VERIFICATION_PRECISION, AGREEMENT_SIGNIFICANT_DIGITS):
        raise AssertionError("/precision_recheck: digits")
    if not isinstance(precision["comparison"], str):
        raise AssertionError("/precision_recheck/comparison")
    replay = _require_keys(
        top["binary64_replay"],
        ("canonical_measurements", "objective_absolute_error_bound", "log_variance_absolute_error_bound",
         "variance_relative_error_bound", "one_step_forecast_relative_error_bound", "extended_real_rule"),
        "/binary64_replay",
    )
    for key in ("objective_absolute_error_bound", "log_variance_absolute_error_bound", "variance_relative_error_bound", "one_step_forecast_relative_error_bound"):
        _numeric_string(replay[key], f"/binary64_replay/{key}")
    if not isinstance(replay["canonical_measurements"], str) or not isinstance(replay["extended_real_rule"], str):
        raise AssertionError("/binary64_replay: prose")
    _validate_rust_tolerances(
        top["rust_kernel_tolerances"], "/rust_kernel_tolerances",
        ("objective_absolute", "log_variance_absolute", "physical_variance_relative",
         "one_step_forecast_relative", "physical_gradient_absolute"),
    )
    _validate_rust_tolerances(
        top["rust_fit_tolerances"], "/rust_fit_tolerances",
        ("coefficient_absolute", "omega_absolute", "objective_absolute",
         "log_variance_absolute", "physical_variance_relative", "one_step_forecast_relative"),
    )
    return top


def _canonical_bytes(document: dict[str, object]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _deep_build() -> tuple[dict[str, object], dict[str, Decimal], Decimal, Decimal]:
    primary = _build_raw(ORACLE_PRECISION)
    verification = _build_raw(VERIFICATION_PRECISION)
    with localcontext() as ctx:
        ctx.prec = 140
        ctx.Emax = 999_999_999
        ctx.Emin = -999_999_999
        agreement = _compare_raw(primary, verification)
        gradient_error = _gradient_audit(verification)
        binary_errors = _binary64_measurements(verification)
    formatted = _format_decimal_tree(primary)
    assert isinstance(formatted, dict)
    _validate_schema(formatted)
    return formatted, binary_errors, agreement, gradient_error


def _load(path: Path) -> tuple[dict[str, object], bytes]:
    data = path.read_bytes()
    document = json.loads(data.decode("utf-8"))
    if not isinstance(document, dict):
        raise AssertionError("artifact root must be a mapping")
    return document, data


def fast_check(path: Path) -> None:
    document, data = _load(path)
    _validate_schema(document)
    if data != _canonical_bytes(document):
        raise AssertionError("artifact is not canonical pretty-printed UTF-8 JSON")
    digest = _sha256(data)
    if EXPECTED_ARTIFACT_SHA256 == "TO_BE_REPLACED_AFTER_DEEP_CHECK":
        raise AssertionError(f"artifact hash is not pinned; measured {digest}")
    if digest != EXPECTED_ARTIFACT_SHA256:
        raise AssertionError(f"artifact SHA-256 {digest} != pinned {EXPECTED_ARTIFACT_SHA256}")
    # The constant audit is inexpensive and protects the most important embedded
    # transcendental even in the sub-second path.
    with localcontext() as ctx:
        ctx.prec = 60
        _sqrt_2_over_pi()
    print(f"fast-check ok: {path} sha256={digest}")


def deep_check(path: Path) -> None:
    started = time.perf_counter()
    expected, binary_errors, agreement, gradient_error = _deep_build()
    actual, data = _load(path)
    _validate_schema(actual)
    if actual != expected:
        raise AssertionError("artifact differs from independently regenerated 80/120 Decimal oracle")
    if data != _canonical_bytes(actual):
        raise AssertionError("artifact bytes are not canonical")
    digest = _sha256(data)
    if EXPECTED_ARTIFACT_SHA256 != "TO_BE_REPLACED_AFTER_DEEP_CHECK" and digest != EXPECTED_ARTIFACT_SHA256:
        raise AssertionError(f"artifact SHA-256 {digest} != pinned {EXPECTED_ARTIFACT_SHA256}")
    elapsed = time.perf_counter() - started
    print(
        "deep-check ok: "
        f"sha256={digest} runtime={elapsed:.3f}s raw_relative={agreement} "
        f"gradient_abs={gradient_error} binary64={binary_errors}"
    )


def emit(path: Path) -> None:
    started = time.perf_counter()
    document, binary_errors, agreement, gradient_error = _deep_build()
    data = _canonical_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    elapsed = time.perf_counter() - started
    print(
        f"emitted {path} sha256={_sha256(data)} runtime={elapsed:.3f}s "
        f"raw_relative={agreement} gradient_abs={gradient_error} binary64={binary_errors}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("emit", "fast-check", "deep-check", "check"))
    parser.add_argument("path", nargs="?", type=Path, default=Path("golden/egarch_qml.json"))
    arguments = parser.parse_args()
    if arguments.command == "emit":
        emit(arguments.path)
    elif arguments.command == "fast-check":
        fast_check(arguments.path)
    else:
        deep_check(arguments.path)


if __name__ == "__main__":
    main()
